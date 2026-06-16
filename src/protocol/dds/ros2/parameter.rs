//! CDR codec for `rcl_interfaces/msg/ParameterEvent`.

use heapless::Vec;

use super::error::Ros2Error;
use super::log::BuiltinTime;
use super::{cdr_str_len, read_cdr_str, write_cdr_str};
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};

// ─── ParameterType ───────────────────────────────────────────────────────────

/// Discriminant for `ParameterValue`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ParameterType {
    NotSet = 0,
    Bool = 1,
    Integer = 2,
    Double = 3,
    String = 4,
    ByteArray = 5,
    BoolArray = 6,
    IntegerArray = 7,
    DoubleArray = 8,
    StringArray = 9,
}

impl ParameterType {
    fn from_u8(v: u8) -> Result<Self, Ros2Error> {
        match v {
            0 => Ok(ParameterType::NotSet),
            1 => Ok(ParameterType::Bool),
            2 => Ok(ParameterType::Integer),
            3 => Ok(ParameterType::Double),
            4 => Ok(ParameterType::String),
            5 => Ok(ParameterType::ByteArray),
            6 => Ok(ParameterType::BoolArray),
            7 => Ok(ParameterType::IntegerArray),
            8 => Ok(ParameterType::DoubleArray),
            9 => Ok(ParameterType::StringArray),
            _ => Err(Ros2Error::UnknownParameterType),
        }
    }
}

// ─── ParameterValue ──────────────────────────────────────────────────────────

/// Typed value of a ROS2 parameter.
///
/// The CDR wire format for `ParameterValue` is a union-like struct where ALL
/// fields are always present on the wire regardless of the type discriminant.
///
/// The large variant size is intentional: heapless `Vec<_, 32>` is stack-allocated
/// and sized at compile time; no heap allocation occurs.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ParameterValue<'a> {
    NotSet,
    Bool(bool),
    Integer(i64),
    Double(f64),
    String(&'a str),
    ByteArray(Vec<u8, 32>),
    BoolArray(Vec<bool, 32>),
    IntegerArray(Vec<i64, 32>),
    DoubleArray(Vec<f64, 32>),
    StringArray(Vec<&'a str, 32>),
}

impl<'a> ParameterValue<'a> {
    fn type_discriminant(&self) -> u8 {
        match self {
            ParameterValue::NotSet => 0,
            ParameterValue::Bool(_) => 1,
            ParameterValue::Integer(_) => 2,
            ParameterValue::Double(_) => 3,
            ParameterValue::String(_) => 4,
            ParameterValue::ByteArray(_) => 5,
            ParameterValue::BoolArray(_) => 6,
            ParameterValue::IntegerArray(_) => 7,
            ParameterValue::DoubleArray(_) => 8,
            ParameterValue::StringArray(_) => 9,
        }
    }
}

// ─── CDR ParameterValue parse/serialize ──────────────────────────────────────
//
// Layout (ALL fields always on the wire):
//   type:                 u8 + 3 pad  → 4 bytes
//   bool_value:           u8 + 3 pad  → 4 bytes
//   integer_value:        i64         → 8 bytes (8-byte aligned)
//   double_value:         f64         → 8 bytes
//   string_value:         CDR string
//   byte_array_value:     u32 len + bytes
//   bool_array_value:     u32 len + u8 per element
//   integer_array_value:  u32 len + (8-byte-align then) i64 per element
//   double_array_value:   u32 len + (8-byte-align then) f64 per element
//   string_array_value:   u32 len + CDR strings

fn parse_parameter_value<'a>(cur: &mut ByteCursor<'a>) -> Result<ParameterValue<'a>, Ros2Error> {
    // type: u8 + 3 pad
    let type_raw = cur.read_u8().map_err(Ros2Error::from)?;
    let param_type = ParameterType::from_u8(type_raw)?;
    cur.skip(3).map_err(Ros2Error::from)?;

    // bool_value: u8 + 3 pad
    let bool_raw = cur.read_u8().map_err(Ros2Error::from)?;
    cur.skip(3).map_err(Ros2Error::from)?;

    // integer_value: 8-byte aligned i64
    cur.align_to(8).map_err(Ros2Error::from)?;
    let integer_raw = cur.read_i64().map_err(Ros2Error::from)?;

    // double_value: f64 (follows immediately after i64, stays 8-byte aligned)
    let double_raw = {
        let bits = cur.read_u64().map_err(Ros2Error::from)?;
        f64::from_bits(bits)
    };

    // string_value: CDR string
    let string_raw = read_cdr_str(cur)?;

    // byte_array_value: u32 len + bytes
    let byte_array_len = cur.read_u32().map_err(Ros2Error::from)? as usize;
    let byte_slice = cur.read_bytes(byte_array_len).map_err(Ros2Error::from)?;
    let mut byte_array: Vec<u8, 32> = Vec::new();
    for &b in byte_slice {
        byte_array
            .push(b)
            .map_err(|_| Ros2Error::TooManyArrayElements)?;
    }

    // bool_array_value: u32 len + u8 per element
    let bool_array_len = cur.read_u32().map_err(Ros2Error::from)? as usize;
    let bool_slice = cur.read_bytes(bool_array_len).map_err(Ros2Error::from)?;
    let mut bool_array: Vec<bool, 32> = Vec::new();
    for &b in bool_slice {
        bool_array
            .push(b != 0)
            .map_err(|_| Ros2Error::TooManyArrayElements)?;
    }

    // integer_array_value: u32 len + i64 per element (8-byte align before body)
    let int_array_len = cur.read_u32().map_err(Ros2Error::from)? as usize;
    let mut int_array: Vec<i64, 32> = Vec::new();
    if int_array_len > 0 {
        cur.align_to(8).map_err(Ros2Error::from)?;
        for _ in 0..int_array_len {
            let v = cur.read_i64().map_err(Ros2Error::from)?;
            int_array
                .push(v)
                .map_err(|_| Ros2Error::TooManyArrayElements)?;
        }
    }

    // double_array_value: u32 len + f64 per element (8-byte align before body)
    let dbl_array_len = cur.read_u32().map_err(Ros2Error::from)? as usize;
    let mut dbl_array: Vec<f64, 32> = Vec::new();
    if dbl_array_len > 0 {
        cur.align_to(8).map_err(Ros2Error::from)?;
        for _ in 0..dbl_array_len {
            let bits = cur.read_u64().map_err(Ros2Error::from)?;
            dbl_array
                .push(f64::from_bits(bits))
                .map_err(|_| Ros2Error::TooManyArrayElements)?;
        }
    }

    // string_array_value: u32 len + CDR strings
    let str_array_len = cur.read_u32().map_err(Ros2Error::from)? as usize;
    let mut str_array: Vec<&'a str, 32> = Vec::new();
    for _ in 0..str_array_len {
        let s = read_cdr_str(cur)?;
        str_array
            .push(s)
            .map_err(|_| Ros2Error::TooManyArrayElements)?;
    }

    // Select the active value based on discriminant
    let value = match param_type {
        ParameterType::NotSet => ParameterValue::NotSet,
        ParameterType::Bool => ParameterValue::Bool(bool_raw != 0),
        ParameterType::Integer => ParameterValue::Integer(integer_raw),
        ParameterType::Double => ParameterValue::Double(double_raw),
        ParameterType::String => ParameterValue::String(string_raw),
        ParameterType::ByteArray => ParameterValue::ByteArray(byte_array),
        ParameterType::BoolArray => ParameterValue::BoolArray(bool_array),
        ParameterType::IntegerArray => ParameterValue::IntegerArray(int_array),
        ParameterType::DoubleArray => ParameterValue::DoubleArray(dbl_array),
        ParameterType::StringArray => ParameterValue::StringArray(str_array),
    };

    Ok(value)
}

fn serialize_parameter_value(
    w: &mut ByteWriter<'_>,
    value: &ParameterValue<'_>,
) -> Result<(), Ros2Error> {
    let disc = value.type_discriminant();

    // type: u8 + 3 pad
    w.write_u8(disc).map_err(Ros2Error::from)?;
    w.write_bytes(&[0u8, 0u8, 0u8]).map_err(Ros2Error::from)?;

    // bool_value: u8 + 3 pad
    let bool_val: u8 = match value {
        ParameterValue::Bool(b) => u8::from(*b),
        _ => 0,
    };
    w.write_u8(bool_val).map_err(Ros2Error::from)?;
    w.write_bytes(&[0u8, 0u8, 0u8]).map_err(Ros2Error::from)?;

    // integer_value: 8-byte align then i64
    w.align_to(8).map_err(Ros2Error::from)?;
    let int_val: i64 = match value {
        ParameterValue::Integer(v) => *v,
        _ => 0,
    };
    w.write_i64(int_val).map_err(Ros2Error::from)?;

    // double_value: f64 (immediately follows i64)
    let dbl_val: f64 = match value {
        ParameterValue::Double(v) => *v,
        _ => 0.0,
    };
    w.write_u64(dbl_val.to_bits()).map_err(Ros2Error::from)?;

    // string_value: CDR string
    let str_val: &str = match value {
        ParameterValue::String(s) => s,
        _ => "",
    };
    write_cdr_str(w, str_val)?;

    // byte_array_value: u32 len + bytes
    let byte_slice: &[u8] = match value {
        ParameterValue::ByteArray(v) => v.as_slice(),
        _ => &[],
    };
    w.write_u32(byte_slice.len() as u32)
        .map_err(Ros2Error::from)?;
    w.write_bytes(byte_slice).map_err(Ros2Error::from)?;

    // bool_array_value: u32 len + u8 per element
    let bool_count: u32 = match value {
        ParameterValue::BoolArray(v) => v.len() as u32,
        _ => 0,
    };
    w.write_u32(bool_count).map_err(Ros2Error::from)?;
    if let ParameterValue::BoolArray(v) = value {
        for &b in v.iter() {
            w.write_u8(u8::from(b)).map_err(Ros2Error::from)?;
        }
    }

    // integer_array_value: u32 len + (8-byte align) + i64 per element
    let int_count: u32 = match value {
        ParameterValue::IntegerArray(v) => v.len() as u32,
        _ => 0,
    };
    w.write_u32(int_count).map_err(Ros2Error::from)?;
    if let ParameterValue::IntegerArray(v) = value {
        if !v.is_empty() {
            w.align_to(8).map_err(Ros2Error::from)?;
            for &iv in v.iter() {
                w.write_i64(iv).map_err(Ros2Error::from)?;
            }
        }
    }

    // double_array_value: u32 len + (8-byte align) + f64 per element
    let dbl_count: u32 = match value {
        ParameterValue::DoubleArray(v) => v.len() as u32,
        _ => 0,
    };
    w.write_u32(dbl_count).map_err(Ros2Error::from)?;
    if let ParameterValue::DoubleArray(v) = value {
        if !v.is_empty() {
            w.align_to(8).map_err(Ros2Error::from)?;
            for &dv in v.iter() {
                w.write_u64(dv.to_bits()).map_err(Ros2Error::from)?;
            }
        }
    }

    // string_array_value: u32 len + CDR strings
    let str_count: u32 = match value {
        ParameterValue::StringArray(v) => v.len() as u32,
        _ => 0,
    };
    w.write_u32(str_count).map_err(Ros2Error::from)?;
    if let ParameterValue::StringArray(v) = value {
        for &s in v.iter() {
            write_cdr_str(w, s)?;
        }
    }

    Ok(())
}

/// Compute the byte length of a serialized `ParameterValue`.
///
/// The `base_offset` is the position in the output buffer where this value starts,
/// needed to compute alignment padding for i64/f64 fields.
fn parameter_value_serialized_len(value: &ParameterValue<'_>, base_offset: usize) -> usize {
    let mut len = 0usize;

    // type: u8 + 3 pad
    len += 4;
    // bool_value: u8 + 3 pad
    len += 4;

    // integer_value: align to 8, then i64 (8 bytes)
    let pos_before_align = base_offset + len;
    let rem = pos_before_align % 8;
    if rem != 0 {
        len += 8 - rem; // alignment padding
    }
    len += 8; // i64

    // double_value: f64 (8 bytes, already 8-byte aligned after i64)
    len += 8;

    // string_value: CDR string
    let str_val: &str = match value {
        ParameterValue::String(s) => s,
        _ => "",
    };
    len += cdr_str_len(str_val);

    // byte_array_value: u32 len + bytes
    let byte_len = match value {
        ParameterValue::ByteArray(v) => v.len(),
        _ => 0,
    };
    len += 4 + byte_len;

    // bool_array_value: u32 len + u8 per element
    let bool_cnt = match value {
        ParameterValue::BoolArray(v) => v.len(),
        _ => 0,
    };
    len += 4 + bool_cnt;

    // integer_array_value: u32 len + align(8) + i64 per element
    let int_cnt = match value {
        ParameterValue::IntegerArray(v) => v.len(),
        _ => 0,
    };
    len += 4;
    if int_cnt > 0 {
        let pos = base_offset + len;
        let rem2 = pos % 8;
        if rem2 != 0 {
            len += 8 - rem2;
        }
        len += 8 * int_cnt;
    }

    // double_array_value: u32 len + align(8) + f64 per element
    let dbl_cnt = match value {
        ParameterValue::DoubleArray(v) => v.len(),
        _ => 0,
    };
    len += 4;
    if dbl_cnt > 0 {
        let pos = base_offset + len;
        let rem3 = pos % 8;
        if rem3 != 0 {
            len += 8 - rem3;
        }
        len += 8 * dbl_cnt;
    }

    // string_array_value: u32 len + CDR strings
    let str_cnt = match value {
        ParameterValue::StringArray(v) => v.len(),
        _ => 0,
    };
    len += 4;
    if let ParameterValue::StringArray(v) = value {
        for &s in v.iter() {
            len += cdr_str_len(s);
        }
    } else {
        let _ = str_cnt; // suppress warning
    }

    len
}

// ─── Parameter ───────────────────────────────────────────────────────────────

/// A single named ROS2 parameter with its typed value.
#[derive(Debug, Clone)]
pub struct Parameter<'a> {
    pub name: &'a str,
    pub value: ParameterValue<'a>,
}

fn parse_parameter<'a>(cur: &mut ByteCursor<'a>) -> Result<Parameter<'a>, Ros2Error> {
    let name = read_cdr_str(cur)?;
    let value = parse_parameter_value(cur)?;
    Ok(Parameter { name, value })
}

fn serialize_parameter(w: &mut ByteWriter<'_>, param: &Parameter<'_>) -> Result<(), Ros2Error> {
    write_cdr_str(w, param.name)?;
    serialize_parameter_value(w, &param.value)
}

fn parameter_serialized_len(param: &Parameter<'_>, base_offset: usize) -> usize {
    let name_len = cdr_str_len(param.name);
    let value_offset = base_offset + name_len;
    name_len + parameter_value_serialized_len(&param.value, value_offset)
}

// ─── ParameterEventMsg ───────────────────────────────────────────────────────

/// A parsed `rcl_interfaces/msg/ParameterEvent` message.
///
/// CDR layout:
/// ```text
/// stamp.sec:     i32  (4 bytes)
/// stamp.nanosec: u32  (4 bytes)
/// node:          CDR string
/// new_parameters:     CDR sequence of Parameter
/// changed_parameters: CDR sequence of Parameter
/// deleted_parameters: CDR sequence of string
/// ```
#[derive(Debug, Clone)]
pub struct ParameterEventMsg<'a> {
    pub stamp: BuiltinTime,
    pub node: &'a str,
    pub new_parameters: Vec<Parameter<'a>, 16>,
    pub changed_parameters: Vec<Parameter<'a>, 16>,
    pub deleted_parameters: Vec<&'a str, 16>,
}

impl<'a> ParameterEventMsg<'a> {
    /// Parse from a CDR little-endian byte cursor.
    pub fn parse(cur: &mut ByteCursor<'a>) -> Result<Self, Ros2Error> {
        // stamp
        let sec = cur.read_i32().map_err(Ros2Error::from)?;
        let nanosec = cur.read_u32().map_err(Ros2Error::from)?;
        let stamp = BuiltinTime { sec, nanosec };

        // node
        let node = read_cdr_str(cur)?;

        // new_parameters
        let new_count = cur.read_u32().map_err(Ros2Error::from)? as usize;
        let mut new_parameters: Vec<Parameter<'a>, 16> = Vec::new();
        for _ in 0..new_count {
            let p = parse_parameter(cur)?;
            new_parameters
                .push(p)
                .map_err(|_| Ros2Error::TooManyParameters)?;
        }

        // changed_parameters
        let changed_count = cur.read_u32().map_err(Ros2Error::from)? as usize;
        let mut changed_parameters: Vec<Parameter<'a>, 16> = Vec::new();
        for _ in 0..changed_count {
            let p = parse_parameter(cur)?;
            changed_parameters
                .push(p)
                .map_err(|_| Ros2Error::TooManyParameters)?;
        }

        // deleted_parameters (sequence of string)
        let deleted_count = cur.read_u32().map_err(Ros2Error::from)? as usize;
        let mut deleted_parameters: Vec<&'a str, 16> = Vec::new();
        for _ in 0..deleted_count {
            let s = read_cdr_str(cur)?;
            deleted_parameters
                .push(s)
                .map_err(|_| Ros2Error::TooManyParameters)?;
        }

        Ok(ParameterEventMsg {
            stamp,
            node,
            new_parameters,
            changed_parameters,
            deleted_parameters,
        })
    }

    /// Serialize to the given `ByteWriter`.
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), Ros2Error> {
        // stamp
        w.write_i32(self.stamp.sec).map_err(Ros2Error::from)?;
        w.write_u32(self.stamp.nanosec).map_err(Ros2Error::from)?;

        // node
        write_cdr_str(w, self.node)?;

        // new_parameters
        w.write_u32(self.new_parameters.len() as u32)
            .map_err(Ros2Error::from)?;
        for p in &self.new_parameters {
            serialize_parameter(w, p)?;
        }

        // changed_parameters
        w.write_u32(self.changed_parameters.len() as u32)
            .map_err(Ros2Error::from)?;
        for p in &self.changed_parameters {
            serialize_parameter(w, p)?;
        }

        // deleted_parameters
        w.write_u32(self.deleted_parameters.len() as u32)
            .map_err(Ros2Error::from)?;
        for &s in &self.deleted_parameters {
            write_cdr_str(w, s)?;
        }

        Ok(())
    }

    /// Compute the serialized byte length.
    pub fn serialized_len(&self) -> usize {
        let mut len = 0usize;

        // stamp: i32 + u32 = 8 bytes
        len += 8;

        // node: CDR string
        len += cdr_str_len(self.node);

        // new_parameters: u32 count + each param
        len += 4;
        for p in &self.new_parameters {
            let base = len;
            len += parameter_serialized_len(p, base);
        }

        // changed_parameters: u32 count + each param
        len += 4;
        for p in &self.changed_parameters {
            let base = len;
            len += parameter_serialized_len(p, base);
        }

        // deleted_parameters: u32 count + CDR strings
        len += 4;
        for &s in &self.deleted_parameters {
            len += cdr_str_len(s);
        }

        len
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::byte_cursor::{ByteWriter, Endianness};

    fn round_trip(msg: &ParameterEventMsg<'_>) -> Vec<u8, 1024> {
        let len = msg.serialized_len();
        let mut buf: Vec<u8, 1024> = Vec::new();
        buf.resize(len, 0).unwrap();
        let mut w = ByteWriter::new(buf.as_mut_slice(), Endianness::Little);
        msg.serialize(&mut w).unwrap();
        buf
    }

    fn make_empty_event<'a>() -> ParameterEventMsg<'a> {
        ParameterEventMsg {
            stamp: BuiltinTime {
                sec: 100,
                nanosec: 200,
            },
            node: "/test_node",
            new_parameters: Vec::new(),
            changed_parameters: Vec::new(),
            deleted_parameters: Vec::new(),
        }
    }

    #[test]
    fn parameter_value_bool_round_trip() {
        let event = ParameterEventMsg {
            stamp: BuiltinTime { sec: 1, nanosec: 0 },
            node: "/node",
            new_parameters: {
                let mut v: Vec<Parameter<'_>, 16> = Vec::new();
                v.push(Parameter {
                    name: "my_bool",
                    value: ParameterValue::Bool(true),
                })
                .unwrap();
                v
            },
            changed_parameters: Vec::new(),
            deleted_parameters: Vec::new(),
        };
        let buf = round_trip(&event);
        let mut cur = ByteCursor::new(buf.as_slice(), Endianness::Little);
        let parsed = ParameterEventMsg::parse(&mut cur).unwrap();
        assert_eq!(parsed.new_parameters.len(), 1);
        assert_eq!(parsed.new_parameters[0].name, "my_bool");
        match &parsed.new_parameters[0].value {
            ParameterValue::Bool(b) => assert!(*b),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn parameter_value_string_round_trip() {
        let event = ParameterEventMsg {
            stamp: BuiltinTime { sec: 0, nanosec: 0 },
            node: "/node",
            new_parameters: {
                let mut v: Vec<Parameter<'_>, 16> = Vec::new();
                v.push(Parameter {
                    name: "greeting",
                    value: ParameterValue::String("hello"),
                })
                .unwrap();
                v
            },
            changed_parameters: Vec::new(),
            deleted_parameters: Vec::new(),
        };
        let buf = round_trip(&event);
        let mut cur = ByteCursor::new(buf.as_slice(), Endianness::Little);
        let parsed = ParameterEventMsg::parse(&mut cur).unwrap();
        match &parsed.new_parameters[0].value {
            ParameterValue::String(s) => assert_eq!(*s, "hello"),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn parameter_value_integer_array_round_trip() {
        let vals: Vec<i64, 32> = {
            let mut v = Vec::new();
            for x in [10i64, -20, 300, -4000, 50000] {
                v.push(x).unwrap();
            }
            v
        };
        let event = ParameterEventMsg {
            stamp: BuiltinTime { sec: 0, nanosec: 0 },
            node: "/node",
            new_parameters: {
                let mut v: Vec<Parameter<'_>, 16> = Vec::new();
                v.push(Parameter {
                    name: "int_arr",
                    value: ParameterValue::IntegerArray(vals),
                })
                .unwrap();
                v
            },
            changed_parameters: Vec::new(),
            deleted_parameters: Vec::new(),
        };
        let buf = round_trip(&event);
        let mut cur = ByteCursor::new(buf.as_slice(), Endianness::Little);
        let parsed = ParameterEventMsg::parse(&mut cur).unwrap();
        match &parsed.new_parameters[0].value {
            ParameterValue::IntegerArray(v) => {
                assert_eq!(v.as_slice(), &[10i64, -20, 300, -4000, 50000]);
            }
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn parameter_event_with_one_changed_parameter_round_trips() {
        let event = ParameterEventMsg {
            stamp: BuiltinTime {
                sec: 42,
                nanosec: 0,
            },
            node: "/my_node",
            new_parameters: Vec::new(),
            changed_parameters: {
                let mut v: Vec<Parameter<'_>, 16> = Vec::new();
                v.push(Parameter {
                    name: "speed",
                    value: ParameterValue::Double(1.5),
                })
                .unwrap();
                v
            },
            deleted_parameters: Vec::new(),
        };
        let buf = round_trip(&event);
        let mut cur = ByteCursor::new(buf.as_slice(), Endianness::Little);
        let parsed = ParameterEventMsg::parse(&mut cur).unwrap();
        assert_eq!(parsed.changed_parameters.len(), 1);
        assert_eq!(parsed.changed_parameters[0].name, "speed");
        match &parsed.changed_parameters[0].value {
            ParameterValue::Double(d) => assert!((*d - 1.5f64).abs() < 1e-10),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn parameter_event_with_three_deletions_round_trips() {
        let event = ParameterEventMsg {
            stamp: BuiltinTime { sec: 0, nanosec: 0 },
            node: "/node",
            new_parameters: Vec::new(),
            changed_parameters: Vec::new(),
            deleted_parameters: {
                let mut v: Vec<&str, 16> = Vec::new();
                v.push("alpha").unwrap();
                v.push("beta").unwrap();
                v.push("gamma").unwrap();
                v
            },
        };
        let buf = round_trip(&event);
        let mut cur = ByteCursor::new(buf.as_slice(), Endianness::Little);
        let parsed = ParameterEventMsg::parse(&mut cur).unwrap();
        assert_eq!(parsed.deleted_parameters.len(), 3);
        assert_eq!(parsed.deleted_parameters[0], "alpha");
        assert_eq!(parsed.deleted_parameters[1], "beta");
        assert_eq!(parsed.deleted_parameters[2], "gamma");
    }

    #[test]
    fn parameter_event_full_payload_round_trip() {
        // 2 new + 3 changed + 1 deleted, mixed types
        let event = ParameterEventMsg {
            stamp: BuiltinTime {
                sec: 999,
                nanosec: 123,
            },
            node: "/full_node",
            new_parameters: {
                let mut v: Vec<Parameter<'_>, 16> = Vec::new();
                v.push(Parameter {
                    name: "rate",
                    value: ParameterValue::Double(10.0),
                })
                .unwrap();
                v.push(Parameter {
                    name: "label",
                    value: ParameterValue::String("fast"),
                })
                .unwrap();
                v
            },
            changed_parameters: {
                let mut v: Vec<Parameter<'_>, 16> = Vec::new();
                v.push(Parameter {
                    name: "count",
                    value: ParameterValue::Integer(7),
                })
                .unwrap();
                v.push(Parameter {
                    name: "active",
                    value: ParameterValue::Bool(false),
                })
                .unwrap();
                v.push(Parameter {
                    name: "empty",
                    value: ParameterValue::NotSet,
                })
                .unwrap();
                v
            },
            deleted_parameters: {
                let mut v: Vec<&str, 16> = Vec::new();
                v.push("old_param").unwrap();
                v
            },
        };

        // First serialization
        let buf1 = round_trip(&event);

        // Parse
        let mut cur = ByteCursor::new(buf1.as_slice(), Endianness::Little);
        let parsed = ParameterEventMsg::parse(&mut cur).unwrap();

        // Re-serialize
        let buf2 = round_trip(&parsed);

        // Must produce identical bytes (idempotent)
        assert_eq!(buf1.as_slice(), buf2.as_slice());

        // Spot check
        assert_eq!(parsed.stamp.sec, 999);
        assert_eq!(parsed.node, "/full_node");
        assert_eq!(parsed.new_parameters.len(), 2);
        assert_eq!(parsed.changed_parameters.len(), 3);
        assert_eq!(parsed.deleted_parameters.len(), 1);
        assert_eq!(parsed.deleted_parameters[0], "old_param");
    }

    #[test]
    fn parameter_event_serialized_len_matches_byte_count() {
        let event = make_empty_event();
        let expected_len = event.serialized_len();
        let mut buf = vec![0u8; expected_len];
        let mut w = ByteWriter::new(&mut buf, Endianness::Little);
        event.serialize(&mut w).unwrap();
        assert_eq!(w.position(), expected_len);
    }
}
