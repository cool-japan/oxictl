//! `std_msgs` ROS2 message types.
//!
//! Provides `Header`, `ColorRGBA`, `Empty`, `StdString`, `Bool`,
//! and all numeric primitive message types with CDR serialization.

use heapless::String as HString;

use crate::protocol::dds::api::dds_type::DdsType;
use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};

use super::builtin_interfaces::Time;
use super::{make_cursor, make_writer};

// ─── Header ──────────────────────────────────────────────────────────────────

/// `std_msgs/msg/Header` — ROS2 message header with timestamp and frame ID.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Header {
    /// Timestamp.
    pub stamp: Time,
    /// Coordinate frame ID.
    pub frame_id: HString<256>,
}

impl Header {
    /// Serialize fields (without CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        self.stamp.serialize_inner(w)?;
        w.write_cdr_string(self.frame_id.as_str())?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let stamp = Time::deserialize_inner(r)?;
        let s = r.read_cdr_string()?;
        let mut frame_id = HString::<256>::new();
        frame_id
            .push_str(s)
            .map_err(|_| DdsApiError::Serialization("frame_id exceeds 256-byte capacity"))?;
        Ok(Self { stamp, frame_id })
    }
}

impl DdsType for Header {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::Header_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.serialize_inner(&mut w)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        Self::deserialize_inner(&mut r)
    }
}

// ─── ColorRGBA ───────────────────────────────────────────────────────────────

/// `std_msgs/msg/ColorRGBA` — RGBA colour with float components [0.0, 1.0].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColorRGBA {
    /// Red component.
    pub r: f32,
    /// Green component.
    pub g: f32,
    /// Blue component.
    pub b: f32,
    /// Alpha component.
    pub a: f32,
}

impl DdsType for ColorRGBA {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::ColorRGBA_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_f32(self.r)?;
        w.write_f32(self.g)?;
        w.write_f32(self.b)?;
        w.write_f32(self.a)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let rv = r.read_f32()?;
        let g = r.read_f32()?;
        let b = r.read_f32()?;
        let a = r.read_f32()?;
        Ok(Self { r: rv, g, b, a })
    }
}

// ─── Empty ───────────────────────────────────────────────────────────────────

/// `std_msgs/msg/Empty` — a message with no payload fields.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Empty {}

impl DdsType for Empty {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::Empty_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let w = make_writer(buf)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let _r = make_cursor(payload)?;
        Ok(Self {})
    }
}

// ─── StdString ───────────────────────────────────────────────────────────────

/// `std_msgs/msg/String` — a single string field.
///
/// Note: Rust type is named `StdString` to avoid collision with `std::string::String`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StdString {
    /// The string data.
    pub data: HString<256>,
}

impl DdsType for StdString {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::String_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_cdr_string(self.data.as_str())?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let s = r.read_cdr_string()?;
        let mut data = HString::<256>::new();
        data.push_str(s)
            .map_err(|_| DdsApiError::Serialization("string exceeds 256-byte capacity"))?;
        Ok(Self { data })
    }
}

// ─── Bool ────────────────────────────────────────────────────────────────────

/// `std_msgs/msg/Bool` — a single boolean field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bool {
    /// The boolean data (CDR: u8 where 0 = false, 1 = true).
    pub data: bool,
}

impl DdsType for Bool {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::Bool_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_u8(if self.data { 1 } else { 0 })?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let data = r.read_u8()? != 0;
        Ok(Self { data })
    }
}

// ─── Int8 ────────────────────────────────────────────────────────────────────

/// `std_msgs/msg/Int8` — a single signed 8-bit integer field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Int8 {
    /// The integer data.
    pub data: i8,
}

impl DdsType for Int8 {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::Int8_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_u8(self.data as u8)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let data = r.read_u8()? as i8;
        Ok(Self { data })
    }
}

// ─── Int16 ───────────────────────────────────────────────────────────────────

/// `std_msgs/msg/Int16` — a single signed 16-bit integer field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Int16 {
    /// The integer data.
    pub data: i16,
}

impl DdsType for Int16 {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::Int16_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_i16(self.data)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let data = r.read_i16()?;
        Ok(Self { data })
    }
}

// ─── Int32 ───────────────────────────────────────────────────────────────────

/// `std_msgs/msg/Int32` — a single signed 32-bit integer field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Int32 {
    /// The integer data.
    pub data: i32,
}

impl DdsType for Int32 {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::Int32_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_i32(self.data)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let data = r.read_i32()?;
        Ok(Self { data })
    }
}

// ─── Int64 ───────────────────────────────────────────────────────────────────

/// `std_msgs/msg/Int64` — a single signed 64-bit integer field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Int64 {
    /// The integer data.
    pub data: i64,
}

impl DdsType for Int64 {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::Int64_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_i64(self.data)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let data = r.read_i64()?;
        Ok(Self { data })
    }
}

// ─── UInt8 ───────────────────────────────────────────────────────────────────

/// `std_msgs/msg/UInt8` — a single unsigned 8-bit integer field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UInt8 {
    /// The integer data.
    pub data: u8,
}

impl DdsType for UInt8 {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::UInt8_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_u8(self.data)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let data = r.read_u8()?;
        Ok(Self { data })
    }
}

// ─── UInt16 ──────────────────────────────────────────────────────────────────

/// `std_msgs/msg/UInt16` — a single unsigned 16-bit integer field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UInt16 {
    /// The integer data.
    pub data: u16,
}

impl DdsType for UInt16 {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::UInt16_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_u16(self.data)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let data = r.read_u16()?;
        Ok(Self { data })
    }
}

// ─── UInt32 ──────────────────────────────────────────────────────────────────

/// `std_msgs/msg/UInt32` — a single unsigned 32-bit integer field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UInt32 {
    /// The integer data.
    pub data: u32,
}

impl DdsType for UInt32 {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::UInt32_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_u32(self.data)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let data = r.read_u32()?;
        Ok(Self { data })
    }
}

// ─── UInt64 ──────────────────────────────────────────────────────────────────

/// `std_msgs/msg/UInt64` — a single unsigned 64-bit integer field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UInt64 {
    /// The integer data.
    pub data: u64,
}

impl DdsType for UInt64 {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::UInt64_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_u64(self.data)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let data = r.read_u64()?;
        Ok(Self { data })
    }
}

// ─── Float32 ─────────────────────────────────────────────────────────────────

/// `std_msgs/msg/Float32` — a single 32-bit float field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Float32 {
    /// The float data.
    pub data: f32,
}

impl DdsType for Float32 {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::Float32_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_f32(self.data)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let data = r.read_f32()?;
        Ok(Self { data })
    }
}

// ─── Float64 ─────────────────────────────────────────────────────────────────

/// `std_msgs/msg/Float64` — a single 64-bit float field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Float64 {
    /// The float data.
    pub data: f64,
}

impl DdsType for Float64 {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::Float64_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_f64(self.data)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let data = r.read_f64()?;
        Ok(Self { data })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_type_name() {
        assert_eq!(Header::TYPE_NAME, "std_msgs::msg::dds_::Header_");
    }

    #[test]
    fn header_round_trip() {
        let mut frame_id = HString::<256>::new();
        frame_id.push_str("world").unwrap();
        let original = Header {
            stamp: Time {
                sec: 42,
                nanosec: 100,
            },
            frame_id,
        };
        let mut buf = [0u8; 512];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Header::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn header_byte_layout() {
        // Header with stamp={0,0} and frame_id="map"
        let mut frame_id = HString::<256>::new();
        frame_id.push_str("map").unwrap();
        let h = Header {
            stamp: Time { sec: 0, nanosec: 0 },
            frame_id,
        };
        let mut buf = [0u8; 128];
        let written = h.serialize(&mut buf).unwrap();
        // Layout: 4 (header) + 4 (sec=0) + 4 (nanosec=0) + 8 (CDR string "map": 4+4) = 20
        assert_eq!(written, 20);
        // CDR LE header
        assert_eq!(&buf[0..4], &[0x00, 0x01, 0x00, 0x00]);
        // sec=0
        assert_eq!(&buf[4..8], &[0x00, 0x00, 0x00, 0x00]);
        // nanosec=0
        assert_eq!(&buf[8..12], &[0x00, 0x00, 0x00, 0x00]);
        // CDR string "map": length=4 (including NUL), then "map\0" (already 4-byte aligned, no extra pad)
        assert_eq!(&buf[12..16], &[0x04, 0x00, 0x00, 0x00]); // length=4 LE
        assert_eq!(&buf[16..20], &[b'm', b'a', b'p', 0x00]); // "map\0"
    }

    #[test]
    fn color_rgba_round_trip() {
        let original = ColorRGBA {
            r: 1.0,
            g: 0.5,
            b: 0.25,
            a: 0.75,
        };
        let mut buf = [0u8; 128];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = ColorRGBA::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn color_rgba_type_name() {
        assert_eq!(ColorRGBA::TYPE_NAME, "std_msgs::msg::dds_::ColorRGBA_");
    }

    #[test]
    fn empty_round_trip() {
        let original = Empty {};
        let mut buf = [0u8; 16];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Empty::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
        // Only header, no payload
        assert_eq!(written, 4);
    }

    #[test]
    fn empty_type_name() {
        assert_eq!(Empty::TYPE_NAME, "std_msgs::msg::dds_::Empty_");
    }

    #[test]
    fn std_string_round_trip() {
        let mut data = HString::<256>::new();
        data.push_str("hello, ROS2!").unwrap();
        let original = StdString { data };
        let mut buf = [0u8; 128];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = StdString::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn std_string_type_name() {
        assert_eq!(StdString::TYPE_NAME, "std_msgs::msg::dds_::String_");
    }

    #[test]
    fn bool_round_trip() {
        for val in [true, false] {
            let original = Bool { data: val };
            let mut buf = [0u8; 16];
            let written = original.serialize(&mut buf).unwrap();
            let decoded = Bool::deserialize(&buf[..written]).unwrap();
            assert_eq!(original, decoded);
        }
    }

    #[test]
    fn bool_type_name() {
        assert_eq!(Bool::TYPE_NAME, "std_msgs::msg::dds_::Bool_");
    }

    #[test]
    fn int8_round_trip() {
        let original = Int8 { data: -42 };
        let mut buf = [0u8; 16];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Int8::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn int8_type_name() {
        assert_eq!(Int8::TYPE_NAME, "std_msgs::msg::dds_::Int8_");
    }

    #[test]
    fn int16_round_trip() {
        let original = Int16 { data: -1000 };
        let mut buf = [0u8; 16];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Int16::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn int16_type_name() {
        assert_eq!(Int16::TYPE_NAME, "std_msgs::msg::dds_::Int16_");
    }

    #[test]
    fn int32_round_trip() {
        let original = Int32 { data: -100_000 };
        let mut buf = [0u8; 16];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Int32::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn int32_type_name() {
        assert_eq!(Int32::TYPE_NAME, "std_msgs::msg::dds_::Int32_");
    }

    #[test]
    fn int64_round_trip() {
        let original = Int64 {
            data: -9_000_000_000,
        };
        let mut buf = [0u8; 16];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Int64::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn int64_type_name() {
        assert_eq!(Int64::TYPE_NAME, "std_msgs::msg::dds_::Int64_");
    }

    #[test]
    fn uint8_round_trip() {
        let original = UInt8 { data: 255 };
        let mut buf = [0u8; 16];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = UInt8::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn uint8_type_name() {
        assert_eq!(UInt8::TYPE_NAME, "std_msgs::msg::dds_::UInt8_");
    }

    #[test]
    fn uint16_round_trip() {
        let original = UInt16 { data: 65000 };
        let mut buf = [0u8; 16];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = UInt16::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn uint16_type_name() {
        assert_eq!(UInt16::TYPE_NAME, "std_msgs::msg::dds_::UInt16_");
    }

    #[test]
    fn uint32_round_trip() {
        let original = UInt32 {
            data: 4_000_000_000,
        };
        let mut buf = [0u8; 16];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = UInt32::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn uint32_type_name() {
        assert_eq!(UInt32::TYPE_NAME, "std_msgs::msg::dds_::UInt32_");
    }

    #[test]
    fn uint64_round_trip() {
        let original = UInt64 {
            data: 18_000_000_000_000_000_000,
        };
        let mut buf = [0u8; 16];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = UInt64::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn uint64_type_name() {
        assert_eq!(UInt64::TYPE_NAME, "std_msgs::msg::dds_::UInt64_");
    }

    #[test]
    fn float32_round_trip() {
        let original = Float32 {
            data: std::f32::consts::PI,
        };
        let mut buf = [0u8; 16];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Float32::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn float32_type_name() {
        assert_eq!(Float32::TYPE_NAME, "std_msgs::msg::dds_::Float32_");
    }

    #[test]
    fn float64_round_trip() {
        let original = Float64 {
            data: std::f64::consts::E,
        };
        let mut buf = [0u8; 16];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Float64::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn float64_type_name() {
        assert_eq!(Float64::TYPE_NAME, "std_msgs::msg::dds_::Float64_");
    }
}
