//! Built-in `DdsType` implementations for commonly-used types.
//!
//! - `heapless::String<256>` — raw CDR string payload (no message envelope)
//! - `LogOwned`              — owned counterpart of `LogMsg<'_>` (ROS2 /rosout)
//! - `ParameterEventOwned`   — owned counterpart of `ParameterEventMsg<'_>` (ROS2 /parameter_events)

use heapless::String as HString;

use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};
use crate::protocol::dds::ros2::log::LogSeverity;

use super::dds_type::DdsType;
use super::error::DdsApiError;

// ─── CDR encapsulation header helpers ────────────────────────────────────────

/// CDR little-endian encapsulation header: `[0x00, 0x01, 0x00, 0x00]`.
const CDR_LE_HEADER: [u8; 4] = [0x00, 0x01, 0x00, 0x00];

/// Write the 4-byte CDR LE header and return a `ByteWriter` over the remainder.
fn make_writer(buf: &mut [u8]) -> Result<ByteWriter<'_>, DdsApiError> {
    if buf.len() < 4 {
        return Err(DdsApiError::PayloadBufferTooSmall);
    }
    buf[..4].copy_from_slice(&CDR_LE_HEADER);
    Ok(ByteWriter::new(&mut buf[4..], Endianness::Little))
}

/// Parse the 4-byte encapsulation header and return an appropriately-endian cursor.
fn make_cursor(payload: &[u8]) -> Result<ByteCursor<'_>, DdsApiError> {
    if payload.len() < 4 {
        return Err(DdsApiError::Serialization(
            "payload shorter than CDR header",
        ));
    }
    // CDR encapsulation byte 1: 0x01 = CDR_LE, 0x00 = CDR_BE.
    // Bit 0 of byte 1 set → little-endian.
    let endianness = if payload[1] & 0x01 != 0 {
        Endianness::Little
    } else {
        Endianness::Big
    };
    Ok(ByteCursor::new(&payload[4..], endianness))
}

// ─── heapless::String<256> ───────────────────────────────────────────────────

/// Wraps `heapless::String<256>` as a DDS CDR string type.
///
/// Wire format: `[CDR header (4 bytes)][u32 length (includes NUL)][bytes][NUL][padding]`.
impl DdsType for HString<256> {
    const TYPE_NAME: &'static str = "std_msgs::msg::dds_::String_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.write_cdr_string(self.as_str())
            .map_err(DdsApiError::from)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut cur = make_cursor(payload)?;
        let s = cur.read_cdr_string().map_err(DdsApiError::from)?;
        let mut result = HString::<256>::new();
        result
            .push_str(s)
            .map_err(|_| DdsApiError::Serialization("string exceeds 256-byte capacity"))?;
        Ok(result)
    }
}

// ─── LogOwned ────────────────────────────────────────────────────────────────

/// Owned version of `LogMsg<'_>` for use in typed queues.
///
/// Stores the same fields but with `heapless::String<256>` owned storage.
#[derive(Debug, Clone)]
pub struct LogOwned {
    /// Log severity level.
    pub severity: LogSeverity,
    /// Node name.
    pub name: HString<256>,
    /// Log message text.
    pub msg: HString<256>,
    /// Source file.
    pub file: HString<256>,
    /// Source function.
    pub function: HString<256>,
    /// Log line number.
    pub line: u32,
    /// Stamp seconds.
    pub stamp_sec: i32,
    /// Stamp nanoseconds.
    pub stamp_nsec: u32,
}

impl Default for LogOwned {
    fn default() -> Self {
        Self {
            severity: LogSeverity::Info,
            name: HString::new(),
            msg: HString::new(),
            file: HString::new(),
            function: HString::new(),
            line: 0,
            stamp_sec: 0,
            stamp_nsec: 0,
        }
    }
}

impl DdsType for LogOwned {
    const TYPE_NAME: &'static str = "rcl_interfaces::msg::dds_::Log_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        // stamp: sec(i32) + nsec(u32)
        w.write_i32(self.stamp_sec).map_err(DdsApiError::from)?;
        w.write_u32(self.stamp_nsec).map_err(DdsApiError::from)?;
        // level: u8
        w.write_u8(self.severity as u8).map_err(DdsApiError::from)?;
        // Pad to 4-byte alignment (wrote 9 bytes so far: 4 header + 8 stamp + 1 level = 9 in payload)
        // After the 4-byte header, payload starts. We've written 8 + 1 = 9 bytes into the payload
        // which is position 9 in the writer. Align to 4: need 3 pad bytes.
        w.align_to(4).map_err(DdsApiError::from)?;
        // strings: name, msg, file, function
        w.write_cdr_string(self.name.as_str())
            .map_err(DdsApiError::from)?;
        w.write_cdr_string(self.msg.as_str())
            .map_err(DdsApiError::from)?;
        w.write_cdr_string(self.file.as_str())
            .map_err(DdsApiError::from)?;
        w.write_cdr_string(self.function.as_str())
            .map_err(DdsApiError::from)?;
        // line: u32
        w.write_u32(self.line).map_err(DdsApiError::from)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut cur = make_cursor(payload)?;
        let stamp_sec = cur.read_i32().map_err(DdsApiError::from)?;
        let stamp_nsec = cur.read_u32().map_err(DdsApiError::from)?;
        let level_raw = cur.read_u8().map_err(DdsApiError::from)?;
        cur.align_to(4).map_err(DdsApiError::from)?;
        let severity = log_severity_from_u8(level_raw);
        let name_s = cur.read_cdr_string().map_err(DdsApiError::from)?;
        let msg_s = cur.read_cdr_string().map_err(DdsApiError::from)?;
        let file_s = cur.read_cdr_string().map_err(DdsApiError::from)?;
        let function_s = cur.read_cdr_string().map_err(DdsApiError::from)?;
        let line = cur.read_u32().map_err(DdsApiError::from)?;

        let mut name = HString::<256>::new();
        name.push_str(name_s)
            .map_err(|_| DdsApiError::Serialization("log name too long"))?;
        let mut msg = HString::<256>::new();
        msg.push_str(msg_s)
            .map_err(|_| DdsApiError::Serialization("log msg too long"))?;
        let mut file = HString::<256>::new();
        file.push_str(file_s)
            .map_err(|_| DdsApiError::Serialization("log file too long"))?;
        let mut function = HString::<256>::new();
        function
            .push_str(function_s)
            .map_err(|_| DdsApiError::Serialization("log function too long"))?;

        Ok(Self {
            severity,
            name,
            msg,
            file,
            function,
            line,
            stamp_sec,
            stamp_nsec,
        })
    }
}

fn log_severity_from_u8(v: u8) -> LogSeverity {
    match v {
        10 => LogSeverity::Debug,
        20 => LogSeverity::Info,
        30 => LogSeverity::Warn,
        40 => LogSeverity::Error,
        50 => LogSeverity::Fatal,
        _ => LogSeverity::Info,
    }
}

// ─── ParameterEventOwned ─────────────────────────────────────────────────────

/// Simplified owned version of `ParameterEventMsg<'_>` for use in typed queues.
///
/// Stores the timestamp and node name plus counts of newly created/changed/deleted
/// parameters.  Full `ParameterValue` data is omitted — decode the raw payload via
/// `ParameterEventMsg` directly if per-value inspection is needed.
#[derive(Debug, Clone, Default)]
pub struct ParameterEventOwned {
    /// Stamp seconds.
    pub stamp_sec: i32,
    /// Stamp nanoseconds.
    pub stamp_nsec: u32,
    /// Node name.
    pub node: HString<256>,
    /// Number of new parameters in this event.
    pub new_parameters_count: u32,
    /// Number of changed parameters in this event.
    pub changed_parameters_count: u32,
    /// Number of deleted parameters in this event.
    pub deleted_parameters_count: u32,
}

impl DdsType for ParameterEventOwned {
    const TYPE_NAME: &'static str = "rcl_interfaces::msg::dds_::ParameterEvent_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        // stamp: sec(i32) + nsec(u32)
        w.write_i32(self.stamp_sec).map_err(DdsApiError::from)?;
        w.write_u32(self.stamp_nsec).map_err(DdsApiError::from)?;
        // node name
        w.write_cdr_string(self.node.as_str())
            .map_err(DdsApiError::from)?;
        // new_parameters: length=u32 + 0 entries (simplified — no parameter values)
        w.write_u32(self.new_parameters_count)
            .map_err(DdsApiError::from)?;
        // changed_parameters: length=u32
        w.write_u32(self.changed_parameters_count)
            .map_err(DdsApiError::from)?;
        // deleted_parameters: length=u32
        w.write_u32(self.deleted_parameters_count)
            .map_err(DdsApiError::from)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut cur = make_cursor(payload)?;
        let stamp_sec = cur.read_i32().map_err(DdsApiError::from)?;
        let stamp_nsec = cur.read_u32().map_err(DdsApiError::from)?;
        let node_s = cur.read_cdr_string().map_err(DdsApiError::from)?;
        let new_count = cur.read_u32().map_err(DdsApiError::from)?;
        let changed_count = cur.read_u32().map_err(DdsApiError::from)?;
        let deleted_count = cur.read_u32().map_err(DdsApiError::from)?;

        let mut node = HString::<256>::new();
        node.push_str(node_s)
            .map_err(|_| DdsApiError::Serialization("parameter event node name too long"))?;

        Ok(Self {
            stamp_sec,
            stamp_nsec,
            node,
            new_parameters_count: new_count,
            changed_parameters_count: changed_count,
            deleted_parameters_count: deleted_count,
        })
    }
}
