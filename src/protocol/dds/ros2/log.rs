//! CDR codec for `rcl_interfaces/msg/Log`.

use super::error::Ros2Error;
use super::{cdr_str_len, read_cdr_str, write_cdr_str};
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};

// ─── BuiltinTime ─────────────────────────────────────────────────────────────

/// ROS2 builtin timestamp (used in Log and ParameterEvent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinTime {
    pub sec: i32,
    pub nanosec: u32,
}

// ─── LogSeverity ─────────────────────────────────────────────────────────────

/// ROS2 log severity constants matching `rcl_interfaces::msg::Log` level field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LogSeverity {
    Debug = 10,
    Info = 20,
    Warn = 30,
    Error = 40,
    Fatal = 50,
}

impl LogSeverity {
    pub fn from_u8(v: u8) -> Result<Self, Ros2Error> {
        match v {
            10 => Ok(LogSeverity::Debug),
            20 => Ok(LogSeverity::Info),
            30 => Ok(LogSeverity::Warn),
            40 => Ok(LogSeverity::Error),
            50 => Ok(LogSeverity::Fatal),
            _ => Err(Ros2Error::InvalidLogLevel),
        }
    }
}

// ─── LogMsg ──────────────────────────────────────────────────────────────────

/// A borrowed view of a parsed `rcl_interfaces/msg/Log` message.
///
/// CDR layout (little-endian):
/// ```text
/// level:    u8  (1 byte) + 3 bytes padding → 4-byte aligned
/// stamp.sec:     i32  (4 bytes)
/// stamp.nanosec: u32  (4 bytes)
/// name:     CDR string (u32 length including NUL, bytes, NUL, pad to 4)
/// msg:      CDR string
/// file:     CDR string
/// function: CDR string
/// line:     u32 (4 bytes)
/// ```
#[derive(Debug, Clone)]
pub struct LogMsg<'a> {
    pub level: LogSeverity,
    pub stamp: BuiltinTime,
    pub name: &'a str,
    pub msg: &'a str,
    pub file: &'a str,
    pub function: &'a str,
    pub line: u32,
}

impl<'a> LogMsg<'a> {
    /// Parse a `LogMsg` from a CDR little-endian byte cursor.
    pub fn parse(cur: &mut ByteCursor<'a>) -> Result<Self, Ros2Error> {
        // level: u8 + 3 padding bytes
        let level_raw = cur.read_u8()?;
        let level = LogSeverity::from_u8(level_raw)?;
        // 3 bytes padding
        cur.skip(3).map_err(Ros2Error::from)?;

        // stamp: i32 sec + u32 nanosec
        let sec = cur.read_i32().map_err(Ros2Error::from)?;
        let nanosec = cur.read_u32().map_err(Ros2Error::from)?;
        let stamp = BuiltinTime { sec, nanosec };

        // CDR strings
        let name = read_cdr_str(cur)?;
        let msg = read_cdr_str(cur)?;
        let file = read_cdr_str(cur)?;
        let function = read_cdr_str(cur)?;

        // line: u32
        let line = cur.read_u32().map_err(Ros2Error::from)?;

        Ok(LogMsg {
            level,
            stamp,
            name,
            msg,
            file,
            function,
            line,
        })
    }

    /// Serialize this `LogMsg` into the given `ByteWriter`.
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), Ros2Error> {
        // level: u8 + 3 padding bytes
        w.write_u8(self.level as u8).map_err(Ros2Error::from)?;
        w.write_bytes(&[0u8, 0u8, 0u8]).map_err(Ros2Error::from)?;

        // stamp
        w.write_i32(self.stamp.sec).map_err(Ros2Error::from)?;
        w.write_u32(self.stamp.nanosec).map_err(Ros2Error::from)?;

        // CDR strings
        write_cdr_str(w, self.name)?;
        write_cdr_str(w, self.msg)?;
        write_cdr_str(w, self.file)?;
        write_cdr_str(w, self.function)?;

        // line
        w.write_u32(self.line).map_err(Ros2Error::from)?;

        Ok(())
    }

    /// Compute the serialized byte length without actually writing.
    pub fn serialized_len(&self) -> usize {
        4 // level u8 + 3 pad
        + 8 // stamp.sec i32 + stamp.nanosec u32
        + cdr_str_len(self.name)
        + cdr_str_len(self.msg)
        + cdr_str_len(self.file)
        + cdr_str_len(self.function)
        + 4 // line u32
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::byte_cursor::{ByteWriter, Endianness};

    fn make_log<'a>() -> LogMsg<'a> {
        LogMsg {
            level: LogSeverity::Info,
            stamp: BuiltinTime {
                sec: 1234,
                nanosec: 5678,
            },
            name: "my_node",
            msg: "hello world",
            file: "node.cpp",
            function: "main",
            line: 42,
        }
    }

    #[test]
    fn log_severity_constants_match_ros2() {
        assert_eq!(LogSeverity::Debug as u8, 10);
        assert_eq!(LogSeverity::Info as u8, 20);
        assert_eq!(LogSeverity::Warn as u8, 30);
        assert_eq!(LogSeverity::Error as u8, 40);
        assert_eq!(LogSeverity::Fatal as u8, 50);
    }

    #[test]
    fn log_round_trip_info_level() {
        let original = make_log();
        let len = original.serialized_len();
        let mut buf = vec![0u8; len];
        let mut w = ByteWriter::new(&mut buf, Endianness::Little);
        original.serialize(&mut w).unwrap();

        let mut cur = ByteCursor::new(&buf, Endianness::Little);
        let parsed = LogMsg::parse(&mut cur).unwrap();

        assert_eq!(parsed.level, original.level);
        assert_eq!(parsed.stamp.sec, original.stamp.sec);
        assert_eq!(parsed.stamp.nanosec, original.stamp.nanosec);
        assert_eq!(parsed.name, original.name);
        assert_eq!(parsed.msg, original.msg);
        assert_eq!(parsed.file, original.file);
        assert_eq!(parsed.function, original.function);
        assert_eq!(parsed.line, original.line);
    }

    #[test]
    fn log_serialized_len_matches_serialize_byte_count() {
        let log = make_log();
        let expected_len = log.serialized_len();
        let mut buf = vec![0u8; expected_len];
        let mut w = ByteWriter::new(&mut buf, Endianness::Little);
        log.serialize(&mut w).unwrap();
        assert_eq!(w.position(), expected_len);
    }

    #[test]
    fn log_with_empty_strings_round_trips() {
        let original = LogMsg {
            level: LogSeverity::Debug,
            stamp: BuiltinTime { sec: 0, nanosec: 0 },
            name: "",
            msg: "",
            file: "",
            function: "",
            line: 0,
        };
        let len = original.serialized_len();
        let mut buf = vec![0u8; len];
        let mut w = ByteWriter::new(&mut buf, Endianness::Little);
        original.serialize(&mut w).unwrap();

        let mut cur = ByteCursor::new(&buf, Endianness::Little);
        let parsed = LogMsg::parse(&mut cur).unwrap();

        assert_eq!(parsed.level, original.level);
        assert_eq!(parsed.name, "");
        assert_eq!(parsed.msg, "");
        assert_eq!(parsed.line, 0);
    }

    #[test]
    fn log_parse_truncated_buffer_returns_error() {
        let buf = [0u8; 4]; // only 4 bytes — not enough for a complete LogMsg
        let mut cur = ByteCursor::new(&buf, Endianness::Little);
        let result = LogMsg::parse(&mut cur);
        assert!(result.is_err());
    }

    #[test]
    fn log_parse_invalid_severity_returns_error() {
        // Build a valid-looking header but set level=99
        let mut buf = vec![0u8; 256];
        buf[0] = 99; // level = 99 (invalid)
                     // padding bytes 1-3 already zero
                     // stamp: sec=0, nanosec=0 at bytes 4-11 already zero
                     // need valid CDR strings for name/msg/file/function
                     // Each empty string: u32(1) + 0x00 + 3 pad = 8 bytes
        let mut cursor_pos = 12usize;
        for _ in 0..4 {
            // Write CDR empty string: length=1 (u32 LE), NUL byte, 3 padding bytes
            buf[cursor_pos] = 1;
            buf[cursor_pos + 1] = 0;
            buf[cursor_pos + 2] = 0;
            buf[cursor_pos + 3] = 0;
            buf[cursor_pos + 4] = 0; // NUL
                                     // 3 pad bytes already zero
            cursor_pos += 8;
        }
        // line: u32 at cursor_pos (already zero)

        let mut cur = ByteCursor::new(&buf, Endianness::Little);
        let result = LogMsg::parse(&mut cur);
        assert_eq!(result.unwrap_err(), Ros2Error::InvalidLogLevel);
    }

    #[test]
    fn log_round_trip_fatal_level() {
        let original = LogMsg {
            level: LogSeverity::Fatal,
            stamp: BuiltinTime {
                sec: -1,
                nanosec: 999_999_999,
            },
            name: "crash_node",
            msg: "fatal error occurred",
            file: "main.cpp",
            function: "shutdown",
            line: 100,
        };
        let len = original.serialized_len();
        let mut buf = vec![0u8; len];
        let mut w = ByteWriter::new(&mut buf, Endianness::Little);
        original.serialize(&mut w).unwrap();

        let mut cur = ByteCursor::new(&buf, Endianness::Little);
        let parsed = LogMsg::parse(&mut cur).unwrap();

        assert_eq!(parsed.level, LogSeverity::Fatal);
        assert_eq!(parsed.stamp.sec, -1);
        assert_eq!(parsed.name, "crash_node");
    }
}
