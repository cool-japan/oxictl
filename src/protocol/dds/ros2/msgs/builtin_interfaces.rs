//! `builtin_interfaces` ROS2 message types.
//!
//! Provides `Time` and `Duration` with CDR serialization.

use crate::protocol::dds::api::dds_type::DdsType;
use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};

use super::{make_cursor, make_writer};

// ─── Time ────────────────────────────────────────────────────────────────────

/// `builtin_interfaces/msg/Time` — ROS2 timestamp (seconds + nanoseconds).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Time {
    /// Seconds component (signed to allow times before epoch).
    pub sec: i32,
    /// Nanoseconds component [0, 999_999_999].
    pub nanosec: u32,
}

impl Time {
    /// Serialize fields (without CDR header) into `w`.
    ///
    /// Called by parent types that embed `Time` inline.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.write_i32(self.sec)?;
        w.write_u32(self.nanosec)?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    ///
    /// Called by parent types that embed `Time` inline.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let sec = r.read_i32()?;
        let nanosec = r.read_u32()?;
        Ok(Self { sec, nanosec })
    }
}

impl DdsType for Time {
    const TYPE_NAME: &'static str = "builtin_interfaces::msg::dds_::Time_";

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

// ─── Duration ────────────────────────────────────────────────────────────────

/// `builtin_interfaces/msg/Duration` — ROS2 duration (seconds + nanoseconds).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Duration {
    /// Seconds component (signed).
    pub sec: i32,
    /// Nanoseconds component [0, 999_999_999].
    pub nanosec: u32,
}

impl Duration {
    /// Serialize fields (without CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.write_i32(self.sec)?;
        w.write_u32(self.nanosec)?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let sec = r.read_i32()?;
        let nanosec = r.read_u32()?;
        Ok(Self { sec, nanosec })
    }
}

impl DdsType for Duration {
    const TYPE_NAME: &'static str = "builtin_interfaces::msg::dds_::Duration_";

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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_type_name() {
        assert_eq!(Time::TYPE_NAME, "builtin_interfaces::msg::dds_::Time_");
    }

    #[test]
    fn time_round_trip() {
        let original = Time {
            sec: 100,
            nanosec: 500_000,
        };
        let mut buf = [0u8; 64];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Time::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn time_byte_layout() {
        let t = Time { sec: 1, nanosec: 2 };
        let mut buf = [0u8; 64];
        let written = t.serialize(&mut buf).unwrap();
        // Total = 4 (header) + 4 (i32) + 4 (u32) = 12 bytes
        assert_eq!(written, 12);
        // CDR LE header
        assert_eq!(&buf[0..4], &[0x00, 0x01, 0x00, 0x00]);
        // sec = 1 as LE i32
        assert_eq!(&buf[4..8], &[0x01, 0x00, 0x00, 0x00]);
        // nanosec = 2 as LE u32
        assert_eq!(&buf[8..12], &[0x02, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn duration_type_name() {
        assert_eq!(
            Duration::TYPE_NAME,
            "builtin_interfaces::msg::dds_::Duration_"
        );
    }

    #[test]
    fn duration_round_trip() {
        let original = Duration {
            sec: -5,
            nanosec: 123_456_789,
        };
        let mut buf = [0u8; 64];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Duration::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }
}
