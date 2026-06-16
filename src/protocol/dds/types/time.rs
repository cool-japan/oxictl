use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};
use crate::protocol::dds::error::RtpsError;

pub const TIME_ZERO: Time = Time {
    seconds: 0,
    fraction: 0,
};
pub const TIME_INVALID: Time = Time {
    seconds: -1,
    fraction: 0xFFFF_FFFF,
};
pub const TIME_INFINITE: Time = Time {
    seconds: 0x7FFF_FFFF,
    fraction: 0xFFFF_FFFF,
};

/// RTPS timestamp (NTP-style). `fraction` = nanoseconds * 2^32 / 10^9 (approximately).
/// Wire: [seconds i32][fraction u32], 8 bytes, in submessage endianness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Time {
    pub seconds: i32,
    pub fraction: u32,
}

impl Time {
    pub fn is_invalid(&self) -> bool {
        self == &TIME_INVALID
    }

    pub fn is_infinite(&self) -> bool {
        self == &TIME_INFINITE
    }

    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let seconds = cur.read_i32()?;
        let fraction = cur.read_u32()?;
        Ok(Self { seconds, fraction })
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.seconds)?;
        w.write_u32(self.fraction)
    }
}

/// RTPS duration. Same wire format as Time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Duration {
    pub seconds: i32,
    pub fraction: u32,
}

impl Duration {
    pub fn zero() -> Self {
        Self {
            seconds: 0,
            fraction: 0,
        }
    }

    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let seconds = cur.read_i32()?;
        let fraction = cur.read_u32()?;
        Ok(Self { seconds, fraction })
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.seconds)?;
        w.write_u32(self.fraction)
    }
}
