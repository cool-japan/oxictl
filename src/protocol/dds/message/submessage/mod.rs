pub mod acknack;
pub mod data;
pub mod gap;
pub mod heartbeat;
pub mod info;

pub use acknack::{AckNack, NackFrag};
pub use data::{Data, DataFrag};
pub use gap::Gap;
pub use heartbeat::{Heartbeat, HeartbeatFrag};
pub use info::{InfoDestination, InfoReply, InfoReplyIp4, InfoSource, InfoTimestamp};

use crate::protocol::dds::byte_cursor::{ByteWriter, Endianness};
use crate::protocol::dds::error::RtpsError;

/// RTPS 2.3 submessage kind identifiers (Table 8.14).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmessageKind {
    Pad = 0x01,
    AckNack = 0x06,
    Heartbeat = 0x07,
    Gap = 0x08,
    InfoTimestamp = 0x09,
    InfoSource = 0x0C,
    InfoReplyIp4 = 0x0D,
    InfoDestination = 0x0E,
    InfoReply = 0x0F,
    NackFrag = 0x12,
    HeartbeatFrag = 0x13,
    Data = 0x15,
    DataFrag = 0x16,
}

impl SubmessageKind {
    /// Convert from the raw wire byte. Returns `None` for unknown kinds.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::Pad),
            0x06 => Some(Self::AckNack),
            0x07 => Some(Self::Heartbeat),
            0x08 => Some(Self::Gap),
            0x09 => Some(Self::InfoTimestamp),
            0x0C => Some(Self::InfoSource),
            0x0D => Some(Self::InfoReplyIp4),
            0x0E => Some(Self::InfoDestination),
            0x0F => Some(Self::InfoReply),
            0x12 => Some(Self::NackFrag),
            0x13 => Some(Self::HeartbeatFrag),
            0x15 => Some(Self::Data),
            0x16 => Some(Self::DataFrag),
            _ => None,
        }
    }
}

/// Header that precedes every RTPS submessage (4 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmessageHeader {
    pub kind: SubmessageKind,
    pub flags: u8,
    pub octets_to_next_header: u16,
}

impl SubmessageHeader {
    /// Endianness determined by bit 0 of the flags byte.
    pub fn endianness(&self) -> Endianness {
        Endianness::from_flags(self.flags)
    }
}

/// A fully-parsed RTPS submessage (any of the 13 standard kinds + Pad).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Submessage<'a> {
    Data(Data<'a>),
    DataFrag(DataFrag<'a>),
    Heartbeat(Heartbeat),
    HeartbeatFrag(HeartbeatFrag),
    AckNack(AckNack),
    NackFrag(NackFrag),
    Gap(Gap),
    InfoTimestamp(InfoTimestamp),
    InfoSource(InfoSource),
    InfoDestination(InfoDestination),
    InfoReply(InfoReply),
    InfoReplyIp4(InfoReplyIp4),
    Pad,
}

impl<'a> Submessage<'a> {
    /// The submessage kind byte for this variant.
    pub fn kind(&self) -> SubmessageKind {
        match self {
            Self::Data(_) => SubmessageKind::Data,
            Self::DataFrag(_) => SubmessageKind::DataFrag,
            Self::Heartbeat(_) => SubmessageKind::Heartbeat,
            Self::HeartbeatFrag(_) => SubmessageKind::HeartbeatFrag,
            Self::AckNack(_) => SubmessageKind::AckNack,
            Self::NackFrag(_) => SubmessageKind::NackFrag,
            Self::Gap(_) => SubmessageKind::Gap,
            Self::InfoTimestamp(_) => SubmessageKind::InfoTimestamp,
            Self::InfoSource(_) => SubmessageKind::InfoSource,
            Self::InfoDestination(_) => SubmessageKind::InfoDestination,
            Self::InfoReply(_) => SubmessageKind::InfoReply,
            Self::InfoReplyIp4(_) => SubmessageKind::InfoReplyIp4,
            Self::Pad => SubmessageKind::Pad,
        }
    }

    /// Endianness for this submessage (PAD returns Little by convention).
    pub fn endianness(&self) -> Endianness {
        match self {
            Self::Data(v) => v.endianness,
            Self::DataFrag(v) => v.endianness,
            Self::Heartbeat(v) => v.endianness,
            Self::HeartbeatFrag(v) => v.endianness,
            Self::AckNack(v) => v.endianness,
            Self::NackFrag(v) => v.endianness,
            Self::Gap(v) => v.endianness,
            Self::InfoTimestamp(v) => v.endianness,
            Self::InfoSource(v) => v.endianness,
            Self::InfoDestination(v) => v.endianness,
            Self::InfoReply(v) => v.endianness,
            Self::InfoReplyIp4(v) => v.endianness,
            Self::Pad => Endianness::Little,
        }
    }

    /// Flags byte (kind-specific bits + E flag) for this submessage.
    pub fn flags_byte(&self) -> u8 {
        match self {
            Self::Data(v) => v.flags_byte(),
            Self::DataFrag(v) => v.flags_byte(),
            Self::Heartbeat(v) => v.flags_byte(),
            Self::HeartbeatFrag(v) => v.flags_byte(),
            Self::AckNack(v) => v.flags_byte(),
            Self::NackFrag(v) => v.flags_byte(),
            Self::Gap(v) => v.flags_byte(),
            Self::InfoTimestamp(v) => v.flags_byte(),
            Self::InfoSource(v) => v.flags_byte(),
            Self::InfoDestination(v) => v.flags_byte(),
            Self::InfoReply(v) => v.flags_byte(),
            Self::InfoReplyIp4(v) => v.flags_byte(),
            Self::Pad => Endianness::Little.into_flags(0u8),
        }
    }

    /// Serialize the body (everything after the 4-byte SubmessageHeader).
    ///
    /// PAD has no body and returns Ok(()) immediately.
    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        match self {
            Self::Data(v) => v.serialize_body(w),
            Self::DataFrag(v) => v.serialize_body(w),
            Self::Heartbeat(v) => v.serialize_body(w),
            Self::HeartbeatFrag(v) => v.serialize_body(w),
            Self::AckNack(v) => v.serialize_body(w),
            Self::NackFrag(v) => v.serialize_body(w),
            Self::Gap(v) => v.serialize_body(w),
            Self::InfoTimestamp(v) => v.serialize_body(w),
            Self::InfoSource(v) => v.serialize_body(w),
            Self::InfoDestination(v) => v.serialize_body(w),
            Self::InfoReply(v) => v.serialize_body(w),
            Self::InfoReplyIp4(v) => v.serialize_body(w),
            Self::Pad => Ok(()),
        }
    }
}
