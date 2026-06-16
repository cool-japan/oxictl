pub mod header;
pub mod submessage;

pub use crate::protocol::dds::message::header::MessageHeader;
pub use crate::protocol::dds::message::submessage::{
    AckNack, Data, DataFrag, Gap, Heartbeat, HeartbeatFrag, InfoDestination, InfoReply,
    InfoReplyIp4, InfoSource, InfoTimestamp, NackFrag, Submessage, SubmessageKind,
};

/// RTPS magic bytes at the start of every message.
pub const RTPS_MAGIC: [u8; 4] = *b"RTPS";

/// A fully-parsed RTPS message: 20-byte header + up to 64 submessages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message<'a> {
    pub header: MessageHeader,
    pub submessages: heapless::Vec<Submessage<'a>, 64>,
}

impl<'a> Message<'a> {
    /// Iterate over all submessages in this message.
    pub fn iter_submessages(&self) -> impl Iterator<Item = &Submessage<'a>> {
        self.submessages.iter()
    }
}
