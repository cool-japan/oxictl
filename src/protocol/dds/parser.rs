use crate::protocol::dds::byte_cursor::Endianness;
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::message::submessage::{
    AckNack, Data, DataFrag, Gap, Heartbeat, HeartbeatFrag, InfoDestination, InfoReply,
    InfoReplyIp4, InfoSource, InfoTimestamp, NackFrag,
};
use crate::protocol::dds::message::{Message, MessageHeader, Submessage, SubmessageKind};

/// Parse a complete RTPS 2.3 message from `bytes`.
///
/// Zero-alloc: submessage payloads borrow from the input slice.
///
/// Per RTPS 8.3.7.1.2, unknown submessage kinds are silently skipped (forward compatibility).
/// Returns `TooManySubmessages` if more than 64 submessages are present.
pub fn parse_message<'a>(bytes: &'a [u8]) -> Result<Message<'a>, RtpsError> {
    let (header, mut remaining) = MessageHeader::parse(bytes)?;
    let mut submessages: heapless::Vec<Submessage<'a>, 64> = heapless::Vec::new();

    while !remaining.is_empty() {
        if remaining.len() < 4 {
            return Err(RtpsError::TruncatedHeader);
        }

        let kind_byte = remaining[0];
        let flags = remaining[1];
        let endianness = Endianness::from_flags(flags);

        let onh_bytes = [remaining[2], remaining[3]];
        let octets_to_next_header = match endianness {
            Endianness::Little => u16::from_le_bytes(onh_bytes),
            Endianness::Big => u16::from_be_bytes(onh_bytes),
        };

        remaining = &remaining[4..];

        let (body, next) = if octets_to_next_header == 0 {
            // Value 0 means submessage extends to end of message
            (remaining, &remaining[remaining.len()..])
        } else {
            let end = octets_to_next_header as usize;
            if end > remaining.len() {
                return Err(RtpsError::BufferTooSmall);
            }
            (&remaining[..end], &remaining[end..])
        };

        let sub = match SubmessageKind::from_u8(kind_byte) {
            None => {
                // Unknown kind: skip per RTPS 8.3.7.1.2 forward compat
                remaining = next;
                continue;
            }
            Some(SubmessageKind::Pad) => Submessage::Pad,
            Some(SubmessageKind::Data) => Submessage::Data(Data::parse(flags, body)?),
            Some(SubmessageKind::DataFrag) => Submessage::DataFrag(DataFrag::parse(flags, body)?),
            Some(SubmessageKind::Heartbeat) => {
                Submessage::Heartbeat(Heartbeat::parse(flags, body)?)
            }
            Some(SubmessageKind::HeartbeatFrag) => {
                Submessage::HeartbeatFrag(HeartbeatFrag::parse(flags, body)?)
            }
            Some(SubmessageKind::AckNack) => Submessage::AckNack(AckNack::parse(flags, body)?),
            Some(SubmessageKind::NackFrag) => Submessage::NackFrag(NackFrag::parse(flags, body)?),
            Some(SubmessageKind::Gap) => Submessage::Gap(Gap::parse(flags, body)?),
            Some(SubmessageKind::InfoTimestamp) => {
                Submessage::InfoTimestamp(InfoTimestamp::parse(flags, body)?)
            }
            Some(SubmessageKind::InfoSource) => {
                Submessage::InfoSource(InfoSource::parse(flags, body)?)
            }
            Some(SubmessageKind::InfoDestination) => {
                Submessage::InfoDestination(InfoDestination::parse(flags, body)?)
            }
            Some(SubmessageKind::InfoReply) => {
                Submessage::InfoReply(InfoReply::parse(flags, body)?)
            }
            Some(SubmessageKind::InfoReplyIp4) => {
                Submessage::InfoReplyIp4(InfoReplyIp4::parse(flags, body)?)
            }
        };

        submessages
            .push(sub)
            .map_err(|_| RtpsError::TooManySubmessages)?;
        remaining = next;
    }

    Ok(Message {
        header,
        submessages,
    })
}
