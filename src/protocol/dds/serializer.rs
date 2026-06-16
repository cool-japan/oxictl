use crate::protocol::dds::byte_cursor::{ByteWriter, Endianness};
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::message::Message;

/// Serialize a complete RTPS message into `buf`.
///
/// Returns the total number of bytes written.
/// Returns `BufferTooSmall` if `buf` is not large enough.
///
/// For each submessage: writes the 4-byte SubmessageHeader, then the body, then
/// backfills `octets_to_next_header` with the actual body length.
pub fn serialize_message(msg: &Message<'_>, buf: &mut [u8]) -> Result<usize, RtpsError> {
    let header_len = msg.header.serialize(buf)?; // writes 20 bytes
    let mut total = header_len;

    for submessage in msg.submessages.iter() {
        if total + 4 > buf.len() {
            return Err(RtpsError::BufferTooSmall);
        }

        let kind = submessage.kind() as u8;
        let flags = submessage.flags_byte();
        let endianness = submessage.endianness();

        buf[total] = kind;
        buf[total + 1] = flags;
        let onh_pos = total + 2;
        buf[onh_pos] = 0;
        buf[onh_pos + 1] = 0;
        total += 4;

        let body_start = total;
        if body_start > buf.len() {
            return Err(RtpsError::BufferTooSmall);
        }
        let mut w = ByteWriter::new(&mut buf[total..], endianness);
        submessage.serialize_body(&mut w)?;
        let body_len = w.position();
        total += body_len;

        // Backfill octets_to_next_header with the actual body length
        let onh = body_len as u16;
        let onh_bytes = match endianness {
            Endianness::Little => onh.to_le_bytes(),
            Endianness::Big => onh.to_be_bytes(),
        };
        buf[onh_pos] = onh_bytes[0];
        buf[onh_pos + 1] = onh_bytes[1];
    }

    Ok(total)
}
