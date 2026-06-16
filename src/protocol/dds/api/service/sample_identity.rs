//! `SampleIdentity` — request/reply correlation header for ROS2 services.
//!
//! Wire layout (24 bytes, CDR body-relative):
//!   bytes  0–15: writer GUID ([u8; 16] = GuidPrefix[12] + EntityId[3] + kind[1])
//!   bytes 16–23: sequence number as CDR int64 little-endian
//!
//! NOTE: the sequence number is a plain CDR int64, NOT the RTPS SequenceNumber
//! wire format (which uses [high:i32][low:u32]).  This matches the cyclonedds/rmw
//! payload-embedded convention.

use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};

/// Correlation header embedded at the start of every service request and reply body.
///
/// The server echoes the request's `SampleIdentity` verbatim into each reply so
/// that clients can match replies to outstanding requests by comparing
/// `header.writer_guid` against the client's own request-publisher GUID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SampleIdentity {
    /// 16-byte GUID of the request-writer: `[prefix(12)][entity_key(3)][kind(1)]`.
    pub writer_guid: [u8; 16],
    /// Monotonically increasing request sequence number (client-local).
    pub sequence_number: i64,
}

impl SampleIdentity {
    /// Construct from explicit components.
    pub fn new(writer_guid: [u8; 16], sequence_number: i64) -> Self {
        Self {
            writer_guid,
            sequence_number,
        }
    }

    /// Write the 24-byte body (no CDR encapsulation header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.write_bytes(&self.writer_guid)?;
        w.write_i64(self.sequence_number)?;
        Ok(())
    }

    /// Read a 24-byte body (no CDR encapsulation header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let raw = r.read_bytes(16)?;
        let mut writer_guid = [0u8; 16];
        writer_guid.copy_from_slice(raw);
        let sequence_number = r.read_i64()?;
        Ok(Self {
            writer_guid,
            sequence_number,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::byte_cursor::Endianness;

    fn make_w(buf: &mut [u8]) -> ByteWriter<'_> {
        ByteWriter::new(buf, Endianness::Little)
    }

    fn make_r(buf: &[u8]) -> ByteCursor<'_> {
        ByteCursor::new(buf, Endianness::Little)
    }

    #[test]
    fn sample_identity_round_trip() {
        let id = SampleIdentity {
            writer_guid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 2],
            sequence_number: 42,
        };
        let mut buf = [0u8; 32];
        let mut w = make_w(&mut buf);
        id.serialize_inner(&mut w).unwrap();
        assert_eq!(w.position(), 24);

        let mut r = make_r(&buf[..24]);
        let decoded = SampleIdentity::deserialize_inner(&mut r).unwrap();
        assert_eq!(id, decoded);
    }

    #[test]
    fn sample_identity_wire_size() {
        let id = SampleIdentity {
            writer_guid: [0u8; 16],
            sequence_number: 1,
        };
        let mut buf = [0u8; 32];
        let mut w = make_w(&mut buf);
        id.serialize_inner(&mut w).unwrap();
        // body = 16 (guid) + 8 (i64) = 24 bytes
        assert_eq!(w.position(), 24);
        // seq_number=1 as CDR int64 LE: 01 00 00 00 00 00 00 00
        assert_eq!(
            &buf[16..24],
            &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn sample_identity_guid_bytes_order() {
        let guid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let id = SampleIdentity::new(guid, 0);
        let mut buf = [0u8; 32];
        let mut w = make_w(&mut buf);
        id.serialize_inner(&mut w).unwrap();
        // GUID bytes must appear in order at offset 0
        assert_eq!(&buf[..16], &guid);
    }
}
