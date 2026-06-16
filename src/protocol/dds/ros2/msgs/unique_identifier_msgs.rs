//! `unique_identifier_msgs` ROS2 message type.
//!
//! Provides `Uuid` — a 16-byte opaque goal identifier used in ROS2 actions.
//!
//! Wire format: 16 raw octets with no length prefix (CDR fixed array `uint8[16]`).

use crate::protocol::dds::api::dds_type::DdsType;
use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};

use super::{make_cursor, make_writer};

/// `unique_identifier_msgs/msg/UUID` — 16-byte opaque goal identifier.
///
/// Wire format: 16 raw octets, no length prefix.  CDR fixed-array `uint8[16]`.
/// DDS type name: `"unique_identifier_msgs::msg::dds_::UUID_"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Uuid {
    pub uuid: [u8; 16],
}

impl Uuid {
    /// The all-zeros nil UUID.
    pub fn nil() -> Self {
        Self { uuid: [0u8; 16] }
    }

    /// Construct from raw bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self { uuid: bytes }
    }

    /// Serialize body (no CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.write_bytes(&self.uuid)?;
        Ok(())
    }

    /// Deserialize body (no CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let raw = r.read_bytes(16)?;
        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(raw);
        Ok(Self { uuid })
    }
}

impl DdsType for Uuid {
    const TYPE_NAME: &'static str = "unique_identifier_msgs::msg::dds_::UUID_";

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
    fn uuid_type_name() {
        assert_eq!(Uuid::TYPE_NAME, "unique_identifier_msgs::msg::dds_::UUID_");
    }

    #[test]
    fn uuid_round_trip() {
        let original = Uuid::from_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mut buf = [0u8; 64];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Uuid::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn uuid_wire_size() {
        let u = Uuid::nil();
        let mut buf = [0u8; 64];
        let written = u.serialize(&mut buf).unwrap();
        // Total = 4 (CDR header) + 16 (uuid bytes) = 20 bytes, NO length prefix
        assert_eq!(written, 20);
        // CDR LE header
        assert_eq!(&buf[0..4], &[0x00, 0x01, 0x00, 0x00]);
        // uuid bytes start at offset 4
        assert_eq!(&buf[4..20], &[0u8; 16]);
    }

    #[test]
    fn uuid_byte_layout_not_a_sequence() {
        // Sequence would start with a u32 length prefix.
        // Fixed array does NOT — first 4 bytes of body are UUID bytes, not a length.
        let u = Uuid::from_bytes([0xFF; 16]);
        let mut buf = [0u8; 64];
        let written = u.serialize(&mut buf).unwrap();
        // body starts at buf[4]; must be 0xFF not a length field
        assert_eq!(buf[4], 0xFF);
        assert_eq!(written, 20);
    }

    #[test]
    fn uuid_nil() {
        assert_eq!(Uuid::nil().uuid, [0u8; 16]);
    }

    #[test]
    fn uuid_from_bytes() {
        let bytes = [42u8; 16];
        let u = Uuid::from_bytes(bytes);
        assert_eq!(u.uuid, bytes);
    }
}
