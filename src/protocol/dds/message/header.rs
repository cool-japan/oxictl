use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::message::RTPS_MAGIC;
use crate::protocol::dds::types::guid::{GuidPrefix, ProtocolVersion, VendorId};

/// RTPS message header (20 bytes): [RTPS magic 4][version 2][vendorId 2][guidPrefix 12].
///
/// The header fields are byte arrays; endianness is moot for magic/version/vendorId.
/// GuidPrefix is also a raw byte array, so no byte-order conversion is needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageHeader {
    pub version: ProtocolVersion,
    pub vendor_id: VendorId,
    pub guid_prefix: GuidPrefix,
}

impl MessageHeader {
    /// Parse a 20-byte RTPS header. Returns `(header, remaining_bytes)`.
    pub fn parse(bytes: &[u8]) -> Result<(Self, &[u8]), RtpsError> {
        if bytes.len() < 20 {
            return Err(RtpsError::TruncatedHeader);
        }
        if bytes[0..4] != RTPS_MAGIC {
            return Err(RtpsError::InvalidMagic);
        }
        // version and vendorId are NOT endianness-sensitive (byte sequences).
        // Use Big endian cursor (moot; read_u8 / read_bytes are endian-independent).
        let mut cur = ByteCursor::new(&bytes[4..20], Endianness::Big);
        let version = ProtocolVersion::parse(&mut cur)?;
        if !version.is_compatible_2x() {
            return Err(RtpsError::UnsupportedVersion);
        }
        let vendor_id = VendorId::parse(&mut cur)?;
        let guid_prefix = GuidPrefix::parse(&mut cur)?;
        Ok((
            Self {
                version,
                vendor_id,
                guid_prefix,
            },
            &bytes[20..],
        ))
    }

    /// Serialize into `buf`. Returns bytes written (always 20 on success).
    pub fn serialize(&self, buf: &mut [u8]) -> Result<usize, RtpsError> {
        if buf.len() < 20 {
            return Err(RtpsError::BufferTooSmall);
        }
        buf[0..4].copy_from_slice(&RTPS_MAGIC);
        let mut w = ByteWriter::new(&mut buf[4..20], Endianness::Big);
        self.version.serialize(&mut w)?;
        self.vendor_id.serialize(&mut w)?;
        self.guid_prefix.serialize(&mut w)?;
        Ok(20)
    }
}
