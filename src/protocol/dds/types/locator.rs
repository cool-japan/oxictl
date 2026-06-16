use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};
use crate::protocol::dds::error::RtpsError;

pub const LOCATOR_KIND_INVALID: i32 = -1;
pub const LOCATOR_KIND_RESERVED: i32 = 0;
pub const LOCATOR_KIND_UDP_V4: i32 = 1;
pub const LOCATOR_KIND_UDP_V6: i32 = 2;

pub const LOCATOR_INVALID: Locator = Locator {
    kind: LOCATOR_KIND_INVALID,
    port: 0,
    address: [0u8; 16],
};

/// RTPS network locator (address + port + kind).
/// Wire size: 24 bytes (kind i32 + port u32 + address [u8; 16]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Locator {
    pub kind: i32,
    pub port: u32,
    pub address: [u8; 16],
}

impl Locator {
    /// Construct a UDPv4 locator. `ipv4` is the 4-byte IPv4 address.
    /// The address field is zero-padded on the left (bytes 0–11 are 0, bytes 12–15 are the IPv4 address).
    pub fn udp_v4(port: u32, ipv4: [u8; 4]) -> Self {
        let mut address = [0u8; 16];
        address[12..16].copy_from_slice(&ipv4);
        Self {
            kind: LOCATOR_KIND_UDP_V4,
            port,
            address,
        }
    }

    /// Construct a UDPv6 locator.
    pub fn udp_v6(port: u32, ipv6: [u8; 16]) -> Self {
        Self {
            kind: LOCATOR_KIND_UDP_V6,
            port,
            address: ipv6,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.kind != LOCATOR_KIND_INVALID
    }

    /// Extract IPv4 address bytes if this is a UDPv4 locator.
    pub fn address_v4(&self) -> Option<[u8; 4]> {
        if self.kind == LOCATOR_KIND_UDP_V4 {
            let mut v4 = [0u8; 4];
            v4.copy_from_slice(&self.address[12..16]);
            Some(v4)
        } else {
            None
        }
    }

    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let kind = cur.read_i32()?;
        let port = cur.read_u32()?;
        let raw: [u8; 16] = cur
            .read_bytes(16)?
            .try_into()
            .map_err(|_| RtpsError::TruncatedHeader)?;
        Ok(Self {
            kind,
            port,
            address: raw,
        })
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.kind)?;
        w.write_u32(self.port)?;
        w.write_bytes(&self.address)
    }

    pub const WIRE_SIZE: usize = 24;
}
