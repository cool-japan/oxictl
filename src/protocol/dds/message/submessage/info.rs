use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::types::guid::{GuidPrefix, ProtocolVersion, VendorId};
use crate::protocol::dds::types::locator::Locator;
use crate::protocol::dds::types::time::Time;

/// INFO_TIMESTAMP submessage body (RTPS 2.3 Section 8.3.7.9).
///
/// Flags: E(0) I(1:invalidate)
/// If I flag NOT set: timestamp present (8 bytes). If I flag set: no body bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfoTimestamp {
    pub endianness: Endianness,
    pub invalidate_flag: bool,
    pub timestamp: Option<Time>,
}

impl InfoTimestamp {
    pub fn parse(flags: u8, body: &[u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let invalidate_flag = flags & 0x02 != 0;

        let timestamp = if invalidate_flag {
            None
        } else {
            let mut cur = ByteCursor::new(body, endianness);
            Some(Time::parse(&mut cur)?)
        };

        Ok(Self {
            endianness,
            invalidate_flag,
            timestamp,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        if let Some(ref ts) = self.timestamp {
            ts.serialize(w)?;
        }
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        let mut f = self.endianness.into_flags(0u8);
        if self.invalidate_flag {
            f |= 0x02;
        }
        f
    }
}

/// INFO_SOURCE submessage body (RTPS 2.3 Section 8.3.7.10).
///
/// Body: unused u32 (0) + protocol_version(2) + vendor_id(2) + guid_prefix(12) = 20 bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfoSource {
    pub endianness: Endianness,
    pub protocol_version: ProtocolVersion,
    pub vendor_id: VendorId,
    pub guid_prefix: GuidPrefix,
}

impl InfoSource {
    pub fn parse(flags: u8, body: &[u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let mut cur = ByteCursor::new(body, endianness);

        // unused u32 — always 0, read and discard in submessage endianness
        let _unused = cur.read_u32()?;

        let protocol_version = ProtocolVersion::parse(&mut cur)?;
        let vendor_id = VendorId::parse(&mut cur)?;
        let guid_prefix = GuidPrefix::parse(&mut cur)?;

        Ok(Self {
            endianness,
            protocol_version,
            vendor_id,
            guid_prefix,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_u32(0)?; // unused, always 0
        self.protocol_version.serialize(w)?;
        self.vendor_id.serialize(w)?;
        self.guid_prefix.serialize(w)?;
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        self.endianness.into_flags(0u8)
    }
}

/// INFO_DESTINATION submessage body (RTPS 2.3 Section 8.3.7.8).
///
/// Body: guid_prefix(12 bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfoDestination {
    pub endianness: Endianness,
    pub guid_prefix: GuidPrefix,
}

impl InfoDestination {
    pub fn parse(flags: u8, body: &[u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let mut cur = ByteCursor::new(body, endianness);
        let guid_prefix = GuidPrefix::parse(&mut cur)?;

        Ok(Self {
            endianness,
            guid_prefix,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        self.guid_prefix.serialize(w)?;
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        self.endianness.into_flags(0u8)
    }
}

/// INFO_REPLY submessage body (RTPS 2.3 Section 8.3.7.7).
///
/// Flags: E(0) M(1:multicast)
/// Body: unicast locator list + optional multicast locator list.
/// Capped at 8 locators per list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfoReply {
    pub endianness: Endianness,
    pub multicast_flag: bool,
    pub unicast_locator_list: heapless::Vec<Locator, 8>,
    pub multicast_locator_list: heapless::Vec<Locator, 8>,
}

impl InfoReply {
    pub fn parse(flags: u8, body: &[u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let multicast_flag = flags & 0x02 != 0;

        let mut cur = ByteCursor::new(body, endianness);

        let unicast_count = cur.read_u32()? as usize;
        if unicast_count > 8 {
            return Err(RtpsError::TooManyParameters);
        }
        let mut unicast_locator_list: heapless::Vec<Locator, 8> = heapless::Vec::new();
        for _ in 0..unicast_count {
            let loc = Locator::parse(&mut cur)?;
            unicast_locator_list
                .push(loc)
                .map_err(|_| RtpsError::TooManyParameters)?;
        }

        let mut multicast_locator_list: heapless::Vec<Locator, 8> = heapless::Vec::new();
        if multicast_flag {
            let multicast_count = cur.read_u32()? as usize;
            if multicast_count > 8 {
                return Err(RtpsError::TooManyParameters);
            }
            for _ in 0..multicast_count {
                let loc = Locator::parse(&mut cur)?;
                multicast_locator_list
                    .push(loc)
                    .map_err(|_| RtpsError::TooManyParameters)?;
            }
        }

        Ok(Self {
            endianness,
            multicast_flag,
            unicast_locator_list,
            multicast_locator_list,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_u32(self.unicast_locator_list.len() as u32)?;
        for loc in &self.unicast_locator_list {
            loc.serialize(w)?;
        }
        if self.multicast_flag {
            w.write_u32(self.multicast_locator_list.len() as u32)?;
            for loc in &self.multicast_locator_list {
                loc.serialize(w)?;
            }
        }
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        let mut f = self.endianness.into_flags(0u8);
        if self.multicast_flag {
            f |= 0x02;
        }
        f
    }
}

/// INFO_REPLY_IP4 submessage body (RTPS 2.3 Section 8.3.7.6.1).
///
/// Flags: E(0) M(1:multicast)
/// Body: unicast_locator(24) + optional multicast_locator(24)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfoReplyIp4 {
    pub endianness: Endianness,
    pub multicast_flag: bool,
    pub unicast_locator: Locator,
    pub multicast_locator: Option<Locator>,
}

impl InfoReplyIp4 {
    pub fn parse(flags: u8, body: &[u8]) -> Result<Self, RtpsError> {
        let endianness = Endianness::from_flags(flags);
        let multicast_flag = flags & 0x02 != 0;

        let mut cur = ByteCursor::new(body, endianness);
        let unicast_locator = Locator::parse(&mut cur)?;
        let multicast_locator = if multicast_flag {
            Some(Locator::parse(&mut cur)?)
        } else {
            None
        };

        Ok(Self {
            endianness,
            multicast_flag,
            unicast_locator,
            multicast_locator,
        })
    }

    pub fn serialize_body(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        self.unicast_locator.serialize(w)?;
        if let Some(ref mc) = self.multicast_locator {
            mc.serialize(w)?;
        }
        Ok(())
    }

    pub fn flags_byte(&self) -> u8 {
        let mut f = self.endianness.into_flags(0u8);
        if self.multicast_flag {
            f |= 0x02;
        }
        f
    }
}
