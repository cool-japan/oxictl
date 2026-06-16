use heapless::Vec;

use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::types::guid::{
    Guid, ProtocolVersion, VendorId, GUID_UNKNOWN, PROTOCOL_VERSION_2_3, VENDOR_ID_OXICTL,
};
use crate::protocol::dds::types::locator::Locator;
use crate::protocol::dds::types::parameter::{
    ParameterList, PID_BUILTIN_ENDPOINT_SET, PID_DEFAULT_MULTICAST_LOCATOR,
    PID_DEFAULT_UNICAST_LOCATOR, PID_METATRAFFIC_MULTICAST_LOCATOR,
    PID_METATRAFFIC_UNICAST_LOCATOR, PID_PARTICIPANT_GUID, PID_PARTICIPANT_LEASE_DURATION,
    PID_PARTICIPANT_MANUAL_LIVELINESS_COUNT, PID_PROTOCOL_VERSION, PID_SENTINEL, PID_VENDOR_ID,
};
use crate::protocol::dds::types::time::Duration;

use super::error::DiscoveryError;

// ─── Builtin endpoint set constants ──────────────────────────────────────────

pub const BUILTIN_ENDPOINT_PARTICIPANT_ANNOUNCER: u32 = 0x00000001;
pub const BUILTIN_ENDPOINT_PARTICIPANT_DETECTOR: u32 = 0x00000002;
pub const BUILTIN_ENDPOINT_PUBLICATIONS_ANNOUNCER: u32 = 0x00000004;
pub const BUILTIN_ENDPOINT_PUBLICATIONS_DETECTOR: u32 = 0x00000008;
pub const BUILTIN_ENDPOINT_SUBSCRIPTIONS_ANNOUNCER: u32 = 0x00000010;
pub const BUILTIN_ENDPOINT_SUBSCRIPTIONS_DETECTOR: u32 = 0x00000020;

// ─── ParticipantBuiltinTopicData ──────────────────────────────────────────────

/// Data announced by each RTPS participant via SPDP.
///
/// CDR-encoded as a PL_CDR_LE (or PL_CDR_BE) ParameterList in DATA submessage payloads.
/// Uses `heapless::Vec` for bounded, `no_std`-compatible storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParticipantBuiltinTopicData {
    pub participant_guid: Guid,
    pub protocol_version: ProtocolVersion,
    pub vendor_id: VendorId,
    pub metatraffic_unicast_locators: Vec<Locator, 4>,
    pub metatraffic_multicast_locators: Vec<Locator, 4>,
    pub default_unicast_locators: Vec<Locator, 4>,
    pub default_multicast_locators: Vec<Locator, 4>,
    pub lease_duration: Duration,
    pub builtin_endpoint_set: u32,
    pub manual_liveliness_count: i32,
}

impl Default for ParticipantBuiltinTopicData {
    fn default() -> Self {
        Self {
            participant_guid: GUID_UNKNOWN,
            protocol_version: PROTOCOL_VERSION_2_3,
            vendor_id: VENDOR_ID_OXICTL,
            metatraffic_unicast_locators: Vec::new(),
            metatraffic_multicast_locators: Vec::new(),
            default_unicast_locators: Vec::new(),
            default_multicast_locators: Vec::new(),
            lease_duration: Duration {
                seconds: 10,
                fraction: 0,
            },
            builtin_endpoint_set: BUILTIN_ENDPOINT_PARTICIPANT_ANNOUNCER
                | BUILTIN_ENDPOINT_PARTICIPANT_DETECTOR,
            manual_liveliness_count: 0,
        }
    }
}

impl ParticipantBuiltinTopicData {
    /// Serialize this participant data into a CDR PL_CDR_LE payload.
    ///
    /// The first 4 bytes are the CDR encapsulation header `[0x00, 0x03, 0x00, 0x00]`
    /// (PL_CDR_LE), followed by the ParameterList entries and a PID_SENTINEL terminator.
    ///
    /// Returns the total number of bytes written into `buf`.
    pub fn serialize_to_payload(&self, buf: &mut [u8]) -> Result<usize, RtpsError> {
        if buf.len() < 4 {
            return Err(RtpsError::BufferTooSmall);
        }
        // CDR PL_LE representation header
        buf[0] = 0x00;
        buf[1] = 0x03;
        buf[2] = 0x00;
        buf[3] = 0x00;

        let mut w = ByteWriter::new(&mut buf[4..], Endianness::Little);

        write_param_version_and_vendor(&mut w, &self.protocol_version, &self.vendor_id)?;
        write_param_guid(&mut w, PID_PARTICIPANT_GUID, &self.participant_guid)?;

        for loc in &self.metatraffic_unicast_locators {
            write_param_locator(&mut w, PID_METATRAFFIC_UNICAST_LOCATOR, loc)?;
        }
        for loc in &self.metatraffic_multicast_locators {
            write_param_locator(&mut w, PID_METATRAFFIC_MULTICAST_LOCATOR, loc)?;
        }
        for loc in &self.default_unicast_locators {
            write_param_locator(&mut w, PID_DEFAULT_UNICAST_LOCATOR, loc)?;
        }
        for loc in &self.default_multicast_locators {
            write_param_locator(&mut w, PID_DEFAULT_MULTICAST_LOCATOR, loc)?;
        }

        write_param_duration(&mut w, PID_PARTICIPANT_LEASE_DURATION, &self.lease_duration)?;
        write_param_u32(&mut w, PID_BUILTIN_ENDPOINT_SET, self.builtin_endpoint_set)?;
        write_param_i32(
            &mut w,
            PID_PARTICIPANT_MANUAL_LIVELINESS_COUNT,
            self.manual_liveliness_count,
        )?;

        // PID_SENTINEL: pid=0x0001, length=0
        w.write_u16(PID_SENTINEL)?;
        w.write_u16(0)?;

        Ok(4 + w.position())
    }

    /// Parse participant data from a CDR PL_CDR_LE or PL_CDR_BE payload.
    ///
    /// The first 4 bytes must be the CDR encapsulation header:
    /// - `[0x00, 0x02, 0x00, 0x00]` = PL_CDR_BE
    /// - `[0x00, 0x03, 0x00, 0x00]` = PL_CDR_LE (most common)
    /// - Anything else: assumed LE.
    ///
    /// Unknown PIDs are silently skipped for forward compatibility.
    pub fn parse_from_payload(payload: &[u8]) -> Result<Self, DiscoveryError> {
        if payload.len() < 4 {
            return Err(DiscoveryError::PayloadTooSmall);
        }
        // Detect endianness from CDR rep-ID (bytes 0–1).
        // PL_CDR_BE = [0x00, 0x02], PL_CDR_LE = [0x00, 0x03]
        let endianness = if payload[1] == 0x02 {
            Endianness::Big
        } else {
            Endianness::Little // default: PL_CDR_LE
        };

        let param_bytes = &payload[4..];
        let mut cur = ByteCursor::new(param_bytes, endianness);
        let list = ParameterList::parse(&mut cur)?;

        let mut data = Self::default();
        for param in list.iter() {
            let mut vc = ByteCursor::new(param.value, endianness);
            match param.pid {
                PID_PROTOCOL_VERSION if param.value.len() >= 2 => {
                    data.protocol_version = ProtocolVersion::parse(&mut vc)?;
                }
                PID_VENDOR_ID if param.value.len() >= 2 => {
                    data.vendor_id = VendorId::parse(&mut vc)?;
                }
                PID_PARTICIPANT_GUID if param.value.len() >= 16 => {
                    data.participant_guid = Guid::parse(&mut vc)?;
                }
                PID_METATRAFFIC_UNICAST_LOCATOR if param.value.len() >= 24 => {
                    let loc = Locator::parse(&mut vc)?;
                    let _ = data.metatraffic_unicast_locators.push(loc);
                }
                PID_METATRAFFIC_MULTICAST_LOCATOR if param.value.len() >= 24 => {
                    let loc = Locator::parse(&mut vc)?;
                    let _ = data.metatraffic_multicast_locators.push(loc);
                }
                PID_DEFAULT_UNICAST_LOCATOR if param.value.len() >= 24 => {
                    let loc = Locator::parse(&mut vc)?;
                    let _ = data.default_unicast_locators.push(loc);
                }
                PID_DEFAULT_MULTICAST_LOCATOR if param.value.len() >= 24 => {
                    let loc = Locator::parse(&mut vc)?;
                    let _ = data.default_multicast_locators.push(loc);
                }
                PID_PARTICIPANT_LEASE_DURATION if param.value.len() >= 8 => {
                    data.lease_duration = Duration::parse(&mut vc)?;
                }
                PID_BUILTIN_ENDPOINT_SET if param.value.len() >= 4 => {
                    data.builtin_endpoint_set = vc.read_u32()?;
                }
                PID_PARTICIPANT_MANUAL_LIVELINESS_COUNT if param.value.len() >= 4 => {
                    data.manual_liveliness_count = vc.read_i32()?;
                }
                _ => {} // unknown PIDs or undersized values ignored (forward compat)
            }
        }

        Ok(data)
    }
}

// ─── Private serialization helpers ───────────────────────────────────────────

/// Write one parameter entry: pid(u16 LE) + aligned_length(u16 LE) + value bytes + zero padding.
fn write_param(w: &mut ByteWriter<'_>, pid: u16, value_bytes: &[u8]) -> Result<(), RtpsError> {
    let aligned_len = (value_bytes.len() + 3) & !3;
    w.write_u16(pid)?;
    w.write_u16(aligned_len as u16)?;
    w.write_bytes(value_bytes)?;
    let pad = aligned_len - value_bytes.len();
    if pad > 0 {
        w.write_bytes(&[0u8, 0u8, 0u8][..pad])?;
    }
    Ok(())
}

/// Serialize one locator into a 24-byte stack buffer, then write as parameter.
fn write_param_locator(w: &mut ByteWriter<'_>, pid: u16, loc: &Locator) -> Result<(), RtpsError> {
    let mut lbuf = [0u8; 24];
    {
        let mut lw = ByteWriter::new(&mut lbuf, Endianness::Little);
        loc.serialize(&mut lw)?;
    }
    write_param(w, pid, &lbuf)
}

fn write_param_guid(w: &mut ByteWriter<'_>, pid: u16, guid: &Guid) -> Result<(), RtpsError> {
    let mut gbuf = [0u8; 16];
    {
        let mut gw = ByteWriter::new(&mut gbuf, Endianness::Little);
        guid.serialize(&mut gw)?;
    }
    write_param(w, pid, &gbuf)
}

fn write_param_u32(w: &mut ByteWriter<'_>, pid: u16, v: u32) -> Result<(), RtpsError> {
    let bytes = v.to_le_bytes();
    write_param(w, pid, &bytes)
}

fn write_param_i32(w: &mut ByteWriter<'_>, pid: u16, v: i32) -> Result<(), RtpsError> {
    let bytes = v.to_le_bytes();
    write_param(w, pid, &bytes)
}

fn write_param_duration(w: &mut ByteWriter<'_>, pid: u16, d: &Duration) -> Result<(), RtpsError> {
    let mut dbuf = [0u8; 8];
    {
        let mut dw = ByteWriter::new(&mut dbuf, Endianness::Little);
        d.serialize(&mut dw)?;
    }
    write_param(w, pid, &dbuf)
}

/// Write PID_PROTOCOL_VERSION and PID_VENDOR_ID as individual parameters.
fn write_param_version_and_vendor(
    w: &mut ByteWriter<'_>,
    version: &ProtocolVersion,
    vendor: &VendorId,
) -> Result<(), RtpsError> {
    // PID_PROTOCOL_VERSION: value = [major, minor, 0, 0] (4 bytes, already aligned)
    write_param(
        w,
        PID_PROTOCOL_VERSION,
        &[version.major, version.minor, 0, 0],
    )?;
    // PID_VENDOR_ID: value = [v0, v1, 0, 0] (4 bytes, already aligned)
    write_param(w, PID_VENDOR_ID, &[vendor.0[0], vendor.0[1], 0, 0])
}
