//! SEDP builtin topic data: publications and subscriptions.
//!
//! `PublicationBuiltinTopicData` and `SubscriptionBuiltinTopicData` carry the endpoint
//! metadata exchanged over SEDP.  Both encode/decode as CDR PL_CDR_LE ParameterList
//! payloads, identical to SPDP participant data.

use heapless::{String as HString, Vec as HVec};

use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::types::guid::{Guid, GUID_UNKNOWN};
use crate::protocol::dds::types::locator::Locator;
use crate::protocol::dds::types::parameter::{
    ParameterList, PID_DEADLINE, PID_DURABILITY, PID_ENDPOINT_GUID, PID_HISTORY, PID_LIVELINESS,
    PID_MULTICAST_LOCATOR, PID_RELIABILITY, PID_SENTINEL, PID_TOPIC_NAME, PID_TYPE_NAME,
    PID_UNICAST_LOCATOR,
};

use super::qos::{
    DeadlineQosPolicy, DurabilityQosPolicy, HistoryQosPolicy, LivelinessQosPolicy,
    ReliabilityQosPolicy,
};

// ─── PublicationBuiltinTopicData ─────────────────────────────────────────────

/// SEDP builtin topic data for a publication (writer) endpoint.
///
/// Serialized as a CDR PL_CDR_LE ParameterList in DATA submessage payloads
/// on the builtin publications topic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicationBuiltinTopicData {
    pub endpoint_guid: Guid,
    pub topic_name: HString<256>,
    pub type_name: HString<256>,
    pub unicast_locators: HVec<Locator, 4>,
    pub multicast_locators: HVec<Locator, 4>,
    pub reliability: ReliabilityQosPolicy,
    pub history: HistoryQosPolicy,
    pub durability: DurabilityQosPolicy,
    pub liveliness: LivelinessQosPolicy,
    pub deadline: DeadlineQosPolicy,
}

impl Default for PublicationBuiltinTopicData {
    fn default() -> Self {
        Self {
            endpoint_guid: GUID_UNKNOWN,
            topic_name: HString::new(),
            type_name: HString::new(),
            unicast_locators: HVec::new(),
            multicast_locators: HVec::new(),
            reliability: ReliabilityQosPolicy::default(),
            history: HistoryQosPolicy::default(),
            durability: DurabilityQosPolicy::default(),
            liveliness: LivelinessQosPolicy::default(),
            deadline: DeadlineQosPolicy::default(),
        }
    }
}

impl PublicationBuiltinTopicData {
    /// Serialize this data into a CDR PL_CDR_LE payload.
    ///
    /// Writes a 4-byte CDR encapsulation header `[0x00, 0x03, 0x00, 0x00]` (PL_CDR_LE)
    /// followed by the ParameterList entries and a PID_SENTINEL terminator.
    ///
    /// Returns the total number of bytes written into `buf`.
    pub fn serialize_to_payload(&self, buf: &mut [u8]) -> Result<usize, RtpsError> {
        if buf.len() < 4 {
            return Err(RtpsError::BufferTooSmall);
        }
        buf[0] = 0x00;
        buf[1] = 0x03;
        buf[2] = 0x00;
        buf[3] = 0x00;

        let mut w = ByteWriter::new(&mut buf[4..], Endianness::Little);

        write_param_guid(&mut w, PID_ENDPOINT_GUID, &self.endpoint_guid)?;
        write_param_string(&mut w, PID_TOPIC_NAME, self.topic_name.as_str())?;
        write_param_string(&mut w, PID_TYPE_NAME, self.type_name.as_str())?;

        for loc in &self.unicast_locators {
            write_param_locator(&mut w, PID_UNICAST_LOCATOR, loc)?;
        }
        for loc in &self.multicast_locators {
            write_param_locator(&mut w, PID_MULTICAST_LOCATOR, loc)?;
        }

        // Reliability (12 bytes)
        let mut rbuf = [0u8; 12];
        {
            let mut rw = ByteWriter::new(&mut rbuf, Endianness::Little);
            self.reliability.serialize(&mut rw)?;
        }
        write_param_raw(&mut w, PID_RELIABILITY, &rbuf)?;

        // History (8 bytes)
        let mut hbuf = [0u8; 8];
        {
            let mut hw = ByteWriter::new(&mut hbuf, Endianness::Little);
            self.history.serialize(&mut hw)?;
        }
        write_param_raw(&mut w, PID_HISTORY, &hbuf)?;

        // Durability (4 bytes)
        let mut dbuf = [0u8; 4];
        {
            let mut dw = ByteWriter::new(&mut dbuf, Endianness::Little);
            self.durability.serialize(&mut dw)?;
        }
        write_param_raw(&mut w, PID_DURABILITY, &dbuf)?;

        // Liveliness (12 bytes)
        let mut lbuf = [0u8; 12];
        {
            let mut lw = ByteWriter::new(&mut lbuf, Endianness::Little);
            self.liveliness.serialize(&mut lw)?;
        }
        write_param_raw(&mut w, PID_LIVELINESS, &lbuf)?;

        // Deadline (8 bytes)
        let mut dlbuf = [0u8; 8];
        {
            let mut dlw = ByteWriter::new(&mut dlbuf, Endianness::Little);
            self.deadline.serialize(&mut dlw)?;
        }
        write_param_raw(&mut w, PID_DEADLINE, &dlbuf)?;

        // PID_SENTINEL
        w.write_u16(PID_SENTINEL)?;
        w.write_u16(0)?;

        Ok(4 + w.position())
    }

    /// Parse publication endpoint data from a CDR PL_CDR_LE or PL_CDR_BE payload.
    ///
    /// Detects endianness from the CDR encapsulation header. Unknown PIDs are silently
    /// skipped for forward compatibility.
    pub fn parse_from_payload(payload: &[u8]) -> Result<Self, RtpsError> {
        if payload.len() < 4 {
            return Err(RtpsError::TruncatedHeader);
        }
        let endianness = if payload[1] == 0x02 {
            Endianness::Big
        } else {
            Endianness::Little
        };

        let param_bytes = &payload[4..];
        let mut cur = ByteCursor::new(param_bytes, endianness);
        let list = ParameterList::parse(&mut cur)?;

        let mut data = Self::default();
        for param in list.iter() {
            match param.pid {
                PID_ENDPOINT_GUID if param.value.len() >= 16 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.endpoint_guid = Guid::parse(&mut vc)?;
                }
                PID_TOPIC_NAME => {
                    data.topic_name = read_param_cdr_string::<256>(param.value, endianness)?;
                }
                PID_TYPE_NAME => {
                    data.type_name = read_param_cdr_string::<256>(param.value, endianness)?;
                }
                PID_UNICAST_LOCATOR if param.value.len() >= 24 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    let loc = Locator::parse(&mut vc)?;
                    let _ = data.unicast_locators.push(loc);
                }
                PID_MULTICAST_LOCATOR if param.value.len() >= 24 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    let loc = Locator::parse(&mut vc)?;
                    let _ = data.multicast_locators.push(loc);
                }
                PID_RELIABILITY if param.value.len() >= 12 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.reliability = ReliabilityQosPolicy::parse(&mut vc)?;
                }
                PID_HISTORY if param.value.len() >= 8 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.history = HistoryQosPolicy::parse(&mut vc)?;
                }
                PID_DURABILITY if param.value.len() >= 4 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.durability = DurabilityQosPolicy::parse(&mut vc)?;
                }
                PID_LIVELINESS if param.value.len() >= 12 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.liveliness = LivelinessQosPolicy::parse(&mut vc)?;
                }
                PID_DEADLINE if param.value.len() >= 8 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.deadline = DeadlineQosPolicy::parse(&mut vc)?;
                }
                _ => {} // unknown PIDs silently skipped
            }
        }
        Ok(data)
    }
}

// ─── SubscriptionBuiltinTopicData ────────────────────────────────────────────

/// SEDP builtin topic data for a subscription (reader) endpoint.
///
/// Serialized as a CDR PL_CDR_LE ParameterList in DATA submessage payloads
/// on the builtin subscriptions topic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubscriptionBuiltinTopicData {
    pub endpoint_guid: Guid,
    pub topic_name: HString<256>,
    pub type_name: HString<256>,
    pub unicast_locators: HVec<Locator, 4>,
    pub multicast_locators: HVec<Locator, 4>,
    pub reliability: ReliabilityQosPolicy,
    pub history: HistoryQosPolicy,
    pub durability: DurabilityQosPolicy,
    pub liveliness: LivelinessQosPolicy,
    pub deadline: DeadlineQosPolicy,
}

impl Default for SubscriptionBuiltinTopicData {
    fn default() -> Self {
        Self {
            endpoint_guid: GUID_UNKNOWN,
            topic_name: HString::new(),
            type_name: HString::new(),
            unicast_locators: HVec::new(),
            multicast_locators: HVec::new(),
            reliability: ReliabilityQosPolicy::default(),
            history: HistoryQosPolicy::default(),
            durability: DurabilityQosPolicy::default(),
            liveliness: LivelinessQosPolicy::default(),
            deadline: DeadlineQosPolicy::default(),
        }
    }
}

impl SubscriptionBuiltinTopicData {
    /// Serialize this data into a CDR PL_CDR_LE payload.
    pub fn serialize_to_payload(&self, buf: &mut [u8]) -> Result<usize, RtpsError> {
        if buf.len() < 4 {
            return Err(RtpsError::BufferTooSmall);
        }
        buf[0] = 0x00;
        buf[1] = 0x03;
        buf[2] = 0x00;
        buf[3] = 0x00;

        let mut w = ByteWriter::new(&mut buf[4..], Endianness::Little);

        write_param_guid(&mut w, PID_ENDPOINT_GUID, &self.endpoint_guid)?;
        write_param_string(&mut w, PID_TOPIC_NAME, self.topic_name.as_str())?;
        write_param_string(&mut w, PID_TYPE_NAME, self.type_name.as_str())?;

        for loc in &self.unicast_locators {
            write_param_locator(&mut w, PID_UNICAST_LOCATOR, loc)?;
        }
        for loc in &self.multicast_locators {
            write_param_locator(&mut w, PID_MULTICAST_LOCATOR, loc)?;
        }

        let mut rbuf = [0u8; 12];
        {
            let mut rw = ByteWriter::new(&mut rbuf, Endianness::Little);
            self.reliability.serialize(&mut rw)?;
        }
        write_param_raw(&mut w, PID_RELIABILITY, &rbuf)?;

        let mut hbuf = [0u8; 8];
        {
            let mut hw = ByteWriter::new(&mut hbuf, Endianness::Little);
            self.history.serialize(&mut hw)?;
        }
        write_param_raw(&mut w, PID_HISTORY, &hbuf)?;

        let mut dbuf = [0u8; 4];
        {
            let mut dw = ByteWriter::new(&mut dbuf, Endianness::Little);
            self.durability.serialize(&mut dw)?;
        }
        write_param_raw(&mut w, PID_DURABILITY, &dbuf)?;

        let mut lbuf = [0u8; 12];
        {
            let mut lw = ByteWriter::new(&mut lbuf, Endianness::Little);
            self.liveliness.serialize(&mut lw)?;
        }
        write_param_raw(&mut w, PID_LIVELINESS, &lbuf)?;

        let mut dlbuf = [0u8; 8];
        {
            let mut dlw = ByteWriter::new(&mut dlbuf, Endianness::Little);
            self.deadline.serialize(&mut dlw)?;
        }
        write_param_raw(&mut w, PID_DEADLINE, &dlbuf)?;

        w.write_u16(PID_SENTINEL)?;
        w.write_u16(0)?;

        Ok(4 + w.position())
    }

    /// Parse subscription endpoint data from a CDR PL_CDR_LE or PL_CDR_BE payload.
    pub fn parse_from_payload(payload: &[u8]) -> Result<Self, RtpsError> {
        if payload.len() < 4 {
            return Err(RtpsError::TruncatedHeader);
        }
        let endianness = if payload[1] == 0x02 {
            Endianness::Big
        } else {
            Endianness::Little
        };

        let param_bytes = &payload[4..];
        let mut cur = ByteCursor::new(param_bytes, endianness);
        let list = ParameterList::parse(&mut cur)?;

        let mut data = Self::default();
        for param in list.iter() {
            match param.pid {
                PID_ENDPOINT_GUID if param.value.len() >= 16 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.endpoint_guid = Guid::parse(&mut vc)?;
                }
                PID_TOPIC_NAME => {
                    data.topic_name = read_param_cdr_string::<256>(param.value, endianness)?;
                }
                PID_TYPE_NAME => {
                    data.type_name = read_param_cdr_string::<256>(param.value, endianness)?;
                }
                PID_UNICAST_LOCATOR if param.value.len() >= 24 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    let loc = Locator::parse(&mut vc)?;
                    let _ = data.unicast_locators.push(loc);
                }
                PID_MULTICAST_LOCATOR if param.value.len() >= 24 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    let loc = Locator::parse(&mut vc)?;
                    let _ = data.multicast_locators.push(loc);
                }
                PID_RELIABILITY if param.value.len() >= 12 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.reliability = ReliabilityQosPolicy::parse(&mut vc)?;
                }
                PID_HISTORY if param.value.len() >= 8 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.history = HistoryQosPolicy::parse(&mut vc)?;
                }
                PID_DURABILITY if param.value.len() >= 4 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.durability = DurabilityQosPolicy::parse(&mut vc)?;
                }
                PID_LIVELINESS if param.value.len() >= 12 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.liveliness = LivelinessQosPolicy::parse(&mut vc)?;
                }
                PID_DEADLINE if param.value.len() >= 8 => {
                    let mut vc = ByteCursor::new(param.value, endianness);
                    data.deadline = DeadlineQosPolicy::parse(&mut vc)?;
                }
                _ => {}
            }
        }
        Ok(data)
    }
}

// ─── Private serialization helpers ───────────────────────────────────────────

/// Write one parameter entry: pid(u16 LE) + aligned_length(u16 LE) + value_bytes + zero padding.
fn write_param_raw(w: &mut ByteWriter<'_>, pid: u16, value_bytes: &[u8]) -> Result<(), RtpsError> {
    let aligned_len = (value_bytes.len() + 3) & !3;
    w.write_u16(pid)?;
    w.write_u16(aligned_len as u16)?;
    w.write_bytes(value_bytes)?;
    let pad = aligned_len - value_bytes.len();
    if pad > 0 {
        let zeros = [0u8; 3];
        w.write_bytes(&zeros[..pad])?;
    }
    Ok(())
}

/// Write a CDR string parameter entry.
///
/// The CDR string value = `[u32 length (includes null)][bytes][null][zero padding to 4-byte boundary]`.
/// Total parameter value length = `4 + aligned(len(s)+1, 4)`.
fn write_param_string(w: &mut ByteWriter<'_>, pid: u16, s: &str) -> Result<(), RtpsError> {
    let with_null = s.len() + 1; // content + null terminator
    let aligned_str = (with_null + 3) & !3; // aligned byte count for the string part
    let value_len = 4 + aligned_str; // u32 length prefix + aligned string content

    w.write_u16(pid)?;
    w.write_u16(value_len as u16)?;
    w.write_cdr_string(s)
}

/// Write a 16-byte GUID as a raw parameter.
fn write_param_guid(w: &mut ByteWriter<'_>, pid: u16, guid: &Guid) -> Result<(), RtpsError> {
    let mut buf = [0u8; 16];
    {
        let mut gw = ByteWriter::new(&mut buf, Endianness::Little);
        guid.serialize(&mut gw)?;
    }
    write_param_raw(w, pid, &buf)
}

/// Write a 24-byte Locator as a raw parameter.
fn write_param_locator(w: &mut ByteWriter<'_>, pid: u16, loc: &Locator) -> Result<(), RtpsError> {
    let mut buf = [0u8; 24];
    {
        let mut lw = ByteWriter::new(&mut buf, Endianness::Little);
        loc.serialize(&mut lw)?;
    }
    write_param_raw(w, pid, &buf)
}

// ─── Private parse helpers ────────────────────────────────────────────────────

/// Parse a CDR string from a raw parameter value slice.
///
/// Format: `[u32 length (includes null)][string bytes][null terminator][zero padding]`.
/// The returned string has the null terminator trimmed.
fn read_param_cdr_string<const N: usize>(
    value: &[u8],
    endianness: Endianness,
) -> Result<HString<N>, RtpsError> {
    let mut cur = ByteCursor::new(value, endianness);
    let s = cur.read_cdr_string()?;
    let mut result = HString::<N>::new();
    result.push_str(s).map_err(|_| RtpsError::BufferTooSmall)?;
    Ok(result)
}
