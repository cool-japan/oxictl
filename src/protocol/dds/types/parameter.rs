use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};
use crate::protocol::dds::error::RtpsError;
use heapless::Vec;

// ─── PID constants ───────────────────────────────────────────────────────────

pub const PID_PAD: u16 = 0x0000;
pub const PID_SENTINEL: u16 = 0x0001;
pub const PID_USER_DATA: u16 = 0x002C;
pub const PID_TOPIC_NAME: u16 = 0x0005;
pub const PID_TYPE_NAME: u16 = 0x0007;
pub const PID_GROUP_DATA: u16 = 0x002D;
pub const PID_TOPIC_DATA: u16 = 0x002E;
pub const PID_DURABILITY: u16 = 0x001D;
pub const PID_DURABILITY_SERVICE: u16 = 0x001E;
pub const PID_DEADLINE: u16 = 0x0023;
pub const PID_LATENCY_BUDGET: u16 = 0x0027;
pub const PID_LIVELINESS: u16 = 0x001B;
pub const PID_RELIABILITY: u16 = 0x001A;
pub const PID_LIFESPAN: u16 = 0x002B;
pub const PID_DESTINATION_ORDER: u16 = 0x0025;
pub const PID_HISTORY: u16 = 0x0040;
pub const PID_RESOURCE_LIMITS: u16 = 0x0041;
pub const PID_OWNERSHIP: u16 = 0x001F;
pub const PID_OWNERSHIP_STRENGTH: u16 = 0x0006;
pub const PID_PRESENTATION: u16 = 0x0021;
pub const PID_PARTITION: u16 = 0x0029;
pub const PID_TIME_BASED_FILTER: u16 = 0x0004;
pub const PID_TRANSPORT_PRIORITY: u16 = 0x0049;
pub const PID_PROTOCOL_VERSION: u16 = 0x0015;
pub const PID_VENDORID: u16 = 0x0016;
pub const PID_VENDOR_ID: u16 = PID_VENDORID;
pub const PID_UNICAST_LOCATOR: u16 = 0x002F;
pub const PID_MULTICAST_LOCATOR: u16 = 0x0030;
pub const PID_DEFAULT_UNICAST_LOCATOR: u16 = 0x0031;
pub const PID_DEFAULT_MULTICAST_LOCATOR: u16 = 0x0048;
pub const PID_METATRAFFIC_UNICAST_LOCATOR: u16 = 0x0032;
pub const PID_METATRAFFIC_MULTICAST_LOCATOR: u16 = 0x0033;
pub const PID_EXPECTS_INLINE_QOS: u16 = 0x0043;
pub const PID_PARTICIPANT_MANUAL_LIVELINESS_COUNT: u16 = 0x0034;
pub const PID_PARTICIPANT_BUILTIN_ENDPOINTS: u16 = 0x0044;
pub const PID_PARTICIPANT_LEASE_DURATION: u16 = 0x0002;
pub const PID_CONTENT_FILTER_PROPERTY: u16 = 0x0035;
pub const PID_PARTICIPANT_GUID: u16 = 0x0050;
pub const PID_PARTICIPANT_ENTITYID: u16 = 0x0051;
pub const PID_GROUP_GUID: u16 = 0x0052;
pub const PID_GROUP_ENTITYID: u16 = 0x0053;
pub const PID_BUILTIN_ENDPOINT_SET: u16 = 0x0058;
pub const PID_PROPERTY_LIST: u16 = 0x0059;
pub const PID_TYPE_MAX_SIZE_SERIALIZED: u16 = 0x0060;
pub const PID_ENTITY_NAME: u16 = 0x0062;
pub const PID_KEY_HASH: u16 = 0x0070;
pub const PID_STATUS_INFO: u16 = 0x0071;
pub const PID_ENDPOINT_GUID: u16 = 0x005A;

// ─── Parameter ───────────────────────────────────────────────────────────────

/// A single RTPS ParameterList entry. Zero-copy: `value` borrows from the input slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parameter<'a> {
    pub pid: u16,
    pub value: &'a [u8],
}

// ─── ParameterList ────────────────────────────────────────────────────────────

/// A bounded list of RTPS parameters. Capacity: 32 entries.
pub struct ParameterList<'a> {
    params: Vec<Parameter<'a>, 32>,
}

impl<'a> ParameterList<'a> {
    /// Create an empty parameter list.
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    /// Push a parameter. Returns `TooManyParameters` if the list is already at capacity.
    pub fn push(&mut self, p: Parameter<'a>) -> Result<(), RtpsError> {
        self.params
            .push(p)
            .map_err(|_| RtpsError::TooManyParameters)
    }

    /// Iterate over all parameters in order.
    pub fn iter(&self) -> impl Iterator<Item = &Parameter<'a>> {
        self.params.iter()
    }

    /// Find the first parameter with the given PID.
    pub fn find(&self, pid: u16) -> Option<&Parameter<'a>> {
        self.params.iter().find(|p| p.pid == pid)
    }

    /// Number of parameters in the list.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// True if the list contains no parameters.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Byte length when serialized: sum of aligned entries + 4-byte sentinel.
    ///
    /// Each entry occupies 4 (pid + length) + `value.len()` rounded up to 4-byte boundary.
    pub fn serialized_len(&self) -> usize {
        let mut total = 0usize;
        for p in &self.params {
            let aligned_len = (p.value.len() + 3) & !3;
            total += 4 + aligned_len;
        }
        // sentinel: pid=0x0001 (2 bytes) + length=0 (2 bytes)
        total + 4
    }

    /// Parse a PID_SENTINEL-terminated parameter list from the cursor.
    ///
    /// Each entry: `pid` u16 | `length` u16 | `value` bytes (`length` bytes, 4-byte aligned).
    /// PID_PAD entries are skipped. Stops at PID_SENTINEL.
    /// Returns `TooManyParameters` if more than 32 non-PAD entries are encountered.
    pub fn parse(cur: &mut ByteCursor<'a>) -> Result<Self, RtpsError> {
        let mut list = Self::new();
        loop {
            let pid = cur.read_u16()?;
            let length = cur.read_u16()? as usize;

            if pid == PID_SENTINEL {
                // Sentinel has no value; parsing complete.
                break;
            }
            if pid == PID_PAD {
                // PAD entries are skipped; advance past declared length bytes.
                cur.skip(length)?;
                // Align to 4-byte boundary after skip.
                cur.align_to(4)?;
                continue;
            }

            // Validate that `length` bytes are available.
            if cur.remaining() < length {
                return Err(RtpsError::InvalidParameterLength);
            }
            let value = cur.read_bytes(length)?;

            // Advance to next 4-byte boundary.
            cur.align_to(4)?;

            list.push(Parameter { pid, value })?;
        }
        Ok(list)
    }

    /// Serialize the parameter list into the writer.
    ///
    /// Each parameter: `pid` u16 | `length` u16 | `value` bytes | zero padding to 4-byte boundary.
    /// Terminated by PID_SENTINEL with length 0.
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        for p in &self.params {
            w.write_u16(p.pid)?;
            // Length must be 4-byte aligned per CDR.
            let aligned_len = (p.value.len() + 3) & !3;
            w.write_u16(aligned_len as u16)?;
            w.write_bytes(p.value)?;
            // Zero padding to reach aligned boundary.
            let pad = aligned_len - p.value.len();
            if pad > 0 {
                let zeros = [0u8; 3];
                w.write_bytes(&zeros[..pad])?;
            }
        }
        // Write sentinel.
        w.write_u16(PID_SENTINEL)?;
        w.write_u16(0)?;
        Ok(())
    }
}

impl Default for ParameterList<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for ParameterList<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ParameterList")
            .field("params", &self.params)
            .finish()
    }
}

impl PartialEq for ParameterList<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params
    }
}

impl Eq for ParameterList<'_> {}

impl<'a> Clone for ParameterList<'a> {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
        }
    }
}
