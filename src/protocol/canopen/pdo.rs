//! CANopen PDO (Process Data Object) communication.
//!
//! PDOs transmit real-time process data in CAN frames.
//! Each PDO carries up to 8 bytes of data.
//!
//! - RPDO: Receive PDO (node receives from network)
//! - TPDO: Transmit PDO (node sends to network)

/// PDO mapping entry: index + sub_index in object dictionary.
#[derive(Debug, Clone, Copy)]
pub struct PdoMapEntry {
    pub index: u16,
    pub sub_index: u8,
    /// Bit length of mapped object.
    pub bit_length: u8,
}

/// PDO communication parameters.
#[derive(Debug, Clone, Copy)]
pub struct PdoComm {
    /// COB-ID (CAN Identifier).
    pub cob_id: u32,
    /// Transmission type:
    ///   0 = acyclic synchronous
    ///   1-240 = every N SYNC
    ///   254 = event-driven (async)
    ///   255 = event-driven (async, RTR)
    pub transmission_type: u8,
    /// Inhibit time (100µs units).
    pub inhibit_time: u16,
    /// Event timer (ms).
    pub event_timer_ms: u16,
}

impl PdoComm {
    /// Default TPDO1 for node (COB-ID = 0x180 + node_id).
    pub fn tpdo1(node_id: u8) -> Self {
        Self {
            cob_id: 0x180 + node_id as u32,
            transmission_type: 1,
            inhibit_time: 0,
            event_timer_ms: 0,
        }
    }

    /// Default RPDO1 for node (COB-ID = 0x200 + node_id).
    pub fn rpdo1(node_id: u8) -> Self {
        Self {
            cob_id: 0x200 + node_id as u32,
            transmission_type: 254,
            inhibit_time: 0,
            event_timer_ms: 0,
        }
    }
}

/// A CAN frame (8 bytes max payload).
#[derive(Debug, Clone, Copy)]
pub struct CanFrame {
    pub cob_id: u32,
    pub dlc: u8, // data length code (0..8)
    pub data: [u8; 8],
}

impl CanFrame {
    pub fn new(cob_id: u32, data: &[u8]) -> Self {
        let dlc = data.len().min(8) as u8;
        let mut d = [0u8; 8];
        d[..dlc as usize].copy_from_slice(&data[..dlc as usize]);
        Self {
            cob_id,
            dlc,
            data: d,
        }
    }
}

/// PDO object: manages mapping and frame assembly/parsing.
///
/// `MAX_ENTRIES` = max OD entries mapped into this PDO (CiA 301: max 8 bytes → 1-8 entries).
#[derive(Debug)]
pub struct Pdo<const MAX_ENTRIES: usize> {
    pub comm: PdoComm,
    pub entries: [Option<PdoMapEntry>; MAX_ENTRIES],
    entry_count: usize,
    /// Raw PDO data (8 bytes).
    pub data: [u8; 8],
}

impl<const MAX_ENTRIES: usize> Pdo<MAX_ENTRIES> {
    pub fn new(comm: PdoComm) -> Self {
        Self {
            comm,
            entries: core::array::from_fn(|_| None),
            entry_count: 0,
            data: [0u8; 8],
        }
    }

    pub fn add_mapping(&mut self, entry: PdoMapEntry) -> bool {
        if self.entry_count >= MAX_ENTRIES {
            return false;
        }
        self.entries[self.entry_count] = Some(entry);
        self.entry_count += 1;
        true
    }

    /// Assemble PDO frame from current data.
    pub fn assemble(&self) -> CanFrame {
        CanFrame::new(self.comm.cob_id, &self.data)
    }

    /// Parse received CAN frame into PDO data.
    pub fn parse(&mut self, frame: &CanFrame) {
        if frame.cob_id != self.comm.cob_id {
            return;
        }
        let n = frame.dlc as usize;
        self.data[..n].copy_from_slice(&frame.data[..n]);
    }

    /// Write a u16 value at byte offset in PDO data.
    pub fn write_u16(&mut self, offset: usize, val: u16) {
        if offset + 1 < 8 {
            let b = val.to_le_bytes();
            self.data[offset] = b[0];
            self.data[offset + 1] = b[1];
        }
    }

    /// Read a u16 value at byte offset.
    pub fn read_u16(&self, offset: usize) -> u16 {
        if offset + 1 < 8 {
            u16::from_le_bytes([self.data[offset], self.data[offset + 1]])
        } else {
            0
        }
    }
}

// ─── Extended PDO types ───────────────────────────────────────────────────────

use super::object_dict::{OdEntryValue, OdError, StaticOd};

/// Error type for extended PDO operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdoError {
    /// The OD entry could not be read.
    OdReadError(OdError),
    /// The OD entry could not be written.
    OdWriteError(OdError),
    /// The packed PDO payload would exceed 8 bytes.
    PayloadTooLarge,
    /// The received CAN frame has an unexpected COB-ID.
    CobIdMismatch,
    /// The frame DLC is too short to decode the mapped entries.
    FrameTooShort,
    /// No mapping entries are configured.
    NoMappings,
}

/// A single OD-to-PDO mapping entry.
///
/// Records which OD object (`index`:`subindex`) maps to which byte `offset`
/// within the PDO payload.  `byte_count` is the width of the value (1, 2, or 4).
#[derive(Debug, Clone, Copy)]
pub struct PdoMapping {
    pub index: u16,
    pub subindex: u8,
    /// Byte offset within the 8-byte PDO payload.
    pub offset: usize,
    /// Width of the value in bytes (1, 2, or 4).
    pub byte_count: usize,
}

/// Transmit PDO communication parameters.
#[derive(Debug, Clone, Copy)]
pub struct TpdoConfig {
    /// CAN Object Identifier for this TPDO.
    pub cob_id: u32,
    /// Transmission type:
    ///   0=acyclic synchronous, 1-240=every N SYNC, 254/255=event-driven.
    pub transmission_type: u8,
    /// Event timer (ms); 0 = disabled.
    pub event_timer_ms: u16,
}

impl TpdoConfig {
    /// Default TPDO1 for `node_id` (event-driven, no timer).
    pub fn tpdo1(node_id: u8) -> Self {
        Self {
            cob_id: 0x180 + node_id as u32,
            transmission_type: 254,
            event_timer_ms: 0,
        }
    }

    /// Default TPDO2 for `node_id`.
    pub fn tpdo2(node_id: u8) -> Self {
        Self {
            cob_id: 0x280 + node_id as u32,
            transmission_type: 254,
            event_timer_ms: 0,
        }
    }
}

/// Receive PDO communication parameters.
#[derive(Debug, Clone, Copy)]
pub struct RpdoConfig {
    /// CAN Object Identifier for this RPDO.
    pub cob_id: u32,
    /// Transmission type (informational; same coding as TPDO).
    pub transmission_type: u8,
}

impl RpdoConfig {
    /// Default RPDO1 for `node_id`.
    pub fn rpdo1(node_id: u8) -> Self {
        Self {
            cob_id: 0x200 + node_id as u32,
            transmission_type: 254,
        }
    }

    /// Default RPDO2 for `node_id`.
    pub fn rpdo2(node_id: u8) -> Self {
        Self {
            cob_id: 0x300 + node_id as u32,
            transmission_type: 254,
        }
    }
}

// ─── TpdoProducer ─────────────────────────────────────────────────────────────

/// Transmit PDO producer: reads up to `N` OD entries and packs them into a
/// CAN frame in little-endian byte order.
///
/// `N` is the maximum number of OD-entry mappings (up to 8 is sensible, since
/// a PDO payload is at most 8 bytes).
pub struct TpdoProducer<const N: usize> {
    config: TpdoConfig,
    mappings: [Option<PdoMapping>; N],
    mapping_count: usize,
    /// Total payload bytes summed across all mappings.
    payload_bytes: usize,
    /// Event-timer accumulator (ms).
    timer_elapsed_ms: u32,
}

impl<const N: usize> TpdoProducer<N> {
    /// Create a new `TpdoProducer` with the given configuration.
    pub fn new(config: TpdoConfig) -> Self {
        Self {
            config,
            mappings: [None; N],
            mapping_count: 0,
            payload_bytes: 0,
            timer_elapsed_ms: 0,
        }
    }

    /// COB-ID used for transmitted frames.
    pub fn cob_id(&self) -> u32 {
        self.config.cob_id
    }

    /// Add a mapping from `(index, subindex)` in the OD to a byte range in
    /// the PDO payload.  The mapping records the offset and byte width.
    ///
    /// # Errors
    /// Returns `Err(PdoError::PayloadTooLarge)` if adding this entry would
    /// make the payload exceed 8 bytes, or if the mapping table is full.
    pub fn add_mapping(
        &mut self,
        index: u16,
        subindex: u8,
        byte_count: usize,
    ) -> Result<(), PdoError> {
        if self.mapping_count >= N || self.payload_bytes + byte_count > 8 {
            return Err(PdoError::PayloadTooLarge);
        }
        let offset = self.payload_bytes;
        self.mappings[self.mapping_count] = Some(PdoMapping {
            index,
            subindex,
            offset,
            byte_count,
        });
        self.mapping_count += 1;
        self.payload_bytes += byte_count;
        Ok(())
    }

    /// Number of configured mappings.
    pub fn mapping_count(&self) -> usize {
        self.mapping_count
    }

    /// Read all mapped OD entries and pack their values into a CAN frame.
    ///
    /// Values are packed in little-endian byte order starting from byte 0.
    pub fn produce<const OD: usize>(&self, od: &StaticOd<OD>) -> Result<CanFrame, PdoError> {
        if self.mapping_count == 0 {
            return Err(PdoError::NoMappings);
        }
        let mut payload = [0u8; 8];
        for slot in self.mappings[..self.mapping_count].iter() {
            let m = slot.ok_or(PdoError::NoMappings)?;
            let value = od
                .read(m.index, m.subindex)
                .map_err(PdoError::OdReadError)?;
            let bytes = value_to_le_bytes(value);
            let end = m.offset + m.byte_count;
            if end > 8 {
                return Err(PdoError::PayloadTooLarge);
            }
            payload[m.offset..end].copy_from_slice(&bytes[..m.byte_count]);
        }
        Ok(CanFrame::new(
            self.config.cob_id,
            &payload[..self.payload_bytes],
        ))
    }

    /// Tick the event timer by `dt_ms` milliseconds.
    ///
    /// Returns `true` when the event timer fires (i.e., a TPDO should be
    /// produced).  Always returns `false` if `event_timer_ms == 0`.
    pub fn tick(&mut self, dt_ms: u32) -> bool {
        if self.config.event_timer_ms == 0 {
            return false;
        }
        self.timer_elapsed_ms += dt_ms;
        if self.timer_elapsed_ms >= self.config.event_timer_ms as u32 {
            self.timer_elapsed_ms = 0;
            return true;
        }
        false
    }
}

// ─── RpdoConsumer ─────────────────────────────────────────────────────────────

/// Receive PDO consumer: unpacks a received CAN frame and writes the bytes
/// into up to `N` OD entries.
///
/// `N` is the maximum number of OD-entry mappings.
pub struct RpdoConsumer<const N: usize> {
    config: RpdoConfig,
    mappings: [Option<PdoMapping>; N],
    mapping_count: usize,
    /// Total payload bytes summed across all mappings.
    payload_bytes: usize,
    /// How many frames have been successfully consumed.
    frames_consumed: u32,
}

impl<const N: usize> RpdoConsumer<N> {
    /// Create a new `RpdoConsumer` with the given configuration.
    pub fn new(config: RpdoConfig) -> Self {
        Self {
            config,
            mappings: [None; N],
            mapping_count: 0,
            payload_bytes: 0,
            frames_consumed: 0,
        }
    }

    /// COB-ID this consumer listens on.
    pub fn cob_id(&self) -> u32 {
        self.config.cob_id
    }

    /// Total frames successfully consumed.
    pub fn frames_consumed(&self) -> u32 {
        self.frames_consumed
    }

    /// Add a mapping from a byte range in the PDO payload to `(index, subindex)`
    /// in the OD.  `byte_count` must match the target OD entry's byte width.
    ///
    /// # Errors
    /// Returns `Err(PdoError::PayloadTooLarge)` if adding this entry would
    /// make the expected payload exceed 8 bytes.
    pub fn add_mapping(
        &mut self,
        index: u16,
        subindex: u8,
        byte_count: usize,
    ) -> Result<(), PdoError> {
        if self.mapping_count >= N || self.payload_bytes + byte_count > 8 {
            return Err(PdoError::PayloadTooLarge);
        }
        let offset = self.payload_bytes;
        self.mappings[self.mapping_count] = Some(PdoMapping {
            index,
            subindex,
            offset,
            byte_count,
        });
        self.mapping_count += 1;
        self.payload_bytes += byte_count;
        Ok(())
    }

    /// Unpack a received CAN frame and write the values to the OD.
    ///
    /// The frame COB-ID must match the configured COB-ID.  Each mapped byte
    /// range in the payload is decoded using the *current* type of the OD
    /// entry (so the OD entry must already exist with the correct type).
    pub fn consume<const OD: usize>(
        &mut self,
        frame: &CanFrame,
        od: &mut StaticOd<OD>,
    ) -> Result<(), PdoError> {
        if frame.cob_id != self.config.cob_id {
            return Err(PdoError::CobIdMismatch);
        }
        if (frame.dlc as usize) < self.payload_bytes {
            return Err(PdoError::FrameTooShort);
        }
        for slot in self.mappings[..self.mapping_count].iter() {
            let m = slot.ok_or(PdoError::NoMappings)?;
            let end = m.offset + m.byte_count;
            if end > 8 {
                return Err(PdoError::PayloadTooLarge);
            }
            let slice = &frame.data[m.offset..end];

            // Read existing value to know the target type.
            let existing = od
                .read(m.index, m.subindex)
                .map_err(PdoError::OdReadError)?;
            let new_val = bytes_to_value(existing, slice).ok_or(PdoError::FrameTooShort)?;
            od.write(m.index, m.subindex, new_val)
                .map_err(PdoError::OdWriteError)?;
        }
        self.frames_consumed += 1;
        Ok(())
    }
}

// ─── Helper functions ─────────────────────────────────────────────────────────

/// Encode an `OdEntryValue` to a little-endian 8-byte array.
fn value_to_le_bytes(value: OdEntryValue) -> [u8; 8] {
    value.to_le_bytes()
}

/// Decode bytes from a PDO payload slice into the same variant as `existing`.
fn bytes_to_value(existing: OdEntryValue, bytes: &[u8]) -> Option<OdEntryValue> {
    match existing {
        OdEntryValue::U8(_) => bytes.first().map(|&b| OdEntryValue::U8(b)),
        OdEntryValue::I8(_) => bytes.first().map(|&b| OdEntryValue::I8(b as i8)),
        OdEntryValue::U16(_) => {
            if bytes.len() >= 2 {
                Some(OdEntryValue::U16(u16::from_le_bytes([bytes[0], bytes[1]])))
            } else {
                None
            }
        }
        OdEntryValue::I16(_) => {
            if bytes.len() >= 2 {
                Some(OdEntryValue::I16(i16::from_le_bytes([bytes[0], bytes[1]])))
            } else {
                None
            }
        }
        OdEntryValue::U32(_) => {
            if bytes.len() >= 4 {
                Some(OdEntryValue::U32(u32::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3],
                ])))
            } else {
                None
            }
        }
        OdEntryValue::I32(_) => {
            if bytes.len() >= 4 {
                Some(OdEntryValue::I32(i32::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3],
                ])))
            } else {
                None
            }
        }
        OdEntryValue::Bool(_) => bytes.first().map(|&b| OdEntryValue::Bool(b != 0)),
        OdEntryValue::OctetString(_) => {
            if bytes.len() >= 8 {
                let mut s = [0u8; 8];
                s.copy_from_slice(&bytes[..8]);
                Some(OdEntryValue::OctetString(s))
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::object_dict::{AccessType, DataType, OdEntry, OdEntryValue, StaticOd};
    use super::*;

    #[test]
    fn pdo_assemble_parse() {
        let comm = PdoComm::tpdo1(1);
        let mut pdo = Pdo::<4>::new(comm);
        pdo.write_u16(0, 0x6041);
        let frame = pdo.assemble();
        assert_eq!(frame.cob_id, 0x181);

        let mut rpdo = Pdo::<4>::new(PdoComm {
            cob_id: 0x181,
            ..comm
        });
        rpdo.parse(&frame);
        assert_eq!(rpdo.read_u16(0), 0x6041);
    }

    #[test]
    fn pdo_wrong_cob_id_ignored() {
        let comm = PdoComm::rpdo1(1);
        let mut pdo = Pdo::<2>::new(comm);
        let frame = CanFrame::new(0x999, &[0xFF]);
        pdo.parse(&frame);
        assert_eq!(pdo.data[0], 0); // unchanged
    }

    // ── TpdoProducer / RpdoConsumer tests ────────────────────────────────────

    fn make_od() -> StaticOd<32> {
        let mut od = StaticOd::<32>::new();
        od.insert(OdEntry::new(
            0x6041,
            0,
            DataType::Unsigned16,
            AccessType::RO,
            OdEntryValue::U16(0x0237),
        ))
        .unwrap();
        od.insert(OdEntry::new(
            0x6064,
            0,
            DataType::Integer32,
            AccessType::RO,
            OdEntryValue::I32(12345),
        ))
        .unwrap();
        od.insert(OdEntry::new(
            0x6040,
            0,
            DataType::Unsigned16,
            AccessType::RW,
            OdEntryValue::U16(0),
        ))
        .unwrap();
        od.insert(OdEntry::new(
            0x607A,
            0,
            DataType::Integer32,
            AccessType::RW,
            OdEntryValue::I32(0),
        ))
        .unwrap();
        od
    }

    #[test]
    fn tpdo_producer_basic() {
        let od = make_od();
        let config = TpdoConfig::tpdo1(1);
        let mut prod = TpdoProducer::<4>::new(config);
        prod.add_mapping(0x6041, 0, 2).unwrap(); // U16 status word
        prod.add_mapping(0x6064, 0, 4).unwrap(); // I32 position
        let frame = prod.produce(&od).unwrap();
        assert_eq!(frame.cob_id, 0x181);
        assert_eq!(frame.dlc, 6);
        let sw = u16::from_le_bytes([frame.data[0], frame.data[1]]);
        assert_eq!(sw, 0x0237);
        let pos = i32::from_le_bytes([frame.data[2], frame.data[3], frame.data[4], frame.data[5]]);
        assert_eq!(pos, 12345);
    }

    #[test]
    fn tpdo_producer_no_mappings_error() {
        let od = make_od();
        let config = TpdoConfig::tpdo1(1);
        let prod = TpdoProducer::<4>::new(config);
        assert_eq!(prod.produce(&od).unwrap_err(), PdoError::NoMappings);
    }

    #[test]
    fn tpdo_producer_payload_overflow() {
        let config = TpdoConfig::tpdo1(1);
        let mut prod = TpdoProducer::<4>::new(config);
        prod.add_mapping(0x6041, 0, 4).unwrap();
        prod.add_mapping(0x6064, 0, 4).unwrap();
        // Third mapping would overflow 8 bytes
        let result = prod.add_mapping(0x6040, 0, 2);
        assert_eq!(result, Err(PdoError::PayloadTooLarge));
    }

    #[test]
    fn rpdo_consumer_basic() {
        let mut od = make_od();
        let config = RpdoConfig::rpdo1(1);
        let mut consumer = RpdoConsumer::<4>::new(config);
        consumer.add_mapping(0x6040, 0, 2).unwrap(); // U16 control word
        consumer.add_mapping(0x607A, 0, 4).unwrap(); // I32 target position

        // Pack: cw=0x000F (2 bytes), pos=-500 (4 bytes)
        let cw: u16 = 0x000F;
        let pos: i32 = -500;
        let mut payload = [0u8; 8];
        payload[..2].copy_from_slice(&cw.to_le_bytes());
        payload[2..6].copy_from_slice(&pos.to_le_bytes());
        let frame = CanFrame::new(0x201, &payload[..6]);

        consumer.consume(&frame, &mut od).unwrap();
        assert_eq!(od.read(0x6040, 0).unwrap(), OdEntryValue::U16(0x000F));
        assert_eq!(od.read(0x607A, 0).unwrap(), OdEntryValue::I32(-500));
        assert_eq!(consumer.frames_consumed(), 1);
    }

    #[test]
    fn rpdo_consumer_cob_id_mismatch() {
        let mut od = make_od();
        let config = RpdoConfig::rpdo1(1);
        let mut consumer = RpdoConsumer::<4>::new(config);
        consumer.add_mapping(0x6040, 0, 2).unwrap();

        let frame = CanFrame::new(0x999, &[0, 0]);
        let err = consumer.consume(&frame, &mut od).unwrap_err();
        assert_eq!(err, PdoError::CobIdMismatch);
    }

    #[test]
    fn rpdo_consumer_frame_too_short() {
        let mut od = make_od();
        let config = RpdoConfig::rpdo1(1);
        let mut consumer = RpdoConsumer::<4>::new(config);
        consumer.add_mapping(0x6040, 0, 2).unwrap();
        consumer.add_mapping(0x607A, 0, 4).unwrap();

        // Provide only 2 bytes (need 6).
        let frame = CanFrame::new(0x201, &[0x0F, 0x00]);
        let err = consumer.consume(&frame, &mut od).unwrap_err();
        assert_eq!(err, PdoError::FrameTooShort);
    }

    #[test]
    fn tpdo_event_timer_fires() {
        let config = TpdoConfig {
            cob_id: 0x181,
            transmission_type: 255,
            event_timer_ms: 10,
        };
        let mut prod = TpdoProducer::<4>::new(config);
        assert!(!prod.tick(5));
        assert!(!prod.tick(4));
        assert!(prod.tick(1)); // 10ms elapsed
        assert!(!prod.tick(5)); // reset
    }

    #[test]
    fn tpdo_config_default_cob_ids() {
        let t1 = TpdoConfig::tpdo1(3);
        assert_eq!(t1.cob_id, 0x183);
        let t2 = TpdoConfig::tpdo2(3);
        assert_eq!(t2.cob_id, 0x283);
    }

    #[test]
    fn rpdo_config_default_cob_ids() {
        let r1 = RpdoConfig::rpdo1(5);
        assert_eq!(r1.cob_id, 0x205);
        let r2 = RpdoConfig::rpdo2(5);
        assert_eq!(r2.cob_id, 0x305);
    }
}
