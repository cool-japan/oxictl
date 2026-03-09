//! CANopen Emergency protocol (CiA 301).
//!
//! The Emergency (EMCY) service is triggered by internal device errors.
//! The EMCY message contains an error code, error register, and 5 bytes
//! of device-specific data. A history ring buffer stores recent events.

use heapless::Deque;

/// Default Emergency COB-ID = 0x80 + node_id.
pub const EMCY_COB_ID_BASE: u16 = 0x80;

// -------------------------------------------------------------------------
// DS-301 Error Register bit definitions (0x1001:00)
// -------------------------------------------------------------------------

/// Generic error (bit 0).
pub const ERR_REG_GENERIC: u8 = 0x01;
/// Current error (bit 1).
pub const ERR_REG_CURRENT: u8 = 0x02;
/// Voltage error (bit 2).
pub const ERR_REG_VOLTAGE: u8 = 0x04;
/// Temperature error (bit 3).
pub const ERR_REG_TEMPERATURE: u8 = 0x08;
/// Communication error (bit 4).
pub const ERR_REG_COMMUNICATION: u8 = 0x10;
/// Device profile specific error (bit 5).
pub const ERR_REG_PROFILE: u8 = 0x20;
/// Manufacturer-specific error (bit 7).
pub const ERR_REG_MANUFACTURER: u8 = 0x80;

// -------------------------------------------------------------------------
// Pre-defined emergency error codes (CiA 301 Table 25)
// -------------------------------------------------------------------------

/// No error (error reset).
pub const EMCY_NO_ERROR: u16 = 0x0000;
/// Generic error.
pub const EMCY_GENERIC: u16 = 0x1000;
/// Current - device input side.
pub const EMCY_CURRENT_INPUT: u16 = 0x2100;
/// Voltage - mains voltage.
pub const EMCY_VOLTAGE_MAINS: u16 = 0x3100;
/// Temperature - ambient.
pub const EMCY_TEMP_AMBIENT: u16 = 0x4100;
/// Communication - CAN overrun.
pub const EMCY_COM_CAN_OVERRUN: u16 = 0x8110;
/// Communication - bus off.
pub const EMCY_COM_BUS_OFF: u16 = 0x8140;
/// Protocol error - PDO length.
pub const EMCY_PROTO_PDO_LEN: u16 = 0x8200;
/// External error.
pub const EMCY_EXTERNAL: u16 = 0x9000;
/// Manufacturer-specific.
pub const EMCY_MANUFACTURER: u16 = 0xFF00;

/// A single emergency event record.
#[derive(Debug, Clone, Copy)]
pub struct EmergencyEvent {
    /// EMCY error code (16-bit).
    pub error_code: u16,
    /// Error register value.
    pub error_reg: u8,
    /// Device-specific data (5 bytes).
    pub data: [u8; 5],
    /// Sequence number (monotonically increasing).
    pub sequence: u32,
}

impl EmergencyEvent {
    /// Create a new emergency event.
    pub fn new(error_code: u16, error_reg: u8, data: [u8; 5], sequence: u32) -> Self {
        Self {
            error_code,
            error_reg,
            data,
            sequence,
        }
    }

    /// Serialize to 8-byte CAN frame data.
    pub fn to_bytes(&self) -> [u8; 8] {
        let ec = self.error_code.to_le_bytes();
        [
            ec[0],
            ec[1],
            self.error_reg,
            self.data[0],
            self.data[1],
            self.data[2],
            self.data[3],
            self.data[4],
        ]
    }

    /// Parse from 8-byte CAN frame data.
    pub fn from_bytes(b: &[u8; 8], sequence: u32) -> Self {
        Self {
            error_code: u16::from_le_bytes([b[0], b[1]]),
            error_reg: b[2],
            data: [b[3], b[4], b[5], b[6], b[7]],
            sequence,
        }
    }

    /// Is this an error reset event (error_code == 0)?
    pub fn is_error_reset(&self) -> bool {
        self.error_code == EMCY_NO_ERROR
    }
}

/// Emergency producer: emits EMCY CAN frames.
///
/// Maintains an 8-entry ring buffer of recent events.
pub struct EmergencyProducer {
    /// Node-specific COB-ID.
    cob_id: u16,
    /// Current error register state.
    error_reg: u8,
    /// History of recent emergency events.
    history: Deque<EmergencyEvent, 8>,
    /// Total emergencies emitted.
    total_emitted: u32,
    /// Sequence counter for events.
    sequence: u32,
    /// Whether any active error exists.
    active_error: bool,
}

impl EmergencyProducer {
    /// Create a new emergency producer for `node_id`.
    pub fn new(node_id: u8) -> Self {
        Self {
            cob_id: EMCY_COB_ID_BASE + node_id as u16,
            error_reg: 0,
            history: Deque::new(),
            total_emitted: 0,
            sequence: 0,
            active_error: false,
        }
    }

    /// COB-ID for this producer.
    pub fn cob_id(&self) -> u16 {
        self.cob_id
    }

    /// Current error register.
    pub fn error_reg(&self) -> u8 {
        self.error_reg
    }

    /// Whether any active error is present.
    pub fn has_active_error(&self) -> bool {
        self.active_error
    }

    /// Total emergencies emitted.
    pub fn total_emitted(&self) -> u32 {
        self.total_emitted
    }

    /// Number of events in history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Emit an emergency event.
    ///
    /// `code` is the 16-bit error code, `error_reg` is the OR of ERR_REG_* bits,
    /// and `data` is 5 bytes of device-specific information.
    ///
    /// Returns the 8-byte CAN frame to transmit on the EMCY COB-ID.
    pub fn emit_error(&mut self, code: u16, error_reg: u8, data: [u8; 5]) -> [u8; 8] {
        self.error_reg |= error_reg;
        self.active_error = code != EMCY_NO_ERROR;
        self.sequence += 1;
        let event = EmergencyEvent::new(code, self.error_reg, data, self.sequence);
        let frame = event.to_bytes();

        // Add to history ring buffer (drop oldest if full)
        if self.history.is_full() {
            self.history.pop_front();
        }
        let _ = self.history.push_back(event);
        self.total_emitted += 1;
        frame
    }

    /// Emit an error reset (code=0x0000).
    pub fn reset_error(&mut self, error_reg_bits: u8) -> [u8; 8] {
        self.error_reg &= !error_reg_bits;
        if self.error_reg == 0 {
            self.active_error = false;
        }
        self.emit_error(EMCY_NO_ERROR, 0, [0u8; 5])
    }

    /// Get the most recent event from history.
    pub fn last_event(&self) -> Option<&EmergencyEvent> {
        self.history.back()
    }

    /// Get an event by index from oldest (0) to newest.
    pub fn event_at(&self, idx: usize) -> Option<&EmergencyEvent> {
        self.history.iter().nth(idx)
    }

    /// Clear the history ring buffer.
    pub fn clear_history(&mut self) {
        while self.history.pop_front().is_some() {}
    }
}

/// Emergency consumer: receives and processes EMCY frames from other nodes.
#[derive(Debug, Clone)]
pub struct EmergencyConsumer {
    /// COB-ID to listen for.
    cob_id: u16,
    /// Last received event.
    last_event: Option<EmergencyEvent>,
    /// Total received.
    total_received: u32,
    /// Sequence counter.
    sequence: u32,
}

impl EmergencyConsumer {
    /// Create a consumer listening on the given COB-ID.
    pub fn new(cob_id: u16) -> Self {
        Self {
            cob_id,
            last_event: None,
            total_received: 0,
            sequence: 0,
        }
    }

    /// Process a received EMCY frame.
    pub fn on_emcy(&mut self, frame: &[u8; 8]) {
        self.sequence += 1;
        self.total_received += 1;
        self.last_event = Some(EmergencyEvent::from_bytes(frame, self.sequence));
    }

    /// Last received event.
    pub fn last_event(&self) -> Option<&EmergencyEvent> {
        self.last_event.as_ref()
    }

    /// Total received.
    pub fn total_received(&self) -> u32 {
        self.total_received
    }

    /// COB-ID being listened to.
    pub fn cob_id(&self) -> u16 {
        self.cob_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_error() {
        let mut prod = EmergencyProducer::new(1);
        let frame = prod.emit_error(
            EMCY_COM_BUS_OFF,
            ERR_REG_COMMUNICATION,
            [0xDE, 0xAD, 0, 0, 0],
        );
        // frame[0..2] = error_code LE
        let code = u16::from_le_bytes([frame[0], frame[1]]);
        assert_eq!(code, EMCY_COM_BUS_OFF);
        assert_eq!(frame[2], ERR_REG_COMMUNICATION);
        assert_eq!(frame[3], 0xDE);
        assert!(prod.has_active_error());
        assert_eq!(prod.history_len(), 1);
    }

    #[test]
    fn test_history_ring_buffer_overflow() {
        let mut prod = EmergencyProducer::new(2);
        for i in 0..10u8 {
            prod.emit_error(EMCY_GENERIC, ERR_REG_GENERIC, [i, 0, 0, 0, 0]);
        }
        // History is capped at 8
        assert_eq!(prod.history_len(), 8);
        // Last event should have data[0]=9
        assert_eq!(prod.last_event().unwrap().data[0], 9);
    }

    #[test]
    fn test_reset_error() {
        let mut prod = EmergencyProducer::new(1);
        prod.emit_error(EMCY_GENERIC, ERR_REG_GENERIC | ERR_REG_VOLTAGE, [0; 5]);
        assert!(prod.has_active_error());
        prod.reset_error(ERR_REG_GENERIC | ERR_REG_VOLTAGE);
        assert!(!prod.has_active_error());
        assert_eq!(prod.error_reg(), 0);
    }

    #[test]
    fn test_emergency_consumer() {
        let mut consumer = EmergencyConsumer::new(0x81);
        let frame = [0x00u8, 0x10, 0x01, 0xAA, 0, 0, 0, 0]; // EMCY_GENERIC, ERR_REG_GENERIC
        consumer.on_emcy(&frame);
        assert_eq!(consumer.total_received(), 1);
        let ev = consumer.last_event().unwrap();
        assert_eq!(ev.error_code, EMCY_GENERIC);
        assert_eq!(ev.error_reg, ERR_REG_GENERIC);
        assert_eq!(ev.data[0], 0xAA);
    }

    #[test]
    fn test_frame_roundtrip() {
        let mut prod = EmergencyProducer::new(5);
        let frame = prod.emit_error(EMCY_TEMP_AMBIENT, ERR_REG_TEMPERATURE, [1, 2, 3, 4, 5]);
        let mut consumer = EmergencyConsumer::new(0x85);
        consumer.on_emcy(&frame);
        let ev = consumer.last_event().unwrap();
        assert_eq!(ev.error_code, EMCY_TEMP_AMBIENT);
        assert_eq!(ev.error_reg, ERR_REG_TEMPERATURE);
        assert_eq!(ev.data, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_error_register_accumulation() {
        let mut prod = EmergencyProducer::new(1);
        prod.emit_error(EMCY_GENERIC, ERR_REG_GENERIC, [0; 5]);
        prod.emit_error(EMCY_VOLTAGE_MAINS, ERR_REG_VOLTAGE, [0; 5]);
        // Both bits should be set
        assert_eq!(prod.error_reg(), ERR_REG_GENERIC | ERR_REG_VOLTAGE);
    }

    #[test]
    fn test_event_serialization() {
        let ev = EmergencyEvent::new(EMCY_COM_BUS_OFF, ERR_REG_COMMUNICATION, [1, 2, 3, 4, 5], 1);
        let bytes = ev.to_bytes();
        let parsed = EmergencyEvent::from_bytes(&bytes, 1);
        assert_eq!(parsed.error_code, EMCY_COM_BUS_OFF);
        assert_eq!(parsed.error_reg, ERR_REG_COMMUNICATION);
        assert_eq!(parsed.data, [1, 2, 3, 4, 5]);
    }
}
