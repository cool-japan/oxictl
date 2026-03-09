//! CANopen Network Management (NMT) state machine.
//!
//! NMT controls the lifecycle of CANopen nodes:
//!   INITIALIZING → PRE-OPERATIONAL → OPERATIONAL ↔ STOPPED
//!
//! NMT commands are broadcast by the NMT master to all nodes or a specific node.

/// CANopen NMT state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmtState {
    /// Power-on initialization.
    Initializing,
    /// Pre-operational: SDO/NMT allowed, PDO not allowed.
    PreOperational,
    /// Operational: all services allowed.
    Operational,
    /// Stopped: only NMT and heartbeat allowed.
    Stopped,
}

/// NMT command specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NmtCommand {
    StartRemoteNode = 0x01,
    StopRemoteNode = 0x02,
    EnterPreOperational = 0x80,
    ResetNode = 0x81,
    ResetCommunication = 0x82,
}

/// NMT message: command + target node ID (0 = all nodes).
#[derive(Debug, Clone, Copy)]
pub struct NmtMessage {
    pub command: NmtCommand,
    pub node_id: u8,
}

impl NmtMessage {
    pub fn broadcast(command: NmtCommand) -> Self {
        Self {
            command,
            node_id: 0,
        }
    }

    pub fn to_node(command: NmtCommand, node_id: u8) -> Self {
        Self { command, node_id }
    }
}

/// NMT state machine for a single CANopen node.
#[derive(Debug, Clone, Copy)]
pub struct NmtStateMachine {
    pub state: NmtState,
    pub node_id: u8,
}

impl NmtStateMachine {
    pub fn new(node_id: u8) -> Self {
        Self {
            state: NmtState::Initializing,
            node_id,
        }
    }

    /// Process an NMT command. Returns true if the command applies to this node.
    pub fn process(&mut self, msg: &NmtMessage) -> bool {
        if msg.node_id != 0 && msg.node_id != self.node_id {
            return false;
        }
        match msg.command {
            NmtCommand::StartRemoteNode => {
                if self.state == NmtState::PreOperational || self.state == NmtState::Stopped {
                    self.state = NmtState::Operational;
                }
            }
            NmtCommand::StopRemoteNode => {
                if self.state != NmtState::Initializing {
                    self.state = NmtState::Stopped;
                }
            }
            NmtCommand::EnterPreOperational => {
                if self.state != NmtState::Initializing {
                    self.state = NmtState::PreOperational;
                }
            }
            NmtCommand::ResetNode | NmtCommand::ResetCommunication => {
                self.state = NmtState::Initializing;
            }
        }
        true
    }

    /// Boot-up: transitions from Initializing to Pre-Operational.
    pub fn boot_up(&mut self) {
        if self.state == NmtState::Initializing {
            self.state = NmtState::PreOperational;
        }
    }
}

/// Heartbeat producer for NMT error control.
#[derive(Debug, Clone, Copy)]
pub struct HeartbeatProducer {
    pub node_id: u8,
    /// Heartbeat interval (ms).
    pub interval_ms: u16,
    /// Time since last heartbeat (ms).
    elapsed_ms: u32,
}

impl HeartbeatProducer {
    pub fn new(node_id: u8, interval_ms: u16) -> Self {
        Self {
            node_id,
            interval_ms,
            elapsed_ms: 0,
        }
    }

    /// Tick by dt_ms. Returns true if a heartbeat should be sent.
    pub fn tick(&mut self, dt_ms: u32) -> bool {
        self.elapsed_ms += dt_ms;
        if self.interval_ms > 0 && self.elapsed_ms >= self.interval_ms as u32 {
            self.elapsed_ms = 0;
            return true;
        }
        false
    }
}

// ─── Extended NMT types ──────────────────────────────────────────────────────

/// Error type for NMT operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmtError {
    /// The received command byte does not map to any known NmtCommand.
    UnknownCommand(u8),
    /// The node ID is out of valid range (must be 0–127; 0 = broadcast).
    InvalidNodeId(u8),
}

/// A heartbeat CAN frame produced by `HeartbeatProducer::tick_frame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeartbeatFrame {
    /// Node ID that is transmitting this heartbeat.
    pub node_id: u8,
    /// Current NMT state encoded as the heartbeat payload byte.
    pub state: NmtState,
}

impl HeartbeatFrame {
    /// Encode the heartbeat state byte per CiA 301 table 22.
    ///
    /// | NMT state         | Byte value |
    /// |-------------------|------------|
    /// | Initializing      | 0x00       |
    /// | Stopped           | 0x04       |
    /// | Operational       | 0x05       |
    /// | PreOperational    | 0x7F       |
    pub fn state_byte(state: NmtState) -> u8 {
        match state {
            NmtState::Initializing => 0x00,
            NmtState::Stopped => 0x04,
            NmtState::Operational => 0x05,
            NmtState::PreOperational => 0x7F,
        }
    }

    /// Decode a heartbeat payload byte into an `NmtState`.
    ///
    /// Returns `None` for reserved values not defined by CiA 301.
    pub fn decode_state_byte(byte: u8) -> Option<NmtState> {
        match byte {
            0x00 => Some(NmtState::Initializing),
            0x04 => Some(NmtState::Stopped),
            0x05 => Some(NmtState::Operational),
            0x7F => Some(NmtState::PreOperational),
            _ => None,
        }
    }

    /// Serialize this heartbeat frame to an 8-byte CAN data array.
    ///
    /// Per CiA 301 the DLC of a heartbeat frame is 1, so only `data[0]`
    /// carries the state byte; the remainder is zero-padded.
    pub fn to_can_data(&self) -> [u8; 8] {
        let mut data = [0u8; 8];
        data[0] = Self::state_byte(self.state);
        data
    }
}

/// Controller that applies raw NMT command bytes (from a received CAN frame)
/// to a local `NmtStateMachine`, producing an optional new state.
///
/// The NMT command frame layout per CiA 301 is:
///   byte 0 — command specifier
///   byte 1 — target node ID (0 = all nodes)
///
/// `NmtController` is intentionally stateless beyond the owned
/// `NmtStateMachine`; it does *not* own the heartbeat producer so that users
/// can compose them freely.
#[derive(Debug, Clone, Copy)]
pub struct NmtController {
    pub machine: NmtStateMachine,
}

impl NmtController {
    /// Create a new `NmtController` for the given `node_id` (1–127).
    pub fn new(node_id: u8) -> Self {
        Self {
            machine: NmtStateMachine::new(node_id),
        }
    }

    /// Boot the node from Initializing to PreOperational.
    pub fn boot_up(&mut self) {
        self.machine.boot_up();
    }

    /// Current NMT state.
    pub fn state(&self) -> NmtState {
        self.machine.state
    }

    /// Node ID.
    pub fn node_id(&self) -> u8 {
        self.machine.node_id
    }

    /// Parse a 2-byte NMT command frame and apply it to the internal state
    /// machine if it targets this node.
    ///
    /// Returns `Some(new_state)` if the command was applied, `None` if the
    /// command was addressed to a different node.
    ///
    /// # Errors
    ///
    /// Returns `Err(NmtError::UnknownCommand)` if `cmd_byte` is not a valid
    /// NMT command specifier.
    /// Returns `Err(NmtError::InvalidNodeId)` if `target_node` is > 127.
    pub fn process_command(
        &mut self,
        cmd_byte: u8,
        target_node: u8,
    ) -> Result<Option<NmtState>, NmtError> {
        if target_node > 127 {
            return Err(NmtError::InvalidNodeId(target_node));
        }
        let cmd = NmtCommand::from_u8(cmd_byte).ok_or(NmtError::UnknownCommand(cmd_byte))?;
        let msg = NmtMessage {
            command: cmd,
            node_id: target_node,
        };
        let applied = self.machine.process(&msg);
        if applied {
            Ok(Some(self.machine.state))
        } else {
            Ok(None)
        }
    }
}

impl NmtCommand {
    /// Try to parse a raw command-specifier byte per CiA 301 table 21.
    pub fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0x01 => Some(Self::StartRemoteNode),
            0x02 => Some(Self::StopRemoteNode),
            0x80 => Some(Self::EnterPreOperational),
            0x81 => Some(Self::ResetNode),
            0x82 => Some(Self::ResetCommunication),
            _ => None,
        }
    }

    /// Return the wire byte for this command specifier.
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

impl HeartbeatProducer {
    /// Tick by `dt_ms` milliseconds and, if a heartbeat is due, return a
    /// `HeartbeatFrame` carrying the given `state`.
    ///
    /// This is a richer variant of the plain `tick()` method; it embeds the
    /// NMT state into the returned frame so callers can emit a well-formed
    /// CAN heartbeat without additional bookkeeping.
    pub fn tick_frame(&mut self, dt_ms: u32, state: NmtState) -> Option<HeartbeatFrame> {
        if self.tick(dt_ms) {
            Some(HeartbeatFrame {
                node_id: self.node_id,
                state,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nmt_boot_up() {
        let mut nmt = NmtStateMachine::new(1);
        nmt.boot_up();
        assert_eq!(nmt.state, NmtState::PreOperational);
    }

    #[test]
    fn nmt_start_remote_node() {
        let mut nmt = NmtStateMachine::new(1);
        nmt.boot_up();
        nmt.process(&NmtMessage::broadcast(NmtCommand::StartRemoteNode));
        assert_eq!(nmt.state, NmtState::Operational);
    }

    #[test]
    fn nmt_stop_remote_node() {
        let mut nmt = NmtStateMachine::new(1);
        nmt.boot_up();
        nmt.process(&NmtMessage::to_node(NmtCommand::StartRemoteNode, 1));
        nmt.process(&NmtMessage::broadcast(NmtCommand::StopRemoteNode));
        assert_eq!(nmt.state, NmtState::Stopped);
    }

    #[test]
    fn nmt_node_id_filter() {
        let mut nmt = NmtStateMachine::new(1);
        nmt.boot_up();
        // Command for node 2 should not affect node 1
        nmt.process(&NmtMessage::to_node(NmtCommand::StartRemoteNode, 2));
        assert_eq!(nmt.state, NmtState::PreOperational); // unchanged
    }

    #[test]
    fn heartbeat_fires_at_interval() {
        let mut hb = HeartbeatProducer::new(1, 100);
        assert!(!hb.tick(50));
        assert!(!hb.tick(49));
        assert!(hb.tick(1)); // 100ms elapsed
        assert!(!hb.tick(50)); // reset
    }

    // ── NmtController tests ──────────────────────────────────────────────────

    #[test]
    fn nmt_controller_boot_up() {
        let mut ctrl = NmtController::new(5);
        ctrl.boot_up();
        assert_eq!(ctrl.state(), NmtState::PreOperational);
    }

    #[test]
    fn nmt_controller_process_broadcast_start() {
        let mut ctrl = NmtController::new(3);
        ctrl.boot_up();
        let result = ctrl.process_command(0x01, 0).unwrap(); // broadcast Start
        assert_eq!(result, Some(NmtState::Operational));
        assert_eq!(ctrl.state(), NmtState::Operational);
    }

    #[test]
    fn nmt_controller_process_targeted_different_node() {
        let mut ctrl = NmtController::new(3);
        ctrl.boot_up();
        let result = ctrl.process_command(0x01, 7).unwrap(); // target node 7, not 3
        assert_eq!(result, None);
        assert_eq!(ctrl.state(), NmtState::PreOperational); // unchanged
    }

    #[test]
    fn nmt_controller_unknown_command() {
        let mut ctrl = NmtController::new(1);
        assert_eq!(
            ctrl.process_command(0xFF, 0),
            Err(NmtError::UnknownCommand(0xFF))
        );
    }

    #[test]
    fn nmt_controller_invalid_node_id() {
        let mut ctrl = NmtController::new(1);
        assert_eq!(
            ctrl.process_command(0x01, 200),
            Err(NmtError::InvalidNodeId(200))
        );
    }

    #[test]
    fn nmt_controller_reset_node() {
        let mut ctrl = NmtController::new(2);
        ctrl.boot_up();
        ctrl.process_command(0x01, 0).unwrap(); // → Operational
        ctrl.process_command(0x81, 0).unwrap(); // Reset → Initializing
        assert_eq!(ctrl.state(), NmtState::Initializing);
    }

    #[test]
    fn nmt_command_round_trip() {
        let cmds = [
            (0x01u8, NmtCommand::StartRemoteNode),
            (0x02, NmtCommand::StopRemoteNode),
            (0x80, NmtCommand::EnterPreOperational),
            (0x81, NmtCommand::ResetNode),
            (0x82, NmtCommand::ResetCommunication),
        ];
        for (byte, cmd) in cmds {
            assert_eq!(NmtCommand::from_u8(byte), Some(cmd));
            assert_eq!(cmd.to_u8(), byte);
        }
    }

    #[test]
    fn heartbeat_frame_state_encoding() {
        assert_eq!(HeartbeatFrame::state_byte(NmtState::Initializing), 0x00);
        assert_eq!(HeartbeatFrame::state_byte(NmtState::Stopped), 0x04);
        assert_eq!(HeartbeatFrame::state_byte(NmtState::Operational), 0x05);
        assert_eq!(HeartbeatFrame::state_byte(NmtState::PreOperational), 0x7F);
    }

    #[test]
    fn heartbeat_frame_decode_round_trip() {
        for state in [
            NmtState::Initializing,
            NmtState::Stopped,
            NmtState::Operational,
            NmtState::PreOperational,
        ] {
            let byte = HeartbeatFrame::state_byte(state);
            assert_eq!(HeartbeatFrame::decode_state_byte(byte), Some(state));
        }
        assert_eq!(HeartbeatFrame::decode_state_byte(0x99), None);
    }

    #[test]
    fn heartbeat_tick_frame_emits_correctly() {
        let mut hb = HeartbeatProducer::new(7, 50);
        assert!(hb.tick_frame(30, NmtState::Operational).is_none());
        let frame = hb
            .tick_frame(20, NmtState::Operational)
            .expect("should emit after 50ms");
        assert_eq!(frame.node_id, 7);
        assert_eq!(frame.state, NmtState::Operational);
        // Immediately after reset, nothing emitted yet
        assert!(hb.tick_frame(10, NmtState::Operational).is_none());
    }

    #[test]
    fn heartbeat_frame_to_can_data() {
        let frame = HeartbeatFrame {
            node_id: 1,
            state: NmtState::Operational,
        };
        let data = frame.to_can_data();
        assert_eq!(data[0], 0x05);
        assert_eq!(&data[1..], &[0u8; 7]);
    }
}
