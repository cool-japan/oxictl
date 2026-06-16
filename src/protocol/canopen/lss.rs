//! CANopen Layer Setting Services (LSS) - CiA 305.
//!
//! LSS provides a mechanism to configure node-IDs and baudrates
//! for unconfigured nodes using vendor ID and serial number addressing.

/// LSS client state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LssClientCoState {
    /// Waiting for a command to begin.
    Waiting,
    /// Identification process started.
    Identifying,
    /// Slave has been identified.
    Identified,
    /// Node ID has been configured.
    Configured,
    /// Configuration has been stored to NVM.
    Stored,
}

/// LSS command service codes.
mod cs {
    pub const SWITCH_STATE_GLOBAL: u8 = 0x04;
    pub const IDENTIFY_SLAVE_RESP: u8 = 0x4F;
    pub const CONFIGURE_NODE_ID: u8 = 0x11;
    pub const CONFIGURE_BAUDRATE: u8 = 0x13;
    pub const STORE_CONFIGURATION: u8 = 0x17;
    pub const SWITCH_STATE_SELECTIVE_VENDOR: u8 = 0x40;
    pub const SWITCH_STATE_SELECTIVE_PRODUCT: u8 = 0x41;
    pub const SWITCH_STATE_SELECTIVE_REVISION: u8 = 0x42;
    pub const SWITCH_STATE_SELECTIVE_SERIAL: u8 = 0x43;
}

/// LSS error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LssCoError {
    /// Not in the required state for this operation.
    InvalidState,
    /// Node ID is out of range (must be 1–127).
    InvalidNodeId,
    /// Baudrate is not in the allowed set.
    InvalidBaudrate,
    /// Identity does not match the target.
    IdentityMismatch,
    /// CAN frame buffer is too small.
    BufferTooSmall,
}

/// LSS Client (master-side) for CANopen CiA 305.
///
/// Manages node address assignment via vendor ID / serial number.
#[derive(Debug, Clone)]
pub struct LssClientCo {
    state: LssClientCoState,
    /// Target slave vendor ID.
    vendor_id: u32,
    /// Target slave product code.
    product_code: u32,
    /// Target slave revision number.
    revision: u32,
    /// Target slave serial number.
    serial_number: u32,
    /// Assigned node ID.
    node_id: u8,
    /// Configured baudrate.
    baudrate: u32,
    /// Number of identify frames exchanged.
    identify_count: u8,
}

impl LssClientCo {
    /// Create a new LSS client targeting a specific device identity.
    pub fn new(vendor_id: u32, product_code: u32, revision: u32, serial_number: u32) -> Self {
        Self {
            state: LssClientCoState::Waiting,
            vendor_id,
            product_code,
            revision,
            serial_number,
            node_id: 0xFF,
            baudrate: 250_000,
            identify_count: 0,
        }
    }

    /// Current state.
    pub fn state(&self) -> LssClientCoState {
        self.state
    }

    /// Assigned node ID (0xFF = not yet assigned).
    pub fn node_id(&self) -> u8 {
        self.node_id
    }

    /// Configured baudrate.
    pub fn baudrate(&self) -> u32 {
        self.baudrate
    }

    /// Begin the selective identify process.
    /// Builds 4 CAN frames into `out` (need 4 * 8 bytes = 32 bytes min).
    pub fn begin_identify(&mut self, out: &mut [[u8; 8]; 4]) {
        self.state = LssClientCoState::Identifying;
        self.identify_count = 0;

        // Frame 0: vendor ID (cs=0x40)
        out[0] = Self::build_lss_frame(cs::SWITCH_STATE_SELECTIVE_VENDOR, self.vendor_id);
        // Frame 1: product code (cs=0x41)
        out[1] = Self::build_lss_frame(cs::SWITCH_STATE_SELECTIVE_PRODUCT, self.product_code);
        // Frame 2: revision (cs=0x42)
        out[2] = Self::build_lss_frame(cs::SWITCH_STATE_SELECTIVE_REVISION, self.revision);
        // Frame 3: serial (cs=0x43)
        out[3] = Self::build_lss_frame(cs::SWITCH_STATE_SELECTIVE_SERIAL, self.serial_number);
    }

    fn build_lss_frame(command: u8, value: u32) -> [u8; 8] {
        let vb = value.to_le_bytes();
        [command, vb[0], vb[1], vb[2], vb[3], 0, 0, 0]
    }

    /// Process an identify response frame (cs=0x4F).
    /// Returns true if this is a valid identify response.
    pub fn on_identify_response(&mut self, frame: &[u8; 8]) -> bool {
        if self.state != LssClientCoState::Identifying {
            return false;
        }
        if frame[0] != cs::IDENTIFY_SLAVE_RESP {
            return false;
        }
        self.identify_count += 1;
        self.state = LssClientCoState::Identified;
        true
    }

    /// Assign a node ID to the identified slave.
    /// Returns the CAN frame to send, or an error.
    pub fn assign_node_id(&mut self, node_id: u8) -> Result<[u8; 8], LssCoError> {
        if self.state != LssClientCoState::Identified {
            return Err(LssCoError::InvalidState);
        }
        if node_id == 0 || node_id > 127 {
            return Err(LssCoError::InvalidNodeId);
        }
        self.node_id = node_id;
        self.state = LssClientCoState::Configured;
        Ok([cs::CONFIGURE_NODE_ID, node_id, 0, 0, 0, 0, 0, 0])
    }

    /// Configure baudrate for the identified slave.
    /// Returns the CAN frame to send, or an error.
    pub fn set_baudrate(&mut self, baudrate: u32) -> Result<[u8; 8], LssCoError> {
        if self.state != LssClientCoState::Identified && self.state != LssClientCoState::Configured
        {
            return Err(LssCoError::InvalidState);
        }
        let table_idx = Self::baudrate_table_index(baudrate)?;
        self.baudrate = baudrate;
        Ok([cs::CONFIGURE_BAUDRATE, 0, table_idx, 0, 0, 0, 0, 0])
    }

    fn baudrate_table_index(baud: u32) -> Result<u8, LssCoError> {
        match baud {
            1_000_000 => Ok(0),
            800_000 => Ok(1),
            500_000 => Ok(2),
            250_000 => Ok(3),
            125_000 => Ok(4),
            50_000 => Ok(6),
            20_000 => Ok(7),
            10_000 => Ok(8),
            _ => Err(LssCoError::InvalidBaudrate),
        }
    }

    /// Store configuration to NVM.
    /// Returns the CAN frame to send, or an error.
    pub fn store_config(&mut self) -> Result<[u8; 8], LssCoError> {
        if self.state != LssClientCoState::Configured {
            return Err(LssCoError::InvalidState);
        }
        self.state = LssClientCoState::Stored;
        Ok([cs::STORE_CONFIGURATION, 0x65, 0x76, 0x61, 0x73, 0, 0, 0])
    }

    /// Reset to waiting state.
    pub fn reset(&mut self) {
        self.state = LssClientCoState::Waiting;
        self.node_id = 0xFF;
        self.identify_count = 0;
    }

    /// Build a global switch-state frame (switch all nodes to waiting/configuration mode).
    /// mode: 0=waiting, 1=configuration
    pub fn build_switch_global(mode: u8) -> [u8; 8] {
        [cs::SWITCH_STATE_GLOBAL, mode, 0, 0, 0, 0, 0, 0]
    }

    /// Number of identify responses received.
    pub fn identify_count(&self) -> u8 {
        self.identify_count
    }

    /// Target vendor ID.
    pub fn target_vendor_id(&self) -> u32 {
        self.vendor_id
    }

    /// Target serial number.
    pub fn target_serial(&self) -> u32 {
        self.serial_number
    }
}

impl Default for LssClientCo {
    fn default() -> Self {
        Self::new(0, 0, 0, 0)
    }
}

/// LSS Slave (server-side) state for node-ID configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LssSlaveState {
    Waiting,
    Configuration,
}

/// LSS Slave — responds to LSS master commands.
#[derive(Debug, Clone)]
pub struct LssSlave {
    state: LssSlaveState,
    vendor_id: u32,
    product_code: u32,
    revision: u32,
    serial_number: u32,
    /// Currently configured node ID.
    node_id: u8,
    /// Baudrate in bps (updated on configure baudrate command).
    baudrate: u32,
    /// Identify match count (selective identify: 4 consecutive matches).
    match_count: u8,
}

impl LssSlave {
    /// Create a new LSS slave.
    pub fn new(
        vendor_id: u32,
        product_code: u32,
        revision: u32,
        serial_number: u32,
        node_id: u8,
    ) -> Self {
        Self {
            state: LssSlaveState::Waiting,
            vendor_id,
            product_code,
            revision,
            serial_number,
            node_id,
            baudrate: 250_000,
            match_count: 0,
        }
    }

    /// Current slave state.
    pub fn state(&self) -> LssSlaveState {
        self.state
    }

    /// Current node ID.
    pub fn node_id(&self) -> u8 {
        self.node_id
    }

    /// Current baudrate in bps.
    pub fn baudrate(&self) -> u32 {
        self.baudrate
    }

    /// Process an incoming LSS frame.
    /// Returns Some(response_frame) if a response should be sent.
    pub fn process_frame(&mut self, frame: &[u8; 8]) -> Option<[u8; 8]> {
        match frame[0] {
            cs::SWITCH_STATE_GLOBAL => {
                self.state = if frame[1] == 1 {
                    self.match_count = 0;
                    LssSlaveState::Configuration
                } else {
                    LssSlaveState::Waiting
                };
                None
            }
            cs::SWITCH_STATE_SELECTIVE_VENDOR => {
                let v = u32::from_le_bytes([frame[1], frame[2], frame[3], frame[4]]);
                self.match_count = if v == self.vendor_id { 1 } else { 0 };
                None
            }
            cs::SWITCH_STATE_SELECTIVE_PRODUCT => {
                let v = u32::from_le_bytes([frame[1], frame[2], frame[3], frame[4]]);
                if self.match_count == 1 && v == self.product_code {
                    self.match_count = 2;
                } else {
                    self.match_count = 0;
                }
                None
            }
            cs::SWITCH_STATE_SELECTIVE_REVISION => {
                let v = u32::from_le_bytes([frame[1], frame[2], frame[3], frame[4]]);
                if self.match_count == 2 && v == self.revision {
                    self.match_count = 3;
                } else {
                    self.match_count = 0;
                }
                None
            }
            cs::SWITCH_STATE_SELECTIVE_SERIAL => {
                let v = u32::from_le_bytes([frame[1], frame[2], frame[3], frame[4]]);
                if self.match_count == 3 && v == self.serial_number {
                    self.match_count = 4;
                    self.state = LssSlaveState::Configuration;
                    // Send identify response
                    let mut resp = [0u8; 8];
                    resp[0] = cs::IDENTIFY_SLAVE_RESP;
                    return Some(resp);
                }
                self.match_count = 0;
                None
            }
            cs::CONFIGURE_NODE_ID if self.state == LssSlaveState::Configuration => {
                let nid = frame[1];
                if nid > 0 && nid <= 127 {
                    self.node_id = nid;
                    Some([0x4F, 0x00, 0, 0, 0, 0, 0, 0])
                } else {
                    Some([0x4F, 0x01, 0, 0, 0, 0, 0, 0]) // error
                }
            }
            _ => None,
        }
    }
}

impl LssClientCo {
    // Test helper exposed for slave tests
    #[cfg(test)]
    pub fn build_lss_frame_pub(command: u8, value: u32) -> [u8; 8] {
        Self::build_lss_frame(command, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identify_workflow() {
        let mut client = LssClientCo::new(0xABCD, 0x1234, 0x0001, 0xDEAD);
        let mut frames = [[0u8; 8]; 4];
        client.begin_identify(&mut frames);
        assert_eq!(client.state(), LssClientCoState::Identifying);
        // Frame 0 should be vendor ID selective command
        assert_eq!(frames[0][0], cs::SWITCH_STATE_SELECTIVE_VENDOR);
        // Simulate response
        let resp = [cs::IDENTIFY_SLAVE_RESP, 0, 0, 0, 0, 0, 0, 0];
        assert!(client.on_identify_response(&resp));
        assert_eq!(client.state(), LssClientCoState::Identified);
    }

    #[test]
    fn test_assign_node_id() {
        let mut client = LssClientCo::new(0x1, 0x2, 0x3, 0x4);
        // Force to identified state
        let mut frames = [[0u8; 8]; 4];
        client.begin_identify(&mut frames);
        let resp = [cs::IDENTIFY_SLAVE_RESP, 0, 0, 0, 0, 0, 0, 0];
        client.on_identify_response(&resp);

        let frame = client.assign_node_id(42).unwrap();
        assert_eq!(frame[0], cs::CONFIGURE_NODE_ID);
        assert_eq!(frame[1], 42);
        assert_eq!(client.node_id(), 42);
        assert_eq!(client.state(), LssClientCoState::Configured);
    }

    #[test]
    fn test_invalid_node_id() {
        let mut client = LssClientCo::new(0x1, 0x2, 0x3, 0x4);
        let mut frames = [[0u8; 8]; 4];
        client.begin_identify(&mut frames);
        let resp = [cs::IDENTIFY_SLAVE_RESP, 0, 0, 0, 0, 0, 0, 0];
        client.on_identify_response(&resp);
        assert_eq!(client.assign_node_id(0), Err(LssCoError::InvalidNodeId));
        assert_eq!(client.assign_node_id(128), Err(LssCoError::InvalidNodeId));
    }

    #[test]
    fn test_store_config() {
        let mut client = LssClientCo::new(0x1, 0x2, 0x3, 0x4);
        let mut frames = [[0u8; 8]; 4];
        client.begin_identify(&mut frames);
        let resp = [cs::IDENTIFY_SLAVE_RESP, 0, 0, 0, 0, 0, 0, 0];
        client.on_identify_response(&resp);
        client.assign_node_id(5).unwrap();
        let store_frame = client.store_config().unwrap();
        assert_eq!(store_frame[0], cs::STORE_CONFIGURATION);
        assert_eq!(client.state(), LssClientCoState::Stored);
    }

    #[test]
    fn test_baudrate_invalid() {
        let mut client = LssClientCo::new(0x1, 0x2, 0x3, 0x4);
        let mut frames = [[0u8; 8]; 4];
        client.begin_identify(&mut frames);
        let resp = [cs::IDENTIFY_SLAVE_RESP, 0, 0, 0, 0, 0, 0, 0];
        client.on_identify_response(&resp);
        assert_eq!(client.set_baudrate(9600), Err(LssCoError::InvalidBaudrate));
        assert!(client.set_baudrate(500_000).is_ok());
    }

    #[test]
    fn test_lss_slave_selective_identify() {
        let mut slave = LssSlave::new(0xABCD, 0x1234, 0x0001, 0xDEAD, 0xFF);
        let f0 = LssClientCo::build_lss_frame_pub(cs::SWITCH_STATE_SELECTIVE_VENDOR, 0xABCD);
        let f1 = LssClientCo::build_lss_frame_pub(cs::SWITCH_STATE_SELECTIVE_PRODUCT, 0x1234);
        let f2 = LssClientCo::build_lss_frame_pub(cs::SWITCH_STATE_SELECTIVE_REVISION, 0x0001);
        let f3 = LssClientCo::build_lss_frame_pub(cs::SWITCH_STATE_SELECTIVE_SERIAL, 0xDEAD);
        slave.process_frame(&f0);
        slave.process_frame(&f1);
        slave.process_frame(&f2);
        let resp = slave.process_frame(&f3);
        assert!(resp.is_some());
        assert_eq!(resp.unwrap()[0], cs::IDENTIFY_SLAVE_RESP);
        assert_eq!(slave.state(), LssSlaveState::Configuration);
    }
}
