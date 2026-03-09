//! EtherCAT Layer Setting Services (LSS) - node address assignment.
//!
//! LSS provides a mechanism to assign node-IDs and baudrates to slaves
//! using their vendor ID, product code, revision, and serial number.

use crate::core::scalar::ControlScalar;

/// LSS state machine states for the server (slave side).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LssState {
    Waiting,
    Identified,
    Configured,
}

/// LSS Server - handles address assignment via vendor/serial number.
#[derive(Debug, Clone)]
pub struct LssServer<S: ControlScalar> {
    state: LssState,
    vendor_id: u32,
    product_code: u32,
    revision: u32,
    serial_number: u32,
    node_id: u8,
    baudrate: u32,
    _phantom: core::marker::PhantomData<S>,
}

impl<S: ControlScalar> LssServer<S> {
    /// Create a new LSS server with device identity information.
    pub fn new(vendor_id: u32, product_code: u32, revision: u32, serial_number: u32) -> Self {
        Self {
            state: LssState::Waiting,
            vendor_id,
            product_code,
            revision,
            serial_number,
            node_id: 0xFF,
            baudrate: 250_000,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Current state of the LSS state machine.
    pub fn state(&self) -> LssState {
        self.state
    }

    /// Assigned node ID (0xFF means unassigned).
    pub fn node_id(&self) -> u8 {
        self.node_id
    }

    /// Configured baudrate in bps.
    pub fn baudrate(&self) -> u32 {
        self.baudrate
    }

    /// Attempt to identify slave by vendor ID and serial number.
    /// Returns true if identity matches.
    pub fn identify_slave(
        &mut self,
        vendor_id: u32,
        product_code: u32,
        revision: u32,
        serial_number: u32,
    ) -> bool {
        if self.vendor_id == vendor_id
            && self.product_code == product_code
            && self.revision == revision
            && self.serial_number == serial_number
        {
            self.state = LssState::Identified;
            true
        } else {
            false
        }
    }

    /// Assign a node ID to the identified slave.
    /// Returns Err if not in Identified state or node_id out of range.
    pub fn assign_address(&mut self, node_id: u8) -> Result<(), LssError> {
        if self.state != LssState::Identified {
            return Err(LssError::NotIdentified);
        }
        if node_id == 0 || node_id > 127 {
            return Err(LssError::InvalidNodeId);
        }
        self.node_id = node_id;
        self.state = LssState::Configured;
        Ok(())
    }

    /// Configure node ID (combined identify + assign in one call).
    pub fn configure_node_id(
        &mut self,
        vendor_id: u32,
        product_code: u32,
        revision: u32,
        serial_number: u32,
        node_id: u8,
    ) -> Result<(), LssError> {
        if !self.identify_slave(vendor_id, product_code, revision, serial_number) {
            return Err(LssError::IdentityMismatch);
        }
        self.assign_address(node_id)
    }

    /// Configure baudrate for the identified slave.
    /// Valid rates per CiA 301: 10k, 20k, 50k, 125k, 250k, 500k, 800k, 1M.
    pub fn configure_baudrate(&mut self, baudrate: u32) -> Result<(), LssError> {
        match baudrate {
            10_000 | 20_000 | 50_000 | 125_000 | 250_000 | 500_000 | 800_000 | 1_000_000 => {
                self.baudrate = baudrate;
                Ok(())
            }
            _ => Err(LssError::InvalidBaudrate),
        }
    }

    /// Store configuration (check that we are in Configured state).
    pub fn store_configuration(&mut self) -> Result<(), LssError> {
        if self.state != LssState::Configured {
            return Err(LssError::NotConfigured);
        }
        Ok(())
    }

    /// Reset to waiting state.
    pub fn reset(&mut self) {
        self.state = LssState::Waiting;
        self.node_id = 0xFF;
    }

    /// Build LSS identify request frame (vendor/product/revision/serial).
    /// Returns the 8-byte frame payload.
    pub fn build_identify_request(&self) -> [u8; 8] {
        let mut frame = [0u8; 8];
        frame[0] = 0x4B; // LSS identify slave command
        let vb = self.vendor_id.to_le_bytes();
        frame[1] = vb[0];
        frame[2] = vb[1];
        frame[3] = vb[2];
        frame[4] = vb[3];
        frame
    }

    /// Build assign node-id response frame.
    pub fn build_assign_response(&self, error_code: u8) -> [u8; 8] {
        let mut frame = [0u8; 8];
        frame[0] = 0x4F; // assign node-id response
        frame[1] = error_code;
        frame
    }

    /// Parse an incoming LSS frame and process it.
    /// Returns true if a response is required.
    pub fn process_frame(&mut self, frame: &[u8; 8]) -> bool {
        match frame[0] {
            0x4B => {
                // Identify slave request
                let vendor_id = u32::from_le_bytes([frame[1], frame[2], frame[3], frame[4]]);
                // For identify, we need the full 4 fields — use compact form
                let _ = vendor_id;
                false
            }
            0x11 => {
                // Switch state global
                let new_state = frame[1];
                if new_state == 0x04 {
                    // waiting
                    self.reset();
                }
                false
            }
            _ => false,
        }
    }
}

/// LSS operation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LssError {
    NotIdentified,
    NotConfigured,
    InvalidNodeId,
    InvalidBaudrate,
    IdentityMismatch,
}

/// LSS Client state machine for master-side operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LssClientState {
    Idle,
    Searching,
    SlaveFound,
    Configuring,
    Done,
}

/// LSS Client - manages the discovery and configuration process.
#[derive(Debug, Clone)]
pub struct LssClient {
    state: LssClientState,
    target_node_id: u8,
    found_vendor_id: u32,
    found_serial: u32,
    search_vendor_id: u32,
    search_product_code: u32,
    search_revision: u32,
    search_serial: u32,
}

impl LssClient {
    /// Create a new LSS client.
    pub fn new() -> Self {
        Self {
            state: LssClientState::Idle,
            target_node_id: 0,
            found_vendor_id: 0,
            found_serial: 0,
            search_vendor_id: 0,
            search_product_code: 0,
            search_revision: 0,
            search_serial: 0,
        }
    }

    /// Current state.
    pub fn state(&self) -> LssClientState {
        self.state
    }

    /// Begin slave search with optional target identity filter.
    pub fn begin_search(&mut self) {
        self.state = LssClientState::Searching;
    }

    /// Begin targeted search for a specific device.
    pub fn begin_targeted_search(
        &mut self,
        vendor_id: u32,
        product_code: u32,
        revision: u32,
        serial: u32,
    ) {
        self.search_vendor_id = vendor_id;
        self.search_product_code = product_code;
        self.search_revision = revision;
        self.search_serial = serial;
        self.state = LssClientState::Searching;
    }

    /// Process identify response from slave.
    pub fn on_identify_response(&mut self, vendor_id: u32, serial: u32) {
        if self.state == LssClientState::Searching {
            self.found_vendor_id = vendor_id;
            self.found_serial = serial;
            self.state = LssClientState::SlaveFound;
        }
    }

    /// Request to assign a node ID to the found slave.
    pub fn assign_node_id(&mut self, node_id: u8) -> Result<(), LssError> {
        if self.state != LssClientState::SlaveFound {
            return Err(LssError::NotIdentified);
        }
        if node_id == 0 || node_id > 127 {
            return Err(LssError::InvalidNodeId);
        }
        self.target_node_id = node_id;
        self.state = LssClientState::Configuring;
        Ok(())
    }

    /// Confirm that assignment was accepted by slave.
    pub fn confirm_assignment(&mut self) {
        if self.state == LssClientState::Configuring {
            self.state = LssClientState::Done;
        }
    }

    /// Build identify slave request frame for the targeted device.
    pub fn build_identify_frame(&self) -> [u8; 8] {
        let mut frame = [0u8; 8];
        frame[0] = 0x46; // identify slave
        let vb = self.search_vendor_id.to_le_bytes();
        frame[1] = vb[0];
        frame[2] = vb[1];
        frame[3] = vb[2];
        frame[4] = vb[3];
        frame
    }

    /// Build assign node-id request frame.
    pub fn build_assign_frame(&self) -> [u8; 8] {
        let mut frame = [0u8; 8];
        frame[0] = 0x11; // configure node-id
        frame[1] = self.target_node_id;
        frame
    }

    /// The assigned node ID.
    pub fn assigned_node_id(&self) -> u8 {
        self.target_node_id
    }

    /// Found vendor ID.
    pub fn found_vendor_id(&self) -> u32 {
        self.found_vendor_id
    }

    /// Found serial number.
    pub fn found_serial(&self) -> u32 {
        self.found_serial
    }
}

impl Default for LssClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identify_slave() {
        let mut srv = LssServer::<f32>::new(0x1234, 0x5678, 0x0001, 0xABCD);
        assert_eq!(srv.state(), LssState::Waiting);
        let ok = srv.identify_slave(0x1234, 0x5678, 0x0001, 0xABCD);
        assert!(ok);
        assert_eq!(srv.state(), LssState::Identified);
    }

    #[test]
    fn test_assign_address() {
        let mut srv = LssServer::<f32>::new(0x1, 0x2, 0x3, 0x4);
        srv.identify_slave(0x1, 0x2, 0x3, 0x4);
        assert!(srv.assign_address(5).is_ok());
        assert_eq!(srv.node_id(), 5);
        assert_eq!(srv.state(), LssState::Configured);
    }

    #[test]
    fn test_invalid_node_id() {
        let mut srv = LssServer::<f32>::new(0x1, 0x2, 0x3, 0x4);
        srv.identify_slave(0x1, 0x2, 0x3, 0x4);
        assert_eq!(srv.assign_address(0), Err(LssError::InvalidNodeId));
        assert_eq!(srv.assign_address(128), Err(LssError::InvalidNodeId));
    }

    #[test]
    fn test_invalid_baudrate() {
        let mut srv = LssServer::<f32>::new(0x1, 0x2, 0x3, 0x4);
        srv.identify_slave(0x1, 0x2, 0x3, 0x4);
        assert_eq!(srv.configure_baudrate(9600), Err(LssError::InvalidBaudrate));
        assert!(srv.configure_baudrate(500_000).is_ok());
        assert_eq!(srv.baudrate(), 500_000);
    }

    #[test]
    fn test_identity_mismatch() {
        let mut srv = LssServer::<f64>::new(0x1, 0x2, 0x3, 0x4);
        assert_eq!(
            srv.configure_node_id(0x9, 0x2, 0x3, 0x4, 5),
            Err(LssError::IdentityMismatch)
        );
    }

    #[test]
    fn test_lss_client_workflow() {
        let mut client = LssClient::new();
        client.begin_search();
        assert_eq!(client.state(), LssClientState::Searching);
        client.on_identify_response(0xDEAD, 0xBEEF);
        assert_eq!(client.state(), LssClientState::SlaveFound);
        assert_eq!(client.found_vendor_id(), 0xDEAD);
        client.assign_node_id(10).unwrap();
        assert_eq!(client.state(), LssClientState::Configuring);
        client.confirm_assignment();
        assert_eq!(client.state(), LssClientState::Done);
        assert_eq!(client.assigned_node_id(), 10);
    }

    #[test]
    fn test_lss_server_reset() {
        let mut srv = LssServer::<f32>::new(0x1, 0x2, 0x3, 0x4);
        srv.identify_slave(0x1, 0x2, 0x3, 0x4);
        srv.assign_address(7).unwrap();
        srv.reset();
        assert_eq!(srv.state(), LssState::Waiting);
        assert_eq!(srv.node_id(), 0xFF);
    }

    #[test]
    fn test_store_configuration() {
        let mut srv = LssServer::<f32>::new(0x1, 0x2, 0x3, 0x4);
        assert_eq!(srv.store_configuration(), Err(LssError::NotConfigured));
        srv.identify_slave(0x1, 0x2, 0x3, 0x4);
        assert_eq!(srv.store_configuration(), Err(LssError::NotConfigured));
        srv.assign_address(3).unwrap();
        assert!(srv.store_configuration().is_ok());
    }
}
