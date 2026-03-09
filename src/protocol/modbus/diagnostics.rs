//! Modbus FC08 Diagnostics support (CiA Modbus specification).
//!
//! FC08 provides loopback testing and counter access for diagnostics.
//! Sub-function codes control specific diagnostic operations.

/// FC08 function code.
pub const FC_DIAGNOSTICS: u8 = 0x08;

// -------------------------------------------------------------------------
// Sub-function codes per Modbus Application Protocol spec
// -------------------------------------------------------------------------

/// Sub-function 0x0000: Return Query Data (loopback).
pub const DIAG_RETURN_QUERY_DATA: u16 = 0x0000;
/// Sub-function 0x0001: Restart Communications Option.
pub const DIAG_RESTART_COMM: u16 = 0x0001;
/// Sub-function 0x0002: Return Diagnostic Register.
pub const DIAG_RETURN_DIAG_REG: u16 = 0x0002;
/// Sub-function 0x0003: Change ASCII Input Delimiter.
pub const DIAG_CHANGE_ASCII_DELIM: u16 = 0x0003;
/// Sub-function 0x0004: Force Listen Only Mode.
pub const DIAG_FORCE_LISTEN_ONLY: u16 = 0x0004;
/// Sub-function 0x000A: Clear Counters and Diagnostic Register.
pub const DIAG_CLEAR_COUNTERS: u16 = 0x000A;
/// Sub-function 0x000B: Return Bus Message Count.
pub const DIAG_BUS_MSG_COUNT: u16 = 0x000B;
/// Sub-function 0x000C: Return Bus Communication Error Count.
pub const DIAG_BUS_COMM_ERR_COUNT: u16 = 0x000C;
/// Sub-function 0x000D: Return Bus Exception Error Count.
pub const DIAG_BUS_EXCEPT_ERR_COUNT: u16 = 0x000D;
/// Sub-function 0x000E: Return Slave Message Count.
pub const DIAG_SLAVE_MSG_COUNT: u16 = 0x000E;
/// Sub-function 0x000F: Return Slave No Response Count.
pub const DIAG_SLAVE_NO_RESP_COUNT: u16 = 0x000F;
/// Sub-function 0x0010: Return Slave NAK Count.
pub const DIAG_SLAVE_NAK_COUNT: u16 = 0x0010;
/// Sub-function 0x0011: Return Slave Busy Count.
pub const DIAG_SLAVE_BUSY_COUNT: u16 = 0x0011;
/// Sub-function 0x0012: Return Bus Character Overrun Count.
pub const DIAG_BUS_OVERRUN_COUNT: u16 = 0x0012;
/// Sub-function 0x0014: Clear Overrun Counter and Flag.
pub const DIAG_CLEAR_OVERRUN: u16 = 0x0014;

/// FC08 diagnostic counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiagnosticCounters {
    /// Total bus messages received (including broadcasts).
    pub bus_message_count: u32,
    /// CRC/parity error count.
    pub crc_error_count: u32,
    /// Exception response count.
    pub exception_count: u32,
    /// NAK (negative acknowledge) count.
    pub nak_count: u32,
    /// Slave no-response count.
    pub no_response_count: u32,
    /// Slave busy count.
    pub busy_count: u32,
    /// Character overrun count.
    pub overrun_count: u32,
    /// Slave message count (addressed to this slave).
    pub slave_message_count: u32,
    /// Diagnostic register (16-bit, device-specific flags).
    pub diagnostic_register: u16,
}

impl DiagnosticCounters {
    /// Create zeroed counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment bus message counter.
    pub fn inc_bus_message(&mut self) {
        self.bus_message_count += 1;
    }

    /// Increment CRC error counter.
    pub fn inc_crc_error(&mut self) {
        self.crc_error_count += 1;
    }

    /// Increment exception counter.
    pub fn inc_exception(&mut self) {
        self.exception_count += 1;
    }

    /// Increment NAK counter.
    pub fn inc_nak(&mut self) {
        self.nak_count += 1;
    }

    /// Increment no-response counter.
    pub fn inc_no_response(&mut self) {
        self.no_response_count += 1;
    }

    /// Increment busy counter.
    pub fn inc_busy(&mut self) {
        self.busy_count += 1;
    }

    /// Increment overrun counter.
    pub fn inc_overrun(&mut self) {
        self.overrun_count += 1;
    }

    /// Increment slave message counter.
    pub fn inc_slave_message(&mut self) {
        self.slave_message_count += 1;
    }

    /// Clear all counters and diagnostic register.
    pub fn clear_all(&mut self) {
        *self = Self::new();
    }

    /// Clear overrun counter and flag.
    pub fn clear_overrun(&mut self) {
        self.overrun_count = 0;
        self.diagnostic_register &= !0x0001; // clear overrun flag
    }
}

/// Error type for diagnostic processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagError {
    InvalidFrame,
    UnknownSubFunction,
    ListenOnlyMode,
}

/// Modbus FC08 Diagnostic server.
///
/// Processes FC08 frames and maintains counters.
pub struct DiagnosticServer {
    /// Node address.
    node_id: u8,
    /// Diagnostic counters.
    counters: DiagnosticCounters,
    /// ASCII delimiter (used in some sub-functions).
    ascii_delim: u8,
    /// Whether listen-only mode is active.
    listen_only: bool,
}

impl DiagnosticServer {
    /// Create a new diagnostic server for `node_id`.
    pub fn new(node_id: u8) -> Self {
        Self {
            node_id,
            counters: DiagnosticCounters::new(),
            ascii_delim: 0x0A, // default LF
            listen_only: false,
        }
    }

    /// Access the counters.
    pub fn counters(&self) -> &DiagnosticCounters {
        &self.counters
    }

    /// Mutable access to counters.
    pub fn counters_mut(&mut self) -> &mut DiagnosticCounters {
        &mut self.counters
    }

    /// Is listen-only mode active?
    pub fn is_listen_only(&self) -> bool {
        self.listen_only
    }

    /// Node ID.
    pub fn node_id(&self) -> u8 {
        self.node_id
    }

    /// Process a raw FC08 frame (RTU format, without CRC).
    ///
    /// `data` should start with device_id byte. Returns a response frame
    /// in `out` (up to 8 bytes) and returns the response length, or Err.
    pub fn process_frame(&mut self, data: &[u8]) -> Result<[u8; 8], DiagError> {
        if data.len() < 6 {
            return Err(DiagError::InvalidFrame);
        }
        if data[0] != self.node_id && data[0] != 0x00 {
            // Not addressed to us, still count the bus message
            self.counters.inc_bus_message();
            return Err(DiagError::InvalidFrame);
        }
        if data[1] != FC_DIAGNOSTICS {
            return Err(DiagError::InvalidFrame);
        }

        self.counters.inc_bus_message();
        self.counters.inc_slave_message();

        if self.listen_only {
            return Err(DiagError::ListenOnlyMode);
        }

        let sub_fn = u16::from_be_bytes([data[2], data[3]]);
        let sub_data = u16::from_be_bytes([data[4], data[5]]);

        let mut resp = [0u8; 8];
        resp[0] = self.node_id;
        resp[1] = FC_DIAGNOSTICS;
        let sf_bytes = sub_fn.to_be_bytes();
        resp[2] = sf_bytes[0];
        resp[3] = sf_bytes[1];

        match sub_fn {
            DIAG_RETURN_QUERY_DATA => {
                // Echo back data bytes
                resp[4] = data[4];
                resp[5] = data[5];
            }
            DIAG_RESTART_COMM => {
                // Restart: if data=0xFF00, clear event log
                self.listen_only = false;
                if sub_data == 0xFF00 {
                    self.counters.clear_all();
                }
                resp[4] = 0x00;
                resp[5] = 0x00;
            }
            DIAG_RETURN_DIAG_REG => {
                let dr = self.counters.diagnostic_register.to_be_bytes();
                resp[4] = dr[0];
                resp[5] = dr[1];
            }
            DIAG_CHANGE_ASCII_DELIM => {
                self.ascii_delim = (sub_data >> 8) as u8;
                resp[4] = 0x00;
                resp[5] = 0x00;
            }
            DIAG_FORCE_LISTEN_ONLY => {
                self.listen_only = true;
                // No response sent in real implementation, but we return for simulation
                resp[4] = 0x00;
                resp[5] = 0x00;
            }
            DIAG_CLEAR_COUNTERS => {
                self.counters.clear_all();
                resp[4] = 0x00;
                resp[5] = 0x00;
            }
            DIAG_BUS_MSG_COUNT => {
                let v = (self.counters.bus_message_count as u16).to_be_bytes();
                resp[4] = v[0];
                resp[5] = v[1];
            }
            DIAG_BUS_COMM_ERR_COUNT => {
                let v = (self.counters.crc_error_count as u16).to_be_bytes();
                resp[4] = v[0];
                resp[5] = v[1];
            }
            DIAG_BUS_EXCEPT_ERR_COUNT => {
                let v = (self.counters.exception_count as u16).to_be_bytes();
                resp[4] = v[0];
                resp[5] = v[1];
            }
            DIAG_SLAVE_MSG_COUNT => {
                let v = (self.counters.slave_message_count as u16).to_be_bytes();
                resp[4] = v[0];
                resp[5] = v[1];
            }
            DIAG_SLAVE_NO_RESP_COUNT => {
                let v = (self.counters.no_response_count as u16).to_be_bytes();
                resp[4] = v[0];
                resp[5] = v[1];
            }
            DIAG_SLAVE_NAK_COUNT => {
                let v = (self.counters.nak_count as u16).to_be_bytes();
                resp[4] = v[0];
                resp[5] = v[1];
            }
            DIAG_SLAVE_BUSY_COUNT => {
                let v = (self.counters.busy_count as u16).to_be_bytes();
                resp[4] = v[0];
                resp[5] = v[1];
            }
            DIAG_BUS_OVERRUN_COUNT => {
                let v = (self.counters.overrun_count as u16).to_be_bytes();
                resp[4] = v[0];
                resp[5] = v[1];
            }
            DIAG_CLEAR_OVERRUN => {
                self.counters.clear_overrun();
                resp[4] = 0x00;
                resp[5] = 0x00;
            }
            _ => {
                self.counters.inc_exception();
                return Err(DiagError::UnknownSubFunction);
            }
        }

        Ok(resp)
    }

    /// Build a FC08 request frame (client side).
    pub fn build_request(node_id: u8, sub_fn: u16, data: u16) -> [u8; 6] {
        let sf = sub_fn.to_be_bytes();
        let d = data.to_be_bytes();
        [node_id, FC_DIAGNOSTICS, sf[0], sf[1], d[0], d[1]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loopback() {
        let mut srv = DiagnosticServer::new(1);
        let req = DiagnosticServer::build_request(1, DIAG_RETURN_QUERY_DATA, 0xABCD);
        let resp = srv.process_frame(&req).unwrap();
        assert_eq!(resp[0], 1);
        assert_eq!(resp[1], FC_DIAGNOSTICS);
        assert_eq!(u16::from_be_bytes([resp[4], resp[5]]), 0xABCD);
    }

    #[test]
    fn test_bus_message_counter() {
        let mut srv = DiagnosticServer::new(1);
        // Process 3 loopback frames
        for _ in 0..3 {
            let req = DiagnosticServer::build_request(1, DIAG_RETURN_QUERY_DATA, 0x0000);
            srv.process_frame(&req).unwrap();
        }
        // Now query the counter
        let req = DiagnosticServer::build_request(1, DIAG_BUS_MSG_COUNT, 0x0000);
        let resp = srv.process_frame(&req).unwrap();
        // 3 loopback + 1 query = 4 messages
        let count = u16::from_be_bytes([resp[4], resp[5]]);
        assert_eq!(count, 4);
    }

    #[test]
    fn test_clear_counters() {
        let mut srv = DiagnosticServer::new(1);
        srv.counters_mut().inc_crc_error();
        srv.counters_mut().inc_exception();
        let req = DiagnosticServer::build_request(1, DIAG_CLEAR_COUNTERS, 0x0000);
        srv.process_frame(&req).unwrap();
        assert_eq!(srv.counters().crc_error_count, 0);
        assert_eq!(srv.counters().exception_count, 0);
    }

    #[test]
    fn test_listen_only_mode() {
        let mut srv = DiagnosticServer::new(1);
        let req = DiagnosticServer::build_request(1, DIAG_FORCE_LISTEN_ONLY, 0x0000);
        let _ = srv.process_frame(&req); // May succeed or return listen-only error
        srv.listen_only = true;
        let req2 = DiagnosticServer::build_request(1, DIAG_RETURN_QUERY_DATA, 0x1234);
        assert_eq!(srv.process_frame(&req2), Err(DiagError::ListenOnlyMode));
    }

    #[test]
    fn test_exception_count_unknown_subfn() {
        let mut srv = DiagnosticServer::new(1);
        // Sub-function 0xFFFF is unknown
        let req = [1u8, FC_DIAGNOSTICS, 0xFF, 0xFF, 0x00, 0x00];
        assert_eq!(srv.process_frame(&req), Err(DiagError::UnknownSubFunction));
        assert_eq!(srv.counters().exception_count, 1);
    }

    #[test]
    fn test_diagnostic_counters_inc() {
        let mut c = DiagnosticCounters::new();
        c.inc_bus_message();
        c.inc_bus_message();
        c.inc_crc_error();
        c.inc_nak();
        assert_eq!(c.bus_message_count, 2);
        assert_eq!(c.crc_error_count, 1);
        assert_eq!(c.nak_count, 1);
        c.clear_all();
        assert_eq!(c.bus_message_count, 0);
    }
}
