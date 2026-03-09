//! Modbus TCP framing — Ethernet/IP encapsulation (MBAP header).
//!
//! Modbus TCP adds a 6-byte MBAP (Modbus Application Protocol) header
//! to the standard PDU:
//!
//! ```text
//! [Transaction ID: 2] [Protocol ID: 2=0x0000] [Length: 2] [Unit ID: 1] [PDU: N]
//! ```
//!
//! This module provides:
//! - `encode_tcp` / `decode_tcp` — pure framing functions (no I/O)
//! - `TcpSession` — transaction counter and high-level encode/decode helpers
//!
//! No actual TCP socket is created here; I/O is the caller's responsibility.

use super::register::ModbusError;

// ─── MBAP header ─────────────────────────────────────────────────────────────

/// Modbus Application Protocol (MBAP) header — 7 bytes on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MbapHeader {
    /// Transaction identifier (echoed by server).
    pub transaction_id: u16,
    /// Protocol identifier (always 0 for Modbus).
    pub protocol_id: u16,
    /// Length of remaining frame (unit_id + PDU).
    pub length: u16,
    /// Unit identifier (slave ID on serial subnet).
    pub unit_id: u8,
}

impl MbapHeader {
    /// Construct a valid MBAP header for a PDU of `pdu_len` bytes.
    pub fn new(transaction_id: u16, unit_id: u8, pdu_len: u16) -> Self {
        Self {
            transaction_id,
            protocol_id: 0x0000,
            length: 1 + pdu_len, // unit_id byte + PDU
            unit_id,
        }
    }

    /// Serialize to 7 bytes (big-endian).
    pub fn to_bytes(&self) -> [u8; 7] {
        let tid = self.transaction_id.to_be_bytes();
        let pid = self.protocol_id.to_be_bytes();
        let len = self.length.to_be_bytes();
        [tid[0], tid[1], pid[0], pid[1], len[0], len[1], self.unit_id]
    }

    /// Parse from a 7-byte slice.
    pub fn from_bytes(b: &[u8; 7]) -> Self {
        Self {
            transaction_id: u16::from_be_bytes([b[0], b[1]]),
            protocol_id: u16::from_be_bytes([b[2], b[3]]),
            length: u16::from_be_bytes([b[4], b[5]]),
            unit_id: b[6],
        }
    }
}

// ─── TcpFrame ────────────────────────────────────────────────────────────────

/// Modbus TCP frame (MBAP header + PDU).
#[derive(Debug, Clone)]
pub struct TcpFrame {
    pub header: MbapHeader,
    pub function_code: u8,
    pub data: heapless::Vec<u8, 253>,
}

impl TcpFrame {
    /// Build Read Holding Registers request.
    pub fn read_holding_registers(
        transaction_id: u16,
        unit_id: u8,
        start: u16,
        count: u16,
    ) -> Self {
        let mut data = heapless::Vec::new();
        let _ = data.extend_from_slice(&start.to_be_bytes());
        let _ = data.extend_from_slice(&count.to_be_bytes());
        let pdu_len = 1 + data.len() as u16; // FC + data
        Self {
            header: MbapHeader::new(transaction_id, unit_id, pdu_len),
            function_code: 0x03,
            data,
        }
    }

    /// Build Write Single Register request.
    pub fn write_single_register(transaction_id: u16, unit_id: u8, reg: u16, val: u16) -> Self {
        let mut data = heapless::Vec::new();
        let _ = data.extend_from_slice(&reg.to_be_bytes());
        let _ = data.extend_from_slice(&val.to_be_bytes());
        let pdu_len = 1 + data.len() as u16;
        Self {
            header: MbapHeader::new(transaction_id, unit_id, pdu_len),
            function_code: 0x06,
            data,
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> heapless::Vec<u8, 260> {
        let mut buf: heapless::Vec<u8, 260> = heapless::Vec::new();
        let _ = buf.extend_from_slice(&self.header.to_bytes());
        let _ = buf.push(self.function_code);
        let _ = buf.extend_from_slice(&self.data);
        buf
    }

    /// Parse from bytes. Returns None if too short or invalid protocol ID.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }
        let header_bytes: [u8; 7] = bytes[..7].try_into().ok()?;
        let header = MbapHeader::from_bytes(&header_bytes);
        if header.protocol_id != 0 {
            return None;
        }
        let function_code = bytes[7];
        let mut data = heapless::Vec::new();
        if bytes.len() > 8 {
            let _ = data.extend_from_slice(&bytes[8..]);
        }
        Some(Self {
            header,
            function_code,
            data,
        })
    }
}

// ─── encode_tcp / decode_tcp ──────────────────────────────────────────────────

/// Encode a Modbus TCP frame into `buf`.
///
/// The frame format: `[MBAP header: 7][PDU: pdu.len()]`.
/// Total frame size = 7 + pdu.len() bytes.
///
/// Returns the total number of bytes written, or `ModbusError::IllegalDataValue`
/// if `buf` is too small.
pub fn encode_tcp(
    transaction_id: u16,
    unit_id: u8,
    pdu: &[u8],
    buf: &mut [u8],
) -> Result<usize, ModbusError> {
    let pdu_len = pdu.len();
    let total = 7 + pdu_len; // MBAP (7) + PDU
    if buf.len() < total {
        return Err(ModbusError::IllegalDataValue);
    }
    let header = MbapHeader::new(transaction_id, unit_id, pdu_len as u16);
    let hdr_bytes = header.to_bytes();
    buf[..7].copy_from_slice(&hdr_bytes);
    buf[7..7 + pdu_len].copy_from_slice(pdu);
    Ok(total)
}

/// Decode a Modbus TCP frame from `buf`.
///
/// Returns `(transaction_id, unit_id, pdu_slice)` on success.
///
/// Errors:
/// - `IllegalDataValue` — buffer shorter than MBAP header (7 bytes), or declared
///   length field is inconsistent with buffer length.
/// - `IllegalFunction` — protocol_id is not 0 (not Modbus).
pub fn decode_tcp(buf: &[u8]) -> Result<(u16, u8, &[u8]), ModbusError> {
    if buf.len() < 7 {
        return Err(ModbusError::IllegalDataValue);
    }
    let hdr_bytes: [u8; 7] = buf[..7]
        .try_into()
        .map_err(|_| ModbusError::IllegalDataValue)?;
    let header = MbapHeader::from_bytes(&hdr_bytes);

    if header.protocol_id != 0x0000 {
        return Err(ModbusError::IllegalFunction);
    }

    // `length` = unit_id (1) + PDU length, so PDU length = length - 1
    let declared_pdu_and_uid = header.length as usize;
    if declared_pdu_and_uid < 1 {
        return Err(ModbusError::IllegalDataValue);
    }
    let pdu_len = declared_pdu_and_uid - 1; // subtract unit_id byte
    let total_expected = 7 + pdu_len;
    if buf.len() < total_expected {
        return Err(ModbusError::IllegalDataValue);
    }
    let pdu = &buf[7..7 + pdu_len];
    Ok((header.transaction_id, header.unit_id, pdu))
}

// ─── TcpSession ──────────────────────────────────────────────────────────────

/// Modbus TCP session — tracks the transaction ID counter.
///
/// Each request generated by this session gets a unique, monotonically
/// incrementing transaction ID.  The session wraps at `u16::MAX` → 0.
///
/// No socket or I/O: the caller owns the transport layer.
#[derive(Debug, Default)]
pub struct TcpSession {
    next_transaction_id: u16,
}

impl TcpSession {
    /// Create a new session (transaction ID starts at 0).
    pub const fn new() -> Self {
        Self {
            next_transaction_id: 0,
        }
    }

    /// Allocate and return the next transaction ID.
    pub fn next_transaction_id(&mut self) -> u16 {
        let id = self.next_transaction_id;
        self.next_transaction_id = self.next_transaction_id.wrapping_add(1);
        id
    }

    /// Encode a request PDU into `buf`, automatically assigning a transaction ID.
    ///
    /// Returns `(transaction_id, bytes_written)` on success, or
    /// `ModbusError::IllegalDataValue` if `buf` is too small.
    pub fn encode_request(
        &mut self,
        unit_id: u8,
        pdu: &[u8],
        buf: &mut [u8],
    ) -> Result<(u16, usize), ModbusError> {
        let tid = self.next_transaction_id();
        let n = encode_tcp(tid, unit_id, pdu, buf)?;
        Ok((tid, n))
    }

    /// Decode a response from `buf`.
    ///
    /// Returns `(transaction_id, unit_id, pdu_slice)`.  The caller is
    /// responsible for matching `transaction_id` to an outstanding request.
    pub fn decode_response<'a>(&self, buf: &'a [u8]) -> Result<(u16, u8, &'a [u8]), ModbusError> {
        decode_tcp(buf)
    }

    /// Peek at the transaction ID that will be assigned to the *next* request
    /// (without incrementing the counter).
    pub fn peek_next_transaction_id(&self) -> u16 {
        self.next_transaction_id
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MbapHeader ────────────────────────────────────────────────────────────

    #[test]
    fn mbap_header_roundtrip() {
        let h = MbapHeader::new(0x1234, 1, 5);
        let b = h.to_bytes();
        let h2 = MbapHeader::from_bytes(&b);
        assert_eq!(h2.transaction_id, 0x1234);
        assert_eq!(h2.unit_id, 1);
        assert_eq!(h2.length, 6); // 1 + 5
        assert_eq!(h2.protocol_id, 0x0000);
    }

    // ── TcpFrame ─────────────────────────────────────────────────────────────

    #[test]
    fn tcp_frame_roundtrip() {
        let frame = TcpFrame::read_holding_registers(42, 1, 0, 10);
        let bytes = frame.to_bytes();
        let parsed = TcpFrame::from_bytes(&bytes).expect("parse failed");
        assert_eq!(parsed.header.transaction_id, 42);
        assert_eq!(parsed.function_code, 0x03);
    }

    #[test]
    fn tcp_frame_invalid_protocol_id() {
        let frame = TcpFrame::write_single_register(1, 1, 100, 200);
        let mut bytes = frame.to_bytes();
        bytes[2] = 0x01; // corrupt protocol ID
        assert!(TcpFrame::from_bytes(&bytes).is_none());
    }

    // ── encode_tcp / decode_tcp ───────────────────────────────────────────────

    #[test]
    fn encode_decode_tcp_roundtrip() {
        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x0A]; // FC03: read 10 registers
        let mut buf = [0u8; 256];
        let n = encode_tcp(0x1234, 1, &pdu, &mut buf).expect("encode");
        assert_eq!(n, 7 + pdu.len());

        let (tid, uid, pdu_back) = decode_tcp(&buf[..n]).expect("decode");
        assert_eq!(tid, 0x1234);
        assert_eq!(uid, 1);
        assert_eq!(pdu_back, &pdu);
    }

    #[test]
    fn encode_tcp_buffer_too_small() {
        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x01];
        let mut buf = [0u8; 5]; // needs 12
        assert_eq!(
            encode_tcp(1, 1, &pdu, &mut buf),
            Err(ModbusError::IllegalDataValue)
        );
    }

    #[test]
    fn decode_tcp_too_short() {
        let buf = [0u8; 6]; // needs 7
        assert_eq!(decode_tcp(&buf), Err(ModbusError::IllegalDataValue));
    }

    #[test]
    fn decode_tcp_wrong_protocol_id() {
        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x01];
        let mut buf = [0u8; 256];
        let n = encode_tcp(1, 1, &pdu, &mut buf).expect("encode");
        buf[2] = 0x01; // corrupt protocol ID
        assert_eq!(decode_tcp(&buf[..n]), Err(ModbusError::IllegalFunction));
    }

    #[test]
    fn decode_tcp_length_mismatch() {
        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x01];
        let mut buf = [0u8; 256];
        let n = encode_tcp(1, 1, &pdu, &mut buf).expect("encode");
        // Inflate the declared length so it exceeds the actual buffer
        buf[4] = 0x7F;
        buf[5] = 0xFF;
        assert_eq!(decode_tcp(&buf[..n]), Err(ModbusError::IllegalDataValue));
    }

    // ── TcpSession ────────────────────────────────────────────────────────────

    #[test]
    fn tcp_session_transaction_id_increments() {
        let mut session = TcpSession::new();
        assert_eq!(session.peek_next_transaction_id(), 0);
        assert_eq!(session.next_transaction_id(), 0);
        assert_eq!(session.next_transaction_id(), 1);
        assert_eq!(session.next_transaction_id(), 2);
        assert_eq!(session.peek_next_transaction_id(), 3);
    }

    #[test]
    fn tcp_session_transaction_id_wraps() {
        let mut session = TcpSession {
            next_transaction_id: u16::MAX,
        };
        assert_eq!(session.next_transaction_id(), u16::MAX);
        assert_eq!(session.next_transaction_id(), 0); // wraps
    }

    #[test]
    fn tcp_session_encode_request_assigns_tid() {
        let mut session = TcpSession::new();
        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x01];
        let mut buf = [0u8; 256];
        let (tid, n) = session.encode_request(1, &pdu, &mut buf).expect("encode");
        assert_eq!(tid, 0);
        assert_eq!(n, 7 + pdu.len());
        // Second request gets tid=1
        let (tid2, _) = session.encode_request(1, &pdu, &mut buf).expect("encode 2");
        assert_eq!(tid2, 1);
    }

    #[test]
    fn tcp_session_decode_response() {
        let mut session = TcpSession::new();
        let pdu = [0x03u8, 0x02, 0xAB, 0xCD]; // FC03 response: 1 register = 0xABCD
        let mut buf = [0u8; 256];
        let (tid_sent, n) = session.encode_request(1, &pdu, &mut buf).expect("encode");
        let (tid_recv, uid, pdu_back) = session.decode_response(&buf[..n]).expect("decode");
        assert_eq!(tid_sent, tid_recv);
        assert_eq!(uid, 1);
        assert_eq!(pdu_back, &pdu);
    }

    #[test]
    fn tcp_session_encode_request_buffer_too_small() {
        let mut session = TcpSession::new();
        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x01];
        let mut buf = [0u8; 3]; // too small
        assert_eq!(
            session.encode_request(1, &pdu, &mut buf),
            Err(ModbusError::IllegalDataValue)
        );
    }
}
