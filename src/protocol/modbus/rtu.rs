//! Modbus RTU framing — serial (RS-485/RS-232) frame encoding/decoding.
//!
//! RTU frame format:
//!   [Device Address (1)] [Function Code (1)] [Data (N)] [CRC16 (2)]
//!
//! No start/stop delimiters — timing gaps between frames identify frame
//! boundaries.  The minimum inter-frame gap is 3.5 character times.
//!
//! # RtuMaster state machine
//!
//! `RtuMaster<W>` is a generic, no-std state machine that drives the master
//! side of an RTU transaction.  The caller supplies a writer (anything
//! implementing `RtuWriter`) and drives the state machine by feeding bytes
//! and tick counts.

use super::register::ModbusError;

// ─── CRC-16/Modbus ────────────────────────────────────────────────────────────

/// Compute CRC-16/Modbus (polynomial 0xA001, initial value 0xFFFF).
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
        crc ^= byte as u16;
        for _ in 0..8 {
            if crc & 0x0001 != 0 {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    crc
}

// ─── Function codes ───────────────────────────────────────────────────────────

/// Modbus function codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FunctionCode {
    ReadCoils = 0x01,
    ReadDiscreteInputs = 0x02,
    ReadHoldingRegisters = 0x03,
    ReadInputRegisters = 0x04,
    WriteSingleCoil = 0x05,
    WriteSingleRegister = 0x06,
    WriteMultipleCoils = 0x0F,
    WriteMultipleRegisters = 0x10,
    ReportSlaveId = 0x11,
}

// ─── RtuFrame ─────────────────────────────────────────────────────────────────

/// Modbus RTU frame.
#[derive(Debug, Clone)]
pub struct RtuFrame {
    pub device_address: u8,
    pub function_code: u8,
    pub data: heapless::Vec<u8, 256>,
}

impl RtuFrame {
    /// Build a Read Holding Registers (FC03) request frame.
    pub fn read_holding_registers(addr: u8, start: u16, count: u16) -> Self {
        let mut data = heapless::Vec::new();
        let _ = data.extend_from_slice(&start.to_be_bytes());
        let _ = data.extend_from_slice(&count.to_be_bytes());
        Self {
            device_address: addr,
            function_code: 0x03,
            data,
        }
    }

    /// Build a Write Single Register (FC06) request frame.
    pub fn write_single_register(addr: u8, reg: u16, val: u16) -> Self {
        let mut data = heapless::Vec::new();
        let _ = data.extend_from_slice(&reg.to_be_bytes());
        let _ = data.extend_from_slice(&val.to_be_bytes());
        Self {
            device_address: addr,
            function_code: 0x06,
            data,
        }
    }

    /// Serialize to bytes (with CRC16).
    pub fn to_bytes(&self) -> heapless::Vec<u8, 260> {
        let mut buf: heapless::Vec<u8, 260> = heapless::Vec::new();
        let _ = buf.push(self.device_address);
        let _ = buf.push(self.function_code);
        let _ = buf.extend_from_slice(&self.data);
        let crc = crc16(&buf);
        let _ = buf.extend_from_slice(&crc.to_le_bytes()); // RTU: CRC low byte first
        buf
    }

    /// Parse from raw bytes. Returns None if CRC fails or frame too short.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let payload_len = bytes.len() - 2;
        let crc_calc = crc16(&bytes[..payload_len]);
        let crc_recv = u16::from_le_bytes([bytes[payload_len], bytes[payload_len + 1]]);
        if crc_calc != crc_recv {
            return None;
        }

        let mut data = heapless::Vec::new();
        let _ = data.extend_from_slice(&bytes[2..payload_len]);
        Some(Self {
            device_address: bytes[0],
            function_code: bytes[1],
            data,
        })
    }
}

// ─── encode_rtu / decode_rtu ──────────────────────────────────────────────────

/// Encode an RTU frame into `frame_buf`.
///
/// `pdu_buf` contains `[function_code, data...]` (N bytes).
/// The encoded frame is: `[addr][pdu_buf...][crc_lo][crc_hi]`.
///
/// Returns the number of bytes written, or `ModbusError::IllegalDataValue`
/// if `frame_buf` is too small (needs `pdu_buf.len() + 3` bytes).
pub fn encode_rtu(addr: u8, pdu_buf: &[u8], frame_buf: &mut [u8]) -> Result<usize, ModbusError> {
    let needed = pdu_buf.len() + 3; // addr + PDU + 2-byte CRC
    if frame_buf.len() < needed {
        return Err(ModbusError::IllegalDataValue);
    }
    frame_buf[0] = addr;
    frame_buf[1..1 + pdu_buf.len()].copy_from_slice(pdu_buf);
    let crc = crc16(&frame_buf[..1 + pdu_buf.len()]);
    let crc_offset = 1 + pdu_buf.len();
    frame_buf[crc_offset] = (crc & 0xFF) as u8;
    frame_buf[crc_offset + 1] = (crc >> 8) as u8;
    Ok(needed)
}

/// Decode an RTU frame from `frame`.
///
/// Validates CRC and checks minimum length.  Returns `(addr, pdu_slice)` on
/// success where `pdu_slice` is `[function_code, data...]`.
///
/// Errors:
/// - `IllegalDataValue` — frame too short (< 4 bytes)
/// - `SlaveDeviceFailure` — CRC mismatch
pub fn decode_rtu(frame: &[u8]) -> Result<(u8, &[u8]), ModbusError> {
    if frame.len() < 4 {
        return Err(ModbusError::IllegalDataValue);
    }
    let payload_len = frame.len() - 2;
    let crc_calc = crc16(&frame[..payload_len]);
    let crc_recv = u16::from_le_bytes([frame[payload_len], frame[payload_len + 1]]);
    if crc_calc != crc_recv {
        return Err(ModbusError::SlaveDeviceFailure); // CRC error → slave device failure
    }
    let addr = frame[0];
    let pdu = &frame[1..payload_len];
    Ok((addr, pdu))
}

// ─── Silent interval ──────────────────────────────────────────────────────────

/// Compute the minimum inter-frame silent interval in **microseconds**.
///
/// Per Modbus spec: 3.5 character times.  At baud ≥ 19200, a fixed 1750 µs
/// interval is used; at lower baud rates, the time is calculated precisely.
///
/// `bits_per_char` is typically 11 (1 start + 8 data + 1 parity + 1 stop or
/// 1 start + 8 data + 2 stop = 11 bits).
pub const fn silent_interval_us(baud_rate: u32, bits_per_char: u32) -> u32 {
    if baud_rate >= 19_200 {
        // Fixed 1750 µs per Modbus standard (3.5 × 1/19200 × 11 ≈ 2004 µs,
        // but the spec mandates 1.75 ms for baud ≥ 19200)
        1_750
    } else {
        // 3.5 × bits_per_char × 1_000_000 / baud_rate (µs)
        // Multiply first to keep integer precision.
        (3_500_000u64 * bits_per_char as u64 / baud_rate as u64) as u32
    }
}

// ─── RtuWriter trait ──────────────────────────────────────────────────────────

/// Abstraction over a serial write+flush operation used by `RtuMaster`.
///
/// Implementations are expected to be infallible at the trait level; they
/// return an opaque `E` for transport-layer errors which `RtuMaster` wraps
/// into `ModbusError::SlaveDeviceFailure`.
pub trait RtuWriter {
    /// Error type for write / flush failures.
    type Error;

    /// Write `bytes` to the serial line.
    fn write(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;

    /// Flush the transmit buffer.
    fn flush(&mut self) -> Result<(), Self::Error>;
}

// ─── RtuMaster state machine ──────────────────────────────────────────────────

/// State of the RTU master.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtuMasterState {
    /// Idle, ready to send a new request.
    Idle,
    /// Waiting for the first byte of the response.
    WaitingForResponse,
    /// Accumulating response bytes.
    ReceivingResponse,
    /// A complete response has been received and is ready to decode.
    ResponseReady,
    /// An error occurred; master must be reset.
    Error,
}

/// RTU master state machine.
///
/// `W` is the writer type implementing `RtuWriter`. The caller feeds received
/// bytes via `feed_byte` and elapsed time via `tick`.
///
/// # Usage
///
/// ```text
/// let mut master = RtuMaster::new(writer, 9600, 11);
/// master.send_request(1, &pdu_buf[..pdu_len])?;
/// // ... call feed_byte / tick as bytes arrive ...
/// if let Some(response) = master.take_response() {
///     // decode response PDU
/// }
/// ```
pub struct RtuMaster<W: RtuWriter> {
    writer: W,
    state: RtuMasterState,
    /// Response accumulation buffer.
    rx_buf: heapless::Vec<u8, 260>,
    /// Expected response length (if known; 0 = unknown, accumulate until gap).
    expected_len: usize,
    /// Microseconds elapsed since last byte received.
    silence_us: u32,
    /// Inter-frame gap threshold in microseconds.
    gap_threshold_us: u32,
    /// Pending response length when state = ResponseReady.
    ready_len: usize,
}

impl<W: RtuWriter> RtuMaster<W> {
    /// Create a new RTU master.
    ///
    /// `baud_rate` and `bits_per_char` are used to compute the inter-frame gap.
    pub fn new(writer: W, baud_rate: u32, bits_per_char: u32) -> Self {
        Self {
            writer,
            state: RtuMasterState::Idle,
            rx_buf: heapless::Vec::new(),
            expected_len: 0,
            silence_us: 0,
            gap_threshold_us: silent_interval_us(baud_rate, bits_per_char),
            ready_len: 0,
        }
    }

    /// Borrow the inner writer.
    pub fn writer(&self) -> &W {
        &self.writer
    }

    /// Mutably borrow the inner writer.
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.writer
    }

    /// Current state.
    pub fn state(&self) -> RtuMasterState {
        self.state
    }

    /// Send a request.  `pdu_buf` must start with the function code.
    ///
    /// Returns `ModbusError::SlaveDeviceFailure` if the underlying writer fails,
    /// and `ModbusError::IllegalDataValue` if the master is not idle.
    pub fn send_request(&mut self, addr: u8, pdu_buf: &[u8]) -> Result<(), ModbusError> {
        if self.state != RtuMasterState::Idle {
            return Err(ModbusError::IllegalDataValue);
        }
        let mut frame_buf = [0u8; 260];
        let n = encode_rtu(addr, pdu_buf, &mut frame_buf)?;
        self.writer
            .write(&frame_buf[..n])
            .map_err(|_| ModbusError::SlaveDeviceFailure)?;
        self.writer
            .flush()
            .map_err(|_| ModbusError::SlaveDeviceFailure)?;
        self.rx_buf.clear();
        self.expected_len = 0;
        self.silence_us = 0;
        self.ready_len = 0;
        self.state = RtuMasterState::WaitingForResponse;
        Ok(())
    }

    /// Feed a received byte into the state machine.
    pub fn feed_byte(&mut self, byte: u8) {
        match self.state {
            RtuMasterState::WaitingForResponse | RtuMasterState::ReceivingResponse => {
                if self.rx_buf.push(byte).is_err() {
                    // Buffer overflow — frame is too large.
                    self.state = RtuMasterState::Error;
                    return;
                }
                self.silence_us = 0;
                self.state = RtuMasterState::ReceivingResponse;

                // If we know the expected length and have received enough, mark ready.
                if self.expected_len > 0 && self.rx_buf.len() >= self.expected_len {
                    self.ready_len = self.rx_buf.len();
                    self.state = RtuMasterState::ResponseReady;
                }
            }
            _ => {} // ignore bytes in idle / ready / error states
        }
    }

    /// Advance the silence timer by `elapsed_us` microseconds.
    ///
    /// When the inter-frame gap expires while receiving, the frame is
    /// considered complete and the state transitions to `ResponseReady`.
    pub fn tick(&mut self, elapsed_us: u32) {
        if self.state != RtuMasterState::ReceivingResponse {
            return;
        }
        self.silence_us = self.silence_us.saturating_add(elapsed_us);
        if self.silence_us >= self.gap_threshold_us && !self.rx_buf.is_empty() {
            self.ready_len = self.rx_buf.len();
            self.state = RtuMasterState::ResponseReady;
        }
    }

    /// Set the expected response frame length (optional optimisation).
    ///
    /// When set, `feed_byte` will transition to `ResponseReady` as soon as
    /// `expected_len` bytes have been received, without waiting for the gap.
    pub fn set_expected_response_len(&mut self, len: usize) {
        self.expected_len = len;
    }

    /// Take the raw response frame if ready.
    ///
    /// Returns a reference to the received bytes and resets to `Idle`.
    /// Returns `None` if the state is not `ResponseReady`.
    pub fn take_response(&mut self) -> Option<heapless::Vec<u8, 260>> {
        if self.state != RtuMasterState::ResponseReady {
            return None;
        }
        let frame = self.rx_buf.clone();
        self.rx_buf.clear();
        self.ready_len = 0;
        self.state = RtuMasterState::Idle;
        Some(frame)
    }

    /// Decode and validate the ready response.
    ///
    /// Returns `(addr, pdu_slice_clone)` where `pdu_slice_clone` is a
    /// `heapless::Vec<u8, 256>` containing the PDU bytes.
    ///
    /// Transitions to `Idle` on success or `Error` on CRC failure.
    pub fn decode_response(&mut self) -> Result<(u8, heapless::Vec<u8, 256>), ModbusError> {
        if self.state != RtuMasterState::ResponseReady {
            return Err(ModbusError::IllegalDataValue);
        }
        let raw = self.rx_buf.clone();
        match decode_rtu(&raw) {
            Ok((addr, pdu_slice)) => {
                let mut pdu: heapless::Vec<u8, 256> = heapless::Vec::new();
                pdu.extend_from_slice(pdu_slice)
                    .map_err(|_| ModbusError::IllegalDataValue)?;
                self.rx_buf.clear();
                self.ready_len = 0;
                self.state = RtuMasterState::Idle;
                Ok((addr, pdu))
            }
            Err(e) => {
                self.state = RtuMasterState::Error;
                Err(e)
            }
        }
    }

    /// Reset the state machine to `Idle`.
    pub fn reset(&mut self) {
        self.state = RtuMasterState::Idle;
        self.rx_buf.clear();
        self.expected_len = 0;
        self.silence_us = 0;
        self.ready_len = 0;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CRC tests ─────────────────────────────────────────────────────────────

    #[test]
    fn crc16_known_value() {
        // CRC-16/Modbus for [0x01, 0x03, 0x00, 0x00, 0x00, 0x0A] = 0xCDC5
        let data = [0x01u8, 0x03, 0x00, 0x00, 0x00, 0x0A];
        let crc = crc16(&data);
        assert_eq!(crc, 0xCDC5, "CRC=0x{crc:04X}");
    }

    #[test]
    fn crc16_empty() {
        assert_eq!(crc16(&[]), 0xFFFF);
    }

    // ── RtuFrame tests ────────────────────────────────────────────────────────

    #[test]
    fn rtu_frame_roundtrip() {
        let frame = RtuFrame::read_holding_registers(1, 0, 10);
        let bytes = frame.to_bytes();
        let parsed = RtuFrame::from_bytes(&bytes).expect("parse failed");
        assert_eq!(parsed.device_address, 1);
        assert_eq!(parsed.function_code, 0x03);
    }

    #[test]
    fn rtu_frame_bad_crc() {
        let frame = RtuFrame::write_single_register(1, 100, 0xABCD);
        let mut bytes = frame.to_bytes();
        let len = bytes.len();
        bytes[len - 1] ^= 0xFF;
        assert!(RtuFrame::from_bytes(&bytes).is_none());
    }

    // ── encode_rtu / decode_rtu ───────────────────────────────────────────────

    #[test]
    fn encode_decode_rtu_roundtrip() {
        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x0A]; // FC03 + data
        let mut frame_buf = [0u8; 260];
        let n = encode_rtu(0x01, &pdu, &mut frame_buf).expect("encode failed");
        assert_eq!(n, pdu.len() + 3);

        let (addr, pdu_back) = decode_rtu(&frame_buf[..n]).expect("decode failed");
        assert_eq!(addr, 0x01);
        assert_eq!(pdu_back, &pdu);
    }

    #[test]
    fn encode_rtu_buffer_too_small() {
        let pdu = [0x03u8; 5];
        let mut buf = [0u8; 4]; // needs 8
        assert_eq!(
            encode_rtu(1, &pdu, &mut buf),
            Err(ModbusError::IllegalDataValue)
        );
    }

    #[test]
    fn decode_rtu_too_short() {
        let frame = [0x01u8, 0x03, 0x00]; // only 3 bytes, needs ≥ 4
        assert_eq!(decode_rtu(&frame), Err(ModbusError::IllegalDataValue));
    }

    #[test]
    fn decode_rtu_bad_crc() {
        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x01];
        let mut frame_buf = [0u8; 260];
        let n = encode_rtu(1, &pdu, &mut frame_buf).expect("encode");
        frame_buf[n - 1] ^= 0xFF; // corrupt CRC
        assert_eq!(
            decode_rtu(&frame_buf[..n]),
            Err(ModbusError::SlaveDeviceFailure)
        );
    }

    // ── Silent interval ───────────────────────────────────────────────────────

    #[test]
    fn silent_interval_high_baud() {
        // baud ≥ 19200 → always 1750 µs
        assert_eq!(silent_interval_us(19_200, 11), 1_750);
        assert_eq!(silent_interval_us(115_200, 11), 1_750);
    }

    #[test]
    fn silent_interval_low_baud() {
        // baud = 9600, bits = 11 → 3.5 × 11 × 1e6 / 9600 ≈ 4010 µs
        let us = silent_interval_us(9_600, 11);
        // 3_500_000 × 11 / 9600 = 4010 (integer division)
        assert_eq!(us, 4010);
    }

    // ── RtuMaster state machine ───────────────────────────────────────────────

    /// A simple in-memory writer for unit testing.
    struct BufWriter {
        pub buf: heapless::Vec<u8, 512>,
    }

    impl BufWriter {
        fn new() -> Self {
            Self {
                buf: heapless::Vec::new(),
            }
        }
    }

    impl RtuWriter for BufWriter {
        type Error = ();
        fn write(&mut self, bytes: &[u8]) -> Result<(), ()> {
            self.buf.extend_from_slice(bytes).map_err(|_| ())
        }
        fn flush(&mut self) -> Result<(), ()> {
            Ok(())
        }
    }

    #[test]
    fn master_send_and_receive_response() {
        let writer = BufWriter::new();
        let mut master = RtuMaster::new(writer, 9600, 11);

        // Send a FC03 request to addr 1: read 1 register at 0x0000
        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x01];
        master.send_request(1, &pdu).expect("send");
        assert_eq!(master.state(), RtuMasterState::WaitingForResponse);

        // Build the response frame (addr=1, FC03, byte_count=2, 0xABCD)
        let resp_pdu = [0x03u8, 0x02, 0xAB, 0xCD];
        let mut resp_frame = [0u8; 260];
        let resp_n = encode_rtu(1, &resp_pdu, &mut resp_frame).expect("resp encode");

        // Feed all bytes except the last two (CRC), then feed CRC
        master.set_expected_response_len(resp_n);
        for &b in &resp_frame[..resp_n] {
            master.feed_byte(b);
        }
        assert_eq!(master.state(), RtuMasterState::ResponseReady);

        let (addr, pdu_back) = master.decode_response().expect("decode");
        assert_eq!(addr, 1);
        assert_eq!(pdu_back.as_slice(), &resp_pdu);
        assert_eq!(master.state(), RtuMasterState::Idle);
    }

    #[test]
    fn master_detects_response_via_gap() {
        let writer = BufWriter::new();
        let mut master = RtuMaster::new(writer, 9600, 11);

        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x01];
        master.send_request(1, &pdu).expect("send");

        let resp_pdu = [0x03u8, 0x02, 0x00, 0x01];
        let mut resp_frame = [0u8; 260];
        let resp_n = encode_rtu(1, &resp_pdu, &mut resp_frame).expect("resp encode");

        // Feed bytes
        for &b in &resp_frame[..resp_n] {
            master.feed_byte(b);
        }
        // Still receiving — expected_len not set
        assert_eq!(master.state(), RtuMasterState::ReceivingResponse);

        // Simulate silence > threshold (9600 baud → 4010 µs)
        master.tick(5_000);
        assert_eq!(master.state(), RtuMasterState::ResponseReady);
    }

    #[test]
    fn master_reset_clears_state() {
        let writer = BufWriter::new();
        let mut master = RtuMaster::new(writer, 9600, 11);
        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x01];
        master.send_request(1, &pdu).expect("send");
        master.reset();
        assert_eq!(master.state(), RtuMasterState::Idle);
    }

    #[test]
    fn master_send_while_not_idle_fails() {
        let writer = BufWriter::new();
        let mut master = RtuMaster::new(writer, 9600, 11);
        let pdu = [0x03u8, 0x00, 0x00, 0x00, 0x01];
        master.send_request(1, &pdu).expect("first send");
        let result = master.send_request(1, &pdu);
        assert_eq!(result, Err(ModbusError::IllegalDataValue));
    }
}
