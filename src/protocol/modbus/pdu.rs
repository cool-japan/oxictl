//! Modbus Protocol Data Unit (PDU) encoding and decoding.
//!
//! A PDU consists of a function code (1 byte) followed by function-code-specific data.
//! This module covers FC01–FC06 and FC16, plus exception responses.
//!
//! All data payloads use `heapless::Vec<u8, 256>` — no heap allocation.

use heapless::Vec;

use super::register::ModbusError;

// ─── Exception code ──────────────────────────────────────────────────────────

/// Modbus exception codes (returned in the high-bit response).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExceptionCode {
    /// Function code not supported by this device.
    IllegalFunction = 0x01,
    /// Data address out of range.
    IllegalDataAddress = 0x02,
    /// Data value not accepted.
    IllegalDataValue = 0x03,
    /// Slave device failure.
    SlaveDeviceFailure = 0x04,
}

impl ExceptionCode {
    /// Parse an exception code byte; returns `None` for unrecognised values.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::IllegalFunction),
            0x02 => Some(Self::IllegalDataAddress),
            0x03 => Some(Self::IllegalDataValue),
            0x04 => Some(Self::SlaveDeviceFailure),
            _ => None,
        }
    }
}

impl From<ModbusError> for ExceptionCode {
    fn from(e: ModbusError) -> Self {
        match e {
            ModbusError::IllegalFunction => ExceptionCode::IllegalFunction,
            ModbusError::IllegalDataAddress => ExceptionCode::IllegalDataAddress,
            ModbusError::IllegalDataValue => ExceptionCode::IllegalDataValue,
            ModbusError::SlaveDeviceFailure => ExceptionCode::SlaveDeviceFailure,
        }
    }
}

// ─── Request PDU ─────────────────────────────────────────────────────────────

/// Modbus request PDU variants for FC01–FC06 and FC16.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    /// FC01 — Read Coils: read `count` coils starting at `start`.
    ReadCoils { start: u16, count: u16 },
    /// FC02 — Read Discrete Inputs.
    ReadDiscreteInputs { start: u16, count: u16 },
    /// FC03 — Read Holding Registers.
    ReadHoldingRegisters { start: u16, count: u16 },
    /// FC04 — Read Input Registers.
    ReadInputRegisters { start: u16, count: u16 },
    /// FC05 — Write Single Coil.  `value` must be 0x0000 (off) or 0xFF00 (on).
    WriteSingleCoil { address: u16, value: u16 },
    /// FC06 — Write Single Register.
    WriteSingleRegister { address: u16, value: u16 },
    /// FC16 — Write Multiple Registers.
    WriteMultipleRegisters {
        start: u16,
        count: u16,
        /// Raw register bytes (big-endian pairs).  Max 246 bytes = 123 registers.
        data: Vec<u8, 246>,
    },
}

/// Modbus response PDU variants matching the request set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Response {
    /// FC01 response — coil data, bit-packed.
    ReadCoils { byte_count: u8, data: Vec<u8, 32> },
    /// FC02 response — discrete input data, bit-packed.
    ReadDiscreteInputs { byte_count: u8, data: Vec<u8, 32> },
    /// FC03 response — holding register bytes (big-endian u16 pairs).
    ReadHoldingRegisters { byte_count: u8, data: Vec<u8, 250> },
    /// FC04 response — input register bytes (big-endian u16 pairs).
    ReadInputRegisters { byte_count: u8, data: Vec<u8, 250> },
    /// FC05 echo response.
    WriteSingleCoil { address: u16, value: u16 },
    /// FC06 echo response.
    WriteSingleRegister { address: u16, value: u16 },
    /// FC16 response — confirms write.
    WriteMultipleRegisters { start: u16, count: u16 },
    /// Exception response — function code with high bit set, plus exception code.
    Exception {
        function_code: u8,
        exception: ExceptionCode,
    },
}

// ─── Encode request ───────────────────────────────────────────────────────────

/// Encode a `Request` into `buf` and return the number of bytes written.
///
/// `buf` must be at least 256 bytes.  Returns `ModbusError::IllegalDataValue`
/// if the buffer is too small or data is malformed.
pub fn encode_request(req: &Request, buf: &mut [u8]) -> Result<usize, ModbusError> {
    match req {
        Request::ReadCoils { start, count }
        | Request::ReadDiscreteInputs { start, count }
        | Request::ReadHoldingRegisters { start, count }
        | Request::ReadInputRegisters { start, count } => {
            if buf.len() < 5 {
                return Err(ModbusError::IllegalDataValue);
            }
            buf[0] = request_fc(req);
            let s = start.to_be_bytes();
            let c = count.to_be_bytes();
            buf[1] = s[0];
            buf[2] = s[1];
            buf[3] = c[0];
            buf[4] = c[1];
            Ok(5)
        }
        Request::WriteSingleCoil { address, value }
        | Request::WriteSingleRegister { address, value } => {
            if buf.len() < 5 {
                return Err(ModbusError::IllegalDataValue);
            }
            buf[0] = request_fc(req);
            let a = address.to_be_bytes();
            let v = value.to_be_bytes();
            buf[1] = a[0];
            buf[2] = a[1];
            buf[3] = v[0];
            buf[4] = v[1];
            Ok(5)
        }
        Request::WriteMultipleRegisters { start, count, data } => {
            let byte_count = data.len();
            let needed = 6 + byte_count; // FC + addr(2) + count(2) + byte_cnt(1) + data
            if buf.len() < needed {
                return Err(ModbusError::IllegalDataValue);
            }
            if byte_count > 246 || byte_count % 2 != 0 {
                return Err(ModbusError::IllegalDataValue);
            }
            let s = start.to_be_bytes();
            let c = count.to_be_bytes();
            buf[0] = 0x10;
            buf[1] = s[0];
            buf[2] = s[1];
            buf[3] = c[0];
            buf[4] = c[1];
            buf[5] = byte_count as u8;
            buf[6..6 + byte_count].copy_from_slice(data);
            Ok(needed)
        }
    }
}

/// Return the function code byte for a given request.
fn request_fc(req: &Request) -> u8 {
    match req {
        Request::ReadCoils { .. } => 0x01,
        Request::ReadDiscreteInputs { .. } => 0x02,
        Request::ReadHoldingRegisters { .. } => 0x03,
        Request::ReadInputRegisters { .. } => 0x04,
        Request::WriteSingleCoil { .. } => 0x05,
        Request::WriteSingleRegister { .. } => 0x06,
        Request::WriteMultipleRegisters { .. } => 0x10,
    }
}

// ─── Decode response ──────────────────────────────────────────────────────────

/// Decode a response PDU from `buf`.
///
/// Returns `ModbusError::IllegalDataValue` when the buffer is too short or
/// inconsistent, and `ModbusError::IllegalFunction` for unknown function codes.
pub fn decode_response(buf: &[u8]) -> Result<Response, ModbusError> {
    if buf.is_empty() {
        return Err(ModbusError::IllegalDataValue);
    }
    let fc = buf[0];

    // Exception: high bit set on FC byte.
    if fc & 0x80 != 0 {
        if buf.len() < 2 {
            return Err(ModbusError::IllegalDataValue);
        }
        let exc = ExceptionCode::from_u8(buf[1]).ok_or(ModbusError::IllegalDataValue)?;
        return Ok(Response::Exception {
            function_code: fc & 0x7F,
            exception: exc,
        });
    }

    match fc {
        0x01 => decode_bit_response(buf, |bc, data| Response::ReadCoils {
            byte_count: bc,
            data,
        }),
        0x02 => decode_bit_response(buf, |bc, data| Response::ReadDiscreteInputs {
            byte_count: bc,
            data,
        }),
        0x03 => decode_register_response(buf, |bc, data| Response::ReadHoldingRegisters {
            byte_count: bc,
            data,
        }),
        0x04 => decode_register_response(buf, |bc, data| Response::ReadInputRegisters {
            byte_count: bc,
            data,
        }),
        0x05 => {
            if buf.len() < 5 {
                return Err(ModbusError::IllegalDataValue);
            }
            Ok(Response::WriteSingleCoil {
                address: u16::from_be_bytes([buf[1], buf[2]]),
                value: u16::from_be_bytes([buf[3], buf[4]]),
            })
        }
        0x06 => {
            if buf.len() < 5 {
                return Err(ModbusError::IllegalDataValue);
            }
            Ok(Response::WriteSingleRegister {
                address: u16::from_be_bytes([buf[1], buf[2]]),
                value: u16::from_be_bytes([buf[3], buf[4]]),
            })
        }
        0x10 => {
            if buf.len() < 5 {
                return Err(ModbusError::IllegalDataValue);
            }
            Ok(Response::WriteMultipleRegisters {
                start: u16::from_be_bytes([buf[1], buf[2]]),
                count: u16::from_be_bytes([buf[3], buf[4]]),
            })
        }
        _ => Err(ModbusError::IllegalFunction),
    }
}

/// Helper: decode bit-data (coils / discrete inputs) response.
fn decode_bit_response<F>(buf: &[u8], make: F) -> Result<Response, ModbusError>
where
    F: Fn(u8, Vec<u8, 32>) -> Response,
{
    if buf.len() < 2 {
        return Err(ModbusError::IllegalDataValue);
    }
    let byte_count = buf[1] as usize;
    if buf.len() < 2 + byte_count || byte_count > 32 {
        return Err(ModbusError::IllegalDataValue);
    }
    let mut data: Vec<u8, 32> = Vec::new();
    data.extend_from_slice(&buf[2..2 + byte_count])
        .map_err(|_| ModbusError::IllegalDataValue)?;
    Ok(make(byte_count as u8, data))
}

/// Helper: decode register-data (holding / input registers) response.
fn decode_register_response<F>(buf: &[u8], make: F) -> Result<Response, ModbusError>
where
    F: Fn(u8, Vec<u8, 250>) -> Response,
{
    if buf.len() < 2 {
        return Err(ModbusError::IllegalDataValue);
    }
    let byte_count = buf[1] as usize;
    if buf.len() < 2 + byte_count || byte_count > 250 || byte_count % 2 != 0 {
        return Err(ModbusError::IllegalDataValue);
    }
    let mut data: Vec<u8, 250> = Vec::new();
    data.extend_from_slice(&buf[2..2 + byte_count])
        .map_err(|_| ModbusError::IllegalDataValue)?;
    Ok(make(byte_count as u8, data))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Encode tests ──────────────────────────────────────────────────────────

    #[test]
    fn encode_read_coils() {
        let req = Request::ReadCoils {
            start: 0x0013,
            count: 0x0013,
        };
        let mut buf = [0u8; 256];
        let n = encode_request(&req, &mut buf).expect("encode failed");
        assert_eq!(n, 5);
        assert_eq!(buf[0], 0x01);
        assert_eq!(u16::from_be_bytes([buf[1], buf[2]]), 0x0013);
        assert_eq!(u16::from_be_bytes([buf[3], buf[4]]), 0x0013);
    }

    #[test]
    fn encode_read_holding_registers() {
        let req = Request::ReadHoldingRegisters {
            start: 0x006B,
            count: 0x0003,
        };
        let mut buf = [0u8; 256];
        let n = encode_request(&req, &mut buf).expect("encode failed");
        assert_eq!(n, 5);
        assert_eq!(buf[0], 0x03);
        assert_eq!(u16::from_be_bytes([buf[1], buf[2]]), 0x006B);
        assert_eq!(u16::from_be_bytes([buf[3], buf[4]]), 0x0003);
    }

    #[test]
    fn encode_write_single_register() {
        let req = Request::WriteSingleRegister {
            address: 0x0001,
            value: 0x0003,
        };
        let mut buf = [0u8; 256];
        let n = encode_request(&req, &mut buf).expect("encode failed");
        assert_eq!(n, 5);
        assert_eq!(buf[0], 0x06);
    }

    #[test]
    fn encode_write_multiple_registers() {
        let mut data: Vec<u8, 246> = Vec::new();
        let _ = data.extend_from_slice(&[0x00, 0x0A, 0x01, 0x02]); // two registers
        let req = Request::WriteMultipleRegisters {
            start: 0x0001,
            count: 0x0002,
            data,
        };
        let mut buf = [0u8; 256];
        let n = encode_request(&req, &mut buf).expect("encode failed");
        // FC(1)+addr(2)+count(2)+byte_cnt(1)+data(4) = 10
        assert_eq!(n, 10);
        assert_eq!(buf[0], 0x10);
        assert_eq!(buf[5], 4); // byte count
    }

    #[test]
    fn encode_buffer_too_small() {
        let req = Request::ReadCoils { start: 0, count: 1 };
        let mut buf = [0u8; 4]; // needs 5
        let result = encode_request(&req, &mut buf);
        assert_eq!(result, Err(ModbusError::IllegalDataValue));
    }

    // ── Decode tests ──────────────────────────────────────────────────────────

    #[test]
    fn decode_read_coils_response() {
        // FC01 response: byte_count=3, 3 bytes of coil data
        let raw = [0x01u8, 0x03, 0xCD, 0x6B, 0x05];
        let resp = decode_response(&raw).expect("decode failed");
        match resp {
            Response::ReadCoils { byte_count, data } => {
                assert_eq!(byte_count, 3);
                assert_eq!(data.as_slice(), &[0xCD, 0x6B, 0x05]);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn decode_read_holding_registers_response() {
        // FC03 response: byte_count=6, 3 registers = 0x022B, 0x0000, 0x0064
        let raw = [0x03u8, 0x06, 0x02, 0x2B, 0x00, 0x00, 0x00, 0x64];
        let resp = decode_response(&raw).expect("decode failed");
        match resp {
            Response::ReadHoldingRegisters { byte_count, data } => {
                assert_eq!(byte_count, 6);
                assert_eq!(u16::from_be_bytes([data[0], data[1]]), 0x022B);
                assert_eq!(u16::from_be_bytes([data[4], data[5]]), 0x0064);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn decode_write_single_coil_response() {
        let raw = [0x05u8, 0x00, 0xAC, 0xFF, 0x00];
        let resp = decode_response(&raw).expect("decode failed");
        match resp {
            Response::WriteSingleCoil { address, value } => {
                assert_eq!(address, 0x00AC);
                assert_eq!(value, 0xFF00);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn decode_exception_response() {
        // FC03 exception: illegal data address
        let raw = [0x83u8, 0x02];
        let resp = decode_response(&raw).expect("decode failed");
        match resp {
            Response::Exception {
                function_code,
                exception,
            } => {
                assert_eq!(function_code, 0x03);
                assert_eq!(exception, ExceptionCode::IllegalDataAddress);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn decode_write_multiple_registers_response() {
        let raw = [0x10u8, 0x00, 0x01, 0x00, 0x02];
        let resp = decode_response(&raw).expect("decode failed");
        match resp {
            Response::WriteMultipleRegisters { start, count } => {
                assert_eq!(start, 0x0001);
                assert_eq!(count, 0x0002);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn decode_empty_buffer_error() {
        assert_eq!(decode_response(&[]), Err(ModbusError::IllegalDataValue));
    }

    #[test]
    fn decode_unknown_fc_error() {
        let raw = [0x42u8];
        assert_eq!(decode_response(&raw), Err(ModbusError::IllegalFunction));
    }

    #[test]
    fn exception_code_roundtrip() {
        for (byte, expected) in [
            (0x01, ExceptionCode::IllegalFunction),
            (0x02, ExceptionCode::IllegalDataAddress),
            (0x03, ExceptionCode::IllegalDataValue),
            (0x04, ExceptionCode::SlaveDeviceFailure),
        ] {
            assert_eq!(ExceptionCode::from_u8(byte), Some(expected));
        }
        assert_eq!(ExceptionCode::from_u8(0xFF), None);
    }
}
