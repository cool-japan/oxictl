//! Modbus server (slave) — processes requests and generates responses.
//!
//! Dispatches function codes to the register map and builds response PDUs.

use super::register::{ModbusError, RegisterMap};

/// Modbus PDU (Protocol Data Unit): function code + data.
#[derive(Debug, Clone)]
pub struct ModbusPdu {
    pub function_code: u8,
    pub data: heapless::Vec<u8, 254>,
}

impl ModbusPdu {
    pub fn new(function_code: u8) -> Self {
        Self {
            function_code,
            data: heapless::Vec::new(),
        }
    }

    pub fn error_response(function_code: u8, error_code: ModbusError) -> Self {
        let mut pdu = Self::new(function_code | 0x80); // error: high bit set
        let _ = pdu.data.push(error_code as u8);
        pdu
    }
}

/// Modbus server dispatcher.
///
/// Processes an incoming request PDU and returns a response PDU.
pub struct ModbusServer<const COILS: usize, const DI: usize, const IR: usize, const HR: usize> {
    pub registers: RegisterMap<COILS, DI, IR, HR>,
    /// Server (slave) address.
    pub address: u8,
}

impl<const COILS: usize, const DI: usize, const IR: usize, const HR: usize>
    ModbusServer<COILS, DI, IR, HR>
{
    pub fn new(address: u8) -> Self {
        Self {
            registers: RegisterMap::new(),
            address,
        }
    }

    /// Process a request PDU, return response PDU.
    pub fn process(&mut self, req: &ModbusPdu) -> ModbusPdu {
        match req.function_code {
            0x01 => self.handle_read_coils(&req.data),
            0x02 => self.handle_read_discrete_inputs(&req.data),
            0x03 => self.handle_read_holding_registers(&req.data),
            0x04 => self.handle_read_input_registers(&req.data),
            0x05 => self.handle_write_coil(&req.data),
            0x06 => self.handle_write_register(&req.data),
            0x10 => self.handle_write_multiple_registers(&req.data),
            _ => ModbusPdu::error_response(req.function_code, ModbusError::IllegalFunction),
        }
    }

    fn parse_start_count(data: &[u8]) -> Option<(u16, u16)> {
        if data.len() < 4 {
            return None;
        }
        let start = u16::from_be_bytes([data[0], data[1]]);
        let count = u16::from_be_bytes([data[2], data[3]]);
        Some((start, count))
    }

    fn handle_read_coils(&self, data: &[u8]) -> ModbusPdu {
        let Some((start, count)) = Self::parse_start_count(data) else {
            return ModbusPdu::error_response(0x01, ModbusError::IllegalDataValue);
        };
        match self.registers.read_coils(start, count) {
            Err(e) => ModbusPdu::error_response(0x01, e),
            Ok(coils) => {
                let mut pdu = ModbusPdu::new(0x01);
                let byte_count = (count as usize).div_ceil(8);
                let _ = pdu.data.push(byte_count as u8);
                let mut byte = 0u8;
                for (i, &c) in coils.iter().enumerate() {
                    if c {
                        byte |= 1 << (i % 8);
                    }
                    if i % 8 == 7 || i == coils.len() - 1 {
                        let _ = pdu.data.push(byte);
                        byte = 0;
                    }
                }
                pdu
            }
        }
    }

    fn handle_read_discrete_inputs(&self, data: &[u8]) -> ModbusPdu {
        let Some((start, count)) = Self::parse_start_count(data) else {
            return ModbusPdu::error_response(0x02, ModbusError::IllegalDataValue);
        };
        match self.registers.read_discrete_inputs(start, count) {
            Err(e) => ModbusPdu::error_response(0x02, e),
            Ok(inputs) => {
                let mut pdu = ModbusPdu::new(0x02);
                let byte_count = (count as usize).div_ceil(8);
                let _ = pdu.data.push(byte_count as u8);
                let mut byte = 0u8;
                for (i, &inp) in inputs.iter().enumerate() {
                    if inp {
                        byte |= 1 << (i % 8);
                    }
                    if i % 8 == 7 || i == inputs.len() - 1 {
                        let _ = pdu.data.push(byte);
                        byte = 0;
                    }
                }
                pdu
            }
        }
    }

    fn handle_read_holding_registers(&self, data: &[u8]) -> ModbusPdu {
        let Some((start, count)) = Self::parse_start_count(data) else {
            return ModbusPdu::error_response(0x03, ModbusError::IllegalDataValue);
        };
        match self.registers.read_holding_registers(start, count) {
            Err(e) => ModbusPdu::error_response(0x03, e),
            Ok(regs) => {
                let mut pdu = ModbusPdu::new(0x03);
                let _ = pdu.data.push((regs.len() * 2) as u8);
                for &r in regs {
                    let _ = pdu.data.extend_from_slice(&r.to_be_bytes());
                }
                pdu
            }
        }
    }

    fn handle_read_input_registers(&self, data: &[u8]) -> ModbusPdu {
        let Some((start, count)) = Self::parse_start_count(data) else {
            return ModbusPdu::error_response(0x04, ModbusError::IllegalDataValue);
        };
        match self.registers.read_input_registers(start, count) {
            Err(e) => ModbusPdu::error_response(0x04, e),
            Ok(regs) => {
                let mut pdu = ModbusPdu::new(0x04);
                let _ = pdu.data.push((regs.len() * 2) as u8);
                for &r in regs {
                    let _ = pdu.data.extend_from_slice(&r.to_be_bytes());
                }
                pdu
            }
        }
    }

    fn handle_write_coil(&mut self, data: &[u8]) -> ModbusPdu {
        if data.len() < 4 {
            return ModbusPdu::error_response(0x05, ModbusError::IllegalDataValue);
        }
        let addr = u16::from_be_bytes([data[0], data[1]]);
        let val = u16::from_be_bytes([data[2], data[3]]);
        let coil_val = val == 0xFF00;
        match self.registers.write_coil(addr, coil_val) {
            Err(e) => ModbusPdu::error_response(0x05, e),
            Ok(()) => {
                let mut pdu = ModbusPdu::new(0x05);
                let _ = pdu.data.extend_from_slice(data);
                pdu
            }
        }
    }

    fn handle_write_register(&mut self, data: &[u8]) -> ModbusPdu {
        if data.len() < 4 {
            return ModbusPdu::error_response(0x06, ModbusError::IllegalDataValue);
        }
        let addr = u16::from_be_bytes([data[0], data[1]]);
        let val = u16::from_be_bytes([data[2], data[3]]);
        match self.registers.write_register(addr, val) {
            Err(e) => ModbusPdu::error_response(0x06, e),
            Ok(()) => {
                let mut pdu = ModbusPdu::new(0x06);
                let _ = pdu.data.extend_from_slice(data);
                pdu
            }
        }
    }

    fn handle_write_multiple_registers(&mut self, data: &[u8]) -> ModbusPdu {
        if data.len() < 5 {
            return ModbusPdu::error_response(0x10, ModbusError::IllegalDataValue);
        }
        let start = u16::from_be_bytes([data[0], data[1]]);
        let count = u16::from_be_bytes([data[2], data[3]]);
        let byte_count = data[4] as usize;
        if data.len() < 5 + byte_count {
            return ModbusPdu::error_response(0x10, ModbusError::IllegalDataValue);
        }
        let reg_bytes = &data[5..5 + byte_count];
        let regs: heapless::Vec<u16, 125> = reg_bytes
            .chunks(2)
            .filter_map(|c| {
                if c.len() == 2 {
                    Some(u16::from_be_bytes([c[0], c[1]]))
                } else {
                    None
                }
            })
            .collect();

        match self.registers.write_registers(start, &regs) {
            Err(e) => ModbusPdu::error_response(0x10, e),
            Ok(()) => {
                let mut pdu = ModbusPdu::new(0x10);
                let _ = pdu.data.extend_from_slice(&start.to_be_bytes());
                let _ = pdu.data.extend_from_slice(&count.to_be_bytes());
                pdu
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Server = ModbusServer<16, 8, 8, 32>;

    #[test]
    fn read_holding_registers() {
        let mut srv = Server::new(1);
        srv.registers.write_register(5, 0x1234).unwrap();

        let mut req = ModbusPdu::new(0x03);
        let _ = req.data.extend_from_slice(&[0x00, 0x05, 0x00, 0x01]); // start=5, count=1
        let resp = srv.process(&req);

        assert_eq!(resp.function_code, 0x03);
        assert_eq!(resp.data[0], 2); // byte count
        assert_eq!(u16::from_be_bytes([resp.data[1], resp.data[2]]), 0x1234);
    }

    #[test]
    fn write_single_register() {
        let mut srv = Server::new(1);
        let mut req = ModbusPdu::new(0x06);
        let _ = req.data.extend_from_slice(&[0x00, 0x0A, 0xAB, 0xCD]); // addr=10, val=0xABCD
        let resp = srv.process(&req);
        assert_eq!(resp.function_code, 0x06);
        assert_eq!(srv.registers.holding_registers[10], 0xABCD);
    }

    #[test]
    fn illegal_function_code() {
        let mut srv = Server::new(1);
        let req = ModbusPdu::new(0x99);
        let resp = srv.process(&req);
        assert_eq!(resp.function_code, 0x99 | 0x80);
        assert_eq!(resp.data[0], ModbusError::IllegalFunction as u8);
    }

    #[test]
    fn read_out_of_range_error() {
        let mut srv = Server::new(1);
        let mut req = ModbusPdu::new(0x03);
        let _ = req.data.extend_from_slice(&[0x00, 0x1E, 0x00, 0x0A]); // start=30, count=10 → >32
        let resp = srv.process(&req);
        assert_eq!(resp.function_code, 0x03 | 0x80);
    }
}
