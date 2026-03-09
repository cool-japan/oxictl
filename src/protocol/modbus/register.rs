//! Modbus register map definition and access.
//!
//! Modbus defines four register/coil spaces:
//!   - Coils (0x): 1-bit read-write digital outputs
//!   - Discrete inputs (1x): 1-bit read-only digital inputs
//!   - Input registers (3x): 16-bit read-only analog inputs
//!   - Holding registers (4x): 16-bit read-write

/// Modbus register type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegType {
    Coil,
    DiscreteInput,
    InputRegister,
    HoldingRegister,
}

/// Result type for register access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModbusError {
    IllegalFunction = 0x01,
    IllegalDataAddress = 0x02,
    IllegalDataValue = 0x03,
    SlaveDeviceFailure = 0x04,
}

pub type ModbusResult<T> = Result<T, ModbusError>;

/// Register map for a Modbus server (slave device).
///
/// Fixed-size arrays for each register space.
#[derive(Debug)]
pub struct RegisterMap<const COILS: usize, const DI: usize, const IR: usize, const HR: usize> {
    /// Coils (read-write bits).
    pub coils: [bool; COILS],
    /// Discrete inputs (read-only bits).
    pub discrete_inputs: [bool; DI],
    /// Input registers (read-only 16-bit).
    pub input_registers: [u16; IR],
    /// Holding registers (read-write 16-bit).
    pub holding_registers: [u16; HR],
}

impl<const COILS: usize, const DI: usize, const IR: usize, const HR: usize>
    RegisterMap<COILS, DI, IR, HR>
{
    pub fn new() -> Self {
        Self {
            coils: [false; COILS],
            discrete_inputs: [false; DI],
            input_registers: [0u16; IR],
            holding_registers: [0u16; HR],
        }
    }

    /// FC01: Read Coils
    pub fn read_coils(&self, start: u16, count: u16) -> ModbusResult<&[bool]> {
        let s = start as usize;
        let e = s + count as usize;
        if e > COILS {
            return Err(ModbusError::IllegalDataAddress);
        }
        Ok(&self.coils[s..e])
    }

    /// FC02: Read Discrete Inputs
    pub fn read_discrete_inputs(&self, start: u16, count: u16) -> ModbusResult<&[bool]> {
        let s = start as usize;
        let e = s + count as usize;
        if e > DI {
            return Err(ModbusError::IllegalDataAddress);
        }
        Ok(&self.discrete_inputs[s..e])
    }

    /// FC03: Read Holding Registers
    pub fn read_holding_registers(&self, start: u16, count: u16) -> ModbusResult<&[u16]> {
        let s = start as usize;
        let e = s + count as usize;
        if e > HR {
            return Err(ModbusError::IllegalDataAddress);
        }
        Ok(&self.holding_registers[s..e])
    }

    /// FC04: Read Input Registers
    pub fn read_input_registers(&self, start: u16, count: u16) -> ModbusResult<&[u16]> {
        let s = start as usize;
        let e = s + count as usize;
        if e > IR {
            return Err(ModbusError::IllegalDataAddress);
        }
        Ok(&self.input_registers[s..e])
    }

    /// FC05: Write Single Coil
    pub fn write_coil(&mut self, address: u16, value: bool) -> ModbusResult<()> {
        if address as usize >= COILS {
            return Err(ModbusError::IllegalDataAddress);
        }
        self.coils[address as usize] = value;
        Ok(())
    }

    /// FC06: Write Single Register
    pub fn write_register(&mut self, address: u16, value: u16) -> ModbusResult<()> {
        if address as usize >= HR {
            return Err(ModbusError::IllegalDataAddress);
        }
        self.holding_registers[address as usize] = value;
        Ok(())
    }

    /// FC15: Write Multiple Coils
    pub fn write_coils(&mut self, start: u16, values: &[bool]) -> ModbusResult<()> {
        let s = start as usize;
        let e = s + values.len();
        if e > COILS {
            return Err(ModbusError::IllegalDataAddress);
        }
        self.coils[s..e].copy_from_slice(values);
        Ok(())
    }

    /// FC16: Write Multiple Registers
    pub fn write_registers(&mut self, start: u16, values: &[u16]) -> ModbusResult<()> {
        let s = start as usize;
        let e = s + values.len();
        if e > HR {
            return Err(ModbusError::IllegalDataAddress);
        }
        self.holding_registers[s..e].copy_from_slice(values);
        Ok(())
    }
}

impl<const COILS: usize, const DI: usize, const IR: usize, const HR: usize> Default
    for RegisterMap<COILS, DI, IR, HR>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Map = RegisterMap<16, 8, 8, 32>;

    #[test]
    fn read_write_coil() {
        let mut map = Map::new();
        map.write_coil(3, true).unwrap();
        assert_eq!(map.read_coils(3, 1).unwrap(), &[true]);
    }

    #[test]
    fn read_write_holding_register() {
        let mut map = Map::new();
        map.write_register(10, 0x1234).unwrap();
        let regs = map.read_holding_registers(10, 1).unwrap();
        assert_eq!(regs[0], 0x1234);
    }

    #[test]
    fn out_of_range_returns_error() {
        let map = Map::new();
        assert_eq!(
            map.read_holding_registers(30, 5),
            Err(ModbusError::IllegalDataAddress)
        );
    }

    #[test]
    fn write_multiple_registers() {
        let mut map = Map::new();
        map.write_registers(0, &[1, 2, 3, 4]).unwrap();
        let r = map.read_holding_registers(0, 4).unwrap();
        assert_eq!(r, &[1, 2, 3, 4]);
    }
}
