//! Fixed-size register and coil banks for Modbus slave devices.
//!
//! `RegisterBank<N>` holds N u16 holding registers.
//! `CoilBank<N, BYTES>` holds N boolean coils bit-packed into a `[u8; BYTES]` array.
//!
//! # Const-generic sizing for `CoilBank`
//!
//! Stable Rust does not support `[u8; (N+7)/8]` in the type signature because
//! generic const expressions are unstable.  Therefore `CoilBank` requires two
//! const parameters: the number of coils (`N`) and the backing byte-array size
//! (`BYTES`, which must be `≥ (N+7)/8`).  A runtime check guards against
//! under-sized `BYTES`.
//!
//! Both types are `no_std` compatible and heap-free.

use super::register::ModbusError;

// ─── RegisterBank ─────────────────────────────────────────────────────────────

/// A fixed-size bank of `N` 16-bit holding registers.
///
/// Registers are zero-initialised at construction.
#[derive(Debug)]
pub struct RegisterBank<const N: usize> {
    data: [u16; N],
}

impl<const N: usize> RegisterBank<N> {
    /// Construct a zeroed register bank.
    pub const fn new() -> Self {
        Self { data: [0u16; N] }
    }

    /// Read a contiguous slice of registers `[start, start+count)`.
    ///
    /// Returns `IllegalDataAddress` if the range exceeds the bank.
    pub fn read_registers(&self, start: u16, count: u16) -> Result<&[u16], ModbusError> {
        let s = start as usize;
        let c = count as usize;
        let end = s.checked_add(c).ok_or(ModbusError::IllegalDataAddress)?;
        if end > N {
            return Err(ModbusError::IllegalDataAddress);
        }
        Ok(&self.data[s..end])
    }

    /// Write a single register at `addr`.
    ///
    /// Returns `IllegalDataAddress` if `addr >= N`.
    pub fn write_register(&mut self, addr: u16, value: u16) -> Result<(), ModbusError> {
        if addr as usize >= N {
            return Err(ModbusError::IllegalDataAddress);
        }
        self.data[addr as usize] = value;
        Ok(())
    }

    /// Write a contiguous block of registers starting at `start`.
    ///
    /// An empty `values` slice is a no-op.  Returns `IllegalDataAddress`
    /// if `start as usize + values.len() > N`.
    pub fn write_registers(&mut self, start: u16, values: &[u16]) -> Result<(), ModbusError> {
        let s = start as usize;
        let end = s
            .checked_add(values.len())
            .ok_or(ModbusError::IllegalDataAddress)?;
        if end > N {
            return Err(ModbusError::IllegalDataAddress);
        }
        self.data[s..end].copy_from_slice(values);
        Ok(())
    }

    /// Read a single register.
    pub fn get(&self, addr: u16) -> Result<u16, ModbusError> {
        if addr as usize >= N {
            return Err(ModbusError::IllegalDataAddress);
        }
        Ok(self.data[addr as usize])
    }

    /// Return the full data slice (all N registers).
    pub fn as_slice(&self) -> &[u16] {
        &self.data
    }
}

impl<const N: usize> Default for RegisterBank<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── CoilBank ─────────────────────────────────────────────────────────────────

/// Compute the minimum number of backing bytes required for `n` coils.
/// Used to document the relationship: `BYTES >= (N + 7) / 8`.
pub const fn min_coil_bytes(n: usize) -> usize {
    n.div_ceil(8)
}

/// A fixed-size bank of `N` boolean coils, bit-packed into a `[u8; BYTES]` array.
///
/// `BYTES` must satisfy `BYTES >= (N + 7) / 8`.  The `new()` constructor
/// enforces this at runtime (panics if violated in debug, returns garbage-free
/// zero-initialised data in release).
///
/// Bit layout per byte: bit 0 (LSB) = lowest-index coil. This matches the
/// Modbus wire format for FC01/FC05.
///
/// # Sizing guidance
///
/// ```text
/// CoilBank<16, 2>  // 16 coils → 2 bytes
/// CoilBank<17, 3>  // 17 coils → ceil(17/8) = 3 bytes
/// CoilBank<8,  1>  // 8  coils → 1 byte
/// ```
#[derive(Debug)]
pub struct CoilBank<const N: usize, const BYTES: usize> {
    bits: [u8; BYTES],
}

impl<const N: usize, const BYTES: usize> CoilBank<N, BYTES> {
    /// Construct a bank with all coils cleared (false).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `BYTES < (N + 7) / 8`.
    pub const fn new() -> Self {
        // Compile-time assertion not directly possible without nightly features,
        // but we catch it at construction time.
        debug_assert!(BYTES >= min_coil_bytes(N), "BYTES must be >= (N+7)/8");
        Self { bits: [0u8; BYTES] }
    }

    /// Validate that BYTES is large enough for N coils.
    fn check_sizing() -> Result<(), ModbusError> {
        if BYTES < min_coil_bytes(N) {
            Err(ModbusError::IllegalDataValue)
        } else {
            Ok(())
        }
    }

    /// Read a single coil.
    ///
    /// Returns `IllegalDataAddress` if `addr >= N`.
    pub fn read_coil(&self, addr: u16) -> Result<bool, ModbusError> {
        let idx = addr as usize;
        if idx >= N {
            return Err(ModbusError::IllegalDataAddress);
        }
        let byte = self.bits[idx / 8];
        Ok((byte >> (idx % 8)) & 0x01 != 0)
    }

    /// Write a single coil.
    ///
    /// Returns `IllegalDataAddress` if `addr >= N`.
    pub fn write_coil(&mut self, addr: u16, value: bool) -> Result<(), ModbusError> {
        let idx = addr as usize;
        if idx >= N {
            return Err(ModbusError::IllegalDataAddress);
        }
        let bit = 1u8 << (idx % 8);
        if value {
            self.bits[idx / 8] |= bit;
        } else {
            self.bits[idx / 8] &= !bit;
        }
        Ok(())
    }

    /// Read `count` coils starting at `start`, bit-packed into `buf`.
    ///
    /// Bit layout: first coil → LSB of buf[0], coil N+1 → bit 1 of buf[0], etc.
    /// Trailing bits in the last byte are zeroed.
    ///
    /// Returns the number of data bytes written into `buf`.
    ///
    /// # Errors
    ///
    /// - `IllegalDataAddress` — range `[start, start+count)` exceeds the bank.
    /// - `IllegalDataValue` — `buf` is too small to hold `ceil(count/8)` bytes.
    pub fn read_coils(&self, start: u16, count: u16, buf: &mut [u8]) -> Result<u8, ModbusError> {
        Self::check_sizing()?;
        let s = start as usize;
        let c = count as usize;
        let end = s.checked_add(c).ok_or(ModbusError::IllegalDataAddress)?;
        if end > N {
            return Err(ModbusError::IllegalDataAddress);
        }
        let byte_count = c.div_ceil(8);
        if buf.len() < byte_count {
            return Err(ModbusError::IllegalDataValue);
        }
        // Zero the output bytes (trailing bits must be 0 per spec).
        buf[..byte_count].iter_mut().for_each(|b| *b = 0);
        for i in 0..c {
            let src_idx = s + i;
            let src_byte = self.bits[src_idx / 8];
            let src_bit = (src_byte >> (src_idx % 8)) & 0x01;
            buf[i / 8] |= src_bit << (i % 8);
        }
        Ok(byte_count as u8)
    }

    /// Write `count` coils starting at `start` from bit-packed `data`.
    ///
    /// `data` must contain at least `ceil(count/8)` bytes.
    ///
    /// # Errors
    ///
    /// - `IllegalDataAddress` — range exceeds the bank.
    /// - `IllegalDataValue` — `data` is too short.
    pub fn write_coils(&mut self, start: u16, count: u16, data: &[u8]) -> Result<(), ModbusError> {
        Self::check_sizing()?;
        let s = start as usize;
        let c = count as usize;
        let end = s.checked_add(c).ok_or(ModbusError::IllegalDataAddress)?;
        if end > N {
            return Err(ModbusError::IllegalDataAddress);
        }
        let byte_count = c.div_ceil(8);
        if data.len() < byte_count {
            return Err(ModbusError::IllegalDataValue);
        }
        for i in 0..c {
            let bit = (data[i / 8] >> (i % 8)) & 0x01;
            let dst_idx = s + i;
            let mask = 1u8 << (dst_idx % 8);
            if bit != 0 {
                self.bits[dst_idx / 8] |= mask;
            } else {
                self.bits[dst_idx / 8] &= !mask;
            }
        }
        Ok(())
    }

    /// Return the raw packed bits as a byte slice.
    pub fn as_raw_bytes(&self) -> &[u8] {
        &self.bits
    }
}

impl<const N: usize, const BYTES: usize> Default for CoilBank<N, BYTES> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RegisterBank tests ────────────────────────────────────────────────────

    #[test]
    fn register_bank_read_write_single() {
        let mut bank: RegisterBank<32> = RegisterBank::new();
        bank.write_register(10, 0xABCD).expect("write failed");
        assert_eq!(bank.get(10).expect("get failed"), 0xABCD);
    }

    #[test]
    fn register_bank_read_slice() {
        let mut bank: RegisterBank<16> = RegisterBank::new();
        bank.write_registers(2, &[1, 2, 3, 4])
            .expect("bulk write failed");
        let slice = bank.read_registers(2, 4).expect("read failed");
        assert_eq!(slice, &[1u16, 2, 3, 4]);
    }

    #[test]
    fn register_bank_out_of_bounds_read() {
        let bank: RegisterBank<8> = RegisterBank::new();
        assert_eq!(
            bank.read_registers(6, 4),
            Err(ModbusError::IllegalDataAddress)
        );
    }

    #[test]
    fn register_bank_out_of_bounds_write() {
        let mut bank: RegisterBank<8> = RegisterBank::new();
        assert_eq!(
            bank.write_register(8, 0xFFFF),
            Err(ModbusError::IllegalDataAddress)
        );
    }

    #[test]
    fn register_bank_bulk_write_out_of_bounds() {
        let mut bank: RegisterBank<4> = RegisterBank::new();
        assert_eq!(
            bank.write_registers(3, &[1, 2]),
            Err(ModbusError::IllegalDataAddress)
        );
    }

    #[test]
    fn register_bank_zero_count_read() {
        let bank: RegisterBank<8> = RegisterBank::new();
        let slice = bank.read_registers(0, 0).expect("zero-count read failed");
        assert_eq!(slice.len(), 0);
    }

    #[test]
    fn register_bank_as_slice_length() {
        let bank: RegisterBank<10> = RegisterBank::new();
        assert_eq!(bank.as_slice().len(), 10);
    }

    #[test]
    fn register_bank_write_registers_empty_slice() {
        let mut bank: RegisterBank<8> = RegisterBank::new();
        bank.write_registers(0, &[]).expect("empty write");
        assert_eq!(bank.get(0).expect("get"), 0);
    }

    // ── CoilBank tests ────────────────────────────────────────────────────────

    // CoilBank<16, 2>: 16 coils → 2 bytes
    type Bank16 = CoilBank<16, 2>;

    #[test]
    fn coil_bank_read_write_single() {
        let mut bank = Bank16::new();
        bank.write_coil(7, true).expect("write failed");
        assert!(bank.read_coil(7).expect("read failed"));
        assert!(!bank.read_coil(6).expect("read adjacent"));
    }

    #[test]
    fn coil_bank_toggle() {
        let mut bank: CoilBank<8, 1> = CoilBank::new();
        bank.write_coil(3, true).expect("set");
        assert!(bank.read_coil(3).expect("check"));
        bank.write_coil(3, false).expect("clear");
        assert!(!bank.read_coil(3).expect("check cleared"));
    }

    #[test]
    fn coil_bank_out_of_bounds() {
        let mut bank = Bank16::new();
        assert_eq!(
            bank.write_coil(16, true),
            Err(ModbusError::IllegalDataAddress)
        );
        assert_eq!(bank.read_coil(16), Err(ModbusError::IllegalDataAddress));
    }

    #[test]
    fn coil_bank_read_coils_packed() {
        let mut bank = Bank16::new();
        // Set coils 0, 2, 4 → bits 0,2,4 of byte 0 → 0b0001_0101 = 0x15
        bank.write_coil(0, true).expect("0");
        bank.write_coil(2, true).expect("2");
        bank.write_coil(4, true).expect("4");

        let mut buf = [0u8; 2];
        let byte_count = bank.read_coils(0, 8, &mut buf).expect("read_coils");
        assert_eq!(byte_count, 1);
        assert_eq!(buf[0], 0b0001_0101);
    }

    #[test]
    fn coil_bank_write_coils_packed() {
        let mut bank = Bank16::new();
        // Pack coils 0,3,7 → byte = 0b1000_1001 = 0x89
        let data = [0x89u8, 0x00];
        bank.write_coils(0, 8, &data).expect("write_coils");
        assert!(bank.read_coil(0).expect("c0"));
        assert!(!bank.read_coil(1).expect("c1"));
        assert!(!bank.read_coil(2).expect("c2"));
        assert!(bank.read_coil(3).expect("c3"));
        assert!(!bank.read_coil(4).expect("c4"));
        assert!(!bank.read_coil(5).expect("c5"));
        assert!(!bank.read_coil(6).expect("c6"));
        assert!(bank.read_coil(7).expect("c7"));
    }

    #[test]
    fn coil_bank_read_coils_cross_byte_boundary() {
        let mut bank = Bank16::new();
        bank.write_coil(6, true).expect("set bit 6");
        bank.write_coil(9, true).expect("set bit 9");

        let mut buf = [0u8; 4];
        // Read 6 coils starting at index 5: coils 5,6,7,8,9,10
        // coil 6 → local index 1 → buf[0] bit 1 = 0x02
        // coil 9 → local index 4 → buf[0] bit 4 = 0x10
        let byte_count = bank.read_coils(5, 6, &mut buf).expect("read");
        assert_eq!(byte_count, 1);
        assert_ne!(buf[0] & (1 << 1), 0, "coil 6 not set");
        assert_ne!(buf[0] & (1 << 4), 0, "coil 9 not set");
    }

    #[test]
    fn coil_bank_read_coils_out_of_bounds() {
        let bank: CoilBank<8, 1> = CoilBank::new();
        let mut buf = [0u8; 4];
        assert_eq!(
            bank.read_coils(6, 4, &mut buf),
            Err(ModbusError::IllegalDataAddress)
        );
    }

    #[test]
    fn coil_bank_read_coils_buf_too_small() {
        let bank = Bank16::new();
        let mut buf = [0u8; 0]; // needs at least 1 byte for 8 coils
        assert_eq!(
            bank.read_coils(0, 8, &mut buf),
            Err(ModbusError::IllegalDataValue)
        );
    }

    #[test]
    fn coil_bank_raw_bytes_size() {
        // CoilBank<17, 3>: needs ceil(17/8)=3 bytes
        let bank: CoilBank<17, 3> = CoilBank::new();
        assert_eq!(bank.as_raw_bytes().len(), 3);
    }

    #[test]
    fn coil_bank_write_coils_roundtrip() {
        let mut bank: CoilBank<16, 2> = CoilBank::new();
        // Write 16 coils as two bytes
        let data = [0xA5u8, 0x3C]; // arbitrary bit pattern
        bank.write_coils(0, 16, &data).expect("write");
        let mut readback = [0u8; 2];
        bank.read_coils(0, 16, &mut readback).expect("read");
        assert_eq!(readback, data);
    }

    #[test]
    fn coil_bank_write_coils_partial() {
        let mut bank: CoilBank<16, 2> = CoilBank::new();
        let data = [0xFFu8]; // 8 coils all set
        bank.write_coils(8, 8, &data).expect("write upper byte");
        // Lower 8 coils should still be clear
        let mut lower = [0u8; 1];
        bank.read_coils(0, 8, &mut lower).expect("read lower");
        assert_eq!(lower[0], 0x00);
        // Upper 8 coils should all be set
        let mut upper = [0u8; 1];
        bank.read_coils(8, 8, &mut upper).expect("read upper");
        assert_eq!(upper[0], 0xFF);
    }
}
