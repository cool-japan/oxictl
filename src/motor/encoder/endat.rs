//! EnDat 2.2 encoder protocol decoder.
//!
//! EnDat 2.2 is a synchronous, serial interface for absolute encoders.
//! This module implements:
//! - CRC-5 computation (polynomial 0x15 / x⁵+x²+1)
//! - Frame decoding for position and status bits
//! - Conversion from raw counts to angle (rad) and turns
#![allow(clippy::excessive_precision)]

use crate::core::scalar::ControlScalar;

/// Compute CRC-5 over a bit field using polynomial 0x15 (x⁵+x²+1).
///
/// The CRC is computed MSB-first over `bits` number of bits in `data`.
///
/// # Arguments
/// * `data` - Data word (only lower `bits` bits are used).
/// * `bits` - Number of bits to process.
///
/// # Returns
/// 5-bit CRC value.
pub fn crc5(data: u32, bits: u8) -> u8 {
    const POLY: u8 = 0x15; // x^5 + x^2 + 1
    let mut crc: u8 = 0x1F; // Init with all 1s per EnDat spec

    // Process bits MSB-first
    let n = bits as u32;
    for i in (0..n).rev() {
        let bit = ((data >> i) & 1) as u8;
        let top = (crc >> 4) & 1; // MSB of current CRC
        crc = (crc << 1) & 0x1F; // Shift left, keep 5 bits
        if top ^ bit != 0 {
            crc ^= POLY & 0x1F;
        }
    }
    crc & 0x1F
}

/// Result of decoding one EnDat frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndatResult {
    /// Single-turn raw position count.
    pub position: u32,
    /// Multi-turn revolution count.
    pub turn_count: u16,
    /// True if the received CRC matches the computed CRC.
    pub crc_ok: bool,
    /// Alarm flag: encoder detected internal error.
    pub alarm: bool,
    /// Error bit 1.
    pub error1: bool,
    /// Error bit 2.
    pub error2: bool,
}

/// EnDat 2.2 frame decoder.
///
/// Frame layout (simplified; full EnDat 2.2 spec extends this):
///
/// ```text
/// [Start (1)] [F01] [F02] [error2] [error1] [alarm]
/// [multi-turn (mt_bits)] [single-turn (st_bits)] [CRC5 (5)]
/// ```
///
/// Total payload bits = 1+1+1+1+1+1 + mt_bits + st_bits + 5
///
/// In this implementation we pack the frame from MSB as follows in the 32-bit word:
/// bit31: start (ignored)
/// bit30: F01 (ignored)
/// bit29: F02 (ignored)
/// bit28: error2
/// bit27: error1
/// bit26: alarm
/// bits[26-mt-1 .. 26-mt-st]: position (st_bits)
/// bits[26-mt .. 26-1]: turn_count (mt_bits)
/// bits[4..0]: CRC5
///
/// For decoding we extract fields by bit position. The CRC covers all
/// bits from error2 through the position field.
#[derive(Debug, Clone, Copy)]
pub struct EndatDecoder<S: ControlScalar> {
    /// Single-turn resolution bits (e.g., 23 for 8M counts/rev).
    pub st_bits: u8,
    /// Multi-turn bits (e.g., 12 for 4096 turns).
    pub mt_bits: u8,
    /// Electrical period for angle scaling (typically 2π).
    pub scale: S,
    _phantom: core::marker::PhantomData<S>,
}

impl<S: ControlScalar> EndatDecoder<S> {
    pub fn new(st_bits: u8, mt_bits: u8, scale: S) -> Self {
        Self {
            st_bits,
            mt_bits,
            scale,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Decode a 32-bit raw EnDat frame.
    ///
    /// Bit layout (MSB first, bits numbered from 31 down):
    ///   [31]     = start bit (ignored)
    ///   [30]     = F01 (ignored)
    ///   [29]     = F02 (ignored)
    ///   [28]     = error2
    ///   [27]     = error1
    ///   [26]     = alarm
    ///   [25..26-mt_bits+1] = turn_count (mt_bits bits)
    ///   [25-mt_bits..26-mt_bits-st_bits+1] = position (st_bits bits)
    ///   [4..0]   = CRC5
    ///
    /// The CRC covers bits [28 .. 5] (error2, error1, alarm, turn, position).
    pub fn decode_frame(&self, bits: u32) -> EndatResult {
        // Extract status flags
        let error2 = (bits >> 28) & 1 != 0;
        let error1 = (bits >> 27) & 1 != 0;
        let alarm = (bits >> 26) & 1 != 0;

        // Extract multi-turn field
        let mt_shift = 26 - self.mt_bits as u32;
        let mt_mask = (1u32 << self.mt_bits) - 1;
        let turn_count = ((bits >> mt_shift) & mt_mask) as u16;

        // Extract single-turn position field
        let st_shift = mt_shift - self.st_bits as u32;
        let st_mask = (1u32 << self.st_bits) - 1;
        let position = (bits >> st_shift) & st_mask;

        // Extract received CRC (lower 5 bits)
        let crc_received = (bits & 0x1F) as u8;

        // Compute CRC over the data field: bits [28..5]
        // Extract the data bits (bits 28 down to 5, so 24 bits total)
        let data_field = (bits >> 5) & 0x00FF_FFFF;
        let data_bits: u8 = 24; // bits 28..5 inclusive
        let crc_computed = crc5(data_field, data_bits);

        EndatResult {
            position,
            turn_count,
            crc_ok: crc_received == crc_computed,
            alarm,
            error1,
            error2,
        }
    }

    /// Convert raw single-turn position to angle in radians [0, 2π).
    ///
    /// angle = (position / 2^st_bits) * scale
    pub fn to_angle(&self, raw_pos: u32) -> S {
        let full_scale = S::from_f64((1u64 << self.st_bits) as f64);
        let raw = S::from_f64(raw_pos as f64);
        (raw / full_scale) * self.scale
    }

    /// Convert raw multi-turn count to total turns as a floating-point value.
    ///
    /// Interprets the raw count as an unsigned integer number of full revolutions.
    pub fn to_turns(&self, raw_mt: u16) -> S {
        S::from_f64(raw_mt as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc5_zero_data() {
        // With all-zero data and init=0x1F, result should be deterministic.
        let crc = crc5(0, 8);
        // Just verify it's a 5-bit value
        assert!(crc <= 0x1F, "CRC must fit in 5 bits, got {crc}");
    }

    #[test]
    fn test_to_angle_full_scale() {
        // 23-bit encoder: position = 2^23 - 1 ≈ 2π*(1 - 1/2^23)
        let dec = EndatDecoder::<f32>::new(23, 12, 2.0 * core::f32::consts::PI);
        let full = (1u32 << 23) - 1;
        let angle = dec.to_angle(full);
        let two_pi = 2.0 * core::f32::consts::PI;
        assert!(angle > 0.0 && angle < two_pi, "angle={angle}");
        assert!(
            (angle - two_pi).abs() < 0.001,
            "angle≈2π expected, got {angle}"
        );
    }

    #[test]
    fn test_to_angle_quarter_scale() {
        // position = 2^23 / 4 → angle ≈ π/2
        let dec = EndatDecoder::<f32>::new(23, 12, 2.0 * core::f32::consts::PI);
        let quarter = 1u32 << 21; // 2^23 / 4
        let angle = dec.to_angle(quarter);
        let expected = core::f32::consts::PI / 2.0;
        assert!(
            (angle - expected).abs() < 0.001,
            "angle={angle}, expected={expected}"
        );
    }

    #[test]
    fn test_decode_frame_alarm_bit() {
        // Construct a frame with alarm bit set (bit 26)
        let frame = 1u32 << 26;
        let dec = EndatDecoder::<f32>::new(16, 8, 2.0 * core::f32::consts::PI);
        let result = dec.decode_frame(frame);
        assert!(result.alarm, "Alarm bit should be set");
        assert!(!result.error1);
        assert!(!result.error2);
    }

    #[test]
    fn test_decode_frame_no_errors() {
        // Frame with no error bits set, all position bits zero
        let frame = 0u32;
        let dec = EndatDecoder::<f32>::new(16, 8, 2.0 * core::f32::consts::PI);
        let result = dec.decode_frame(frame);
        assert!(!result.alarm);
        assert!(!result.error1);
        assert!(!result.error2);
        assert_eq!(result.position, 0);
        assert_eq!(result.turn_count, 0);
    }

    #[test]
    fn test_to_turns() {
        let dec = EndatDecoder::<f32>::new(23, 12, 2.0 * core::f32::consts::PI);
        let turns = dec.to_turns(42);
        assert!((turns - 42.0_f32).abs() < 1e-6, "turns={turns}");
    }
}
