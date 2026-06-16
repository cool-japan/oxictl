//! Unit tests for fixed-point arithmetic types and operations.

use super::convert::{fixed_from_f32_saturating, fixed_to_f32};
use super::types::{Q15_16, Q3_29, Q7_24};
use crate::core::scalar::PidScalar;

#[test]
fn zero_one_roundtrip() {
    assert_eq!(Q15_16::ZERO + Q15_16::ONE, Q15_16::ONE);
}

#[test]
fn from_int_positive() {
    let x = Q15_16::from_int(5);
    let y = Q15_16::from_int(3);
    let sum = x + y;
    let result = fixed_to_f32(sum);
    assert!((result - 8.0_f32).abs() < 0.001, "got {}", result);
}

#[test]
fn saturating_overflow() {
    use fixed::types::I16F16;
    let max = Q15_16(I16F16::MAX);
    let one = Q15_16::ONE;
    let sat = max + one;
    assert_eq!(sat, Q15_16(I16F16::MAX), "overflow should saturate");
}

#[test]
fn saturating_underflow() {
    use fixed::types::I16F16;
    let min = Q15_16(I16F16::MIN);
    let one = Q15_16::ONE;
    let sat = min - one;
    assert_eq!(sat, Q15_16(I16F16::MIN), "underflow should saturate");
}

#[test]
fn div_result_close_to_expected() {
    let four = Q15_16::from_int(4);
    let two = Q15_16::from_int(2);
    let result = fixed_to_f32(four / two);
    assert!(
        (result - 2.0_f32).abs() < 0.001,
        "4/2 should be ~2, got {}",
        result
    );
}

#[test]
fn div_by_zero_saturates() {
    use fixed::types::I16F16;
    let four = Q15_16::from_int(4);
    let zero = Q15_16::ZERO;
    let result = four / zero;
    // checked_div(0) returns None; fallback: rhs.0 == ZERO is not > ZERO → MIN
    assert!(
        result == Q15_16(I16F16::MAX) || result == Q15_16(I16F16::MIN),
        "div by zero should saturate, got {:?}",
        result
    );
}

#[test]
fn abs_positive_unchanged() {
    let x = Q15_16::from_int(3);
    assert_eq!(x.abs(), x);
}

#[test]
fn abs_negative() {
    let neg = -Q15_16::from_int(3);
    let pos = Q15_16::from_int(3);
    assert_eq!(neg.abs(), pos);
}

#[test]
fn epsilon_is_positive() {
    assert!(Q15_16::EPSILON > Q15_16::ZERO);
}

#[test]
fn roundtrip_f32_q15_16() {
    // Use a value that's not a known constant to avoid clippy::approx_constant
    let val = 3.0_f32 + 0.14159_f32;
    let fixed = fixed_from_f32_saturating(val);
    let back = fixed_to_f32(fixed);
    // Q15.16 has ~0.000015 resolution; this value should roundtrip within 0.001
    assert!(
        (back - val).abs() < 0.001,
        "roundtrip: {} -> {} -> {}",
        val,
        fixed.0,
        back
    );
}

#[test]
fn from_int_all_formats() {
    let _ = Q3_29::from_int(0);
    let _ = Q3_29::from_int(2);
    // Q7_24 (I8F24) range: [-128, 128), use value well within range
    let _ = Q7_24::from_int(10);
}
