//! Property-based tests verifying Q15_16 fixed-point arithmetic properties.

use oxictl::core::fixed_point::convert::{fixed_from_f32_saturating, fixed_to_f32};
use proptest::prelude::*;

proptest! {
    #[test]
    fn add_sub_roundtrip_near_zero(
        a_f32 in -10.0_f32..10.0_f32,
        b_f32 in -10.0_f32..10.0_f32,
    ) {
        let a = fixed_from_f32_saturating(a_f32);
        let b = fixed_from_f32_saturating(b_f32);
        // (a + b) - b should equal a (within 2 ULPs of Q15.16)
        let result = (a + b) - b;
        let diff = fixed_to_f32(result) - fixed_to_f32(a);
        // Q15.16 ULP = 2^(-16) ≈ 0.0000153; allow 2 ULP
        prop_assert!(diff.abs() < 0.0001,
            "roundtrip failed: a={}, b={}, diff={}", a_f32, b_f32, diff);
    }

    #[test]
    fn mul_commutativity(
        a_f32 in -1.0_f32..1.0_f32,
        b_f32 in -1.0_f32..1.0_f32,
    ) {
        let a = fixed_from_f32_saturating(a_f32);
        let b = fixed_from_f32_saturating(b_f32);
        let ab = fixed_to_f32(a * b);
        let ba = fixed_to_f32(b * a);
        prop_assert!((ab - ba).abs() < 0.0001,
            "a*b != b*a: {}*{}=({},{})", a_f32, b_f32, ab, ba);
    }
}
