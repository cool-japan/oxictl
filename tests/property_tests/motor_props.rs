//! Property-based tests for motor transform invariants.

use oxictl::motor::transform::{clarke, clarke_inverse, park, park_inverse, svpwm, AlphaBeta};
use proptest::prelude::*;

proptest! {
    /// Clarke-Park-InversePark round-trip preserves αβ magnitude.
    #[test]
    fn clarke_park_roundtrip(
        ia in -10.0f64..10.0,
        ib in -10.0f64..10.0,
        theta in -core::f64::consts::PI..core::f64::consts::PI,
    ) {
        let ic = -ia - ib;
        let ab = clarke(ia, ib, ic);
        let dq = park(&ab, theta);
        let ab2 = park_inverse(&dq, theta);
        prop_assert!((ab2.alpha - ab.alpha).abs() < 1e-10,
            "alpha: {} vs {}", ab2.alpha, ab.alpha);
        prop_assert!((ab2.beta - ab.beta).abs() < 1e-10,
            "beta: {} vs {}", ab2.beta, ab.beta);
    }

    /// Clarke inverse recovers original balanced 3-phase currents.
    #[test]
    fn clarke_inverse_is_left_inverse(
        ia in -10.0f64..10.0,
        ib in -10.0f64..10.0,
    ) {
        let ic = -ia - ib;
        let ab = clarke(ia, ib, ic);
        let (ra, rb, rc) = clarke_inverse(&ab);
        prop_assert!((ra - ia).abs() < 1e-10, "ia: {ra} vs {ia}");
        prop_assert!((rb - ib).abs() < 1e-10, "ib: {rb} vs {ib}");
        prop_assert!((rc - ic).abs() < 1e-10, "ic: {rc} vs {ic}");
    }

    /// Amplitude-invariant Clarke: alpha^2+beta^2 = (2/3)*(a^2+b^2+c^2) for balanced 3-phase.
    #[test]
    fn clarke_power_preservation(
        ia in -10.0f64..10.0,
        ib in -10.0f64..10.0,
    ) {
        let ic = -ia - ib;
        let ab = clarke(ia, ib, ic);
        let power_abc = (2.0 / 3.0) * (ia * ia + ib * ib + ic * ic);
        let power_ab = ab.alpha * ab.alpha + ab.beta * ab.beta;
        prop_assert!((power_ab - power_abc).abs() < 1e-9 * power_abc.abs().max(1.0),
            "power_ab={power_ab}, power_abc={power_abc}");
    }

    /// Park transform preserves magnitude: d^2+q^2 = alpha^2+beta^2.
    #[test]
    fn park_preserves_magnitude(
        alpha in -10.0f64..10.0,
        beta in -10.0f64..10.0,
        theta in -core::f64::consts::PI..core::f64::consts::PI,
    ) {
        let ab = AlphaBeta { alpha, beta, zero: 0.0 };
        let dq = park(&ab, theta);
        let mag_in = alpha * alpha + beta * beta;
        let mag_out = dq.d * dq.d + dq.q * dq.q;
        prop_assert!((mag_out - mag_in).abs() < 1e-9 * mag_in.abs().max(1.0),
            "mag_in={mag_in}, mag_out={mag_out}");
    }

    /// SVPWM: all duty cycles in [0, 1] for voltage vectors in linear modulation region.
    #[test]
    fn svpwm_duties_in_unit_interval(
        // alpha_n, beta_n are normalized to [-0.5, 0.5]; magnitude check filters overmodulation
        alpha_n in -0.5f64..0.5,
        beta_n in -0.5f64..0.5,
        vdc in 10.0f64..100.0,
    ) {
        // Linear modulation: |v_ab| ≤ Vdc/√3
        let mag_n = (alpha_n * alpha_n + beta_n * beta_n).sqrt();
        let limit = 1.0 / 3.0_f64.sqrt();
        prop_assume!(mag_n <= limit * 0.95);
        let ab = AlphaBeta { alpha: alpha_n * vdc, beta: beta_n * vdc, zero: 0.0 };
        let duty = svpwm(&ab, vdc);
        prop_assert!(duty.ta >= -1e-9 && duty.ta <= 1.0 + 1e-9, "ta={}", duty.ta);
        prop_assert!(duty.tb >= -1e-9 && duty.tb <= 1.0 + 1e-9, "tb={}", duty.tb);
        prop_assert!(duty.tc >= -1e-9 && duty.tc <= 1.0 + 1e-9, "tc={}", duty.tc);
    }

    /// SVPWM: zero voltage vector gives 0.5 duty on all phases.
    #[test]
    fn svpwm_zero_vector_gives_half_duty(vdc in 1.0f64..100.0) {
        let ab = AlphaBeta { alpha: 0.0, beta: 0.0, zero: 0.0 };
        let duty = svpwm(&ab, vdc);
        prop_assert!((duty.ta - 0.5).abs() < 1e-9, "ta={}", duty.ta);
        prop_assert!((duty.tb - 0.5).abs() < 1e-9, "tb={}", duty.tb);
        prop_assert!((duty.tc - 0.5).abs() < 1e-9, "tc={}", duty.tc);
    }
}
