//! Property-based tests for PID controller invariants.

use oxictl::core::signal::{Feedback, Setpoint};
use oxictl::core::traits::Controller;
use oxictl::pid::standard::PidConfig;
use proptest::prelude::*;

proptest! {
    /// P-only: output is exactly Kp * error on first call.
    #[test]
    fn p_only_output_is_kp_times_error(
        kp in 0.001f64..100.0,
        sp_val in -100.0f64..100.0,
        fb_val in -100.0f64..100.0,
    ) {
        prop_assume!(kp.is_finite() && sp_val.is_finite() && fb_val.is_finite());
        let mut pid = PidConfig::p(kp).build();
        let sp = Setpoint::new(sp_val);
        let fb = Feedback::new(fb_val);
        let out = pid.update(&sp, &fb, 0.01f64);
        let expected = kp * (sp_val - fb_val);
        prop_assert!(
            (out.value() - expected).abs() < 1e-9 * expected.abs().max(1.0),
            "kp={kp}, error={}, out={}, expected={expected}", sp_val - fb_val, out.value()
        );
    }

    /// Output is always finite for finite inputs and gains.
    #[test]
    fn output_is_finite(
        kp in 0.001f64..10.0,
        ki in 0.0f64..5.0,
        kd in 0.0f64..1.0,
        sp_val in -10.0f64..10.0,
        fb_val in -10.0f64..10.0,
        dt in 0.0001f64..0.1,
    ) {
        prop_assume!(kp.is_finite() && ki.is_finite() && kd.is_finite());
        prop_assume!(sp_val.is_finite() && fb_val.is_finite() && dt.is_finite());
        let mut pid = PidConfig::pid(kp, ki, kd).build();
        for _ in 0..10 {
            let out = pid.update(&Setpoint::new(sp_val), &Feedback::new(fb_val), dt);
            prop_assert!(out.value().is_finite(), "Output became non-finite");
        }
    }

    /// With output limits, output is always clamped.
    #[test]
    fn output_respects_limits(
        kp in 0.1f64..100.0,
        ki in 0.1f64..50.0,
        error in -100.0f64..100.0,
        lo in -50.0f64..-1.0,
        hi in 1.0f64..50.0,
        n_steps in 1usize..50,
    ) {
        prop_assume!(lo < hi && lo.is_finite() && hi.is_finite());
        prop_assume!(kp.is_finite() && ki.is_finite() && error.is_finite());
        let mut pid = PidConfig::pi(kp, ki)
            .with_limits(lo, hi)
            .build();
        let sp = Setpoint::new(0.0f64);
        let fb = Feedback::new(-error);
        for _ in 0..n_steps {
            let out = pid.update(&sp, &fb, 0.01);
            prop_assert!(
                out.value() >= lo - 1e-9 && out.value() <= hi + 1e-9,
                "output={} not in [{lo},{hi}]", out.value()
            );
        }
    }

    /// Reset clears integral: output after reset equals P+I for first step only.
    #[test]
    fn reset_clears_integral(
        kp in 0.001f64..100.0,
        ki in 0.001f64..10.0,
        error in -10.0f64..10.0,
    ) {
        prop_assume!(kp.is_finite() && ki.is_finite() && error.is_finite());
        let mut pid = PidConfig::pi(kp, ki).build();
        let sp = Setpoint::new(0.0f64);
        let fb = Feedback::new(-error);
        // Build up integral over many steps
        for _ in 0..20 {
            pid.update(&sp, &fb, 0.01);
        }
        // Sample mid-run: integral should be large
        let out_before_reset = pid.update(&sp, &fb, 0.01);

        pid.reset();
        // First output after reset: P + first-step integral (ki*error*dt), no prior accumulation
        let dt = 0.01f64;
        let out = pid.update(&sp, &fb, dt);
        let expected = kp * error + ki * error * dt;
        prop_assert!(out.value().is_finite());
        // After reset, accumulated integral is zero — output must equal P + one-step I
        prop_assert!(
            (out.value() - expected).abs() < 1e-9 * expected.abs().max(1.0),
            "After reset, out={}, expected={expected} (P+1-step-I)", out.value()
        );
        // The output before reset had large integral, should differ significantly when ki*error large
        let large_integral = (ki * error * 20.0 * dt).abs() > 0.01;
        if large_integral {
            prop_assert!(
                out_before_reset.value().abs() > out.value().abs() - 1e-6
                    || (out_before_reset.value() - out.value()).abs() > 1e-6,
                "Reset should have changed the output"
            );
        }
    }

    /// Zero error, zero integral state → zero output (P+D).
    #[test]
    fn zero_error_zero_output_on_fresh_controller(
        kp in 0.001f64..100.0,
        ki in 0.0f64..10.0,
        kd in 0.0f64..5.0,
    ) {
        prop_assume!(kp.is_finite() && ki.is_finite() && kd.is_finite());
        let mut pid = PidConfig::pid(kp, ki, kd).build();
        let sp = Setpoint::new(0.0f64);
        let fb = Feedback::new(0.0f64);
        let out = pid.update(&sp, &fb, 0.01);
        prop_assert!(
            out.value().abs() < 1e-12,
            "Zero error should give zero output: {}", out.value()
        );
    }
}
