//! Integration test: PID with output limits saturates correctly in Q15_16.

use oxictl::core::fixed_point::convert::fixed_from_f32_saturating;
use oxictl::core::fixed_point::types::Q15_16;
use oxictl::core::saturation::OutputLimiter;
use oxictl::core::scalar::PidScalar;
use oxictl::core::signal::{Feedback, Setpoint};
use oxictl::core::traits::Controller;
use oxictl::pid::anti_windup::AntiWindupMethod;
use oxictl::pid::standard::PidConfig;

#[test]
fn pid_with_limits_saturates_correctly() {
    let kp = fixed_from_f32_saturating(2.0);
    let ki = fixed_from_f32_saturating(0.5);
    let kd = Q15_16::ZERO;

    let config = PidConfig {
        kp,
        ki,
        kd,
        beta: Q15_16::ONE,
        gamma: Q15_16::ZERO,
        output_limiter: Some(OutputLimiter::new(
            fixed_from_f32_saturating(-5.0),
            fixed_from_f32_saturating(5.0),
        )),
        anti_windup: AntiWindupMethod::Clamping,
        derivative_filter_tau: None,
    };
    let mut pid = config.build();

    // Large setpoint step → should saturate at +5.0
    let sp = Setpoint::new(fixed_from_f32_saturating(100.0));
    let fb = Feedback::new(fixed_from_f32_saturating(0.0));
    let dt = fixed_from_f32_saturating(0.01);

    let out = pid.update(&sp, &fb, dt);
    let out_f32 = oxictl::core::fixed_point::convert::fixed_to_f32(out.value());

    assert!(
        out_f32 <= 5.01,
        "Output should be clamped to 5.0, got {}",
        out_f32
    );
    assert!(out.is_saturated(), "Controller should report saturation");
}
