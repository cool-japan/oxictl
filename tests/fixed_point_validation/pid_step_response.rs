//! Integration test: PI controller with Q15_16 fixed-point drives a first-order plant to setpoint.

use oxictl::core::fixed_point::convert::{fixed_from_f32_saturating, fixed_to_f32};
use oxictl::core::fixed_point::types::Q15_16;
use oxictl::core::scalar::PidScalar;
use oxictl::core::signal::{Feedback, Setpoint};
use oxictl::core::traits::Controller;
use oxictl::pid::anti_windup::AntiWindupMethod;
use oxictl::pid::standard::PidConfig;

#[test]
fn pi_pid_drives_first_order_plant_to_setpoint_q15_16() {
    // Q15_16 gains for a PI controller: Kp=0.5, Ki=1.0, Kd=0
    let kp = fixed_from_f32_saturating(0.5);
    let ki = fixed_from_f32_saturating(1.0);
    let kd = Q15_16::ZERO;

    let config = PidConfig {
        kp,
        ki,
        kd,
        beta: Q15_16::ONE,
        gamma: Q15_16::ZERO,
        output_limiter: None,
        anti_windup: AntiWindupMethod::Clamping,
        derivative_filter_tau: None,
    };
    let mut pid = config.build();

    // First-order plant: y[k] = y[k-1] + dt * (-y[k-1] + u[k-1])
    // tau=1, kplant=1
    let mut y = Q15_16::ZERO;
    let setpoint = fixed_from_f32_saturating(1.0);
    let dt = fixed_from_f32_saturating(0.01);

    for _ in 0..300_usize {
        let sp = Setpoint::new(setpoint);
        let fb = Feedback::new(y);
        let u = pid.update(&sp, &fb, dt);
        // Plant step: y += dt * (-y + u)
        y = y + dt * ((-y) + u.value());
    }

    let y_f32 = fixed_to_f32(y);
    // Should converge within 5% of setpoint (1.0) to account for fixed-point quantization
    assert!(
        (y_f32 - 1.0_f32).abs() < 0.05,
        "PI controller should drive plant near setpoint 1.0, got y={}",
        y_f32
    );
}
