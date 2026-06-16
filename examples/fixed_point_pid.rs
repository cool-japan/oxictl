//! Demonstrates a PID controller running in Q15.16 fixed-point arithmetic.
//!
//! Run: cargo run --example fixed_point_pid --features "fixed_point,pid,std"

use oxictl::core::fixed_point::convert::{fixed_from_f32_saturating, fixed_to_f32};
use oxictl::core::fixed_point::types::Q15_16;
use oxictl::core::scalar::PidScalar;
use oxictl::core::signal::{Feedback, Setpoint};
use oxictl::core::traits::Controller;
use oxictl::pid::anti_windup::AntiWindupMethod;
use oxictl::pid::standard::PidConfig;

fn main() {
    let kp = fixed_from_f32_saturating(1.0_f32);
    let ki = fixed_from_f32_saturating(2.0_f32);
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

    let setpoint = fixed_from_f32_saturating(1.0_f32);
    let dt = fixed_from_f32_saturating(0.01_f32);
    let mut y = Q15_16::ZERO;

    for step in 0..500_usize {
        let sp = Setpoint::new(setpoint);
        let fb = Feedback::new(y);
        let u = pid.update(&sp, &fb, dt);

        // First-order plant: dy/dt = -y + u  →  y[k] = y[k-1] + dt*(-y+u)
        y = y + dt * ((-y) + u.value());

        #[cfg(feature = "std")]
        if step % 50 == 0 {
            println!(
                "step={:3}  y={:.4}  u={:.4}",
                step,
                fixed_to_f32(y),
                fixed_to_f32(u.value())
            );
        }
        let _ = step; // suppress unused variable warning in no_std builds
    }

    let final_y = fixed_to_f32(y);
    #[cfg(feature = "std")]
    println!("Final: y={:.4} (target 1.0)", final_y);

    assert!(
        (final_y - 1.0_f32).abs() < 0.05,
        "PID should converge: final y={}",
        final_y
    );
}
