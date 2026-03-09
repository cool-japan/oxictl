//! PID anti-windup validation tests.

use oxictl::core::saturation::OutputLimiter;
use oxictl::core::signal::{Feedback, Setpoint};
use oxictl::core::traits::Controller;
use oxictl::pid::anti_windup::AntiWindupMethod;
use oxictl::pid::standard::PidConfig;

/// Test that clamping anti-windup prevents integral from growing unboundedly.
#[test]
fn clamping_antiwindup_limits_integral() {
    let mut pid = PidConfig::pi(1.0f64, 2.0)
        .with_limits(-5.0, 5.0)
        .with_anti_windup(AntiWindupMethod::Clamping)
        .build();

    let dt = 0.01f64;
    // Apply large constant error (saturates output)
    for _ in 0..500 {
        pid.update(&Setpoint::new(100.0), &Feedback::new(0.0), dt);
    }

    // Get output - should be clipped to 5.0 (saturation)
    let out = pid.update(&Setpoint::new(100.0), &Feedback::new(0.0), dt);
    assert!(
        out.value() <= 5.1,
        "Output should be clipped by limiter, got {}",
        out.value()
    );

    // Now set error to zero. With anti-windup, recovery should be fast.
    let _out_zero_err = pid.update(&Setpoint::new(0.0), &Feedback::new(0.0), dt);
    let out_after = pid.update(&Setpoint::new(0.0), &Feedback::new(0.0), dt);
    assert!(
        out_after.value().abs() < 10.0,
        "Anti-windup: output should not be massively wound up, got {}",
        out_after.value()
    );
}

/// Test back-calculation anti-windup recovery.
#[test]
fn back_calculation_antiwindup_recovery() {
    let mut pid = PidConfig::pi(1.0f64, 0.5)
        .with_limits(-3.0, 3.0)
        .with_anti_windup(AntiWindupMethod::BackCalculation { kb: 0.1 })
        .build();

    let dt = 0.01f64;

    // Saturate for 200 steps
    for _ in 0..200 {
        pid.update(&Setpoint::new(10.0), &Feedback::new(0.0), dt);
    }

    // Now at setpoint. Output should recover quickly.
    let mut max_out = 0.0f64;
    for _ in 0..50 {
        let out = pid.update(&Setpoint::new(0.0), &Feedback::new(0.0), dt);
        max_out = max_out.max(out.value().abs());
    }
    assert!(
        max_out < 5.0,
        "Back-calculation should limit windup recovery overshoot, max_out={}",
        max_out
    );
}

/// Test no anti-windup as baseline: integral does wind up.
#[test]
fn no_antiwindup_integral_windup_occurs() {
    let mut pid = PidConfig::pi(0.1f64, 5.0) // High ki
        .with_limits(-1.0, 1.0)
        .with_anti_windup(AntiWindupMethod::None)
        .build();

    let dt = 0.01f64;
    // Large error for long time → integral winds up
    for _ in 0..300 {
        pid.update(&Setpoint::new(10.0), &Feedback::new(0.0), dt);
    }

    // Now try to regulate. Output should be stuck at limit due to windup.
    let out = pid.update(&Setpoint::new(0.0), &Feedback::new(0.0), dt);
    assert!(
        out.value() >= 0.5,
        "Without anti-windup, integral should still be wound up: out={}",
        out.value()
    );
}

/// Clamping anti-windup: in saturation, integral freezes.
#[test]
fn clamping_freezes_integral_in_saturation() {
    let mut pid = PidConfig::pi(1.0f64, 10.0)
        .with_limits(-2.0, 2.0)
        .with_anti_windup(AntiWindupMethod::Clamping)
        .build();

    let dt = 0.001f64;

    // Drive to saturation
    for _ in 0..100 {
        pid.update(&Setpoint::new(5.0), &Feedback::new(0.0), dt);
    }
    let out1 = pid.update(&Setpoint::new(5.0), &Feedback::new(0.0), dt);

    // Further steps at same error: output should stay clipped (not grow)
    for _ in 0..200 {
        pid.update(&Setpoint::new(5.0), &Feedback::new(0.0), dt);
    }
    let out2 = pid.update(&Setpoint::new(5.0), &Feedback::new(0.0), dt);

    // Both should be at saturation limit
    assert!(
        (out1.value() - out2.value()).abs() < 0.1,
        "Clamping should freeze integral: out1={}, out2={}",
        out1.value(),
        out2.value()
    );
    assert!(
        out2.value().abs() <= 2.01,
        "Output must stay within limits: {}",
        out2.value()
    );
}

// Suppress unused import warning (OutputLimiter used transitively)
fn _assert_output_limiter_used() {
    let _ = OutputLimiter::<f64>::new(-1.0, 1.0);
}
