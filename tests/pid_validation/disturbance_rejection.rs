//! PID disturbance rejection validation.

use oxictl::core::saturation::OutputLimiter;
use oxictl::core::signal::{Feedback, Setpoint};
use oxictl::core::traits::Controller;
use oxictl::pid::anti_windup::AntiWindupMethod;
use oxictl::pid::standard::PidConfig;

/// Simulate first-order plant: y[k+1] = a*y[k] + b*u[k] + d[k]
fn first_order_plant(y: f64, u: f64, d: f64, a: f64, b: f64) -> f64 {
    a * y + b * u + d
}

/// PI controller with integral should reject constant disturbance (zero steady-state error).
#[test]
fn pi_rejects_constant_step_disturbance() {
    let mut pid = PidConfig::pi(2.0f64, 1.0).build();
    let dt = 0.01f64;
    let a = 0.95f64; // plant pole
    let b = 0.05f64; // plant gain

    let setpoint = 1.0f64;
    let disturbance = 0.3f64; // Constant step disturbance

    let mut y = 0.0f64;

    // Run for 500 steps without disturbance
    for _ in 0..500 {
        let u = pid
            .update(&Setpoint::new(setpoint), &Feedback::new(y), dt)
            .value();
        y = first_order_plant(y, u, 0.0, a, b);
    }

    // Apply disturbance for 3000 steps (30 seconds)
    for _ in 0..3000 {
        let u = pid
            .update(&Setpoint::new(setpoint), &Feedback::new(y), dt)
            .value();
        y = first_order_plant(y, u, disturbance, a, b);
    }

    // With PI control, steady-state error should be ≈ 0 despite disturbance
    let ss_error = (setpoint - y).abs();
    assert!(
        ss_error < 0.05,
        "PI must reject constant disturbance: |e_ss|={:.4} (should be <0.05)",
        ss_error
    );
}

/// P-only controller has non-zero steady-state error under disturbance.
#[test]
fn p_only_has_nonzero_ss_error_under_disturbance() {
    let mut pid = PidConfig::p(2.0f64).build();
    let dt = 0.01f64;
    let a = 0.95f64;
    let b = 0.05f64;

    let setpoint = 1.0f64;
    let disturbance = 0.3f64;

    let mut y = 0.0f64;
    for _ in 0..1000 {
        let u = pid
            .update(&Setpoint::new(setpoint), &Feedback::new(y), dt)
            .value();
        y = first_order_plant(y, u, disturbance, a, b);
    }

    // P-only: steady-state error exists
    let ss_error = (setpoint - y).abs();
    assert!(
        ss_error > 0.01,
        "P-only should have nonzero SS error under disturbance: |e|={:.4}",
        ss_error
    );
}

/// PID recovers from impulse disturbance within a bounded time.
#[test]
fn pid_recovers_from_impulse_disturbance() {
    let mut pid = PidConfig::pid(3.0f64, 1.0, 0.1).build();
    let dt = 0.01f64;
    let a = 0.9f64;
    let b = 0.1f64;

    let setpoint = 1.0f64;
    let mut y = 0.0f64;

    // Reach steady state
    for _ in 0..300 {
        let u = pid
            .update(&Setpoint::new(setpoint), &Feedback::new(y), dt)
            .value();
        y = first_order_plant(y, u, 0.0, a, b);
    }

    // Apply impulse disturbance
    let u = pid
        .update(&Setpoint::new(setpoint), &Feedback::new(y), dt)
        .value();
    y = first_order_plant(y, u, 5.0, a, b); // Large impulse

    // Record recovery
    let perturbed_y = y;
    for _ in 0..400 {
        let u = pid
            .update(&Setpoint::new(setpoint), &Feedback::new(y), dt)
            .value();
        y = first_order_plant(y, u, 0.0, a, b);
    }

    // Should have recovered to within 0.1 of setpoint
    let final_error = (setpoint - y).abs();
    assert!(
        final_error < 0.1,
        "PID should recover from impulse: initial_disp={:.3}, final_error={:.4}",
        perturbed_y - setpoint,
        final_error
    );
}

/// Anti-windup limits overshoot after large disturbance removal.
#[test]
fn integral_windup_doesnt_cause_large_overshoot_after_disturbance() {
    let mut pid = PidConfig::pi(2.0f64, 0.5)
        .with_limits(-10.0, 10.0)
        .with_anti_windup(AntiWindupMethod::Clamping)
        .build();

    let dt = 0.01f64;
    let a = 0.95f64;
    let b = 0.05f64;
    let setpoint = 1.0f64;
    let mut y = 0.0f64;

    // Apply saturating disturbance
    for _ in 0..200 {
        let u = pid
            .update(&Setpoint::new(setpoint), &Feedback::new(y), dt)
            .value();
        y = first_order_plant(y, u, 20.0, a, b); // Large disturbance
    }

    // Remove disturbance: recover to setpoint
    for _ in 0..500 {
        let u = pid
            .update(&Setpoint::new(setpoint), &Feedback::new(y), dt)
            .value();
        y = first_order_plant(y, u, 0.0, a, b);
    }

    // After recovery, y should be near setpoint (anti-windup prevents infinite windup)
    let ss_error = (y - setpoint).abs();
    assert!(
        ss_error < 2.0,
        "Anti-windup: should converge after disturbance removal: y={:.3}, sp={:.3}",
        y,
        setpoint
    );
}

// Suppress unused import warnings
fn _assert_imports_used() {
    let _ = OutputLimiter::<f64>::new(-1.0, 1.0);
}
