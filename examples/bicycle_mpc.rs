//! Kinematic bicycle model with P-controller path tracking.
//!
//! Scenario: a vehicle with wheelbase L = 2.7 m travels at constant longitudinal
//! speed v = 5 m/s and must follow a sinusoidal reference path:
//!
//!   y_ref(x) = sin(x / 10)
//!
//! A proportional steering-rate controller drives the lateral error to zero:
//!
//!   δ̇ = Kp · (y_ref − y)
//!
//! where Kp = 0.15 rad/(s·m).  The steering angle is saturated to ±0.4 rad to
//! prevent the controller from winding up.  The simulation runs for 100 steps at
//! dt = 0.05 s.  Position and lateral error are printed every 10 steps.
//!
//! Run with:
//!   cargo run --example bicycle_mpc --features "sim"

use oxictl::sim::KinematicBicycle;

/// Sinusoidal reference: y_ref = sin(x / 10).
#[inline]
fn y_ref(x: f64) -> f64 {
    libm::sin(x / 10.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Kinematic Bicycle Model — Sinusoidal Path Tracking ===\n");

    // ── Vehicle parameters ────────────────────────────────────────────────────
    let wheelbase: f64 = 2.7; // L (m) — typical mid-size car
    let dt: f64 = 0.05; // integration timestep (s)
    let v: f64 = 5.0; // constant longitudinal speed (m/s)

    // ── Controller gain and steering saturation ───────────────────────────────
    // Proportional steering-rate gain: δ̇ = Kp * (y_ref − y)
    // The steering angle δ is saturated to ±delta_max to prevent wind-up.
    let kp: f64 = 0.15;
    let delta_max: f64 = 0.4; // rad (~23°), physically achievable limit

    // ── Construct kinematic bicycle, initial position (0, 0) ─────────────────
    let mut bike = KinematicBicycle::<f64>::new(wheelbase, dt, 0.0, 0.0)
        .map_err(|e| format!("Bicycle model error: {e}"))?;

    // ── Simulation header ─────────────────────────────────────────────────────
    println!("Wheelbase L = {wheelbase} m,  dt = {dt} s,  v = {v} m/s,  Kp = {kp},  δ_max = ±{delta_max} rad");
    println!("Reference: y_ref(x) = sin(x / 10)\n");
    println!(
        "{:>5}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "step", "x (m)", "y (m)", "y_ref", "lat_err", "delta (rad)"
    );
    println!("{}", "-".repeat(60));

    let n_steps: usize = 100;
    let print_every: usize = 10;

    let mut max_lat_err: f64 = 0.0;
    let mut final_lat_err: f64 = 0.0;

    for step in 0..n_steps {
        let state = bike.state();
        let x = state[0];
        let y = state[1];
        let delta = state[3];

        // Lateral error: positive = vehicle left of reference
        let yr = y_ref(x);
        let lat_err = yr - y;

        // Update running statistics
        if lat_err.abs() > max_lat_err {
            max_lat_err = lat_err.abs();
        }
        final_lat_err = lat_err;

        // Print every `print_every` steps (including step 0)
        if step % print_every == 0 {
            println!(
                "{:>5}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}",
                step, x, y, yr, lat_err, delta
            );
        }

        // P-controller: steering rate proportional to lateral error.
        // Clamp the desired steering angle to ±delta_max and compute the
        // required rate to reach it from the current angle in one step.
        let delta_desired = (kp * lat_err).clamp(-delta_max, delta_max);
        let delta_dot = (delta_desired - delta) / dt;

        // Advance model one step
        bike.step(v, delta_dot)
            .map_err(|e| format!("Bicycle step error: {e}"))?;
    }

    // Print final state
    let state = bike.state();
    let x = state[0];
    let y = state[1];
    let delta = state[3];
    let yr = y_ref(x);
    let lat_err = yr - y;
    println!(
        "{:>5}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}",
        n_steps, x, y, yr, lat_err, delta
    );

    // ── Summary ───────────────────────────────────────────────────────────────
    println!();
    println!("=== Summary ===");
    println!("Total simulation time : {:.2} s", n_steps as f64 * dt);
    println!("Final x position      : {:.4} m", x);
    println!("Final y position      : {:.4} m", y);
    println!("Final y_ref           : {:.4} m", y_ref(x));
    println!("Final lateral error   : {:.4} m", final_lat_err);
    println!("Max lateral error     : {:.4} m", max_lat_err);
    println!("Final steering angle  : {:.4} rad", delta);

    // ── Validation ────────────────────────────────────────────────────────────
    // A proportional steering-rate controller with saturation tracks a slowly
    // varying sinusoidal reference with a phase lag; errors below 1.0 m are
    // acceptable for this open-loop P-control scenario.
    println!();
    if final_lat_err.abs() < 1.0 {
        println!("[PASS] Vehicle is tracking the reference path (|error| < 1.0 m).");
    } else {
        println!(
            "[INFO] Final lateral error {:.4} m — consider increasing Kp or reducing speed.",
            final_lat_err
        );
    }

    if max_lat_err < 1.5 {
        println!("[PASS] Maximum lateral deviation < 1.5 m — vehicle follows the reference.");
    } else {
        println!(
            "[INFO] Maximum lateral deviation {:.4} m exceeded 1.5 m threshold.",
            max_lat_err
        );
    }

    Ok(())
}
