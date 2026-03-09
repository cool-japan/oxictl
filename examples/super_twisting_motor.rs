//! Super-Twisting Algorithm (STA) motor speed control with disturbance rejection.
//!
//! # System model
//! First-order motor plant:
//!
//!   ẏ = -a·y + b·u + d
//!
//! with a = 10.0, b = 100.0, reference y_ref = 1.0 (step).
//!
//! # Sliding variable
//!   σ = y_ref − y  (tracking error)
//!
//! The super-twisting algorithm drives σ and σ̇ to zero in finite time despite
//! bounded matched disturbances.  A step disturbance d = 2.0 is injected at
//! step 100 to demonstrate rejection.
//!
//! Run: `cargo run --example super_twisting_motor --features state_feedback`

use oxictl::state_feedback::SuperTwistingController;

/// Euler step of the first-order motor plant.
///
///   ẏ = -a·y + b·u + d
///
/// Returns updated output y.
fn plant_step(y: f64, u: f64, d: f64, a: f64, b: f64, dt: f64) -> f64 {
    y + dt * (-a * y + b * u + d)
}

fn main() -> Result<(), String> {
    // ------------ Plant parameters ------------
    let a = 10.0_f64; // first-order pole
    let b = 100.0_f64; // input gain
    let dt = 0.001_f64; // 1 kHz sampling

    // ------------ Reference (step) ------------
    let y_ref = 1.0_f64;

    // ------------ Super-Twisting controller ------------
    // Stability conditions: k1 > 2·√W, k2 > W + k1²/2
    // With disturbance bound W = d_max/b ≈ 0.02 → k1=2, k2=5 is conservative.
    let mut ctrl = SuperTwistingController::<f64>::new(2.0, 5.0, dt).map_err(|e| e.to_string())?;

    // ------------ Simulation state ------------
    let mut y = 0.0_f64; // plant output (initial)

    println!("# Super-Twisting Motor Control: a={a}, b={b}, y_ref={y_ref}, k1=2.0, k2=5.0");
    println!("# Disturbance d=2.0 injected after step 100");
    println!("{:>6}  {:>10}  {:>10}  {:>10}", "step", "y", "sigma", "u");

    let n_steps = 200_usize;
    for step in 0..n_steps {
        // Disturbance: zero for first 100 steps, then d=2.0
        let d = if step >= 100 { 2.0_f64 } else { 0.0_f64 };

        // Sliding variable: tracking error
        let sigma = y_ref - y;

        // Super-twisting control update
        let u = ctrl.update(sigma).map_err(|e| e.to_string())?;

        // Print every 20 steps
        if step % 20 == 0 {
            println!("{step:6}  {y:10.5}  {sigma:10.5}  {u:10.5}");
        }

        // Advance plant (Euler integration)
        y = plant_step(y, u, d, a, b, dt);
    }

    // Final state
    let final_sigma = y_ref - y;
    eprintln!("\n=== Super-Twisting Motor Summary ===");
    eprintln!("Plant:       ẏ = -{a}·y + {b}·u + d");
    eprintln!("Controller:  STA k1=2.0, k2=5.0, dt={dt}");
    eprintln!("Disturbance: d=2.0 from step 100 onward");
    eprintln!("Final y:     {y:.5}  (reference = {y_ref})");
    eprintln!("Final σ:     {final_sigma:.5}");
    eprintln!("Integral v:  {:.5}", ctrl.integral_state());

    if final_sigma.abs() < 0.05 {
        eprintln!("PASS: output tracks reference within 5% despite disturbance.");
    } else {
        eprintln!(
            "WARN: tracking error exceeds 5% (|σ| = {:.5}).",
            final_sigma.abs()
        );
    }

    Ok(())
}
