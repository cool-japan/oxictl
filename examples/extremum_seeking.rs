//! Gradient-Based Extremum Seeking Control: Unknown Static Map Maximisation.
//!
//! Demonstrates the [`GradientEsc`] controller finding the peak of an unknown
//! static map:
//!
//!   y = -(u - 3.5)² + 10     (peak at u* = 3.5, y* = 10)
//!
//! The ESC starts far from the optimum (u_hat = 0.0) and converges to
//! u_hat ≈ 3.5 using sinusoidal probing + two-filter demodulation.
//!
//! Parameter rationale (Krstić & Wang 2000 time-scale separation):
//!   ω = 20 rad/s  (probing frequency)
//!   h_y = 1 rad/s (output HPF removes DC offset of y)
//!   h   = 5 rad/s (gradient LPF smooths demodulated product)
//!   k_int = 5.0   (integrator gain; must satisfy ω >> k_int)
//!   dt  = 0.01 s  → 3000 steps = 30 s simulation
//!
//! Run with:
//!   cargo run --example extremum_seeking --features "extremum"

use oxictl::extremum::GradientEsc;

/// Unknown static map with peak at u_star = 3.5, peak value = 10.0.
/// The ESC sees only the output y; it has no knowledge of the functional form.
fn unknown_static_map(u: f64) -> f64 {
    -(u - 3.5) * (u - 3.5) + 10.0
}

fn main() -> Result<(), String> {
    println!("=== Gradient Extremum Seeking: Maximising Unknown Static Map ===\n");
    println!("Target map:  y = -(u - 3.5)² + 10   (peak at u* = 3.5, y* = 10.0)");
    println!("Initial estimate:  u_hat = 0.0  (far from optimum)");
    println!();

    // ── ESC parameters ────────────────────────────────────────────────────────
    // Time-scale separation: ω >> k_int ensures averaged analysis holds.
    // HPF bandwidth h_y << ω removes absolute DC offset of y without attenuating
    // probing-period variations.  LPF bandwidth h << ω smooths 2ω harmonics.
    let u_init: f64 = 0.0; // starting estimate (far from u* = 3.5)
    let amplitude: f64 = 0.2; // probing sinusoid amplitude
    let omega: f64 = 20.0; // probing frequency [rad/s]
    let hpf_bw: f64 = 1.0; // output HPF bandwidth h_y [rad/s]
    let lpf_bw: f64 = 5.0; // gradient LPF bandwidth h [rad/s]
    let k_int: f64 = 5.0; // integrator gain
    let dt: f64 = 0.01; // sample period [s]

    let mut esc = GradientEsc::<f64>::new(
        u_init, amplitude, omega, hpf_bw, lpf_bw, k_int, dt, false, // maximise (not minimise)
    )
    .map_err(|e| format!("GradientEsc init error: {e}"))?;

    // ── Simulation ────────────────────────────────────────────────────────────
    let n_steps: usize = 3000;

    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>12}",
        "Step", "u_hat", "u_probe", "y_measured", "Error |u*-û|"
    );
    println!("{}", "-".repeat(62));

    let u_star: f64 = 3.5;
    let mut final_u_hat: f64 = u_init;

    for step in 0..n_steps {
        // Get current probing input
        let u_probe = esc.probing_input();

        // Evaluate the unknown static map (plant measurement)
        let y = unknown_static_map(u_probe);

        // Feed measurement back to ESC, advance internal state
        let _u_next = esc
            .update(y)
            .map_err(|e| format!("ESC update error at step {step}: {e}"))?;

        let u_hat = esc.estimate();
        final_u_hat = u_hat;

        // Print progress every 500 steps and at step 0
        if step % 500 == 0 {
            let error = (u_hat - u_star).abs();
            println!(
                "{:>6}  {:>12.5}  {:>12.5}  {:>12.5}  {:>12.5}",
                step, u_hat, u_probe, y, error
            );
        }
    }

    // Print final state
    let u_hat_final = esc.estimate();
    let y_final = unknown_static_map(u_hat_final);
    let error_final = (u_hat_final - u_star).abs();

    println!(
        "{:>6}  {:>12.5}  {:>12.5}  {:>12.5}  {:>12.5}",
        n_steps,
        u_hat_final,
        esc.probing_input(),
        y_final,
        error_final
    );

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\n=== Summary ===");
    println!("True optimum:            u* = {u_star:.4},  y* = 10.0000");
    println!("ESC converged estimate:  û  = {final_u_hat:.5}");
    println!("Convergence error:       |u* - û| = {error_final:.5}");

    if error_final < 0.5 {
        println!("[PASS] ESC converged to within 0.5 of the true optimum u* = 3.5.");
    } else {
        println!(
            "[INFO] ESC estimate û = {final_u_hat:.4} — convergence ongoing \
             (more steps or higher k_int may help)."
        );
    }

    println!("\nNote: persistent dithering (±amplitude) at steady state is expected.");
    println!("      The estimate û tracks u* despite having no model of the cost function.");

    Ok(())
}
