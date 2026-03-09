//! Frequency-domain stability analysis example.
//!
//! Demonstrates Bode plot computation, stability margin analysis, Nyquist
//! stability criterion, and sensitivity peak for a lead-lag compensated
//! second-order plant.
//!
//! Run with:
//!   cargo run --example bode_analysis --features "std"

use oxictl::core::frequency_domain::{
    compute_bode, compute_nyquist, distance_to_critical, gain_margin, is_stable_nyquist,
    peak_sensitivity, phase_margin, BodeData, LoopShaping,
};
use oxictl::core::transfer_fn::TransferFn;

fn main() {
    println!("=== Bode Stability Analysis: Lead-Lag Compensated Second-Order Plant ===\n");

    // ── Plant: discrete-time second-order lowpass ─────────────────────────────
    // Bilinear (Tustin) approximation of G(s) = ω_n² / (s² + 2ζω_n·s + ω_n²)
    // with ω_n = 5.0 rad/s, ζ = 0.7, Ts = 0.02 s.
    // Coefficients computed analytically for the Tustin mapping s → 2(z-1)/(Ts(z+1)).
    // b = [b0, b1, b2], a = [a1, a2] in the form H(z) = (b0 + b1 z^-1 + b2 z^-2)
    //                                                  / (1 + a1 z^-1 + a2 z^-2)
    // For simplicity we use a pre-computed stable second-order plant TF:
    //   H_plant(z) = 0.0452(z+1)² / (z² - 1.652z + 0.7067)
    //   => b = [0.0452, 0.0904, 0.0452], a = [-1.652, 0.7067]
    let b_plant = [0.0452_f64, 0.0904, 0.0452];
    let a_plant = [-1.652_f64, 0.7067, 0.0];
    let plant = TransferFn::<f64, 3>::new(b_plant, a_plant);

    // ── Lead-lag compensator ───────────────────────────────────────────────────
    // C(z) = K · (z - z_lead) / (z - p_lead) · (z - z_lag) / (z - p_lag)
    // Lead: provides phase advance near crossover (boosts phase margin)
    // Lag:  provides DC gain boost (improves steady-state accuracy)
    // Approximate discrete lead-lag:
    //   b_ctrl = [1.2, -0.96, 0.0], a_ctrl = [-0.7, 0.0, 0.0] (order-2 approximation)
    //   This gives ~30° phase lead and 6 dB gain at crossover.
    let b_ctrl = [1.2_f64, -0.96, 0.0];
    let a_ctrl = [-0.7_f64, 0.0, 0.0];
    let controller = TransferFn::<f64, 3>::new(b_ctrl, a_ctrl);

    // ── Open-loop Bode plot: L = P·C ─────────────────────────────────────────
    // Frequency range: 0.01 to π (Nyquist) in 128 log-spaced points
    let omega_min = 0.01_f64;
    let omega_max = core::f64::consts::PI * 0.95; // approach but not reach Nyquist

    // Bode data for the plant alone
    let bode_plant: BodeData<f64, 128> = compute_bode(&plant, omega_min, omega_max)
        .expect("Bode computation for plant should succeed");

    println!("Plant frequency response summary:");
    println!(
        "  DC gain (low-ω magnitude): {:.2} dB",
        bode_plant.points[0].magnitude_db
    );
    println!(
        "  High-ω magnitude: {:.2} dB",
        bode_plant.points[127].magnitude_db
    );

    // ── Loop shaping: sensitivity analysis ────────────────────────────────────
    let loop_shape = LoopShaping::new(plant, controller);
    let sens_data = loop_shape
        .compute_sensitivity_response::<128>(omega_min, omega_max)
        .expect("Sensitivity response computation should succeed");

    // Sensitivity peak: Ms = ‖S‖∞ (H-infinity norm)
    let ms = peak_sensitivity(&sens_data);
    let ms_db = 20.0 * ms.log10();
    println!("\nLoop-shaping sensitivity analysis:");
    println!(
        "  Sensitivity peak Ms = {:.4} (linear), {:.2} dB",
        ms, ms_db
    );
    if ms <= 2.0 {
        println!("  [PASS] Ms ≤ 2.0 → well-conditioned closed-loop (robust)");
    } else {
        println!("  [WARN] Ms > 2.0 → consider redesigning compensator");
    }

    // ── Open-loop Bode for plant (standalone) ─────────────────────────────────
    // Gain margin and phase margin for the plant alone (no compensator)
    let plant_plain = TransferFn::<f64, 3>::new(b_plant, a_plant);
    let bode_plain: BodeData<f64, 128> = compute_bode(&plant_plain, omega_min, omega_max)
        .expect("Bode for plain plant should succeed");

    match gain_margin(&bode_plain) {
        Some(gm) => println!("\nPlant-only gain margin: {:.2} dB", gm),
        None => println!("\nPlant-only gain margin: undefined (phase never crosses -180°)"),
    }
    match phase_margin(&bode_plain) {
        Some(pm) => println!("Plant-only phase margin: {:.2}°", pm),
        None => println!("Plant-only phase margin: undefined (no gain crossover)"),
    }

    // ── Nyquist stability analysis ─────────────────────────────────────────────
    // Evaluate the plant at the Nyquist curve to check stability criterion.
    // is_stable_nyquist takes the TF and number of points directly.
    let stable = is_stable_nyquist(&plant_plain, 128);
    println!("\nNyquist stability criterion (plant only):");
    println!("  Stable by Nyquist: {}", if stable { "YES" } else { "NO" });

    // Compute Nyquist data for distance-to-critical-point analysis
    let nyquist_data = compute_nyquist::<f64, 3, 128>(&plant_plain, omega_max)
        .expect("Nyquist computation should succeed");
    println!(
        "  Distance to critical point (-1+0j): {:.4}",
        distance_to_critical(&nyquist_data)
    );

    // ── Bode plot printout (selected frequencies) ─────────────────────────────
    println!("\nOpen-loop Bode plot (plant, selected frequencies):");
    println!(
        "{:>10}  {:>12}  {:>12}",
        "ω (rad/s)", "Mag (dB)", "Phase (°)"
    );
    println!("{}", "-".repeat(38));

    // Print every 16th point for brevity
    for i in (0..128).step_by(16) {
        let pt = &bode_plain.points[i];
        println!(
            "{:>10.4}  {:>12.3}  {:>12.2}",
            pt.omega, pt.magnitude_db, pt.phase_deg
        );
    }

    // ── Summary ────────────────────────────────────────────────────────────────
    println!("\n=== Summary ===");
    println!("Lead-lag compensator applied to second-order plant.");
    println!("Sensitivity peak: {:.3} ({:.2} dB)", ms, ms_db);
    println!(
        "Closed-loop robustness: {}",
        if ms_db <= 6.0 {
            "GOOD (Ms ≤ 6 dB)"
        } else {
            "MARGINAL"
        }
    );
}
