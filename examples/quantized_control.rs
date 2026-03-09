//! Quantized Control: Effect of Finite-Bit Quantization on a P-Controller.
//!
//! Simulates a simple first-order discrete plant:
//!   y[k+1] = 0.9·y[k] + 0.1·u[k]
//!
//! with a proportional controller:
//!   u[k] = 5.0·(ref - y[k])
//!
//! Two cases are compared side-by-side:
//!   (a) Ideal: no quantization (continuous control signal u).
//!   (b) Quantized: 4-bit uniform quantizer applied to u before plant entry.
//!       Range: [−10, +10], giving 16 levels and step size Δ = 20/15 ≈ 1.333.
//!
//! Reference: ref = 1.0.  Both plants start at y = 0.0.
//!
//! With 4-bit quantization the control signal cannot represent the exact
//! error-correcting value, causing a limit cycle (y oscillates between two
//! adjacent quantization levels at steady state) instead of monotone settling.
//!
//! SNR of the 4-bit quantizer:  6.02·4 + 1.76 ≈ 25.84 dB
//!
//! Run with:
//!   cargo run --example quantized_control --features "comm"

use oxictl::comm::UniformQuantizer;

fn main() -> Result<(), String> {
    println!("=== Quantized Control: P-Controller with 4-Bit Uniform Quantizer ===\n");

    // ── Plant parameters ──────────────────────────────────────────────────────
    // y[k+1] = a_d·y[k] + b_d·u[k]
    let a_d: f64 = 0.9; // discrete plant pole
    let b_d: f64 = 0.1; // discrete plant gain
    let reference: f64 = 1.0; // step reference
    let kp: f64 = 5.0; // proportional gain

    // ── 4-bit uniform quantizer on u ─────────────────────────────────────────
    // Range [−10, +10] covers the initial transient: u_max ≈ kp·ref = 5.0
    // and provides headroom.
    // Levels: 2^4 = 16.  Δ = 20 / (16−1) = 20/15 ≈ 1.333.
    let bits: u8 = 4;
    let u_min: f64 = -10.0;
    let u_max: f64 = 10.0;

    let quantizer = UniformQuantizer::<f64>::new(bits, u_min, u_max)
        .map_err(|e| format!("UniformQuantizer init error: {e:?}"))?;

    let snr = quantizer.snr_db(0.707); // full-scale sinusoid SNR
    let delta = quantizer.delta();
    let levels = quantizer.levels();

    println!("Quantizer:  {bits}-bit uniform,  range [{u_min}, {u_max}]");
    println!("            levels = {levels},  Δ = {delta:.4},  SNR ≈ {snr:.2} dB");
    println!("Plant:      y[k+1] = {a_d}·y[k] + {b_d}·u[k]");
    println!("Controller: u[k] = {kp}·(ref − y[k]),  ref = {reference}");
    println!();

    // ── Initial state ─────────────────────────────────────────────────────────
    let mut y_exact: f64 = 0.0; // ideal (no quantization)
    let mut y_quant: f64 = 0.0; // with 4-bit quantization

    let n_steps: usize = 50;

    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "Step", "y_exact", "u_exact", "y_quant", "u_quant", "Δu_error"
    );
    println!("{}", "-".repeat(72));

    // Print step 0 (initial state)
    {
        let u_e = kp * (reference - y_exact);
        let u_q_cont = kp * (reference - y_quant);
        let u_q = quantizer.quantize(u_q_cont);
        println!(
            "{:>6}  {:>12.6}  {:>12.6}  {:>12.6}  {:>12.6}  {:>12.6}",
            0,
            y_exact,
            u_e,
            y_quant,
            u_q,
            (u_q - u_e).abs()
        );
    }

    for step in 1..=n_steps {
        // ── Exact (ideal) path ────────────────────────────────────────────────
        let u_exact = kp * (reference - y_exact);
        // Clamp to quantizer range to keep comparison fair
        let u_exact_clamped = u_exact.clamp(u_min, u_max);
        y_exact = a_d * y_exact + b_d * u_exact_clamped;

        // ── Quantized path ────────────────────────────────────────────────────
        let u_continuous = kp * (reference - y_quant);
        let u_quantized = quantizer.quantize(u_continuous);
        y_quant = a_d * y_quant + b_d * u_quantized;

        // Print every 10 steps
        if step % 10 == 0 {
            let u_cont_ref = kp * (reference - y_quant); // for display (pre-quantize)
            let u_q_disp = quantizer.quantize(u_cont_ref);
            let u_e_disp = kp * (reference - y_exact);
            println!(
                "{:>6}  {:>12.6}  {:>12.6}  {:>12.6}  {:>12.6}  {:>12.6}",
                step,
                y_exact,
                u_e_disp,
                y_quant,
                u_q_disp,
                (u_q_disp - u_e_disp).abs()
            );
        }
    }

    // ── Limit cycling analysis ────────────────────────────────────────────────
    // Check the last few steps for oscillation
    let mut y_history: [f64; 6] = [0.0; 6];
    let mut y_quant_check = y_quant;

    for slot in y_history.iter_mut() {
        let u_cont = kp * (reference - y_quant_check);
        let u_q = quantizer.quantize(u_cont);
        y_quant_check = a_d * y_quant_check + b_d * u_q;
        *slot = y_quant_check;
    }

    let amplitude_oscillation = y_history[0..6]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        - y_history[0..6]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\n=== Summary ===");
    println!(
        "Ideal final output:      y_exact = {y_exact:.6}  (error = {:.6})",
        (y_exact - reference).abs()
    );
    println!(
        "Quantized final output:  y_quant = {y_quant:.6}  (error = {:.6})",
        (y_quant - reference).abs()
    );
    println!(
        "Limit-cycle amplitude:   {amplitude_oscillation:.6}  \
         (quantizer Δ = {delta:.4})"
    );

    let ideal_settled = (y_exact - reference).abs() < 1e-4;
    let quant_limit_cycling = amplitude_oscillation > delta * 0.05;

    if ideal_settled {
        println!("[PASS] Ideal controller settled to reference within 1e-4.");
    } else {
        println!("[INFO] Ideal controller output: {y_exact:.6}  (still converging).");
    }

    if quant_limit_cycling {
        println!(
            "[PASS] Quantized controller shows limit cycling \
             (amplitude = {amplitude_oscillation:.4} > 5% of Δ = {:.4}).",
            delta * 0.05
        );
        println!(
            "       This is the expected steady-state behaviour for \
             {bits}-bit ({levels} levels) quantization."
        );
    } else {
        println!(
            "[INFO] Quantized controller settled — quantization resolution \
             may be sufficient for this reference."
        );
    }

    println!("\nIncrease 'bits' to reduce Δ and suppress limit cycling.");

    Ok(())
}
