//! LMS adaptive noise cancellation example.
//!
//! Demonstrates how a 4-tap LMS filter progressively learns to estimate
//! the noise component of a primary signal, leaving only the clean sinusoid.
//!
//! Signal model:
//!   - Clean signal:   s[n] = sin(2π · 0.1 · n)   (amplitude 1.0)
//!   - LCG noise:      v[n]   (deterministic pseudo-random, range ±0.4)
//!   - Primary signal: d[n] = s[n] + v[n]
//!   - Reference:      x[n] = v[n]   (same noise, correlated with primary)
//!
//! The LMS filter W maps x[n] → y[n] ≈ v[n].  The error
//!   e[n] = d[n] − y[n]  →  s[n]   (clean sinusoid)
//! as the filter weight w[0] converges toward 1.0.
//!
//! After convergence the residual noise power (e[n] − s[n]) shrinks
//! dramatically compared with the original noise power.
//!
//! Run with:
//!   cargo run --example lms_noise_cancel --features "std"

use oxictl::core::adaptive_filters::LmsFilter;

/// Galois LCG returning values uniformly in (−0.4, 0.4).
#[inline]
fn lcg_step(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // High 32 bits → integer in [−200, 199] → divide by 500.0 → ±0.4
    let bits = (*state >> 32) as i32 % 400; // range [−200, 199]
    bits as f64 / 500.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LMS Adaptive Noise Cancellation ===\n");
    println!("Filter order N = 4,  step size μ = 0.02");
    println!("Primary: d[n] = sin(2π·0.1·n) + v[n],  |v[n]| ≤ 0.40");
    println!("Reference: x[n] = v[n]  (correlated noise, independent of sinusoid)");
    println!("Goal: residual noise power after filtering  <<  original noise power\n");

    // ── LMS filter: 4 taps, step size μ = 0.02 ───────────────────────────────
    // Noise power ≈ (0.4)²/3 ≈ 0.053.  LMS stability requires mu < 1/(2N·Px)
    // ≈ 2.4.  We use mu = 0.02 for lower misadjustment.
    const N: usize = 4;
    let mu: f64 = 0.02;
    let mut lms = LmsFilter::<f64, N>::new(mu).map_err(|e| format!("LMS init error: {e}"))?;

    // Sliding window of the N most recent reference noise samples.
    // Index 0 = most recent, matching LmsFilter::update convention.
    let mut x_buf = [0.0_f64; N];

    let mut lcg_state: u64 = 12345;
    let n_iter: usize = 800;

    println!(
        "{:>6}  {:>10}  {:>12}  {:>10}  {:>14}  {:>8}",
        "Iter", "Clean s[n]", "Primary d[n]", "Error e[n]", "Resid noise", "w[0]"
    );
    println!("{}", "-".repeat(70));

    let two_pi_01 = 2.0 * core::f64::consts::PI * 0.1;

    for n in 0..n_iter {
        let noise = lcg_step(&mut lcg_state);
        let clean = libm::sin(two_pi_01 * n as f64);
        let primary = clean + noise;

        // Shift reference buffer and insert current noise at index 0
        for i in (1..N).rev() {
            x_buf[i] = x_buf[i - 1];
        }
        x_buf[0] = noise;

        // LMS: desired = primary, reference = noise buffer
        let y = lms
            .update(&x_buf, primary)
            .map_err(|e| format!("LMS update error: {e}"))?;
        let error = primary - y;
        let residual = error - clean; // how much noise remains in e[n]

        if n % 100 == 0 {
            println!(
                "{:>6}  {:>10.5}  {:>12.5}  {:>10.5}  {:>14.6}  {:>8.4}",
                n,
                clean,
                primary,
                error,
                residual,
                lms.weights()[0]
            );
        }
    }

    println!("\n=== Summary ===");

    // Measure residual noise MSE and original noise MSE over 100 fresh steps,
    // using a separate (untrained) filter for the baseline.
    let mut mse_residual: f64 = 0.0;
    let mut mse_original: f64 = 0.0;
    let count = 100usize;

    for n in n_iter..n_iter + count {
        let noise = lcg_step(&mut lcg_state);
        let clean = libm::sin(two_pi_01 * n as f64);
        let primary = clean + noise;
        for i in (1..N).rev() {
            x_buf[i] = x_buf[i - 1];
        }
        x_buf[0] = noise;
        let y = lms
            .update(&x_buf, primary)
            .map_err(|e| format!("LMS update error: {e}"))?;
        let residual = (primary - y) - clean;
        mse_residual += residual * residual;
        mse_original += noise * noise;
    }
    mse_residual /= count as f64;
    mse_original /= count as f64;

    let rmse_residual = libm::sqrt(mse_residual);
    let rmse_original = libm::sqrt(mse_original);
    let noise_reduction_db = 10.0 * libm::log10(mse_original / mse_residual.max(1e-12));

    println!("Original noise RMSE:       {:.6}", rmse_original);
    println!("Residual noise RMSE:       {:.6}", rmse_residual);
    println!("Noise reduction:           {:.2} dB", noise_reduction_db);
    println!(
        "Converged filter weight[0]: {:.6}  (expected → 1.0)",
        lms.weights()[0]
    );

    if noise_reduction_db > 6.0 {
        println!(
            "[PASS] LMS achieved > 6 dB noise reduction after {} iterations.",
            n_iter
        );
    } else {
        println!(
            "[INFO] Noise reduction {:.2} dB; increase n_iter for stronger cancellation.",
            noise_reduction_db
        );
    }

    Ok(())
}
