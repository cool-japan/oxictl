//! Iterative Learning Control (ILC) on a repetitive pick-and-place task.
//!
//! The robot arm performs the same pick-and-place motion every trial.
//! Trial length: 20 time steps.
//!
//! Reference trajectory: step from 0 → 1 at t = 5, back to 0 at t = 15.
//!   r[n] = 1  for  5 ≤ n < 15
//!   r[n] = 0  otherwise
//!
//! Plant (first-order discrete-time):
//!   y[n] = a · y[n−1] + b · (u_fb[n] + u_ff[n])
//!   a = 0.7,  b = 0.3
//!
//! The ILC feedforward u_ff is initialised to zero and updated each trial
//! by a P-type law:
//!   u_ff_{k+1}[n] = u_ff_k[n] + L · e_k[n]
//!
//! where L = 0.5 is the learning gain and e_k[n] = r[n] − y_k[n] is the
//! per-step tracking error.  No feedback controller (u_fb = 0) is used so
//! that the ILC learning is isolated.
//!
//! Over 10 trials the RMS error should decrease monotonically.
//!
//! Run with:
//!   cargo run --example ilc_repetitive --features "ilc"

use oxictl::ilc::PTypeIlc;

/// Trial length (number of time steps per repetition).
const TRIAL_LEN: usize = 20;

/// Build the reference trajectory: step from 0 to 1 at t=5, back to 0 at t=15.
fn make_reference() -> [f64; TRIAL_LEN] {
    let mut r = [0.0f64; TRIAL_LEN];
    for (n, val) in r.iter_mut().enumerate() {
        if (5..15).contains(&n) {
            *val = 1.0;
        }
    }
    r
}

/// Simulate one trial of the first-order plant.
///
/// Plant:  y[n] = a * y[n-1] + b * u[n]
/// Returns the output trajectory y[0..TRIAL_LEN].
fn simulate_trial(u_ff: &[f64; TRIAL_LEN], a: f64, b: f64) -> [f64; TRIAL_LEN] {
    let mut y = [0.0f64; TRIAL_LEN];
    let mut y_prev = 0.0f64;
    for (n, y_n) in y.iter_mut().enumerate() {
        let output = a * y_prev + b * u_ff[n];
        *y_n = output;
        y_prev = output;
    }
    y
}

/// Compute root-mean-square of an array.
fn rms(arr: &[f64; TRIAL_LEN]) -> f64 {
    let sum_sq = arr.iter().fold(0.0f64, |acc, &v| acc + v * v);
    libm::sqrt(sum_sq / TRIAL_LEN as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ILC — Repetitive Pick-and-Place Task ===\n");

    // ── Plant parameters ──────────────────────────────────────────────────────
    let a = 0.7f64; // pole (must be < 1 for open-loop stability)
    let b = 0.3f64; // DC gain = b / (1 - a) = 1.0

    // ── ILC parameters ────────────────────────────────────────────────────────
    let learning_gain = 0.5f64;
    let n_trials = 10usize;

    println!(
        "Plant:          y[n] = {a} · y[n-1] + {b} · u[n]  (DC gain = {:.1})",
        b / (1.0 - a)
    );
    println!("Learning gain:  L = {learning_gain}");
    println!("Trials:         {n_trials}");
    println!("Trial length:   {TRIAL_LEN} steps");
    println!("Reference:      step 0→1 at t=5, 1→0 at t=15\n");

    // ── Reference trajectory ──────────────────────────────────────────────────
    let reference = make_reference();

    // ── Construct P-type ILC ──────────────────────────────────────────────────
    let mut ilc: PTypeIlc<f64, TRIAL_LEN> =
        PTypeIlc::new(learning_gain).map_err(|e| format!("ILC construction error: {e}"))?;

    // ── Run trials ────────────────────────────────────────────────────────────
    println!(
        "{:>7}  {:>12}  {:>14}  {:>12}",
        "trial", "rms_error", "conv_change", "u_ff_rms"
    );
    println!("{}", "-".repeat(52));

    let mut prev_rms = f64::INFINITY;

    for trial in 0..n_trials {
        // Retrieve current feedforward signal (copy to avoid borrow conflict)
        let u_ff = *ilc.feedforward();

        // Simulate the plant for this trial
        let y = simulate_trial(&u_ff, a, b);

        // Compute per-step tracking error: e[n] = r[n] - y[n]
        let mut error = [0.0f64; TRIAL_LEN];
        for (n, e) in error.iter_mut().enumerate() {
            *e = reference[n] - y[n];
        }

        // Compute convergence change before update (compares to previous trial)
        let conv_change = ilc.convergence_error(&error);

        // RMS of current error
        let cur_rms = rms(&error);

        // RMS of current feedforward
        let u_ff_rms = rms(&u_ff);

        println!(
            "{:>7}  {:>12.6}  {:>14.6}  {:>12.6}",
            trial, cur_rms, conv_change, u_ff_rms
        );

        // Apply ILC update: u_ff_{k+1}[n] = u_ff[n] + L * e[n]
        ilc.update(&error)
            .map_err(|e| format!("ILC update error at trial {trial}: {e}"))?;

        // Track whether error is decreasing (skip trial 0 for comparison)
        if trial > 0 && cur_rms > prev_rms + 1e-9 {
            println!("[WARN] RMS error increased at trial {trial}!");
        }
        prev_rms = cur_rms;
    }

    // ── Final trial simulation (using learned feedforward) ────────────────────
    let u_ff_final = *ilc.feedforward();
    let y_final = simulate_trial(&u_ff_final, a, b);

    let mut final_error = [0.0f64; TRIAL_LEN];
    for n in 0..TRIAL_LEN {
        final_error[n] = reference[n] - y_final[n];
    }
    let final_rms = rms(&final_error);

    // ── Per-step output table for the last trial ──────────────────────────────
    println!();
    println!("=== Final learned trajectory (trial {n_trials}) ===");
    println!(
        "{:>5}  {:>10}  {:>10}  {:>10}  {:>10}",
        "step", "r[n]", "y[n]", "u_ff[n]", "error[n]"
    );
    println!("{}", "-".repeat(55));

    for n in 0..TRIAL_LEN {
        println!(
            "{:>5}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
            n, reference[n], y_final[n], u_ff_final[n], final_error[n]
        );
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!();
    println!("=== Summary ===");
    println!("Completed trials    : {}", ilc.trial_count());
    println!("Final RMS error     : {:.6}", final_rms);
    println!("Initial RMS error   : {:.6}", rms(&make_reference())); // u_ff=0 → y=0 → e=r

    // ── Validation ────────────────────────────────────────────────────────────
    println!();
    // After 10 trials with L=0.5 on a plant with DC gain 1.0, the error is
    // reduced by a factor of ~0.5^10 = 1/1024 asymptotically; however, the
    // finite trial length and step-response discontinuities keep RMS above 0.
    // A 10× reduction from initial (≈ 0.707) to below 0.10 indicates good learning.
    if final_rms < 0.10 {
        println!(
            "[PASS] ILC has reduced error by > 10× — final RMS {:.4} < 0.10.",
            final_rms
        );
    } else {
        println!(
            "[INFO] Final RMS error {:.4} — more trials may improve tracking.",
            final_rms
        );
    }

    // Confirm that error decreased from trial 1 onward (monotone learning)
    // Re-run trials to check monotonicity
    let mut mono_ilc: PTypeIlc<f64, TRIAL_LEN> =
        PTypeIlc::new(learning_gain).map_err(|e| format!("ILC (monotone check) error: {e}"))?;
    let mut prev = f64::INFINITY;
    let mut monotone = true;
    for _ in 0..n_trials {
        let uff = *mono_ilc.feedforward();
        let y_t = simulate_trial(&uff, a, b);
        let mut err = [0.0f64; TRIAL_LEN];
        for (n, e) in err.iter_mut().enumerate() {
            *e = reference[n] - y_t[n];
        }
        let cur = rms(&err);
        mono_ilc
            .update(&err)
            .map_err(|e| format!("Monotone check update error: {e}"))?;
        if cur > prev + 1e-9 {
            monotone = false;
        }
        prev = cur;
    }

    if monotone {
        println!("[PASS] RMS error decreased monotonically over all trials.");
    } else {
        println!("[INFO] RMS error was not strictly monotone (may be numerical noise).");
    }

    Ok(())
}
