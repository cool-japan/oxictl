//! Koopman operator identification for a nonlinear pendulum.
//!
//! System: θ̈ = −sin(θ) − 0.1·θ̇ + u  (Euler-discretized, dt = 0.05 s)
//!
//! Lifting map: PolynomialLifting<f64, 2, 5>
//!   ψ([θ, θ̇]) = [θ, θ̇, θ², θ·θ̇, θ̇²]   (2 linear + 3 quadratic terms)
//!
//! Procedure:
//! 1. Simulate a free-decay trajectory (u = 0) from θ₀ = 0.5 rad to collect
//!    30 consecutive lifted snapshot pairs.
//! 2. Fit an EDMD model (Koopman matrix K ∈ ℝ^{5×5}) to those pairs.
//! 3. Evaluate training reconstruction error (should be small).
//! 4. Predict the next lifted state for a separate test point and compare the
//!    first two components (θ, θ̇) against the true Euler step.
//!
//! Run: `cargo run --example koopman_pendulum --features koopman`

use oxictl::koopman::{Edmd, PolynomialLifting};

// ── Pendulum constants ────────────────────────────────────────────────────────

const DT: f64 = 0.05;
const DAMPING: f64 = 0.1;

// ── Euler step for the nonlinear pendulum ─────────────────────────────────────

/// One Euler step:  [θ, θ̇] → [θ + dt·θ̇,  θ̇ + dt·(−sin θ − d·θ̇ + u)]
fn pendulum_step(theta: f64, theta_dot: f64, u: f64) -> (f64, f64) {
    let theta_ddot = -libm::sin(theta) - DAMPING * theta_dot + u;
    (theta + DT * theta_dot, theta_dot + DT * theta_ddot)
}

// ── Lifting helper ────────────────────────────────────────────────────────────

/// Lift a 2-D pendulum state to the 5-D polynomial feature space.
fn lift_state(lifter: &PolynomialLifting<f64, 2, 5>, theta: f64, theta_dot: f64) -> [f64; 5] {
    lifter.lift(&[theta, theta_dot])
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<(), String> {
    println!("=== Koopman Pendulum Example ===");
    println!("System: θ̈ = −sin(θ) − {DAMPING}·θ̇ + u,  dt = {DT} s");
    println!("Lifting: ψ(θ, θ̇) = [θ, θ̇, θ², θ·θ̇, θ̇²]  (L = 5)");
    println!();

    // ── Build lifting map ─────────────────────────────────────────────────────

    let lifter: PolynomialLifting<f64, 2, 5> = PolynomialLifting::new();

    // ── Collect 30 training snapshot pairs (free decay, u = 0) ───────────────

    const DATA: usize = 30;
    let mut psi_x: [[f64; 5]; DATA] = [[0.0; 5]; DATA];
    let mut psi_x_next: [[f64; 5]; DATA] = [[0.0; 5]; DATA];

    let (mut theta, mut theta_dot) = (0.5_f64, 0.0_f64);

    for k in 0..DATA {
        psi_x[k] = lift_state(&lifter, theta, theta_dot);
        let (th_next, thd_next) = pendulum_step(theta, theta_dot, 0.0);
        psi_x_next[k] = lift_state(&lifter, th_next, thd_next);
        theta = th_next;
        theta_dot = thd_next;
    }

    println!("Collected {DATA} snapshot pairs along a free-decay trajectory.");
    println!("  Initial state: θ₀ = 0.5 rad, θ̇₀ = 0 rad/s");
    println!(
        "  Final  state:  θ = {:.4} rad, θ̇ = {:.4} rad/s",
        theta, theta_dot
    );
    println!();

    // ── Fit EDMD ──────────────────────────────────────────────────────────────

    let mut edmd: Edmd<f64, 5, 30> = Edmd::new();
    edmd.fit(&psi_x, &psi_x_next)
        .map_err(|e| format!("EDMD fit failed: {e}"))?;

    println!("EDMD fitting complete.  is_fitted = {}", edmd.is_fitted());

    // ── Training reconstruction error ─────────────────────────────────────────

    let train_err = edmd
        .reconstruction_error(&psi_x, &psi_x_next)
        .map_err(|e| format!("Reconstruction error: {e}"))?;

    println!("Training reconstruction MSE = {train_err:.6e}");

    // ── Test-point prediction ─────────────────────────────────────────────────

    let test_theta = 0.3_f64;
    let test_theta_dot = 0.1_f64;
    let psi_test = lift_state(&lifter, test_theta, test_theta_dot);

    let psi_pred = edmd
        .predict(&psi_test)
        .map_err(|e| format!("Prediction failed: {e}"))?;

    // True next state by Euler integration
    let (true_next_theta, true_next_theta_dot) = pendulum_step(test_theta, test_theta_dot, 0.0);
    let psi_true = lift_state(&lifter, true_next_theta, true_next_theta_dot);

    let err_theta = (psi_pred[0] - psi_true[0]).abs();
    let err_theta_dot = (psi_pred[1] - psi_true[1]).abs();

    println!();
    println!("Test point: θ = {test_theta:.2} rad, θ̇ = {test_theta_dot:.2} rad/s");
    println!(
        "  True  next: θ = {:.5}, θ̇ = {:.5}",
        true_next_theta, true_next_theta_dot
    );
    println!(
        "  Koopman pred: θ = {:.5}, θ̇ = {:.5}",
        psi_pred[0], psi_pred[1]
    );
    println!("  |err θ|   = {err_theta:.4e}");
    println!("  |err θ̇|   = {err_theta_dot:.4e}");
    println!();

    // ── Koopman eigenvalue proxy (diagonal of K) ──────────────────────────────

    let ev = edmd.eigenvalues_real_part();
    println!("Koopman K diagonal (eigenvalue proxy):");
    for (i, v) in ev.iter().enumerate() {
        println!("  K[{i},{i}] = {v:.6}");
    }
    println!();

    // ── Summary ───────────────────────────────────────────────────────────────

    let total_test_err = err_theta + err_theta_dot;
    println!("=== Summary ===");
    println!("Training MSE  : {train_err:.4e}  (should be small for polynomial system)");
    println!("Test |err|    : {total_test_err:.4e}  (θ + θ̇ components combined)");

    if train_err < 1e-3 && total_test_err < 0.1 {
        println!("PASS: Koopman lifting captures nonlinear pendulum dynamics accurately.");
    } else {
        println!(
            "WARN: Errors larger than expected (train={train_err:.2e}, test={total_test_err:.2e})."
        );
        println!("      The polynomial basis may need more terms for large-angle dynamics.");
    }

    Ok(())
}
