//! Gaussian Process regression learning a 1-D nonlinear function.
//!
//! True function:  f(x) = sin(2π x) · exp(−x / 3)
//!
//! Eight training points are placed uniformly in [0, 3].  An RBF (squared
//! exponential) kernel with variance σ² = 1.0 and length-scale l = 0.5 is
//! used together with observation noise variance σ²_n = 0.01.
//!
//! After fitting, the GP is queried at five test points.  Within the training
//! range the posterior mean should closely track the true function; outside
//! the range the predictive standard deviation (uncertainty) grows.
//!
//! Run with:
//!   cargo run --example gp_learning --features "gp"

use oxictl::gp::{GpRegression, RbfKernel};

/// True 1-D function: f(x) = sin(2π x) · exp(−x / 3).
#[inline]
fn true_f(x: f64) -> f64 {
    libm::sin(2.0 * core::f64::consts::PI * x) * libm::exp(-x / 3.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Gaussian Process Regression: f(x) = sin(2πx)·exp(-x/3) ===\n");

    // ── Kernel and GP setup ───────────────────────────────────────────────────
    // RBF kernel: k(x,x') = σ² · exp(−‖x−x'‖² / (2 l²))
    let kernel = RbfKernel::<f64> {
        variance: 1.0,
        length_scale: 0.5,
    };
    let noise_var: f64 = 0.01;

    // GpRegression<S, K, D, N>: scalar f64, RbfKernel, D=1 input dim, N=8 training points
    let mut gp: GpRegression<f64, RbfKernel<f64>, 1, 8> = GpRegression::new(kernel, noise_var);

    // ── Training data: 8 evenly spaced points in [0, 3] ─────────────────────
    // x_i = i · (3.0 / 7.0) for i in 0..8
    let step = 3.0_f64 / 7.0;
    let x_train: [[f64; 1]; 8] = core::array::from_fn(|i| [i as f64 * step]);
    let y_train: [f64; 8] = core::array::from_fn(|i| true_f(x_train[i][0]));

    println!("Training points (x, y_true):");
    for i in 0..8 {
        println!("  x = {:.4}   y = {:.6}", x_train[i][0], y_train[i]);
    }
    println!();

    // ── Fit ───────────────────────────────────────────────────────────────────
    gp.fit(x_train, y_train)
        .map_err(|e| format!("GP fit error: {e}"))?;
    println!("GP fitted successfully.  is_trained = {}", gp.is_trained());

    let lml = gp
        .log_marginal_likelihood()
        .map_err(|e| format!("LML error: {e}"))?;
    println!("Log marginal likelihood: {:.4}\n", lml);

    // ── Predict at 5 test points ──────────────────────────────────────────────
    // Mix of interpolation points (inside [0,3]) and one extrapolation (x=4).
    let test_xs: [f64; 5] = [0.25, 0.75, 1.5, 2.25, 4.0];

    println!(
        "{:>8}  {:>12}  {:>14}  {:>12}  {:>10}",
        "x", "true f(x)", "pred mean", "std dev", "error"
    );
    println!("{}", "-".repeat(62));

    for &x in &test_xs {
        let (mean, var) = gp
            .predict(&[x])
            .map_err(|e| format!("GP predict error: {e}"))?;
        let std_dev = libm::sqrt(var);
        let f_true = true_f(x);
        let err = (mean - f_true).abs();
        println!(
            "{:>8.4}  {:>12.6}  {:>14.6}  {:>12.6}  {:>10.6}",
            x, f_true, mean, std_dev, err
        );
    }

    println!("\n=== Summary ===");
    println!("Kernel: RBF (variance=1.0, length_scale=0.5), noise_var=0.01");
    println!("Training points: 8 in [0, 3]");

    // Verify interpolation quality at interior points (first 4 test points)
    let mut max_interp_err: f64 = 0.0;
    for &x in &test_xs[..4] {
        let (mean, _var) = gp
            .predict(&[x])
            .map_err(|e| format!("GP predict error: {e}"))?;
        let err = (mean - true_f(x)).abs();
        if err > max_interp_err {
            max_interp_err = err;
        }
    }
    println!(
        "Max interpolation error (interior points): {:.6}",
        max_interp_err
    );

    // Verify extrapolation uncertainty is larger than interpolation uncertainty
    let (_m_in, var_in) = gp
        .predict(&[1.5_f64])
        .map_err(|e| format!("GP predict error: {e}"))?;
    let (_m_out, var_out) = gp
        .predict(&[4.0_f64])
        .map_err(|e| format!("GP predict error: {e}"))?;
    println!(
        "Predictive std dev at x=1.5 (interior): {:.6}",
        libm::sqrt(var_in)
    );
    println!(
        "Predictive std dev at x=4.0 (exterior): {:.6}",
        libm::sqrt(var_out)
    );

    if var_out > var_in {
        println!("[PASS] Uncertainty correctly higher outside training range.");
    } else {
        println!("[INFO] Uncertainty check inconclusive (may depend on kernel params).");
    }

    if max_interp_err < 0.1 {
        println!("[PASS] Good interpolation within training range (error < 0.10).");
    } else {
        println!("[INFO] Interpolation error > 0.10; consider tuning kernel hyperparameters.");
    }

    Ok(())
}
