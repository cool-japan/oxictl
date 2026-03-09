//! Instrumental Variables (IV) identification.
#![allow(
    clippy::needless_range_loop,
    clippy::doc_overindented_list_items,
    clippy::manual_memcpy
)]
//!
//! Ordinary Least Squares (ARX) produces biased estimates when noise is correlated
//! with the output regressors. The IV method achieves consistent (asymptotically
//! unbiased) estimates by replacing the regressor matrix Φ with an instrument matrix Z
//! whose columns are correlated with the regressors but uncorrelated with noise.
//!
//! # Algorithm
//!
//! Two-stage IV (basic):
//! 1. Stage 1 — ARX LS: θ̂_LS = (ΦᵀΦ)⁻¹Φᵀy.
//!    Simulate the LS model to obtain noise-free output predictions ŷ(t).
//! 2. Stage 2 — IV: Replace output regressors in Z with lagged ŷ.
//!    θ̂_IV = (ZᵀΦ)⁻¹Zᵀy.
//!
//! Refined IV (RIV) iterates Stage 2 using the latest model to re-generate
//! instruments until ‖Δθ‖₂ < `tol`.
//!
//! # Const parameters
//! - `NA`  — AR order.
//! - `NB`  — exogenous order.
//! - `P`   — regressor dimension = NA + NB (**caller must ensure P == NA + NB**).
//! - `NK`  — dead time (samples).

use crate::core::scalar::ControlScalar;
use crate::sysid::arx::{cholesky_solve_n, ArxModel};
use crate::sysid::SysIdError;

// ── Gaussian elimination with partial pivoting ────────────────────────────────

/// Solve the (possibly non-symmetric) linear system A·x = b via Gaussian
/// elimination with partial pivoting.
///
/// `a` and `b` are modified in place. Returns `Err(SingularMatrix)` if singular.
fn gauss_solve<S: ControlScalar, const N: usize>(
    a: &mut [[S; N]; N],
    b: &mut [S; N],
) -> Result<(), SysIdError> {
    for col in 0..N {
        // Partial pivot
        let mut max_val = a[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..N {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val == S::ZERO {
            return Err(SysIdError::SingularMatrix);
        }
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        let pivot = a[col][col];
        for row in (col + 1)..N {
            let factor = a[row][col] / pivot;
            for k in col..N {
                let val = a[col][k];
                a[row][k] -= factor * val;
            }
            let bval = b[col];
            b[row] -= factor * bval;
        }
    }

    // Back substitution
    let mut i = N;
    while i > 0 {
        i -= 1;
        let mut s = b[i];
        for k in (i + 1)..N {
            s -= a[i][k] * b[k];
        }
        if a[i][i] == S::ZERO {
            return Err(SysIdError::SingularMatrix);
        }
        b[i] = s / a[i][i];
    }

    Ok(())
}

// ── Regressor / instrument builders ──────────────────────────────────────────

/// Build the ARX regressor φ(t) ∈ R^P at time index `t`.
///
/// Layout: `[-y(t-1), …, -y(t-NA), u(t-NK), …, u(t-NK-NB+1)]`
fn build_phi<
    S: ControlScalar,
    const NA: usize,
    const NB: usize,
    const P: usize,
    const NK: usize,
>(
    t: usize,
    y: &[S],
    u: &[S],
) -> [S; P] {
    let mut phi = [S::ZERO; P];
    for i in 0..NA {
        if t > i {
            phi[i] = -y[t - 1 - i];
        }
    }
    for i in 0..NB {
        let delay = NK + i;
        if t >= delay {
            phi[NA + i] = u[t - delay];
        }
    }
    phi
}

/// Build the instrument vector z(t) ∈ R^P at time index `t`.
///
/// Replaces the output regressors with model-predicted (noise-free) outputs `y_hat`.
fn build_z<S: ControlScalar, const NA: usize, const NB: usize, const P: usize, const NK: usize>(
    t: usize,
    y_hat: &[S],
    u: &[S],
) -> [S; P] {
    let mut z = [S::ZERO; P];
    for i in 0..NA {
        if t > i {
            z[i] = -y_hat[t - 1 - i];
        }
    }
    for i in 0..NB {
        let delay = NK + i;
        if t >= delay {
            z[NA + i] = u[t - delay];
        }
    }
    z
}

/// Simulate the ARX model over the data set to obtain noise-free predicted outputs.
///
/// Uses a one-step-ahead predictor based on model-simulated outputs (open-loop
/// on the model's own predictions), bootstrapped from actual data for the first `NA`
/// samples.
fn simulate_model<S: ControlScalar, const NA: usize, const NB: usize>(
    model: &ArxModel<S, NA, NB>,
    y: &[S],
    u: &[S],
    n: usize,
    y_hat_out: &mut [S],
) {
    let mut y_buf = [S::ZERO; NA];
    // Bootstrap from actual data
    for i in 0..NA {
        if NA > 0 && NA > i {
            let idx = NA - 1 - i;
            if idx < n {
                y_buf[i] = y[idx];
            }
        }
    }

    for t in 0..n {
        // u history for u(t-1)…u(t-NB) (NK=1 assumed for instrument building)
        let mut u_hist = [S::ZERO; NB];
        for i in 0..NB {
            if t > i {
                u_hist[i] = u[t - 1 - i];
            }
        }
        let y_new = model.predict(&y_buf, &u_hist);
        y_hat_out[t] = y_new;

        // Shift y_buf newest-first
        let mut j = NA;
        while j > 1 {
            y_buf[j - 1] = y_buf[j - 2];
            j -= 1;
        }
        if NA > 0 {
            y_buf[0] = y_new;
        }
    }
}

// ── IvIdentifier ─────────────────────────────────────────────────────────────

/// Basic two-stage Instrumental Variables estimator.
///
/// # Const parameters
/// - `NA`, `NB` — polynomial orders.
/// - `P`  — regressor dimension = NA + NB.
/// - `NK` — dead time.
pub struct IvIdentifier<
    S: ControlScalar,
    const NA: usize,
    const NB: usize,
    const P: usize,
    const NK: usize,
> {
    _phantom: core::marker::PhantomData<S>,
}

impl<S: ControlScalar, const NA: usize, const NB: usize, const P: usize, const NK: usize>
    IvIdentifier<S, NA, NB, P, NK>
{
    const BUF_CAP: usize = 4096;

    /// Fit an ARX model using the two-stage IV method.
    ///
    /// # Arguments
    /// * `y` — output sequence.
    /// * `u` — input sequence (same length as `y`).
    /// * `max_iter` — IV iterations (1 = basic IV, >1 = iterated IV).
    pub fn fit_iv(y: &[S], u: &[S], max_iter: usize) -> Result<ArxModel<S, NA, NB>, SysIdError> {
        let n = y.len();
        if u.len() != n {
            return Err(SysIdError::InvalidData);
        }
        for &v in y.iter().chain(u.iter()) {
            if !v.is_finite() {
                return Err(SysIdError::InvalidData);
            }
        }

        let min_start = NA.max(NK + NB);
        if n <= min_start {
            return Err(SysIdError::InsufficientData);
        }

        if P == 0 {
            return Ok(ArxModel::zeros());
        }

        // ── Stage 1: ARX LS ──────────────────────────────────────────────────
        let mut ata_ls = [[S::ZERO; P]; P];
        let mut aty_ls = [S::ZERO; P];
        for t in min_start..n {
            let phi = build_phi::<S, NA, NB, P, NK>(t, y, u);
            for i in 0..P {
                for j in 0..P {
                    ata_ls[i][j] += phi[i] * phi[j];
                }
                aty_ls[i] += phi[i] * y[t];
            }
        }
        cholesky_solve_n::<S, P>(&mut ata_ls, &mut aty_ls)?;

        let mut model = ArxModel::zeros();
        for i in 0..NA {
            model.a[i] = aty_ls[i];
        }
        for i in 0..NB {
            model.b[i] = aty_ls[NA + i];
        }

        if max_iter == 0 {
            return Ok(model);
        }

        // ── Stage 2+: IV iterations ──────────────────────────────────────────
        let n_capped = n.min(Self::BUF_CAP);
        let mut y_hat = [S::ZERO; 4096];

        for _iter in 0..max_iter {
            simulate_model::<S, NA, NB>(&model, y, u, n_capped, &mut y_hat);

            let mut zta_phi = [[S::ZERO; P]; P];
            let mut zty = [S::ZERO; P];

            for t in min_start..n_capped {
                let phi = build_phi::<S, NA, NB, P, NK>(t, y, u);
                let z = build_z::<S, NA, NB, P, NK>(t, &y_hat, u);
                for i in 0..P {
                    for j in 0..P {
                        zta_phi[i][j] += z[i] * phi[j];
                    }
                    zty[i] += z[i] * y[t];
                }
            }

            gauss_solve::<S, P>(&mut zta_phi, &mut zty)?;

            let mut new_model = ArxModel::zeros();
            for i in 0..NA {
                new_model.a[i] = zty[i];
            }
            for i in 0..NB {
                new_model.b[i] = zty[NA + i];
            }
            model = new_model;
        }

        Ok(model)
    }
}

// ── RefIvIdentifier ───────────────────────────────────────────────────────────

/// Refined Instrumental Variables (RIV) estimator.
///
/// Iterates the IV step until ‖Δθ‖₂ < `tol` or `max_iter` is reached.
///
/// # Const parameters
/// - `NA`, `NB` — polynomial orders.
/// - `P`  — regressor dimension = NA + NB.
/// - `NK` — dead time.
pub struct RefIvIdentifier<
    S: ControlScalar,
    const NA: usize,
    const NB: usize,
    const P: usize,
    const NK: usize,
> {
    _phantom: core::marker::PhantomData<S>,
}

impl<S: ControlScalar, const NA: usize, const NB: usize, const P: usize, const NK: usize>
    RefIvIdentifier<S, NA, NB, P, NK>
{
    const BUF_CAP: usize = 4096;

    /// Fit an ARX model using Refined IV with convergence checking.
    pub fn fit_riv(
        y: &[S],
        u: &[S],
        max_iter: usize,
        tol: S,
    ) -> Result<ArxModel<S, NA, NB>, SysIdError> {
        let n = y.len();
        if u.len() != n {
            return Err(SysIdError::InvalidData);
        }
        for &v in y.iter().chain(u.iter()) {
            if !v.is_finite() {
                return Err(SysIdError::InvalidData);
            }
        }

        let min_start = NA.max(NK + NB);
        if n <= min_start {
            return Err(SysIdError::InsufficientData);
        }

        if P == 0 {
            return Ok(ArxModel::zeros());
        }

        // Stage 1: ARX LS
        let mut ata_ls = [[S::ZERO; P]; P];
        let mut aty_ls = [S::ZERO; P];
        for t in min_start..n {
            let phi = build_phi::<S, NA, NB, P, NK>(t, y, u);
            for i in 0..P {
                for j in 0..P {
                    ata_ls[i][j] += phi[i] * phi[j];
                }
                aty_ls[i] += phi[i] * y[t];
            }
        }
        cholesky_solve_n::<S, P>(&mut ata_ls, &mut aty_ls)?;

        let mut theta = aty_ls;
        let mut model = ArxModel::zeros();
        for i in 0..NA {
            model.a[i] = theta[i];
        }
        for i in 0..NB {
            model.b[i] = theta[NA + i];
        }

        let n_capped = n.min(Self::BUF_CAP);
        let mut y_hat = [S::ZERO; 4096];
        let mut converged = false;

        for _iter in 0..max_iter {
            simulate_model::<S, NA, NB>(&model, y, u, n_capped, &mut y_hat);

            let mut zta_phi = [[S::ZERO; P]; P];
            let mut zty = [S::ZERO; P];
            for t in min_start..n_capped {
                let phi = build_phi::<S, NA, NB, P, NK>(t, y, u);
                let z = build_z::<S, NA, NB, P, NK>(t, &y_hat, u);
                for i in 0..P {
                    for j in 0..P {
                        zta_phi[i][j] += z[i] * phi[j];
                    }
                    zty[i] += z[i] * y[t];
                }
            }
            gauss_solve::<S, P>(&mut zta_phi, &mut zty)?;

            let mut norm_sq = S::ZERO;
            for i in 0..P {
                let d = zty[i] - theta[i];
                norm_sq += d * d;
            }
            theta = zty;

            let mut new_model = ArxModel::zeros();
            for i in 0..NA {
                new_model.a[i] = theta[i];
            }
            for i in 0..NB {
                new_model.b[i] = theta[NA + i];
            }
            model = new_model;

            if norm_sq.sqrt() < tol {
                converged = true;
                break;
            }
        }

        if !converged && max_iter > 1 {
            return Err(SysIdError::NotConverged);
        }

        Ok(model)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sysid::arx::{fit_percent, ArxIdentifier};

    fn generate_arx1_noiseless(n: usize) -> (heapless::Vec<f64, 4096>, heapless::Vec<f64, 4096>) {
        let a_true = 0.7_f64;
        let b_true = 0.3_f64;
        let mut y: heapless::Vec<f64, 4096> = heapless::Vec::new();
        let mut u: heapless::Vec<f64, 4096> = heapless::Vec::new();
        let _ = y.push(0.0);
        let _ = u.push(0.0);
        for t in 1..n {
            let ut = libm::sin(0.1 * t as f64) + libm::cos(0.07 * t as f64) * 0.5;
            let yt = a_true * y[t - 1] + b_true * u[t - 1];
            let _ = y.push(yt);
            let _ = u.push(ut);
        }
        (y, u)
    }

    fn generate_noisy_fir(
        n: usize,
        noise_std: f64,
    ) -> (heapless::Vec<f64, 4096>, heapless::Vec<f64, 4096>) {
        let b_true = 0.3_f64;
        let mut y: heapless::Vec<f64, 4096> = heapless::Vec::new();
        let mut u: heapless::Vec<f64, 4096> = heapless::Vec::new();
        let _ = y.push(0.0);
        let _ = u.push(0.0);
        let mut lcg: u64 = 12345;
        for t in 1..n {
            let ut = libm::sin(0.1 * t as f64) + libm::cos(0.07 * t as f64) * 0.5;
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((lcg >> 33) as f64 / (u64::MAX >> 33) as f64 - 0.5) * 2.0 * noise_std;
            let yt = b_true * u[t - 1] + noise;
            let _ = y.push(yt);
            let _ = u.push(ut);
        }
        (y, u)
    }

    /// On noiseless ARX data, IV and LS should give very similar results.
    #[test]
    fn iv_matches_ls_on_noiseless_data() {
        let (y, u) = generate_arx1_noiseless(600);

        // NA=1, NB=1, P=2, NK=1
        let ls_model =
            ArxIdentifier::<f64, 1, 1, 2, 1>::fit(y.as_slice(), u.as_slice()).expect("LS fit");
        let iv_model =
            IvIdentifier::<f64, 1, 1, 2, 1>::fit_iv(y.as_slice(), u.as_slice(), 3).expect("IV fit");

        let mut pred_ls: heapless::Vec<f64, 4096> = heapless::Vec::new();
        let mut pred_iv: heapless::Vec<f64, 4096> = heapless::Vec::new();
        for t in 1..y.len() {
            let y_hist = [y[t - 1]];
            let u_hist = [u[t - 1]];
            let _ = pred_ls.push(ls_model.predict(&y_hist, &u_hist));
            let _ = pred_iv.push(iv_model.predict(&y_hist, &u_hist));
        }
        let fp_ls = fit_percent(pred_ls.as_slice(), &y.as_slice()[1..]);
        let fp_iv = fit_percent(pred_iv.as_slice(), &y.as_slice()[1..]);
        assert!(fp_ls > 99.0, "LS FIT% {fp_ls:.2}");
        assert!(fp_iv > 99.0, "IV FIT% {fp_iv:.2}");
    }

    /// On noisy data, IV should produce a valid (finite) estimate.
    #[test]
    fn iv_gives_finite_estimate_on_noisy_data() {
        let (y, u) = generate_noisy_fir(800, 0.05);
        let model = IvIdentifier::<f64, 1, 1, 2, 1>::fit_iv(y.as_slice(), u.as_slice(), 5)
            .expect("IV fit on noisy data");
        assert!(model.a[0].is_finite());
        assert!(model.b[0].is_finite());
    }

    /// Refined IV converges within tolerance on noiseless data.
    #[test]
    fn riv_converges_on_noiseless_data() {
        let (y, u) = generate_arx1_noiseless(600);
        let model =
            RefIvIdentifier::<f64, 1, 1, 2, 1>::fit_riv(y.as_slice(), u.as_slice(), 20, 1e-10)
                .expect("RIV fit should converge");

        let mut pred: heapless::Vec<f64, 4096> = heapless::Vec::new();
        for t in 1..y.len() {
            let y_hist = [y[t - 1]];
            let u_hist = [u[t - 1]];
            let _ = pred.push(model.predict(&y_hist, &u_hist));
        }
        let fp = fit_percent(pred.as_slice(), &y.as_slice()[1..]);
        assert!(fp > 99.0, "RIV FIT% {fp:.2}");
    }

    /// IV on clearly insufficient data returns an error.
    #[test]
    fn iv_insufficient_data_returns_error() {
        let y = [1.0_f64, 2.0];
        let u = [0.5_f64, 0.3];
        // NA=2, NB=2, P=4, NK=1
        let result = IvIdentifier::<f64, 2, 2, 4, 1>::fit_iv(&y, &u, 1);
        assert!(
            matches!(result, Err(SysIdError::InsufficientData)),
            "expected InsufficientData, got {:?}",
            result.err()
        );
    }
}
