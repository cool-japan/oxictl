//! ARMAX (AutoRegressive Moving Average with eXogenous input) model identification.
#![allow(
    clippy::needless_range_loop,
    clippy::doc_overindented_list_items,
    clippy::doc_lazy_continuation,
    clippy::manual_memcpy
)]
//!
//! The ARMAX model in shift-operator form is:
//! ```text
//!   A(q)·y(t) = B(q)·u(t) + C(q)·e(t)
//! ```
//! where
//! ```text
//!   A(q) = 1 + a₁q⁻¹ + … + aₙₐq⁻ⁿᵃ
//!   B(q) = b₀q⁻¹ + … + bₙᵦ₋₁q⁻ⁿᵇ
//!   C(q) = 1 + c₁q⁻¹ + … + cₙ꜀q⁻ⁿ꜀
//! ```
//!
//! The C polynomial models the colored noise component. Extended Least Squares (ELS)
//! corrects the bias of ordinary ARX least squares by iteratively refining the
//! regression with past residual estimates.
//!
//! # Algorithm: Extended Least Squares (ELS)
//!
//! ELS iterates the following two steps:
//! 1. Stage A/B/C: Regressor φ(t) = [-y(t-1)…-y(t-NA), u(t-1)…u(t-NB), ê(t-1)…ê(t-NC)]
//!    Solve normal equations → θ = [a, b, c].
//! 2. Update residuals: ê(t) = y(t) − φ(t)ᵀ·θ.
//! Repeat until ‖Δθ‖₂ < `tol`.
//!
//! # Const parameters
//! - `NA` — AR order.
//! - `NB` — exogenous order.
//! - `NC` — moving-average order.
//! - `P`  — total regressor dimension = NA + NB + NC (**caller must ensure P == NA + NB + NC**).

use crate::core::scalar::ControlScalar;
use crate::sysid::arx::cholesky_solve_n;
use crate::sysid::SysIdError;

// ── ArmaxModel ────────────────────────────────────────────────────────────────

/// Identified ARMAX model.
///
/// Stores coefficients for all three polynomials A, B, C.
/// The one-step-ahead prediction given past outputs, inputs, and residuals is:
/// ```text
///   ŷ(t) = −a₀·y(t-1) − … − aₙₐ₋₁·y(t-NA)
///          + b₀·u(t-1) + … + bₙᵦ₋₁·u(t-NB)
///          + c₀·e(t-1) + … + cₙ꜀₋₁·e(t-NC)
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ArmaxModel<S: ControlScalar, const NA: usize, const NB: usize, const NC: usize> {
    /// AR coefficients (A polynomial, excluding leading 1).
    pub a: [S; NA],
    /// Exogenous coefficients (B polynomial).
    pub b: [S; NB],
    /// MA coefficients (C polynomial, excluding leading 1).
    pub c: [S; NC],
}

impl<S: ControlScalar, const NA: usize, const NB: usize, const NC: usize>
    ArmaxModel<S, NA, NB, NC>
{
    /// Create a zero-initialised model.
    pub fn zeros() -> Self {
        Self {
            a: [S::ZERO; NA],
            b: [S::ZERO; NB],
            c: [S::ZERO; NC],
        }
    }

    /// One-step-ahead prediction using past outputs, inputs, and residuals.
    ///
    /// All history slices are **newest first**:
    /// - `y_hist[0] = y(t-1)`, …
    /// - `u_hist[0] = u(t-1)`, …
    /// - `e_hist[0] = e(t-1)`, …
    pub fn predict_one_step(&self, y_hist: &[S; NA], u_hist: &[S; NB], e_hist: &[S; NC]) -> S {
        let mut y_hat = S::ZERO;
        for i in 0..NA {
            y_hat -= self.a[i] * y_hist[i];
        }
        for i in 0..NB {
            y_hat += self.b[i] * u_hist[i];
        }
        for i in 0..NC {
            y_hat += self.c[i] * e_hist[i];
        }
        y_hat
    }
}

// ── ELSIdentifier ─────────────────────────────────────────────────────────────

/// Extended Least Squares identifier for ARMAX models.
///
/// # Const parameters
/// - `NA`, `NB`, `NC` — polynomial orders.
/// - `P` — total regressor dimension = NA + NB + NC (**must equal NA + NB + NC**).
pub struct ELSIdentifier<
    S: ControlScalar,
    const NA: usize,
    const NB: usize,
    const NC: usize,
    const P: usize,
> {
    _phantom: core::marker::PhantomData<S>,
}

impl<S: ControlScalar, const NA: usize, const NB: usize, const NC: usize, const P: usize>
    ELSIdentifier<S, NA, NB, NC, P>
{
    /// Maximum number of data points processed in the normal-equations accumulation.
    const BUF_CAP: usize = 4096;

    /// Fit an ARMAX model to the provided data using Extended Least Squares.
    ///
    /// # Arguments
    /// * `y` — output sequence.
    /// * `u` — input sequence (same length as `y`).
    /// * `max_iter` — maximum number of ELS iterations (typically 10–30).
    /// * `tol` — convergence threshold on ‖Δθ‖₂.
    ///
    /// Returns `Err(InsufficientData)` if fewer samples are available than needed.
    /// Returns `Err(NotConverged)` if the algorithm does not converge within `max_iter`.
    pub fn fit(
        y: &[S],
        u: &[S],
        max_iter: usize,
        tol: S,
    ) -> Result<ArmaxModel<S, NA, NB, NC>, SysIdError> {
        let n = y.len();
        if u.len() != n {
            return Err(SysIdError::InvalidData);
        }
        for &v in y.iter().chain(u.iter()) {
            if !v.is_finite() {
                return Err(SysIdError::InvalidData);
            }
        }

        let start = NA.max(NB).max(NC);
        if n <= start {
            return Err(SysIdError::InsufficientData);
        }

        if P == 0 {
            return Ok(ArmaxModel::zeros());
        }

        let n_capped = n.min(Self::BUF_CAP);

        // Residual buffer indexed by sample time, initialised to zero
        let mut e_buf = [S::ZERO; 4096];

        let mut theta_prev = [S::ZERO; P];
        let mut theta_cur = [S::ZERO; P];
        let mut converged = false;

        for _iter in 0..max_iter {
            // Accumulate normal equations; add small Tikhonov regularisation
            // to prevent singularity when residual regressors are near-zero
            // (e.g. first ELS iteration or noiseless data with NC > 0).
            let reg = S::from_f64(1e-10);
            let mut ata = [[S::ZERO; P]; P];
            for i in 0..P {
                ata[i][i] = reg;
            }
            let mut aty = [S::ZERO; P];

            for t in start..n_capped {
                let mut phi = [S::ZERO; P];
                for i in 0..NA {
                    phi[i] = -y[t - 1 - i];
                }
                for i in 0..NB {
                    if t > i {
                        phi[NA + i] = u[t - 1 - i];
                    }
                }
                for i in 0..NC {
                    let tidx = t as isize - 1 - i as isize;
                    if tidx >= 0 {
                        phi[NA + NB + i] = e_buf[tidx as usize];
                    }
                }

                let yt = y[t];
                for i in 0..P {
                    for j in 0..P {
                        ata[i][j] += phi[i] * phi[j];
                    }
                    aty[i] += phi[i] * yt;
                }
            }

            cholesky_solve_n::<S, P>(&mut ata, &mut aty)?;
            theta_cur = aty;

            // Update residuals
            for t in start..n_capped {
                let mut phi = [S::ZERO; P];
                for i in 0..NA {
                    phi[i] = -y[t - 1 - i];
                }
                for i in 0..NB {
                    if t > i {
                        phi[NA + i] = u[t - 1 - i];
                    }
                }
                for i in 0..NC {
                    let tidx = t as isize - 1 - i as isize;
                    if tidx >= 0 {
                        phi[NA + NB + i] = e_buf[tidx as usize];
                    }
                }
                let mut y_hat = S::ZERO;
                for i in 0..P {
                    y_hat += phi[i] * theta_cur[i];
                }
                e_buf[t] = y[t] - y_hat;
            }

            // Convergence check ‖θ_cur − θ_prev‖₂
            let mut norm_sq = S::ZERO;
            for i in 0..P {
                let d = theta_cur[i] - theta_prev[i];
                norm_sq += d * d;
            }
            if norm_sq.sqrt() < tol {
                converged = true;
                break;
            }

            theta_prev = theta_cur;
        }

        if !converged && max_iter > 1 {
            return Err(SysIdError::NotConverged);
        }

        let mut model = ArmaxModel::zeros();
        for i in 0..NA {
            model.a[i] = theta_cur[i];
        }
        for i in 0..NB {
            model.b[i] = theta_cur[NA + i];
        }
        for i in 0..NC {
            model.c[i] = theta_cur[NA + NB + i];
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

    /// When NC=0 (P=2), ELS reduces to ARX least squares.
    /// Verify that the A, B coefficients match those of direct ARX identification.
    #[test]
    fn els_nc0_matches_arx() {
        let (y, u) = generate_arx1_noiseless(500);

        // ARX: NA=1, NB=1, P=2, NK=1
        let arx_model =
            ArxIdentifier::<f64, 1, 1, 2, 1>::fit(y.as_slice(), u.as_slice()).expect("ARX fit");

        // ELS with NC=0, P=2: single pass (max_iter=1) equals LS
        let armax_model =
            ELSIdentifier::<f64, 1, 1, 0, 2>::fit(y.as_slice(), u.as_slice(), 1, 1e-12)
                .expect("ELS fit with NC=0");

        let da = (armax_model.a[0] - arx_model.a[0]).abs();
        let db = (armax_model.b[0] - arx_model.b[0]).abs();
        assert!(da < 1e-8, "A coefficients differ by {da:.2e}");
        assert!(db < 1e-8, "B coefficients differ by {db:.2e}");
    }

    /// ELS on noiseless ARX data achieves FIT% > 99%.
    #[test]
    fn els_armax_fits_noiseless_arx_data() {
        let (y, u) = generate_arx1_noiseless(500);

        // NA=1, NB=1, NC=1, P=3
        let model = ELSIdentifier::<f64, 1, 1, 1, 3>::fit(y.as_slice(), u.as_slice(), 20, 1e-10)
            .expect("ELS fit");

        let mut predicted: heapless::Vec<f64, 4096> = heapless::Vec::new();
        let mut e_hist = [0.0_f64; 1];
        let mut y_hist = [0.0_f64; 1];
        let mut u_hist_buf = [0.0_f64; 1];
        for t in 1..y.len() {
            u_hist_buf[0] = u[t - 1];
            let y_hat = model.predict_one_step(&y_hist, &u_hist_buf, &e_hist);
            let e = y[t] - y_hat;
            e_hist[0] = e;
            y_hist[0] = y[t];
            let _ = predicted.push(y_hat);
        }
        let fp = fit_percent(predicted.as_slice(), &y.as_slice()[1..]);
        assert!(
            fp > 99.0,
            "ELS ARMAX FIT% should exceed 99% for noiseless data, got {fp:.2}%"
        );
    }

    #[test]
    fn armax_model_zeros_predict_returns_zero() {
        let model = ArmaxModel::<f64, 2, 2, 1>::zeros();
        let y_hist = [1.0_f64, 2.0];
        let u_hist = [0.5_f64, 0.3];
        let e_hist = [0.1_f64];
        let pred = model.predict_one_step(&y_hist, &u_hist, &e_hist);
        assert_eq!(pred, 0.0);
    }
}
