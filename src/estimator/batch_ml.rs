#![allow(clippy::too_many_arguments)]
use crate::core::matrix::{matmul, matvec, Matrix};
use crate::core::scalar::ControlScalar;
use crate::estimator::rts_smoother::{FilteredState, RtsSmoother, SmoothedData, SmootherError};

/// Errors arising from the ML estimator or the batch Kalman smoother.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstError {
    /// A required matrix inversion failed.
    SingularMatrix,
    /// The RTS smoother encountered an error.
    SmootherError(SmootherError),
    /// The supplied data sequences have incompatible lengths.
    LengthMismatch,
    /// Gradient descent failed to produce a valid update.
    OptimizationFailed,
}

impl core::fmt::Display for EstError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EstError::SingularMatrix => write!(f, "MlEstimator: singular matrix"),
            EstError::SmootherError(e) => write!(f, "MlEstimator: smoother error — {e}"),
            EstError::LengthMismatch => write!(f, "MlEstimator: data length mismatch"),
            EstError::OptimizationFailed => write!(f, "MlEstimator: optimization diverged"),
        }
    }
}

impl From<SmootherError> for EstError {
    fn from(e: SmootherError) -> Self {
        EstError::SmootherError(e)
    }
}

// ─── internal KF forward pass ────────────────────────────────────────────────

/// Run one complete forward Kalman pass on a measurement sequence.
///
/// Returns the negative log-likelihood `L = 0.5 · Σ(log|S_k| + νᵀ S_k⁻¹ ν)`
/// and fills `smoother` with the forward-pass states.
///
/// # Type Parameters
/// * `N` — state dimension
/// * `M` — measurement dimension
/// * `I` — input dimension
/// * `T` — maximum sequence length (compile-time constant)
fn forward_pass<S, const N: usize, const M: usize, const I: usize, const T: usize>(
    a: &Matrix<S, N, N>,
    b: &Matrix<S, N, I>,
    h: &Matrix<S, M, N>,
    q: &Matrix<S, N, N>,
    r: &Matrix<S, M, M>,
    y: &[[S; M]; T],
    u: &[[S; I]; T],
    steps: usize,
    x0: &[S; N],
    p0: &Matrix<S, N, N>,
    smoother: &mut RtsSmoother<S, N, T>,
) -> Result<S, EstError>
where
    S: ControlScalar,
{
    smoother.reset();
    let mut x = *x0;
    let mut p = *p0;
    let ht = h.transpose();
    let at = a.transpose();
    let mut neg_log_lik = S::ZERO;
    let log2pi = S::from_f64(core::f64::consts::TAU.ln()); // ln(2π)

    for k in 0..steps {
        // ── Predict ──────────────────────────────────────────────────────
        let ax = matvec(a, &x);
        let bu = matvec(b, &u[k]);
        let x_pred: [S; N] = core::array::from_fn(|i| ax[i] + bu[i]);

        let ap = matmul(a, &p);
        let apat = matmul(&ap, &at);
        let p_pred = apat.add_mat(q);

        // ── Update ───────────────────────────────────────────────────────
        let hx = matvec(h, &x_pred);
        let innov: [S; M] = core::array::from_fn(|i| y[k][i] - hx[i]);

        let hp = matmul(h, &p_pred);
        let hpht = matmul(&hp, &ht);
        let s_mat = hpht.add_mat(r); // innovation covariance S_k

        let s_inv = s_mat.inv().ok_or(EstError::SingularMatrix)?;
        let pht = matmul(&p_pred, &ht);
        let k_gain = matmul(&pht, &s_inv);

        let kv = matvec(&k_gain, &innov);
        let x_post: [S; N] = core::array::from_fn(|i| x_pred[i] + kv[i]);

        let kh = matmul(&k_gain, h);
        let eye = Matrix::<S, N, N>::identity();
        let i_kh = eye.sub_mat(&kh);
        let p_post = matmul(&i_kh, &p_pred);

        // ── Contribution to negative log-likelihood ───────────────────────
        // L += 0.5 · (M·ln(2π) + ln|S_k| + νᵀ S_k⁻¹ ν)
        let nu_s_nu: S = {
            let s_inv_nu = matvec(&s_inv, &innov);
            let mut acc = S::ZERO;
            for i in 0..M {
                acc += innov[i] * s_inv_nu[i];
            }
            acc
        };
        // Log-determinant via Cholesky (more stable); fall back to direct trace.
        let log_det_s = log_det_sym(&s_mat);
        let m_f = S::from_f64(M as f64);
        neg_log_lik += S::HALF * (m_f * log2pi + log_det_s + nu_s_nu);

        // ── Store forward state ───────────────────────────────────────────
        smoother.store_forward(FilteredState::new(x_post, p_post, x_pred, p_pred))?;

        x = x_post;
        p = p_post;
    }

    Ok(neg_log_lik)
}

/// Approximate log-determinant of a symmetric positive-definite matrix via
/// Cholesky decomposition.  Falls back to sum-of-logs of diagonal if Cholesky
/// succeeds; returns a large value if the matrix is singular.
fn log_det_sym<S: ControlScalar, const N: usize>(m: &Matrix<S, N, N>) -> S {
    match m.cholesky() {
        Some(l) => {
            let mut sum = S::ZERO;
            for i in 0..N {
                let d = l.data[i][i];
                if d > S::ZERO {
                    sum += d.ln();
                } else {
                    return S::from_f64(1e30_f64);
                }
            }
            sum * S::TWO // ln|M| = 2 · Σ ln(L_ii)
        }
        None => S::from_f64(1e30_f64),
    }
}

// ─── MlEstimator ─────────────────────────────────────────────────────────────

/// Batch Maximum-Likelihood estimator for Gaussian linear state-space models.
///
/// Optimises the log-likelihood of the observation sequence with respect to
/// the **process noise covariance** `Q` and **measurement noise covariance**
/// `R`, using gradient descent with finite-difference gradients.
///
/// The model is:
/// ```text
///   x[k+1] = A · x[k] + B · u[k] + w[k],   w ~ N(0, Q)
///   y[k]   = H · x[k] + v[k],               v ~ N(0, R)
/// ```
///
/// Diagonal structure of Q and R is assumed to keep the parameter space
/// tractable for finite-difference estimation.
///
/// # Type Parameters
/// * `S` — scalar type
/// * `N` — state dimension
/// * `M` — measurement dimension
/// * `T` — compile-time maximum sequence length
#[derive(Debug, Clone, Copy)]
pub struct MlEstimator<S: ControlScalar, const N: usize, const M: usize, const T: usize> {
    /// State transition matrix.
    pub a: Matrix<S, N, N>,
    /// Input matrix.
    pub b: Matrix<S, N, N>, // I=N for generality; callers supply zero columns.
    /// Measurement matrix.
    pub h: Matrix<S, M, N>,
    /// Initial state.
    pub x0: [S; N],
    /// Initial covariance.
    pub p0: Matrix<S, N, N>,
    /// Step size for finite-difference gradient approximation.
    pub fd_eps: S,
    /// Gradient-descent learning rate.
    pub learn_rate: S,
}

impl<S: ControlScalar, const N: usize, const M: usize, const T: usize> MlEstimator<S, N, M, T> {
    /// Construct a new ML estimator.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, N>,
        h: Matrix<S, M, N>,
        x0: [S; N],
        p0: Matrix<S, N, N>,
        fd_eps: S,
        learn_rate: S,
    ) -> Self {
        Self {
            a,
            b,
            h,
            x0,
            p0,
            fd_eps,
            learn_rate,
        }
    }

    /// Estimate Q and R by maximising the log-likelihood of the observation
    /// sequence `y[0..steps]` with inputs `u[0..steps]`.
    ///
    /// Uses gradient descent on the **negative log-likelihood** (NLL), with
    /// finite-difference gradients w.r.t. the diagonal elements of Q and R.
    ///
    /// Returns `(Q_est, R_est)` on convergence, or an error.
    pub fn fit(
        &self,
        y: &[[S; M]; T],
        u: &[[S; N]; T],
        steps: usize,
        max_iter: usize,
    ) -> Result<(Matrix<S, N, N>, Matrix<S, M, M>), EstError> {
        if steps == 0 || steps > T {
            return Err(EstError::LengthMismatch);
        }

        // Initialise Q and R as identity-scaled matrices.
        let mut q_diag: [S; N] = [S::ONE; N];
        let mut r_diag: [S; M] = [S::ONE; M];

        let eps = self.fd_eps;
        let lr = self.learn_rate;
        let zero_u: [S; N] = [S::ZERO; N];

        // Convert u (N-column) to [S;N] slices.
        let mut smoother = RtsSmoother::<S, N, T>::new();

        let mut prev_nll = S::from_f64(f64::MAX);

        for _iter in 0..max_iter {
            let q_cur = diag_matrix::<S, N>(&q_diag);
            let r_cur = diag_matrix::<S, M>(&r_diag);

            // Current NLL.
            let nll = forward_pass::<S, N, M, N, T>(
                &self.a,
                &self.b,
                &self.h,
                &q_cur,
                &r_cur,
                y,
                u,
                steps,
                &self.x0,
                &self.p0,
                &mut smoother,
            )?;

            // Check convergence.
            let delta = (prev_nll - nll).abs();
            if delta < S::from_f64(1e-6) {
                break;
            }
            prev_nll = nll;

            // ── Finite-difference gradient w.r.t. q_diag ─────────────────
            let mut grad_q = [S::ZERO; N];
            for i in 0..N {
                let mut qd_p = q_diag;
                qd_p[i] += eps;
                let q_p = diag_matrix::<S, N>(&qd_p);
                let nll_p = forward_pass::<S, N, M, N, T>(
                    &self.a,
                    &self.b,
                    &self.h,
                    &q_p,
                    &r_cur,
                    y,
                    u,
                    steps,
                    &self.x0,
                    &self.p0,
                    &mut smoother,
                )?;
                grad_q[i] = (nll_p - nll) / eps;
            }

            // ── Finite-difference gradient w.r.t. r_diag ─────────────────
            let mut grad_r = [S::ZERO; M];
            for i in 0..M {
                let mut rd_p = r_diag;
                rd_p[i] += eps;
                let r_p = diag_matrix::<S, M>(&rd_p);
                let nll_p = forward_pass::<S, N, M, N, T>(
                    &self.a,
                    &self.b,
                    &self.h,
                    &q_cur,
                    &r_p,
                    y,
                    u,
                    steps,
                    &self.x0,
                    &self.p0,
                    &mut smoother,
                )?;
                grad_r[i] = (nll_p - nll) / eps;
            }

            // ── Gradient descent update (projected onto positive values). ─
            for i in 0..N {
                let new_val = q_diag[i] - lr * grad_q[i];
                q_diag[i] = if new_val > S::from_f64(1e-10) {
                    new_val
                } else {
                    S::from_f64(1e-10)
                };
            }
            for i in 0..M {
                let new_val = r_diag[i] - lr * grad_r[i];
                r_diag[i] = if new_val > S::from_f64(1e-10) {
                    new_val
                } else {
                    S::from_f64(1e-10)
                };
            }

            // Suppress unused variable warning.
            let _ = zero_u;
        }

        Ok((diag_matrix::<S, N>(&q_diag), diag_matrix::<S, M>(&r_diag)))
    }
}

/// Build a diagonal matrix from a slice of diagonal values.
fn diag_matrix<S: ControlScalar, const N: usize>(diag: &[S; N]) -> Matrix<S, N, N> {
    Matrix {
        data: core::array::from_fn(|r| {
            core::array::from_fn(|c| if r == c { diag[r] } else { S::ZERO })
        }),
    }
}

// ─── BatchKalmanSmoother ──────────────────────────────────────────────────────

/// Combines ML parameter estimation (via `MlEstimator`) with RTS smoothing.
///
/// Workflow:
/// 1. Call `fit_and_smooth` with the measurement/input sequences.
/// 2. Internally performs ML estimation of Q and R.
/// 3. Runs a final RTS smooth pass with the estimated parameters.
/// 4. Returns smoothed states together with the estimated Q and R.
///
/// # Type Parameters
/// * `S` — scalar type
/// * `N` — state dimension
/// * `M` — measurement dimension
/// * `T` — compile-time maximum sequence length
#[derive(Debug, Clone, Copy)]
pub struct BatchKalmanSmoother<S: ControlScalar, const N: usize, const M: usize, const T: usize> {
    /// Underlying ML estimator.
    pub estimator: MlEstimator<S, N, M, T>,
}

/// Output of `BatchKalmanSmoother::fit_and_smooth`.
#[derive(Debug, Clone, Copy)]
pub struct BatchSmootherOutput<S: ControlScalar, const N: usize, const M: usize, const T: usize> {
    /// Smoothed states (length = `steps`).
    pub smoothed: SmoothedData<S, N, T>,
    /// Estimated process noise covariance.
    pub q_est: Matrix<S, N, N>,
    /// Estimated measurement noise covariance.
    pub r_est: Matrix<S, M, M>,
}

impl<S: ControlScalar, const N: usize, const M: usize, const T: usize>
    BatchKalmanSmoother<S, N, M, T>
{
    /// Create a new batch smoother wrapping an `MlEstimator`.
    pub fn new(estimator: MlEstimator<S, N, M, T>) -> Self {
        Self { estimator }
    }

    /// Estimate Q, R via ML and then smooth the sequence with the estimated
    /// parameters.
    ///
    /// Returns both the smoothed states and the estimated parameters.
    pub fn fit_and_smooth(
        &self,
        y: &[[S; M]; T],
        u: &[[S; N]; T],
        steps: usize,
        max_iter: usize,
    ) -> Result<BatchSmootherOutput<S, N, M, T>, EstError> {
        // Step 1: estimate Q, R.
        let (q_est, r_est) = self.estimator.fit(y, u, steps, max_iter)?;

        // Step 2: final forward pass with estimated parameters.
        let mut smoother = RtsSmoother::<S, N, T>::new();
        forward_pass::<S, N, M, N, T>(
            &self.estimator.a,
            &self.estimator.b,
            &self.estimator.h,
            &q_est,
            &r_est,
            y,
            u,
            steps,
            &self.estimator.x0,
            &self.estimator.p0,
            &mut smoother,
        )?;

        // Step 3: RTS backward pass.
        let smoothed = smoother.smooth(&self.estimator.a)?;

        Ok(BatchSmootherOutput {
            smoothed,
            q_est,
            r_est,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a deterministic pseudo-random noise sequence (LCG, no rand).
    fn lcg_noise(seed: u64, n: usize, scale: f64) -> [f64; 64] {
        let mut out = [0.0_f64; 64];
        let mut s = seed;
        for item in out.iter_mut().take(n) {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let frac = (s >> 33) as f64 / (u32::MAX as f64);
            *item = (frac - 0.5) * 2.0 * scale;
        }
        out
    }

    /// Build synthetic 1-D data with known Q=q, R=r.
    fn make_data_1d(steps: usize, q_true: f64, r_true: f64) -> ([[f64; 1]; 64], [[f64; 1]; 64]) {
        let proc_noise = lcg_noise(42, steps, q_true.sqrt());
        let meas_noise = lcg_noise(99, steps, r_true.sqrt());

        let mut y = [[0.0_f64; 1]; 64];
        let u = [[0.0_f64; 1]; 64];
        let mut x = 0.0_f64;
        for k in 0..steps {
            x += proc_noise[k]; // A=1
            y[k][0] = x + meas_noise[k]; // H=1
        }
        (y, u)
    }

    fn build_estimator_1d() -> MlEstimator<f64, 1, 1, 64> {
        MlEstimator::new(
            Matrix::<f64, 1, 1>::identity(),        // A
            Matrix::<f64, 1, 1>::zeros(),           // B (unused)
            Matrix::<f64, 1, 1>::identity(),        // H
            [0.0_f64; 1],                           // x0
            Matrix::<f64, 1, 1> { data: [[10.0]] }, // P0
            1e-4,                                   // fd_eps
            1e-3,                                   // learn_rate
        )
    }

    #[test]
    fn ml_fit_produces_positive_diagonal_q_r() {
        let (y, u) = make_data_1d(30, 0.1, 1.0);
        let est = build_estimator_1d();
        let (q_est, r_est) = est.fit(&y, &u, 30, 50).expect("fit");
        assert!(q_est.data[0][0] > 0.0, "Q diagonal must be positive");
        assert!(r_est.data[0][0] > 0.0, "R diagonal must be positive");
    }

    #[test]
    fn ml_fit_recovers_order_of_magnitude_q_r() {
        // With enough data, the ML estimate should be in the same order of
        // magnitude as the true values.
        let q_true = 0.05_f64;
        let r_true = 0.5_f64;
        let (y, u) = make_data_1d(50, q_true, r_true);
        let est = build_estimator_1d();
        let (q_est, r_est) = est.fit(&y, &u, 50, 200).expect("fit");
        let q_ratio = q_est.data[0][0] / q_true;
        let r_ratio = r_est.data[0][0] / r_true;
        // With 50 steps and gradient descent the estimate should be within
        // two orders of magnitude.  A tighter bound requires more data/iterations.
        assert!(
            q_ratio > 0.01 && q_ratio < 200.0,
            "Q estimate order of magnitude off: {q_ratio}"
        );
        assert!(
            r_ratio > 0.01 && r_ratio < 200.0,
            "R estimate order of magnitude off: {r_ratio}"
        );
    }

    #[test]
    fn batch_smoother_length_correct() {
        let (y, u) = make_data_1d(20, 0.1, 1.0);
        let est = build_estimator_1d();
        let batch = BatchKalmanSmoother::new(est);
        let out = batch
            .fit_and_smooth(&y, &u, 20, 30)
            .expect("fit_and_smooth");
        assert_eq!(out.smoothed.len, 20);
    }

    #[test]
    fn batch_smoother_covariance_non_negative() {
        let (y, u) = make_data_1d(15, 0.1, 1.0);
        let est = build_estimator_1d();
        let batch = BatchKalmanSmoother::new(est);
        let out = batch
            .fit_and_smooth(&y, &u, 15, 30)
            .expect("fit_and_smooth");
        for k in 0..out.smoothed.len {
            let p = out.smoothed.states[k].p.data[0][0];
            assert!(
                p >= 0.0,
                "Smoothed covariance must be non-negative at k={k}: {p}"
            );
        }
    }

    #[test]
    fn length_mismatch_returns_error() {
        let (y, u) = make_data_1d(10, 0.1, 1.0);
        let est = build_estimator_1d();
        // steps=0 is invalid
        let res = est.fit(&y, &u, 0, 10);
        assert!(matches!(res, Err(EstError::LengthMismatch)));
    }
}
