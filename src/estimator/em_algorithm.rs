#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
use crate::core::matrix::{matmul, matvec, outer, Matrix};
use crate::core::scalar::ControlScalar;
use crate::estimator::rts_smoother::{FilteredState, RtsSmoother, SmootherError};

/// Error type for the EM algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmError {
    /// A required matrix inversion failed.
    SingularMatrix,
    /// Smoother internal error.
    SmootherError(SmootherError),
    /// The supplied sequence length is invalid (0 or > T).
    InvalidLength,
}

impl core::fmt::Display for EmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EmError::SingularMatrix => write!(f, "EmAlgorithm: singular matrix"),
            EmError::SmootherError(e) => write!(f, "EmAlgorithm: smoother error — {e}"),
            EmError::InvalidLength => write!(f, "EmAlgorithm: invalid sequence length"),
        }
    }
}

impl From<SmootherError> for EmError {
    fn from(e: SmootherError) -> Self {
        EmError::SmootherError(e)
    }
}

/// Learned model parameters produced by the EM algorithm.
///
/// All matrices use the compile-time dimensions of `EmAlgorithm<S,N,M,T>`.
#[derive(Debug, Clone, Copy)]
pub struct EmModel<S: ControlScalar, const N: usize, const M: usize> {
    /// State transition matrix A (N×N).
    pub a: Matrix<S, N, N>,
    /// Measurement matrix C (M×N).
    pub c: Matrix<S, M, N>,
    /// Process noise covariance Q (N×N).
    pub q: Matrix<S, N, N>,
    /// Measurement noise covariance R (M×M).
    pub r: Matrix<S, M, M>,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Run a forward Kalman pass and fill the smoother buffer.
///
/// Returns nothing (the smoother buffer is populated in place).
fn kf_forward<S, const N: usize, const M: usize, const T: usize>(
    a: &Matrix<S, N, N>,
    c: &Matrix<S, M, N>,
    q: &Matrix<S, N, N>,
    r: &Matrix<S, M, M>,
    y: &[[S; M]; T],
    steps: usize,
    x0: &[S; N],
    p0: &Matrix<S, N, N>,
    smoother: &mut RtsSmoother<S, N, T>,
) -> Result<(), EmError>
where
    S: ControlScalar,
{
    smoother.reset();
    let mut x = *x0;
    let mut p = *p0;
    let ct = c.transpose();
    let at = a.transpose();

    for k in 0..steps {
        // Predict
        let x_pred = matvec(a, &x);
        let ap = matmul(a, &p);
        let apat = matmul(&ap, &at);
        let p_pred = apat.add_mat(q);

        // Update
        let cx = matvec(c, &x_pred);
        let innov: [S; M] = core::array::from_fn(|i| y[k][i] - cx[i]);

        let cp = matmul(c, &p_pred);
        let cpct = matmul(&cp, &ct);
        let s_mat = cpct.add_mat(r);
        let s_inv = s_mat.inv().ok_or(EmError::SingularMatrix)?;
        let pct = matmul(&p_pred, &ct);
        let k_gain = matmul(&pct, &s_inv);

        let kv = matvec(&k_gain, &innov);
        let x_post: [S; N] = core::array::from_fn(|i| x_pred[i] + kv[i]);

        let kc = matmul(&k_gain, c);
        let eye = Matrix::<S, N, N>::identity();
        let i_kc = eye.sub_mat(&kc);
        let p_post = matmul(&i_kc, &p_pred);

        smoother.store_forward(FilteredState::new(x_post, p_post, x_pred, p_pred))?;

        x = x_post;
        p = p_post;
    }

    Ok(())
}

// ─── EmAlgorithm ─────────────────────────────────────────────────────────────

/// Expectation-Maximization (EM) learner for linear Gaussian state-space models.
///
/// ## Model
/// ```text
///   x[k+1] = A · x[k] + w[k],   w ~ N(0, Q)
///   y[k]   = C · x[k] + v[k],   v ~ N(0, R)
/// ```
///
/// ## Algorithm
///
/// **E-step**: Run the RTS Kalman smoother with current parameters to obtain
/// the smoothed sufficient statistics:
/// - `E[x_k]`, `E[x_k x_kᵀ]`, `E[x_k x_{k-1}ᵀ]`
///
/// **M-step**: Update parameters by maximising the expected complete-data
/// log-likelihood:
/// ```text
///   A_new = P_{xk, xk-1} · P_{xk-1, xk-1}⁻¹
///   C_new = (Σ y_k x_{k|T}ᵀ) · (Σ x_{k|T} x_{k|T}ᵀ)⁻¹
///   Q_new = (1/(T-1)) · (P_{xk,xk} - A_new · P_{xk-1,xk}ᵀ)
///   R_new = (1/T)     · (Σ y_k y_kᵀ - C_new · Σ x_{k|T} y_kᵀ)
/// ```
///
/// # Type Parameters
/// * `S` — scalar type
/// * `N` — state dimension
/// * `M` — measurement dimension
/// * `T` — compile-time maximum sequence length
#[derive(Debug, Clone, Copy)]
pub struct EmAlgorithm<S: ControlScalar, const N: usize, const M: usize, const T: usize> {
    /// Initial state transition matrix A.
    pub a_init: Matrix<S, N, N>,
    /// Initial measurement matrix C.
    pub c_init: Matrix<S, M, N>,
    /// Initial process noise covariance Q.
    pub q_init: Matrix<S, N, N>,
    /// Initial measurement noise covariance R.
    pub r_init: Matrix<S, M, M>,
    /// Initial state mean.
    pub x0: [S; N],
    /// Initial state covariance.
    pub p0: Matrix<S, N, N>,
}

impl<S: ControlScalar, const N: usize, const M: usize, const T: usize> EmAlgorithm<S, N, M, T> {
    /// Create a new EM learner with initial parameters.
    pub fn new(
        a_init: Matrix<S, N, N>,
        c_init: Matrix<S, M, N>,
        q_init: Matrix<S, N, N>,
        r_init: Matrix<S, M, M>,
        x0: [S; N],
        p0: Matrix<S, N, N>,
    ) -> Self {
        Self {
            a_init,
            c_init,
            q_init,
            r_init,
            x0,
            p0,
        }
    }

    /// Run the EM algorithm on measurement sequence `y[0..steps]`.
    ///
    /// Iterates at most `max_iter` times; stops early when the Frobenius-norm
    /// change in A and C is below `tol`.
    ///
    /// Returns the learned `EmModel` on success.
    pub fn fit(
        &self,
        y: &[[S; M]; T],
        steps: usize,
        max_iter: usize,
        tol: S,
    ) -> Result<EmModel<S, N, M>, EmError> {
        if steps == 0 || steps > T {
            return Err(EmError::InvalidLength);
        }

        let mut a = self.a_init;
        let mut c = self.c_init;
        let mut q = self.q_init;
        let mut r = self.r_init;

        let mut smoother = RtsSmoother::<S, N, T>::new();

        for _iter in 0..max_iter {
            // ── E-step: run RTS smoother ──────────────────────────────────
            kf_forward(&a, &c, &q, &r, y, steps, &self.x0, &self.p0, &mut smoother)?;
            let smoothed = smoother.smooth(&a)?;

            // ── Compute smoothed sufficient statistics ────────────────────
            // Pxx  = Σ_{k=1}^{T-1} E[x_k x_kᵀ]  = Σ P_{k|T} + x_{k|T} x_{k|T}ᵀ
            // Pxx1 = Σ_{k=1}^{T-1} E[x_k x_{k-1}ᵀ]
            // Pxx0 = Σ_{k=0}^{T-2} E[x_k x_kᵀ]   (lagged version)
            //
            // Also:
            // Syx  = Σ_{k=0}^{T-1} y_k x_{k|T}ᵀ
            // Sxx  = Σ_{k=0}^{T-1} E[x_k x_kᵀ]
            // Syy  = Σ_{k=0}^{T-1} y_k y_kᵀ

            let mut pxx: Matrix<S, N, N> = Matrix::zeros(); // Σ E[x_k x_kᵀ] for k=1..T-1
            let mut pxx1: Matrix<S, N, N> = Matrix::zeros(); // Σ E[x_k x_{k-1}ᵀ] for k=1..T-1
            let mut pxx0: Matrix<S, N, N> = Matrix::zeros(); // Σ E[x_k x_kᵀ] for k=0..T-2

            let mut syx: Matrix<S, M, N> = Matrix::zeros(); // Σ y_k x_{k|T}ᵀ  all k
            let mut sxx: Matrix<S, N, N> = Matrix::zeros(); // Σ E[x_k x_kᵀ]   all k
            let mut syy: Matrix<S, M, M> = Matrix::zeros(); // Σ y_k y_kᵀ      all k

            // Accumulate E[x_k x_kᵀ] = P_{k|T} + x_{k|T} x_{k|T}ᵀ
            // Accumulate cross-cov E[x_k x_{k-1}ᵀ] using the lag-1 smoother
            // cross-covariance: P_{k,k-1|T} = G_{k-1} · P_{k|T}
            // (Standard result from Shumway & Stoffer 2000.)

            for k in 0..steps {
                let xk = smoothed.states[k].x;
                let pk = smoothed.states[k].p;

                // E[x_k x_kᵀ]
                let exxk = pk.add_mat(&outer(&xk, &xk));

                sxx = sxx.add_mat(&exxk);

                // y_k x_{k|T}ᵀ
                let yk = y[k];
                let yx_k = outer(&yk, &xk);
                syx = syx.add_mat(&yx_k);

                // y_k y_kᵀ
                let yy_k = outer(&yk, &yk);
                syy = syy.add_mat(&yy_k);

                // Separate sums for A M-step (k=1..T-1 vs k=0..T-2).
                if k > 0 {
                    pxx = pxx.add_mat(&exxk);
                }
                if k < steps - 1 {
                    pxx0 = pxx0.add_mat(&exxk);
                }
            }

            // Lag-1 cross-covariance: P_{k,k-1|T} = G_{k-1} · P_{k|T}
            // Smoother gain G_{k-1} = P_{k-1|k-1} · Aᵀ · P_{k|k-1}⁻¹
            // (stored in forward buffer from the last kf_forward call).
            let at_cur = a.transpose();
            for k in 1..steps {
                let fk_prev = smoother
                    .get_state(k - 1)
                    .ok_or(EmError::SmootherError(SmootherError::IndexOutOfRange))?;
                let p_pred_k = fk_prev.p_pred; // P_{k|k-1}
                let p_km1_post = fk_prev.p; // P_{k-1|k-1}

                let p_pred_k_inv = p_pred_k.inv().ok_or(EmError::SingularMatrix)?;
                // G_{k-1} = P_{k-1|k-1} · Aᵀ · P_{k|k-1}⁻¹
                let p_at = matmul(&p_km1_post, &at_cur);
                let g_km1 = matmul(&p_at, &p_pred_k_inv);

                // P_{k,k-1|T} = G_{k-1} · P_{k|T}
                let pk_smooth = smoothed.states[k].p;
                let cross = matmul(&g_km1, &pk_smooth);

                // E[x_k x_{k-1}ᵀ] = P_{k,k-1|T} + x_{k|T} x_{k-1|T}ᵀ
                let xk = smoothed.states[k].x;
                let xkm1 = smoothed.states[k - 1].x;
                let exxcross = cross.add_mat(&outer(&xk, &xkm1));

                pxx1 = pxx1.add_mat(&exxcross);
            }

            // ── M-step ────────────────────────────────────────────────────

            // A_new = P_{xk,xk-1} · P_{xk-1,xk-1}⁻¹
            //       = pxx1 · pxx0⁻¹
            let a_new = if steps > 1 {
                let pxx0_inv = pxx0.inv().ok_or(EmError::SingularMatrix)?;
                matmul(&pxx1, &pxx0_inv)
            } else {
                a
            };

            // C_new = Syx · Sxx⁻¹
            let sxx_inv = sxx.inv().ok_or(EmError::SingularMatrix)?;
            let c_new = matmul(&syx, &sxx_inv);

            // Q_new = (1/(T-1)) · (Pxx - A_new · Pxx1ᵀ)
            let q_new = if steps > 1 {
                let t_minus_1 = S::from_f64((steps - 1) as f64);
                let pxx1_t = pxx1.transpose();
                let a_pxx1t = matmul(&a_new, &pxx1_t);
                let diff = pxx.sub_mat(&a_pxx1t);
                symmetrize(diff.scale(S::ONE / t_minus_1))
            } else {
                q
            };

            // R_new = (1/T) · (Syy - C_new · Syxᵀ)
            let t_f = S::from_f64(steps as f64);
            let syx_t = syx.transpose();
            let c_syxt = matmul(&c_new, &syx_t);
            let r_diff = syy.sub_mat(&c_syxt);
            let r_new = symmetrize(r_diff.scale(S::ONE / t_f));

            // ── Convergence check ─────────────────────────────────────────
            let da = a_new.sub_mat(&a).frob_norm();
            let dc = c_new.sub_mat(&c).frob_norm();

            a = a_new;
            c = c_new;
            q = q_new;
            r = r_new;

            if da < tol && dc < tol {
                break;
            }
        }

        Ok(EmModel { a, c, q, r })
    }
}

/// Force a matrix to be symmetric by averaging with its transpose.
fn symmetrize<S: ControlScalar, const N: usize>(m: Matrix<S, N, N>) -> Matrix<S, N, N> {
    let mt = m.transpose();
    m.add_mat(&mt).scale(S::HALF)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple LCG pseudo-random noise (no external crates).
    fn lcg(seed: u64, n: usize, scale: f64) -> [f64; 64] {
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

    /// Generate 1-D observation sequence with true A=a_true, C=1, Q=q_true, R=r_true.
    fn make_obs(steps: usize, a_true: f64, q_true: f64, r_true: f64) -> [[f64; 1]; 64] {
        let proc_n = lcg(7, steps, q_true.sqrt());
        let meas_n = lcg(13, steps, r_true.sqrt());
        let mut y = [[0.0_f64; 1]; 64];
        let mut x = 0.0_f64;
        for k in 0..steps {
            x = a_true * x + proc_n[k];
            y[k][0] = x + meas_n[k];
        }
        y
    }

    fn build_em_1d() -> EmAlgorithm<f64, 1, 1, 64> {
        EmAlgorithm::new(
            Matrix::<f64, 1, 1>::identity(),        // A init
            Matrix::<f64, 1, 1>::identity(),        // C init
            Matrix::<f64, 1, 1> { data: [[1.0]] },  // Q init
            Matrix::<f64, 1, 1> { data: [[1.0]] },  // R init
            [0.0_f64; 1],                           // x0
            Matrix::<f64, 1, 1> { data: [[10.0]] }, // P0
        )
    }

    #[test]
    fn em_converges_to_valid_model() {
        let y = make_obs(40, 0.95, 0.1, 0.5);
        let em = build_em_1d();
        let model = em.fit(&y, 40, 80, 1e-5).expect("EM fit");

        // Q and R must be positive (variance cannot be negative).
        assert!(
            model.q.data[0][0] > 0.0,
            "Q must be positive: {}",
            model.q.data[0][0]
        );
        assert!(
            model.r.data[0][0] > 0.0,
            "R must be positive: {}",
            model.r.data[0][0]
        );
    }

    #[test]
    fn em_recovers_a_approximately() {
        // True A = 0.9; with enough data EM should move A toward 0.9.
        let a_true = 0.9_f64;
        let y = make_obs(60, a_true, 0.1, 0.3);
        let em = build_em_1d();
        let model = em.fit(&y, 60, 120, 1e-6).expect("EM fit");

        let a_est = model.a.data[0][0];
        assert!(
            (a_est - a_true).abs() < 0.4,
            "A estimate should be within 0.4 of true {a_true}: got {a_est}"
        );
    }

    #[test]
    fn em_recovers_c_approximately() {
        // True C = 1 (identity); EM should maintain it.
        let y = make_obs(50, 0.95, 0.05, 0.2);
        let em = build_em_1d();
        let model = em.fit(&y, 50, 100, 1e-6).expect("EM fit");

        let c_est = model.c.data[0][0];
        // C should be in a reasonable range (not 0 or exploding).
        assert!(
            c_est.abs() > 0.01 && c_est.abs() < 100.0,
            "C estimate out of reasonable range: {c_est}"
        );
    }

    #[test]
    fn em_single_step_does_not_panic() {
        let y = make_obs(1, 0.95, 0.1, 0.5);
        let em = build_em_1d();
        // steps=1 means no A update (T-1=0); should succeed without error.
        let model = em.fit(&y, 1, 10, 1e-6).expect("EM fit single step");
        assert!(model.q.data[0][0] > 0.0);
    }

    #[test]
    fn em_invalid_length_returns_error() {
        let y = make_obs(10, 0.95, 0.1, 0.5);
        let em = build_em_1d();
        let res = em.fit(&y, 0, 10, 1e-6);
        assert!(matches!(res, Err(EmError::InvalidLength)));
    }
}
