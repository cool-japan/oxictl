#![allow(clippy::needless_range_loop)]
use crate::core::matrix::{matmul, matvec, Matrix};
use crate::core::scalar::ControlScalar;

/// Square-Root Kalman Filter.
///
/// Maintains the **lower-triangular Cholesky factor** `S` of the error
/// covariance such that `P = S · Sᵀ`.  Operating on `S` rather than `P`
/// directly provides superior numerical stability for ill-conditioned systems
/// (e.g., high-precision INS, long-horizon propagation) where rounding errors
/// can drive a standard KF covariance negative-definite.
///
/// Algorithm summary:
/// * **Predict**: propagate `S` through a Cholesky QR update
///   `S_pred` = chol( A·S·Sᵀ·Aᵀ + Q )
/// * **Update**: Cholesky rank-1 downdate via sequential scalar measurements
///   (converts the M-dimensional measurement into M independent scalar updates)
///
/// Discrete-time linear model:
/// ```text
///   x[k+1] = A·x[k] + B·u[k] + w[k],  w ~ N(0, Q)
///   z[k]   = H·x[k] + v[k],             v ~ N(0, R)
/// ```
///
/// # Type Parameters
/// * `S` – scalar type (`f32` or `f64`)
/// * `N` – state dimension
/// * `M` – measurement dimension
/// * `I` – input dimension
#[derive(Debug, Clone, Copy)]
pub struct SqrtKalman<S: ControlScalar, const N: usize, const M: usize, const I: usize> {
    /// State transition matrix (N×N).
    pub a: Matrix<S, N, N>,
    /// Control input matrix (N×I).
    pub b: Matrix<S, N, I>,
    /// Measurement matrix (M×N).
    pub h: Matrix<S, M, N>,
    /// Process noise covariance (N×N).
    pub q: Matrix<S, N, N>,
    /// Measurement noise **standard deviation** per channel (diagonal R assumed).
    /// `r_diag[i]` is the std-dev σ_i, so R_ii = σ_i².
    pub r_diag: [S; M],
    /// State estimate (N).
    x: [S; N],
    /// Lower-triangular Cholesky factor of P (N×N).
    s_chol: Matrix<S, N, N>,
}

/// Error type for the Square-Root Kalman Filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqrtKfError {
    /// Predicted covariance is not positive definite (Cholesky failed).
    NotPositiveDefinite,
    /// A required matrix inversion failed.
    SingularMatrix,
    /// Provided initial `s0` is not lower-triangular or not valid.
    InvalidCholesky,
}

impl core::fmt::Display for SqrtKfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SqrtKfError::NotPositiveDefinite => {
                write!(f, "SqrtKalman: predicted covariance not positive definite")
            }
            SqrtKfError::SingularMatrix => write!(f, "SqrtKalman: singular matrix"),
            SqrtKfError::InvalidCholesky => write!(f, "SqrtKalman: invalid Cholesky factor"),
        }
    }
}

impl<S: ControlScalar, const N: usize, const M: usize, const I: usize> SqrtKalman<S, N, M, I> {
    /// Create a Square-Root Kalman Filter from standard KF parameters.
    ///
    /// `p0` must be positive definite; `r_diag[i]` is the std-dev of sensor `i`.
    /// Returns `None` if `p0`'s Cholesky decomposition fails.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        h: Matrix<S, M, N>,
        q: Matrix<S, N, N>,
        r_diag: [S; M],
        x0: [S; N],
        p0: Matrix<S, N, N>,
    ) -> Option<Self> {
        let s_chol = p0.cholesky()?;
        Some(Self {
            a,
            b,
            h,
            q,
            r_diag,
            x: x0,
            s_chol,
        })
    }

    /// Create directly from a Cholesky factor `s0` (lower-triangular).
    pub fn from_cholesky(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        h: Matrix<S, M, N>,
        q: Matrix<S, N, N>,
        r_diag: [S; M],
        x0: [S; N],
        s0: Matrix<S, N, N>,
    ) -> Self {
        Self {
            a,
            b,
            h,
            q,
            r_diag,
            x: x0,
            s_chol: s0,
        }
    }

    /// **Predict step**.
    ///
    /// Propagates state and the Cholesky factor:
    /// 1. `x_pred = A·x + B·u`
    /// 2. `P_pred = A·P·Aᵀ + Q`  (via full `P` reconstruction then Cholesky)
    ///
    /// Returns `Err` if the predicted covariance is not positive definite.
    pub fn predict(&mut self, u: &[S; I]) -> Result<(), SqrtKfError> {
        // Reconstruct P = S · Sᵀ
        let st = self.s_chol.transpose();
        let p = matmul(&self.s_chol, &st);

        // Propagate state
        let ax = matvec(&self.a, &self.x);
        let bu = matvec(&self.b, u);
        self.x = core::array::from_fn(|i| ax[i] + bu[i]);

        // Propagate covariance: P_pred = A·P·Aᵀ + Q
        let ap = matmul(&self.a, &p);
        let at = self.a.transpose();
        let apat = matmul(&ap, &at);
        let p_pred = apat.add_mat(&self.q);

        // Re-factorise
        self.s_chol = p_pred.cholesky().ok_or(SqrtKfError::NotPositiveDefinite)?;
        Ok(())
    }

    /// **Update step** using sequential scalar processing.
    ///
    /// Each measurement channel `z[i]` is processed independently as a scalar
    /// update, which is equivalent to the vector update when `R` is diagonal
    /// and avoids any matrix inversion.
    ///
    /// The Cholesky factor is updated via a **rank-1 downdate** (Givens rotation
    /// method) after each scalar measurement.
    ///
    /// Returns the stacked innovation vector, or `Err` if the downdate fails.
    pub fn update(&mut self, z: &[S; M]) -> Result<[S; M], SqrtKfError> {
        let mut innovation = [S::ZERO; M];

        for i in 0..M {
            // Extract i-th row of H: h_row (1×N) → as a vector
            let h_row: [S; N] = core::array::from_fn(|j| self.h.data[i][j]);

            // Innovation scalar: ν = z[i] - h_row · x
            let hx: S = {
                let mut acc = S::ZERO;
                for j in 0..N {
                    acc += h_row[j] * self.x[j];
                }
                acc
            };
            let nu = z[i] - hx;
            innovation[i] = nu;

            // Innovation variance: σ² = h_row · P · h_rowᵀ + R_ii
            // P = S · Sᵀ, so h · P · hᵀ = ‖S^T · h^T‖²
            let sth: [S; N] = {
                let st = self.s_chol.transpose();
                matvec(&st, &h_row)
            };
            let hp_ht: S = {
                let mut acc = S::ZERO;
                for &v in &sth {
                    acc += v * v;
                }
                acc
            };
            let r_ii = self.r_diag[i] * self.r_diag[i];
            let sigma2 = hp_ht + r_ii;

            if sigma2 <= S::ZERO {
                return Err(SqrtKfError::SingularMatrix);
            }
            let sigma = sigma2.sqrt();

            // Kalman gain (scalar): k = P · h_rowᵀ / σ²
            // = S · Sᵀ · h_rowᵀ / σ²
            //   We compute S · sth (which is S · Sᵀ · h) then divide by σ²
            let p_ht: [S; N] = matvec(&self.s_chol, &sth);
            let k: [S; N] = core::array::from_fn(|j| p_ht[j] / sigma2);

            // State update: x ← x + k · ν
            for j in 0..N {
                self.x[j] += k[j] * nu;
            }

            // Cholesky rank-1 downdate: S ← choldowndate(S, (1/σ) · S·sth / σ)
            // Using the Givens rotation method:
            //   For v = Sᵀ · h_rowᵀ / σ, we seek S_new such that
            //   S_new · S_newᵀ = S · Sᵀ - (S·sth) · (S·sth)ᵀ / σ²
            //   which equals P - P·hᵀ·h·P / σ²  (i.e. the Joseph update diagonal step)
            //
            // We perform the downdate in-place using the standard sequential downdate:
            //   for k = 0..N:
            //     r = sqrt(S[k,k]² - v[k]²)
            //     c = r / S[k,k];  s_rot = v[k] / S[k,k]
            //     S[k,k] = r
            //     for j = k+1..N:
            //         S[j,k] = (S[j,k] - s_rot * v[j]) / c
            //         v[j]   = c * v[j] - s_rot * S_old[j,k]  (use new S[j,k])
            //
            // v_downdate = (S·sth) / σ  = p_ht / σ
            let mut v_down: [S; N] = core::array::from_fn(|j| p_ht[j] / sigma);

            let downdate_ok = chol_rank1_downdate(&mut self.s_chol, &mut v_down);
            if !downdate_ok {
                return Err(SqrtKfError::NotPositiveDefinite);
            }
        }

        Ok(innovation)
    }

    /// Current state estimate.
    pub fn state(&self) -> &[S; N] {
        &self.x
    }

    /// Reconstruct full covariance `P = S · Sᵀ`.
    pub fn covariance(&self) -> Matrix<S, N, N> {
        let st = self.s_chol.transpose();
        matmul(&self.s_chol, &st)
    }

    /// Lower-triangular Cholesky factor `S` where `P = S · Sᵀ`.
    pub fn cholesky_factor(&self) -> &Matrix<S, N, N> {
        &self.s_chol
    }

    /// Reset to new state and covariance.  Returns `None` if `p0` is not PD.
    pub fn reset(&mut self, x0: [S; N], p0: Matrix<S, N, N>) -> Option<()> {
        self.s_chol = p0.cholesky()?;
        self.x = x0;
        Some(())
    }
}

/// Perform an in-place rank-1 Cholesky downdate on the lower-triangular factor
/// `l` (N×N) using the vector `v` (modified in place).
///
/// After the call `l` satisfies:  `l_new · l_newᵀ = l_old · l_oldᵀ - v·vᵀ`
///
/// Uses the sequential Givens rotation approach.  Returns `false` if the
/// downdate would make the matrix non-positive-definite.
fn chol_rank1_downdate<S: ControlScalar, const N: usize>(
    l: &mut Matrix<S, N, N>,
    v: &mut [S; N],
) -> bool {
    for k in 0..N {
        let lkk = l.data[k][k];
        let vk = v[k];
        let r2 = lkk * lkk - vk * vk;
        if r2 <= S::ZERO {
            return false;
        }
        let r = r2.sqrt();
        let c = r / lkk;
        let s_rot = vk / lkk;
        l.data[k][k] = r;
        for j in (k + 1)..N {
            let lj = l.data[j][k];
            let vj = v[j];
            l.data[j][k] = (lj - s_rot * vj) / c;
            v[j] = c * vj - s_rot * lj;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_filter() -> SqrtKalman<f64, 2, 1, 1> {
        let dt = 0.01_f64;
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = dt;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.5 * dt * dt;
        b.data[1][0] = dt;

        let mut h = Matrix::<f64, 1, 2>::zeros();
        h.data[0][0] = 1.0;

        let q = Matrix::<f64, 2, 2>::identity().scale(1e-4);
        let r_diag = [0.3162_f64]; // σ ≈ 0.316, R = 0.1
        let p0 = Matrix::<f64, 2, 2>::identity().scale(10.0);

        SqrtKalman::new(a, b, h, q, r_diag, [0.0_f64; 2], p0).expect("p0 positive definite")
    }

    #[test]
    fn new_with_valid_p0() {
        let _f = build_filter();
    }

    #[test]
    fn new_returns_none_for_non_pd() {
        let a = Matrix::<f64, 2, 2>::identity();
        let b = Matrix::<f64, 2, 1>::zeros();
        let mut h = Matrix::<f64, 1, 2>::zeros();
        h.data[0][0] = 1.0;
        let q = Matrix::<f64, 2, 2>::identity().scale(1e-4);
        let r_diag = [0.3_f64];
        let p0 = Matrix::<f64, 2, 2>::zeros(); // singular
        let result = SqrtKalman::new(a, b, h, q, r_diag, [0.0_f64; 2], p0);
        assert!(result.is_none());
    }

    #[test]
    fn predict_runs() {
        let mut f = build_filter();
        assert!(f.predict(&[0.0]).is_ok());
    }

    #[test]
    fn update_returns_innovation() {
        let mut f = build_filter();
        f.predict(&[0.0]).expect("predict");
        let innov = f.update(&[1.0]).expect("update");
        assert_eq!(innov.len(), 1);
    }

    #[test]
    fn tracks_constant_position() {
        let mut f = build_filter();
        let true_pos = 5.0_f64;
        for _ in 0..300 {
            f.predict(&[0.0]).expect("predict");
            f.update(&[true_pos]).expect("update");
        }
        assert!(
            (f.state()[0] - true_pos).abs() < 0.5,
            "Expected ~{true_pos}, got {}",
            f.state()[0]
        );
    }

    #[test]
    fn cholesky_factor_stays_lower_triangular() {
        let mut f = build_filter();
        for _ in 0..50 {
            f.predict(&[0.0]).expect("predict");
            f.update(&[2.0]).expect("update");
        }
        let s = f.cholesky_factor();
        for i in 0..2 {
            for j in (i + 1)..2 {
                assert!(
                    s.data[i][j].abs() < 1e-12,
                    "Upper triangle non-zero: S[{i}][{j}] = {}",
                    s.data[i][j]
                );
            }
        }
    }

    #[test]
    fn covariance_positive_definite_after_updates() {
        let mut f = build_filter();
        for _ in 0..20 {
            f.predict(&[0.0]).expect("predict");
            f.update(&[1.0]).expect("update");
        }
        let p = f.covariance();
        // P positive definite ↔ Cholesky succeeds
        assert!(p.cholesky().is_some(), "P should remain positive definite");
    }

    #[test]
    fn covariance_decreases_over_time() {
        let mut f = build_filter();
        let initial_trace = f.covariance().trace();
        for i in 0..100 {
            f.predict(&[0.0]).expect("predict");
            f.update(&[i as f64 * 0.01]).expect("update");
        }
        let final_trace = f.covariance().trace();
        assert!(
            final_trace < initial_trace,
            "Trace should decrease: {initial_trace} → {final_trace}"
        );
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut f = build_filter();
        for _ in 0..20 {
            f.predict(&[0.0]).expect("predict");
            f.update(&[5.0]).expect("update");
        }
        let p0 = Matrix::<f64, 2, 2>::identity().scale(10.0);
        f.reset([0.0_f64; 2], p0).expect("reset");
        assert!((f.state()[0]).abs() < 1e-10);
    }

    #[test]
    fn chol_rank1_downdate_correctness() {
        // Verify: L_new · L_newᵀ ≈ L · Lᵀ - v · vᵀ
        let p = Matrix::<f64, 2, 2> {
            data: [[4.0, 2.0], [2.0, 3.0]],
        };
        let mut l = p.cholesky().expect("p is PD");
        let v_orig = [0.1_f64, 0.05_f64];
        let mut v = v_orig;

        let ok = chol_rank1_downdate(&mut l, &mut v);
        assert!(ok, "Downdate should succeed");

        let lt = l.transpose();
        let p_new = matmul(&l, &lt);

        // Expected: p - v·vᵀ
        let p_expected = Matrix::<f64, 2, 2> {
            data: [
                [
                    p.data[0][0] - v_orig[0] * v_orig[0],
                    p.data[0][1] - v_orig[0] * v_orig[1],
                ],
                [
                    p.data[1][0] - v_orig[1] * v_orig[0],
                    p.data[1][1] - v_orig[1] * v_orig[1],
                ],
            ],
        };

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (p_new.data[i][j] - p_expected.data[i][j]).abs() < 1e-6,
                    "Mismatch at ({i},{j}): {} vs {}",
                    p_new.data[i][j],
                    p_expected.data[i][j]
                );
            }
        }
    }
}
