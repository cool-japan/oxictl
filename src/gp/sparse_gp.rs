//! Sparse Gaussian Process with FITC (Fully Independent Training Conditional)
//! approximation.
//!
//! For N_train >> M (number of inducing points), the FITC approximation reduces
//! the O(N³) cost of exact GP training to O(NM² + M³).  Prediction is O(M).
//!
//! Reference: Snelson & Ghahramani (2006), "Sparse Gaussian Processes using
//! Pseudo-inputs", NeurIPS.

#![allow(clippy::needless_range_loop)]

use super::cholesky::{cholesky, cholesky_solve, forward_sub};
use super::kernel::Kernel;
use super::GpError;
use crate::core::scalar::ControlScalar;

// ──────────────────────────────────────────────────────────────────────────────
// SparseGp struct
// ──────────────────────────────────────────────────────────────────────────────

/// Sparse GP with FITC approximation.
///
/// # Type Parameters
/// * `S`  – scalar type (f32 or f64)
/// * `K`  – kernel type implementing [`Kernel<S, D>`]
/// * `D`  – input dimensionality (compile-time constant)
/// * `M`  – number of inducing points (compile-time constant)
/// * `N`  – number of training points (compile-time constant)
pub struct SparseGp<S, K, const D: usize, const M: usize, const N: usize>
where
    S: ControlScalar,
    K: Kernel<S, D>,
{
    /// Covariance kernel.
    kernel: K,
    /// Observation noise variance σ²_n.
    noise_var: S,
    /// Inducing point inputs (M × D).
    inducing: [[S; D]; M],
    /// Whether `fit` has been called successfully.
    trained: bool,
    /// Posterior mean weights at inducing points (M-vector).
    mu: [S; M],
    /// Posterior Cholesky factor at inducing points (M × M).
    l_post: [[S; M]; M],
    /// Cholesky factor of Kmm (for prior subtraction in variance).
    kmm_chol: [[S; M]; M],
}

impl<S, K, const D: usize, const M: usize, const N: usize> SparseGp<S, K, D, M, N>
where
    S: ControlScalar,
    K: Kernel<S, D>,
{
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create an untrained Sparse GP.
    pub fn new(kernel: K, noise_var: S, inducing: [[S; D]; M]) -> Self {
        Self {
            kernel,
            noise_var,
            inducing,
            trained: false,
            mu: [S::ZERO; M],
            l_post: [[S::ZERO; M]; M],
            kmm_chol: [[S::ZERO; M]; M],
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Dot product of two M-vectors.
    #[inline]
    fn dot_m(a: &[S; M], b: &[S; M]) -> S {
        let mut acc = S::ZERO;
        for i in 0..M {
            acc += a[i] * b[i];
        }
        acc
    }

    // ── Training ──────────────────────────────────────────────────────────────

    /// Fit the sparse GP to training data using the FITC approximation.
    ///
    /// # Algorithm
    /// 1. Build Kmm (M×M inducing kernel matrix).
    /// 2. Cholesky-factorize Kmm (with jitter for stability).
    /// 3. Build Kmn (M×N cross-covariance).
    /// 4. Compute FITC diagonal Λ = diag(Knn) − diag(Qnn) + σ²_n.
    /// 5. Build Q_post = Kmm + Kmn Λ⁻¹ Kmn^T.
    /// 6. Solve for posterior mean μ via Cholesky on Q_post.
    pub fn fit(&mut self, x_train: [[S; D]; N], y_train: [S; N]) -> Result<(), GpError> {
        let jitter = S::from_f64(1e-8);

        // Step 1: Build Kmm (M×M)
        let mut kmm = [[S::ZERO; M]; M];
        for i in 0..M {
            for j in 0..=i {
                let kij = self.kernel.eval(&self.inducing[i], &self.inducing[j]);
                kmm[i][j] = kij;
                kmm[j][i] = kij;
            }
            kmm[i][i] += jitter;
        }

        // Step 2: Cholesky(Kmm)
        self.kmm_chol = cholesky(&kmm)?;

        // Step 3: Build Kmn (M×N)
        let mut kmn = [[S::ZERO; N]; M];
        for i in 0..M {
            for j in 0..N {
                kmn[i][j] = self.kernel.eval(&self.inducing[i], &x_train[j]);
            }
        }

        // Step 4: FITC diagonal
        // Qnn_diag[j] = col_j^T * Kmm^{-1} * col_j  where col_j = Kmn[:,j]
        // via col_j^T Kmm^{-1} col_j = ||Lmm^{-1} col_j||²
        let min_lambda = S::from_f64(1e-8);
        let mut lambda = [S::ZERO; N];
        for j in 0..N {
            // extract column j from Kmn as an M-vector
            let mut col_j = [S::ZERO; M];
            for i in 0..M {
                col_j[i] = kmn[i][j];
            }
            let v_j = forward_sub(&self.kmm_chol, &col_j)?;
            let qnn_j = Self::dot_m(&v_j, &v_j);
            let knn_j = self.kernel.eval(&x_train[j], &x_train[j]);
            let raw = knn_j - qnn_j + self.noise_var;
            lambda[j] = if raw < min_lambda { min_lambda } else { raw };
        }

        // Step 5: Build Q_post = Kmm + Kmn Λ^{-1} Kmn^T  (M×M)
        let mut q_post = [[S::ZERO; M]; M];
        // Start from Kmm
        for i in 0..M {
            for j in 0..M {
                q_post[i][j] = kmm[i][j];
            }
        }
        // Add Kmn * diag(1/lambda) * Kmn^T
        for i in 0..M {
            for j in 0..M {
                let mut acc = S::ZERO;
                for k in 0..N {
                    acc += kmn[i][k] * kmn[j][k] / lambda[k];
                }
                q_post[i][j] += acc;
            }
            // Add jitter on diagonal for numerical stability
            q_post[i][i] += jitter;
        }

        // Cholesky of Q_post for storing posterior
        self.l_post = cholesky(&q_post)?;

        // Step 6: rhs = Kmn * (Λ^{-1} y)  (M-vector)
        let mut rhs = [S::ZERO; M];
        for i in 0..M {
            let mut acc = S::ZERO;
            for k in 0..N {
                acc += kmn[i][k] * y_train[k] / lambda[k];
            }
            rhs[i] = acc;
        }

        // mu = Q_post^{-1} * rhs
        self.mu = cholesky_solve(&q_post, &rhs)?;

        self.trained = true;
        Ok(())
    }

    // ── Prediction ────────────────────────────────────────────────────────────

    /// Predict posterior mean and variance at `x_star`.
    ///
    /// # Errors
    /// Returns [`GpError::NotTrained`] if `fit` has not been called.
    pub fn predict(&self, x_star: &[S; D]) -> Result<(S, S), GpError> {
        if !self.trained {
            return Err(GpError::NotTrained);
        }

        // k_m[i] = k(x_star, z_i)  where z_i are inducing inputs
        let mut k_m = [S::ZERO; M];
        for i in 0..M {
            k_m[i] = self.kernel.eval(x_star, &self.inducing[i]);
        }

        // Posterior mean: μ* = k_m^T μ
        let mean = Self::dot_m(&k_m, &self.mu);

        // Posterior variance:
        //   σ²* = k(x*,x*) + σ²_n
        //         - k_m^T Kmm^{-1} k_m          (prior reduction)
        //         + k_m^T Q_post^{-1} k_m        (posterior boost)
        let prior_kss = self.kernel.eval(x_star, x_star) + self.noise_var;

        // v1 = Lmm^{-1} k_m  →  ||v1||² = k_m^T Kmm^{-1} k_m
        let v1 = forward_sub(&self.kmm_chol, &k_m)?;
        let prior_reduction = Self::dot_m(&v1, &v1);

        // v2 = L_post^{-1} k_m  →  ||v2||² = k_m^T Q_post^{-1} k_m
        let v2 = forward_sub(&self.l_post, &k_m)?;
        let post_boost = Self::dot_m(&v2, &v2);

        let var_raw = prior_kss - prior_reduction + post_boost;
        let var = if var_raw < S::ZERO { S::ZERO } else { var_raw };

        Ok((mean, var))
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Whether the model has been fitted.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Posterior mean weights at inducing points.
    pub fn mu(&self) -> &[S; M] {
        &self.mu
    }

    /// Noise variance.
    pub fn noise_var(&self) -> S {
        self.noise_var
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::kernel::{Matern52Kernel, RbfKernel};

    fn make_rbf() -> RbfKernel<f64> {
        RbfKernel {
            variance: 1.0,
            length_scale: 1.0,
        }
    }

    // ── 1. Basic fit N=5, M=2 ─────────────────────────────────────────────────

    #[test]
    fn fit_n5_m2_succeeds() {
        let inducing = [[0.5_f64], [1.5_f64]];
        let mut sgp: SparseGp<f64, _, 1, 2, 5> = SparseGp::new(make_rbf(), 0.1, inducing);
        let x = [[0.0_f64], [0.5], [1.0], [1.5], [2.0]];
        let y = [0.0_f64, 0.5, 1.0, 0.5, 0.0];
        assert!(sgp.fit(x, y).is_ok(), "fit must succeed");
        assert!(sgp.is_trained());
    }

    // ── 2. Predict within training range has positive variance ────────────────

    #[test]
    fn predict_variance_positive() {
        let inducing = [[0.5_f64], [1.5_f64]];
        let mut sgp: SparseGp<f64, _, 1, 2, 5> = SparseGp::new(make_rbf(), 0.1, inducing);
        let x = [[0.0_f64], [0.5], [1.0], [1.5], [2.0]];
        let y = [0.0_f64, 0.5, 1.0, 0.5, 0.0];
        sgp.fit(x, y).expect("fit must succeed");
        let (_m, var) = sgp.predict(&[1.0_f64]).expect("predict must succeed");
        assert!(var >= 0.0, "variance {var} must be non-negative");
    }

    // ── 3. M=1 degenerate case with N=3 ──────────────────────────────────────

    #[test]
    fn m1_degenerate_fits() {
        let inducing = [[1.0_f64]];
        let mut sgp: SparseGp<f64, _, 1, 1, 3> = SparseGp::new(make_rbf(), 0.1, inducing);
        let x = [[0.0_f64], [1.0], [2.0]];
        let y = [0.0_f64, 1.0, 0.0];
        assert!(sgp.fit(x, y).is_ok(), "M=1 fit must succeed");
    }

    // ── 4. Variance is always >= 0 ───────────────────────────────────────────

    #[test]
    fn variance_always_nonneg() {
        let inducing = [[0.5_f64], [1.5_f64]];
        let mut sgp: SparseGp<f64, _, 1, 2, 5> = SparseGp::new(make_rbf(), 0.01, inducing);
        let x = [[0.0_f64], [0.5], [1.0], [1.5], [2.0]];
        let y = [0.0_f64, 0.5, 1.0, 0.5, 0.0];
        sgp.fit(x, y).expect("fit must succeed");
        for xstar in [-10.0_f64, 0.0, 0.5, 1.0, 1.5, 2.0, 10.0, 100.0] {
            let (_m, var) = sgp.predict(&[xstar]).expect("predict failed");
            assert!(var >= 0.0, "variance {var} < 0 at x={xstar}");
        }
    }

    // ── 5. mu is updated (nonzero) after fit with nonzero targets ─────────────

    #[test]
    fn mu_updated_after_fit() {
        let inducing = [[0.5_f64], [1.5_f64]];
        let mut sgp: SparseGp<f64, _, 1, 2, 5> = SparseGp::new(make_rbf(), 0.1, inducing);
        let x = [[0.0_f64], [0.5], [1.0], [1.5], [2.0]];
        let y = [0.0_f64, 5.0, 10.0, 5.0, 0.0];
        sgp.fit(x, y).expect("fit must succeed");
        let mu_norm: f64 = sgp.mu().iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(
            mu_norm > 1e-6,
            "mu should be non-trivial after fit with large y"
        );
    }

    // ── 6. predict before fit returns NotTrained ──────────────────────────────

    #[test]
    fn predict_before_fit_returns_not_trained() {
        let inducing = [[0.5_f64], [1.5_f64]];
        let sgp: SparseGp<f64, _, 1, 2, 5> = SparseGp::new(make_rbf(), 0.1, inducing);
        let result = sgp.predict(&[1.0_f64]);
        assert_eq!(result, Err(GpError::NotTrained));
    }

    // ── 7. Matern52 kernel works with sparse GP ────────────────────────────────

    #[test]
    fn sparse_gp_matern52_fits() {
        let kernel = Matern52Kernel::<f64> {
            variance: 1.0,
            length_scale: 1.0,
        };
        let inducing = [[0.5_f64], [1.5_f64], [2.5_f64]];
        let mut sgp: SparseGp<f64, _, 1, 3, 6> = SparseGp::new(kernel, 0.1, inducing);
        let x = [[0.0_f64], [0.5], [1.0], [1.5], [2.0], [3.0]];
        let y = [0.0_f64, 1.0, 2.0, 1.5, 1.0, 0.5];
        assert!(sgp.fit(x, y).is_ok());
        let (_m, var) = sgp.predict(&[1.5_f64]).expect("predict failed");
        assert!(var >= 0.0, "variance must be non-negative");
    }

    // ── 8. 2D input sparse GP ─────────────────────────────────────────────────

    #[test]
    fn sparse_gp_2d_input() {
        let inducing = [[0.5_f64, 0.5], [1.5_f64, 1.5]];
        let mut sgp: SparseGp<f64, _, 2, 2, 4> = SparseGp::new(make_rbf(), 0.1, inducing);
        let x = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
        let y = [0.0_f64, 1.0, 1.0, 4.0];
        assert!(sgp.fit(x, y).is_ok());
        let (mean, var) = sgp.predict(&[1.0_f64, 1.0]).expect("predict failed");
        assert!(var >= 0.0, "variance {var} must be non-negative");
        assert!(mean.is_finite(), "mean {mean} must be finite");
    }
}
