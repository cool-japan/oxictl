//! Exact Gaussian Process Regression.
//!
//! Implements the standard GP regression model:
//!
//!   y = f(x) + ε,  f ~ GP(0, k),  ε ~ N(0, σ²_n)
//!
//! Training is O(N³) via Cholesky decomposition of the (N×N) kernel matrix.
//! Prediction is O(N) for the mean and O(N²) for the variance.

#![allow(clippy::needless_range_loop)]

use super::cholesky::{backward_sub, cholesky, forward_sub};
use super::kernel::Kernel;
use super::GpError;
use crate::core::scalar::ControlScalar;

// ──────────────────────────────────────────────────────────────────────────────
// GpRegression struct
// ──────────────────────────────────────────────────────────────────────────────

/// Exact GP regression with const-generic training set size `N`.
///
/// # Type Parameters
/// * `S`  – scalar type (f32 or f64)
/// * `K`  – kernel type implementing [`Kernel<S, D>`]
/// * `D`  – input dimensionality (compile-time constant)
/// * `N`  – number of training points (compile-time constant)
pub struct GpRegression<S, K, const D: usize, const N: usize>
where
    S: ControlScalar,
    K: Kernel<S, D>,
{
    /// Covariance kernel.
    kernel: K,
    /// Observation noise variance σ²_n.
    noise_var: S,
    /// Training inputs.
    x_train: [[S; D]; N],
    /// Training targets.
    y_train: [S; N],
    /// Lower-triangular Cholesky factor of (K + σ²_n I).
    l_chol: [[S; N]; N],
    /// α = (K + σ²_n I)⁻¹ y  (precomputed for fast prediction).
    alpha: [S; N],
    /// Whether `fit` has been called successfully.
    trained: bool,
}

impl<S, K, const D: usize, const N: usize> GpRegression<S, K, D, N>
where
    S: ControlScalar,
    K: Kernel<S, D>,
{
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new, untrained GP regression model.
    pub fn new(kernel: K, noise_var: S) -> Self {
        Self {
            kernel,
            noise_var,
            x_train: [[S::ZERO; D]; N],
            y_train: [S::ZERO; N],
            l_chol: [[S::ZERO; N]; N],
            alpha: [S::ZERO; N],
            trained: false,
        }
    }

    // ── Training ──────────────────────────────────────────────────────────────

    /// Fit the GP to training data.
    ///
    /// Builds the kernel matrix K, adds noise on the diagonal, computes the
    /// Cholesky factor L and the vector α = (K + σ²I)⁻¹ y.
    ///
    /// # Errors
    /// Returns [`GpError::NotPositiveDefinite`] if the kernel matrix is not
    /// numerically positive definite (e.g. very small noise_var with a
    /// near-singular kernel matrix).
    pub fn fit(&mut self, x_train: [[S; D]; N], y_train: [S; N]) -> Result<(), GpError> {
        self.x_train = x_train;
        self.y_train = y_train;

        // Build kernel matrix K (symmetric, so only compute lower triangle
        // and mirror).
        let mut k_mat = [[S::ZERO; N]; N];
        for i in 0..N {
            for j in 0..=i {
                let kij = self.kernel.eval(&x_train[i], &x_train[j]);
                k_mat[i][j] = kij;
                k_mat[j][i] = kij;
            }
            // Add noise variance on diagonal
            k_mat[i][i] += self.noise_var;
        }

        // Cholesky factorization
        self.l_chol = cholesky(&k_mat)?;

        // Compute α = L⁻ᵀ (L⁻¹ y)
        let v = forward_sub(&self.l_chol, &y_train)?;
        self.alpha = backward_sub(&self.l_chol, &v)?;

        self.trained = true;
        Ok(())
    }

    // ── Prediction ────────────────────────────────────────────────────────────

    /// Predict the posterior mean and variance at a test point `x_star`.
    ///
    /// Returns `(mean, variance)`.
    ///
    /// The variance is clamped to ≥ 0 to prevent numerical artifacts.
    ///
    /// # Errors
    /// Returns [`GpError::NotTrained`] if `fit` has not been called.
    pub fn predict(&self, x_star: &[S; D]) -> Result<(S, S), GpError> {
        if !self.trained {
            return Err(GpError::NotTrained);
        }

        // k_star[i] = k(x_star, x_train[i])
        let mut k_star = [S::ZERO; N];
        for i in 0..N {
            k_star[i] = self.kernel.eval(x_star, &self.x_train[i]);
        }

        // Posterior mean: μ* = k_star · α
        let mut mean = S::ZERO;
        for i in 0..N {
            mean += k_star[i] * self.alpha[i];
        }

        // Posterior variance: σ²* = k(x*,x*) + σ²_n - vᵀv  where  v = L⁻¹ k_star
        let v = forward_sub(&self.l_chol, &k_star)?;
        let mut v_dot_v = S::ZERO;
        for i in 0..N {
            v_dot_v += v[i] * v[i];
        }
        let prior_var = self.kernel.eval(x_star, x_star) + self.noise_var;
        let var_raw = prior_var - v_dot_v;
        let var = if var_raw < S::ZERO { S::ZERO } else { var_raw };

        Ok((mean, var))
    }

    // ── Log Marginal Likelihood ───────────────────────────────────────────────

    /// Compute the log marginal likelihood of the training data.
    ///
    /// log p(y | X) = −½ yᵀ α − Σᵢ log L[i][i] − N/2 · log(2π)
    ///
    /// This can be used for kernel hyperparameter optimisation.
    ///
    /// # Errors
    /// Returns [`GpError::NotTrained`] if `fit` has not been called.
    pub fn log_marginal_likelihood(&self) -> Result<S, GpError> {
        if !self.trained {
            return Err(GpError::NotTrained);
        }

        // term1 = -0.5 * yᵀ α
        let mut yt_alpha = 0.0_f64;
        for i in 0..N {
            yt_alpha += self.y_train[i].to_f64() * self.alpha[i].to_f64();
        }
        let term1 = -0.5 * yt_alpha;

        // term2 = -Σᵢ log(L[i][i])
        let mut log_det = 0.0_f64;
        for i in 0..N {
            log_det += libm::log(self.l_chol[i][i].to_f64());
        }
        let term2 = -log_det;

        // term3 = -N/2 * log(2π)
        let term3 = -(N as f64) * 0.5 * libm::log(2.0 * core::f64::consts::PI);

        Ok(S::from_f64(term1 + term2 + term3))
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Whether the GP has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Kernel hyperparameters (read-only reference).
    pub fn kernel(&self) -> &K {
        &self.kernel
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
    use crate::gp::kernel::{LinearKernel, Matern52Kernel, RbfKernel};

    fn make_rbf() -> RbfKernel<f64> {
        RbfKernel {
            variance: 1.0,
            length_scale: 1.0,
        }
    }

    // ── 1. fit on 1D data succeeds ────────────────────────────────────────────

    #[test]
    fn fit_1d_rbf_no_error() {
        let mut gp: GpRegression<f64, _, 1, 3> = GpRegression::new(make_rbf(), 0.01);
        let x = [[0.0_f64], [1.0], [2.0]];
        let y = [0.0_f64, 1.0, 0.0];
        assert!(gp.fit(x, y).is_ok());
    }

    // ── 2. predict at training point ─────────────────────────────────────────

    #[test]
    fn predict_at_training_point_close_to_target() {
        let mut gp: GpRegression<f64, _, 1, 3> = GpRegression::new(make_rbf(), 1e-6);
        let x = [[0.0_f64], [1.0], [2.0]];
        let y = [0.0_f64, 1.0, 0.0];
        gp.fit(x, y).expect("fit must succeed");

        let (mean, _var) = gp.predict(&[1.0_f64]).expect("predict must succeed");
        // With very small noise the mean should be close to y=1.0 at x=1.0
        assert!((mean - 1.0).abs() < 0.01, "mean={mean} expected ~1.0");
    }

    // ── 3. variance at training point < variance at distant point ─────────────

    #[test]
    fn variance_increases_away_from_training() {
        let mut gp: GpRegression<f64, _, 1, 3> = GpRegression::new(make_rbf(), 1e-6);
        let x = [[0.0_f64], [1.0], [2.0]];
        let y = [0.0_f64, 1.0, 0.0];
        gp.fit(x, y).expect("fit must succeed");

        let (_m_near, var_near) = gp.predict(&[1.0_f64]).expect("predict failed");
        let (_m_far, var_far) = gp.predict(&[100.0_f64]).expect("predict failed");
        assert!(
            var_far > var_near,
            "far variance {var_far} should exceed near variance {var_near}"
        );
    }

    // ── 4. log marginal likelihood is finite ──────────────────────────────────

    #[test]
    fn log_marginal_likelihood_finite() {
        let mut gp: GpRegression<f64, _, 1, 3> = GpRegression::new(make_rbf(), 0.1);
        let x = [[0.0_f64], [1.0], [2.0]];
        let y = [0.0_f64, 1.0, 0.0];
        gp.fit(x, y).expect("fit must succeed");
        let lml = gp.log_marginal_likelihood().expect("lml must succeed");
        assert!(lml.is_finite(), "lml={lml} must be finite");
    }

    // ── 5. predict before fit returns NotTrained ──────────────────────────────

    #[test]
    fn predict_before_fit_returns_not_trained() {
        let gp: GpRegression<f64, _, 1, 3> = GpRegression::new(make_rbf(), 0.1);
        let result = gp.predict(&[0.5_f64]);
        assert_eq!(result, Err(GpError::NotTrained));
    }

    // ── 6. lml before fit returns NotTrained ─────────────────────────────────

    #[test]
    fn lml_before_fit_returns_not_trained() {
        let gp: GpRegression<f64, _, 1, 3> = GpRegression::new(make_rbf(), 0.1);
        assert_eq!(gp.log_marginal_likelihood(), Err(GpError::NotTrained));
    }

    // ── 7. fit with Matern52 succeeds ─────────────────────────────────────────

    #[test]
    fn fit_matern52_succeeds() {
        let kernel = Matern52Kernel::<f64> {
            variance: 1.0,
            length_scale: 1.0,
        };
        let mut gp: GpRegression<f64, _, 1, 4> = GpRegression::new(kernel, 0.1);
        let x = [[0.0_f64], [1.0], [2.0], [3.0]];
        let y = [1.0_f64, 2.0, 1.5, 0.5];
        assert!(gp.fit(x, y).is_ok());
    }

    // ── 8. fit with LinearKernel succeeds ────────────────────────────────────

    #[test]
    fn fit_linear_kernel_succeeds() {
        let kernel = LinearKernel::<f64> {
            variance: 1.0,
            bias: 1.0,
            degree: 1,
        };
        let mut gp: GpRegression<f64, _, 1, 3> = GpRegression::new(kernel, 0.1);
        let x = [[1.0_f64], [2.0], [3.0]];
        let y = [2.0_f64, 4.0, 6.0];
        assert!(gp.fit(x, y).is_ok());
    }

    // ── 9. variance is always >= 0 ────────────────────────────────────────────

    #[test]
    fn variance_always_nonneg() {
        let mut gp: GpRegression<f64, _, 1, 3> = GpRegression::new(make_rbf(), 1e-6);
        let x = [[0.0_f64], [1.0], [2.0]];
        let y = [0.0_f64, 1.0, 0.0];
        gp.fit(x, y).expect("fit must succeed");

        for xstar in [-5.0_f64, 0.0, 0.5, 1.0, 2.0, 10.0, 100.0] {
            let (_m, var) = gp.predict(&[xstar]).expect("predict failed");
            assert!(var >= 0.0, "variance {var} < 0 at x={xstar}");
        }
    }

    // ── 10. 2D input GP fits without error ────────────────────────────────────

    #[test]
    fn fit_2d_rbf_succeeds() {
        let kernel = RbfKernel::<f64> {
            variance: 1.0,
            length_scale: 1.0,
        };
        let mut gp: GpRegression<f64, _, 2, 4> = GpRegression::new(kernel, 0.1);
        let x = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = [0.0_f64, 1.0, 1.0, 2.0];
        assert!(gp.fit(x, y).is_ok());
        let (mean, var) = gp.predict(&[0.5_f64, 0.5]).expect("predict failed");
        assert!(var >= 0.0, "variance must be non-negative, got {var}");
        assert!(mean.is_finite(), "mean must be finite, got {mean}");
    }
}
