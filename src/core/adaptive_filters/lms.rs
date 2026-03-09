//! LMS family of adaptive FIR filters.
//!
//! Provides three variants:
//! - [`LmsFilter`]: Standard Least Mean Squares
//! - [`NlmsFilter`]: Normalized LMS (step size normalized by signal power)
//! - [`VssLmsFilter`]: Variable Step-Size LMS (adapts mu based on error magnitude)
//!
//! All filters are generic over a [`ControlScalar`] type and a const filter
//! length `N`, and are `no_std` compatible.

use crate::core::scalar::ControlScalar;
use core::fmt;

// ─────────────────────────────────────────────────────────────
//  Error type
// ─────────────────────────────────────────────────────────────

/// Errors returned by adaptive filter operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptiveFilterError {
    /// Step size mu is zero, negative, or otherwise invalid.
    InvalidStepSize,
    /// Filter has diverged (weights or error became non-finite).
    Divergence,
    /// Input data is malformed (e.g., zero-length).
    InvalidInput,
}

impl fmt::Display for AdaptiveFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStepSize => write!(f, "invalid step size"),
            Self::Divergence => write!(f, "filter diverged"),
            Self::InvalidInput => write!(f, "invalid input"),
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Helper: dot product
// ─────────────────────────────────────────────────────────────

#[inline]
fn dot<S: ControlScalar, const N: usize>(a: &[S; N], b: &[S; N]) -> S {
    let mut acc = S::ZERO;
    for i in 0..N {
        acc += a[i] * b[i];
    }
    acc
}

// ─────────────────────────────────────────────────────────────
//  LmsFilter
// ─────────────────────────────────────────────────────────────

/// Standard LMS adaptive FIR filter.
///
/// Weight update rule:
/// ```text
/// y[n]   = w[n]^T * x[n]
/// e[n]   = d[n] - y[n]
/// w[n+1] = w[n] + 2·mu·e[n]·x[n]
/// ```
///
/// Convergence requires `0 < mu < 1 / (N * P_x)` where `P_x` is the input power.
#[derive(Debug, Clone)]
pub struct LmsFilter<S: ControlScalar, const N: usize> {
    weights: [S; N],
    mu: S,
}

impl<S: ControlScalar, const N: usize> LmsFilter<S, N> {
    /// Create a new LMS filter with step size `mu` and zero-initialized weights.
    ///
    /// # Errors
    /// Returns [`AdaptiveFilterError::InvalidStepSize`] if `mu <= 0`.
    pub fn new(mu: S) -> Result<Self, AdaptiveFilterError> {
        if mu <= S::ZERO {
            return Err(AdaptiveFilterError::InvalidStepSize);
        }
        Ok(Self {
            weights: [S::ZERO; N],
            mu,
        })
    }

    /// Process one sample: compute output, update weights, return output.
    ///
    /// # Arguments
    /// * `x` — input signal vector of length N (most recent sample at index 0)
    /// * `d` — desired (reference) signal sample
    ///
    /// # Errors
    /// Returns [`AdaptiveFilterError::Divergence`] if weights become non-finite.
    pub fn update(&mut self, x: &[S; N], d: S) -> Result<S, AdaptiveFilterError> {
        let y = dot(&self.weights, x);
        let e = d - y;
        let two_mu = S::TWO * self.mu;
        for (w, &xi) in self.weights.iter_mut().zip(x.iter()) {
            *w += two_mu * e * xi;
            if !w.is_finite() {
                return Err(AdaptiveFilterError::Divergence);
            }
        }
        Ok(y)
    }

    /// Return a reference to the current weight vector.
    pub fn weights(&self) -> &[S; N] {
        &self.weights
    }

    /// Reset weights to zero.
    pub fn reset(&mut self) {
        self.weights = [S::ZERO; N];
    }
}

// ─────────────────────────────────────────────────────────────
//  NlmsFilter
// ─────────────────────────────────────────────────────────────

/// Normalized LMS (NLMS) adaptive FIR filter.
///
/// The step size is normalized by the instantaneous input signal power,
/// which improves convergence speed and robustness to input power variations.
///
/// Weight update rule:
/// ```text
/// y[n]   = w[n]^T * x[n]
/// e[n]   = d[n] - y[n]
/// w[n+1] = w[n] + (mu / (x^T x + eps)) · e[n] · x[n]
/// ```
///
/// Stability is guaranteed for `0 < mu < 2`.
#[derive(Debug, Clone)]
pub struct NlmsFilter<S: ControlScalar, const N: usize> {
    weights: [S; N],
    mu: S,
    eps: S,
}

impl<S: ControlScalar, const N: usize> NlmsFilter<S, N> {
    /// Create a new NLMS filter.
    ///
    /// # Arguments
    /// * `mu`  — normalized step size; must be in `(0, 2)` for stability
    /// * `eps` — regularization constant to prevent division by zero; must be > 0
    ///
    /// # Errors
    /// Returns [`AdaptiveFilterError::InvalidStepSize`] if `mu <= 0` or `eps <= 0`.
    pub fn new(mu: S, eps: S) -> Result<Self, AdaptiveFilterError> {
        if mu <= S::ZERO || eps <= S::ZERO {
            return Err(AdaptiveFilterError::InvalidStepSize);
        }
        Ok(Self {
            weights: [S::ZERO; N],
            mu,
            eps,
        })
    }

    /// Process one sample and return the filter output.
    ///
    /// # Errors
    /// Returns [`AdaptiveFilterError::Divergence`] if weights become non-finite.
    pub fn update(&mut self, x: &[S; N], d: S) -> Result<S, AdaptiveFilterError> {
        let y = dot(&self.weights, x);
        let e = d - y;
        let power = dot(x, x) + self.eps;
        let step = self.mu / power;
        for (w, &xi) in self.weights.iter_mut().zip(x.iter()) {
            *w += step * e * xi;
            if !w.is_finite() {
                return Err(AdaptiveFilterError::Divergence);
            }
        }
        Ok(y)
    }

    /// Return a reference to the current weight vector.
    pub fn weights(&self) -> &[S; N] {
        &self.weights
    }

    /// Reset weights to zero.
    pub fn reset(&mut self) {
        self.weights = [S::ZERO; N];
    }
}

// ─────────────────────────────────────────────────────────────
//  VssLmsFilter
// ─────────────────────────────────────────────────────────────

/// Variable Step-Size LMS (VSS-LMS) adaptive FIR filter.
///
/// The step size `mu` is adapted at each iteration based on the instantaneous
/// squared error and input power, allowing faster initial convergence and lower
/// steady-state misadjustment than fixed-step LMS.
///
/// Step-size update:
/// ```text
/// mu[n+1] = clamp(alpha·mu[n] + (1-alpha)·e²[n] / (x^T x + eps), mu_min, mu_max)
/// ```
///
/// Weight update:
/// ```text
/// w[n+1] = w[n] + mu[n]·e[n]·x[n]
/// ```
#[derive(Debug, Clone)]
pub struct VssLmsFilter<S: ControlScalar, const N: usize> {
    weights: [S; N],
    mu: S,
    mu_min: S,
    mu_max: S,
    alpha: S,
}

impl<S: ControlScalar, const N: usize> VssLmsFilter<S, N> {
    /// Create a new VSS-LMS filter.
    ///
    /// # Arguments
    /// * `mu_init` — initial step size; must be positive
    /// * `mu_min`  — minimum allowed step size; must be positive
    /// * `mu_max`  — maximum allowed step size; must be > `mu_min`
    /// * `alpha`   — exponential smoothing factor for step-size; must be in `(0, 1)`
    ///
    /// # Errors
    /// Returns [`AdaptiveFilterError::InvalidStepSize`] if any parameter is invalid.
    pub fn new(mu_init: S, mu_min: S, mu_max: S, alpha: S) -> Result<Self, AdaptiveFilterError> {
        if mu_init <= S::ZERO || mu_min <= S::ZERO || mu_max <= mu_min {
            return Err(AdaptiveFilterError::InvalidStepSize);
        }
        if alpha <= S::ZERO || alpha >= S::ONE {
            return Err(AdaptiveFilterError::InvalidStepSize);
        }
        Ok(Self {
            weights: [S::ZERO; N],
            mu: mu_init,
            mu_min,
            mu_max,
            alpha,
        })
    }

    /// Process one sample and return the filter output.
    ///
    /// The step size is adapted *before* the weight update so that the current
    /// error drives the next step size.
    ///
    /// # Errors
    /// Returns [`AdaptiveFilterError::Divergence`] if weights become non-finite.
    pub fn update(&mut self, x: &[S; N], d: S) -> Result<S, AdaptiveFilterError> {
        let y = dot(&self.weights, x);
        let e = d - y;
        let power = dot(x, x) + S::from_f64(1e-10);
        // Adapt step size
        let one_minus_alpha = S::ONE - self.alpha;
        let mu_new = self.alpha * self.mu + one_minus_alpha * e * e / power;
        self.mu = mu_new.clamp_val(self.mu_min, self.mu_max);
        // Weight update
        let mu = self.mu;
        for (w, &xi) in self.weights.iter_mut().zip(x.iter()) {
            *w += mu * e * xi;
            if !w.is_finite() {
                return Err(AdaptiveFilterError::Divergence);
            }
        }
        Ok(y)
    }

    /// Return a reference to the current weight vector.
    pub fn weights(&self) -> &[S; N] {
        &self.weights
    }

    /// Return the current step size.
    pub fn mu(&self) -> S {
        self.mu
    }

    /// Reset weights to zero and restore the initial step size.
    pub fn reset(&mut self, mu_init: S) {
        self.weights = [S::ZERO; N];
        self.mu = mu_init;
    }
}

// ─────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Simple deterministic LCG for reproducible test signals (no rand crate).
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_f64(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map to [-1, 1]
            let u = (self.state >> 11) as f64 / (1u64 << 53) as f64;
            u * 2.0 - 1.0
        }
    }

    /// Apply a 4-tap FIR filter to produce a desired signal.
    fn apply_fir(x_history: &[f64], coeffs: &[f64; 4]) -> f64 {
        let mut y = 0.0_f64;
        for (k, &c) in coeffs.iter().enumerate() {
            if k < x_history.len() {
                y += c * x_history[x_history.len() - 1 - k];
            }
        }
        y
    }

    #[test]
    fn lms_converges_to_fir() {
        // Unknown system: 4-tap FIR with coefficients [0.5, 0.3, 0.1, 0.05]
        let true_coeffs: [f64; 4] = [0.5, 0.3, 0.1, 0.05];
        let mut filter = LmsFilter::<f64, 4>::new(0.01).expect("valid mu");

        let mut lcg = Lcg::new(42);
        let mut x_history = [0.0_f64; 4];
        let mut mse = 0.0_f64;

        let n_iter = 5000usize;
        for iter in 0..n_iter {
            // Shift input history
            x_history[3] = x_history[2];
            x_history[2] = x_history[1];
            x_history[1] = x_history[0];
            x_history[0] = lcg.next_f64();

            let d = apply_fir(&x_history, &true_coeffs);
            let x_arr: [f64; 4] = x_history;
            let y = filter.update(&x_arr, d).expect("no divergence");
            let e = d - y;
            if iter >= n_iter - 500 {
                mse += e * e;
            }
        }
        mse /= 500.0;
        assert!(mse < 1e-4, "LMS did not converge: MSE = {mse}");
    }

    #[test]
    fn nlms_converges_to_fir() {
        let true_coeffs: [f64; 4] = [0.5, 0.3, 0.1, 0.05];
        let mut filter = NlmsFilter::<f64, 4>::new(0.5, 1e-6).expect("valid params");

        let mut lcg = Lcg::new(123);
        let mut x_history = [0.0_f64; 4];
        let mut mse = 0.0_f64;
        let n_iter = 3000usize;

        for iter in 0..n_iter {
            x_history[3] = x_history[2];
            x_history[2] = x_history[1];
            x_history[1] = x_history[0];
            x_history[0] = lcg.next_f64();

            let d = apply_fir(&x_history, &true_coeffs);
            let x_arr: [f64; 4] = x_history;
            let y = filter.update(&x_arr, d).expect("no divergence");
            let e = d - y;
            if iter >= n_iter - 500 {
                mse += e * e;
            }
        }
        mse /= 500.0;
        assert!(mse < 1e-4, "NLMS did not converge: MSE = {mse}");
    }

    #[test]
    fn vslms_stability_and_convergence() {
        let true_coeffs: [f64; 4] = [0.4, 0.2, 0.15, 0.1];
        let mut filter = VssLmsFilter::<f64, 4>::new(0.05, 1e-4, 0.5, 0.9).expect("valid params");

        let mut lcg = Lcg::new(7);
        let mut x_history = [0.0_f64; 4];
        let mut mse = 0.0_f64;
        let n_iter = 5000usize;

        for iter in 0..n_iter {
            x_history[3] = x_history[2];
            x_history[2] = x_history[1];
            x_history[1] = x_history[0];
            x_history[0] = lcg.next_f64();

            let d = apply_fir(&x_history, &true_coeffs);
            let x_arr: [f64; 4] = x_history;
            let y = filter.update(&x_arr, d).expect("no divergence");
            let e = d - y;

            // Verify step size stays within bounds
            let mu = filter.mu();
            assert!((1e-4..=0.5).contains(&mu), "mu out of bounds: {mu}");

            if iter >= n_iter - 500 {
                mse += e * e;
            }
        }
        mse /= 500.0;
        assert!(mse < 1e-3, "VSS-LMS did not converge: MSE = {mse}");
    }

    #[test]
    fn nlms_converges_faster_than_lms_at_early_iterations() {
        // Measure MSE after a fixed number of early iterations.
        // NLMS with mu=0.5 should outperform LMS with mu=0.01 initially.
        let true_coeffs: [f64; 4] = [0.5, 0.3, 0.1, 0.05];
        let mut lms = LmsFilter::<f64, 4>::new(0.01).expect("valid mu");
        let mut nlms = NlmsFilter::<f64, 4>::new(0.5, 1e-6).expect("valid params");

        let mut lcg_lms = Lcg::new(99);
        let mut lcg_nlms = Lcg::new(99);
        let mut x_lms = [0.0_f64; 4];
        let mut x_nlms = [0.0_f64; 4];
        let n_early = 300usize;
        let mut mse_lms = 0.0_f64;
        let mut mse_nlms = 0.0_f64;

        for iter in 0..n_early {
            // Same random input for both
            let new_sample_lms = lcg_lms.next_f64();
            let new_sample_nlms = lcg_nlms.next_f64();
            x_lms[3] = x_lms[2];
            x_lms[2] = x_lms[1];
            x_lms[1] = x_lms[0];
            x_lms[0] = new_sample_lms;
            x_nlms[3] = x_nlms[2];
            x_nlms[2] = x_nlms[1];
            x_nlms[1] = x_nlms[0];
            x_nlms[0] = new_sample_nlms;

            let d_lms = apply_fir(&x_lms, &true_coeffs);
            let d_nlms = apply_fir(&x_nlms, &true_coeffs);
            let y_lms = lms.update(&x_lms, d_lms).expect("ok");
            let y_nlms = nlms.update(&x_nlms, d_nlms).expect("ok");

            if iter >= n_early - 100 {
                mse_lms += (d_lms - y_lms).powi(2);
                mse_nlms += (d_nlms - y_nlms).powi(2);
            }
        }
        mse_lms /= 100.0;
        mse_nlms /= 100.0;
        // NLMS should have lower MSE at early iterations
        assert!(
            mse_nlms < mse_lms,
            "Expected NLMS MSE ({mse_nlms:.6}) < LMS MSE ({mse_lms:.6}) at early iterations"
        );
    }

    #[test]
    fn lms_invalid_step_size() {
        assert_eq!(
            LmsFilter::<f64, 4>::new(0.0).unwrap_err(),
            AdaptiveFilterError::InvalidStepSize
        );
        assert_eq!(
            LmsFilter::<f64, 4>::new(-0.1).unwrap_err(),
            AdaptiveFilterError::InvalidStepSize
        );
    }

    #[test]
    fn nlms_invalid_params() {
        assert_eq!(
            NlmsFilter::<f64, 4>::new(0.0, 1e-6).unwrap_err(),
            AdaptiveFilterError::InvalidStepSize
        );
        assert_eq!(
            NlmsFilter::<f64, 4>::new(0.5, 0.0).unwrap_err(),
            AdaptiveFilterError::InvalidStepSize
        );
    }

    #[test]
    fn vslms_invalid_params() {
        // mu_max <= mu_min
        assert!(VssLmsFilter::<f64, 4>::new(0.05, 0.1, 0.05, 0.9).is_err());
        // alpha out of (0,1)
        assert!(VssLmsFilter::<f64, 4>::new(0.05, 1e-4, 0.5, 0.0).is_err());
        assert!(VssLmsFilter::<f64, 4>::new(0.05, 1e-4, 0.5, 1.0).is_err());
    }

    #[test]
    fn lms_weight_retrieval() {
        let mut filter = LmsFilter::<f64, 4>::new(0.01).expect("valid");
        let x = [1.0_f64, 0.0, 0.0, 0.0];
        let _ = filter.update(&x, 0.5).expect("ok");
        let w = filter.weights();
        // After one step: w[0] += 2 * 0.01 * 0.5 * 1.0 = 0.01
        assert!((w[0] - 0.01).abs() < 1e-12, "w[0] = {}", w[0]);
    }

    #[test]
    fn lms_reset_clears_weights() {
        let mut filter = LmsFilter::<f64, 4>::new(0.05).expect("valid");
        let x = [1.0_f64, 1.0, 1.0, 1.0];
        for _ in 0..100 {
            let _ = filter.update(&x, 1.0).expect("ok");
        }
        filter.reset();
        for &w in filter.weights().iter() {
            assert_eq!(w, 0.0, "weight not zero after reset");
        }
    }

    #[test]
    fn lms_f32_convergence() {
        // Verify the filter also works with f32 precision
        let true_coeffs: [f32; 4] = [0.5, 0.3, 0.1, 0.05];
        let mut filter = LmsFilter::<f32, 4>::new(0.005_f32).expect("valid");
        let mut lcg = Lcg::new(55);
        let mut x_history = [0.0_f32; 4];
        let mut mse = 0.0_f32;
        let n_iter = 8000usize;
        for iter in 0..n_iter {
            x_history[3] = x_history[2];
            x_history[2] = x_history[1];
            x_history[1] = x_history[0];
            x_history[0] = lcg.next_f64() as f32;
            let d: f32 = true_coeffs
                .iter()
                .zip(x_history.iter())
                .map(|(c, x)| c * x)
                .sum();
            let y = filter.update(&x_history, d).expect("ok");
            if iter >= n_iter - 500 {
                let e = d - y;
                mse += e * e;
            }
        }
        mse /= 500.0;
        assert!(mse < 1e-3_f32, "LMS f32 did not converge: MSE = {mse}");
    }
}
