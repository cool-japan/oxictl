/// Grünwald-Letnikov (GL) fractional calculus operators for control systems.
///
/// The GL approximation of a fractional derivative/integral of order α is:
///
///   D^α f(t) ≈ (1/h^α) · Σ_{k=0}^{N-1} w_k · f(t - k·h)
///
/// where the binomial weights satisfy the recurrence:
///   w_0 = 1
///   w_k = w_{k-1} · (k - 1 - α) / k
///
/// For α > 0: fractional derivative
/// For α < 0: fractional integral
/// For α = 1: recovers standard backward-difference derivative
/// For α = -1: recovers cumulative sum scaled by dt (standard integral)
use crate::core::scalar::ControlScalar;

use super::FracError;

/// Grünwald-Letnikov fractional operator with a fixed window of N samples.
///
/// Generic over scalar type `S` and window size `N`.
/// The window stores the last N input samples (newest at index 0).
#[derive(Debug, Clone)]
pub struct GrunwaldLeibniz<S: ControlScalar, const N: usize> {
    /// Fractional order α (may be positive or negative).
    alpha: S,
    /// Sample time h in seconds.
    sample_time: S,
    /// Precomputed GL binomial weights w_0 … w_{N-1}.
    weights: [S; N],
    /// Circular buffer storing the last N input samples (index 0 = most recent).
    window: [S; N],
    /// Number of samples seen so far (saturates at N).
    count: usize,
    /// Write position in the circular buffer (points to where the *next* write goes).
    head: usize,
}

impl<S: ControlScalar, const N: usize> GrunwaldLeibniz<S, N> {
    /// Construct a GL operator of fractional order `alpha` with sample time `h`.
    ///
    /// # Errors
    /// Returns [`FracError::InvalidOrder`] if `alpha` is NaN or infinite.
    /// Returns [`FracError::InvalidSampleTime`] if `h <= 0`.
    /// Returns [`FracError::WindowTooSmall`] if `N == 0`.
    pub fn new(alpha: S, sample_time: S) -> Result<Self, FracError> {
        if N == 0 {
            return Err(FracError::WindowTooSmall);
        }
        if !alpha.is_finite() {
            return Err(FracError::InvalidOrder);
        }
        if sample_time <= S::ZERO || !sample_time.is_finite() {
            return Err(FracError::InvalidSampleTime);
        }

        let weights = compute_gl_weights::<S, N>(alpha);

        Ok(Self {
            alpha,
            sample_time,
            weights,
            window: [S::ZERO; N],
            count: 0,
            head: 0,
        })
    }

    /// Return the fractional order.
    #[inline]
    pub fn alpha(&self) -> S {
        self.alpha
    }

    /// Feed a new sample into the operator and return D^α x at this time step.
    ///
    /// The output is scaled by `(1 / h^alpha)` so that the units are consistent
    /// with the chosen sample time.
    pub fn update(&mut self, x: S) -> S {
        // Write new sample at head position
        self.window[self.head] = x;
        self.head = (self.head + 1) % N;
        if self.count < N {
            self.count += 1;
        }

        // Accumulate: sum_{k=0}^{count-1} w_k * window[newest - k]
        let mut acc = S::ZERO;
        let effective = self.count;
        for k in 0..effective {
            // Index of the k-th oldest sample from the most-recent write
            // head was just incremented, so newest sample is at (head - 1 + N) % N
            let idx = (self.head + N - 1 - k) % N;
            acc += self.weights[k] * self.window[idx];
        }

        // Scale by h^{-alpha}
        let h_alpha = self.sample_time.powf(self.alpha);
        if h_alpha.abs() < S::EPSILON {
            S::ZERO
        } else {
            acc / h_alpha
        }
    }

    /// Reset the operator state (clear the window buffer).
    pub fn reset(&mut self) {
        self.window = [S::ZERO; N];
        self.count = 0;
        self.head = 0;
    }
}

/// Compute the N binomial GL weights for order `alpha`.
///
/// w_0 = 1,  w_k = w_{k-1} * (k - 1 - alpha) / k
fn compute_gl_weights<S: ControlScalar, const N: usize>(alpha: S) -> [S; N] {
    let mut w = [S::ZERO; N];
    if N == 0 {
        return w;
    }
    w[0] = S::ONE;
    for k in 1..N {
        // w_k = w_{k-1} * (k-1 - alpha) / k
        let k_s = S::from_f64(k as f64);
        let km1_s = S::from_f64((k - 1) as f64);
        w[k] = w[k - 1] * (km1_s - alpha) / k_s;
    }
    w
}

/// Fractional integrator: GL with order `alpha < 0`.
///
/// Convenience wrapper that enforces α ∈ (-2, 0).
#[derive(Debug, Clone)]
pub struct FracIntegrator<S: ControlScalar, const N: usize> {
    inner: GrunwaldLeibniz<S, N>,
}

impl<S: ControlScalar, const N: usize> FracIntegrator<S, N> {
    /// Construct a fractional integrator of order `lambda` ∈ (0, 2).
    ///
    /// Internally uses α = -lambda so the GL operator integrates.
    ///
    /// # Errors
    /// Returns [`FracError::InvalidOrder`] if `lambda` is outside (0, 2) or non-finite.
    pub fn new(lambda: S, sample_time: S) -> Result<Self, FracError> {
        let two = S::TWO;
        if lambda <= S::ZERO || lambda >= two || !lambda.is_finite() {
            return Err(FracError::InvalidOrder);
        }
        let alpha = S::ZERO - lambda; // α = -lambda → integrating
        Ok(Self {
            inner: GrunwaldLeibniz::new(alpha, sample_time)?,
        })
    }

    /// Feed a new sample and return the fractional integral.
    #[inline]
    pub fn update(&mut self, x: S) -> S {
        self.inner.update(x)
    }

    /// Reset the integrator state.
    #[inline]
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return the integration order λ (positive).
    #[inline]
    pub fn lambda(&self) -> S {
        S::ZERO - self.inner.alpha()
    }
}

/// Fractional differentiator: GL with order `alpha > 0`.
///
/// Convenience wrapper that enforces α ∈ (0, 2).
#[derive(Debug, Clone)]
pub struct FracDifferentiator<S: ControlScalar, const N: usize> {
    inner: GrunwaldLeibniz<S, N>,
}

impl<S: ControlScalar, const N: usize> FracDifferentiator<S, N> {
    /// Construct a fractional differentiator of order `mu` ∈ (0, 2).
    ///
    /// # Errors
    /// Returns [`FracError::InvalidOrder`] if `mu` is outside (0, 2) or non-finite.
    pub fn new(mu: S, sample_time: S) -> Result<Self, FracError> {
        let two = S::TWO;
        if mu <= S::ZERO || mu >= two || !mu.is_finite() {
            return Err(FracError::InvalidOrder);
        }
        Ok(Self {
            inner: GrunwaldLeibniz::new(mu, sample_time)?,
        })
    }

    /// Feed a new sample and return the fractional derivative.
    #[inline]
    pub fn update(&mut self, x: S) -> S {
        self.inner.update(x)
    }

    /// Reset the differentiator state.
    #[inline]
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return the differentiation order μ.
    #[inline]
    pub fn mu(&self) -> S {
        self.inner.alpha()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: run N updates with a constant input, return last output
    // -----------------------------------------------------------------------
    fn run_gl<const N: usize>(alpha: f64, h: f64, inputs: &[f64]) -> f64 {
        let mut gl = GrunwaldLeibniz::<f64, N>::new(alpha, h).expect("valid GL");
        let mut out = 0.0_f64;
        for &x in inputs {
            out = gl.update(x);
        }
        out
    }

    // -----------------------------------------------------------------------
    // Weight computation tests
    // -----------------------------------------------------------------------

    #[test]
    fn grunwald_weights_alpha_one_standard_difference() {
        // For α = 1 the GL weights are:
        //   w_0 = 1, w_1 = (0-1)/1 = -1, w_2 = -1*(1-1)/2 = 0, ...
        // i.e. the backward difference operator  f(t) - f(t-h)
        let w = compute_gl_weights::<f64, 4>(1.0_f64);
        assert!((w[0] - 1.0).abs() < 1e-12, "w[0]={}", w[0]);
        assert!((w[1] - (-1.0)).abs() < 1e-12, "w[1]={}", w[1]);
        assert!(w[2].abs() < 1e-12, "w[2]={}", w[2]);
        assert!(w[3].abs() < 1e-12, "w[3]={}", w[3]);
    }

    #[test]
    fn grunwald_weights_alpha_two_second_difference() {
        // α = 2: w_0=1, w_1=-2, w_2=1, rest 0
        let w = compute_gl_weights::<f64, 4>(2.0_f64);
        assert!((w[0] - 1.0).abs() < 1e-12);
        assert!((w[1] - (-2.0)).abs() < 1e-12);
        assert!((w[2] - 1.0).abs() < 1e-12);
        assert!(w[3].abs() < 1e-12);
    }

    #[test]
    fn grunwald_weights_alpha_neg_one_cumsum() {
        // α = -1: w_k = 1 for all k (cumulative sum / integration)
        let w = compute_gl_weights::<f64, 5>(-1.0_f64);
        for (i, &wi) in w.iter().enumerate() {
            assert!((wi - 1.0).abs() < 1e-12, "w[{}]={} expected 1.0", i, wi);
        }
    }

    // -----------------------------------------------------------------------
    // Integer-order derivative coincides with standard backward difference
    // -----------------------------------------------------------------------

    #[test]
    fn gl_alpha_one_recovers_standard_derivative() {
        // D^1 f(t) ≈ (f(t) - f(t-h)) / h
        // For f(t) = t → derivative ≈ 1.0
        let h = 0.1_f64;
        let n = 20_usize;
        let inputs: Vec<f64> = (0..n).map(|i| i as f64 * h).collect();

        // After two samples: (2h - h)/h = 1.0
        // After many samples: still 1.0 (linear ramp)
        let mut gl = GrunwaldLeibniz::<f64, 8>::new(1.0, h).expect("valid");
        let mut last = 0.0_f64;
        for &x in &inputs {
            last = gl.update(x);
        }
        // Allow tolerance for floating-point accumulation of the window tail
        // (higher-order terms of the GL expansion for α=1 are exactly 0)
        assert!((last - 1.0).abs() < 1e-10, "Expected ~1.0, got {}", last);
    }

    #[test]
    fn gl_alpha_two_recovers_second_derivative_of_linear() {
        // For f(t) = t (linear), D^2 f ≈ 0
        let h = 0.1_f64;
        let inputs: Vec<f64> = (0..10).map(|i| i as f64 * h).collect();
        let mut gl = GrunwaldLeibniz::<f64, 4>::new(2.0, h).expect("valid");
        let mut last = 0.0_f64;
        for &x in &inputs {
            last = gl.update(x);
        }
        assert!(
            last.abs() < 1e-9,
            "D^2 of linear should be ~0, got {}",
            last
        );
    }

    #[test]
    fn gl_alpha_two_recovers_second_derivative_of_quadratic() {
        // f(t) = t^2 / 2 → D^2 f = 1
        let h = 0.01_f64;
        let n = 40_usize;
        let inputs: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * h;
                t * t / 2.0
            })
            .collect();
        // With N=4 the GL window truncation dominates early; use larger N
        let mut gl = GrunwaldLeibniz::<f64, 32>::new(2.0, h).expect("valid");
        let mut last = 0.0_f64;
        for &x in &inputs {
            last = gl.update(x);
        }
        // Close to 1.0 (allow 1% error due to window truncation)
        assert!(
            (last - 1.0).abs() < 0.02,
            "D^2 of t^2/2 should be ~1.0, got {}",
            last
        );
    }

    // -----------------------------------------------------------------------
    // Step response integral test
    // -----------------------------------------------------------------------

    #[test]
    fn gl_neg_one_integrates_step() {
        // D^{-1} of unit step over time T should equal T
        // (area under a unit step of height 1 for N steps of size h = N*h)
        let h = 0.01_f64;
        let n = 50_usize;
        // After n steps with unit input, integral ≈ n*h
        let mut gl = GrunwaldLeibniz::<f64, 64>::new(-1.0, h).expect("valid");
        let mut last = 0.0_f64;
        for _ in 0..n {
            last = gl.update(1.0);
        }
        let expected = n as f64 * h; // 0.5
                                     // The GL integral accumulates: sum_{k=0}^{n-1} 1 * h = n * h
        assert!(
            (last - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            last
        );
    }

    // -----------------------------------------------------------------------
    // Half-order derivative (non-integer)
    // -----------------------------------------------------------------------

    #[test]
    fn gl_half_order_derivative_is_finite() {
        // Just verify that D^{0.5} of a ramp does not diverge or NaN
        let h = 0.01_f64;
        let out = run_gl::<16>(0.5, h, &[0.0, 0.01, 0.02, 0.03, 0.04]);
        assert!(out.is_finite(), "D^0.5 should be finite, got {}", out);
    }

    // -----------------------------------------------------------------------
    // Error conditions
    // -----------------------------------------------------------------------

    #[test]
    fn gl_error_on_nan_alpha() {
        let result = GrunwaldLeibniz::<f64, 4>::new(f64::NAN, 0.01);
        assert!(matches!(result, Err(FracError::InvalidOrder)));
    }

    #[test]
    fn gl_error_on_infinite_alpha() {
        let result = GrunwaldLeibniz::<f64, 4>::new(f64::INFINITY, 0.01);
        assert!(matches!(result, Err(FracError::InvalidOrder)));
    }

    #[test]
    fn gl_error_on_zero_sample_time() {
        let result = GrunwaldLeibniz::<f64, 4>::new(1.0, 0.0);
        assert!(matches!(result, Err(FracError::InvalidSampleTime)));
    }

    #[test]
    fn gl_error_on_negative_sample_time() {
        let result = GrunwaldLeibniz::<f64, 4>::new(1.0, -0.01);
        assert!(matches!(result, Err(FracError::InvalidSampleTime)));
    }

    // -----------------------------------------------------------------------
    // FracIntegrator wrapper
    // -----------------------------------------------------------------------

    #[test]
    fn frac_integrator_lambda_one_matches_standard_integral() {
        let h = 0.01_f64;
        let n = 50_usize;
        let mut fi = FracIntegrator::<f64, 64>::new(1.0, h).expect("valid");
        let mut last = 0.0_f64;
        for _ in 0..n {
            last = fi.update(1.0);
        }
        let expected = n as f64 * h;
        assert!(
            (last - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            last
        );
    }

    #[test]
    fn frac_integrator_invalid_lambda_errors() {
        assert!(matches!(
            FracIntegrator::<f64, 4>::new(0.0, 0.01),
            Err(FracError::InvalidOrder)
        ));
        assert!(matches!(
            FracIntegrator::<f64, 4>::new(2.0, 0.01),
            Err(FracError::InvalidOrder)
        ));
        assert!(matches!(
            FracIntegrator::<f64, 4>::new(-0.5, 0.01),
            Err(FracError::InvalidOrder)
        ));
    }

    // -----------------------------------------------------------------------
    // FracDifferentiator wrapper
    // -----------------------------------------------------------------------

    #[test]
    fn frac_differentiator_mu_one_matches_standard_derivative() {
        let h = 0.1_f64;
        let mut fd = FracDifferentiator::<f64, 8>::new(1.0, h).expect("valid");
        // Linear ramp: derivative should be 1
        let inputs: Vec<f64> = (0..15).map(|i| i as f64 * h).collect();
        let mut last = 0.0_f64;
        for &x in &inputs {
            last = fd.update(x);
        }
        assert!((last - 1.0).abs() < 1e-10, "Expected ~1.0, got {}", last);
    }

    #[test]
    fn frac_differentiator_invalid_mu_errors() {
        assert!(matches!(
            FracDifferentiator::<f64, 4>::new(0.0, 0.01),
            Err(FracError::InvalidOrder)
        ));
        assert!(matches!(
            FracDifferentiator::<f64, 4>::new(2.0, 0.01),
            Err(FracError::InvalidOrder)
        ));
    }

    // -----------------------------------------------------------------------
    // Reset clears state
    // -----------------------------------------------------------------------

    #[test]
    fn gl_reset_clears_window() {
        let mut gl = GrunwaldLeibniz::<f64, 4>::new(1.0, 0.1).expect("valid");
        for i in 0..4 {
            gl.update(i as f64);
        }
        gl.reset();
        // After reset and a single zero input, output should be zero
        let out = gl.update(0.0);
        assert_eq!(out, 0.0, "After reset, output should be zero");
    }

    #[test]
    fn frac_integrator_reset_works() {
        let mut fi = FracIntegrator::<f64, 8>::new(1.0, 0.01).expect("valid");
        for _ in 0..10 {
            fi.update(1.0);
        }
        fi.reset();
        let out = fi.update(0.0);
        assert_eq!(out, 0.0);
    }
}
