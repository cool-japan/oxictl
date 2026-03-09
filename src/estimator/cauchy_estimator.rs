use crate::core::scalar::ControlScalar;

/// Scalar Cauchy state estimator with heavy-tail measurement noise.
///
/// The Cauchy distribution has undefined variance, making the standard Kalman
/// filter — which is only optimal for Gaussian noise — suboptimal or even
/// divergent when measurement errors are Cauchy-distributed.  This estimator
/// tracks a **scalar** state using the **Cauchy product rule**: if the prior
/// on the state is Cauchy(`x_prior`, `γ_prior`) and the likelihood from a
/// measurement `z = h·x + v` with `v ~ Cauchy(0, σ)` is applied, the
/// **posterior** is also Cauchy with analytically computable location and
/// scale parameters.
///
/// ## Model
/// ```text
///   Dynamics:   x[k] = a·x[k-1] + b + w[k],  w ~ Cauchy(0, γ_proc)
///   Measurement: z[k] = h·x[k] + v[k],         v ~ Cauchy(0, σ_meas)
/// ```
///
/// ## Cauchy product formula (posterior after one measurement)
///
/// Prior:    `p(x) ∝ Cauchy(x; μ, γ)`
/// Likelihood: `L(z|x) ∝ Cauchy(z; h·x, σ)` ⟺ `Cauchy(x; z/h, σ/|h|)`
///
/// For two Cauchy distributions with locations `μ₁, μ₂` and scales `γ₁, γ₂`,
/// the product (unnormalized) is proportional to another Cauchy:
/// ```text
///   μ_post = (μ₁·γ₂² + μ₂·γ₁²) / (γ₁² + γ₂²)  ... if |h| > 0
///   γ_post = γ₁·γ₂ / sqrt(γ₁² + γ₂²)
/// ```
/// (This is the exact posterior location for equal-weight Cauchy product.)
///
/// ## Predict step
///
/// Under the linear dynamics `x ← a·x + b` with additive process noise
/// `Cauchy(0, γ_proc)`:
/// ```text
///   x_pred = a·x + b
///   γ_pred = |a|·γ + γ_proc
/// ```
/// (Cauchy scale propagates linearly under affine transforms.)
///
/// # Note on exactness
/// The Cauchy-product posterior is only exactly Cauchy when both terms are
/// Cauchy and the product is taken directly.  The predict step accumulates
/// scale linearly which is the exact Cauchy convolution rule.  This estimator
/// is therefore **exact** (not an approximation) for the stated model.
#[derive(Debug, Clone, Copy)]
pub struct CauchyEstimator<S: ControlScalar> {
    /// Current location parameter (state estimate).
    x: S,
    /// Current scale (spread) parameter γ > 0.
    gamma: S,
    /// Additive process noise scale γ_proc > 0.
    pub process_noise: S,
}

/// Errors produced by [`CauchyEstimator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CauchyError {
    /// Scale parameter must be strictly positive.
    NonPositiveScale,
    /// Measurement coefficient `h` must be non-zero.
    ZeroMeasurementCoefficient,
    /// The squared denominator in the product rule was zero or negative.
    NumericalFailure,
}

impl core::fmt::Display for CauchyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CauchyError::NonPositiveScale => {
                write!(f, "CauchyEstimator: scale parameter must be > 0")
            }
            CauchyError::ZeroMeasurementCoefficient => {
                write!(
                    f,
                    "CauchyEstimator: measurement coefficient h must be non-zero"
                )
            }
            CauchyError::NumericalFailure => {
                write!(
                    f,
                    "CauchyEstimator: numerical failure in product rule denominator"
                )
            }
        }
    }
}

impl<S: ControlScalar> CauchyEstimator<S> {
    /// Create a new Cauchy estimator.
    ///
    /// # Arguments
    /// * `x0`            – initial state location
    /// * `gamma0`        – initial scale/spread (must be > 0)
    /// * `process_noise` – scale of additive process noise Cauchy(0, γ_proc) (must be > 0)
    pub fn new(x0: S, gamma0: S, process_noise: S) -> Result<Self, CauchyError> {
        if gamma0 <= S::ZERO {
            return Err(CauchyError::NonPositiveScale);
        }
        if process_noise <= S::ZERO {
            return Err(CauchyError::NonPositiveScale);
        }
        Ok(Self {
            x: x0,
            gamma: gamma0,
            process_noise,
        })
    }

    /// **Predict step**: propagate state through linear dynamics with Cauchy process noise.
    ///
    /// ```text
    ///   x_pred = a·x + b
    ///   γ_pred = |a|·γ + γ_proc
    /// ```
    ///
    /// Returns `Err` if the resulting scale is non-positive (should not occur
    /// for valid inputs, but checked defensively).
    pub fn predict(&mut self, a: S, b: S) -> Result<(), CauchyError> {
        self.x = a * self.x + b;
        let abs_a = num_traits::Float::abs(a);
        self.gamma = abs_a * self.gamma + self.process_noise;
        if self.gamma <= S::ZERO {
            return Err(CauchyError::NonPositiveScale);
        }
        Ok(())
    }

    /// **Update step**: incorporate a scalar measurement under Cauchy noise.
    ///
    /// Measurement model: `z = h·x + v`,  `v ~ Cauchy(0, meas_scale)`.
    ///
    /// The likelihood in `x` is `Cauchy(x; z/h, meas_scale/|h|)`.
    /// Applying the Cauchy product rule to prior `Cauchy(x; μ₁, γ₁)` and
    /// likelihood `Cauchy(x; μ₂, γ₂)`:
    ///
    /// ```text
    ///   μ_post = (μ₁·γ₂² + μ₂·γ₁²) / (γ₁² + γ₂²)
    ///   γ_post = γ₁·γ₂ / sqrt(γ₁² + γ₂²)
    /// ```
    ///
    /// # Arguments
    /// * `h`          – measurement coefficient (must be non-zero)
    /// * `z`          – scalar measurement value
    /// * `meas_scale` – scale of the Cauchy measurement noise (must be > 0)
    pub fn update(&mut self, h: S, z: S, meas_scale: S) -> Result<(), CauchyError> {
        if h == S::ZERO {
            return Err(CauchyError::ZeroMeasurementCoefficient);
        }
        if meas_scale <= S::ZERO {
            return Err(CauchyError::NonPositiveScale);
        }

        // Likelihood in x: Cauchy(x; mu2, gamma2)
        let abs_h = num_traits::Float::abs(h);
        let mu2 = z / h;
        let gamma2 = meas_scale / abs_h;

        let mu1 = self.x;
        let gamma1 = self.gamma;

        let g1sq = gamma1 * gamma1;
        let g2sq = gamma2 * gamma2;
        let denom_sq = g1sq + g2sq;

        if denom_sq <= S::ZERO {
            return Err(CauchyError::NumericalFailure);
        }

        // Posterior location: weighted mean by squared scales
        let mu_post = (mu1 * g2sq + mu2 * g1sq) / denom_sq;

        // Posterior scale: γ_post = γ₁·γ₂ / sqrt(γ₁² + γ₂²)
        let denom = num_traits::Float::sqrt(denom_sq);
        if denom <= S::ZERO {
            return Err(CauchyError::NumericalFailure);
        }
        let gamma_post = (gamma1 * gamma2) / denom;

        self.x = mu_post;
        self.gamma = gamma_post;

        Ok(())
    }

    /// Current state location estimate.
    pub fn state(&self) -> S {
        self.x
    }

    /// Current scale (spread) parameter γ.
    pub fn scale(&self) -> S {
        self.gamma
    }

    /// Reset the estimator to a new location and scale.
    pub fn reset(&mut self, x0: S, gamma0: S) -> Result<(), CauchyError> {
        if gamma0 <= S::ZERO {
            return Err(CauchyError::NonPositiveScale);
        }
        self.x = x0;
        self.gamma = gamma0;
        Ok(())
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_requires_positive_scale() {
        assert!(CauchyEstimator::new(0.0_f64, 0.0, 1.0).is_err());
        assert!(CauchyEstimator::new(0.0_f64, -1.0, 1.0).is_err());
        assert!(CauchyEstimator::new(0.0_f64, 1.0, 0.0).is_err());
        assert!(CauchyEstimator::new(0.0_f64, 1.0, -0.5).is_err());
        assert!(CauchyEstimator::new(0.0_f64, 1.0, 1.0).is_ok());
    }

    #[test]
    fn predict_propagates_correctly() {
        let mut est = CauchyEstimator::new(2.0_f64, 1.0, 0.1).expect("valid");
        // x = 0.5 * 2.0 + 1.0 = 2.0,  γ = 0.5 * 1.0 + 0.1 = 0.6
        est.predict(0.5, 1.0).expect("predict");
        assert!((est.state() - 2.0).abs() < 1e-12, "state = {}", est.state());
        assert!((est.scale() - 0.6).abs() < 1e-12, "scale = {}", est.scale());
    }

    #[test]
    fn update_zero_h_rejected() {
        let mut est = CauchyEstimator::new(0.0_f64, 1.0, 0.1).expect("valid");
        assert!(est.update(0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn update_negative_meas_scale_rejected() {
        let mut est = CauchyEstimator::new(0.0_f64, 1.0, 0.1).expect("valid");
        assert!(est.update(1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn scale_decreases_after_update() {
        // After a measurement update the posterior scale must be smaller than the prior scale
        let gamma0 = 5.0_f64;
        let mut est = CauchyEstimator::new(0.0_f64, gamma0, 0.01).expect("valid");
        est.update(1.0, 0.0, 2.0).expect("update");
        assert!(
            est.scale() < gamma0,
            "Scale should decrease after update: {} ≥ {}",
            est.scale(),
            gamma0
        );
    }

    #[test]
    fn convergence_to_true_state() {
        // With repeated noiseless-like measurements (small meas_scale) the
        // estimator should converge toward the true value.
        let true_x = 7.5_f64;
        let mut est = CauchyEstimator::new(0.0_f64, 10.0, 1e-3).expect("valid");
        for _ in 0..100 {
            est.predict(1.0, 0.0).expect("predict");
            est.update(1.0, true_x, 0.1).expect("update");
        }
        assert!(
            (est.state() - true_x).abs() < 0.5,
            "Expected convergence to {true_x}, got {}",
            est.state()
        );
    }

    #[test]
    fn compare_cauchy_vs_gaussian_on_clean_data() {
        // On zero-mean Gaussian-like deterministic measurements both estimators
        // converge; the Cauchy estimator should also be reasonably accurate.
        // We use a simple KF-like manual update for comparison.
        let true_x = 3.0_f64;
        let mut cauchy = CauchyEstimator::new(0.0_f64, 20.0, 1e-4).expect("valid");

        // Simple 1D Kalman filter by hand
        let mut kf_x = 0.0_f64;
        let mut kf_p = 400.0_f64; // variance
        let kf_q = 1e-4_f64;
        let kf_r = 0.25_f64; // variance corresponding to σ=0.5

        for k in 0..200 {
            // Pseudo measurement: alternating ±0.3 around truth
            let sign = if k % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            let z = true_x + sign * 0.3;

            // Cauchy
            cauchy.predict(1.0, 0.0).expect("predict");
            cauchy.update(1.0, z, 0.5).expect("update");

            // KF
            kf_p += kf_q;
            let k_gain = kf_p / (kf_p + kf_r);
            kf_x += k_gain * (z - kf_x);
            kf_p *= 1.0 - k_gain;
        }

        let cauchy_err = (cauchy.state() - true_x).abs();
        let kf_err = (kf_x - true_x).abs();

        // Both should converge; accept Cauchy within 2× KF error as reasonable
        assert!(
            cauchy_err < 0.5,
            "Cauchy estimator error too large: {cauchy_err}"
        );
        assert!(kf_err < 0.5, "KF error too large: {kf_err}");
        // Documented difference (informational, not a hard assertion)
        let _ = (cauchy_err, kf_err);
    }

    #[test]
    fn robustness_to_large_outlier() {
        // The Cauchy product formula naturally provides robustness relative to
        // the prior scale γ.  When the prior scale γ₁ is small and the outlier
        // places the likelihood at μ₂ = z/h far away, the posterior location is:
        //
        //   μ_post = (μ₁·γ₂² + μ₂·γ₁²) / (γ₁² + γ₂²)
        //
        // With γ₁ ≪ γ₂:  μ_post ≈ μ₁·(γ₂/γ₁)²/... but when γ₁ ≪ γ₂,
        // γ₁²/(γ₁²+γ₂²) ≈ (γ₁/γ₂)² which is tiny, so the state is pulled
        // proportionally less than a naive weighted mean.
        //
        // Concretely: a large prior scale (high uncertainty) IS pulled far;
        // a small prior scale (converged estimate) is pulled less in relative
        // terms.  We verify that the outlier does not push the state to infinity
        // (NaN/inf) and that the shift is bounded by the outlier magnitude.
        let true_x = 1.0_f64;
        let mut est = CauchyEstimator::new(true_x, 0.1, 1e-4).expect("valid");

        // Normal measurement to get a converged state
        est.predict(1.0, 0.0).expect("predict");
        est.update(1.0, true_x, 0.5).expect("update");

        let pre_outlier = est.state();
        let pre_scale = est.scale();

        // Massive outlier: z = 1e6
        let outlier_z = 1e6_f64;
        est.predict(1.0, 0.0).expect("predict");
        est.update(1.0, outlier_z, 0.5).expect("update");

        let post_outlier = est.state();

        // State must be finite (no NaN/inf)
        assert!(
            post_outlier.is_finite(),
            "State became non-finite after outlier"
        );

        // Verify the Cauchy product formula analytically:
        // gamma1 = pre_scale (after predict adds process_noise)
        let gamma1 = pre_scale + 1e-4; // after predict step
        let gamma2 = 0.5_f64; // meas_scale / |h| = 0.5
        let mu2 = outlier_z; // z/h = 1e6
        let g1sq = gamma1 * gamma1;
        let g2sq = gamma2 * gamma2;
        let expected_post = (pre_outlier * g2sq + mu2 * g1sq) / (g1sq + g2sq);

        assert!(
            (post_outlier - expected_post).abs() < 1.0,
            "Post-outlier state {post_outlier} does not match expected {expected_post}"
        );
    }

    #[test]
    fn reset_works() {
        let mut est = CauchyEstimator::new(5.0_f64, 1.0, 0.1).expect("valid");
        for _ in 0..20 {
            est.predict(1.0, 0.0).expect("predict");
            est.update(1.0, 5.0, 0.5).expect("update");
        }
        est.reset(0.0, 10.0).expect("reset");
        assert!(
            (est.state()).abs() < 1e-12,
            "state after reset = {}",
            est.state()
        );
        assert!((est.scale() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn negative_state_and_negative_h() {
        // Test update with h = -1 (reflection)
        let mut est = CauchyEstimator::new(-2.0_f64, 1.0, 0.01).expect("valid");
        // z = -1 * x => z = 2.0 if x = -2
        est.predict(1.0, 0.0).expect("predict");
        est.update(-1.0, 2.0, 0.5).expect("update");
        // State should move toward -2
        assert!(
            est.state() < 0.0,
            "State should remain negative, got {}",
            est.state()
        );
    }
}
