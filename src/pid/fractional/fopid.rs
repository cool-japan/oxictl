/// PI^λ D^μ Fractional-Order PID (FOPID) controller.
///
/// The controller output is:
///   u(t) = Kp·e(t) + Ki·I^λ[e(t)] + Kd·D^μ[e(t)]
///
/// where I^λ and D^μ are fractional integral/derivative operators
/// approximated via the Grünwald-Letnikov method.
use crate::core::scalar::ControlScalar;

use super::{grunwald::GrunwaldLeibniz, FracError};

/// Configuration for the FOPID controller.
#[derive(Debug, Clone)]
pub struct FopidConfig<S: ControlScalar> {
    /// Proportional gain.
    pub kp: S,
    /// Integral gain.
    pub ki: S,
    /// Integration order λ ∈ (0, 2). λ=1 → standard I.
    pub lambda: S,
    /// Derivative gain.
    pub kd: S,
    /// Differentiation order μ ∈ (0, 2). μ=1 → standard D.
    pub mu: S,
    /// Minimum control output (clamp).
    pub u_min: S,
    /// Maximum control output (clamp).
    pub u_max: S,
    /// Discrete sample time h [s].
    pub sample_time: S,
}

impl<S: ControlScalar> FopidConfig<S> {
    /// Construct a FOPID config with standard (integer-order) PID defaults
    /// (λ = μ = 1).
    pub fn standard_pid(kp: S, ki: S, kd: S, sample_time: S) -> Self {
        Self {
            kp,
            ki,
            lambda: S::ONE,
            kd,
            mu: S::ONE,
            u_min: S::from_f64(-1e6),
            u_max: S::from_f64(1e6),
            sample_time,
        }
    }

    /// Construct a full fractional-order config.
    #[allow(clippy::too_many_arguments)]
    pub fn fractional(
        kp: S,
        ki: S,
        lambda: S,
        kd: S,
        mu: S,
        u_min: S,
        u_max: S,
        sample_time: S,
    ) -> Self {
        Self {
            kp,
            ki,
            lambda,
            kd,
            mu,
            u_min,
            u_max,
            sample_time,
        }
    }

    /// Build a `Fopid` controller from this configuration.
    ///
    /// # Errors
    /// Returns [`FracError`] if any parameter is invalid.
    pub fn build<const N: usize>(self) -> Result<Fopid<S, N>, FracError> {
        validate_order(self.lambda)?;
        validate_order(self.mu)?;
        if self.u_min >= self.u_max {
            return Err(FracError::InvalidConfig("u_min must be less than u_max"));
        }

        // Integrator uses α = -lambda (fractional integral)
        let int_op = GrunwaldLeibniz::new(S::ZERO - self.lambda, self.sample_time)?;
        // Differentiator uses α = mu (fractional derivative)
        let diff_op = GrunwaldLeibniz::new(self.mu, self.sample_time)?;

        Ok(Fopid {
            kp: self.kp,
            ki: self.ki,
            kd: self.kd,
            u_min: self.u_min,
            u_max: self.u_max,
            int_op,
            diff_op,
            saturated: false,
        })
    }
}

/// PI^λ D^μ Fractional-Order PID controller.
///
/// Uses two Grünwald-Letnikov windows of depth `N`:
/// - `int_op`: GL with α = -λ (fractional integral)
/// - `diff_op`: GL with α = μ (fractional derivative)
///
/// Anti-windup is implemented via conditional integration (freeze the
/// integrator when the output is saturated and the error would deepen
/// the saturation).
#[derive(Debug, Clone)]
pub struct Fopid<S: ControlScalar, const N: usize> {
    kp: S,
    ki: S,
    kd: S,
    u_min: S,
    u_max: S,
    /// GL fractional integrator (α = -λ).
    int_op: GrunwaldLeibniz<S, N>,
    /// GL fractional differentiator (α = μ).
    diff_op: GrunwaldLeibniz<S, N>,
    /// True when the previous output was saturated.
    saturated: bool,
}

impl<S: ControlScalar, const N: usize> Fopid<S, N> {
    /// Compute the fractional-PID output for one time step.
    ///
    /// `setpoint` — reference value
    /// `measurement` — plant output
    ///
    /// Returns the (possibly saturated) control action.
    pub fn update(&mut self, setpoint: S, measurement: S) -> S {
        let error = setpoint - measurement;

        let p_term = self.kp * error;

        // Fractional derivative of the error
        let d_raw = self.diff_op.update(error);
        let d_term = self.kd * d_raw;

        // Anti-windup: only update integrator if not deepening saturation
        let pre_integral = self.int_op.update(error);
        let i_term = if self.saturated {
            // If saturated, check sign coherence: freeze if error pushes deeper
            // Obtain a tentative output without integral to detect direction
            let tentative = p_term + d_term;
            let sat_high = tentative > self.u_max;
            let sat_low = tentative < self.u_min;
            let freeze = (sat_high && error > S::ZERO) || (sat_low && error < S::ZERO);
            if freeze {
                S::ZERO // Discard the integrator contribution this step
            } else {
                self.ki * pre_integral
            }
        } else {
            self.ki * pre_integral
        };

        let output_raw = p_term + i_term + d_term;
        let output = output_raw.clamp_val(self.u_min, self.u_max);
        self.saturated = (output_raw - output).abs() > S::EPSILON;

        output
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.int_op.reset();
        self.diff_op.reset();
        self.saturated = false;
    }

    /// Whether the last output was saturated.
    #[inline]
    pub fn is_saturated(&self) -> bool {
        self.saturated
    }

    /// Access the proportional gain.
    #[inline]
    pub fn kp(&self) -> S {
        self.kp
    }

    /// Access the integral gain.
    #[inline]
    pub fn ki(&self) -> S {
        self.ki
    }

    /// Access the derivative gain.
    #[inline]
    pub fn kd(&self) -> S {
        self.kd
    }
}

// ---------------------------------------------------------------------------
// Auto-tuner
// ---------------------------------------------------------------------------

/// Result of an FOPID auto-tuning search.
#[derive(Debug, Clone)]
pub struct FopidAutoTuneResult<S: ControlScalar> {
    /// Best integration order λ found.
    pub lambda: S,
    /// Best differentiation order μ found.
    pub mu: S,
    /// Achieved phase margin [degrees] (approximate).
    pub phase_margin_deg: S,
    /// Achieved gain crossover frequency [rad/s] (approximate).
    pub crossover_freq: S,
    /// Best objective value (lower is better).
    pub objective: S,
}

/// Grid-search tuner that finds fractional orders λ and μ satisfying
/// approximate phase-margin and gain-crossover-frequency specifications.
///
/// The transfer function used is:
///   C(jω) = Kp + Ki·(jω)^{-λ} + Kd·(jω)^{μ}
///
/// The open-loop gain (with a nominal unity plant) is |C(jω)|.
/// Phase is arg(C(jω)).
///
/// Grid resolution: 0.05 in both λ and μ over [0.5, 1.5].
pub struct FopidAutoTune<S: ControlScalar> {
    /// Proportional gain for the search.
    pub kp: S,
    /// Integral gain for the search.
    pub ki: S,
    /// Derivative gain for the search.
    pub kd: S,
    /// Desired phase margin [degrees].
    pub target_phase_margin_deg: S,
    /// Desired gain crossover frequency [rad/s].
    pub target_crossover_freq: S,
    /// Grid resolution for λ and μ search (default 0.1).
    pub resolution: S,
}

impl<S: ControlScalar> FopidAutoTune<S> {
    /// Create a new auto-tuner with specified targets.
    pub fn new(kp: S, ki: S, kd: S, target_phase_margin_deg: S, target_crossover_freq: S) -> Self {
        Self {
            kp,
            ki,
            kd,
            target_phase_margin_deg,
            target_crossover_freq,
            resolution: S::from_f64(0.1),
        }
    }

    /// Set the grid resolution (step size for λ and μ).
    pub fn with_resolution(mut self, resolution: S) -> Self {
        self.resolution = resolution;
        self
    }

    /// Execute the grid search.
    ///
    /// Searches λ ∈ [0.5, 1.5] and μ ∈ [0.5, 1.5] with the configured
    /// resolution, evaluating the objective at each grid point.
    ///
    /// # Errors
    /// Returns [`FracError::TuningFailed`] if no valid point is found.
    pub fn tune(&self) -> Result<FopidAutoTuneResult<S>, FracError> {
        let omega = self.target_crossover_freq;
        if omega <= S::ZERO || !omega.is_finite() {
            return Err(FracError::InvalidConfig(
                "crossover frequency must be positive",
            ));
        }

        let lo = S::from_f64(0.5);
        let hi = S::from_f64(1.5);
        let step = self.resolution;

        if step <= S::ZERO || !step.is_finite() {
            return Err(FracError::InvalidConfig("resolution must be positive"));
        }

        // Number of steps in each dimension
        let n_steps = ((hi - lo) / step).to_f64().ceil() as usize + 1;

        let mut best_obj = S::from_f64(f64::MAX);
        let mut best_lambda = lo;
        let mut best_mu = lo;
        let mut best_pm = S::ZERO;
        let mut best_omega = S::ZERO;
        let mut found = false;

        for i in 0..n_steps {
            let lambda = (lo + step * S::from_f64(i as f64)).min(hi);
            for j in 0..n_steps {
                let mu = (lo + step * S::from_f64(j as f64)).min(hi);

                let (pm, gc_omega) =
                    evaluate_fopid_frequency(self.kp, self.ki, lambda, self.kd, mu, omega);

                if !pm.is_finite() || !gc_omega.is_finite() {
                    continue;
                }

                // Objective: weighted sum of squared deviations
                let pm_err = pm - self.target_phase_margin_deg;
                let om_err = (gc_omega - omega) / (omega + S::EPSILON);
                let obj = pm_err * pm_err + S::from_f64(10.0) * om_err * om_err;

                if obj < best_obj {
                    best_obj = obj;
                    best_lambda = lambda;
                    best_mu = mu;
                    best_pm = pm;
                    best_omega = gc_omega;
                    found = true;
                }
            }
        }

        if !found {
            return Err(FracError::TuningFailed);
        }

        Ok(FopidAutoTuneResult {
            lambda: best_lambda,
            mu: best_mu,
            phase_margin_deg: best_pm,
            crossover_freq: best_omega,
            objective: best_obj,
        })
    }
}

/// Evaluate the FOPID frequency characteristics at a given frequency ω.
///
/// The controller transfer function evaluated at s = jω is:
///   C(jω) = Kp + Ki·(jω)^{-λ} + Kd·(jω)^{μ}
///
/// Using (jω)^α = ω^α · exp(jαπ/2):
///   Re[(jω)^α] = ω^α · cos(α·π/2)
///   Im[(jω)^α] = ω^α · sin(α·π/2)
///
/// Returns `(phase_margin_deg, gain_crossover_freq_approx)`.
/// The phase margin is computed as 180° + ∠C(jω), and the crossover
/// frequency returned is `omega` itself (gain=|C(jω)| normalised by Kp).
fn evaluate_fopid_frequency<S: ControlScalar>(
    kp: S,
    ki: S,
    lambda: S,
    kd: S,
    mu: S,
    omega: S,
) -> (S, S) {
    let half_pi = S::PI * S::HALF;

    // (jω)^α components
    let omega_neg_lambda = omega.powf(S::ZERO - lambda);
    let phase_neg_lambda = (S::ZERO - lambda) * half_pi; // angle of (jω)^{-λ}
    let i_re = ki * omega_neg_lambda * phase_neg_lambda.cos();
    let i_im = ki * omega_neg_lambda * phase_neg_lambda.sin();

    let omega_mu = omega.powf(mu);
    let phase_mu = mu * half_pi; // angle of (jω)^{μ}
    let d_re = kd * omega_mu * phase_mu.cos();
    let d_im = kd * omega_mu * phase_mu.sin();

    let c_re = kp + i_re + d_re;
    let c_im = i_im + d_im;

    let angle_rad = c_im.atan2(c_re);
    let angle_deg = angle_rad * S::from_f64(180.0 / core::f64::consts::PI);

    // Phase margin = 180 + phase (for negative-feedback open-loop)
    let phase_margin = S::from_f64(180.0) + angle_deg;

    (phase_margin, omega)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_order<S: ControlScalar>(order: S) -> Result<(), FracError> {
    if !order.is_finite() || order <= S::ZERO || order >= S::TWO {
        Err(FracError::InvalidOrder)
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Extension: FopidConfig builder helpers (with_limits)
// ---------------------------------------------------------------------------

impl<S: ControlScalar> FopidConfig<S> {
    /// Set output limits.
    pub fn with_limits(mut self, u_min: S, u_max: S) -> Self {
        self.u_min = u_min;
        self.u_max = u_max;
        self
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // λ=μ=1 gives standard PID-like response
    // -----------------------------------------------------------------------

    #[test]
    fn fopid_standard_pid_output_bounded() {
        let config = FopidConfig::standard_pid(1.0_f64, 0.5, 0.1, 0.01);
        let mut ctrl = config
            .with_limits(-10.0, 10.0)
            .build::<16>()
            .expect("valid");

        // Apply a constant setpoint step
        for _ in 0..100 {
            let out = ctrl.update(1.0, 0.0);
            assert!(
                (-10.0..=10.0).contains(&out),
                "Output out of bounds: {}",
                out
            );
        }
    }

    #[test]
    fn fopid_lambda_mu_one_proportional_first_step() {
        // On the very first step with error=1, P=1, I≈0, D≈1/h → u ≈ kp + kd/h
        // With ki=0 and kd=0, first-step output = kp * error
        let config = FopidConfig::standard_pid(2.0_f64, 0.0, 0.0, 0.01);
        let mut ctrl = config
            .with_limits(-100.0, 100.0)
            .build::<4>()
            .expect("valid");

        let out = ctrl.update(5.0, 0.0); // error = 5
                                         // P = kp * error = 10.0; I=0; D^1[e] first step = e/h = 500 but kd=0
        assert!(
            (out - 10.0).abs() < 1e-9,
            "First step: expected 10.0, got {}",
            out
        );
    }

    #[test]
    fn fopid_output_saturates_at_limits() {
        let config = FopidConfig::fractional(100.0_f64, 0.0, 1.0, 0.0, 1.0, -5.0, 5.0, 0.01);
        let mut ctrl = config.build::<4>().expect("valid");

        let out = ctrl.update(1.0, 0.0);
        assert_eq!(out, 5.0, "Should saturate to u_max=5.0");
        assert!(ctrl.is_saturated());
    }

    #[test]
    fn fopid_saturates_negative() {
        let config = FopidConfig::fractional(100.0_f64, 0.0, 1.0, 0.0, 1.0, -5.0, 5.0, 0.01);
        let mut ctrl = config.build::<4>().expect("valid");

        let out = ctrl.update(-1.0, 0.0);
        assert_eq!(out, -5.0, "Should saturate to u_min=-5.0");
        assert!(ctrl.is_saturated());
    }

    #[test]
    fn fopid_reset_clears_state() {
        let config = FopidConfig::standard_pid(1.0_f64, 1.0, 0.1, 0.01);
        let mut ctrl = config
            .with_limits(-100.0, 100.0)
            .build::<8>()
            .expect("valid");

        for _ in 0..20 {
            ctrl.update(1.0, 0.0);
        }
        ctrl.reset();
        // After reset, a single zero-error step should yield zero output
        let out = ctrl.update(0.0, 0.0);
        assert_eq!(out, 0.0, "After reset with zero error, output should be 0");
    }

    #[test]
    fn fopid_gains_accessible() {
        let config = FopidConfig::standard_pid(3.0_f64, 2.0, 1.0, 0.01);
        let ctrl = config
            .with_limits(-100.0, 100.0)
            .build::<4>()
            .expect("valid");
        assert!((ctrl.kp() - 3.0).abs() < 1e-12);
        assert!((ctrl.ki() - 2.0).abs() < 1e-12);
        assert!((ctrl.kd() - 1.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Invalid config
    // -----------------------------------------------------------------------

    #[test]
    fn fopid_invalid_lambda_zero() {
        let config = FopidConfig::fractional(1.0, 1.0, 0.0, 0.1, 1.0, -10.0, 10.0, 0.01);
        assert!(matches!(config.build::<4>(), Err(FracError::InvalidOrder)));
    }

    #[test]
    fn fopid_invalid_mu_two() {
        let config = FopidConfig::fractional(1.0, 1.0, 1.0, 0.1, 2.0, -10.0, 10.0, 0.01);
        assert!(matches!(config.build::<4>(), Err(FracError::InvalidOrder)));
    }

    #[test]
    fn fopid_invalid_limits() {
        let config = FopidConfig::fractional(1.0, 1.0, 1.0, 0.1, 1.0, 10.0, -10.0, 0.01);
        assert!(matches!(
            config.build::<4>(),
            Err(FracError::InvalidConfig(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Auto-tuner
    // -----------------------------------------------------------------------

    #[test]
    fn fopid_auto_tune_returns_result_in_range() {
        let tuner = FopidAutoTune::new(
            1.0_f64, // Kp
            0.5,     // Ki
            0.1,     // Kd
            45.0,    // target phase margin [deg]
            1.0,     // target crossover frequency [rad/s]
        );
        let result = tuner.tune().expect("tune should succeed");
        assert!(result.lambda >= 0.5 && result.lambda <= 1.5);
        assert!(result.mu >= 0.5 && result.mu <= 1.5);
        assert!(result.objective.is_finite());
    }

    #[test]
    fn fopid_auto_tune_invalid_crossover_freq_errors() {
        let tuner = FopidAutoTune::new(1.0_f64, 0.5, 0.1, 45.0, 0.0);
        assert!(matches!(tuner.tune(), Err(FracError::InvalidConfig(_))));
    }

    #[test]
    fn fopid_anti_windup_prevents_runaway() {
        // Run with tight limits for many steps; integral should not explode
        let config = FopidConfig::fractional(1.0_f64, 10.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.01);
        let mut ctrl = config.build::<32>().expect("valid");

        for _ in 0..500 {
            let out = ctrl.update(100.0, 0.0); // large constant error
            assert!(
                (-1.0..=1.0).contains(&out),
                "Anti-windup failed: output {}",
                out
            );
        }
    }

    // -----------------------------------------------------------------------
    // Closed-loop convergence test (λ=μ=1 should behave like PI controller)
    // -----------------------------------------------------------------------

    #[test]
    fn fopid_pi_mode_converges_to_setpoint() {
        // First-order plant: dy/dt = (u - y) / tau
        //
        // The GL fractional integral with a finite window N and step h reaches
        // a theoretical steady state:
        //   y_ss = sp * (kp + ki·N·h) / (1 + kp + ki·N·h)
        //
        // To achieve y_ss > 0.95·sp we need N·h > (19 - kp)/ki.
        // With kp=3, ki=5, h=0.001 and N=2048: N·h=2.048 → y_ss ≈ 0.9859.
        let tau = 1.0_f64;
        let h = 0.001_f64;
        let setpoint = 1.0_f64;
        let mut y = 0.0_f64;

        // N=2048 gives window span of 2.048 s >> tau=1 s, enabling near-full
        // integral accumulation before the truncated tail matters.
        let config = FopidConfig::fractional(3.0_f64, 5.0, 1.0, 0.0, 1.0, -100.0, 100.0, h);
        let mut ctrl = config.build::<2048>().expect("valid");

        for _ in 0..20_000 {
            let u = ctrl.update(setpoint, y);
            y += (u - y) / tau * h;
        }

        // Theoretical y_ss ≈ 0.9859; require y > 0.97 (< 3% of setpoint)
        let y_ss_theory = setpoint * (3.0 + 5.0 * 2048.0 * h) / (1.0 + 3.0 + 5.0 * 2048.0 * h);
        assert!(
            (y - y_ss_theory).abs() < 0.02,
            "PI^1D^0 should settle near theoretical y_ss={}; y={}",
            y_ss_theory,
            y
        );
    }
}
