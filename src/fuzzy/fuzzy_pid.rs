//! Fuzzy-PID hybrid controller.
//!
//! A `SugenoEngine` uses error `e` and error-rate `ė` as inputs to
//! compute online adjustments to the base PID gains (`Kp`, `Ki`, `Kd`).
//! The adjusted gains are then used by an embedded PID update law.

use crate::core::scalar::ControlScalar;
use crate::fuzzy::inference::SugenoEngine;
use crate::fuzzy::membership::MembershipFn;
use crate::fuzzy::FuzzyError;

// ────────────────────────────────────────────────────────────────────────────
// FuzzyPidConfig
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the fuzzy-PID hybrid controller.
#[derive(Debug, Clone, Copy)]
pub struct FuzzyPidConfig<S: ControlScalar> {
    /// Nominal proportional gain.
    pub kp_base: S,
    /// Nominal integral gain.
    pub ki_base: S,
    /// Nominal derivative gain.
    pub kd_base: S,
    /// Maximum fuzzy adjustment for Kp (symmetric, ±).
    pub kp_delta_max: S,
    /// Maximum fuzzy adjustment for Ki (symmetric, ±).
    pub ki_delta_max: S,
    /// Maximum fuzzy adjustment for Kd (symmetric, ±).
    pub kd_delta_max: S,
    /// Output clamp: `[-out_max, out_max]`. None = no clamp.
    pub out_max: Option<S>,
}

impl<S: ControlScalar> FuzzyPidConfig<S> {
    /// Create a basic configuration with no output clamping.
    pub fn new(kp_base: S, ki_base: S, kd_base: S) -> Self {
        Self {
            kp_base,
            ki_base,
            kd_base,
            kp_delta_max: S::ZERO,
            ki_delta_max: S::ZERO,
            kd_delta_max: S::ZERO,
            out_max: None,
        }
    }

    /// Set fuzzy adjustment ranges.
    pub fn with_deltas(mut self, kp_delta_max: S, ki_delta_max: S, kd_delta_max: S) -> Self {
        self.kp_delta_max = kp_delta_max;
        self.ki_delta_max = ki_delta_max;
        self.kd_delta_max = kd_delta_max;
        self
    }

    /// Set symmetric output clamp.
    pub fn with_out_max(mut self, out_max: S) -> Self {
        self.out_max = Some(out_max);
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FuzzyPid
// ────────────────────────────────────────────────────────────────────────────

/// Fuzzy-PID hybrid controller.
///
/// Internally maintains integral accumulator and previous error for the
/// underlying PID law. The `SugenoEngine` with `R` rules adjusts the PID
/// gains based on current error and error-rate.
///
/// Type parameters:
/// - `S`: floating-point scalar.
/// - `R`: maximum number of Sugeno rules.
pub struct FuzzyPid<S: ControlScalar, const R: usize> {
    config: FuzzyPidConfig<S>,
    /// Sugeno engine for gain scheduling (inputs: [error, error_rate]).
    /// The engine outputs a single value in `[-1, 1]` which is scaled by
    /// `*_delta_max` to give the gain adjustments.
    kp_engine: SugenoEngine<S, R>,
    ki_engine: SugenoEngine<S, R>,
    kd_engine: SugenoEngine<S, R>,
    /// Current PID integral term.
    integral: S,
    /// Error from the previous time step, used to compute error-rate.
    prev_error: Option<S>,
}

impl<S: ControlScalar, const R: usize> FuzzyPid<S, R> {
    /// Construct a new fuzzy-PID controller.
    pub fn new(
        config: FuzzyPidConfig<S>,
        kp_engine: SugenoEngine<S, R>,
        ki_engine: SugenoEngine<S, R>,
        kd_engine: SugenoEngine<S, R>,
    ) -> Self {
        Self {
            config,
            kp_engine,
            ki_engine,
            kd_engine,
            integral: S::ZERO,
            prev_error: None,
        }
    }

    /// Reset integrator and error history.
    pub fn reset(&mut self) {
        self.integral = S::ZERO;
        self.prev_error = None;
    }

    /// Compute the control output for the current sample.
    ///
    /// # Arguments
    /// - `setpoint`: desired value.
    /// - `measurement`: current measured value.
    /// - `dt`: time step (must be `> 0`).
    /// - `input_mfs`: membership functions shared by all three Sugeno engines.
    ///   `input_mfs[0]` = error terms, `input_mfs[1]` = error-rate terms.
    ///
    /// Returns `FuzzyError::InvalidParameter` for non-positive `dt`.
    pub fn update(
        &mut self,
        setpoint: S,
        measurement: S,
        dt: S,
        input_mfs: &[&[&dyn MembershipFn<S>]],
    ) -> Result<S, FuzzyError> {
        if dt <= S::ZERO {
            return Err(FuzzyError::InvalidParameter(
                "FuzzyPid: dt must be positive",
            ));
        }

        let error = setpoint - measurement;
        let error_rate = match self.prev_error {
            Some(pe) => (error - pe) / dt,
            None => S::ZERO,
        };

        let crisp_inputs = [error, error_rate];

        // Query each engine; if all rules have zero firing (unlikely after
        // setup but possible with edge inputs) fall back to zero adjustment.
        let delta_kp = self
            .kp_engine
            .infer(&crisp_inputs, input_mfs)
            .unwrap_or(S::ZERO);
        let delta_ki = self
            .ki_engine
            .infer(&crisp_inputs, input_mfs)
            .unwrap_or(S::ZERO);
        let delta_kd = self
            .kd_engine
            .infer(&crisp_inputs, input_mfs)
            .unwrap_or(S::ZERO);

        // Adjusted gains (clamp to non-negative to keep stability)
        let kp = (self.config.kp_base + delta_kp * self.config.kp_delta_max)
            .clamp_val(S::ZERO, S::from_f64(f64::MAX));
        let ki = (self.config.ki_base + delta_ki * self.config.ki_delta_max)
            .clamp_val(S::ZERO, S::from_f64(f64::MAX));
        let kd = (self.config.kd_base + delta_kd * self.config.kd_delta_max)
            .clamp_val(S::ZERO, S::from_f64(f64::MAX));

        // PID law
        self.integral += ki * error * dt;
        let derivative = kd * error_rate;
        let output_raw = kp * error + self.integral + derivative;

        let output = match self.config.out_max {
            Some(max_val) => output_raw.clamp_val(-max_val, max_val),
            None => output_raw,
        };

        self.prev_error = Some(error);
        Ok(output)
    }

    /// Read current integral accumulator (useful for diagnostics).
    pub fn integral(&self) -> S {
        self.integral
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzy::inference::SugenoRule;
    use crate::fuzzy::membership::Trapezoidal;
    use crate::fuzzy::rule_base::{Antecedent, TNorm};

    fn build_constant_sugeno_engine(output: f64) -> SugenoEngine<f64, 4> {
        // Single rule that always fires (full-universe antecedent via trapezoid)
        let universal_mf = Trapezoidal::new(-1e6_f64, -1e5, 1e5, 1e6).unwrap();
        // We'll store the MF externally and pass it in at infer-time.
        // The engine itself just holds the rule coefficients.
        let _ = universal_mf;
        let mut engine: SugenoEngine<f64, 4> = SugenoEngine::new(TNorm::Min);
        let mut ant = Antecedent::new();
        ant.add(0, 0).unwrap(); // references var=0, set=0
        engine
            .add_rule(SugenoRule::new(ant, 0.0, 0.0, output))
            .unwrap();
        engine
    }

    #[test]
    fn fuzzy_pid_proportional_only_no_adjustment() {
        // With kp_delta_max = 0 and no ki/kd, acts as pure P controller
        let config = FuzzyPidConfig::new(2.0_f64, 0.0, 0.0);
        let kp_eng = build_constant_sugeno_engine(0.0);
        let ki_eng = build_constant_sugeno_engine(0.0);
        let kd_eng = build_constant_sugeno_engine(0.0);

        let mut controller = FuzzyPid::new(config, kp_eng, ki_eng, kd_eng);

        // Universal membership function for the fuzzy engine inputs
        let universal_mf = Trapezoidal::new(-1e6_f64, -1e5, 1e5, 1e6).unwrap();
        let err_mfs: &[&dyn MembershipFn<f64>] = &[&universal_mf];
        let rate_mfs: &[&dyn MembershipFn<f64>] = &[&universal_mf];
        let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[err_mfs, rate_mfs];

        let output = controller.update(10.0, 4.0, 0.01, input_mfs).unwrap();
        // error = 6.0, kp = 2.0, ki=0, kd=0 → output = 12.0
        assert!((output - 12.0).abs() < 1e-9, "Expected 12.0, got {output}");
    }

    #[test]
    fn fuzzy_pid_integrates_over_time() {
        let config = FuzzyPidConfig::new(0.0_f64, 1.0, 0.0);
        let kp_eng = build_constant_sugeno_engine(0.0);
        let ki_eng = build_constant_sugeno_engine(0.0);
        let kd_eng = build_constant_sugeno_engine(0.0);

        let mut controller = FuzzyPid::new(config, kp_eng, ki_eng, kd_eng);

        let universal_mf = Trapezoidal::new(-1e6_f64, -1e5, 1e5, 1e6).unwrap();
        let err_mfs: &[&dyn MembershipFn<f64>] = &[&universal_mf];
        let rate_mfs: &[&dyn MembershipFn<f64>] = &[&universal_mf];
        let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[err_mfs, rate_mfs];

        // Constant error = 5.0 for 10 steps at dt=0.1 → integral = 5.0
        for _ in 0..10 {
            controller.update(5.0, 0.0, 0.1, input_mfs).unwrap();
        }
        assert!(
            (controller.integral() - 5.0).abs() < 1e-9,
            "Integral should be 5.0, got {}",
            controller.integral()
        );
    }

    #[test]
    fn fuzzy_pid_reset_clears_state() {
        let config = FuzzyPidConfig::new(1.0_f64, 1.0, 0.0);
        let kp_eng = build_constant_sugeno_engine(0.0);
        let ki_eng = build_constant_sugeno_engine(0.0);
        let kd_eng = build_constant_sugeno_engine(0.0);

        let mut controller = FuzzyPid::new(config, kp_eng, ki_eng, kd_eng);

        let universal_mf = Trapezoidal::new(-1e6_f64, -1e5, 1e5, 1e6).unwrap();
        let err_mfs: &[&dyn MembershipFn<f64>] = &[&universal_mf];
        let rate_mfs: &[&dyn MembershipFn<f64>] = &[&universal_mf];
        let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[err_mfs, rate_mfs];

        controller.update(10.0, 0.0, 0.1, input_mfs).unwrap();
        assert!(controller.integral() != 0.0);

        controller.reset();
        assert_eq!(controller.integral(), 0.0);
    }

    #[test]
    fn fuzzy_pid_invalid_dt_returns_error() {
        let config = FuzzyPidConfig::new(1.0_f64, 0.0, 0.0);
        let kp_eng = build_constant_sugeno_engine(0.0);
        let ki_eng = build_constant_sugeno_engine(0.0);
        let kd_eng = build_constant_sugeno_engine(0.0);

        let mut controller = FuzzyPid::new(config, kp_eng, ki_eng, kd_eng);

        let universal_mf = Trapezoidal::new(-1e6_f64, -1e5, 1e5, 1e6).unwrap();
        let err_mfs: &[&dyn MembershipFn<f64>] = &[&universal_mf];
        let rate_mfs: &[&dyn MembershipFn<f64>] = &[&universal_mf];
        let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[err_mfs, rate_mfs];

        let result = controller.update(10.0, 0.0, 0.0, input_mfs);
        assert!(
            matches!(result, Err(FuzzyError::InvalidParameter(_))),
            "Expected InvalidParameter for dt=0"
        );
    }

    #[test]
    fn fuzzy_pid_output_clamped() {
        let config = FuzzyPidConfig::new(100.0_f64, 0.0, 0.0).with_out_max(5.0);
        let kp_eng = build_constant_sugeno_engine(0.0);
        let ki_eng = build_constant_sugeno_engine(0.0);
        let kd_eng = build_constant_sugeno_engine(0.0);

        let mut controller = FuzzyPid::new(config, kp_eng, ki_eng, kd_eng);

        let universal_mf = Trapezoidal::new(-1e6_f64, -1e5, 1e5, 1e6).unwrap();
        let err_mfs: &[&dyn MembershipFn<f64>] = &[&universal_mf];
        let rate_mfs: &[&dyn MembershipFn<f64>] = &[&universal_mf];
        let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[err_mfs, rate_mfs];

        // error = 10, kp = 100 → raw output = 1000, clamped to 5
        let output = controller.update(10.0, 0.0, 0.01, input_mfs).unwrap();
        assert_eq!(output, 5.0, "Expected clamped output 5.0, got {output}");
    }
}
