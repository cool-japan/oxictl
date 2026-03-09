// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright COOLJAPAN OU (Team Kitasan)
//
// Predictive Functional Control (PFC).
//
// Theory
// ------
// PFC selects the control action u_k so that the model prediction at the
// coincidence horizon H matches a first-order reference trajectory:
//
//   y_ref_H = y_sp + (y0 − y_sp)·exp(−H / τ_ref)
//
// where y_sp is the long-term set-point and y0 is the current output.
//
// Prediction at horizon H (SISO, linear plant):
//   y(t+H) = G·u_k + F(past inputs)
//
// G  = step-response coefficient at H  (impulse-response integral from 0..H)
// F  = free response due to past inputs (with u_k = 0)
//
// Analytical solution (unconstrained):
//   u_k = (y_ref_H − F) / G
//
// Constraints are applied by saturating u_k after the analytical solve.
//
// Implementation details
// ----------------------
// * Plant model: a discrete TransferFn<S, NP> driven by u_k.
// * The step-response coefficient G is computed once in the constructor by
//   driving a clone of the model with a unit step for H samples and recording
//   the output at H (this avoids storing a full impulse-response table).
// * The free response F(k) is computed each call by driving a *separate*
//   state clone of the model forward H steps with u=0, using the current
//   state of the internal model.
// * Past inputs are stored in a `heapless::Deque<S, 32>`.
//
// Limitations (inherent to no-alloc, const-generic design)
// ---------------------------------------------------------
// * Horizon H ≤ 32 is enforced at the type level via heapless capacity.
// * The free-response clone is computed by iterating H times, so call
//   overhead scales as O(H · NP).

use heapless::Deque;

use crate::core::scalar::ControlScalar;
use crate::core::transfer_fn::TransferFn;

use super::ImcError;

// ──────────────────────────────────────────────────────────────
// Config
// ──────────────────────────────────────────────────────────────

/// Maximum supported coincidence horizon (matches `heapless::Deque` capacity).
pub const MAX_HORIZON: usize = 32;

/// Configuration for [`PfcController`].
#[derive(Debug, Clone, Copy)]
pub struct PfcConfig<S: ControlScalar, const NP: usize> {
    /// Numerator of the discrete plant model (length NP).
    pub model_b: [S; NP],
    /// Denominator of the discrete plant model (length NP, implicit leading 1).
    pub model_a: [S; NP],
    /// Coincidence horizon H (number of samples).  Must satisfy 1 ≤ H ≤ MAX_HORIZON.
    pub horizon: usize,
    /// Reference trajectory time constant τ_ref (in samples).
    /// The reference trajectory decays from y0 towards y_sp with this time constant.
    pub reference_time_constant: S,
    /// Control output lower saturation limit.
    pub u_min: S,
    /// Control output upper saturation limit.
    pub u_max: S,
}

impl<S: ControlScalar, const NP: usize> PfcConfig<S, NP> {
    /// Construct configuration without saturation.
    pub fn new(
        model_b: [S; NP],
        model_a: [S; NP],
        horizon: usize,
        reference_time_constant: S,
    ) -> Self {
        let big = S::from_f64(1e9);
        Self {
            model_b,
            model_a,
            horizon,
            reference_time_constant,
            u_min: -big,
            u_max: big,
        }
    }

    /// Attach saturation limits on the control output.
    pub fn with_limits(mut self, u_min: S, u_max: S) -> Self {
        self.u_min = u_min;
        self.u_max = u_max;
        self
    }
}

// ──────────────────────────────────────────────────────────────
// Controller
// ──────────────────────────────────────────────────────────────

/// Predictive Functional Control (PFC) controller.
///
/// Generic parameters
/// ------------------
/// * `S`  – scalar type (`f32` or `f64`)
/// * `NP` – order of the plant model transfer function
#[derive(Debug, Clone)]
pub struct PfcController<S: ControlScalar, const NP: usize> {
    /// Internal plant model (tracks the actual plant output trajectory).
    plant_model: TransferFn<S, NP>,
    /// Step-response coefficient at horizon H (pre-computed in constructor).
    g_coefficient: S,
    /// Coincidence horizon H.
    coincidence_horizon: usize,
    /// Reference trajectory time constant τ_ref (samples).
    reference_time_constant: S,
    /// History of past control inputs (newest at back).
    u_history: Deque<S, MAX_HORIZON>,
    /// Saturation limits.
    u_min: S,
    u_max: S,
    /// Coefficients for free-response computation (cached from config).
    model_b: [S; NP],
    model_a: [S; NP],
}

impl<S: ControlScalar, const NP: usize> PfcController<S, NP> {
    /// Construct from a [`PfcConfig`].
    ///
    /// Returns `Err` if:
    /// * `horizon == 0` or `horizon > MAX_HORIZON`
    /// * `reference_time_constant ≤ 0`
    /// * `u_min ≥ u_max`
    pub fn new(cfg: &PfcConfig<S, NP>) -> Result<Self, ImcError> {
        if cfg.horizon == 0 || cfg.horizon > MAX_HORIZON {
            return Err(ImcError::InvalidParameter(
                "horizon must be in [1, MAX_HORIZON]",
            ));
        }
        if cfg.reference_time_constant <= S::ZERO {
            return Err(ImcError::InvalidParameter(
                "reference_time_constant must be > 0",
            ));
        }
        if cfg.u_min >= cfg.u_max {
            return Err(ImcError::InvalidParameter("u_min must be < u_max"));
        }

        // Pre-compute G: step-response value at horizon H.
        // G = Σ_{i=1}^{H} g(i)  where g is the impulse response of P_m.
        // Equivalently: drive P_m with a unit step for H samples and read
        // the *cumulative* output (i.e., the step response at step H).
        // Since TransferFn already integrates the impulse, we drive with 1.0
        // and the output at step H is the step response at H.
        let g_coefficient = Self::compute_g(cfg.model_b, cfg.model_a, cfg.horizon);

        if g_coefficient.abs() < S::from_f64(1e-12) {
            return Err(ImcError::NumericalError(
                "step-response coefficient G is near zero; horizon may be too short",
            ));
        }

        let plant_model = TransferFn::<S, NP>::new(cfg.model_b, cfg.model_a);

        Ok(Self {
            plant_model,
            g_coefficient,
            coincidence_horizon: cfg.horizon,
            reference_time_constant: cfg.reference_time_constant,
            u_history: Deque::new(),
            u_min: cfg.u_min,
            u_max: cfg.u_max,
            model_b: cfg.model_b,
            model_a: cfg.model_a,
        })
    }

    /// Compute step-response coefficient G at horizon H.
    ///
    /// This drives a fresh copy of the TF with a unit step for H samples and
    /// returns the output at step H.  That is the step response h(H).
    fn compute_g(b: [S; NP], a: [S; NP], horizon: usize) -> S {
        let mut tf = TransferFn::<S, NP>::new(b, a);
        let mut y = S::ZERO;
        for _ in 0..horizon {
            y = tf.process(S::ONE);
        }
        y
    }

    /// Compute the free response F: what P_m predicts H steps ahead if u=0
    /// from *now* (i.e., driven by zero future input).
    ///
    /// This clones the current model state conceptually by running H iterations
    /// of a fresh TF initialised with the *current internal model state*.
    ///
    /// Because `TransferFn` does not expose its internal state directly (only
    /// via `Copy`), we can clone the entire struct and iterate zero-input.
    fn compute_free_response(&self) -> S {
        let mut model_clone = self.plant_model;
        let mut y = S::ZERO;
        for _ in 0..self.coincidence_horizon {
            y = model_clone.process(S::ZERO);
        }
        y
    }

    /// Compute the reference trajectory target at horizon H.
    ///
    ///   y_ref_H = y_sp + (y0 − y_sp)·exp(−H / τ_ref)
    fn reference_at_horizon(&self, y_sp: S, y0: S) -> S {
        let h = S::from_f64(self.coincidence_horizon as f64);
        let decay = (-h / self.reference_time_constant).exp();
        y_sp + (y0 - y_sp) * decay
    }

    /// Compute next control output.
    ///
    /// Arguments
    /// ---------
    /// * `setpoint`    – long-term desired output y_sp
    /// * `measurement` – current plant output y(k)
    ///
    /// Returns the (saturated) control signal u(k).
    pub fn update(&mut self, setpoint: S, measurement: S) -> Result<S, ImcError> {
        // Step 1: compute reference trajectory target at horizon H.
        let y_ref_h = self.reference_at_horizon(setpoint, measurement);

        // Step 2: compute free response (what model predicts if u=0 from now).
        let f = self.compute_free_response();

        // Step 3: analytical PFC law  u_k = (y_ref_H − F) / G
        let u_raw = (y_ref_h - f) / self.g_coefficient;

        // Step 4: saturate.
        let u_sat = u_raw.clamp_val(self.u_min, self.u_max);

        // Step 5: advance internal model with the actually applied u.
        let _ = self.plant_model.process(u_sat);

        // Step 6: record in history (cap at MAX_HORIZON, drop oldest if full).
        if self.u_history.is_full() {
            let _ = self.u_history.pop_front();
        }
        let push_result = self.u_history.push_back(u_sat);
        // push_back can only fail if full, which we handle above.
        if push_result.is_err() {
            return Err(ImcError::NumericalError("u_history buffer overflow"));
        }

        Ok(u_sat)
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.plant_model.reset();
        self.u_history.clear();
    }

    /// Return the pre-computed step-response coefficient G (diagnostic).
    #[inline]
    pub fn g_coefficient(&self) -> S {
        self.g_coefficient
    }

    /// Return the current coincidence horizon H.
    #[inline]
    pub fn coincidence_horizon(&self) -> usize {
        self.coincidence_horizon
    }

    /// Recompute G for a different horizon (e.g., for gain-scheduling).
    ///
    /// Returns `Err` if the new horizon is out of range or G is degenerate.
    pub fn set_horizon(&mut self, new_horizon: usize) -> Result<(), ImcError> {
        if new_horizon == 0 || new_horizon > MAX_HORIZON {
            return Err(ImcError::InvalidParameter(
                "horizon must be in [1, MAX_HORIZON]",
            ));
        }
        let g = Self::compute_g(self.model_b, self.model_a, new_horizon);
        if g.abs() < S::from_f64(1e-12) {
            return Err(ImcError::NumericalError(
                "new G is near zero; choose a longer horizon",
            ));
        }
        self.coincidence_horizon = new_horizon;
        self.g_coefficient = g;
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// First-order plant with unity DC gain: P_m(z) = 0.2 / (1 − 0.8·z^{-1}).
    fn unity_dc_cfg(horizon: usize, tau_ref: f64) -> PfcConfig<f64, 1> {
        PfcConfig::new([0.2], [-0.8], horizon, tau_ref)
    }

    #[test]
    fn pfc_construction_ok() {
        let cfg = unity_dc_cfg(10, 5.0);
        let ctrl = PfcController::<f64, 1>::new(&cfg);
        assert!(ctrl.is_ok(), "Valid config should construct OK");
    }

    #[test]
    fn pfc_construction_invalid_horizon_zero() {
        let cfg = unity_dc_cfg(0, 5.0);
        let ctrl = PfcController::<f64, 1>::new(&cfg);
        assert!(
            matches!(ctrl, Err(ImcError::InvalidParameter(_))),
            "horizon=0 must be rejected"
        );
    }

    #[test]
    fn pfc_construction_invalid_horizon_too_large() {
        let cfg = unity_dc_cfg(MAX_HORIZON + 1, 5.0);
        let ctrl = PfcController::<f64, 1>::new(&cfg);
        assert!(
            matches!(ctrl, Err(ImcError::InvalidParameter(_))),
            "horizon>MAX_HORIZON must be rejected"
        );
    }

    #[test]
    fn pfc_construction_invalid_tau_ref_zero() {
        let cfg = unity_dc_cfg(10, 0.0);
        let ctrl = PfcController::<f64, 1>::new(&cfg);
        assert!(
            matches!(ctrl, Err(ImcError::InvalidParameter(_))),
            "tau_ref=0 must be rejected"
        );
    }

    #[test]
    fn pfc_construction_invalid_limits() {
        let cfg = unity_dc_cfg(10, 5.0).with_limits(1.0, -1.0);
        let ctrl = PfcController::<f64, 1>::new(&cfg);
        assert!(
            matches!(ctrl, Err(ImcError::InvalidParameter(_))),
            "Inverted limits must be rejected"
        );
    }

    /// Step reference tracking with perfect model.
    /// After sufficient steps the output should be near the setpoint.
    #[test]
    fn pfc_step_reference_tracking() {
        let cfg = unity_dc_cfg(8, 6.0);
        let mut ctrl = PfcController::<f64, 1>::new(&cfg).unwrap();

        // Simulate with plant identical to model.
        let mut plant = TransferFn::<f64, 1>::new([0.2], [-0.8]);
        let mut y = 0.0_f64;
        let setpoint = 1.0_f64;
        let mut last_u = 0.0_f64;

        for _ in 0..500 {
            let u = ctrl.update(setpoint, y).unwrap();
            y = plant.process(u);
            last_u = u;
        }

        let error = (y - setpoint).abs();
        assert!(
            error < 0.02,
            "PFC should track step reference: e={:.4}, y={:.4}, u={:.4}",
            error,
            y,
            last_u
        );
    }

    /// Horizon sensitivity: shorter horizon should give faster response
    /// (reaches 90 % of setpoint sooner).
    #[test]
    fn pfc_shorter_horizon_faster_response() {
        // Horizon H=5, τ_ref=3  (aggressive)
        let cfg_fast = PfcConfig::<f64, 1>::new([0.2], [-0.8], 5, 3.0);
        // Horizon H=20, τ_ref=15 (conservative)
        let cfg_slow = PfcConfig::<f64, 1>::new([0.2], [-0.8], 20, 15.0);

        let mut ctrl_fast = PfcController::<f64, 1>::new(&cfg_fast).unwrap();
        let mut ctrl_slow = PfcController::<f64, 1>::new(&cfg_slow).unwrap();

        let mut plant_fast = TransferFn::<f64, 1>::new([0.2], [-0.8]);
        let mut plant_slow = TransferFn::<f64, 1>::new([0.2], [-0.8]);

        let setpoint = 1.0_f64;
        let mut y_fast = 0.0_f64;
        let mut y_slow = 0.0_f64;
        let mut steps_fast_90 = usize::MAX;
        let mut steps_slow_90 = usize::MAX;

        for step in 0..300_usize {
            let u_f = ctrl_fast.update(setpoint, y_fast).unwrap();
            y_fast = plant_fast.process(u_f);
            if y_fast >= 0.9 * setpoint && steps_fast_90 == usize::MAX {
                steps_fast_90 = step;
            }

            let u_s = ctrl_slow.update(setpoint, y_slow).unwrap();
            y_slow = plant_slow.process(u_s);
            if y_slow >= 0.9 * setpoint && steps_slow_90 == usize::MAX {
                steps_slow_90 = step;
            }
        }

        assert!(
            steps_fast_90 <= steps_slow_90,
            "Shorter horizon/τ should reach 90% faster: fast={} slow={}",
            steps_fast_90,
            steps_slow_90
        );
    }

    /// Saturation clamps the control output.
    #[test]
    fn pfc_saturation_respected() {
        let cfg = unity_dc_cfg(10, 5.0).with_limits(-0.1, 0.1);
        let mut ctrl = PfcController::<f64, 1>::new(&cfg).unwrap();

        // Large step → wants large u, must be clamped.
        let u = ctrl.update(100.0, 0.0).unwrap();
        assert!(
            u <= 0.1 + 1e-12,
            "u should be saturated at u_max=0.1, got {:.6}",
            u
        );
    }

    /// After reset, controller returns zero output on zero reference.
    #[test]
    fn pfc_reset_clears_state() {
        let cfg = unity_dc_cfg(10, 5.0);
        let mut ctrl = PfcController::<f64, 1>::new(&cfg).unwrap();
        let mut plant = TransferFn::<f64, 1>::new([0.2], [-0.8]);

        let mut y = 0.0_f64;
        for _ in 0..100 {
            let u = ctrl.update(1.0, y).unwrap();
            y = plant.process(u);
        }

        ctrl.reset();
        let u_post = ctrl.update(0.0, 0.0).unwrap();
        assert!(
            u_post.abs() < 1e-10,
            "After reset on zero input, output should be zero: {:.4e}",
            u_post
        );
    }

    /// G coefficient should be positive for a minimum-phase plant.
    #[test]
    fn pfc_g_coefficient_positive() {
        let cfg = unity_dc_cfg(10, 5.0);
        let ctrl = PfcController::<f64, 1>::new(&cfg).unwrap();
        assert!(
            ctrl.g_coefficient() > 0.0,
            "G should be positive: {}",
            ctrl.g_coefficient()
        );
    }

    /// set_horizon dynamically changes the horizon.
    #[test]
    fn pfc_set_horizon_works() {
        let cfg = unity_dc_cfg(10, 5.0);
        let mut ctrl = PfcController::<f64, 1>::new(&cfg).unwrap();
        assert_eq!(ctrl.coincidence_horizon(), 10);

        ctrl.set_horizon(20).unwrap();
        assert_eq!(ctrl.coincidence_horizon(), 20);
    }

    /// set_horizon rejects out-of-range values.
    #[test]
    fn pfc_set_horizon_invalid() {
        let cfg = unity_dc_cfg(10, 5.0);
        let mut ctrl = PfcController::<f64, 1>::new(&cfg).unwrap();
        assert!(ctrl.set_horizon(0).is_err());
        assert!(ctrl.set_horizon(MAX_HORIZON + 1).is_err());
    }

    /// Second-order plant model compiles and runs without panicking.
    #[test]
    fn pfc_second_order_plant() {
        // Second-order discrete plant
        let cfg = PfcConfig::<f64, 2>::new([0.1, 0.05], [-1.5, 0.7], 12, 8.0);
        let mut ctrl = PfcController::<f64, 2>::new(&cfg).unwrap();
        let mut plant = TransferFn::<f64, 2>::new([0.1, 0.05], [-1.5, 0.7]);
        let mut y = 0.0_f64;

        for _ in 0..300 {
            let u = ctrl.update(1.0, y).unwrap();
            y = plant.process(u);
        }

        assert!(
            y.is_finite(),
            "Second-order PFC response should be finite: y={:.4}",
            y
        );
    }
}
