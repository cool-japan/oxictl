// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright COOLJAPAN OU (Team Kitasan)
//
// Internal Model Control (IMC) controller.
//
// Theory
// ------
// IMC parameterises the controller as:
//   C_imc(z) = Q(z) / (1 - Q(z)·P_m(z))
//
// where:
//   P_m(z) – discrete-time plant model (user-supplied TransferFn)
//   Q(z)   – stable IMC filter (typically a low-pass)
//
// First-order IMC filter (λ ∈ (0,1)):
//   Q(z) = (1 - λ) / (1 - λ·z^{-1})
//
// Implementation structure (equivalent two-degree-of-freedom feedback):
//   ε(k)   = y_plant(k) - y_model(k)          (model-mismatch signal)
//   u(k)   = Q[ r(k) - ε(k) ]                 (filter augmented error)
//   y_model(k+1) updated via P_m driven by u(k)
//
// This is equivalent to the conventional feedback controller
//   C = Q / (1 - Q·P_m)
// but avoids the algebraic combination at each step.

use crate::core::scalar::ControlScalar;
use crate::core::transfer_fn::TransferFn;

use super::ImcError;

// ──────────────────────────────────────────────────────────────
// Config
// ──────────────────────────────────────────────────────────────

/// Configuration for [`ImcController`].
///
/// `N` is the order of both the plant model and the IMC filter (they share the
/// same order for the Q-filter; the Q-filter itself is always first-order, so
/// set `N = 1` for a first-order Q-filter and `M` for an M-th-order plant).
///
/// In practice `N_PLANT` and `N_QFILT` are kept separate so the user can
/// supply a higher-order plant model with a first-order IMC filter.
///
/// For simplicity the struct is generic over a single `NP` (plant order) and
/// `NQ = 1` is specialised via the constructor.
#[derive(Debug, Clone, Copy)]
pub struct ImcConfig<S: ControlScalar, const NP: usize> {
    /// Numerator coefficients of the discrete plant model (length NP).
    pub model_b: [S; NP],
    /// Denominator coefficients of the discrete plant model (length NP).
    /// Implicit leading 1 not included (same convention as [`TransferFn`]).
    pub model_a: [S; NP],
    /// IMC filter pole λ  ∈ (0, 1).  Closer to 1 → slower / more robust.
    /// Closer to 0 → faster / less robust to model mismatch.
    pub filter_lambda: S,
    /// Output constraint – lower bound (applied to the computed control).
    pub u_min: S,
    /// Output constraint – upper bound.
    pub u_max: S,
}

impl<S: ControlScalar, const NP: usize> ImcConfig<S, NP> {
    /// Unconstrained configuration.
    pub fn new(model_b: [S; NP], model_a: [S; NP], filter_lambda: S) -> Self {
        let big = S::from_f64(1e9);
        Self {
            model_b,
            model_a,
            filter_lambda,
            u_min: -big,
            u_max: big,
        }
    }

    /// Add symmetric saturation limits on the control output.
    pub fn with_limits(mut self, u_min: S, u_max: S) -> Self {
        self.u_min = u_min;
        self.u_max = u_max;
        self
    }
}

// ──────────────────────────────────────────────────────────────
// Controller
// ──────────────────────────────────────────────────────────────

/// Internal Model Control (IMC) controller.
///
/// Generic parameters
/// ------------------
/// * `S`  – scalar type (`f32` or `f64`)
/// * `NP` – order of the plant model transfer function
///
/// The IMC Q-filter is always first-order (order = 1), parameterised by λ.
#[derive(Debug, Clone, Copy)]
pub struct ImcController<S: ControlScalar, const NP: usize> {
    /// Internal plant model P_m(z).
    plant_model: TransferFn<S, NP>,
    /// IMC Q-filter Q(z) – first-order low-pass.
    q_filter: TransferFn<S, 1>,
    /// Current model-mismatch signal  ε = y_plant − y_model.
    /// This is the "corrected" disturbance fed back into Q.
    mismatch: S,
    /// Saturation limits.
    u_min: S,
    u_max: S,
}

impl<S: ControlScalar, const NP: usize> ImcController<S, NP> {
    /// Construct from an [`ImcConfig`].
    ///
    /// Returns `Err(ImcError::InvalidParameter)` if `filter_lambda` is outside
    /// the open interval (0, 1).
    pub fn new(cfg: &ImcConfig<S, NP>) -> Result<Self, ImcError> {
        let lambda = cfg.filter_lambda;
        if lambda <= S::ZERO || lambda >= S::ONE {
            return Err(ImcError::InvalidParameter(
                "filter_lambda must be in (0, 1)",
            ));
        }
        if cfg.u_min >= cfg.u_max {
            return Err(ImcError::InvalidParameter("u_min must be < u_max"));
        }

        // Q(z) = (1-λ) / (1 - λ·z^{-1})
        // In TransferFn convention: b=[1-λ], a=[-λ]
        let q_filter = TransferFn::<S, 1>::new([S::ONE - lambda], [-lambda]);
        let plant_model = TransferFn::<S, NP>::new(cfg.model_b, cfg.model_a);

        Ok(Self {
            plant_model,
            q_filter,
            mismatch: S::ZERO,
            u_min: cfg.u_min,
            u_max: cfg.u_max,
        })
    }

    /// Compute the next control output.
    ///
    /// Arguments
    /// ---------
    /// * `reference`    – desired set-point r(k)
    /// * `plant_output` – measured plant output y(k)
    ///
    /// Returns the saturated control signal u(k).
    ///
    /// Internally:
    /// 1. ε(k)  = y_plant(k) − y_model(k)      (mismatch from last step)
    /// 2. u(k)  = Q[ r(k) − ε(k) ]
    /// 3. y_model(k) ← P_m[ u(k) ]             (advance internal model)
    /// 4. Store u(k) for next call.
    pub fn update(&mut self, reference: S, plant_output: S) -> Result<S, ImcError> {
        // Step 1: augmented error = r − ε
        let augmented_error = reference - self.mismatch;

        // Step 2: pass through IMC filter Q
        let u_raw = self.q_filter.process(augmented_error);

        // Step 3: advance internal model with raw (unsaturated) u to keep
        // the model state consistent with actual applied input after saturation.
        // We use the saturated signal so the model tracks what the plant sees.
        let u_saturated = u_raw.clamp_val(self.u_min, self.u_max);
        let y_model = self.plant_model.process(u_saturated);

        // Step 4: update mismatch for next iteration.
        // ε(k+1) = y_plant(k+1) − y_model(k+1)
        // We can only update this at the start of the *next* call, but we
        // pre-compute based on current plant output vs. current model output.
        self.mismatch = plant_output - y_model;

        Ok(u_saturated)
    }

    /// Reset all internal states (filter and model) to zero.
    pub fn reset(&mut self) {
        self.q_filter.reset();
        self.plant_model.reset();
        self.mismatch = S::ZERO;
    }

    /// Access the current model-mismatch signal ε (diagnostic).
    #[inline]
    pub fn mismatch(&self) -> S {
        self.mismatch
    }

    /// Equivalent conventional feedback controller dc-gain (for design checks).
    ///
    /// At DC (z=1): C_dc = Q_dc / (1 − Q_dc · P_m_dc)
    /// Q_dc   = (1-λ) / (1-λ) = 1
    /// P_m_dc = sum(b) / (1 + sum(a))
    pub fn equivalent_dc_gain(&self, cfg: &ImcConfig<S, NP>) -> Result<S, ImcError> {
        let sum_b: S = cfg.model_b.iter().copied().fold(S::ZERO, |acc, x| acc + x);
        let sum_a: S = cfg.model_a.iter().copied().fold(S::ZERO, |acc, x| acc + x);
        let plant_dc = sum_b / (S::ONE + sum_a);
        let q_dc = S::ONE; // DC gain of Q-filter is 1
        let denom = S::ONE - q_dc * plant_dc;
        if denom.abs() < S::EPSILON * S::from_f64(1e6) {
            return Err(ImcError::NumericalError(
                "equivalent_dc_gain: near-zero denominator",
            ));
        }
        Ok(q_dc / denom)
    }
}

// ──────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple first-order plant model: P_m(z) = 0.5 / (1 - 0.5·z^{-1})
    /// DC gain = 0.5 / (1 - 0.5) = 1.0  (unity DC gain → zero steady-state error).
    fn first_order_plant_cfg(lambda: f64) -> ImcConfig<f64, 1> {
        // b=[0.5], a=[-0.5]  → DC gain = 0.5/(1+(-0.5)) = 1.0
        ImcConfig::new([0.5], [-0.5], lambda)
    }

    #[test]
    fn imc_controller_construction_valid() {
        let cfg = first_order_plant_cfg(0.8);
        let ctrl = ImcController::<f64, 1>::new(&cfg);
        assert!(ctrl.is_ok(), "Valid config should construct OK");
    }

    #[test]
    fn imc_controller_construction_invalid_lambda_zero() {
        let cfg = ImcConfig::<f64, 1>::new([0.5], [-0.5], 0.0);
        let ctrl = ImcController::<f64, 1>::new(&cfg);
        assert!(
            matches!(ctrl, Err(ImcError::InvalidParameter(_))),
            "λ=0 must be rejected"
        );
    }

    #[test]
    fn imc_controller_construction_invalid_lambda_one() {
        let cfg = ImcConfig::<f64, 1>::new([0.5], [-0.5], 1.0);
        let ctrl = ImcController::<f64, 1>::new(&cfg);
        assert!(
            matches!(ctrl, Err(ImcError::InvalidParameter(_))),
            "λ=1 must be rejected"
        );
    }

    #[test]
    fn imc_controller_construction_invalid_limits() {
        let cfg = ImcConfig::<f64, 1>::new([0.5], [-0.5], 0.8).with_limits(1.0, -1.0); // inverted limits
        let ctrl = ImcController::<f64, 1>::new(&cfg);
        assert!(
            matches!(ctrl, Err(ImcError::InvalidParameter(_))),
            "Inverted limits must be rejected"
        );
    }

    /// Perfect-model step response: plant IS the model.
    /// After sufficient steps, steady-state error should vanish.
    #[test]
    fn imc_perfect_model_zero_steady_state_error() {
        let cfg = first_order_plant_cfg(0.7);
        let mut ctrl = ImcController::<f64, 1>::new(&cfg).unwrap();

        // Simulate with perfect model: the plant is identical to P_m.
        // We simulate the actual plant externally using the same TF.
        let mut plant_sim = TransferFn::<f64, 1>::new([0.5], [-0.5]);

        let setpoint = 1.0_f64;
        let mut y_plant = 0.0_f64;
        let mut u = 0.0_f64;

        for _ in 0..500 {
            u = ctrl.update(setpoint, y_plant).unwrap();
            y_plant = plant_sim.process(u);
        }

        let error = (y_plant - setpoint).abs();
        assert!(
            error < 1e-4,
            "Steady-state error should be near zero with perfect model, got e={:.6}, y={:.6}, u={:.6}",
            error, y_plant, u
        );
    }

    /// Stability under step reference with imperfect model.
    /// With 20 % model gain error, output should still converge (bounded).
    #[test]
    fn imc_imperfect_model_bounded_response() {
        // Model: gain=1, Plant: gain=1.2  (20% mismatch)
        let cfg = ImcConfig::<f64, 1>::new([0.5], [-0.5], 0.9);
        let mut ctrl = ImcController::<f64, 1>::new(&cfg).unwrap();

        // Actual plant: slightly different (b=0.6 instead of 0.5, same pole)
        let mut plant_sim = TransferFn::<f64, 1>::new([0.6], [-0.5]);

        let setpoint = 1.0_f64;
        let mut y_plant = 0.0_f64;
        let mut last_u = 0.0_f64;

        for _ in 0..800 {
            let u = ctrl.update(setpoint, y_plant).unwrap();
            y_plant = plant_sim.process(u);
            last_u = u;
        }

        // With model mismatch, steady-state output should be bounded and positive.
        assert!(
            y_plant > 0.5 && y_plant < 2.0,
            "With 20% mismatch, output should be bounded: y={:.4}, u={:.4}",
            y_plant,
            last_u
        );
    }

    /// After reset, controller behaves as freshly constructed.
    #[test]
    fn imc_reset_clears_state() {
        let cfg = first_order_plant_cfg(0.8);
        let mut ctrl = ImcController::<f64, 1>::new(&cfg).unwrap();
        let mut plant_sim = TransferFn::<f64, 1>::new([0.5], [-0.5]);

        // Run for 100 steps.
        let mut y = 0.0_f64;
        for _ in 0..100 {
            let u = ctrl.update(1.0, y).unwrap();
            y = plant_sim.process(u);
        }

        ctrl.reset();

        // First output after reset with y=0, r=0 should be zero.
        let u_post = ctrl.update(0.0, 0.0).unwrap();
        assert!(
            u_post.abs() < 1e-10,
            "After reset, output on zero input should be zero, got {:.6e}",
            u_post
        );
    }

    /// Saturation clamps control output.
    #[test]
    fn imc_saturation_respected() {
        let cfg = ImcConfig::<f64, 1>::new([0.5], [-0.5], 0.5).with_limits(-0.1, 0.1);
        let mut ctrl = ImcController::<f64, 1>::new(&cfg).unwrap();

        // Large step reference → controller would want large u, but must saturate.
        let u = ctrl.update(100.0, 0.0).unwrap();
        assert!(
            u <= 0.1 + 1e-12,
            "u should be saturated at u_max=0.1, got {:.6}",
            u
        );
    }

    /// Second-order plant model: ensure NP=2 compiles and runs.
    #[test]
    fn imc_second_order_plant() {
        // Discrete second-order plant (Tustin-transformed double-integrator example)
        // b = [0.25, 0.25], a = [-1.0, 0.0]  (poles at z=0 and z=1)
        let cfg = ImcConfig::<f64, 2>::new([0.25, 0.25], [-1.0, 0.0], 0.85);
        let mut ctrl = ImcController::<f64, 2>::new(&cfg).unwrap();
        let mut plant_sim = TransferFn::<f64, 2>::new([0.25, 0.25], [-1.0, 0.0]);

        let mut y = 0.0_f64;
        let mut last_u = 0.0_f64;
        for _ in 0..600 {
            let u = ctrl.update(1.0, y).unwrap();
            y = plant_sim.process(u);
            last_u = u;
        }

        // Output should not blow up.
        assert!(
            y.is_finite() && last_u.is_finite(),
            "Second-order plant response should be finite: y={:.4}, u={:.4}",
            y,
            last_u
        );
    }
}
