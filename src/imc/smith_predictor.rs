// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright COOLJAPAN OU (Team Kitasan)
//
// Smith Predictor for dead-time (transport-delay) compensation.
//
// Theory
// ------
// Classic Smith predictor restructures the feedback loop so the controller
// sees the plant *without* its dead time:
//
//   Primary controller C(z) sees the augmented error:
//     e_sp(k) = r(k) − [ y(k) + P_m(z)·u(k) − P_m(z)·z^{-d}·u(k) ]
//             = r(k) − y(k) − (y_model_undelayed − y_model_delayed)
//
// where
//   y_model_undelayed = P_m(z) driven by u  (model without dead-time)
//   y_model_delayed   = P_m(z)·z^{-d} driven by u  (model with d-step delay)
//
// The net effect: as long as the model is accurate, the controller effectively
// cancels the dead time and operates on the delay-free plant.
//
// Implementation
// --------------
// * `DELAY` – dead-time in samples (const generic; zero overhead at runtime)
// * `NP`    – order of the primary controller transfer function
// * `NM`    – order of the process model transfer function
//
// The primary controller C(z) is a PI implemented as a TransferFn with
// NP = 1 (first-order IIR equivalent of a bilinear-transform PI).

use crate::core::scalar::ControlScalar;
use crate::core::transfer_fn::TransferFn;

use super::ImcError;

// ──────────────────────────────────────────────────────────────
// Config
// ──────────────────────────────────────────────────────────────

/// Configuration for [`SmithPredictor`].
///
/// The primary controller is a discrete PI:
///   C(z) = Kp + Ki·Ts / (1 − z^{-1})
///
/// which is realised as a TransferFn<S,1> with:
///   b = [Kp + Ki·Ts,  −Kp]
///   a = [−1]
///
/// `NM` is the order of the process model.
#[derive(Debug, Clone, Copy)]
pub struct SmithPredictorConfig<S: ControlScalar, const NM: usize> {
    /// Proportional gain of the primary PI controller.
    pub kp: S,
    /// Integral gain.
    pub ki: S,
    /// Sample time [s] – used only to scale integral gain.
    pub dt: S,
    /// Numerator of the process model (without dead-time), length NM.
    pub model_b: [S; NM],
    /// Denominator of the process model, length NM (implicit leading 1).
    pub model_a: [S; NM],
    /// Control output lower saturation limit.
    pub u_min: S,
    /// Control output upper saturation limit.
    pub u_max: S,
}

impl<S: ControlScalar, const NM: usize> SmithPredictorConfig<S, NM> {
    /// Construct with reasonable defaults (no saturation).
    pub fn new(kp: S, ki: S, dt: S, model_b: [S; NM], model_a: [S; NM]) -> Self {
        let big = S::from_f64(1e9);
        Self {
            kp,
            ki,
            dt,
            model_b,
            model_a,
            u_min: -big,
            u_max: big,
        }
    }

    /// Attach saturation limits.
    pub fn with_limits(mut self, u_min: S, u_max: S) -> Self {
        self.u_min = u_min;
        self.u_max = u_max;
        self
    }
}

// ──────────────────────────────────────────────────────────────
// Controller
// ──────────────────────────────────────────────────────────────

/// Smith Predictor dead-time compensator.
///
/// Generic parameters
/// ------------------
/// * `S`     – scalar type (`f32` or `f64`)
/// * `NM`    – order of the process model transfer function
/// * `DELAY` – dead-time in discrete samples (compile-time constant)
///
/// For `DELAY = 0` the structure degenerates to a plain PI controller.
#[derive(Debug, Clone)]
pub struct SmithPredictor<S: ControlScalar, const NM: usize, const DELAY: usize> {
    /// Primary PI controller realised as a first-order IIR (TransferFn<S,1>).
    inner_controller: TransferFn<S, 1>,
    /// Delay-free process model P_m(z).
    model: TransferFn<S, NM>,
    /// Delayed process model  P_m(z)·z^{-d}: same dynamics, delay applied via
    /// the circular buffer below.
    model_delayed: TransferFn<S, NM>,
    /// Circular delay buffer holding the last DELAY model outputs.
    /// When DELAY = 0 this array is zero-sized and compiles away.
    delay_buffer: [S; DELAY],
    /// Write pointer into the circular buffer.
    delay_head: usize,
    /// Saturation limits.
    u_min: S,
    u_max: S,
    /// Last applied (saturated) control – used for anti-windup conditioning.
    last_u: S,
}

impl<S: ControlScalar, const NM: usize, const DELAY: usize> SmithPredictor<S, NM, DELAY> {
    /// Build a Smith Predictor from configuration.
    ///
    /// Returns `Err` if:
    /// * `dt ≤ 0`
    /// * `u_min ≥ u_max`
    pub fn new(cfg: &SmithPredictorConfig<S, NM>) -> Result<Self, ImcError> {
        if cfg.dt <= S::ZERO {
            return Err(ImcError::InvalidParameter("dt must be > 0"));
        }
        if cfg.u_min >= cfg.u_max {
            return Err(ImcError::InvalidParameter("u_min must be < u_max"));
        }

        // Bilinear-transform PI in TransferFn<S,1> form:
        //   C(z) = (Kp + Ki*Ts/2 + (Ki*Ts/2 - Kp)*z^{-1}) / (1 - z^{-1})
        // Simplified direct PI (forward Euler integrator):
        //   b[0] = Kp + Ki*dt,  b[1]=-Kp  but that needs order-1 → b=[b0], a=[-1]
        // We use the matched-pole zero form for a first-order TransferFn<S,1>:
        //   numerator b0  = Kp
        //   denominator a0= −1  (pure integrator pole at z=1)
        // plus the integral accumulation handled via the IIR:
        //   H(z) = (Kp*(1-z^{-1}) + Ki*dt) / (1 - z^{-1})
        //        = (Kp + Ki*dt - Kp*z^{-1}) / (1 - z^{-1})
        // In TransferFn<S,1>: b=[Kp+Ki*dt], a=[-1] — that captures only
        // proportional+integral for step inputs.  For full fidelity we keep
        // b = [Kp + Ki*dt] and a = [-1] which is the discrete integrator
        // representation (accumulator):
        //
        //   Actually the cleanest approach: store separate P and I terms
        //   inside a 1-state TransferFn using:
        //     b0 = Kp + Ki*dt/2,  b_lag = -(Kp - Ki*dt/2)  [bilinear]
        //     a = -1
        //
        //   But TransferFn<S,N> requires numerator[N] = denominator[N].
        //   For N=1: b=[b0], a=[a0].  We encode bilinear PI as:
        //     b0 = Kp + Ki*dt*0.5
        //     a0 = -1   (pole at z=1)
        //   This gives b0/(1 + a0*z^{-1}) = b0/(1 - z^{-1}) — pure I only.
        //
        // For a well-behaved PI in TransferFn<S,2> (2nd order representation):
        //   H(z) = (b0 + b1*z^{-1}) / (1 + a1*z^{-1})
        //   with b0 = Kp + Ki*dt, b1 = -Kp, a1 = -1
        //
        // However the struct only exposes TransferFn<S,1>.
        // We therefore use TransferFn<S,1> as a PI by noting:
        //
        //   For N=1: only ONE numerator and ONE denominator coefficient.
        //   We implement the PI as: output += Kp*e(k) + Ki*dt*integral
        //   The integrator is baked into the IIR as H(z) = b0/(1 - z^{-1})
        //   with b0 = Ki*dt, giving I(k) = I(k-1) + Ki*dt*e(k).
        //   Then u(k) = Kp*e(k) + I(k).
        //
        // Because TransferFn<S,1> is a single-term IIR and cannot separately
        // implement P+I in one shot, we use TWO separate TransferFns:
        //   - integrator_tf: H(z) = Ki*dt / (1 - z^{-1})   → I term
        //   - But we only have one `inner_controller` field.
        //
        // Decision: implement PI directly in the struct without a second TF
        // by using the TransferFn<S,1> for the integrator term only, and
        // applying the P term inline in `update`.
        //
        // Actually the cleanest approach compatible with TransferFn<S,N>:
        // Use the first-order TF with b=[Kp+Ki*dt, -Kp], a=[-1] would
        // require N=2 with an implicit leading 1 in the denominator.
        //
        // TransferFn<S,2>: b=[b0,b1], a=[a0,a1]
        //   H(z) = (b0 + b1*z^{-1}) / (1 + a0*z^{-1} + a1*z^{-1})
        //   PI:  b0=Kp+Ki*dt, b1=-Kp,  a0=-1, a1=0
        //
        // For simplicity: use TransferFn<S,1> as integrator with b=[Ki*dt],
        // a=[-1], and apply Kp inline.  This is numerically equivalent.
        //
        // HOWEVER the field is `inner_controller: TransferFn<S, 1>`.
        // We encode it as: b=[Ki*dt], a=[-1] and add P in update().

        let ki_dt = cfg.ki * cfg.dt;
        // b=[Ki*dt], a=[-1] → H(z) = Ki*dt / (1 - z^{-1}) (pure integrator)
        let inner_controller = TransferFn::<S, 1>::new([ki_dt], [-S::ONE]);

        let model = TransferFn::<S, NM>::new(cfg.model_b, cfg.model_a);
        let model_delayed = TransferFn::<S, NM>::new(cfg.model_b, cfg.model_a);

        Ok(Self {
            inner_controller,
            model,
            model_delayed,
            delay_buffer: [S::ZERO; DELAY],
            delay_head: 0,
            u_min: cfg.u_min,
            u_max: cfg.u_max,
            last_u: S::ZERO,
        })
    }

    /// Compute next control output.
    ///
    /// Arguments
    /// ---------
    /// * `setpoint`     – desired output r(k)
    /// * `plant_output` – measured plant output y(k)
    ///
    /// Returns the (saturated) control signal u(k).
    pub fn update(&mut self, setpoint: S, plant_output: S, kp: S) -> Result<S, ImcError> {
        // Advance delay-free model with last applied control.
        let y_model_undelayed = self.model.process(self.last_u);

        // Advance the delayed model and push through the delay buffer.
        let y_model_before_delay = self.model_delayed.process(self.last_u);

        // Insert into circular buffer and read the oldest sample.
        let y_model_delayed = if DELAY == 0 {
            // Zero dead-time: bypass buffer entirely.
            y_model_before_delay
        } else {
            let old = self.delay_buffer[self.delay_head];
            self.delay_buffer[self.delay_head] = y_model_before_delay;
            self.delay_head = (self.delay_head + 1) % DELAY;
            old
        };

        // Smith predictor augmented error:
        //   e_sp(k) = r(k) − y(k) − (y_model_undelayed(k) − y_model_delayed(k))
        let predictor_correction = y_model_undelayed - y_model_delayed;
        let error = setpoint - plant_output - predictor_correction;

        // PI: integral term via TransferFn<S,1> (integrator)
        let i_term = self.inner_controller.process(error);

        // Proportional + integral
        let u_raw = kp * error + i_term;

        // Saturate.
        let u_sat = u_raw.clamp_val(self.u_min, self.u_max);
        self.last_u = u_sat;

        Ok(u_sat)
    }

    /// Reset all internal states.
    pub fn reset(&mut self) {
        self.inner_controller.reset();
        self.model.reset();
        self.model_delayed.reset();
        self.delay_buffer = [S::ZERO; DELAY];
        self.delay_head = 0;
        self.last_u = S::ZERO;
    }

    /// Retrieve the current content of the delay buffer (diagnostic).
    pub fn delay_buffer(&self) -> &[S; DELAY] {
        &self.delay_buffer
    }
}

// ──────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple first-order process: P(z) = 0.2 / (1 - 0.8·z^{-1}), DC gain = 1.
    fn first_order_cfg(kp: f64, ki: f64, dt: f64) -> SmithPredictorConfig<f64, 1> {
        // b=[0.2], a=[-0.8]  DC = 0.2/(1-0.8) = 1
        SmithPredictorConfig::new(kp, ki, dt, [0.2], [-0.8])
    }

    #[test]
    fn smith_construction_ok() {
        let cfg = first_order_cfg(2.0, 0.5, 0.1);
        let sp = SmithPredictor::<f64, 1, 5>::new(&cfg);
        assert!(sp.is_ok(), "Valid config should construct OK");
    }

    #[test]
    fn smith_construction_invalid_dt() {
        let cfg = SmithPredictorConfig::<f64, 1>::new(1.0, 0.1, 0.0, [0.2], [-0.8]);
        let sp = SmithPredictor::<f64, 1, 3>::new(&cfg);
        assert!(
            matches!(sp, Err(ImcError::InvalidParameter(_))),
            "dt=0 must be rejected"
        );
    }

    #[test]
    fn smith_construction_invalid_limits() {
        let cfg = SmithPredictorConfig::<f64, 1>::new(1.0, 0.1, 0.1, [0.2], [-0.8])
            .with_limits(5.0, -5.0); // inverted
        let sp = SmithPredictor::<f64, 1, 2>::new(&cfg);
        assert!(
            matches!(sp, Err(ImcError::InvalidParameter(_))),
            "Inverted limits must be rejected"
        );
    }

    /// With DELAY=0, Smith predictor should behave like a plain PI.
    /// The correction term is zero, so it should converge as a normal PI.
    #[test]
    fn smith_zero_delay_converges_to_setpoint() {
        let cfg = first_order_cfg(3.0, 1.0, 0.05);
        let kp = 3.0_f64;
        let mut sp = SmithPredictor::<f64, 1, 0>::new(&cfg).unwrap();

        // Simulate with perfect model (plant = model).
        let mut y = 0.0_f64;
        let mut plant_state = TransferFn::<f64, 1>::new([0.2], [-0.8]);

        let setpoint = 1.0_f64;
        for _ in 0..400 {
            let u = sp.update(setpoint, y, kp).unwrap();
            y = plant_state.process(u);
        }

        let error = (y - setpoint).abs();
        assert!(
            error < 0.05,
            "PI with zero delay should converge: e={:.4}, y={:.4}",
            error,
            y
        );
    }

    /// With dead-time compensation, the controller should still converge.
    /// We simulate a plant with 5-sample pure delay: y(k) = P_m(z)·u(k-5).
    #[test]
    fn smith_delay_compensation_step_response() {
        const D: usize = 5;
        let cfg = first_order_cfg(2.5, 0.8, 0.05);
        let kp = 2.5_f64;
        let mut sp = SmithPredictor::<f64, 1, D>::new(&cfg).unwrap();

        // Plant simulation: TF + D-sample pure delay.
        let mut plant_tf = TransferFn::<f64, 1>::new([0.2], [-0.8]);
        let mut plant_delay: [f64; D] = [0.0; D];
        let mut delay_head = 0_usize;

        let setpoint = 1.0_f64;
        let mut y_delayed = 0.0_f64;

        for _ in 0..600 {
            let u = sp.update(setpoint, y_delayed, kp).unwrap();

            // Plant step: compute un-delayed output.
            let y_plant_raw = plant_tf.process(u);

            // Push through the delay buffer.
            let oldest = plant_delay[delay_head];
            plant_delay[delay_head] = y_plant_raw;
            delay_head = (delay_head + 1) % D;
            y_delayed = oldest;
        }

        let error = (y_delayed - setpoint).abs();
        assert!(
            error < 0.1,
            "Smith predictor should compensate delay: e={:.4}, y={:.4}",
            error,
            y_delayed
        );
    }

    /// Verify that the delay buffer has the right size (compile-time check via
    /// inspecting the public accessor).
    #[test]
    fn smith_delay_buffer_correct_size() {
        let cfg = first_order_cfg(1.0, 0.2, 0.1);
        let sp = SmithPredictor::<f64, 1, 8>::new(&cfg).unwrap();
        assert_eq!(sp.delay_buffer().len(), 8);
    }

    /// After reset, a previously active controller returns zero output on zero input.
    #[test]
    fn smith_reset_clears_state() {
        let cfg = first_order_cfg(2.0, 0.5, 0.1);
        let kp = 2.0_f64;
        let mut sp = SmithPredictor::<f64, 1, 3>::new(&cfg).unwrap();
        let mut plant = TransferFn::<f64, 1>::new([0.2], [-0.8]);

        let mut y = 0.0_f64;
        for _ in 0..100 {
            let u = sp.update(1.0, y, kp).unwrap();
            y = plant.process(u);
        }

        sp.reset();

        let u_after = sp.update(0.0, 0.0, kp).unwrap();
        assert!(
            u_after.abs() < 1e-10,
            "After reset, zero-input should give zero output, got {:.4e}",
            u_after
        );
    }

    /// Saturation test: large setpoint must be clamped.
    #[test]
    fn smith_saturation_respected() {
        let cfg = SmithPredictorConfig::<f64, 1>::new(5.0, 1.0, 0.05, [0.2], [-0.8])
            .with_limits(-0.5, 0.5);
        let mut sp = SmithPredictor::<f64, 1, 2>::new(&cfg).unwrap();

        let u = sp.update(1000.0, 0.0, 5.0).unwrap();
        assert!(
            u <= 0.5 + 1e-12,
            "u should be saturated at 0.5, got {:.4}",
            u
        );
    }
}
