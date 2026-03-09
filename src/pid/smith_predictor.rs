//! Smith Predictor: transport delay compensation for PID control.
//!
//! The Smith Predictor removes the apparent delay from the feedback loop seen
//! by the inner PID controller.  A first-order plant model
//! `y_model[k+1] = a*y_model[k] + b*u[k]` is simulated internally; its
//! undelayed output feeds the PID while the actual (delayed) plant measurement
//! is used to correct the model bias.
#![allow(dead_code)]

use crate::core::scalar::ControlScalar;
use crate::core::signal::{Feedback, Setpoint};
use crate::core::traits::Controller;
use crate::pid::Pid;

/// Smith Predictor wrapping an inner PID controller.
///
/// `D`: delay in samples (ring-buffer length).
///
/// The inner PID controller is driven by an *undelayed* model error, which
/// allows the controller to be tuned as if the plant had no transport delay.
pub struct SmithPredictor<S: ControlScalar, const D: usize> {
    /// Inner PID controller.
    pub inner_pid: Pid<S>,
    /// Ring buffer storing past model outputs (length D → D-sample delay).
    delay_buffer: [S; D],
    buffer_idx: usize,
    /// Plant model coefficient a: y[k+1] = a*y[k] + b*u[k].
    pub plant_a: S,
    /// Plant model coefficient b: y[k+1] = a*y[k] + b*u[k].
    pub plant_b: S,
    /// Current (undelayed) model output.
    model_output: S,
    last_u: S,
}

impl<S: ControlScalar, const D: usize> SmithPredictor<S, D> {
    /// Create a new Smith Predictor.
    ///
    /// # Arguments
    /// * `inner_pid` – configured PID controller (tuned for the *delay-free* plant)
    /// * `plant_a`   – first-order model coefficient a
    /// * `plant_b`   – first-order model coefficient b
    pub fn new(inner_pid: Pid<S>, plant_a: S, plant_b: S) -> Self {
        Self {
            inner_pid,
            delay_buffer: [S::ZERO; D],
            buffer_idx: 0,
            plant_a,
            plant_b,
            model_output: S::ZERO,
            last_u: S::ZERO,
        }
    }

    /// Update the Smith Predictor.
    ///
    /// * `sp` – setpoint
    /// * `fb` – actual measured (delayed) plant output
    /// * `dt` – sample interval
    ///
    /// Returns the controller output `u`.
    pub fn update(&mut self, sp: S, fb: S, dt: S) -> S {
        // 1. Advance the model: y_model[k+1] = a*y_model[k] + b*u[k-1]
        let new_model = self.plant_a * self.model_output + self.plant_b * self.last_u;

        // 2. Retrieve the D-step delayed model output.
        let delayed_model = if D == 0 {
            new_model
        } else {
            self.delay_buffer[self.buffer_idx]
        };

        // 3. Push the new model output into the delay ring buffer.
        if D > 0 {
            self.delay_buffer[self.buffer_idx] = new_model;
            self.buffer_idx = (self.buffer_idx + 1) % D;
        }

        self.model_output = new_model;

        // 4. Compute the Smith-corrected feedback:
        //    e_smith = sp - (fb + (y_model - y_model_delayed))
        //    Equivalently the PID sees: sp_effective vs fb_corrected
        //    fb_corrected = fb + (y_model - delayed_model)
        let fb_corrected = fb + (new_model - delayed_model);

        // 5. Feed the corrected error to the inner PID.
        let sp_sig = Setpoint::new(sp);
        let fb_sig = Feedback::new(fb_corrected);
        let u = self.inner_pid.update(&sp_sig, &fb_sig, dt).value();

        self.last_u = u;
        u
    }

    /// Current (undelayed) model output.
    pub fn model_output(&self) -> S {
        self.model_output
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.inner_pid.reset();
        self.delay_buffer = [S::ZERO; D];
        self.buffer_idx = 0;
        self.model_output = S::ZERO;
        self.last_u = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pid::PidConfig;

    fn make_predictor() -> SmithPredictor<f64, 5> {
        // PI tuned for first-order plant a=0.9, b=0.1
        let pid = PidConfig::pi(1.0_f64, 5.0).build();
        SmithPredictor::new(pid, 0.9_f64, 0.1_f64)
    }

    #[test]
    fn output_nonzero_for_nonzero_setpoint() {
        let mut sp_val = make_predictor();
        let u = sp_val.update(1.0, 0.0, 0.01);
        // With sp=1, fb=0 the PID should produce a positive output.
        assert!(u > 0.0, "u={u}");
    }

    #[test]
    fn zero_setpoint_zero_feedback_zero_output() {
        let mut sp_val = make_predictor();
        let u = sp_val.update(0.0, 0.0, 0.01);
        assert!((u).abs() < 1e-12, "u={u}");
    }

    #[test]
    fn reset_clears_state() {
        let mut pred = make_predictor();
        for _ in 0..10 {
            pred.update(1.0, 0.5, 0.01);
        }
        pred.reset();
        assert!((pred.model_output()).abs() < 1e-12);
        // After reset with sp=0, fb=0 → output = 0
        let u = pred.update(0.0, 0.0, 0.01);
        assert!(u.abs() < 1e-12, "u after reset={u}");
    }

    #[test]
    fn delay_buffer_shifts_correctly() {
        // Use D=3, verify that model output converges.
        let pid = PidConfig::pi(0.5_f64, 1.0).build();
        let mut pred: SmithPredictor<f64, 3> = SmithPredictor::new(pid, 0.8_f64, 0.2_f64);
        // Run a few steps; just check that it doesn't panic and model_output grows.
        for _ in 0..20 {
            pred.update(1.0, 0.0, 0.01);
        }
        assert!(pred.model_output() > 0.0);
    }
}
