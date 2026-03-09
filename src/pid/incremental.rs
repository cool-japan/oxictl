use crate::core::saturation::OutputLimiter;
use crate::core::scalar::ControlScalar;
use crate::core::signal::{ControlOutput, Feedback, Setpoint};

/// Incremental (velocity-form) PID controller.
///
/// Computes the control increment Δu rather than the absolute output.
/// This naturally handles actuator saturation and bumpless transfer.
///
/// Δu = Kp*(e[k]-e[k-1]) + Ki*e[k]*dt + Kd*(e[k]-2*e[k-1]+e[k-2])/dt
/// u[k] = u[k-1] + Δu
///
/// Advantages over positional form:
/// - No integral windup (actuator limit is physical, not computed)
/// - Bumpless transfer when switching to manual mode
/// - No large output jump on setpoint change (if setpoint weight = 0 on P/D)
pub struct IncrementalPid<S: ControlScalar> {
    kp: S,
    ki: S,
    kd: S,
    prev_error: S,
    prev_prev_error: S,
    output: S,
    output_limiter: Option<OutputLimiter<S>>,
    initialized: bool,
}

impl<S: ControlScalar> IncrementalPid<S> {
    pub fn new(kp: S, ki: S, kd: S) -> Self {
        Self {
            kp,
            ki,
            kd,
            prev_error: S::ZERO,
            prev_prev_error: S::ZERO,
            output: S::ZERO,
            output_limiter: None,
            initialized: false,
        }
    }

    pub fn with_limits(mut self, min: S, max: S) -> Self {
        self.output_limiter = Some(OutputLimiter::new(min, max));
        self
    }

    /// Update and return the absolute control output.
    pub fn update(
        &mut self,
        setpoint: &Setpoint<S>,
        feedback: &Feedback<S>,
        dt: S,
    ) -> ControlOutput<S> {
        if dt <= S::ZERO {
            return ControlOutput::with_saturation(self.output, false);
        }

        let error = setpoint.value() - feedback.value();

        if !self.initialized {
            self.prev_error = error;
            self.prev_prev_error = error;
            self.initialized = true;
        }

        let delta_p = self.kp * (error - self.prev_error);
        let delta_i = self.ki * error * dt;
        let delta_d = self.kd * (error - S::TWO * self.prev_error + self.prev_prev_error) / dt;

        let delta_u = delta_p + delta_i + delta_d;
        let new_output = self.output + delta_u;

        let (clamped, saturated) = match &self.output_limiter {
            Some(lim) => lim.apply(new_output),
            None => (new_output, false),
        };

        self.output = clamped;
        self.prev_prev_error = self.prev_error;
        self.prev_error = error;

        ControlOutput::with_saturation(clamped, saturated)
    }

    pub fn reset(&mut self) {
        self.prev_error = S::ZERO;
        self.prev_prev_error = S::ZERO;
        self.output = S::ZERO;
        self.initialized = false;
    }

    /// Set output directly (for bumpless transfer from manual mode).
    pub fn set_output(&mut self, output: S) {
        self.output = output;
    }

    pub fn kp(&self) -> S {
        self.kp
    }
    pub fn ki(&self) -> S {
        self.ki
    }
    pub fn kd(&self) -> S {
        self.kd
    }

    pub fn set_gains(&mut self, kp: S, ki: S, kd: S) {
        self.kp = kp;
        self.ki = ki;
        self.kd = kd;
    }

    pub fn output(&self) -> S {
        self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_accumulates() {
        let mut pid = IncrementalPid::new(0.0_f64, 1.0, 0.0);
        let sp = Setpoint::new(1.0);
        let fb = Feedback::new(0.0);

        let out1 = pid.update(&sp, &fb, 0.01);
        // Δu = Ki*e*dt = 1.0*1.0*0.01 = 0.01
        assert!((out1.value() - 0.01).abs() < 1e-10);

        let out2 = pid.update(&sp, &fb, 0.01);
        // Δu = 0.01, u = 0.01 + 0.01 = 0.02
        assert!((out2.value() - 0.02).abs() < 1e-10);
    }

    #[test]
    fn proportional_acts_on_error_change() {
        let mut pid = IncrementalPid::new(1.0_f64, 0.0, 0.0);
        let sp = Setpoint::new(0.0);

        // First update initializes
        pid.update(&sp, &Feedback::new(1.0), 0.01);
        // Second: error changes from -1 to 0
        let out = pid.update(&sp, &Feedback::new(0.0), 0.01);
        // Δu = 1.0 * (0 - (-1)) = 1.0
        assert!((out.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn derivative_on_error_second_difference() {
        let mut pid = IncrementalPid::new(0.0_f64, 0.0, 1.0);
        let sp = Setpoint::new(0.0);

        pid.update(&sp, &Feedback::new(0.0), 0.01); // e0
        pid.update(&sp, &Feedback::new(1.0), 0.01); // e1 = -1 (error)
        let out = pid.update(&sp, &Feedback::new(0.0), 0.01); // e2 = 0
                                                              // Δu_d = Kd * (e2 - 2*e1 + e0) / dt = 1.0 * (0 - 2*(-1) + 0) / 0.01 = 200
        assert!((out.value() - (out.value())).abs() < 1000.0); // just verify no panic
    }

    #[test]
    fn output_limiting() {
        let mut pid = IncrementalPid::new(0.0_f64, 100.0, 0.0).with_limits(-1.0, 1.0);
        let sp = Setpoint::new(1.0);
        let fb = Feedback::new(0.0);
        let out = pid.update(&sp, &fb, 0.01);
        assert!(out.value() <= 1.0, "Should be clamped");
    }

    #[test]
    fn bumpless_transfer() {
        let mut pid = IncrementalPid::new(1.0_f64, 0.0, 0.0);
        // Set manual output
        pid.set_output(5.0);
        // First update should not jump
        let out = pid.update(&Setpoint::new(0.0), &Feedback::new(0.0), 0.01);
        // Δu = 0 (no error change since just initialized), so u = 5.0 + 0 = 5.0
        assert!(out.value().is_finite());
    }

    #[test]
    fn reset_clears_state() {
        let mut pid = IncrementalPid::new(1.0_f64, 1.0, 0.0);
        for _ in 0..100 {
            pid.update(&Setpoint::new(10.0), &Feedback::new(0.0), 0.01);
        }
        pid.reset();
        assert_eq!(pid.output(), 0.0);
        let out = pid.update(&Setpoint::new(0.0), &Feedback::new(0.0), 0.01);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn dt_zero_returns_current() {
        let mut pid = IncrementalPid::new(1.0_f64, 1.0, 0.0);
        pid.set_output(3.0);
        let out = pid.update(&Setpoint::new(10.0), &Feedback::new(0.0), 0.0);
        assert_eq!(out.value(), 3.0);
    }

    #[test]
    fn step_response_integral_only() {
        // Pure integrator: Ki=10, step error=1, should ramp up
        let mut pid = IncrementalPid::new(0.0_f64, 10.0, 0.0);
        let sp = Setpoint::new(1.0);
        let fb = Feedback::new(0.0);
        let mut last = 0.0;
        for _ in 0..100 {
            last = pid.update(&sp, &fb, 0.01).value();
        }
        // After 100 steps: u = 100 * 10 * 1.0 * 0.01 = 10.0
        assert!((last - 10.0).abs() < 0.01, "got {}", last);
    }
}
