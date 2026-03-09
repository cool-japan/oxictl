//! 2-DOF PID: separate setpoint weights for proportional and derivative terms.
//!
//! Transfer function:
//!   u = Kp*(b*r - y) + Ki*∫(r - y) + Kd*d/dt(c*r - y)
//!
//! * `b` (0..1): proportional setpoint weight — `b=0` eliminates the P kick on
//!   a setpoint step; `b=1` gives standard proportional action.
//! * `c` (0..1): derivative setpoint weight — `c=0` implements derivative-on-
//!   measurement only (no D kick on SP step); `c=1` gives standard D action.
#![allow(dead_code)]

use crate::core::scalar::ControlScalar;

/// 2-DOF PID controller.
#[derive(Debug, Clone, Copy)]
pub struct TwoDofPid<S: ControlScalar> {
    pub kp: S,
    pub ki: S,
    pub kd: S,
    /// Setpoint weight for proportional term (0..1).
    pub b: S,
    /// Setpoint weight for derivative term (0..1).
    pub c: S,
    /// Minimum output.
    pub output_min: S,
    /// Maximum output.
    pub output_max: S,
    integrator: S,
    /// Previous value of (c*r - y) used for derivative estimation.
    prev_c_r_minus_y: S,
    initialized: bool,
}

impl<S: ControlScalar> TwoDofPid<S> {
    /// Create a new 2-DOF PID controller.
    pub fn new(kp: S, ki: S, kd: S, b: S, c: S, output_min: S, output_max: S) -> Self {
        Self {
            kp,
            ki,
            kd,
            b,
            c,
            output_min,
            output_max,
            integrator: S::ZERO,
            prev_c_r_minus_y: S::ZERO,
            initialized: false,
        }
    }

    /// Update the controller.
    ///
    /// * `r`  – reference (setpoint)
    /// * `y`  – measured output (feedback)
    /// * `dt` – sample interval
    ///
    /// Returns the clamped control output.
    pub fn update(&mut self, r: S, y: S, dt: S) -> S {
        if dt <= S::ZERO {
            return S::ZERO;
        }

        // Error for integrator (always full error).
        let error = r - y;

        // Proportional term acts on weighted setpoint minus measurement.
        let p_term = self.kp * (self.b * r - y);

        // Derivative term acts on weighted setpoint minus measurement.
        let c_r_minus_y = self.c * r - y;
        let d_term = if self.initialized {
            self.kd * (c_r_minus_y - self.prev_c_r_minus_y) / dt
        } else {
            S::ZERO
        };

        // Unlimited output (before integrator update).
        let u_unlimited = p_term + self.integrator + d_term;
        let u_clamped = u_unlimited.clamp_val(self.output_min, self.output_max);

        // Anti-windup: only integrate when not saturated, or when the error
        // would reduce saturation (clamping method).
        let saturated_high = u_unlimited > self.output_max;
        let saturated_low = u_unlimited < self.output_min;
        let integrate = !saturated_high || error < S::ZERO;
        let integrate = integrate && (!saturated_low || error > S::ZERO);

        if integrate {
            self.integrator += self.ki * error * dt;
        }

        self.prev_c_r_minus_y = c_r_minus_y;
        self.initialized = true;

        u_clamped
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.integrator = S::ZERO;
        self.prev_c_r_minus_y = S::ZERO;
        self.initialized = false;
    }

    /// Construct a standard (1-DOF) PID: b=1, c=1 (setpoint weighting disabled).
    pub fn standard(kp: S, ki: S, kd: S, output_min: S, output_max: S) -> Self {
        Self::new(kp, ki, kd, S::ONE, S::ONE, output_min, output_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_only_b_one() {
        // b=1, c=1 → standard 2-DOF reduces to standard PID.
        let mut pid = TwoDofPid::<f64>::standard(2.0, 0.0, 0.0, -100.0, 100.0);
        // error = 10-0 = 10, P = 2*10 = 20
        let out = pid.update(10.0, 0.0, 0.01);
        assert!((out - 20.0).abs() < 1e-12, "out={out}");
    }

    #[test]
    fn b_zero_no_p_kick_on_setpoint_step() {
        // With b=0: P = Kp*(0*r - y) = -Kp*y.  For y=0 → P=0.
        let mut pid = TwoDofPid::<f64>::new(2.0, 0.0, 0.0, 0.0, 0.0, -100.0, 100.0);
        let out = pid.update(100.0, 0.0, 0.01);
        // P = Kp*(b*r - y) = 2*(0*100 - 0) = 0; I=0, D=0 → 0
        assert!(out.abs() < 1e-12, "out={out}");
    }

    #[test]
    fn integrator_accumulates() {
        let mut pid = TwoDofPid::<f64>::new(0.0, 1.0, 0.0, 1.0, 1.0, -1000.0, 1000.0);
        // Ki=1, error=5 each step, dt=0.1.
        // Output is computed using the integrator value from the START of the step;
        // then the integrator is updated (forward-Euler):
        //   Step 1: integrator=0 → output=0; then integrator += 5*0.1=0.5
        //   Step 2: integrator=0.5 → output=0.5; then integrator += 0.5=1.0
        //   Step 3: integrator=1.0 → output=1.0
        let dt = 0.1_f64;
        let out1 = pid.update(5.0, 0.0, dt);
        let out2 = pid.update(5.0, 0.0, dt);
        let out3 = pid.update(5.0, 0.0, dt);
        assert!(out1.abs() < 1e-12, "out1={out1}");
        assert!((out2 - 0.5).abs() < 1e-12, "out2={out2}");
        assert!((out3 - 1.0).abs() < 1e-12, "out3={out3}");
    }

    #[test]
    fn output_clamped_to_limits() {
        let mut pid = TwoDofPid::<f64>::standard(100.0, 0.0, 0.0, -5.0, 5.0);
        let out = pid.update(10.0, 0.0, 0.01);
        assert!((out - 5.0).abs() < 1e-12, "out={out}");
    }

    #[test]
    fn reset_clears_state() {
        let mut pid = TwoDofPid::<f64>::standard(1.0, 2.0, 0.0, -100.0, 100.0);
        for _ in 0..10 {
            pid.update(5.0, 0.0, 0.01);
        }
        pid.reset();
        // After reset integrator=0, initialized=false.
        let out = pid.update(0.0, 0.0, 0.01);
        // P=0, I=0, D=0 (not initialized) → 0
        assert!(out.abs() < 1e-12, "out after reset={out}");
    }

    #[test]
    fn c_zero_no_d_kick_on_setpoint_step() {
        // c=0: D acts only on -y (no contribution from setpoint).
        // On first update no D term (not initialized). On second update with
        // same y=0 and different r → c*r - y = 0 - 0 = 0, no change → D=0.
        let mut pid = TwoDofPid::<f64>::new(0.0, 0.0, 10.0, 1.0, 0.0, -100.0, 100.0);
        pid.update(0.0, 0.0, 0.01); // prime
                                    // Change setpoint drastically, y stays 0: c*r-y = 0*100-0 = 0.
        let out = pid.update(100.0, 0.0, 0.01);
        // D = Kd * (0 - 0)/dt = 0
        assert!(out.abs() < 1e-12, "out={out}");
    }
}
