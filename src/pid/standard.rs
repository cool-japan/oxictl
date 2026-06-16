use crate::core::saturation::OutputLimiter;
use crate::core::scalar::PidScalar;
use crate::core::signal::{ControlOutput, Feedback, Setpoint};
use crate::core::traits::Controller;
use crate::pid::anti_windup::AntiWindupMethod;
use crate::pid::derivative_filter::DerivativeFilter;

/// Configuration for constructing a PID controller.
#[derive(Debug, Clone)]
pub struct PidConfig<S: PidScalar> {
    pub kp: S,
    pub ki: S,
    pub kd: S,
    /// Setpoint weight for proportional term (2-DOF). Default 1.0.
    pub beta: S,
    /// Setpoint weight for derivative term (2-DOF). Default 0.0.
    pub gamma: S,
    /// Output limiter (min/max clamp).
    pub output_limiter: Option<OutputLimiter<S>>,
    /// Anti-windup method.
    pub anti_windup: AntiWindupMethod<S>,
    /// Derivative filter time constant. None = no filtering.
    pub derivative_filter_tau: Option<S>,
}

impl<S: PidScalar> PidConfig<S> {
    /// Create a P-only controller.
    pub fn p(kp: S) -> Self {
        Self {
            kp,
            ki: S::ZERO,
            kd: S::ZERO,
            beta: S::ONE,
            gamma: S::ZERO,
            output_limiter: None,
            anti_windup: AntiWindupMethod::None,
            derivative_filter_tau: None,
        }
    }

    /// Create a PI controller.
    pub fn pi(kp: S, ki: S) -> Self {
        Self {
            kp,
            ki,
            kd: S::ZERO,
            beta: S::ONE,
            gamma: S::ZERO,
            output_limiter: None,
            anti_windup: AntiWindupMethod::Clamping,
            derivative_filter_tau: None,
        }
    }

    /// Create a PID controller.
    pub fn pid(kp: S, ki: S, kd: S) -> Self {
        // from_f64 and Float::max are not in PidScalar; use from_int + explicit comparison
        let safe_kp = if kp > S::EPSILON { kp } else { S::EPSILON };
        Self {
            kp,
            ki,
            kd,
            beta: S::ONE,
            gamma: S::ZERO,
            output_limiter: None,
            anti_windup: AntiWindupMethod::Clamping,
            derivative_filter_tau: Some(kd / (S::from_int(10) * safe_kp)),
        }
    }

    /// Set output limits.
    pub fn with_limits(mut self, min: S, max: S) -> Self {
        self.output_limiter = Some(OutputLimiter::new(min, max));
        self
    }

    /// Set anti-windup method.
    pub fn with_anti_windup(mut self, method: AntiWindupMethod<S>) -> Self {
        self.anti_windup = method;
        self
    }

    /// Set 2-DOF setpoint weights.
    pub fn with_setpoint_weights(mut self, beta: S, gamma: S) -> Self {
        self.beta = beta;
        self.gamma = gamma;
        self
    }

    /// Build the PID controller.
    pub fn build(self) -> Pid<S> {
        let d_filter = self.derivative_filter_tau.map(DerivativeFilter::new);

        Pid {
            kp: self.kp,
            ki: self.ki,
            kd: self.kd,
            beta: self.beta,
            gamma: self.gamma,
            integral: S::ZERO,
            prev_error: None,
            prev_measurement: None,
            output_limiter: self.output_limiter,
            anti_windup: self.anti_windup,
            d_filter,
            saturated: false,
        }
    }
}

/// PID controller with 2-DOF support, anti-windup, and derivative filtering.
#[derive(Debug, Clone)]
pub struct Pid<S: PidScalar> {
    kp: S,
    ki: S,
    kd: S,
    beta: S,
    gamma: S,
    integral: S,
    prev_error: Option<S>,
    prev_measurement: Option<S>,
    output_limiter: Option<OutputLimiter<S>>,
    anti_windup: AntiWindupMethod<S>,
    d_filter: Option<DerivativeFilter<S>>,
    saturated: bool,
}

impl<S: PidScalar> Pid<S> {
    pub fn kp(&self) -> S {
        self.kp
    }

    pub fn ki(&self) -> S {
        self.ki
    }

    pub fn kd(&self) -> S {
        self.kd
    }

    pub fn integral(&self) -> S {
        self.integral
    }

    pub fn set_gains(&mut self, kp: S, ki: S, kd: S) {
        self.kp = kp;
        self.ki = ki;
        self.kd = kd;
    }
}

impl<S: PidScalar> Controller<S> for Pid<S> {
    fn update(
        &mut self,
        setpoint: &Setpoint<S>,
        feedback: &Feedback<S>,
        dt: S,
    ) -> ControlOutput<S> {
        // Guard against dt = 0
        if dt <= S::ZERO {
            return ControlOutput::with_saturation(S::ZERO, self.saturated);
        }

        let sp = setpoint.value();
        let pv = feedback.value();
        let error = sp - pv;

        // 2-DOF: Proportional acts on weighted error
        let p_error = self.beta * sp - pv;
        let p_term = self.kp * p_error;

        // Derivative: act on measurement to avoid derivative kick,
        // or on weighted setpoint-measurement difference
        let d_input = self.gamma * sp - pv;
        let raw_derivative = match self.prev_measurement {
            Some(prev_d_input) => {
                // Use stored previous d_input for derivative
                // We store prev_measurement as the previous d_input value
                (d_input - prev_d_input) / dt
            }
            None => S::ZERO,
        };

        let d_term = self.kd
            * match self.d_filter.as_mut() {
                Some(filter) => filter.apply(raw_derivative, dt),
                None => raw_derivative,
            };

        // Compute unlimited output (before anti-windup integral update)
        let output_before_integral = p_term + d_term;

        // Update integral with anti-windup
        let output_unlimited_prev = output_before_integral + self.integral;
        let output_limited_prev = match &self.output_limiter {
            Some(limiter) => limiter.apply(output_unlimited_prev).0,
            None => output_unlimited_prev,
        };

        self.integral = self.anti_windup.correct_integral(
            self.integral,
            output_unlimited_prev,
            output_limited_prev,
            error,
            self.ki,
            dt,
        );

        // Final output
        let output_unlimited = p_term + self.integral + d_term;
        let (output, saturated) = match &self.output_limiter {
            Some(limiter) => limiter.apply(output_unlimited),
            None => (output_unlimited, false),
        };

        self.saturated = saturated;
        self.prev_error = Some(error);
        self.prev_measurement = Some(d_input);

        ControlOutput::with_saturation(output, saturated)
    }

    fn reset(&mut self) {
        self.integral = S::ZERO;
        self.prev_error = None;
        self.prev_measurement = None;
        self.saturated = false;
        if let Some(f) = self.d_filter.as_mut() {
            f.reset();
        }
    }

    fn is_saturated(&self) -> bool {
        self.saturated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_only_proportional() {
        let mut pid = PidConfig::p(2.0_f64).build();
        let sp = Setpoint::new(10.0);
        let fb = Feedback::new(7.0);
        let out = pid.update(&sp, &fb, 0.01);
        // P = 2.0 * (10-7) = 6.0
        assert!((out.value() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn pi_integrates() {
        let mut pid = PidConfig::pi(1.0_f64, 10.0).build();
        let sp = Setpoint::new(10.0);
        let fb = Feedback::new(0.0);

        // First step
        let out1 = pid.update(&sp, &fb, 0.01);
        // P=10, I=10*10*0.01=1.0
        assert!((out1.value() - 11.0).abs() < 1e-6, "got {}", out1.value());

        // Second step (same error)
        let out2 = pid.update(&sp, &fb, 0.01);
        // P=10, I=1.0+1.0=2.0
        assert!((out2.value() - 12.0).abs() < 1e-6, "got {}", out2.value());
    }

    #[test]
    fn pid_derivative_on_measurement() {
        let config = PidConfig {
            kp: 1.0_f64,
            ki: 0.0,
            kd: 0.1,
            beta: 1.0,
            gamma: 0.0, // derivative on measurement only
            output_limiter: None,
            anti_windup: AntiWindupMethod::None,
            derivative_filter_tau: None,
        };
        let mut pid = config.build();
        let sp = Setpoint::new(10.0);

        // First step: no derivative (no prev)
        pid.update(&sp, &Feedback::new(0.0), 0.01);

        // Second step: measurement changes
        let out = pid.update(&sp, &Feedback::new(1.0), 0.01);
        // P = 1*(10-1) = 9, D = 0.1 * (0 - (-1))/0.01 ... gamma=0 so d_input = -pv
        // prev d_input = -0 = 0, curr d_input = -1
        // raw_deriv = (-1 - 0)/0.01 = -100
        // D = 0.1 * -100 = -10
        // total = 9 + (-10) = -1
        assert!((out.value() - (-1.0)).abs() < 1e-6, "got {}", out.value());
    }

    #[test]
    fn output_limiting() {
        let mut pid = PidConfig::p(10.0_f64).with_limits(-5.0, 5.0).build();
        let sp = Setpoint::new(10.0);
        let fb = Feedback::new(0.0);
        let out = pid.update(&sp, &fb, 0.01);
        assert_eq!(out.value(), 5.0);
        assert!(out.is_saturated());
    }

    #[test]
    fn anti_windup_clamping_prevents_windup() {
        let mut pid = PidConfig::pi(1.0_f64, 100.0)
            .with_limits(-10.0, 10.0)
            .with_anti_windup(AntiWindupMethod::Clamping)
            .build();

        let sp = Setpoint::new(100.0);
        let fb = Feedback::new(0.0);

        // Run for many steps to try to wind up
        for _ in 0..1000 {
            pid.update(&sp, &fb, 0.01);
        }

        // Now setpoint goes below feedback - should recover quickly
        let sp_low = Setpoint::new(0.0);
        let fb_high = Feedback::new(5.0);

        // Within a few steps, output should go negative
        let mut went_negative = false;
        for _ in 0..20 {
            let out = pid.update(&sp_low, &fb_high, 0.01);
            if out.value() < 0.0 {
                went_negative = true;
                break;
            }
        }
        assert!(
            went_negative,
            "Controller should recover from saturation quickly with anti-windup"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut pid = PidConfig::pi(1.0_f64, 10.0).build();
        let sp = Setpoint::new(10.0);
        let fb = Feedback::new(0.0);

        for _ in 0..10 {
            pid.update(&sp, &fb, 0.01);
        }
        assert!(pid.integral().abs() > 0.0);

        pid.reset();
        assert_eq!(pid.integral(), 0.0);
        assert!(!pid.is_saturated());
    }

    #[test]
    fn dt_zero_returns_zero() {
        let mut pid = PidConfig::p(1.0_f64).build();
        let out = pid.update(&Setpoint::new(10.0), &Feedback::new(0.0), 0.0);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn two_dof_setpoint_weight() {
        let config = PidConfig {
            kp: 1.0_f64,
            ki: 0.0,
            kd: 0.0,
            beta: 0.5, // Reduced proportional action on setpoint
            gamma: 0.0,
            output_limiter: None,
            anti_windup: AntiWindupMethod::None,
            derivative_filter_tau: None,
        };
        let mut pid = config.build();
        let sp = Setpoint::new(10.0);
        let fb = Feedback::new(0.0);
        let out = pid.update(&sp, &fb, 0.01);
        // P = kp * (beta*sp - pv) = 1.0 * (0.5*10 - 0) = 5.0
        assert!((out.value() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn set_gains_updates_controller() {
        let mut pid = PidConfig::p(1.0_f64).build();
        pid.set_gains(2.0, 0.0, 0.0);
        assert_eq!(pid.kp(), 2.0);
    }

    #[test]
    fn step_response_first_order_system() {
        // Simulate PID controlling a first-order system: dy/dt = (u - y) / tau
        let tau = 1.0_f64;
        let dt = 0.001;
        let setpoint = 1.0;
        let mut y = 0.0_f64;

        let mut pid = PidConfig::pi(5.0_f64, 10.0)
            .with_limits(-100.0, 100.0)
            .build();
        let sp = Setpoint::new(setpoint);

        for _ in 0..10_000 {
            let fb = Feedback::new(y);
            let out = pid.update(&sp, &fb, dt);
            // Plant: first-order with gain=1
            let dy = (out.value() - y) / tau;
            y += dy * dt;
        }

        // After 10 seconds with a PI controller, should be very close to setpoint
        assert!(
            (y - setpoint).abs() < 0.01,
            "Should converge to setpoint: y={}, sp={}",
            y,
            setpoint
        );
    }
}
