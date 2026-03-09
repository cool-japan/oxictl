use crate::core::scalar::ControlScalar;
use crate::core::signal::{Feedback, Setpoint};
use crate::core::traits::Controller;
use crate::pid::anti_windup::AntiWindupMethod;
use crate::pid::standard::{Pid, PidConfig};

/// FOC speed control loop.
///
/// Produces an Iq reference (torque command) from speed error.
pub struct SpeedLoop<S: ControlScalar> {
    controller: Pid<S>,
}

impl<S: ControlScalar> SpeedLoop<S> {
    pub fn new(kp: S, ki: S, iq_limit: S) -> Self {
        let config = PidConfig::pi(kp, ki)
            .with_limits(-iq_limit, iq_limit)
            .with_anti_windup(AntiWindupMethod::Clamping);
        Self {
            controller: config.build(),
        }
    }

    /// Compute Iq reference from speed setpoint and measurement.
    pub fn update(&mut self, speed_ref: S, speed_meas: S, dt: S) -> S {
        let out = self
            .controller
            .update(&Setpoint::new(speed_ref), &Feedback::new(speed_meas), dt);
        out.value()
    }

    pub fn reset(&mut self) {
        self.controller.reset();
    }

    pub fn is_saturated(&self) -> bool {
        self.controller.is_saturated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_error_produces_iq() {
        let mut sl = SpeedLoop::<f64>::new(0.1, 1.0, 10.0);
        let iq = sl.update(100.0, 80.0, 0.001);
        assert!(
            iq > 0.0,
            "Positive speed error should give positive Iq: {}",
            iq
        );
    }

    #[test]
    fn zero_error_zero_iq() {
        let mut sl = SpeedLoop::<f64>::new(0.1, 0.0, 10.0);
        let iq = sl.update(50.0, 50.0, 0.001);
        assert!(iq.abs() < 1e-10);
    }

    #[test]
    fn iq_limited() {
        let mut sl = SpeedLoop::<f64>::new(100.0, 0.0, 5.0);
        let iq = sl.update(1000.0, 0.0, 0.001);
        assert!(iq <= 5.0, "Iq should be limited: {}", iq);
    }
}
