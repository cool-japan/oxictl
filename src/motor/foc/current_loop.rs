use crate::core::scalar::ControlScalar;
use crate::core::signal::{Feedback, Setpoint};
use crate::core::traits::Controller;
use crate::pid::standard::{Pid, PidConfig};

/// FOC current control loops for id and iq.
///
/// The d-axis current controls flux (set to 0 for id=0 MTPA strategy).
/// The q-axis current controls torque.
pub struct CurrentLoop<S: ControlScalar> {
    pub id_controller: Pid<S>,
    pub iq_controller: Pid<S>,
}

impl<S: ControlScalar> CurrentLoop<S> {
    pub fn new(kp: S, ki: S, v_limit: S) -> Self {
        let d_config = PidConfig::pi(kp, ki).with_limits(-v_limit, v_limit);
        let q_config = PidConfig::pi(kp, ki).with_limits(-v_limit, v_limit);
        Self {
            id_controller: d_config.build(),
            iq_controller: q_config.build(),
        }
    }

    /// Compute voltage commands Vd, Vq from current setpoints and measurements.
    pub fn update(&mut self, id_ref: S, iq_ref: S, id_meas: S, iq_meas: S, dt: S) -> (S, S) {
        let vd = self
            .id_controller
            .update(&Setpoint::new(id_ref), &Feedback::new(id_meas), dt)
            .value();
        let vq = self
            .iq_controller
            .update(&Setpoint::new(iq_ref), &Feedback::new(iq_meas), dt)
            .value();
        (vd, vq)
    }

    pub fn reset(&mut self) {
        self.id_controller.reset();
        self.iq_controller.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_error_zero_output() {
        let mut cl = CurrentLoop::<f64>::new(1.0, 0.0, 10.0);
        let (vd, vq) = cl.update(0.0, 5.0, 0.0, 5.0, 0.01);
        assert!((vd).abs() < 1e-10);
        assert!((vq).abs() < 1e-10);
    }

    #[test]
    fn nonzero_error_produces_output() {
        let mut cl = CurrentLoop::<f64>::new(2.0, 0.0, 100.0);
        let (vd, vq) = cl.update(0.0, 5.0, 0.0, 3.0, 0.01);
        assert!(vq > 0.0, "vq={}", vq); // iq error is 2A → positive Vq
        assert!(vd.abs() < 1e-10, "vd={}", vd); // id error is 0
    }

    #[test]
    fn reset_clears_integral() {
        let mut cl = CurrentLoop::<f64>::new(1.0, 10.0, 100.0);
        for _ in 0..100 {
            cl.update(0.0, 1.0, 0.0, 0.0, 0.01);
        }
        cl.reset();
        let (vd, vq) = cl.update(0.0, 0.0, 0.0, 0.0, 0.01);
        assert!(vd.abs() < 1e-10);
        assert!(vq.abs() < 1e-10);
    }
}
