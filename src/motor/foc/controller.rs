use crate::core::scalar::ControlScalar;
use crate::motor::foc::current_loop::CurrentLoop;
use crate::motor::foc::speed_loop::SpeedLoop;
use crate::motor::transform::clarke::clarke_2ph;
use crate::motor::transform::park::{park, park_inverse, Dq};
use crate::motor::transform::svpwm::{svpwm, SvpwmDuty};

/// FOC controller output.
#[derive(Debug, Clone, Copy)]
pub struct FocOutput<S: ControlScalar> {
    /// Three-phase PWM duty cycles.
    pub duty: SvpwmDuty<S>,
    /// d-axis voltage command.
    pub vd: S,
    /// q-axis voltage command.
    pub vq: S,
    /// Estimated electrical angle.
    pub theta: S,
}

/// Complete Field-Oriented Controller (FOC).
///
/// Structure:
///   speed_ref → [Speed Loop] → iq_ref
///   id_ref = 0 (MTPA strategy)
///   [id,iq] → [Current Loop] → [Vd,Vq]
///   [Vd,Vq] + θ → [Inverse Park] → [Vα,Vβ]
///   [Vα,Vβ] + Vdc → [SVPWM] → [Ta,Tb,Tc]
pub struct FocController<S: ControlScalar> {
    speed_loop: SpeedLoop<S>,
    current_loop: CurrentLoop<S>,
    /// DC bus voltage.
    vdc: S,
    /// Current rotor electrical angle estimate.
    theta: S,
    /// Angular velocity (electrical rad/s).
    omega: S,
}

impl<S: ControlScalar> FocController<S> {
    pub fn new(
        speed_kp: S,
        speed_ki: S,
        current_kp: S,
        current_ki: S,
        iq_limit: S,
        v_limit: S,
        vdc: S,
    ) -> Self {
        Self {
            speed_loop: SpeedLoop::new(speed_kp, speed_ki, iq_limit),
            current_loop: CurrentLoop::new(current_kp, current_ki, v_limit),
            vdc,
            theta: S::ZERO,
            omega: S::ZERO,
        }
    }

    /// Run one FOC control cycle.
    ///
    /// - `speed_ref`: speed setpoint (rad/s electrical)
    /// - `speed_meas`: measured speed (rad/s electrical)
    /// - `ia`, `ib`: phase a and b current measurements (c = -a-b for balanced)
    /// - `theta`: rotor electrical angle (radians)
    /// - `dt`: time step
    pub fn update(
        &mut self,
        speed_ref: S,
        speed_meas: S,
        ia: S,
        ib: S,
        theta: S,
        dt: S,
    ) -> FocOutput<S> {
        self.theta = theta;
        self.omega = speed_meas;

        // Step 1: Speed loop → iq reference (id = 0 for MTPA)
        let iq_ref = self.speed_loop.update(speed_ref, speed_meas, dt);
        let id_ref = S::ZERO;

        // Step 2: Clarke transform (2-measurement)
        let ab = clarke_2ph(ia, ib);

        // Step 3: Park transform → dq currents
        let dq = park(&ab, theta);

        // Step 4: Current loop → Vd, Vq
        let (vd, vq) = self.current_loop.update(id_ref, iq_ref, dq.d, dq.q, dt);

        // Step 5: Inverse Park → Vα, Vβ
        let v_dq = Dq { d: vd, q: vq };
        let v_ab = park_inverse(&v_dq, theta);

        // Step 6: SVPWM → duty cycles
        let duty = svpwm(&v_ab, self.vdc);

        FocOutput {
            duty,
            vd,
            vq,
            theta,
        }
    }

    pub fn reset(&mut self) {
        self.speed_loop.reset();
        self.current_loop.reset();
        self.theta = S::ZERO;
        self.omega = S::ZERO;
    }

    pub fn set_vdc(&mut self, vdc: S) {
        self.vdc = vdc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foc_zero_speed_ref_stable() {
        let mut foc = FocController::<f64>::new(0.1, 1.0, 5.0, 100.0, 10.0, 12.0, 24.0);
        // At zero speed ref with zero speed, should produce near-zero output
        let out = foc.update(0.0, 0.0, 0.0, 0.0, 0.0, 0.001);
        assert!((out.vd).abs() < 1e-10);
        assert!((out.vq).abs() < 1e-10);
        // Duty should be ~0.5
        assert!((out.duty.ta - 0.5).abs() < 0.01, "ta={}", out.duty.ta);
    }

    #[test]
    fn foc_produces_nonzero_output_for_speed_error() {
        let mut foc = FocController::<f64>::new(0.5, 2.0, 5.0, 100.0, 10.0, 12.0, 24.0);
        let out = foc.update(100.0, 0.0, 0.0, 0.0, 0.0, 0.001);
        // Speed error → positive iq_ref → current controller should produce Vq
        assert!(
            out.duty.ta != 0.5 || out.duty.tb != 0.5,
            "Should have non-trivial output"
        );
    }

    #[test]
    fn duty_cycles_in_range() {
        let mut foc = FocController::<f64>::new(0.1, 1.0, 5.0, 100.0, 10.0, 12.0, 24.0);
        for angle_deg in (0..360).step_by(30) {
            let theta = (angle_deg as f64) * core::f64::consts::PI / 180.0;
            let out = foc.update(50.0, 30.0, 1.0, -0.5, theta, 0.001);
            assert!(
                out.duty.ta >= 0.0 && out.duty.ta <= 1.0,
                "ta={}",
                out.duty.ta
            );
            assert!(
                out.duty.tb >= 0.0 && out.duty.tb <= 1.0,
                "tb={}",
                out.duty.tb
            );
            assert!(
                out.duty.tc >= 0.0 && out.duty.tc <= 1.0,
                "tc={}",
                out.duty.tc
            );
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut foc = FocController::<f64>::new(0.1, 1.0, 5.0, 100.0, 10.0, 12.0, 24.0);
        for _ in 0..100 {
            foc.update(100.0, 50.0, 1.0, -0.5, 0.3, 0.001);
        }
        foc.reset();
        let out = foc.update(0.0, 0.0, 0.0, 0.0, 0.0, 0.001);
        assert!(out.vd.abs() < 1e-10);
        assert!(out.vq.abs() < 1e-10);
    }
}
