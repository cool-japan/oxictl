use crate::core::scalar::ControlScalar;

/// Current Source Inverter (CSI) controller in dq synchronous reference frame.
///
/// Controls AC current injection using PI controllers in rotating dq frame.
/// Includes cross-coupling decoupling feedforward terms:
///   v_d = Kp*e_d + Ki∫e_d - ω*L*i_q  (d-axis)
///   v_q = Kp*e_q + Ki∫e_q + ω*L*i_d  (q-axis)
///
/// Suitable for grid-connected inverter current injection.
#[derive(Debug, Clone, Copy)]
pub struct CsiController<S: ControlScalar> {
    /// Proportional gain.
    pub kp: S,
    /// Integral gain.
    pub ki: S,
    /// Filter inductance (H) — used for cross-coupling decoupling.
    pub l_filter: S,
    /// Grid angular frequency (rad/s).
    pub omega: S,
    /// d-axis integrator state.
    int_d: S,
    /// q-axis integrator state.
    int_q: S,
    /// Anti-windup limit on integrator output.
    pub int_limit: S,
    /// Output voltage limit (V).
    pub v_limit: S,
}

impl<S: ControlScalar> CsiController<S> {
    pub fn new(kp: S, ki: S, l_filter: S, omega: S, v_limit: S) -> Self {
        Self {
            kp,
            ki,
            l_filter,
            omega,
            int_d: S::ZERO,
            int_q: S::ZERO,
            int_limit: v_limit,
            v_limit,
        }
    }

    /// Update CSI current controller.
    ///
    /// - `id_ref`, `iq_ref`: d-q axis current references (A)
    /// - `id`, `iq`: measured d-q currents (A)
    /// - `vd_ff`, `vq_ff`: grid voltage feedforward (V)
    /// - `dt`: time step (s)
    ///
    /// Returns `(vd_out, vq_out)`: voltage commands in dq frame (V).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        id_ref: S,
        iq_ref: S,
        id: S,
        iq: S,
        vd_ff: S,
        vq_ff: S,
        dt: S,
    ) -> (S, S) {
        let ed = id_ref - id;
        let eq = iq_ref - iq;

        // Integrate with anti-windup clamping
        self.int_d = (self.int_d + self.ki * ed * dt).clamp_val(-self.int_limit, self.int_limit);
        self.int_q = (self.int_q + self.ki * eq * dt).clamp_val(-self.int_limit, self.int_limit);

        // Cross-coupling decoupling feedforward
        let cross_d = -self.omega * self.l_filter * iq;
        let cross_q = self.omega * self.l_filter * id;

        let vd = self.kp * ed + self.int_d + cross_d + vd_ff;
        let vq = self.kp * eq + self.int_q + cross_q + vq_ff;

        // Vector amplitude limit
        let amp = (vd * vd + vq * vq).sqrt();
        if amp > self.v_limit {
            let scale = self.v_limit / amp;
            (vd * scale, vq * scale)
        } else {
            (vd, vq)
        }
    }

    pub fn reset(&mut self) {
        self.int_d = S::ZERO;
        self.int_q = S::ZERO;
    }
}

/// Single-phase CSI controller for H-bridge topology.
/// Uses PI control on instantaneous current error.
#[derive(Debug, Clone, Copy)]
pub struct SinglePhaseCsi<S: ControlScalar> {
    pub kp: S,
    pub ki: S,
    int: S,
    pub int_limit: S,
    pub out_limit: S,
}

impl<S: ControlScalar> SinglePhaseCsi<S> {
    pub fn new(kp: S, ki: S, out_limit: S) -> Self {
        Self {
            kp,
            ki,
            int: S::ZERO,
            int_limit: out_limit,
            out_limit,
        }
    }

    pub fn update(&mut self, i_ref: S, i_meas: S, dt: S) -> S {
        let err = i_ref - i_meas;
        self.int = (self.int + self.ki * err * dt).clamp_val(-self.int_limit, self.int_limit);
        let out = self.kp * err + self.int;
        out.clamp_val(-self.out_limit, self.out_limit)
    }

    pub fn reset(&mut self) {
        self.int = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn csi_tracks_d_current() {
        let omega = 2.0 * PI * 50.0;
        let dt = 1e-4;
        let mut csi = CsiController::new(10.0_f64, 200.0, 5e-3, omega, 400.0);

        let mut id = 0.0f64;
        let l = 5e-3f64;

        for _ in 0..2000 {
            let (vd, _vq) = csi.update(10.0, 0.0, id, 0.0, 0.0, 0.0, dt);
            // Euler: id' = (vd - R*id) / L  (R=0 idealized)
            id += (vd / l) * dt;
            id = id.clamp(-20.0, 20.0);
        }

        assert!((id - 10.0).abs() < 2.0, "id={id:.4}, expected ≈10A");
    }

    #[test]
    fn output_voltage_bounded() {
        let omega = 2.0 * PI * 50.0;
        let mut csi = CsiController::new(100.0_f64, 5000.0, 1e-3, omega, 400.0);

        let (vd, vq) = csi.update(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-4);
        let amp = (vd * vd + vq * vq).sqrt();
        assert!(amp <= 400.0 + 1e-6, "amp={amp:.2} exceeds limit");
    }

    #[test]
    fn single_phase_csi_tracks() {
        let mut csi = SinglePhaseCsi::new(5.0_f64, 100.0, 50.0);
        let mut i = 0.0f64;
        let l = 1e-3f64;
        let dt = 1e-4;

        for _ in 0..2000 {
            let v = csi.update(5.0, i, dt);
            i += (v / l) * dt;
            i = i.clamp(-20.0, 20.0);
        }

        assert!((i - 5.0).abs() < 1.0, "i={i:.4}, expected ≈5A");
    }

    #[test]
    fn csi_reset_clears_integrators() {
        let mut csi = CsiController::new(10.0_f64, 200.0, 5e-3, 314.16, 400.0);
        for _ in 0..100 {
            csi.update(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-4);
        }
        csi.reset();
        assert_eq!(csi.int_d, 0.0);
        assert_eq!(csi.int_q, 0.0);
    }
}
