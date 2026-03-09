use crate::core::scalar::ControlScalar;

/// Permanent Magnet Synchronous Motor (PMSM) model in dq-frame.
///
/// Dynamics (Euler integration):
///   did/dt = (vd - Rs*id + ωe*Lq*iq) / Ld
///   diq/dt = (vq - Rs*iq - ωe*(Ld*id + ψf)) / Lq
///   Te = 1.5 * p * ψf * iq  (non-salient simplification)
///   dωm/dt = (Te - B*ωm - τ_load) / J
///   dθe/dt = p * ωm
#[derive(Debug, Clone, Copy)]
pub struct PmsmModel<S: ControlScalar> {
    /// Stator resistance (Ω).
    pub rs: S,
    /// d-axis inductance (H).
    pub ld: S,
    /// q-axis inductance (H).
    pub lq: S,
    /// Permanent magnet flux linkage (Wb).
    pub psi_f: S,
    /// Number of pole pairs.
    pub pole_pairs: u32,
    /// Moment of inertia (kg·m²).
    pub j: S,
    /// Viscous friction coefficient (N·m·s/rad).
    pub b_friction: S,
    // State
    /// d-axis current (A).
    id: S,
    /// q-axis current (A).
    iq: S,
    /// Mechanical angular velocity (rad/s).
    omega_m: S,
    /// Electrical angle (rad, unbounded accumulation).
    theta_e: S,
}

impl<S: ControlScalar> PmsmModel<S> {
    pub fn new(rs: S, ld: S, lq: S, psi_f: S, pole_pairs: u32, j: S, b_friction: S) -> Self {
        Self {
            rs,
            ld,
            lq,
            psi_f,
            pole_pairs,
            j,
            b_friction,
            id: S::ZERO,
            iq: S::ZERO,
            omega_m: S::ZERO,
            theta_e: S::ZERO,
        }
    }

    /// Typical small PMSM (~100W class) parameters.
    pub fn small_pmsm() -> Self
    where
        S: From<f32>,
    {
        Self::new(
            S::from_f64(0.5),
            S::from_f64(3e-4),
            S::from_f64(3e-4),
            S::from_f64(0.008),
            4,
            S::from_f64(1.5e-5),
            S::from_f64(1e-4),
        )
    }

    /// Advance one timestep with dq-axis voltage inputs.
    ///
    /// - `vd`, `vq`: d and q-axis voltage commands (V)
    /// - `tau_load`: external load torque (N·m)
    /// - `dt`: time step (s)
    pub fn step(&mut self, vd: S, vq: S, tau_load: S, dt: S) {
        let p = S::from_f64(self.pole_pairs as f64);
        let omega_e = self.omega_m * p;

        // Electrical dynamics
        let did = (vd - self.rs * self.id + omega_e * self.lq * self.iq) / self.ld;
        let diq = (vq - self.rs * self.iq - omega_e * (self.ld * self.id + self.psi_f)) / self.lq;

        // Electromagnetic torque (non-salient: Ld=Lq simplification for reluctance term)
        let te = S::from_f64(1.5) * p * self.psi_f * self.iq;

        // Mechanical dynamics
        let domega_m = (te - self.b_friction * self.omega_m - tau_load) / self.j;

        // Euler integration
        self.id += did * dt;
        self.iq += diq * dt;
        self.omega_m += domega_m * dt;
        self.theta_e += omega_e * dt;
    }

    pub fn id(&self) -> S {
        self.id
    }
    pub fn iq(&self) -> S {
        self.iq
    }
    pub fn omega_m(&self) -> S {
        self.omega_m
    }
    pub fn omega_e(&self) -> S {
        self.omega_m * S::from_f64(self.pole_pairs as f64)
    }
    /// Electrical angle (accumulated, not wrapped).
    pub fn theta_e_raw(&self) -> S {
        self.theta_e
    }
    /// Electromagnetic torque estimate.
    pub fn torque_estimate(&self) -> S {
        let p = S::from_f64(self.pole_pairs as f64);
        S::from_f64(1.5) * p * self.psi_f * self.iq
    }

    pub fn reset(&mut self) {
        self.id = S::ZERO;
        self.iq = S::ZERO;
        self.omega_m = S::ZERO;
        self.theta_e = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rest_state_at_zero_input() {
        let mut motor = PmsmModel::<f64>::new(0.5, 3e-4, 3e-4, 0.008, 4, 1.5e-5, 1e-4);
        for _ in 0..1000 {
            motor.step(0.0, 0.0, 0.0, 1e-4);
        }
        assert!(motor.id().abs() < 1e-10);
        assert!(motor.iq().abs() < 1e-10);
        assert!(motor.omega_m().abs() < 1e-10);
    }

    #[test]
    fn vq_drives_rotation() {
        let mut motor = PmsmModel::<f64>::new(0.5, 3e-4, 3e-4, 0.008, 4, 1.5e-5, 1e-4);
        for _ in 0..10_000 {
            motor.step(0.0, 5.0, 0.0, 1e-4);
        }
        assert!(motor.omega_m() > 0.0, "Motor should accelerate with vq>0");
    }

    #[test]
    fn torque_proportional_to_iq() {
        let motor = PmsmModel::<f64> {
            rs: 0.5,
            ld: 3e-4,
            lq: 3e-4,
            psi_f: 0.008,
            pole_pairs: 4,
            j: 1.5e-5,
            b_friction: 1e-4,
            id: 0.0,
            iq: 5.0,
            omega_m: 0.0,
            theta_e: 0.0,
        };
        let te = motor.torque_estimate();
        let expected = 1.5 * 4.0 * 0.008 * 5.0;
        assert!((te - expected).abs() < 1e-10);
    }

    #[test]
    fn reset_returns_to_zero() {
        let mut motor = PmsmModel::<f64>::new(0.5, 3e-4, 3e-4, 0.008, 4, 1.5e-5, 1e-4);
        for _ in 0..100 {
            motor.step(5.0, 10.0, 0.5, 1e-4);
        }
        motor.reset();
        assert_eq!(motor.id(), 0.0);
        assert_eq!(motor.omega_m(), 0.0);
    }
}
