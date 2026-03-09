use crate::core::scalar::ControlScalar;

/// Squirrel-cage induction motor model in stationary (αβ) reference frame.
///
/// 5th-order state-space model:
///   States: [i_α, i_β, λ_αr, λ_βr, ω_r]
///   Inputs: [v_α, v_β, T_load]
///
/// Equations:
///   d_iα/dt = (1/σLs) · [-Rs·iα + (Lm·Rr/Lr²)·λαr + (Lm/Lr)·ωr·λβr + vα]
///   d_iβ/dt = (1/σLs) · [-Rs·iβ + (Lm·Rr/Lr²)·λβr - (Lm/Lr)·ωr·λαr + vβ]
///   d_λαr/dt = Lm·(Rr/Lr)·iα - (Rr/Lr)·λαr - ωr·λβr
///   d_λβr/dt = Lm·(Rr/Lr)·iβ - (Rr/Lr)·λβr + ωr·λαr
///   d_ωr/dt  = (Pp/J) · (Lm/Lr) · (iβ·λαr - iα·λβr) - (B/J)·ωr - T_load/J
///
/// σ = 1 - Lm²/(Ls·Lr)  (leakage factor)
/// ωr is mechanical angular velocity (elec. = Pp·ωr_mech)
#[derive(Debug, Clone, Copy)]
pub struct InductionMotor<S: ControlScalar> {
    /// Stator resistance (Ω).
    pub rs: S,
    /// Rotor resistance (Ω).
    pub rr: S,
    /// Stator inductance Ls = Lls + Lm (H).
    pub ls: S,
    /// Rotor inductance Lr = Llr + Lm (H).
    pub lr: S,
    /// Mutual inductance Lm (H).
    pub lm: S,
    /// Number of pole pairs.
    pub pole_pairs: S,
    /// Rotor inertia (kg·m²).
    pub j: S,
    /// Viscous friction (N·m·s/rad).
    pub b_friction: S,

    // States
    /// α-axis stator current (A).
    i_alpha: S,
    /// β-axis stator current (A).
    i_beta: S,
    /// α-axis rotor flux linkage (Wb).
    lambda_alpha_r: S,
    /// β-axis rotor flux linkage (Wb).
    lambda_beta_r: S,
    /// Mechanical angular velocity (rad/s).
    omega_r: S,
    /// Mechanical rotor angle (rad).
    theta_r: S,
}

impl<S: ControlScalar> InductionMotor<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(rs: S, rr: S, ls: S, lr: S, lm: S, pole_pairs: S, j: S, b_friction: S) -> Self {
        Self {
            rs,
            rr,
            ls,
            lr,
            lm,
            pole_pairs,
            j,
            b_friction,
            i_alpha: S::ZERO,
            i_beta: S::ZERO,
            lambda_alpha_r: S::ZERO,
            lambda_beta_r: S::ZERO,
            omega_r: S::ZERO,
            theta_r: S::ZERO,
        }
    }

    /// Typical small induction motor (1kW, 4-pole).
    pub fn typical_1kw() -> Self {
        Self::new(
            S::from_f64(4.0),   // Rs = 4 Ω
            S::from_f64(2.5),   // Rr = 2.5 Ω
            S::from_f64(0.3),   // Ls = 0.3 H
            S::from_f64(0.3),   // Lr = 0.3 H
            S::from_f64(0.28),  // Lm = 0.28 H
            S::from_f64(2.0),   // 2 pole pairs (4-pole)
            S::from_f64(0.01),  // J = 0.01 kg·m²
            S::from_f64(0.001), // B = 0.001 N·m·s
        )
    }

    /// Euler integration step.
    ///
    /// - `v_alpha`, `v_beta`: stator voltages in αβ frame (V)
    /// - `t_load`: load torque (N·m)
    /// - `dt`: time step (s)
    pub fn step(&mut self, v_alpha: S, v_beta: S, t_load: S, dt: S) {
        let sigma = S::ONE - self.lm * self.lm / (self.ls * self.lr);
        let sigma_ls = sigma * self.ls;
        let rr_lr = self.rr / self.lr;
        let lm_lr = self.lm / self.lr;
        let lm_rr_lr2 = self.lm * self.rr / (self.lr * self.lr);
        let e_omega = self.pole_pairs * self.omega_r;

        // Stator current derivatives
        let d_ia = (sigma_ls).recip()
            * (-self.rs * self.i_alpha
                + lm_rr_lr2 * self.lambda_alpha_r
                + lm_lr * e_omega * self.lambda_beta_r
                + v_alpha);
        let d_ib = (sigma_ls).recip()
            * (-self.rs * self.i_beta + lm_rr_lr2 * self.lambda_beta_r
                - lm_lr * e_omega * self.lambda_alpha_r
                + v_beta);

        // Rotor flux derivatives
        let d_lar = self.lm * rr_lr * self.i_alpha
            - rr_lr * self.lambda_alpha_r
            - e_omega * self.lambda_beta_r;
        let d_lbr = self.lm * rr_lr * self.i_beta - rr_lr * self.lambda_beta_r
            + e_omega * self.lambda_alpha_r;

        // Electromagnetic torque
        let te = self.pole_pairs
            * lm_lr
            * (self.i_beta * self.lambda_alpha_r - self.i_alpha * self.lambda_beta_r);

        // Speed derivative
        let d_omega = (te - self.b_friction * self.omega_r - t_load) / self.j;

        // Euler update
        self.i_alpha += d_ia * dt;
        self.i_beta += d_ib * dt;
        self.lambda_alpha_r += d_lar * dt;
        self.lambda_beta_r += d_lbr * dt;
        self.omega_r += d_omega * dt;
        self.theta_r += self.omega_r * dt;
    }

    pub fn i_alpha(&self) -> S {
        self.i_alpha
    }
    pub fn i_beta(&self) -> S {
        self.i_beta
    }
    pub fn omega_r(&self) -> S {
        self.omega_r
    }
    pub fn theta_r(&self) -> S {
        self.theta_r
    }
    pub fn lambda_alpha_r(&self) -> S {
        self.lambda_alpha_r
    }
    pub fn lambda_beta_r(&self) -> S {
        self.lambda_beta_r
    }

    /// Rotor flux magnitude (Wb).
    pub fn flux_magnitude(&self) -> S {
        (self.lambda_alpha_r * self.lambda_alpha_r + self.lambda_beta_r * self.lambda_beta_r).sqrt()
    }

    /// Electromagnetic torque (N·m).
    pub fn torque(&self) -> S {
        let lm_lr = self.lm / self.lr;
        self.pole_pairs
            * lm_lr
            * (self.i_beta * self.lambda_alpha_r - self.i_alpha * self.lambda_beta_r)
    }

    pub fn reset(&mut self) {
        self.i_alpha = S::ZERO;
        self.i_beta = S::ZERO;
        self.lambda_alpha_r = S::ZERO;
        self.lambda_beta_r = S::ZERO;
        self.omega_r = S::ZERO;
        self.theta_r = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motor_starts_from_rest() {
        let mut motor = InductionMotor::<f64>::typical_1kw();
        // Apply rated voltage (αβ: v_alpha = Vpeak, v_beta = 0 initially)
        let dt = 1e-5_f64;
        for _ in 0..1000 {
            motor.step(100.0, 0.0, 0.0, dt);
        }
        // Currents should build up from zero
        assert!(motor.i_alpha().abs() > 0.1, "ia should grow");
    }

    #[test]
    fn torque_is_zero_at_rest_no_flux() {
        let motor = InductionMotor::<f64>::typical_1kw();
        assert_eq!(motor.torque(), 0.0);
    }

    #[test]
    fn flux_builds_up_with_voltage() {
        let mut motor = InductionMotor::<f64>::typical_1kw();
        let dt = 1e-4_f64;
        // Apply constant v_alpha to magnetize
        for _ in 0..5000 {
            motor.step(50.0, 0.0, 0.0, dt);
        }
        assert!(motor.flux_magnitude() > 0.01, "flux should build up");
    }

    #[test]
    fn motor_accelerates_under_torque_producing_voltage() {
        let mut motor = InductionMotor::<f64>::typical_1kw();
        let dt = 1e-4_f64;
        let omega_s = 2.0 * core::f64::consts::PI * 50.0;
        let v_peak = 100.0_f64;
        let mut t = 0.0_f64;

        for _ in 0..100_000 {
            let va = v_peak * (omega_s * t).cos();
            let vb = v_peak * (omega_s * t - core::f64::consts::PI / 2.0).cos();
            motor.step(va, vb, 0.0, dt);
            t += dt;
        }
        // Motor should start rotating with AC excitation
        assert!(
            motor.omega_r().abs() > 1.0,
            "motor should accelerate, ω={:.2}",
            motor.omega_r()
        );
    }
}
