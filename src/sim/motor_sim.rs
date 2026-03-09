use crate::core::scalar::ControlScalar;
use crate::core::traits::Plant;

/// Simplified DC motor electrical + mechanical simulation.
///
/// State: [i (A), ω (rad/s)]
///   di/dt = (V - R*i - Ke*ω) / L
///   dω/dt = (Kt*i - B*ω - τ_load) / J
#[derive(Debug, Clone, Copy)]
pub struct DcMotorSim<S: ControlScalar> {
    /// Armature resistance (Ω).
    pub r: S,
    /// Armature inductance (H).
    pub l: S,
    /// Back-EMF constant (V·s/rad).
    pub k_e: S,
    /// Torque constant (N·m/A).
    pub k_t: S,
    /// Rotor inertia (kg·m²).
    pub j: S,
    /// Viscous friction (N·m·s/rad).
    pub b: S,
    /// Armature current (A).
    current: S,
    /// Angular velocity (rad/s).
    omega: S,
    /// Angular position (rad).
    theta: S,
    /// External load torque (N·m).
    tau_load: S,
}

impl<S: ControlScalar> DcMotorSim<S> {
    /// Create a DC motor simulation.
    pub fn new(r: S, l: S, k_e: S, k_t: S, j: S, b: S) -> Self {
        Self {
            r,
            l,
            k_e,
            k_t,
            j,
            b,
            current: S::ZERO,
            omega: S::ZERO,
            theta: S::ZERO,
            tau_load: S::ZERO,
        }
    }

    /// Small DC motor (100W class): R=1Ω, L=1mH, Ke=0.05, Kt=0.05, J=0.001, B=0.001
    pub fn small() -> Self {
        Self::new(
            S::from_f64(1.0),
            S::from_f64(0.001),
            S::from_f64(0.05),
            S::from_f64(0.05),
            S::from_f64(0.001),
            S::from_f64(0.001),
        )
    }

    /// Apply voltage input and advance simulation by `dt`.
    pub fn step_voltage(&mut self, v: S, dt: S) {
        let di_dt = (v - self.r * self.current - self.k_e * self.omega) / self.l;
        let domega_dt = (self.k_t * self.current - self.b * self.omega - self.tau_load) / self.j;
        self.current += di_dt * dt;
        self.omega += domega_dt * dt;
        self.theta += self.omega * dt;
    }

    pub fn set_load_torque(&mut self, tau: S) {
        self.tau_load = tau;
    }

    pub fn current(&self) -> S {
        self.current
    }

    pub fn omega(&self) -> S {
        self.omega
    }

    pub fn theta(&self) -> S {
        self.theta
    }

    pub fn reset(&mut self) {
        self.current = S::ZERO;
        self.omega = S::ZERO;
        self.theta = S::ZERO;
        self.tau_load = S::ZERO;
    }
}

impl<S: ControlScalar> Plant<S> for DcMotorSim<S> {
    fn step(&mut self, input: S, dt: S) {
        self.step_voltage(input, dt);
    }

    fn output(&self) -> S {
        self.omega // Output is speed
    }

    fn state(&self) -> &[S] {
        // Return omega as the primary state (current is internal)
        core::slice::from_ref(&self.omega)
    }
}

/// PMSM dq-frame simulation (reuses motor model from `motor/model/pmsm.rs` concept).
///
/// Minimal standalone version for simulation purposes.
/// Full model is in `motor::model::pmsm::PmsmModel`.
#[derive(Debug, Clone, Copy)]
pub struct PmsmSim<S: ControlScalar> {
    /// d-axis inductance (H).
    pub l_d: S,
    /// q-axis inductance (H).
    pub l_q: S,
    /// Stator resistance (Ω).
    pub r_s: S,
    /// Permanent magnet flux linkage (Wb).
    pub psi_f: S,
    /// Number of pole pairs.
    pub n_p: S,
    /// Rotor inertia (kg·m²).
    pub j: S,
    /// Viscous friction (N·m·s/rad).
    pub b: S,

    // State
    id: S,
    iq: S,
    omega_m: S, // mechanical speed (rad/s)
    theta_e: S, // electrical angle (rad)
    tau_load: S,
}

impl<S: ControlScalar> PmsmSim<S> {
    pub fn new(l_d: S, l_q: S, r_s: S, psi_f: S, n_p: S, j: S, b: S) -> Self {
        Self {
            l_d,
            l_q,
            r_s,
            psi_f,
            n_p,
            j,
            b,
            id: S::ZERO,
            iq: S::ZERO,
            omega_m: S::ZERO,
            theta_e: S::ZERO,
            tau_load: S::ZERO,
        }
    }

    /// Small PMSM (100W class).
    pub fn small() -> Self {
        Self::new(
            S::from_f64(0.001),  // Ld
            S::from_f64(0.002),  // Lq
            S::from_f64(0.5),    // Rs
            S::from_f64(0.05),   // ψf
            S::from_f64(2.0),    // pole pairs
            S::from_f64(0.001),  // J
            S::from_f64(0.0001), // B
        )
    }

    /// Advance dq-frame dynamics by dt using Euler integration.
    pub fn step_dq(&mut self, v_d: S, v_q: S, dt: S) {
        let omega_e = self.omega_m * self.n_p;

        let did_dt = (v_d - self.r_s * self.id + omega_e * self.l_q * self.iq) / self.l_d;
        let diq_dt =
            (v_q - self.r_s * self.iq - omega_e * (self.l_d * self.id + self.psi_f)) / self.l_q;

        let torque = S::from_f64(1.5)
            * self.n_p
            * (self.psi_f * self.iq + (self.l_d - self.l_q) * self.id * self.iq);

        let domega_dt = (torque - self.b * self.omega_m - self.tau_load) / self.j;

        self.id += did_dt * dt;
        self.iq += diq_dt * dt;
        self.omega_m += domega_dt * dt;
        self.theta_e += omega_e * dt;

        // Wrap electrical angle to [0, 2π)
        let two_pi = S::TWO * S::PI;
        while self.theta_e >= two_pi {
            self.theta_e -= two_pi;
        }
        while self.theta_e < S::ZERO {
            self.theta_e += two_pi;
        }
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
    pub fn theta_e(&self) -> S {
        self.theta_e
    }

    pub fn torque(&self) -> S {
        S::from_f64(1.5)
            * self.n_p
            * (self.psi_f * self.iq + (self.l_d - self.l_q) * self.id * self.iq)
    }

    pub fn set_load_torque(&mut self, tau: S) {
        self.tau_load = tau;
    }

    pub fn reset(&mut self) {
        self.id = S::ZERO;
        self.iq = S::ZERO;
        self.omega_m = S::ZERO;
        self.theta_e = S::ZERO;
        self.tau_load = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_motor_spins_up_under_voltage() {
        let mut motor = DcMotorSim::small();
        let dt = 0.0001;
        for _ in 0..10000 {
            motor.step_voltage(12.0, dt); // Apply 12V
        }
        // Motor should be spinning
        assert!(motor.omega() > 10.0, "ω={:.2}", motor.omega());
    }

    #[test]
    fn dc_motor_zero_voltage_decelerates() {
        let mut motor = DcMotorSim::small();
        let dt = 0.0001;
        // Spin up
        for _ in 0..10000 {
            motor.step_voltage(12.0, dt);
        }
        let omega_init = motor.omega();
        // Coast down
        for _ in 0..10000 {
            motor.step_voltage(0.0, dt);
        }
        assert!(motor.omega() < omega_init * 0.9, "Should decelerate");
    }

    #[test]
    fn pmsm_iq_drive_generates_torque() {
        let mut pmsm = PmsmSim::small();
        let dt = 0.0001;
        // Apply Vq for a short time (100ms = 1000 steps) before back-EMF reduces Iq.
        // At t=0 and low speed, Iq rises to Vq/Rs = 10A transiently.
        // Check Iq > 0.01A (positive torque direction) and omega is positive.
        for _ in 0..500 {
            pmsm.step_dq(0.0, 5.0, dt);
        }
        assert!(pmsm.iq() > 0.01, "Iq={:.3}", pmsm.iq());
        assert!(pmsm.omega_m() > 0.0, "ω={:.3}", pmsm.omega_m());
    }

    #[test]
    fn dc_motor_implements_plant_trait() {
        let mut motor = DcMotorSim::<f64>::small();
        motor.step(12.0, 0.001);
        assert!(motor.output() >= 0.0);
    }
}
