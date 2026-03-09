use crate::core::scalar::ControlScalar;

/// DC motor plant simulation with RK4 integration.
///
/// Armature circuit:
///   L·di/dt = V - R·i - Ke·ω
///
/// Mechanical dynamics:
///   J·dω/dt = Kt·i - B·ω - TL
///
/// Position:
///   dθ/dt = ω
///
/// State: [i (A), ω (rad/s), θ (rad)]
///
/// Parameters:
///   R:  armature resistance (Ω)
///   L:  armature inductance (H)
///   Ke: back-EMF constant (V·s/rad)
///   Kt: torque constant (N·m/A)
///   J:  rotor inertia (kg·m²)
///   B:  viscous friction (N·m·s/rad)
///
/// For an ideal motor: Ke = Kt (SI units).
#[derive(Debug, Clone, Copy)]
pub struct DcMotorPlant<S: ControlScalar> {
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
    /// Viscous friction coefficient (N·m·s/rad).
    pub b_friction: S,
    /// External load torque (N·m). Positive opposes rotation.
    pub load_torque: S,
    /// State: [current (A), angular_velocity (rad/s), position (rad)].
    state: [S; 3],
}

impl<S: ControlScalar> DcMotorPlant<S> {
    /// Create a DC motor plant.
    pub fn new(r: S, l: S, k_e: S, k_t: S, j: S, b_friction: S) -> Self {
        Self {
            r,
            l,
            k_e,
            k_t,
            j,
            b_friction,
            load_torque: S::ZERO,
            state: [S::ZERO; 3],
        }
    }

    /// Small DC motor suitable for robotics (100W class).
    ///
    /// Parameters: R=1Ω, L=5mH, Ke=0.1 V·s/rad, Kt=0.1 N·m/A,
    ///             J=2e-4 kg·m², B=1e-3 N·m·s/rad.
    pub fn small_robot_motor() -> Self {
        Self::new(
            S::from_f64(1.0),
            S::from_f64(5e-3),
            S::from_f64(0.1),
            S::from_f64(0.1),
            S::from_f64(2e-4),
            S::from_f64(1e-3),
        )
    }

    /// Medium servo motor.
    ///
    /// Parameters: R=0.5Ω, L=2mH, Ke=0.05, Kt=0.05,
    ///             J=1e-3 kg·m², B=5e-4.
    pub fn servo_motor() -> Self {
        Self::new(
            S::from_f64(0.5),
            S::from_f64(2e-3),
            S::from_f64(0.05),
            S::from_f64(0.05),
            S::from_f64(1e-3),
            S::from_f64(5e-4),
        )
    }

    /// Set the external load torque.
    pub fn set_load_torque(&mut self, tau_l: S) {
        self.load_torque = tau_l;
    }

    /// Set initial state.
    pub fn set_state(&mut self, current: S, omega: S, theta: S) {
        self.state = [current, omega, theta];
    }

    /// Armature current (A).
    pub fn current(&self) -> S {
        self.state[0]
    }

    /// Angular velocity (rad/s).
    pub fn omega(&self) -> S {
        self.state[1]
    }

    /// Angular position (rad).
    pub fn theta(&self) -> S {
        self.state[2]
    }

    /// Back-EMF voltage (V).
    pub fn back_emf(&self) -> S {
        self.k_e * self.state[1]
    }

    /// Electromagnetic torque (N·m).
    pub fn torque_em(&self) -> S {
        self.k_t * self.state[0]
    }

    /// Compute state derivatives for given voltage input.
    fn derivatives(&self, s: &[S; 3], voltage: S) -> [S; 3] {
        let i = s[0];
        let omega = s[1];

        let di_dt = if self.l.abs() > S::EPSILON {
            (voltage - self.r * i - self.k_e * omega) / self.l
        } else {
            S::ZERO
        };

        let domega_dt = if self.j.abs() > S::EPSILON {
            (self.k_t * i - self.b_friction * omega - self.load_torque) / self.j
        } else {
            S::ZERO
        };

        let dtheta_dt = omega;

        [di_dt, domega_dt, dtheta_dt]
    }

    /// Advance simulation using RK4 integration.
    ///
    /// `voltage`: applied armature voltage (V).
    /// `dt`: integration step (s).
    pub fn step(&mut self, voltage: S, dt: S) {
        let s = self.state;

        let k1 = self.derivatives(&s, voltage);
        let s2: [S; 3] = core::array::from_fn(|i| s[i] + S::HALF * dt * k1[i]);
        let k2 = self.derivatives(&s2, voltage);
        let s3: [S; 3] = core::array::from_fn(|i| s[i] + S::HALF * dt * k2[i]);
        let k3 = self.derivatives(&s3, voltage);
        let s4: [S; 3] = core::array::from_fn(|i| s[i] + dt * k3[i]);
        let k4 = self.derivatives(&s4, voltage);

        let sixth = S::ONE / S::from_f64(6.0);
        for i in 0..3 {
            self.state[i] += sixth * dt * (k1[i] + S::TWO * k2[i] + S::TWO * k3[i] + k4[i]);
        }
    }

    /// Steady-state angular velocity for given voltage and load torque.
    ///
    /// At steady state di/dt = 0, dω/dt = 0:
    ///   i_ss = (V - Ke*ω_ss) / R
    ///   Kt * i_ss = B*ω_ss + TL
    ///   ω_ss = (Kt*V/R - TL) / (B + Kt*Ke/R)
    pub fn steady_state_omega(&self, voltage: S) -> Option<S> {
        let denom = self.b_friction + self.k_t * self.k_e / self.r;
        if denom.abs() < S::EPSILON || self.r.abs() < S::EPSILON {
            return None;
        }
        Some((self.k_t * voltage / self.r - self.load_torque) / denom)
    }

    /// No-load speed (rad/s) for given voltage.
    pub fn no_load_speed(&self, voltage: S) -> Option<S> {
        let old_load = self.load_torque;
        // Temporarily compute with zero load
        let denom = self.b_friction + self.k_t * self.k_e / self.r;
        if denom.abs() < S::EPSILON || self.r.abs() < S::EPSILON {
            return None;
        }
        let omega_nl = self.k_t * voltage / (self.r * denom);
        let _ = old_load;
        Some(omega_nl)
    }

    /// Electrical time constant τ_e = L/R.
    pub fn electrical_time_constant(&self) -> Option<S> {
        if self.r.abs() < S::EPSILON {
            None
        } else {
            Some(self.l / self.r)
        }
    }

    /// Mechanical time constant τ_m = J*R/(Kt*Ke).
    pub fn mechanical_time_constant(&self) -> Option<S> {
        let denom = self.k_t * self.k_e;
        if denom.abs() < S::EPSILON || self.r.abs() < S::EPSILON {
            None
        } else {
            Some(self.j * self.r / denom)
        }
    }

    /// Reset state to zero.
    pub fn reset(&mut self) {
        self.state = [S::ZERO; 3];
    }

    /// Input power P_in = V*i.
    pub fn input_power(&self, voltage: S) -> S {
        voltage * self.state[0]
    }

    /// Output mechanical power P_out = τ_em * ω.
    pub fn output_power(&self) -> S {
        self.torque_em() * self.state[1]
    }

    /// Efficiency η = P_out / P_in (returns 0 if P_in ≈ 0).
    pub fn efficiency(&self, voltage: S) -> S {
        let p_in = self.input_power(voltage);
        let p_out = self.output_power();
        if p_in.abs() < S::EPSILON {
            S::ZERO
        } else {
            (p_out / p_in).clamp_val(S::ZERO, S::ONE)
        }
    }

    /// Speed in RPM (revolutions per minute).
    pub fn omega_rpm(&self) -> S {
        self.state[1] * S::from_f64(60.0) / (S::TWO * S::PI)
    }

    /// Stall torque at given voltage: τ_stall = Kt * V / R.
    pub fn stall_torque(&self, voltage: S) -> Option<S> {
        if self.r.abs() < S::EPSILON {
            None
        } else {
            Some(self.k_t * voltage / self.r)
        }
    }

    /// Stall current at given voltage: i_stall = V / R.
    pub fn stall_current(&self, voltage: S) -> Option<S> {
        if self.r.abs() < S::EPSILON {
            None
        } else {
            Some(voltage / self.r)
        }
    }

    /// Peak power point: ω_peak = ω_nl / 2, τ_peak = τ_stall / 2.
    ///
    /// Returns (omega_peak, torque_peak) or None if parameters are degenerate.
    pub fn peak_power_point(&self, voltage: S) -> Option<(S, S)> {
        let omega_nl = self.no_load_speed(voltage)?;
        let tau_stall = self.stall_torque(voltage)?;
        Some((omega_nl * S::HALF, tau_stall * S::HALF))
    }

    /// Maximum output power P_max = τ_stall * ω_nl / 4.
    pub fn max_output_power(&self, voltage: S) -> Option<S> {
        let omega_nl = self.no_load_speed(voltage)?;
        let tau_stall = self.stall_torque(voltage)?;
        Some(tau_stall * omega_nl / S::from_f64(4.0))
    }

    /// Copper losses P_cu = R * i².
    pub fn copper_losses(&self) -> S {
        self.r * self.state[0] * self.state[0]
    }

    /// Friction losses P_friction = B * ω².
    pub fn friction_losses(&self) -> S {
        self.b_friction * self.state[1] * self.state[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motor_spins_up_with_voltage() {
        let mut motor = DcMotorPlant::<f64>::small_robot_motor();
        let dt = 1e-5_f64;
        for _ in 0..50000 {
            motor.step(12.0, dt); // 12V
        }
        // Motor should spin after 0.5 seconds
        assert!(
            motor.omega() > 1.0,
            "omega should be positive: {}",
            motor.omega()
        );
    }

    #[test]
    fn motor_reaches_steady_state() {
        let mut motor = DcMotorPlant::<f64>::small_robot_motor();
        let v = 12.0_f64;
        let dt = 1e-5_f64;
        // Run for long enough to reach steady state
        for _ in 0..500000 {
            motor.step(v, dt);
        }
        let omega_ss = motor.steady_state_omega(v).unwrap();
        assert!(
            (motor.omega() - omega_ss).abs() / omega_ss < 0.02,
            "omega={:.2}, ss={:.2}",
            motor.omega(),
            omega_ss
        );
    }

    #[test]
    fn motor_no_load_speed_positive() {
        let motor = DcMotorPlant::<f64>::small_robot_motor();
        let nl = motor.no_load_speed(12.0).unwrap();
        assert!(nl > 0.0, "no-load speed should be positive: {}", nl);
    }

    #[test]
    fn motor_time_constants_positive() {
        let motor = DcMotorPlant::<f64>::small_robot_motor();
        let tau_e = motor.electrical_time_constant().unwrap();
        let tau_m = motor.mechanical_time_constant().unwrap();
        assert!(tau_e > 0.0, "tau_e={}", tau_e);
        assert!(tau_m > 0.0, "tau_m={}", tau_m);
    }

    #[test]
    fn motor_zero_voltage_stays_zero() {
        let mut motor = DcMotorPlant::<f64>::small_robot_motor();
        for _ in 0..1000 {
            motor.step(0.0, 1e-4);
        }
        assert!(motor.omega().abs() < 1e-10, "omega={}", motor.omega());
        assert!(motor.current().abs() < 1e-10, "i={}", motor.current());
    }

    #[test]
    fn motor_back_emf_proportional_to_speed() {
        let mut motor = DcMotorPlant::<f64>::small_robot_motor();
        for _ in 0..100000 {
            motor.step(12.0, 1e-5);
        }
        let expected_bemf = motor.k_e * motor.omega();
        assert!(
            (motor.back_emf() - expected_bemf).abs() < 1e-10,
            "back_emf={}, expected={}",
            motor.back_emf(),
            expected_bemf
        );
    }

    #[test]
    fn motor_load_torque_reduces_speed() {
        let mut m1 = DcMotorPlant::<f64>::small_robot_motor();
        let mut m2 = DcMotorPlant::<f64>::small_robot_motor();
        m2.set_load_torque(0.005);

        let dt = 1e-5_f64;
        for _ in 0..500000 {
            m1.step(12.0, dt);
            m2.step(12.0, dt);
        }
        assert!(
            m2.omega() < m1.omega(),
            "loaded motor ({:.2}) should be slower than unloaded ({:.2})",
            m2.omega(),
            m1.omega()
        );
    }

    #[test]
    fn motor_reset_zeros_state() {
        let mut motor = DcMotorPlant::<f64>::small_robot_motor();
        for _ in 0..100 {
            motor.step(12.0, 1e-4);
        }
        motor.reset();
        assert_eq!(motor.current(), 0.0);
        assert_eq!(motor.omega(), 0.0);
        assert_eq!(motor.theta(), 0.0);
    }

    #[test]
    fn servo_motor_creates_without_panic() {
        let _m = DcMotorPlant::<f64>::servo_motor();
    }

    #[test]
    fn motor_stall_torque_positive() {
        let motor = DcMotorPlant::<f64>::small_robot_motor();
        let tau_s = motor.stall_torque(12.0).unwrap();
        assert!(tau_s > 0.0, "stall_torque={}", tau_s);
    }

    #[test]
    fn motor_stall_current_equals_v_over_r() {
        let motor = DcMotorPlant::<f64>::small_robot_motor();
        let v = 12.0_f64;
        let i_stall = motor.stall_current(v).unwrap();
        assert!((i_stall - v / motor.r).abs() < 1e-10, "i_stall={}", i_stall);
    }

    #[test]
    fn motor_peak_power_point() {
        let motor = DcMotorPlant::<f64>::small_robot_motor();
        let (omega_peak, tau_peak) = motor.peak_power_point(12.0).unwrap();
        let omega_nl = motor.no_load_speed(12.0).unwrap();
        let tau_stall = motor.stall_torque(12.0).unwrap();
        assert!((omega_peak - omega_nl / 2.0).abs() < 1e-9);
        assert!((tau_peak - tau_stall / 2.0).abs() < 1e-9);
    }

    #[test]
    fn motor_max_output_power_positive() {
        let motor = DcMotorPlant::<f64>::small_robot_motor();
        let p_max = motor.max_output_power(12.0).unwrap();
        assert!(p_max > 0.0, "max_power={}", p_max);
    }

    #[test]
    fn motor_rpm_zero_at_start() {
        let motor = DcMotorPlant::<f64>::small_robot_motor();
        assert!((motor.omega_rpm()).abs() < 1e-10);
    }

    #[test]
    fn motor_copper_losses_zero_at_rest() {
        let motor = DcMotorPlant::<f64>::small_robot_motor();
        assert!((motor.copper_losses()).abs() < 1e-10);
    }

    #[test]
    fn motor_energy_balance() {
        // After reaching steady state:
        //   P_in = V*i = R*i² + Kt*i*ω = P_copper + P_electromagnetic
        //
        // P_out (= Kt*i*ω) is the total electromagnetic power transferred to the
        // mechanical side; it covers both friction losses and any useful shaft output.
        // P_friction = B*ω² is already contained within P_out, so the correct
        // electrical energy balance is:
        //
        //   P_in ≈ P_copper + P_out
        //
        // (At no external load all of P_out goes to friction, P_out ≈ P_friction,
        //  but they must not be added separately.)
        let mut motor = DcMotorPlant::<f64>::small_robot_motor();
        let v = 12.0_f64;
        let dt = 1e-5_f64;
        for _ in 0..500000 {
            motor.step(v, dt);
        }
        let p_in = motor.input_power(v);
        let p_out = motor.output_power();
        let p_cu = motor.copper_losses();
        let balance = (p_in - p_out - p_cu).abs();
        // At steady state, balance should be small relative to p_in
        assert!(
            balance / p_in.abs().max(1e-6) < 0.05,
            "energy balance error: {:.4}",
            balance
        );
    }
}
