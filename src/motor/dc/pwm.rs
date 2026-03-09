use crate::core::scalar::ControlScalar;

/// Brushed DC motor model with armature circuit dynamics.
///
/// Dynamics (Euler integration):
///   di/dt = (V_applied - R*i - Ke*ω) / L
///   dω/dt = (Kt*i - B*ω - τ_load) / J
///   dθ/dt = ω
#[derive(Debug, Clone, Copy)]
pub struct DcMotor<S: ControlScalar> {
    /// Armature resistance (Ω).
    pub r: S,
    /// Armature inductance (H).
    pub l: S,
    /// Back-EMF constant (V·s/rad).
    pub ke: S,
    /// Torque constant (N·m/A).
    pub kt: S,
    /// Moment of inertia (kg·m²).
    pub j: S,
    /// Viscous friction (N·m·s/rad).
    pub b_friction: S,
    // State
    current: S,
    omega: S,
    theta: S,
}

impl<S: ControlScalar> DcMotor<S> {
    pub fn new(r: S, l: S, ke: S, kt: S, j: S, b_friction: S) -> Self {
        Self {
            r,
            l,
            ke,
            kt,
            j,
            b_friction,
            current: S::ZERO,
            omega: S::ZERO,
            theta: S::ZERO,
        }
    }

    /// Advance one timestep.
    ///
    /// - `v_applied`: armature voltage (V)
    /// - `tau_load`: external load torque (N·m)
    /// - `dt`: time step (s)
    pub fn step(&mut self, v_applied: S, tau_load: S, dt: S) {
        let di = (v_applied - self.r * self.current - self.ke * self.omega) / self.l;
        let domega = (self.kt * self.current - self.b_friction * self.omega - tau_load) / self.j;

        self.current += di * dt;
        self.omega += domega * dt;
        self.theta += self.omega * dt;
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
    pub fn torque(&self) -> S {
        self.kt * self.current
    }

    /// Steady-state speed for given voltage and load torque.
    /// ω_ss = (kt*Kt*V - R*τ_load) / (B*R + Kt*Ke)  [no inductance]
    pub fn steady_state_speed(&self, v_applied: S, tau_load: S) -> S {
        let num = self.kt * v_applied - self.r * tau_load;
        let den = self.b_friction * self.r + self.kt * self.ke;
        if den <= S::ZERO {
            S::ZERO
        } else {
            num / den
        }
    }

    pub fn reset(&mut self) {
        self.current = S::ZERO;
        self.omega = S::ZERO;
        self.theta = S::ZERO;
    }
}

/// PWM voltage modulator for DC motor drive.
///
/// Converts duty cycle [-1, 1] to effective armature voltage.
#[derive(Debug, Clone, Copy)]
pub struct PwmDrive<S: ControlScalar> {
    /// Supply voltage (V).
    pub vdc: S,
    /// Dead-time fraction (0..1, voltage loss due to switch dead-time).
    pub dead_time_fraction: S,
}

impl<S: ControlScalar> PwmDrive<S> {
    pub fn new(vdc: S) -> Self {
        Self {
            vdc,
            dead_time_fraction: S::ZERO,
        }
    }

    pub fn with_dead_time(mut self, fraction: S) -> Self {
        self.dead_time_fraction = fraction;
        self
    }

    /// Convert duty cycle to armature voltage.
    ///
    /// `duty` in [-1.0, 1.0] where ±1 = full forward/reverse.
    pub fn voltage(&self, duty: S) -> S {
        let clamped = duty.clamp_val(-S::ONE, S::ONE);
        let effective = clamped.abs() - self.dead_time_fraction;
        if effective <= S::ZERO {
            return S::ZERO;
        }
        clamped * effective.abs() / clamped.abs() * self.vdc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_motor() -> DcMotor<f64> {
        DcMotor::new(1.0, 0.01, 0.05, 0.05, 1e-4, 1e-3)
    }

    #[test]
    fn rest_at_zero_voltage() {
        let mut m = build_motor();
        for _ in 0..1000 {
            m.step(0.0, 0.0, 1e-4);
        }
        assert!(m.current().abs() < 1e-10);
        assert!(m.omega().abs() < 1e-10);
    }

    #[test]
    fn accelerates_with_voltage() {
        let mut m = build_motor();
        for _ in 0..10_000 {
            m.step(12.0, 0.0, 1e-4);
        }
        assert!(m.omega() > 0.0, "Motor should accelerate");
    }

    #[test]
    fn reset_clears_state() {
        let mut m = build_motor();
        for _ in 0..100 {
            m.step(12.0, 0.0, 1e-4);
        }
        m.reset();
        assert_eq!(m.current(), 0.0);
        assert_eq!(m.omega(), 0.0);
    }

    #[test]
    fn pwm_drive_full_duty() {
        let pwm = PwmDrive::<f64>::new(24.0);
        assert!((pwm.voltage(1.0) - 24.0).abs() < 1e-10);
        assert!((pwm.voltage(-1.0) + 24.0).abs() < 1e-10);
        assert_eq!(pwm.voltage(0.0), 0.0);
    }

    #[test]
    fn pwm_dead_time_reduces_voltage() {
        let pwm = PwmDrive::<f64>::new(24.0).with_dead_time(0.1);
        let v = pwm.voltage(1.0);
        assert!(v < 24.0 && v > 0.0);
    }
}
