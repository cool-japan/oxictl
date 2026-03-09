use crate::core::scalar::ControlScalar;

/// Synchronous Reference Frame Phase-Locked Loop (SRF-PLL).
///
/// Locks onto the phase of a two-phase (αβ) voltage signal.
///
/// Structure:
///   - Park transform: V_q = -v_α·sin(θ̂) + v_β·cos(θ̂)
///   - PI controller drives V_q → 0 (zero q-axis error ↔ phase lock)
///   - VCO: θ̂ += ω̂·dt
///
/// For single-phase use, generate βquadrature via a 90° delay (e.g. T/4 delay
/// or all-pass filter) before calling `update`.
#[derive(Debug, Clone, Copy)]
pub struct Pll<S: ControlScalar> {
    /// Proportional gain.
    pub kp: S,
    /// Integral gain.
    pub ki: S,
    /// Nominal angular frequency (rad/s), e.g. 2π×50 for 50 Hz grid.
    pub omega_nominal: S,
    /// Estimated electrical angle (rad), wrapped to [-π, π].
    theta_est: S,
    /// Estimated angular frequency (rad/s).
    omega_est: S,
    /// PI integrator state.
    integrator: S,
}

impl<S: ControlScalar> Pll<S> {
    /// Create a SRF-PLL.
    ///
    /// - `omega_nominal`: nominal frequency in rad/s (initializes VCO)
    /// - `kp`, `ki`: PI gains for the phase error loop
    pub fn new(omega_nominal: S, kp: S, ki: S) -> Self {
        Self {
            kp,
            ki,
            omega_nominal,
            theta_est: S::ZERO,
            omega_est: omega_nominal,
            integrator: S::ZERO,
        }
    }

    /// Update the PLL for one time step.
    ///
    /// - `v_alpha`, `v_beta`: stationary-frame (αβ) voltage components
    /// - `dt`: time step (s)
    ///
    /// Returns estimated phase θ̂ ∈ [-π, π].
    pub fn update(&mut self, v_alpha: S, v_beta: S, dt: S) -> S {
        // Park transform to estimated dq frame
        let cos_t = self.theta_est.cos();
        let sin_t = self.theta_est.sin();
        // V_q is the phase-error signal (should be 0 at lock)
        let v_q = -v_alpha * sin_t + v_beta * cos_t;

        // PI loop filter
        self.integrator += v_q * dt;
        let omega_correction = self.kp * v_q + self.ki * self.integrator;

        // VCO
        self.omega_est = self.omega_nominal + omega_correction;
        self.theta_est += self.omega_est * dt;

        // Wrap θ̂ to [-π, π]
        let pi = S::PI;
        let two_pi = S::TWO * pi;
        if self.theta_est > pi {
            self.theta_est -= two_pi;
        } else if self.theta_est < -pi {
            self.theta_est += two_pi;
        }

        self.theta_est
    }

    /// Current estimated phase (rad).
    pub fn theta(&self) -> S {
        self.theta_est
    }

    /// Current estimated frequency (rad/s).
    pub fn omega(&self) -> S {
        self.omega_est
    }

    /// Phase error signal (V_q before PI) — useful for diagnostics.
    pub fn phase_error(&self) -> S {
        self.integrator // reflects accumulated error
    }

    pub fn reset(&mut self) {
        self.theta_est = S::ZERO;
        self.omega_est = self.omega_nominal;
        self.integrator = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn locks_onto_50hz_signal() {
        let omega = 2.0 * PI * 50.0_f64; // 50 Hz
        let mut pll = Pll::new(omega, 200.0, 2000.0);
        let dt = 1e-4_f64;

        for step in 0..10000 {
            let t = step as f64 * dt;
            let v_alpha = (omega * t).cos();
            let v_beta = (omega * t).sin();
            pll.update(v_alpha, v_beta, dt);
        }

        // After 1 second, phase should be locked (error < 0.1 rad)
        let t_end = 10000.0 * dt;
        let theta_true = (omega * t_end) % (2.0 * PI);
        let theta_true_wrapped = if theta_true > PI {
            theta_true - 2.0 * PI
        } else {
            theta_true
        };
        let err = (pll.theta() - theta_true_wrapped).abs();
        // Allow wrap-around: check min of direct diff and 2π-wrapped
        let err_wrapped = err.min((err - 2.0 * PI).abs());
        assert!(
            err_wrapped < 0.1,
            "phase error={:.4} rad, theta_est={:.4}, theta_true={:.4}",
            err_wrapped,
            pll.theta(),
            theta_true_wrapped
        );
    }

    #[test]
    fn estimated_frequency_converges() {
        let omega = 2.0 * PI * 50.0_f64;
        let mut pll = Pll::new(omega, 200.0, 2000.0);
        let dt = 1e-4_f64;

        for step in 0..10000 {
            let t = step as f64 * dt;
            let v_alpha = (omega * t).cos();
            let v_beta = (omega * t).sin();
            pll.update(v_alpha, v_beta, dt);
        }

        let omega_err = (pll.omega() - omega).abs();
        assert!(omega_err < 10.0, "omega_err={:.2} rad/s", omega_err);
    }

    #[test]
    fn reset_clears_state() {
        let omega = 2.0 * PI * 50.0_f64;
        let mut pll = Pll::new(omega, 200.0, 2000.0);
        let dt = 1e-4_f64;

        for step in 0..1000 {
            let t = step as f64 * dt;
            pll.update((omega * t).cos(), (omega * t).sin(), dt);
        }

        pll.reset();
        assert_eq!(pll.theta(), 0.0);
        assert_eq!(pll.omega(), omega);
    }

    #[test]
    fn zero_input_stays_at_nominal() {
        let omega = 100.0_f64;
        let mut pll = Pll::new(omega, 10.0, 100.0);
        let dt = 0.001;

        // With zero input, V_q = 0, PI stays at 0, omega = nominal
        for _ in 0..100 {
            pll.update(0.0, 0.0, dt);
        }
        assert!((pll.omega() - omega).abs() < 1.0);
    }
}
