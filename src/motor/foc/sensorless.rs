use crate::core::scalar::ControlScalar;

/// Sensorless FOC back-EMF observer for PMSM drives.
///
/// Estimates rotor flux (and thus position/speed) from measured
/// stator currents and applied voltages, without a physical position sensor.
///
/// Back-EMF model (stationary αβ frame):
///   eα = -ψf * ωe * sin(θe) = dψα/dt
///   eβ =  ψf * ωe * cos(θe) = dψβ/dt
///
/// Flux observer:
///   dψα/dt = vα - Rs*iα - Ls*diα/dt  (simplified Ls = Ld = Lq)
///   dψβ/dt = vβ - Rs*iβ - Ls*diβ/dt
///
/// Electrical angle estimate: θ̂e = atan2(-ψα, ψβ)
/// Speed estimate: ω̂e = dθ̂e/dt (filtered)
///
/// Uses a first-order low-pass filter on flux to reduce integration drift.
#[derive(Debug, Clone, Copy)]
pub struct BackEmfObserver<S: ControlScalar> {
    /// Stator resistance (Ω).
    pub r_s: S,
    /// Stator inductance (H) — simplified Ld = Lq = Ls.
    pub l_s: S,
    /// Flux observer cutoff frequency (rad/s). Higher → less filtering.
    pub omega_cutoff: S,
    /// Estimated flux α-component.
    psi_alpha: S,
    /// Estimated flux β-component.
    psi_beta: S,
    /// Previous α current for derivative approximation.
    i_alpha_prev: S,
    /// Previous β current.
    i_beta_prev: S,
    /// Estimated electrical angle (rad).
    theta_e: S,
    /// Previous electrical angle for speed estimation.
    theta_e_prev: S,
    /// Estimated electrical speed (rad/s), low-pass filtered.
    omega_e: S,
    /// Speed filter coefficient.
    omega_alpha: S,
    /// Initialized flag.
    initialized: bool,
}

impl<S: ControlScalar> BackEmfObserver<S> {
    /// Create a back-EMF observer.
    ///
    /// - `r_s`: stator resistance (Ω)
    /// - `l_s`: stator inductance (H)
    /// - `omega_cutoff`: flux observer bandwidth (rad/s), e.g. 2π*50
    /// - `omega_filter`: speed estimate filter (rad/s), e.g. 2π*20
    pub fn new(r_s: S, l_s: S, omega_cutoff: S, omega_filter: S) -> Self {
        Self {
            r_s,
            l_s,
            omega_cutoff,
            psi_alpha: S::ZERO,
            psi_beta: S::ZERO,
            i_alpha_prev: S::ZERO,
            i_beta_prev: S::ZERO,
            theta_e: S::ZERO,
            theta_e_prev: S::ZERO,
            omega_e: S::ZERO,
            omega_alpha: omega_filter,
            initialized: false,
        }
    }

    /// Update the observer.
    ///
    /// - `v_alpha`, `v_beta`: applied stator voltage in αβ frame (V)
    /// - `i_alpha`, `i_beta`: measured stator current in αβ frame (A)
    /// - `dt`: time step (s)
    ///
    /// Returns `(theta_e, omega_e)` — estimated electrical angle and speed.
    pub fn update(&mut self, v_alpha: S, v_beta: S, i_alpha: S, i_beta: S, dt: S) -> (S, S) {
        if !self.initialized {
            self.i_alpha_prev = i_alpha;
            self.i_beta_prev = i_beta;
            self.initialized = true;
            return (S::ZERO, S::ZERO);
        }

        // Current derivatives (backward difference)
        let di_alpha_dt = (i_alpha - self.i_alpha_prev) / dt;
        let di_beta_dt = (i_beta - self.i_beta_prev) / dt;

        // Back-EMF = V - R*i - L*di/dt
        let e_alpha = v_alpha - self.r_s * i_alpha - self.l_s * di_alpha_dt;
        let e_beta = v_beta - self.r_s * i_beta - self.l_s * di_beta_dt;

        // Flux observer: dψ/dt = e - ωc*(ψ - ψ_lp) (modified integrator with LP feedback)
        // Simplified: pure integration with LP filter to combat DC drift
        // ψ_new = ψ + (e - ωc*ψ)*dt
        self.psi_alpha += (e_alpha - self.omega_cutoff * self.psi_alpha) * dt;
        self.psi_beta += (e_beta - self.omega_cutoff * self.psi_beta) * dt;

        // Angle estimation: θ̂e = atan2(-ψα, ψβ)
        self.theta_e_prev = self.theta_e;
        self.theta_e = (-self.psi_alpha).atan2(self.psi_beta);

        // Speed estimation: ω̂e = (dθ/dt) filtered
        let mut d_theta = self.theta_e - self.theta_e_prev;

        // Handle wrap-around
        let pi = S::PI;
        let two_pi = S::TWO * pi;
        if d_theta > pi {
            d_theta -= two_pi;
        } else if d_theta < -pi {
            d_theta += two_pi;
        }

        let omega_raw = if dt > S::ZERO { d_theta / dt } else { S::ZERO };

        // Exponential filter on speed
        let tau = if self.omega_alpha > S::ZERO {
            S::ONE / self.omega_alpha
        } else {
            S::from_f64(0.01)
        };
        let alpha = S::ONE - (-dt / tau).exp();
        self.omega_e += alpha * (omega_raw - self.omega_e);

        self.i_alpha_prev = i_alpha;
        self.i_beta_prev = i_beta;

        (self.theta_e, self.omega_e)
    }

    pub fn theta_e(&self) -> S {
        self.theta_e
    }

    pub fn omega_e(&self) -> S {
        self.omega_e
    }

    pub fn flux_alpha(&self) -> S {
        self.psi_alpha
    }

    pub fn flux_beta(&self) -> S {
        self.psi_beta
    }

    pub fn reset(&mut self) {
        self.psi_alpha = S::ZERO;
        self.psi_beta = S::ZERO;
        self.i_alpha_prev = S::ZERO;
        self.i_beta_prev = S::ZERO;
        self.theta_e = S::ZERO;
        self.theta_e_prev = S::ZERO;
        self.omega_e = S::ZERO;
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer_initializes() {
        let mut obs = BackEmfObserver::new(0.5_f64, 0.001, 100.0, 50.0);
        let (theta, omega) = obs.update(0.0, 0.0, 0.0, 0.0, 0.001);
        assert_eq!(theta, 0.0);
        assert_eq!(omega, 0.0);
    }

    #[test]
    fn non_zero_voltage_updates_flux() {
        let mut obs = BackEmfObserver::new(0.5_f64, 0.001, 100.0, 50.0);
        // Apply voltage, run multiple steps
        for _ in 0..100 {
            obs.update(10.0, 0.0, 1.0, 0.0, 0.001);
        }
        // Flux should have grown
        let total_flux =
            (obs.flux_alpha() * obs.flux_alpha() + obs.flux_beta() * obs.flux_beta()).sqrt();
        assert!(total_flux > 0.0, "flux should be non-zero");
    }

    #[test]
    fn reset_clears_state() {
        let mut obs = BackEmfObserver::new(0.5_f64, 0.001, 100.0, 50.0);
        for _ in 0..50 {
            obs.update(5.0, 5.0, 0.5, 0.5, 0.001);
        }
        obs.reset();
        assert_eq!(obs.flux_alpha(), 0.0);
        assert_eq!(obs.flux_beta(), 0.0);
        assert_eq!(obs.omega_e(), 0.0);
    }

    #[test]
    fn sinusoidal_back_emf_produces_angle() {
        let mut obs = BackEmfObserver::new(0.1_f64, 0.001, 50.0, 20.0);
        let dt = 0.0001;
        let omega = 200.0_f64; // 200 rad/s
        let psi_f = 0.05_f64;
        // Simulate back-EMF directly as current (simplified test)
        // Feed synthetic eα = -ψf*ω*sin(ωt), eβ = ψf*ω*cos(ωt)
        for step in 0..2000 {
            let t = step as f64 * dt;
            let e_alpha = -psi_f * omega * (omega * t).sin();
            let e_beta = psi_f * omega * (omega * t).cos();
            // v = e + R*i (with small i)
            obs.update(e_alpha, e_beta, 0.0, 0.0, dt);
        }
        // After 2000 steps (0.2s), omega should be estimated
        let omega_est = obs.omega_e().abs();
        // Very rough check — the filter introduces lag
        assert!(omega_est > 10.0, "omega_est={:.1}", omega_est);
    }
}
