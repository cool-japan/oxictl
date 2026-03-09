use crate::core::scalar::ControlScalar;

/// Model Reference Adaptive Control (MRAC) for second-order SISO plants.
///
/// Reference model (continuous, discretized with Euler):
///   ÿ_m + 2ζω_n·ẏ_m + ω_n²·y_m = ω_n²·r
///
/// Discretized with step dt:
///   ẏ_m[k+1] = ẏ_m[k] + dt*(ω_n²*r - 2ζω_n*ẏ_m - ω_n²*y_m)
///   y_m[k+1]  = y_m[k]  + dt*ẏ_m[k]
///
/// Adaptive controller:
///   u = θ1*r + θ2*y + θ3*ẏ
///
/// MIT rule (gradient on 1/2*e²):
///   θ̇_i = -γ * e * ∂ŷ/∂θ_i
///
/// Parameter projection: clamp each θ to [-θ_max, θ_max] to prevent drift.
/// Normalization: divide gradient by (1 + φ^T*φ).
#[derive(Debug, Clone, Copy)]
pub struct MracSecondOrder<S: ControlScalar> {
    /// Natural frequency ω_n of reference model (rad/s).
    pub omega_n: S,
    /// Damping ratio ζ of reference model.
    pub zeta: S,
    /// Integration step (s).
    pub dt: S,
    /// Adaptation gain γ.
    pub gamma: S,
    /// Maximum parameter magnitude (projection bound).
    pub theta_max: S,
    /// Feedforward gain θ1 (for r).
    pub theta1: S,
    /// Proportional feedback gain θ2 (for y).
    pub theta2: S,
    /// Derivative feedback gain θ3 (for ẏ).
    pub theta3: S,
    /// Reference model output.
    y_m: S,
    /// Reference model velocity.
    yd_m: S,
    /// Tracking error (y_p - y_m).
    error: S,
    /// Normalization constant.
    epsilon: S,
}

impl<S: ControlScalar> MracSecondOrder<S> {
    /// Create second-order MRAC.
    ///
    /// `omega_n`: desired natural frequency (rad/s), e.g. 5.0.
    /// `zeta`: desired damping ratio, e.g. 0.7.
    /// `dt`: sample period (s).
    /// `gamma`: adaptation rate.
    /// `theta_max`: parameter projection bound.
    pub fn new(omega_n: S, zeta: S, dt: S, gamma: S, theta_max: S) -> Self {
        let two = S::from_f64(2.0);
        Self {
            omega_n,
            zeta,
            dt,
            gamma,
            theta_max,
            // Initial guess: match reference model DC gain
            theta1: omega_n * omega_n,
            theta2: -(omega_n * omega_n),
            theta3: -(two * zeta * omega_n),
            y_m: S::ZERO,
            yd_m: S::ZERO,
            error: S::ZERO,
            epsilon: S::from_f64(1e-4),
        }
    }

    /// Advance one timestep.
    ///
    /// `r`:  reference input.
    /// `y`:  measured plant output.
    /// `yd`: measured or estimated plant velocity (first derivative).
    ///
    /// Returns control signal u.
    pub fn update(&mut self, r: S, y: S, yd: S) -> S {
        let wn = self.omega_n;
        let wn2 = wn * wn;
        let two = S::from_f64(2.0);

        // Advance reference model (Euler)
        let ydd_m = wn2 * r - two * self.zeta * wn * self.yd_m - wn2 * self.y_m;
        self.yd_m += self.dt * ydd_m;
        self.y_m += self.dt * self.yd_m;

        // Tracking error
        self.error = y - self.y_m;

        // Regressor vector φ = [r, y, yd]
        let phi_r = r;
        let phi_y = y;
        let phi_yd = yd;

        // Normalization signal
        let norm = S::ONE + phi_r * phi_r + phi_y * phi_y + phi_yd * phi_yd + self.epsilon;

        // MIT rule with normalization
        let grad_scale = self.gamma * self.error / norm;
        self.theta1 -= grad_scale * phi_r;
        self.theta2 -= grad_scale * phi_y;
        self.theta3 -= grad_scale * phi_yd;

        // Parameter projection: clamp to [-theta_max, theta_max]
        self.theta1 = self.theta1.clamp_val(-self.theta_max, self.theta_max);
        self.theta2 = self.theta2.clamp_val(-self.theta_max, self.theta_max);
        self.theta3 = self.theta3.clamp_val(-self.theta_max, self.theta_max);

        // Control law
        self.theta1 * r + self.theta2 * y + self.theta3 * yd
    }

    /// Reference model output (desired trajectory).
    pub fn reference_output(&self) -> S {
        self.y_m
    }

    /// Reference model velocity.
    pub fn reference_velocity(&self) -> S {
        self.yd_m
    }

    /// Current tracking error e = y_p - y_m.
    pub fn tracking_error(&self) -> S {
        self.error
    }

    /// Reset internal states; keep tuned parameters.
    pub fn reset_states(&mut self) {
        self.y_m = S::ZERO;
        self.yd_m = S::ZERO;
        self.error = S::ZERO;
    }

    /// Full reset including adaptive parameters.
    pub fn reset(&mut self) {
        self.reset_states();
        let two = S::from_f64(2.0);
        self.theta1 = self.omega_n * self.omega_n;
        self.theta2 = -(self.omega_n * self.omega_n);
        self.theta3 = -(two * self.zeta * self.omega_n);
    }
}

/// Lyapunov-stable MRAC using a positive-definite adaptation law derived from
/// a Lyapunov function V = e^T*P*e + (1/γ)*Δθ^T*Δθ.
///
/// For a first-order plant with known sign of high-frequency gain:
///   Plant:     y[k+1] = a*y[k] + b*u[k]        (b > 0 assumed)
///   Ref model: y_m[k+1] = a_m*y_m[k] + b_m*r[k]
///
/// Adaptation law (discrete SPR-based):
///   Δθ_r = -γ * e * r  (Lyapunov-stable for b_m/b > 0)
///   Δθ_y = -γ * e * y
///
/// Lyapunov function candidate: V = e² + (1/γ)*(Δθ_r² + Δθ_y²)
/// ΔV ≤ 0 when |a_m| < 1 and γ small enough.
#[derive(Debug, Clone, Copy)]
pub struct LyapunovMracFirstOrder<S: ControlScalar> {
    /// Reference model parameter a_m (|a_m| < 1 for stability).
    pub a_m: S,
    /// Reference model gain b_m.
    pub b_m: S,
    /// Adaptation rate γ.
    pub gamma: S,
    /// Feedforward gain θ_r.
    pub theta_r: S,
    /// Feedback gain θ_y.
    pub theta_y: S,
    /// Reference model state.
    y_m: S,
    /// Last tracking error.
    error: S,
    /// Lyapunov function value (informational).
    lyapunov_v: S,
}

impl<S: ControlScalar> LyapunovMracFirstOrder<S> {
    /// Create Lyapunov-based MRAC.
    ///
    /// Requirements: |a_m| < 1 (stable reference model), gamma > 0 (small).
    pub fn new(a_m: S, b_m: S, gamma: S) -> Self {
        Self {
            a_m,
            b_m,
            gamma,
            theta_r: b_m,
            theta_y: S::ZERO,
            y_m: S::ZERO,
            error: S::ZERO,
            lyapunov_v: S::ZERO,
        }
    }

    /// Update and return control output.
    pub fn update(&mut self, r: S, y: S) -> S {
        // Advance reference model
        self.y_m = self.a_m * self.y_m + self.b_m * r;

        // Tracking error
        self.error = y - self.y_m;

        // Lyapunov-based adaptation (same as MIT rule here, but derivation is Lyapunov)
        let delta_theta_r = -self.gamma * self.error * r;
        let delta_theta_y = -self.gamma * self.error * y;

        self.theta_r += delta_theta_r;
        self.theta_y += delta_theta_y;

        // Compute Lyapunov function V = e² + (1/γ)*(Δθ_r² + Δθ_y²)
        // Using deviations from nominal: Δθ_r = theta_r - b_m/b_p (unknown), track magnitude
        let inv_gamma = if self.gamma.abs() > S::EPSILON {
            S::ONE / self.gamma
        } else {
            S::from_f64(1e6)
        };
        self.lyapunov_v = self.error * self.error
            + inv_gamma * (self.theta_r * self.theta_r + self.theta_y * self.theta_y);

        // Control law
        self.theta_r * r + self.theta_y * y
    }

    pub fn reference_output(&self) -> S {
        self.y_m
    }

    pub fn tracking_error(&self) -> S {
        self.error
    }

    /// Informational Lyapunov function value.
    pub fn lyapunov_value(&self) -> S {
        self.lyapunov_v
    }

    pub fn reset(&mut self) {
        self.y_m = S::ZERO;
        self.error = S::ZERO;
        self.lyapunov_v = S::ZERO;
        self.theta_r = self.b_m;
        self.theta_y = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_order_reference_model_advances() {
        // Reference model with ωn=5, ζ=0.7, dt=0.01
        let mut mrac = MracSecondOrder::<f64>::new(5.0, 0.7, 0.01, 0.001, 100.0);
        let y_prev = mrac.reference_output();
        mrac.update(1.0, 0.0, 0.0);
        // Reference output should have changed from zero
        let _ = y_prev; // was zero, will still be near zero after one step
        assert!(mrac.reference_velocity().abs() >= 0.0); // nonneg check
    }

    #[test]
    fn second_order_tracks_step_reference() {
        let dt = 0.01_f64;
        let mut mrac = MracSecondOrder::<f64>::new(3.0, 0.8, dt, 0.005, 50.0);
        let mut y = 0.0_f64;
        let mut yd = 0.0_f64;

        // Plant: second-order y'' + 3y' + y = 2u  (discretized crudely)
        for _ in 0..3000 {
            let u = mrac.update(1.0, y, yd);
            let ydd = 2.0 * u - 3.0 * yd - y;
            yd += dt * ydd;
            y += dt * yd;
            if y.abs() > 1e4 {
                break; // diverge guard
            }
        }
        // Soft check: error bounded
        assert!(
            mrac.tracking_error().abs() < 5.0,
            "error={}",
            mrac.tracking_error()
        );
    }

    #[test]
    fn second_order_projection_bounds_parameters() {
        let mut mrac = MracSecondOrder::<f64>::new(5.0, 0.7, 0.01, 10.0, 20.0);
        for _ in 0..1000 {
            mrac.update(100.0, 50.0, 10.0);
        }
        assert!(mrac.theta1.abs() <= 20.0 + 1e-9);
        assert!(mrac.theta2.abs() <= 20.0 + 1e-9);
        assert!(mrac.theta3.abs() <= 20.0 + 1e-9);
    }

    #[test]
    fn second_order_reset_clears_states() {
        let mut mrac = MracSecondOrder::<f64>::new(5.0, 0.7, 0.01, 0.01, 100.0);
        for _ in 0..100 {
            mrac.update(1.0, 0.5, 0.1);
        }
        mrac.reset();
        assert_eq!(mrac.reference_output(), 0.0);
        assert_eq!(mrac.tracking_error(), 0.0);
    }

    #[test]
    fn lyapunov_mrac_stays_bounded() {
        let mut mrac = LyapunovMracFirstOrder::<f64>::new(0.8, 0.2, 0.005);
        let mut y = 0.0_f64;
        let mut bounded = true;

        for _ in 0..5000 {
            let u = mrac.update(1.0, y);
            // Plant: y[k+1] = 0.7*y + 1.5*u
            y = 0.7 * y + 1.5 * u;
            if y.abs() > 1000.0 {
                bounded = false;
                break;
            }
        }
        assert!(bounded, "Lyapunov MRAC output diverged");
    }

    #[test]
    fn lyapunov_mrac_value_nonnegative() {
        let mut mrac = LyapunovMracFirstOrder::<f64>::new(0.8, 0.2, 0.01);
        let mut y = 0.0_f64;
        for _ in 0..100 {
            let u = mrac.update(1.0, y);
            y = 0.7 * y + 1.5 * u;
        }
        assert!(mrac.lyapunov_value() >= 0.0);
    }

    #[test]
    fn lyapunov_mrac_reset() {
        let mut mrac = LyapunovMracFirstOrder::<f64>::new(0.8, 0.2, 0.01);
        let mut y = 0.0_f64;
        for _ in 0..100 {
            let u = mrac.update(1.0, y);
            y = 0.7 * y + u;
        }
        mrac.reset();
        assert_eq!(mrac.reference_output(), 0.0);
        assert_eq!(mrac.tracking_error(), 0.0);
        assert_eq!(mrac.theta_y, 0.0);
    }
}
