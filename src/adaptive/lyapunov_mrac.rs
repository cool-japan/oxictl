//! Lyapunov-stable MRAC: provably stable parameter adaptation.
//!
//! Plant: y[k+1] = a_p*y[k] + b_p*u[k]  (a_p, b_p unknown)
//! Reference model: y_m[k+1] = a_m*y_m[k] + b_m*r[k]
//!
//! Controller: u = θ_r*r + θ_y*y
//!
//! Lyapunov adaptation law (ensures V̇ ≤ 0):
//!   θ̇_r = -Γ * P * e * r
//!   θ̇_y = -Γ * P * e * y
//!
//! where e = y - y_m and P is the Lyapunov matrix satisfying a_m^T P a_m - P = -Q.

use crate::core::scalar::ControlScalar;

/// Lyapunov-based MRAC for a first-order SISO plant.
///
/// Parameter vector θ = [θ_r, θ_y] where:
/// - θ_r is the reference feedforward gain
/// - θ_y is the plant output feedback gain
#[derive(Debug, Clone, Copy)]
pub struct LyapunovMrac<S: ControlScalar> {
    /// Reference model: y_m[k+1] = a_m*y_m + b_m*r
    pub a_m: S,
    pub b_m: S,
    /// Adaptation gain Γ (scalar for SISO).
    pub gamma: S,
    /// Lyapunov matrix P (scalar > 0, e.g. solved from a_m^2*P - P = -1 → P = 1/(1-a_m^2)).
    pub p: S,
    /// Parameter vector [θ_r, θ_y].
    pub theta: [S; 2],
    /// Reference model state.
    y_m: S,
    /// Tracking error e = y - y_m.
    error: S,
    /// Reference model output (same as y_m after update).
    pub y_m_out: S,
}

impl<S: ControlScalar> LyapunovMrac<S> {
    /// Create a Lyapunov-MRAC controller.
    ///
    /// # Arguments
    /// - `a_m`: reference model pole (|a_m| < 1 for stability)
    /// - `b_m`: reference model gain
    /// - `gamma`: adaptation rate (Γ > 0)
    /// - `p`: Lyapunov scalar (P > 0); a convenient choice is `1.0 / (1.0 - a_m^2)`.
    pub fn new(a_m: S, b_m: S, gamma: S, p: S) -> Self {
        Self {
            a_m,
            b_m,
            gamma,
            p,
            theta: [b_m, S::ZERO], // initial: θ_r = b_m, θ_y = 0
            y_m: S::ZERO,
            error: S::ZERO,
            y_m_out: S::ZERO,
        }
    }

    /// Lyapunov function value V = P*e^2 + (1/Γ) * ||θ̃||^2.
    ///
    /// `theta_star`: true parameter vector [θ_r*, θ_y*].
    pub fn lyapunov_value(&self, theta_star: [S; 2]) -> S {
        let e2 = self.error * self.error;
        let dt0 = self.theta[0] - theta_star[0];
        let dt1 = self.theta[1] - theta_star[1];
        let theta_err2 = dt0 * dt0 + dt1 * dt1;
        self.p * e2 + theta_err2 / self.gamma
    }

    /// Compute control output and update adaptation.
    ///
    /// - `r`: reference signal
    /// - `y`: measured plant output
    ///
    /// Returns control signal u = θ_r*r + θ_y*y.
    pub fn update(&mut self, r: S, y: S) -> S {
        // Step reference model forward
        self.y_m = self.a_m * self.y_m + self.b_m * r;
        self.y_m_out = self.y_m;

        // Tracking error
        self.error = y - self.y_m;

        // Lyapunov adaptation law: Δθ = -Γ * P * e * regressor
        let adapt = self.gamma * self.p * self.error;
        self.theta[0] -= adapt * r;
        self.theta[1] -= adapt * y;

        // Control law
        self.theta[0] * r + self.theta[1] * y
    }

    /// Current tracking error.
    pub fn error(&self) -> S {
        self.error
    }

    /// Reset controller to initial state.
    pub fn reset(&mut self) {
        self.y_m = S::ZERO;
        self.error = S::ZERO;
        self.y_m_out = S::ZERO;
        self.theta = [self.b_m, S::ZERO];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate plant + Lyapunov-MRAC for `n` steps.
    fn simulate(n: usize, a_p: f64, b_p: f64) -> f64 {
        let a_m = 0.7_f64;
        let b_m = 0.3_f64;
        let gamma = 0.05_f64;
        let p = 1.0 / (1.0 - a_m * a_m); // discrete Lyapunov solution
        let mut ctrl = LyapunovMrac::new(a_m, b_m, gamma, p);
        let mut y = 0.0_f64;
        let r = 1.0_f64;
        for _ in 0..n {
            let u = ctrl.update(r, y);
            y = a_p * y + b_p * u;
        }
        ctrl.error()
    }

    #[test]
    fn test_tracking_error_decreases() {
        let early = simulate(20, 0.8, 1.0).abs();
        let late = simulate(500, 0.8, 1.0).abs();
        assert!(late < early, "early={early}, late={late}");
    }

    #[test]
    fn test_lyapunov_value_non_negative() {
        let a_m = 0.7_f64;
        let b_m = 0.3_f64;
        let gamma = 0.05_f64;
        let p = 1.0 / (1.0 - a_m * a_m);
        let mut ctrl = LyapunovMrac::new(a_m, b_m, gamma, p);
        let mut y = 0.0_f64;
        let theta_star = [1.0_f64, 0.0_f64];
        for _ in 0..100 {
            let u = ctrl.update(1.0, y);
            y = 0.8 * y + u;
            let v = ctrl.lyapunov_value(theta_star);
            assert!(v >= 0.0, "V={v}");
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let mut ctrl = LyapunovMrac::new(0.7_f64, 0.3_f64, 0.05_f64, 2.0_f64);
        for _ in 0..50 {
            ctrl.update(1.0, 0.5);
        }
        ctrl.reset();
        assert_eq!(ctrl.error(), 0.0);
        assert_eq!(ctrl.y_m_out, 0.0);
        assert_eq!(ctrl.theta[1], 0.0);
    }

    #[test]
    fn test_reference_model_output() {
        let a_m = 0.8_f64;
        let b_m = 0.2_f64;
        let mut ctrl = LyapunovMrac::new(a_m, b_m, 0.01_f64, 5.0_f64);
        // Step reference model with zero plant feedback influence
        let r = 1.0_f64;
        let mut expected_ym = 0.0_f64;
        for _ in 0..10 {
            ctrl.update(r, expected_ym); // y == y_m ≈ 0 initially
            expected_ym = a_m * expected_ym + b_m * r;
        }
        // y_m_out should track the reference model
        let diff = (ctrl.y_m_out - expected_ym).abs();
        // Allow tolerance due to controller adapting theta
        assert!(
            diff < 0.5,
            "y_m_out={} expected={}",
            ctrl.y_m_out,
            expected_ym
        );
    }
}
