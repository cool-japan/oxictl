use crate::core::scalar::ControlScalar;

/// Inverted pendulum on a cart simulator.
///
/// Nonlinear dynamics (Euler integration):
///   ẍ = (F + m*l*(θ̈*cos(θ) - θ̇²*sin(θ))) / (M+m)
///   θ̈ = (g*sin(θ) - cos(θ)*(F + m*l*θ̇²*sin(θ)) / (M+m))
///        / (l*(4/3 - m*cos²(θ)/(M+m)))
///
/// State: [x, ẋ, θ, θ̇]
///   - x: cart position (m)
///   - ẋ: cart velocity (m/s)
///   - θ: pole angle from vertical (rad), θ=0 is upright
///   - θ̇: angular velocity (rad/s)
///
/// This is a standard benchmark for control algorithms.
#[derive(Debug, Clone, Copy)]
pub struct InvertedPendulum<S: ControlScalar> {
    /// Cart mass (kg).
    pub m_cart: S,
    /// Pole mass (kg).
    pub m_pole: S,
    /// Half-length of pole (m).
    pub l_pole: S,
    /// Gravitational acceleration (m/s²).
    pub g: S,
    /// Cart friction coefficient.
    pub b_cart: S,
    // State: [x, x_dot, theta, theta_dot]
    state: [S; 4],
}

impl<S: ControlScalar> InvertedPendulum<S> {
    pub fn new(m_cart: S, m_pole: S, l_pole: S, g: S) -> Self {
        Self {
            m_cart,
            m_pole,
            l_pole,
            g,
            b_cart: S::ZERO,
            state: [S::ZERO; 4],
        }
    }

    /// Standard benchmark: cart=1kg, pole=0.1kg, half-length=0.5m, g=9.81.
    pub fn standard() -> Self
    where
        S: From<f32>,
    {
        Self::new(
            S::from_f64(1.0),
            S::from_f64(0.1),
            S::from_f64(0.5),
            S::from_f64(9.81),
        )
    }

    pub fn with_initial_angle(mut self, theta_rad: S) -> Self {
        self.state[2] = theta_rad;
        self
    }

    pub fn with_friction(mut self, b: S) -> Self {
        self.b_cart = b;
        self
    }

    /// Advance one timestep.
    ///
    /// `f`: horizontal force applied to cart (N)
    pub fn step(&mut self, f: S, dt: S) {
        let x = self.state[0];
        let xd = self.state[1];
        let theta = self.state[2];
        let thetad = self.state[3];

        let _ = x; // suppress unused warning

        let m = self.m_pole;
        let mc = self.m_cart;
        let l = self.l_pole;
        let g = self.g;
        let mt = mc + m;

        let sin_t = theta.sin();
        let cos_t = theta.cos();

        // Effective force on cart
        let f_eff = f - self.b_cart * xd;

        // θ̈ denominator
        let denom = l * (S::from_f64(4.0 / 3.0) - m * cos_t * cos_t / mt);

        let theta_ddot = if denom.abs() > S::EPSILON {
            (g * sin_t - cos_t * (f_eff + m * l * thetad * thetad * sin_t) / mt) / denom
        } else {
            S::ZERO
        };

        let x_ddot = (f_eff + m * l * (thetad * thetad * sin_t - theta_ddot * cos_t)) / mt;

        // Euler integration
        self.state[0] += xd * dt;
        self.state[1] += x_ddot * dt;
        self.state[2] += thetad * dt;
        self.state[3] += theta_ddot * dt;
    }

    pub fn state(&self) -> &[S; 4] {
        &self.state
    }

    pub fn cart_position(&self) -> S {
        self.state[0]
    }

    pub fn cart_velocity(&self) -> S {
        self.state[1]
    }

    pub fn pole_angle(&self) -> S {
        self.state[2]
    }

    pub fn pole_angular_velocity(&self) -> S {
        self.state[3]
    }

    pub fn is_fallen(&self, max_angle_rad: S) -> bool {
        self.state[2].abs() > max_angle_rad
    }

    pub fn reset(&mut self) {
        self.state = [S::ZERO; 4];
    }

    pub fn set_state(&mut self, state: [S; 4]) {
        self.state = state;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unforced_falls_over() {
        let mut p = InvertedPendulum::<f64>::new(1.0, 0.1, 0.5, 9.81);
        p.set_state([0.0, 0.0, 0.1, 0.0]); // small initial angle
        for _ in 0..1000 {
            p.step(0.0, 0.001);
        }
        assert!(
            p.pole_angle().abs() > 0.5,
            "Pendulum should fall: θ={:.3}",
            p.pole_angle()
        );
    }

    #[test]
    fn falls_detected() {
        let mut p = InvertedPendulum::<f64>::new(1.0, 0.1, 0.5, 9.81);
        p.set_state([0.0, 0.0, 1.0, 0.0]); // large initial angle
        for _ in 0..1000 {
            p.step(0.0, 0.001);
        }
        assert!(p.is_fallen(1.5));
    }

    #[test]
    fn zero_angle_upright_no_force() {
        let mut p = InvertedPendulum::<f64>::new(1.0, 0.1, 0.5, 9.81);
        // Perfect upright: θ=0, no disturbance
        for _ in 0..100 {
            p.step(0.0, 0.001);
        }
        // Should remain at zero (unstable equilibrium)
        assert!(p.pole_angle().abs() < 1e-10);
    }

    #[test]
    fn force_moves_cart() {
        let mut p = InvertedPendulum::<f64>::new(1.0, 0.1, 0.5, 9.81);
        for _ in 0..1000 {
            p.step(1.0, 0.001);
        }
        assert!(p.cart_position() > 0.0, "Cart should move forward");
    }

    #[test]
    fn reset_clears_state() {
        let mut p = InvertedPendulum::<f64>::new(1.0, 0.1, 0.5, 9.81);
        p.set_state([1.0, 2.0, 0.3, 0.4]);
        p.reset();
        assert_eq!(p.state(), &[0.0_f64; 4]);
    }

    #[test]
    fn lqr_stabilizes_linearized_pendulum() {
        // Pole-only stabilization gains for (m=0.1, M=1.0, l=0.5, g=9.81).
        // F = -(k[0]*x + k[1]*xdot + k[2]*theta + k[3]*thetadot)
        // k[2]=-50, k[3]=-10 gives closed-loop poles at s ≈ -7.32 ± 1.96i (continuous),
        // which stabilizes the pole with time constant ~0.14s. Cart may drift (test ignores it).
        let k = [0.0_f64, 0.0, -50.0, -10.0];
        let mut p = InvertedPendulum::<f64>::new(1.0, 0.1, 0.5, 9.81);
        p.set_state([0.0, 0.0, 0.05, 0.0]); // small perturbation

        for _ in 0..5000 {
            let s = p.state();
            let f = -(k[0] * s[0] + k[1] * s[1] + k[2] * s[2] + k[3] * s[3]);
            p.step(f, 0.001);
            if p.is_fallen(1.0) {
                break;
            }
        }
        assert!(
            p.pole_angle().abs() < 0.1,
            "LQR should stabilize: θ={:.4}",
            p.pole_angle()
        );
    }
}
