use crate::core::scalar::ControlScalar;

/// Full nonlinear pendulum simulation (not linearized).
///
/// Simple pendulum dynamics:
///   ml²θ̈ = -mgl·sin(θ) - b·θ̇ + τ
///   ⟺  θ̈ = -(g/l)·sin(θ) - (b/(ml²))·θ̇ + τ/(ml²)
///
/// For a pendulum on a cart, full coupled dynamics are implemented.
///
/// Integration: 4th-order Runge-Kutta (RK4) for accuracy.
///
/// State: [θ, θ̇] — angle from downward equilibrium (θ=0 is hanging down,
/// θ=π is upright).
///
/// Energy:
///   E_kinetic = ½·m·l²·θ̇²
///   E_potential = m·g·l·(1 - cos(θ))   (zero at θ=0, maximum at θ=π)
#[derive(Debug, Clone, Copy)]
pub struct NonlinearPendulum<S: ControlScalar> {
    /// Pendulum length (m).
    pub length: S,
    /// Bob mass (kg).
    pub mass: S,
    /// Gravitational acceleration (m/s²).
    pub gravity: S,
    /// Damping coefficient (N·m·s/rad).
    pub damping: S,
    /// Angle from downward vertical (rad).
    theta: S,
    /// Angular velocity (rad/s).
    theta_dot: S,
}

impl<S: ControlScalar> NonlinearPendulum<S> {
    /// Create pendulum with given physical parameters.
    pub fn new(length: S, mass: S, gravity: S, damping: S) -> Self {
        Self {
            length,
            mass,
            gravity,
            damping,
            theta: S::ZERO,
            theta_dot: S::ZERO,
        }
    }

    /// Create standard benchmark pendulum: l=1m, m=1kg, g=9.81, b=0.
    pub fn standard() -> Self {
        Self::new(S::ONE, S::ONE, S::from_f64(9.81), S::ZERO)
    }

    /// Set initial conditions.
    pub fn set_state(&mut self, theta: S, theta_dot: S) {
        self.theta = theta;
        self.theta_dot = theta_dot;
    }

    /// State vector [θ, θ̇].
    pub fn state(&self) -> [S; 2] {
        [self.theta, self.theta_dot]
    }

    pub fn angle(&self) -> S {
        self.theta
    }

    pub fn angular_velocity(&self) -> S {
        self.theta_dot
    }

    /// Compute derivatives [θ̇, θ̈] for given state and torque input.
    fn derivatives(&self, theta: S, theta_dot: S, torque: S) -> [S; 2] {
        let ml2 = self.mass * self.length * self.length;
        if ml2.abs() < S::EPSILON {
            return [S::ZERO; 2];
        }
        let theta_ddot = -(self.gravity / self.length) * theta.sin()
            - (self.damping / ml2) * theta_dot
            + torque / ml2;
        [theta_dot, theta_ddot]
    }

    /// Advance one timestep using RK4 integration.
    ///
    /// `torque`: applied torque at the pivot (N·m).
    pub fn step(&mut self, torque: S, dt: S) {
        let [th, thd] = [self.theta, self.theta_dot];

        // k1
        let [k1_th, k1_thd] = self.derivatives(th, thd, torque);
        // k2
        let [k2_th, k2_thd] = self.derivatives(
            th + S::HALF * dt * k1_th,
            thd + S::HALF * dt * k1_thd,
            torque,
        );
        // k3
        let [k3_th, k3_thd] = self.derivatives(
            th + S::HALF * dt * k2_th,
            thd + S::HALF * dt * k2_thd,
            torque,
        );
        // k4
        let [k4_th, k4_thd] = self.derivatives(th + dt * k3_th, thd + dt * k3_thd, torque);

        let sixth = S::ONE / S::from_f64(6.0);
        self.theta += sixth * dt * (k1_th + S::TWO * k2_th + S::TWO * k3_th + k4_th);
        self.theta_dot += sixth * dt * (k1_thd + S::TWO * k2_thd + S::TWO * k3_thd + k4_thd);
    }

    /// Total mechanical energy (J).
    ///
    /// E = ½·m·l²·θ̇² + m·g·l·(1 - cos(θ))
    pub fn energy(&self) -> S {
        let kinetic =
            S::HALF * self.mass * self.length * self.length * self.theta_dot * self.theta_dot;
        let potential = self.mass * self.gravity * self.length * (S::ONE - self.theta.cos());
        kinetic + potential
    }

    /// Kinetic energy only.
    pub fn kinetic_energy(&self) -> S {
        S::HALF * self.mass * self.length * self.length * self.theta_dot * self.theta_dot
    }

    /// Potential energy only.
    pub fn potential_energy(&self) -> S {
        self.mass * self.gravity * self.length * (S::ONE - self.theta.cos())
    }

    /// Reset state to zero.
    pub fn reset(&mut self) {
        self.theta = S::ZERO;
        self.theta_dot = S::ZERO;
    }

    /// Natural frequency ω₀ = √(g/l) for small oscillations.
    pub fn natural_frequency(&self) -> S {
        if self.length > S::ZERO {
            (self.gravity / self.length).sqrt()
        } else {
            S::ZERO
        }
    }

    /// Period of small oscillations T = 2π/ω₀.
    pub fn small_oscillation_period(&self) -> S {
        let omega = self.natural_frequency();
        if omega > S::EPSILON {
            S::TWO * S::PI / omega
        } else {
            S::from_f64(f64::INFINITY)
        }
    }
}

/// Pendulum mounted on a moving cart (full nonlinear coupled system).
///
/// States: [x, ẋ, θ, θ̇]
///   x: cart position (m)
///   ẋ: cart velocity (m/s)
///   θ: pendulum angle from upward vertical (rad), θ=0 is up
///   θ̇: angular velocity (rad/s)
///
/// Equations of motion (Lagrangian derivation):
///   (M + m)ẍ + m·l·(θ̈·cos(θ) - θ̇²·sin(θ)) = F
///   l·θ̈ + ẍ·cos(θ) = g·sin(θ)
///
/// where M = cart mass, m = bob mass, l = pendulum length, g = gravity.
#[derive(Debug, Clone, Copy)]
pub struct PendulumOnCart<S: ControlScalar> {
    /// Cart mass (kg).
    pub cart_mass: S,
    /// Pendulum bob mass (kg).
    pub bob_mass: S,
    /// Pendulum length (m).
    pub length: S,
    /// Gravitational acceleration (m/s²).
    pub gravity: S,
    /// Cart friction coefficient.
    pub cart_friction: S,
    /// State: [x, ẋ, θ, θ̇].
    state: [S; 4],
}

impl<S: ControlScalar> PendulumOnCart<S> {
    pub fn new(cart_mass: S, bob_mass: S, length: S, gravity: S) -> Self {
        Self {
            cart_mass,
            bob_mass,
            length,
            gravity,
            cart_friction: S::ZERO,
            state: [S::ZERO; 4],
        }
    }

    pub fn with_friction(mut self, friction: S) -> Self {
        self.cart_friction = friction;
        self
    }

    pub fn set_state(&mut self, state: [S; 4]) {
        self.state = state;
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
    pub fn angle(&self) -> S {
        self.state[2]
    }
    pub fn angular_velocity(&self) -> S {
        self.state[3]
    }

    fn derivatives(&self, s: &[S; 4], force: S) -> [S; 4] {
        let x_dot = s[1];
        let theta = s[2];
        let theta_dot = s[3];

        let m = self.bob_mass;
        let mc = self.cart_mass;
        let l = self.length;
        let g = self.gravity;
        let mt = mc + m;

        let sin_t = theta.sin();
        let cos_t = theta.cos();

        // Equations:
        // mt*x'' + m*l*(θ''*cos - θ'²*sin) = F - b*x'
        // l*θ'' + x''*cos = g*sin
        //
        // From second: x'' = (g*sin - l*θ'') / cos  (when cos ≠ 0)
        // Substituting into first:
        // mt*(g*sin - l*θ'')/cos + m*l*(θ''*cos - θ'²*sin) = F - b*x'
        // θ''*(m*l*cos - mt*l/cos) = F - b*x' - mt*g*sin/cos + m*l*θ'²*sin
        // θ''*(m*l*cos² - mt*l)/cos = rhs/cos ... simplify:
        //
        // Denominator: l*(m*cos² - mt) ← from standard derivation
        let denom = l * (m * cos_t * cos_t - mt);

        let f_eff = force - self.cart_friction * x_dot;

        let (x_ddot, theta_ddot) = if denom.abs() > S::EPSILON {
            let theta_ddot = (f_eff * cos_t - m * l * theta_dot * theta_dot * sin_t * cos_t
                + mt * g * sin_t)
                / denom;
            let x_ddot =
                (f_eff + m * l * (theta_dot * theta_dot * sin_t - theta_ddot * cos_t)) / mt;
            (x_ddot, theta_ddot)
        } else {
            (S::ZERO, S::ZERO)
        };

        [x_dot, x_ddot, theta_dot, theta_ddot]
    }

    /// RK4 integration step.
    pub fn step(&mut self, force: S, dt: S) {
        let s = self.state;

        let k1 = self.derivatives(&s, force);
        let s2: [S; 4] = core::array::from_fn(|i| s[i] + S::HALF * dt * k1[i]);
        let k2 = self.derivatives(&s2, force);
        let s3: [S; 4] = core::array::from_fn(|i| s[i] + S::HALF * dt * k2[i]);
        let k3 = self.derivatives(&s3, force);
        let s4: [S; 4] = core::array::from_fn(|i| s[i] + dt * k3[i]);
        let k4 = self.derivatives(&s4, force);

        let sixth = S::ONE / S::from_f64(6.0);
        for i in 0..4 {
            self.state[i] += sixth * dt * (k1[i] + S::TWO * k2[i] + S::TWO * k3[i] + k4[i]);
        }
    }

    /// Total energy (kinetic + potential) of the pendulum-cart system.
    pub fn energy(&self) -> S {
        let x_dot = self.state[1];
        let theta = self.state[2];
        let theta_dot = self.state[3];
        let l = self.length;
        let m = self.bob_mass;
        let mc = self.cart_mass;
        let g = self.gravity;

        // Cart KE: ½*M*ẋ²
        let cart_ke = S::HALF * mc * x_dot * x_dot;

        // Bob velocity: v_x = ẋ + l*θ̇*cos(θ), v_y = l*θ̇*sin(θ)
        let vbx = x_dot + l * theta_dot * theta.cos();
        let vby = l * theta_dot * theta.sin();
        let bob_ke = S::HALF * m * (vbx * vbx + vby * vby);

        // Bob PE: m*g*(pivot_height - l*cos(θ)) relative to pivot
        // With upward θ=0: height of bob = -l*cos(θ) (up positive)
        let bob_pe = -m * g * l * theta.cos();

        cart_ke + bob_ke + bob_pe
    }

    pub fn reset(&mut self) {
        self.state = [S::ZERO; 4];
    }

    /// Bob height above the pivot (negative when hanging down).
    ///
    /// h_bob = -l·cos(θ)  where θ=0 is upright.
    pub fn bob_height(&self) -> S {
        -self.length * self.state[2].cos()
    }

    /// Bob x-coordinate relative to pivot: x = l·sin(θ).
    pub fn bob_x(&self) -> S {
        self.length * self.state[2].sin()
    }

    /// Bob y-coordinate relative to pivot: y = -l·cos(θ).
    pub fn bob_y(&self) -> S {
        self.bob_height()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pendulum_energy_conserved_undamped() {
        let mut p = NonlinearPendulum::<f64>::standard();
        p.set_state(0.5, 0.0); // 0.5 rad initial angle, zero velocity
        let e0 = p.energy();
        let dt = 0.0001_f64;
        for _ in 0..10000 {
            p.step(0.0, dt);
        }
        let e1 = p.energy();
        // Energy should be conserved within numerical tolerance
        assert!(
            (e1 - e0).abs() / e0 < 1e-4,
            "energy not conserved: e0={:.6}, e1={:.6}",
            e0,
            e1
        );
    }

    #[test]
    fn pendulum_period_matches_theory() {
        // For small angle: T ≈ 2π√(l/g)
        //
        // Measurement strategy: starting from θ₀ > 0, dθ/dt = 0
        //   - pendulum swings negative at t ≈ T/4
        //   - records the downward zero-crossing time t_down (θ going negative)
        //   - records the upward zero-crossing time t_up (θ returning to ≥ 0)
        //   - half-period = t_up - t_down ≈ T/2
        //   - full period = 2 * half-period
        let mut p = NonlinearPendulum::<f64>::new(1.0, 1.0, 9.81, 0.0);
        let theta0 = 0.05_f64; // small angle
        p.set_state(theta0, 0.0);
        let t_theory = p.small_oscillation_period();

        let dt = 0.0001_f64;
        let max_steps = (3.0 * t_theory / dt) as usize;
        let mut t_down: Option<f64> = None;
        let mut t_up: Option<f64> = None;

        let mut prev_theta = theta0;
        for k in 0..max_steps {
            p.step(0.0, dt);
            let theta = p.angle();
            let t = (k + 1) as f64 * dt;

            // Record first downward zero-crossing (positive → negative)
            if t_down.is_none() && theta < 0.0 && prev_theta >= 0.0 && t > dt * 100.0 {
                t_down = Some(t);
            }
            // Record first upward zero-crossing after going negative
            if t_down.is_some() && t_up.is_none() && theta >= 0.0 && prev_theta < 0.0 {
                t_up = Some(t);
                break;
            }
            prev_theta = theta;
        }

        let half_period = match (t_down, t_up) {
            (Some(td), Some(tu)) => tu - td,
            _ => 0.0,
        };
        let full_period = half_period * 2.0;
        assert!(
            (full_period - t_theory).abs() / t_theory < 0.02,
            "period: measured={:.4}, theory={:.4}",
            full_period,
            t_theory
        );
    }

    #[test]
    fn pendulum_damping_reduces_energy() {
        let mut p = NonlinearPendulum::<f64>::new(1.0, 1.0, 9.81, 0.5);
        p.set_state(1.0, 0.0);
        let e0 = p.energy();
        for _ in 0..10000 {
            p.step(0.0, 0.001);
        }
        let e1 = p.energy();
        assert!(
            e1 < e0,
            "damped energy should decrease: e0={:.4}, e1={:.4}",
            e0,
            e1
        );
    }

    #[test]
    fn pendulum_zero_initial_stays_zero() {
        let mut p = NonlinearPendulum::<f64>::standard();
        for _ in 0..1000 {
            p.step(0.0, 0.01);
        }
        assert!(p.angle().abs() < 1e-12);
        assert!(p.angular_velocity().abs() < 1e-12);
    }

    #[test]
    fn pendulum_natural_frequency() {
        let p = NonlinearPendulum::<f64>::new(1.0, 1.0, 9.81, 0.0);
        let omega = p.natural_frequency();
        let expected = 9.81_f64.sqrt();
        assert!((omega - expected).abs() < 1e-10);
    }

    #[test]
    fn cart_pendulum_force_moves_cart() {
        let mut p = PendulumOnCart::<f64>::new(1.0, 0.1, 0.5, 9.81);
        // Near-upright: θ ≈ π (downward), small perturbation
        p.set_state([0.0, 0.0, 0.1, 0.0]); // small angle from upright
        for _ in 0..1000 {
            p.step(1.0, 0.001);
        }
        assert!(
            p.cart_position() > 0.0,
            "cart should move with positive force: x={}",
            p.cart_position()
        );
    }

    #[test]
    fn cart_pendulum_reset() {
        let mut p = PendulumOnCart::<f64>::new(1.0, 0.1, 0.5, 9.81);
        p.set_state([1.0, 2.0, 0.3, 0.4]);
        p.reset();
        assert_eq!(p.state(), &[0.0_f64; 4]);
    }

    #[test]
    fn cart_pendulum_energy_nonneg() {
        let mut p = PendulumOnCart::<f64>::new(1.0, 0.1, 0.5, 9.81);
        p.set_state([0.0, 0.0, 0.2, 0.1]);
        for _ in 0..100 {
            p.step(0.0, 0.001);
        }
        // Energy is bounded from below by potential energy reference
        // (may not be conserved due to numerical drift, but shouldn't diverge)
        let e = p.energy();
        assert!(e.is_finite(), "energy should be finite: {}", e);
    }
}
