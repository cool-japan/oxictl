/// Cart-pole (inverted pendulum on a cart) plant.
///
/// State: [x, ẋ, θ, θ̇]
///   x     : cart position (m)
///   ẋ     : cart velocity (m/s)
///   θ     : pole angle from **upright** vertical (rad), θ=0 is upright
///   θ̇     : pole angular velocity (rad/s)
///
/// Control: scalar force F applied horizontally to the cart (N).
///
/// Equations of motion (Lagrangian derivation, standard form):
///
///   (M + m)·ẍ + m·l·θ̈·cos(θ) - m·l·θ̇²·sin(θ) = F
///   (m·l²)·θ̈ + m·l·ẍ·cos(θ) - m·g·l·sin(θ) = 0
///
/// Solving for ẍ and θ̈ simultaneously:
///
///   det = (M + m)·m·l² - (m·l·cos θ)² = m·l²·(M + m·sin²θ)
///
///   ẍ   = [m·l(g·sin θ·cos θ - θ̇²·l·sin θ) + F·m·l²/m] / det ... simplified:
///         = [F·m·l² + m·l·(m·l·θ̇²·sin θ - m·g·sin θ·cos θ)] / (m·l²·(M+m·sin²θ)) -- incorrect form
///
///   Using Cramer's rule on the 2×2 system:
///     [M+m,      m·l·cos θ] [ẍ  ]   [F + m·l·θ̇²·sin θ]
///     [m·l·cos θ, m·l²    ] [θ̈  ] = [m·g·l·sin θ       ]
///
///   det = (M+m)·m·l² - m²·l²·cos²θ = m·l²·(M + m·sin²θ)
///
///   ẍ = [ m·l²·(F + m·l·θ̇²·sin θ) - m·l·cos θ·m·g·l·sin θ ] / det
///   θ̈ = [ (M+m)·m·g·l·sin θ - m·l·cos θ·(F + m·l·θ̇²·sin θ) ] / det
///
/// Integration: 4th-order Runge-Kutta.
use crate::core::scalar::ControlScalar;

/// Physical parameters of the cart-pole system.
#[derive(Debug, Clone, Copy)]
pub struct CartPoleParams<S: ControlScalar> {
    /// Cart mass M (kg).
    pub cart_mass: S,
    /// Pole (point) mass m (kg).
    pub pole_mass: S,
    /// Pole half-length l (m).
    pub pole_length: S,
    /// Gravitational acceleration g (m/s²).
    pub gravity: S,
}

impl<S: ControlScalar> CartPoleParams<S> {
    /// Construct with validation.
    ///
    /// Returns `Err` if any parameter is non-positive.
    pub fn new(
        cart_mass: S,
        pole_mass: S,
        pole_length: S,
        gravity: S,
    ) -> Result<Self, CartPoleError> {
        if cart_mass <= S::ZERO {
            return Err(CartPoleError::InvalidParameter(
                "cart_mass must be positive",
            ));
        }
        if pole_mass <= S::ZERO {
            return Err(CartPoleError::InvalidParameter(
                "pole_mass must be positive",
            ));
        }
        if pole_length <= S::ZERO {
            return Err(CartPoleError::InvalidParameter(
                "pole_length must be positive",
            ));
        }
        if gravity <= S::ZERO {
            return Err(CartPoleError::InvalidParameter("gravity must be positive"));
        }
        Ok(Self {
            cart_mass,
            pole_mass,
            pole_length,
            gravity,
        })
    }

    /// Classic OpenAI Gym parameters: M=1 kg, m=0.1 kg, l=0.5 m, g=9.8 m/s².
    pub fn gym_default() -> Self {
        Self {
            cart_mass: S::from_f64(1.0),
            pole_mass: S::from_f64(0.1),
            pole_length: S::from_f64(0.5),
            gravity: S::from_f64(9.8),
        }
    }
}

/// Errors from the cart-pole plant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartPoleError {
    /// A physical parameter is invalid.
    InvalidParameter(&'static str),
    /// The mass matrix is singular (should not occur for positive masses).
    SingularMassMatrix,
}

impl core::fmt::Display for CartPoleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            Self::SingularMassMatrix => write!(f, "Singular mass matrix in cart-pole"),
        }
    }
}

/// Cart-pole state.
#[derive(Debug, Clone, Copy, Default)]
pub struct CartPoleState<S: ControlScalar> {
    /// Cart position (m).
    pub x: S,
    /// Cart velocity (m/s).
    pub x_dot: S,
    /// Pole angle from upright (rad), θ=0 is balanced.
    pub theta: S,
    /// Pole angular velocity (rad/s).
    pub theta_dot: S,
}

impl<S: ControlScalar> CartPoleState<S> {
    pub fn to_array(&self) -> [S; 4] {
        [self.x, self.x_dot, self.theta, self.theta_dot]
    }

    pub fn from_array(a: &[S; 4]) -> Self {
        Self {
            x: a[0],
            x_dot: a[1],
            theta: a[2],
            theta_dot: a[3],
        }
    }
}

/// Cart-pole plant.
#[derive(Debug, Clone, Copy)]
pub struct CartPolePlant<S: ControlScalar> {
    params: CartPoleParams<S>,
    state: CartPoleState<S>,
}

impl<S: ControlScalar> CartPolePlant<S> {
    /// Construct with given parameters, state initialised to zeros.
    pub fn new(params: CartPoleParams<S>) -> Self {
        Self {
            params,
            state: CartPoleState::default(),
        }
    }

    /// Current state.
    pub fn state(&self) -> &CartPoleState<S> {
        &self.state
    }

    /// Set state directly.
    pub fn set_state(&mut self, state: CartPoleState<S>) {
        self.state = state;
    }

    /// Reset to upright equilibrium (all zeros).
    pub fn reset(&mut self) {
        self.state = CartPoleState::default();
    }

    /// Physical parameters.
    pub fn params(&self) -> &CartPoleParams<S> {
        &self.params
    }

    /// Total mechanical energy of the system (cart KE + pole KE + pole PE).
    ///
    /// Measured relative to the upright equilibrium:
    ///   E = ½·(M+m)·ẋ² + ½·m·l²·θ̇² + m·l·ẋ·θ̇·cos(θ) - m·g·l·cos(θ)
    ///
    /// (The cross-term arises from expressing bob velocity in lab frame.)
    pub fn energy(&self) -> S {
        let m = self.params.pole_mass;
        let mc = self.params.cart_mass;
        let l = self.params.pole_length;
        let g = self.params.gravity;
        let x_dot = self.state.x_dot;
        let theta = self.state.theta;
        let theta_dot = self.state.theta_dot;

        // Cart kinetic energy: ½·M·ẋ²
        let cart_ke = S::HALF * mc * x_dot * x_dot;

        // Bob velocity in lab frame:
        //   vbx = ẋ + l·θ̇·cos θ
        //   vby = l·θ̇·sin θ    (vertical)
        let vbx = x_dot + l * theta_dot * theta.cos();
        let vby = l * theta_dot * theta.sin();
        let bob_ke = S::HALF * m * (vbx * vbx + vby * vby);

        // Bob potential energy (positive upward, zero at θ=π, i.e. hanging down):
        //   PE = m·g·l·cos θ  (relative to pivot, θ=0 is upright)
        let bob_pe = m * g * l * theta.cos();

        cart_ke + bob_ke + bob_pe
    }

    /// Compute state derivatives [ẋ, ẍ, θ̇, θ̈] for the given state and force.
    fn derivatives(&self, s: &[S; 4], force: S) -> Result<[S; 4], CartPoleError> {
        let x_dot = s[1];
        let theta = s[2];
        let theta_dot = s[3];

        let m = self.params.pole_mass;
        let mc = self.params.cart_mass;
        let l = self.params.pole_length;
        let g = self.params.gravity;

        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let mt = mc + m;

        // det = m·l²·(M + m·sin²θ)  > 0 for positive masses
        let det = m * l * l * (mt - m * cos_t * cos_t);
        if det.abs() < S::EPSILON * S::from_f64(1e6) {
            return Err(CartPoleError::SingularMassMatrix);
        }

        let rhs_x = force + m * l * theta_dot * theta_dot * sin_t;
        let rhs_t = m * g * l * sin_t;

        // Cramer's rule:
        //   ẍ   = (m·l²·rhs_x  - m·l·cos θ·rhs_t) / det
        //   θ̈   = (mt·rhs_t    - m·l·cos θ·rhs_x) / det
        let x_ddot = (m * l * l * rhs_x - m * l * cos_t * rhs_t) / det;
        let theta_ddot = (mt * rhs_t - m * l * cos_t * rhs_x) / det;

        Ok([x_dot, x_ddot, theta_dot, theta_ddot])
    }

    /// Advance the simulation one step of `dt` seconds with horizontal force `force`.
    ///
    /// Uses 4th-order Runge-Kutta integration.
    ///
    /// # Errors
    /// Returns `Err(CartPoleError::SingularMassMatrix)` if the effective
    /// mass matrix becomes singular (very unusual for physical parameters).
    pub fn step(&mut self, force: S, dt: S) -> Result<(), CartPoleError> {
        let s = self.state.to_array();
        let half = S::HALF;
        let two = S::TWO;
        let sixth = S::ONE / S::from_f64(6.0);

        let k1 = self.derivatives(&s, force)?;

        let s2: [S; 4] = core::array::from_fn(|i| s[i] + half * dt * k1[i]);
        let k2 = self.derivatives(&s2, force)?;

        let s3: [S; 4] = core::array::from_fn(|i| s[i] + half * dt * k2[i]);
        let k3 = self.derivatives(&s3, force)?;

        let s4: [S; 4] = core::array::from_fn(|i| s[i] + dt * k3[i]);
        let k4 = self.derivatives(&s4, force)?;

        let new_s: [S; 4] = core::array::from_fn(|i| {
            s[i] + sixth * dt * (k1[i] + two * k2[i] + two * k3[i] + k4[i])
        });

        self.state = CartPoleState::from_array(&new_s);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At the upright equilibrium (θ=0, θ̇=0, ẋ=0) with zero force,
    /// derivatives should be zero → state should stay zero.
    #[test]
    fn zero_force_at_equilibrium_stays_put() {
        let params = CartPoleParams::gym_default();
        let plant = CartPolePlant::new(params);
        // Equilibrium is unstable; test only the algebraic equilibrium condition.
        let s = [0.0_f64; 4];
        let deriv = plant.derivatives(&s, 0.0).expect("derivatives ok");
        // All derivatives should be zero at [0,0,0,0] with F=0
        for (i, &d) in deriv.iter().enumerate() {
            assert!(
                d.abs() < 1e-12,
                "derivative[{}] = {} ≠ 0 at equilibrium",
                i,
                d
            );
        }
    }

    /// A positive force should accelerate the cart in the positive x direction.
    #[test]
    fn positive_force_moves_cart_right() {
        let params = CartPoleParams::gym_default();
        let mut plant = CartPolePlant::new(params);
        let dt = 0.001_f64;
        for _ in 0..100 {
            plant.step(10.0, dt).expect("step ok");
        }
        assert!(
            plant.state().x > 0.0,
            "positive force should move cart right: x={}",
            plant.state().x
        );
        assert!(
            plant.state().x_dot > 0.0,
            "positive force should give positive velocity: ẋ={}",
            plant.state().x_dot
        );
    }

    /// Energy should be approximately conserved when no external force is applied
    /// and the pole starts with a small angle.  The energy stays within ±0.5%
    /// of initial over 2000 steps (RK4 with dt=1e-4 is nearly symplectic).
    #[test]
    fn energy_approximately_conserved_no_force() {
        let params = CartPoleParams::gym_default();
        let mut plant = CartPolePlant::new(params);
        // Small-angle perturbation from upright
        plant.set_state(CartPoleState {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.05,
            theta_dot: 0.0,
        });
        let e0 = plant.energy();
        let dt = 1e-4_f64;
        for _ in 0..2000 {
            plant.step(0.0, dt).expect("step ok");
        }
        let e1 = plant.energy();
        let rel_err = (e1 - e0).abs() / e0.abs().max(1e-12);
        assert!(
            rel_err < 5e-3,
            "energy not conserved: e0={:.6}, e1={:.6}, rel_err={:.2e}",
            e0,
            e1,
            rel_err
        );
    }

    /// Invalid parameters should be rejected.
    #[test]
    fn invalid_params_rejected() {
        assert!(CartPoleParams::<f64>::new(-1.0, 0.1, 0.5, 9.8).is_err());
        assert!(CartPoleParams::<f64>::new(1.0, -0.1, 0.5, 9.8).is_err());
        assert!(CartPoleParams::<f64>::new(1.0, 0.1, -0.5, 9.8).is_err());
        assert!(CartPoleParams::<f64>::new(1.0, 0.1, 0.5, -9.8).is_err());
    }
}
