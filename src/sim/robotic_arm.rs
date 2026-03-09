/// 2-DOF planar robotic arm plant (Euler–Lagrange dynamics).
///
/// Configuration: two rigid links connected by revolute joints, operating
/// in the horizontal plane (no gravitational effect on torques) or
/// vertical plane (gravity included via G(q) vector).
///
/// State: [q1, q̇1, q2, q̇2]
///   q1  : joint 1 angle (rad), measured from inertial x-axis
///   q̇1  : joint 1 angular velocity (rad/s)
///   q2  : joint 2 angle (rad), relative to link 1 frame
///   q̇2  : joint 2 angular velocity (rad/s)
///
/// Control: [τ1, τ2] (joint torques, N·m).
///
/// Equations of motion:
///   M(q)·q̈ + C(q, q̇)·q̇ + G(q) = τ
///
/// where:
///   M(q) is the 2×2 symmetric positive-definite mass matrix,
///   C(q, q̇) is the 2×2 Coriolis/centripetal matrix,
///   G(q) is the 2×1 gravitational torque vector.
///
/// Solved as: q̈ = M(q)⁻¹·(τ - C(q,q̇)·q̇ - G(q))
///
/// Integration: 4th-order Runge-Kutta.
///
/// Parameters follow the standard robot dynamics textbook conventions:
///   l1, l2 : link lengths (m)
///   lc1, lc2: distance from joint to CoM of each link (m)
///   m1, m2 : link masses (kg)
///   I1, I2 : link moments of inertia about joint axes (kg·m²)
///   g      : gravitational acceleration (m/s²), 0 for horizontal plane
use crate::core::scalar::ControlScalar;

/// Physical parameters of the 2-DOF planar arm.
#[derive(Debug, Clone, Copy)]
pub struct RoboticArmParams<S: ControlScalar> {
    /// Length of link 1 (m).
    pub link1_length: S,
    /// Length of link 2 (m).
    pub link2_length: S,
    /// Distance from joint 1 to CoM of link 1 (m).
    pub link1_com: S,
    /// Distance from joint 2 to CoM of link 2 (m).
    pub link2_com: S,
    /// Mass of link 1 (kg).
    pub link1_mass: S,
    /// Mass of link 2 (kg).
    pub link2_mass: S,
    /// Moment of inertia of link 1 about joint 1 axis (kg·m²).
    pub link1_inertia: S,
    /// Moment of inertia of link 2 about joint 2 axis (kg·m²).
    pub link2_inertia: S,
    /// Gravitational acceleration (m/s²); set to 0 for horizontal plane.
    pub gravity: S,
}

impl<S: ControlScalar> RoboticArmParams<S> {
    /// Construct with validation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        link1_length: S,
        link2_length: S,
        link1_com: S,
        link2_com: S,
        link1_mass: S,
        link2_mass: S,
        link1_inertia: S,
        link2_inertia: S,
        gravity: S,
    ) -> Result<Self, RoboticArmError> {
        if link1_length <= S::ZERO {
            return Err(RoboticArmError::InvalidParameter(
                "link1_length must be positive",
            ));
        }
        if link2_length <= S::ZERO {
            return Err(RoboticArmError::InvalidParameter(
                "link2_length must be positive",
            ));
        }
        if link1_com <= S::ZERO || link1_com > link1_length {
            return Err(RoboticArmError::InvalidParameter(
                "link1_com must be in (0, link1_length]",
            ));
        }
        if link2_com <= S::ZERO || link2_com > link2_length {
            return Err(RoboticArmError::InvalidParameter(
                "link2_com must be in (0, link2_length]",
            ));
        }
        if link1_mass <= S::ZERO {
            return Err(RoboticArmError::InvalidParameter(
                "link1_mass must be positive",
            ));
        }
        if link2_mass <= S::ZERO {
            return Err(RoboticArmError::InvalidParameter(
                "link2_mass must be positive",
            ));
        }
        if link1_inertia <= S::ZERO {
            return Err(RoboticArmError::InvalidParameter(
                "link1_inertia must be positive",
            ));
        }
        if link2_inertia <= S::ZERO {
            return Err(RoboticArmError::InvalidParameter(
                "link2_inertia must be positive",
            ));
        }
        if gravity < S::ZERO {
            return Err(RoboticArmError::InvalidParameter(
                "gravity must be non-negative",
            ));
        }
        Ok(Self {
            link1_length,
            link2_length,
            link1_com,
            link2_com,
            link1_mass,
            link2_mass,
            link1_inertia,
            link2_inertia,
            gravity,
        })
    }

    /// Standard 2-DOF SCARA-like arm: unit links, uniform mass distribution,
    /// vertical plane (g = 9.81).
    pub fn standard() -> Self {
        let l = S::ONE;
        let lc = S::HALF;
        let m = S::ONE;
        // Point mass inertia about joint: I = m·l²/3  (uniform rod)
        let inertia = S::from_f64(1.0 / 3.0);
        Self {
            link1_length: l,
            link2_length: l,
            link1_com: lc,
            link2_com: lc,
            link1_mass: m,
            link2_mass: m,
            link1_inertia: inertia,
            link2_inertia: inertia,
            gravity: S::from_f64(9.81),
        }
    }
}

/// State of the robotic arm.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoboticArmState<S: ControlScalar> {
    /// Joint 1 angle (rad).
    pub q1: S,
    /// Joint 1 velocity (rad/s).
    pub q1_dot: S,
    /// Joint 2 angle (rad).
    pub q2: S,
    /// Joint 2 velocity (rad/s).
    pub q2_dot: S,
}

impl<S: ControlScalar> RoboticArmState<S> {
    pub fn to_array(&self) -> [S; 4] {
        [self.q1, self.q1_dot, self.q2, self.q2_dot]
    }

    pub fn from_array(a: &[S; 4]) -> Self {
        Self {
            q1: a[0],
            q1_dot: a[1],
            q2: a[2],
            q2_dot: a[3],
        }
    }
}

/// Errors from the robotic arm plant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoboticArmError {
    /// A physical parameter is invalid.
    InvalidParameter(&'static str),
    /// The mass matrix is singular (cannot invert).
    SingularMassMatrix,
}

impl core::fmt::Display for RoboticArmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            Self::SingularMassMatrix => write!(f, "Singular mass matrix in robotic arm"),
        }
    }
}

/// 2-DOF planar robotic arm plant.
#[derive(Debug, Clone, Copy)]
pub struct RoboticArmPlant<S: ControlScalar> {
    params: RoboticArmParams<S>,
    state: RoboticArmState<S>,
}

impl<S: ControlScalar> RoboticArmPlant<S> {
    /// Construct with given parameters, state initialised to zeros.
    pub fn new(params: RoboticArmParams<S>) -> Self {
        Self {
            params,
            state: RoboticArmState::default(),
        }
    }

    /// Current state.
    pub fn state(&self) -> &RoboticArmState<S> {
        &self.state
    }

    /// Set state directly.
    pub fn set_state(&mut self, state: RoboticArmState<S>) {
        self.state = state;
    }

    /// Reset to zero state.
    pub fn reset(&mut self) {
        self.state = RoboticArmState::default();
    }

    /// Physical parameters.
    pub fn params(&self) -> &RoboticArmParams<S> {
        &self.params
    }

    /// Compute the 2×2 inertia matrix M(q).
    ///
    /// Standard manipulator dynamics for 2-link planar arm:
    ///   α = I1 + I2 + m1·lc1² + m2·(l1² + lc2²)
    ///   β = m2·l1·lc2
    ///   δ = I2 + m2·lc2²
    ///
    ///   M = [[α + 2β·cos(q2),  δ + β·cos(q2)],
    ///        [δ + β·cos(q2),   δ             ]]
    fn mass_matrix(&self, q2: S) -> [[S; 2]; 2] {
        let p = &self.params;
        let alpha = p.link1_inertia
            + p.link2_inertia
            + p.link1_mass * p.link1_com * p.link1_com
            + p.link2_mass * (p.link1_length * p.link1_length + p.link2_com * p.link2_com);
        let beta = p.link2_mass * p.link1_length * p.link2_com;
        let delta = p.link2_inertia + p.link2_mass * p.link2_com * p.link2_com;

        let cos_q2 = q2.cos();
        let m11 = alpha + S::TWO * beta * cos_q2;
        let m12 = delta + beta * cos_q2;
        let m21 = m12;
        let m22 = delta;
        [[m11, m12], [m21, m22]]
    }

    /// Invert a 2×2 matrix. Returns `None` if singular.
    fn invert_2x2(m: &[[S; 2]; 2]) -> Option<[[S; 2]; 2]> {
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if det.abs() < S::EPSILON * S::from_f64(1e6) {
            return None;
        }
        let inv_det = S::ONE / det;
        Some([
            [m[1][1] * inv_det, -m[0][1] * inv_det],
            [-m[1][0] * inv_det, m[0][0] * inv_det],
        ])
    }

    /// Compute state derivatives given state array and joint torques.
    ///
    /// Returns `Err(SingularMassMatrix)` if M(q) is not invertible.
    fn derivatives(&self, s: &[S; 4], tau: &[S; 2]) -> Result<[S; 4], RoboticArmError> {
        let q1_dot = s[1];
        let q2 = s[2];
        let q2_dot = s[3];

        let p = &self.params;

        let beta = p.link2_mass * p.link1_length * p.link2_com;
        let sin_q2 = q2.sin();

        // Coriolis / centripetal matrix C(q, q̇):
        //   h = -β·sin(q2)
        //
        //   C = [h·q̇2,  h·(q̇1 + q̇2)]
        //       [-h·q̇1, 0            ]
        let h = -beta * sin_q2;
        let c11 = h * q2_dot;
        let c12 = h * (q1_dot + q2_dot);
        let c21 = -h * q1_dot;
        let c22 = S::ZERO;

        // Gravity vector G(q) (vertical-plane arm, x-axis is horizontal):
        //   G1 = (m1·lc1 + m2·l1)·g·cos(q1) + m2·lc2·g·cos(q1+q2)
        //   G2 = m2·lc2·g·cos(q1+q2)
        let q1 = s[0];
        let g = p.gravity;
        let cos_q1 = q1.cos();
        let cos_q1_q2 = (q1 + q2).cos();

        let g1 = (p.link1_mass * p.link1_com + p.link2_mass * p.link1_length) * g * cos_q1
            + p.link2_mass * p.link2_com * g * cos_q1_q2;
        let g2 = p.link2_mass * p.link2_com * g * cos_q1_q2;

        // τ_eff = τ - C·q̇ - G
        let coriolis_1 = c11 * q1_dot + c12 * q2_dot;
        let coriolis_2 = c21 * q1_dot + c22 * q2_dot;
        let tau_eff = [tau[0] - coriolis_1 - g1, tau[1] - coriolis_2 - g2];

        // q̈ = M⁻¹ · τ_eff
        let m = self.mass_matrix(q2);
        let m_inv = Self::invert_2x2(&m).ok_or(RoboticArmError::SingularMassMatrix)?;

        let q1_ddot = m_inv[0][0] * tau_eff[0] + m_inv[0][1] * tau_eff[1];
        let q2_ddot = m_inv[1][0] * tau_eff[0] + m_inv[1][1] * tau_eff[1];

        Ok([q1_dot, q1_ddot, q2_dot, q2_ddot])
    }

    /// Advance the simulation one step of `dt` seconds with joint torques `tau`.
    ///
    /// Uses 4th-order Runge-Kutta integration.
    pub fn step(&mut self, tau: &[S; 2], dt: S) -> Result<(), RoboticArmError> {
        let s = self.state.to_array();
        let half = S::HALF;
        let two = S::TWO;
        let sixth = S::ONE / S::from_f64(6.0);

        let k1 = self.derivatives(&s, tau)?;

        let s2: [S; 4] = core::array::from_fn(|i| s[i] + half * dt * k1[i]);
        let k2 = self.derivatives(&s2, tau)?;

        let s3: [S; 4] = core::array::from_fn(|i| s[i] + half * dt * k2[i]);
        let k3 = self.derivatives(&s3, tau)?;

        let s4: [S; 4] = core::array::from_fn(|i| s[i] + dt * k3[i]);
        let k4 = self.derivatives(&s4, tau)?;

        let new_s: [S; 4] = core::array::from_fn(|i| {
            s[i] + sixth * dt * (k1[i] + two * k2[i] + two * k3[i] + k4[i])
        });

        self.state = RoboticArmState::from_array(&new_s);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Static equilibrium in the vertical plane: at q1=0, q2=0 (both links horizontal),
    /// the required torques to hold the arm static (q̇=0, q̈=0) equal the gravity vector G.
    /// Verify: if we apply those exact torques, derivatives are zero.
    ///
    /// The full G vector for a 2-link arm at q1=0, q2=0 (cos(q1)=1, cos(q1+q2)=1):
    ///   G1 = (m1·lc1 + m2·l1)·g·cos(q1) + m2·lc2·g·cos(q1+q2)
    ///   G2 = m2·lc2·g·cos(q1+q2)
    #[test]
    fn static_equilibrium_horizontal() {
        let params = RoboticArmParams::standard();
        let plant = RoboticArmPlant::new(params);

        // At q1=0, q2=0 (horizontal), q̇=0:
        let p = &params;
        let g = p.gravity;
        let cos_q1 = 1.0_f64; // cos(0)
        let cos_q1q2 = 1.0_f64; // cos(0+0)

        // Full gravity torque (two terms in G1)
        let g1 = (p.link1_mass * p.link1_com + p.link2_mass * p.link1_length) * g * cos_q1
            + p.link2_mass * p.link2_com * g * cos_q1q2;
        let g2 = p.link2_mass * p.link2_com * g * cos_q1q2;

        let s = [0.0_f64, 0.0, 0.0, 0.0];
        let tau = [g1, g2];
        let deriv = plant.derivatives(&s, &tau).expect("should not fail");

        // At static equilibrium with gravity-compensating torques, all derivatives are zero
        for (i, &d) in deriv.iter().enumerate() {
            assert!(
                d.abs() < 1e-10,
                "derivative[{}] = {} should be zero at static equilibrium",
                i,
                d
            );
        }
    }

    /// Positive torque on joint 1 with no gravity (horizontal plane)
    /// should accelerate joint 1 in the positive direction.
    #[test]
    fn positive_tau1_increases_q1_dot() {
        let params = RoboticArmParams::new(1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0 / 3.0, 1.0 / 3.0, 0.0)
            .expect("params ok");
        let mut plant = RoboticArmPlant::new(params);

        let tau = [1.0_f64, 0.0];
        let dt = 0.001_f64;
        for _ in 0..200 {
            plant.step(&tau, dt).expect("step ok");
        }
        assert!(
            plant.state().q1_dot > 0.0,
            "positive τ1 should increase q̇1: got {}",
            plant.state().q1_dot
        );
    }

    /// The mass matrix M(q) must be positive definite for all configurations.
    /// Check the determinant at several q2 values.
    #[test]
    fn mass_matrix_positive_definite() {
        let params = RoboticArmParams::standard();
        let plant = RoboticArmPlant::new(params);

        let q2_values = [0.0_f64, 0.3, 0.9, 1.5, -0.5, -1.2];
        for q2 in q2_values {
            let m = plant.mass_matrix(q2);
            let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            assert!(det > 0.0, "mass matrix not PD at q2={}: det={}", q2, det);
            assert!(
                m[0][0] > 0.0,
                "M[0][0] must be positive at q2={}: got {}",
                q2,
                m[0][0]
            );
        }
    }

    /// Invalid parameters should be rejected.
    #[test]
    fn invalid_params_rejected() {
        assert!(
            RoboticArmParams::<f64>::new(-1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 9.81).is_err()
        );
        assert!(
            RoboticArmParams::<f64>::new(1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 9.81).is_err()
        );
    }
}
