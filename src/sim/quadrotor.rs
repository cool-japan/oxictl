/// Quadrotor UAV rigid body dynamics (6-DOF).
///
/// State vector (12 components):
///   [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
///    0  1  2  3   4   5   6     7      8    9  10 11
///
/// where (x,y,z) = position in NED inertial frame (m),
///       (vx,vy,vz) = linear velocity in inertial frame (m/s),
///       (roll,pitch,yaw) = ZYX Euler angles (rad),
///       (p,q,r) = body-frame angular rates (rad/s).
///
/// Control vector (4 components):
///   [Ω1², Ω2², Ω3², Ω4²]  — squared rotor speeds (rad²/s²)
///
/// Thrust and torque models:
///   T_i  = kT · Ωi²          (thrust from rotor i)
///   Q_i  = kQ · Ωi²          (reaction torque from rotor i)
///
/// Newton–Euler equations (body frame):
///   Translation (inertial):  m · v̇ = R · [0, 0, T_total]ᵀ + [0, 0, mg]ᵀ
///   Rotation (body):         J · ω̇ = τ_body - ω × (J·ω)
///
/// Rotor layout (X-frame, viewed from above, CCW positive):
///   1(front-left) ↔ 3(back-right) spin CW   → negative yaw torque
///   2(front-right)↔ 4(back-left)  spin CCW  → positive yaw torque
///
///   τ_roll  = kT · l · (Ω4² - Ω2²)
///   τ_pitch = kT · l · (Ω3² - Ω1²)
///   τ_yaw   = kQ · (Ω1² - Ω2² + Ω3² - Ω4²)
///
/// Integration: 4th-order Runge-Kutta with timestep dt.
use crate::core::scalar::ControlScalar;

/// Physical parameters for a quadrotor UAV.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadrotorParams<S: ControlScalar> {
    /// Total mass (kg).
    pub mass: S,
    /// Body-frame moment of inertia about x-axis (kg·m²).
    pub ixx: S,
    /// Body-frame moment of inertia about y-axis (kg·m²).
    pub iyy: S,
    /// Body-frame moment of inertia about z-axis (kg·m²).
    pub izz: S,
    /// Motor-to-center arm length (m).
    pub arm_length: S,
    /// Thrust coefficient: T = kT · Ω² (N / (rad²/s²)).
    pub kt: S,
    /// Torque coefficient: Q = kQ · Ω² (N·m / (rad²/s²)).
    pub kq: S,
    /// Gravitational acceleration (m/s²), positive downward.
    pub gravity: S,
}

impl<S: ControlScalar> QuadrotorParams<S> {
    /// Create parameters with validation.
    ///
    /// Returns `Err` if any parameter is non-positive.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mass: S,
        ixx: S,
        iyy: S,
        izz: S,
        arm_length: S,
        kt: S,
        kq: S,
        gravity: S,
    ) -> Result<Self, QuadrotorError> {
        if mass <= S::ZERO {
            return Err(QuadrotorError::InvalidParameter("mass must be positive"));
        }
        if ixx <= S::ZERO || iyy <= S::ZERO || izz <= S::ZERO {
            return Err(QuadrotorError::InvalidParameter(
                "inertia components must be positive",
            ));
        }
        if arm_length <= S::ZERO {
            return Err(QuadrotorError::InvalidParameter(
                "arm_length must be positive",
            ));
        }
        if kt <= S::ZERO {
            return Err(QuadrotorError::InvalidParameter("kt must be positive"));
        }
        if kq <= S::ZERO {
            return Err(QuadrotorError::InvalidParameter("kq must be positive"));
        }
        if gravity <= S::ZERO {
            return Err(QuadrotorError::InvalidParameter("gravity must be positive"));
        }
        Ok(Self {
            mass,
            ixx,
            iyy,
            izz,
            arm_length,
            kt,
            kq,
            gravity,
        })
    }

    /// Standard 250mm-class quadrotor parameters.
    ///
    /// mass=0.5 kg, Ixx=Iyy=4e-3 kg·m², Izz=8e-3 kg·m²,
    /// l=0.12 m, kT=1.5e-5, kQ=3e-7, g=9.81.
    pub fn standard() -> Self {
        Self {
            mass: S::from_f64(0.5),
            ixx: S::from_f64(4.0e-3),
            iyy: S::from_f64(4.0e-3),
            izz: S::from_f64(8.0e-3),
            arm_length: S::from_f64(0.12),
            kt: S::from_f64(1.5e-5),
            kq: S::from_f64(3.0e-7),
            gravity: S::from_f64(9.81),
        }
    }

    /// Hover equilibrium rotor speed squared for a single rotor (rad²/s²).
    ///
    /// At hover: T_total = m·g, so 4·kT·Ω² = m·g → Ω² = m·g / (4·kT).
    pub fn hover_omega_sq(&self) -> S {
        self.mass * self.gravity / (S::from_f64(4.0) * self.kt)
    }
}

/// Full 12-DOF quadrotor state.
#[derive(Debug, Clone, Copy)]
pub struct QuadrotorState<S: ControlScalar> {
    /// Inertial position: x, y, z (m).
    pub x: S,
    pub y: S,
    pub z: S,
    /// Inertial velocity: vx, vy, vz (m/s).
    pub vx: S,
    pub vy: S,
    pub vz: S,
    /// ZYX Euler angles: roll (φ), pitch (θ), yaw (ψ) (rad).
    pub roll: S,
    pub pitch: S,
    pub yaw: S,
    /// Body-frame angular rates: p (roll rate), q (pitch rate), r (yaw rate) (rad/s).
    pub p: S,
    pub q: S,
    pub r: S,
}

impl<S: ControlScalar> Default for QuadrotorState<S> {
    fn default() -> Self {
        Self {
            x: S::ZERO,
            y: S::ZERO,
            z: S::ZERO,
            vx: S::ZERO,
            vy: S::ZERO,
            vz: S::ZERO,
            roll: S::ZERO,
            pitch: S::ZERO,
            yaw: S::ZERO,
            p: S::ZERO,
            q: S::ZERO,
            r: S::ZERO,
        }
    }
}

impl<S: ControlScalar> QuadrotorState<S> {
    /// Convert to 12-element array [x,y,z, vx,vy,vz, roll,pitch,yaw, p,q,r].
    pub fn to_array(&self) -> [S; 12] {
        [
            self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw,
            self.p, self.q, self.r,
        ]
    }

    /// Construct from a 12-element array.
    pub fn from_array(arr: &[S; 12]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
            vx: arr[3],
            vy: arr[4],
            vz: arr[5],
            roll: arr[6],
            pitch: arr[7],
            yaw: arr[8],
            p: arr[9],
            q: arr[10],
            r: arr[11],
        }
    }
}

/// Errors from the quadrotor plant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadrotorError {
    /// A physical parameter has an invalid value.
    InvalidParameter(&'static str),
    /// Numerical singularity during integration (e.g., gimbal lock region).
    NumericalSingularity,
}

impl core::fmt::Display for QuadrotorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            Self::NumericalSingularity => write!(f, "Numerical singularity in quadrotor dynamics"),
        }
    }
}

/// Quadrotor plant with full 6-DOF Newton–Euler rigid body dynamics.
///
/// Integrates with RK4 at each call to `step`.
#[derive(Debug, Clone, Copy)]
pub struct QuadrotorPlant<S: ControlScalar> {
    params: QuadrotorParams<S>,
    state: QuadrotorState<S>,
}

impl<S: ControlScalar> QuadrotorPlant<S> {
    /// Create a new quadrotor plant at the origin with zero velocity.
    pub fn new(params: QuadrotorParams<S>) -> Self {
        Self {
            params,
            state: QuadrotorState::default(),
        }
    }

    /// Current state.
    pub fn state(&self) -> &QuadrotorState<S> {
        &self.state
    }

    /// Set state directly.
    pub fn set_state(&mut self, state: QuadrotorState<S>) {
        self.state = state;
    }

    /// Reset to default (zeroed) state.
    pub fn reset(&mut self) {
        self.state = QuadrotorState::default();
    }

    /// Physical parameters.
    pub fn params(&self) -> &QuadrotorParams<S> {
        &self.params
    }

    /// Compute the ZYX rotation matrix R mapping body→inertial frame.
    ///
    /// Conventions: roll=φ, pitch=θ, yaw=ψ.
    ///   R = Rz(ψ)·Ry(θ)·Rx(φ)
    ///
    /// Returns the 3×3 matrix as [[S;3];3] (row-major).
    fn rotation_matrix(roll: S, pitch: S, yaw: S) -> [[S; 3]; 3] {
        let (sr, cr) = (roll.sin(), roll.cos());
        let (sp, cp) = (pitch.sin(), pitch.cos());
        let (sy, cy) = (yaw.sin(), yaw.cos());

        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    }

    /// Compute state derivatives for the given state array and control input.
    ///
    /// Returns `Err(NumericalSingularity)` if `cos(pitch)` is near zero.
    fn derivatives(&self, s: &[S; 12], u: &[S; 4]) -> Result<[S; 12], QuadrotorError> {
        let vx = s[3];
        let vy = s[4];
        let vz = s[5];
        let roll = s[6];
        let pitch = s[7];
        let yaw = s[8];
        let p = s[9];
        let q = s[10];
        let r = s[11];

        let p_params = &self.params;
        let kt = p_params.kt;
        let kq = p_params.kq;
        let l = p_params.arm_length;
        let m = p_params.mass;
        let g = p_params.gravity;
        let ixx = p_params.ixx;
        let iyy = p_params.iyy;
        let izz = p_params.izz;

        // Individual thrust and total thrust
        let t1 = kt * u[0];
        let t2 = kt * u[1];
        let t3 = kt * u[2];
        let t4 = kt * u[3];
        let t_total = t1 + t2 + t3 + t4;

        // Body torques from rotors (X-frame layout):
        //   τ_roll  = l·kT·(Ω4² - Ω2²)
        //   τ_pitch = l·kT·(Ω3² - Ω1²)
        //   τ_yaw   = kQ·(Ω1² - Ω2² + Ω3² - Ω4²)
        let tau_roll = l * kt * (u[3] - u[1]);
        let tau_pitch = l * kt * (u[2] - u[0]);
        let tau_yaw = kq * (u[0] - u[1] + u[2] - u[3]);

        // Rotation matrix R (body → inertial)
        let rot = Self::rotation_matrix(roll, pitch, yaw);

        // Accelerations in inertial frame: a = R·[0,0,T_total]/m - [0,0,g]
        // (z is positive upward in our sign convention; gravity acts in -z)
        let ax = (rot[0][2] * t_total) / m;
        let ay = (rot[1][2] * t_total) / m;
        let az = (rot[2][2] * t_total) / m - g;

        // Angular acceleration (body frame Euler equations):
        //   J·ω̇ = τ - ω × (J·ω)
        // ω × (J·ω) = [q·Izz·r - r·Iyy·q, r·Ixx·p - p·Izz·r, p·Iyy·q - q·Ixx·p]
        let gyro_x = q * izz * r - r * iyy * q;
        let gyro_y = r * ixx * p - p * izz * r;
        let gyro_z = p * iyy * q - q * ixx * p;

        let p_dot = (tau_roll - gyro_x) / ixx;
        let q_dot = (tau_pitch - gyro_y) / iyy;
        let r_dot = (tau_yaw - gyro_z) / izz;

        // Euler angle kinematics (ZYX convention):
        //   [φ̇]   [1  sin(φ)·tan(θ)  cos(φ)·tan(θ)] [p]
        //   [θ̇] = [0  cos(φ)          -sin(φ)        ] [q]
        //   [ψ̇]   [0  sin(φ)/cos(θ)   cos(φ)/cos(θ) ] [r]
        let cp = pitch.cos();
        if cp.abs() < S::from_f64(1e-6) {
            return Err(QuadrotorError::NumericalSingularity);
        }
        let tp = pitch.sin() / cp;
        let sr = roll.sin();
        let cr = roll.cos();

        let roll_dot = p + (sr * tp) * q + (cr * tp) * r;
        let pitch_dot = cr * q + (-sr) * r;
        let yaw_dot = (sr / cp) * q + (cr / cp) * r;

        Ok([
            vx, vy, vz, ax, ay, az, roll_dot, pitch_dot, yaw_dot, p_dot, q_dot, r_dot,
        ])
    }

    /// Advance the simulation by `dt` seconds under squared rotor speeds `u`.
    ///
    /// Uses 4th-order Runge-Kutta integration.
    ///
    /// # Errors
    /// Returns `Err(QuadrotorError::NumericalSingularity)` if pitch angle
    /// is within 1e-6 rad of ±90°.
    pub fn step(&mut self, u: &[S; 4], dt: S) -> Result<(), QuadrotorError> {
        let s = self.state.to_array();
        let half = S::HALF;
        let two = S::TWO;
        let sixth = S::ONE / S::from_f64(6.0);

        let k1 = self.derivatives(&s, u)?;

        let s2: [S; 12] = core::array::from_fn(|i| s[i] + half * dt * k1[i]);
        let k2 = self.derivatives(&s2, u)?;

        let s3: [S; 12] = core::array::from_fn(|i| s[i] + half * dt * k2[i]);
        let k3 = self.derivatives(&s3, u)?;

        let s4: [S; 12] = core::array::from_fn(|i| s[i] + dt * k3[i]);
        let k4 = self.derivatives(&s4, u)?;

        let new_s: [S; 12] = core::array::from_fn(|i| {
            s[i] + sixth * dt * (k1[i] + two * k2[i] + two * k3[i] + k4[i])
        });

        self.state = QuadrotorState::from_array(&new_s);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At hover, all rotors spin at the same hover speed.
    /// Net thrust ≈ mg, net torque ≈ 0. After N steps the vehicle
    /// should remain at constant altitude (z should not change).
    #[test]
    fn hover_equilibrium_altitude_constant() {
        let params = QuadrotorParams::standard();
        let omega_sq = params.hover_omega_sq();
        let mut plant = QuadrotorPlant::new(params);

        let u = [omega_sq; 4];
        let dt = 0.001_f64;
        let z0 = plant.state().z;
        for _ in 0..500 {
            plant.step(&u, dt).expect("step should succeed");
        }
        // Altitude should be within 1 mm of original after 0.5 s
        let z1 = plant.state().z;
        assert!(
            (z1 - z0).abs() < 1e-3,
            "altitude should be stable at hover: z0={:.6}, z1={:.6}",
            z0,
            z1
        );
    }

    /// A positive roll torque input (Ω4² > Ω2²) should produce positive roll
    /// and cause rightward (positive y) drift.
    #[test]
    fn positive_roll_torque_tilts_right() {
        let params = QuadrotorParams::standard();
        let omega_sq = params.hover_omega_sq();
        let mut plant = QuadrotorPlant::new(params);

        // Increase rotor 4 (back-left → roll right) slightly above hover
        let delta = omega_sq * 0.1_f64;
        let u = [omega_sq, omega_sq, omega_sq, omega_sq + delta];
        let dt = 0.001_f64;
        for _ in 0..200 {
            plant.step(&u, dt).expect("step should succeed");
        }
        assert!(
            plant.state().roll > 0.0,
            "positive roll torque should produce positive roll: roll={}",
            plant.state().roll
        );
    }

    /// Verify that params validation rejects invalid mass.
    #[test]
    fn params_rejects_nonpositive_mass() {
        let result = QuadrotorParams::<f64>::new(-1.0, 4e-3, 4e-3, 8e-3, 0.12, 1.5e-5, 3e-7, 9.81);
        assert!(
            result == Err(QuadrotorError::InvalidParameter("mass must be positive")),
            "expected invalid-mass error"
        );
    }

    /// Verify hover omega squared formula: 4·kT·Ω² = m·g.
    #[test]
    fn hover_omega_sq_formula() {
        let params = QuadrotorParams::<f64>::standard();
        let omega_sq = params.hover_omega_sq();
        let thrust = 4.0_f64 * params.kt * omega_sq;
        let weight = params.mass * params.gravity;
        assert!(
            (thrust - weight).abs() < 1e-10,
            "hover thrust should equal weight: thrust={:.6}, weight={:.6}",
            thrust,
            weight
        );
    }

    /// Rotation matrix should be orthogonal (R^T R = I).
    #[test]
    fn rotation_matrix_is_orthogonal() {
        let roll = 0.3_f64;
        let pitch = 0.2_f64;
        let yaw = 1.1_f64;
        let r = QuadrotorPlant::<f64>::rotation_matrix(roll, pitch, yaw);

        // Compute R^T * R
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0_f64;
                for row in &r {
                    dot += row[i] * row[j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "R^T R [{},{}] = {:.2e} ≠ {:.1}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }
}
