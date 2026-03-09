use super::so3::{rotation_error, vec3_cross, vec3_dot, vec3_norm, SO3};
/// Geometric PD controller on SO(3)/SE(3) for quadrotor UAVs.
///
/// Implements the geometric tracking controller from:
///   T. Lee, M. Leok, N.H. McClamroch, "Geometric Tracking Control of a
///   Quadrotor UAV on SE(3)", CDC 2010.
///
/// # Attitude control (SO(3))
///
/// Attitude error:          e_R = 0.5 · vee(R_dᵀ·R − Rᵀ·R_d)
/// Angular-velocity error:  e_Ω = Ω − Rᵀ·R_d·Ω_d
/// Attitude torque:         τ = −k_R·e_R − k_Ω·e_Ω + Ω × J·Ω
///
/// # Translational control (ℝ³)
///
/// Position error:     e_x = x − x_d
/// Velocity error:     e_v = v − v_d
/// Desired force:      F_des = −k_x·e_x − k_v·e_v + m·g·e₃ + m·a_d
/// Scalar thrust:      T = F_des · R·e₃
/// Desired R_d:        computed from F_des direction and yaw reference.
use crate::core::scalar::ControlScalar;

// ─── State types ──────────────────────────────────────────────────────────────

/// Full SE(3) state of the quadrotor expressed on the Lie group.
///
/// Uses SO(3) rotation matrix for the attitude — no singularity issues.
#[derive(Debug, Clone, Copy)]
pub struct QuadRotorGeomState<S: ControlScalar> {
    /// Inertial position [m].
    pub position: [S; 3],
    /// Inertial velocity [m/s].
    pub velocity: [S; 3],
    /// Body attitude R ∈ SO(3) (body-to-inertial).
    pub attitude: SO3<S>,
    /// Body-frame angular velocity Ω [rad/s].
    pub omega: [S; 3],
}

impl<S: ControlScalar> QuadRotorGeomState<S> {
    /// Zero state (identity attitude, everything else zero).
    pub fn zero() -> Self {
        Self {
            position: [S::ZERO; 3],
            velocity: [S::ZERO; 3],
            attitude: SO3::identity(),
            omega: [S::ZERO; 3],
        }
    }
}

/// Reference trajectory for the geometric controller.
#[derive(Debug, Clone, Copy)]
pub struct GeometricRef<S: ControlScalar> {
    /// Desired position [m].
    pub position: [S; 3],
    /// Desired velocity [m/s].
    pub velocity: [S; 3],
    /// Desired acceleration [m/s²].
    pub acceleration: [S; 3],
    /// Desired yaw angle [rad].
    pub yaw: S,
    /// Desired yaw rate [rad/s].
    pub yaw_rate: S,
}

impl<S: ControlScalar> GeometricRef<S> {
    /// Hover at origin with zero yaw.
    pub fn hover_at_origin() -> Self {
        Self {
            position: [S::ZERO; 3],
            velocity: [S::ZERO; 3],
            acceleration: [S::ZERO; 3],
            yaw: S::ZERO,
            yaw_rate: S::ZERO,
        }
    }
}

// ─── Controller config ────────────────────────────────────────────────────────

/// Geometric controller gains and physical parameters.
#[derive(Debug, Clone, Copy)]
pub struct GeometricConfig<S: ControlScalar> {
    /// Attitude gain k_R (error in SO(3) Lie algebra).
    pub k_r: S,
    /// Angular-velocity gain k_Ω.
    pub k_omega: S,
    /// Position gain k_x.
    pub k_x: S,
    /// Velocity gain k_v.
    pub k_v: S,
    /// Vehicle mass [kg].
    pub mass: S,
    /// Principal moments of inertia [Ixx, Iyy, Izz] [kg·m²].
    pub inertia: [S; 3],
    /// Gravitational acceleration [m/s²].
    pub gravity: S,
}

impl<S: ControlScalar> GeometricConfig<S> {
    /// Reasonable defaults for a 500 g quadrotor.
    pub fn standard() -> Self {
        Self {
            k_r: S::from_f64(8.81),
            k_omega: S::from_f64(2.54),
            k_x: S::from_f64(16.0),
            k_v: S::from_f64(5.6),
            mass: S::from_f64(0.5),
            inertia: [
                S::from_f64(4.0e-3),
                S::from_f64(4.0e-3),
                S::from_f64(8.0e-3),
            ],
            gravity: S::from_f64(9.81),
        }
    }
}

// ─── Controller ───────────────────────────────────────────────────────────────

/// Geometric tracking controller on SE(3).
///
/// On each call to `update` the controller:
/// 1. Computes translational errors and desired force vector.
/// 2. Extracts desired attitude R_d from the force direction and yaw reference.
/// 3. Computes SO(3) attitude error and angular-velocity error.
/// 4. Returns (thrust, torque) that drives the vehicle to the reference.
#[derive(Debug, Clone, Copy)]
pub struct GeometricController<S: ControlScalar> {
    config: GeometricConfig<S>,
}

impl<S: ControlScalar> GeometricController<S> {
    /// Create a new controller with the given configuration.
    pub fn new(config: GeometricConfig<S>) -> Self {
        Self { config }
    }

    /// Compute (thrust [N], torque [N·m; 3]) for one control cycle.
    ///
    /// # Arguments
    /// * `state`     — current vehicle state
    /// * `ref_state` — reference trajectory point
    ///
    /// # Returns
    /// `(thrust, torque_xyz)` where `thrust` is the collective thrust magnitude
    /// and `torque_xyz` is the body-frame torque vector.
    pub fn update(
        &self,
        state: &QuadRotorGeomState<S>,
        ref_state: &GeometricRef<S>,
    ) -> (S, [S; 3]) {
        let cfg = &self.config;

        // ── Translational errors ──────────────────────────────────────────────
        let e_x = [
            state.position[0] - ref_state.position[0],
            state.position[1] - ref_state.position[1],
            state.position[2] - ref_state.position[2],
        ];
        let e_v = [
            state.velocity[0] - ref_state.velocity[0],
            state.velocity[1] - ref_state.velocity[1],
            state.velocity[2] - ref_state.velocity[2],
        ];

        // Unit vector pointing upward in inertial frame (+z direction)
        let e3 = [S::ZERO, S::ZERO, S::ONE];

        // Desired force (in inertial frame):
        //   F_des = −k_x·e_x − k_v·e_v + m·g·e3 + m·a_d
        let f_des = [
            -cfg.k_x * e_x[0] - cfg.k_v * e_v[0]
                + cfg.mass * cfg.gravity * e3[0]
                + cfg.mass * ref_state.acceleration[0],
            -cfg.k_x * e_x[1] - cfg.k_v * e_v[1]
                + cfg.mass * cfg.gravity * e3[1]
                + cfg.mass * ref_state.acceleration[1],
            -cfg.k_x * e_x[2] - cfg.k_v * e_v[2]
                + cfg.mass * cfg.gravity * e3[2]
                + cfg.mass * ref_state.acceleration[2],
        ];

        // Scalar thrust = projection of F_des onto current body z-axis
        let b3 = state.attitude.apply(e3); // Current body +z in world frame
        let thrust = vec3_dot(f_des, b3);

        // ── Desired attitude construction ─────────────────────────────────────
        // R_d is chosen so that its third column (b3_d) aligns with F_des,
        // and its first column (b1_d) is consistent with the yaw reference.
        let r_d = compute_desired_attitude(f_des, ref_state.yaw);

        // Desired angular velocity in body frame (from yaw_rate only, assuming
        // slow trajectory → Ω_d ≈ Rᵀ_d · [0,0,yaw_rate]).
        let omega_d_world = [S::ZERO, S::ZERO, ref_state.yaw_rate];
        let omega_d = r_d.transpose().apply(omega_d_world);

        // ── Attitude errors ───────────────────────────────────────────────────
        // e_R = 0.5·vee(R_dᵀ·R − Rᵀ·R_d)
        let e_r = rotation_error(&r_d, &state.attitude);

        // e_Ω = Ω − Rᵀ·R_d·Ω_d
        let rt_rd_omega_d = state.attitude.transpose().apply(r_d.apply(omega_d));
        let e_omega = [
            state.omega[0] - rt_rd_omega_d[0],
            state.omega[1] - rt_rd_omega_d[1],
            state.omega[2] - rt_rd_omega_d[2],
        ];

        // ── Attitude torque ───────────────────────────────────────────────────
        // τ = −k_R·e_R − k_Ω·e_Ω + Ω × J·Ω
        let j = cfg.inertia;
        let j_omega = [
            j[0] * state.omega[0],
            j[1] * state.omega[1],
            j[2] * state.omega[2],
        ];
        let gyro = vec3_cross(state.omega, j_omega);

        let torque = [
            -cfg.k_r * e_r[0] - cfg.k_omega * e_omega[0] + gyro[0],
            -cfg.k_r * e_r[1] - cfg.k_omega * e_omega[1] + gyro[1],
            -cfg.k_r * e_r[2] - cfg.k_omega * e_omega[2] + gyro[2],
        ];

        (thrust, torque)
    }
}

// ─── Desired attitude computation ─────────────────────────────────────────────

/// Compute desired rotation R_d ∈ SO(3) from a desired force vector and a yaw.
///
/// Strategy (Lee 2010, extended):
/// 1. b3_d = normalise(F_des)              — desired body +z
/// 2. b1_c = [cos(ψ), sin(ψ), 0]          — candidate heading (from yaw ref ψ)
/// 3. b2_d = normalise(b3_d × b1_c)       — desired body +y (right-hand)
/// 4. b1_d = b2_d × b3_d                  — desired body +x
/// 5. R_d  = [b1_d | b2_d | b3_d]         — column-major
///
/// Falls back to identity if F_des is near zero.
fn compute_desired_attitude<S: ControlScalar>(f_des: [S; 3], yaw: S) -> SO3<S> {
    let f_norm = vec3_norm(f_des);
    if f_norm < S::from_f64(1e-6) {
        return SO3::identity();
    }

    let inv_f = S::ONE / f_norm;
    let b3_d = [f_des[0] * inv_f, f_des[1] * inv_f, f_des[2] * inv_f];

    // Candidate first body-axis from yaw angle
    let b1_c = [yaw.cos(), yaw.sin(), S::ZERO];

    // b2_d = normalise(b3_d × b1_c)
    let b2_raw = vec3_cross(b3_d, b1_c);
    let b2_norm = vec3_norm(b2_raw);
    let b2_d = if b2_norm < S::from_f64(1e-6) {
        // b3_d parallel to b1_c (edge case: nearly straight up with 0 yaw)
        // Fall back to [0,1,0] as candidate y-axis
        let b2_raw2 = vec3_cross(b3_d, [S::ZERO, S::ONE, S::ZERO]);
        let b2n2 = vec3_norm(b2_raw2);
        if b2n2 < S::from_f64(1e-6) {
            [S::ONE, S::ZERO, S::ZERO]
        } else {
            let inv_b2 = S::ONE / b2n2;
            [
                b2_raw2[0] * inv_b2,
                b2_raw2[1] * inv_b2,
                b2_raw2[2] * inv_b2,
            ]
        }
    } else {
        let inv_b2 = S::ONE / b2_norm;
        [b2_raw[0] * inv_b2, b2_raw[1] * inv_b2, b2_raw[2] * inv_b2]
    };

    // b1_d = b2_d × b3_d
    let b1_d = vec3_cross(b2_d, b3_d);

    // R_d columns: [b1_d, b2_d, b3_d] → stored row-major:
    //   mat[row][col] = basis_vector_col[row]
    SO3::from_matrix_unchecked([
        [b1_d[0], b2_d[0], b3_d[0]],
        [b1_d[1], b2_d[1], b3_d[1]],
        [b1_d[2], b2_d[2], b3_d[2]],
    ])
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;
    const EPS_LOOSE: f64 = 1e-6;

    /// At hover (state = reference = zero), thrust should equal m·g and
    /// torque should be zero (no errors, no gyroscopic term at Ω=0).
    #[test]
    fn hover_zero_torque_correct_thrust() {
        let cfg = GeometricConfig::<f64>::standard();
        let expected_thrust = cfg.mass * cfg.gravity;
        let ctrl = GeometricController::new(cfg);

        let state = QuadRotorGeomState::<f64>::zero();
        let ref_state = GeometricRef::<f64>::hover_at_origin();

        let (thrust, torque) = ctrl.update(&state, &ref_state);

        assert!(
            (thrust - expected_thrust).abs() < EPS_LOOSE,
            "hover thrust: got {}, expected {}",
            thrust,
            expected_thrust
        );
        for (i, &t) in torque.iter().enumerate() {
            assert!(
                t.abs() < EPS_LOOSE,
                "hover torque[{}] = {} (expected 0)",
                i,
                t
            );
        }
    }

    /// Small angular perturbation: torque should point back toward zero
    /// (restoring sign).  With positive e_R, the τ = −k_R·e_R term is negative.
    #[test]
    fn small_attitude_perturbation_restoring_torque() {
        let cfg = GeometricConfig::<f64>::standard();
        let ctrl = GeometricController::new(cfg);

        // Tilt slightly about x-axis
        let r_perturbed = SO3::<f64>::from_axis_angle([1.0, 0.0, 0.0], 0.05).unwrap();
        let state = QuadRotorGeomState {
            position: [0.0; 3],
            velocity: [0.0; 3],
            attitude: r_perturbed,
            omega: [0.0; 3],
        };
        let ref_state = GeometricRef::<f64>::hover_at_origin();

        let (_thrust, torque) = ctrl.update(&state, &ref_state);

        // Torque about x should be negative (restoring)
        assert!(
            torque[0] < 0.0,
            "restoring torque[0] should be negative, got {}",
            torque[0]
        );
        // Torques about y and z should be small
        assert!(
            torque[1].abs() < EPS_LOOSE,
            "torque[1] = {} should be small",
            torque[1]
        );
    }

    /// Small position offset: thrust vector should have a component pointing
    /// toward the reference, meaning the computed desired direction shifts.
    #[test]
    fn position_error_modifies_thrust() {
        let cfg = GeometricConfig::<f64>::standard();
        let base_thrust = cfg.mass * cfg.gravity;
        let ctrl = GeometricController::new(cfg);

        // Offset upward (z) — F_des z-component increases
        let state = QuadRotorGeomState {
            position: [0.0, 0.0, 1.0], // 1 m above reference
            velocity: [0.0; 3],
            attitude: SO3::identity(),
            omega: [0.0; 3],
        };
        let ref_state = GeometricRef::<f64>::hover_at_origin();

        let (thrust, _torque) = ctrl.update(&state, &ref_state);

        // Corrective force in z should be downward: F_des_z = mg - k_x*1.0
        // For standard config: 0.5*9.81 - 16.0*1.0 = 4.905 - 16 = -11.095
        // Thrust = F_des · b3 where b3 = [0,0,1], so thrust = F_des_z.
        let expected = base_thrust - cfg.k_x * 1.0_f64;
        assert!(
            (thrust - expected).abs() < EPS_LOOSE,
            "thrust with +1m z offset: got {}, expected {}",
            thrust,
            expected
        );
    }

    /// Verify that the desired attitude for a hover (F_des = m*g*e3) with
    /// yaw=0 is the identity rotation.
    #[test]
    fn desired_attitude_hover_is_identity() {
        let m = 0.5_f64;
        let g = 9.81_f64;
        let f_des = [0.0, 0.0, m * g];
        let r_d = compute_desired_attitude(f_des, 0.0_f64);
        // Should be identity (within numerical tolerance)
        let eye = SO3::<f64>::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (r_d.mat[i][j] - eye.mat[i][j]).abs() < EPS_LOOSE,
                    "R_d[{},{}] = {} (expected {})",
                    i,
                    j,
                    r_d.mat[i][j],
                    eye.mat[i][j]
                );
            }
        }
    }

    /// Gyroscopic term: at high angular velocity, gyro term Ω×JΩ should appear
    /// in the torque output.
    #[test]
    fn gyroscopic_torque_at_high_omega() {
        let cfg = GeometricConfig::<f64>::standard();
        let ctrl = GeometricController::new(cfg);

        // Spin about z at 10 rad/s at hover attitude
        let omega_spin = [0.0, 0.0, 10.0_f64];
        let state = QuadRotorGeomState {
            position: [0.0; 3],
            velocity: [0.0; 3],
            attitude: SO3::identity(),
            omega: omega_spin,
        };
        let ref_state = GeometricRef {
            position: [0.0; 3],
            velocity: [0.0; 3],
            acceleration: [0.0; 3],
            yaw: 0.0,
            yaw_rate: 0.0,
        };

        let (_thrust, torque) = ctrl.update(&state, &ref_state);

        // Gyro = Ω × J·Ω = [0,0,10] × [0,0,Izz*10] = [0,0,0] (parallel → zero)
        // But e_Ω = omega - Rᵀ·Rd·Ω_d = [0,0,10] (since Ω_d=0)
        // τ = -k_omega*e_omega + gyro = -k_omega*[0,0,10] + [0,0,0]
        let expected_z = -cfg.k_omega * 10.0_f64;
        assert!(
            (torque[2] - expected_z).abs() < EPS_LOOSE,
            "torque[2] = {} expected {}",
            torque[2],
            expected_z
        );

        // Check that if we ask for the same spin rate as reference, torques vanish.
        let ref_spinning = GeometricRef {
            yaw_rate: 10.0,
            ..ref_state
        };
        let (_thrust2, torque2) = ctrl.update(&state, &ref_spinning);
        // e_Ω should vanish, gyro also zero → all torques near zero
        for (i, &t2) in torque2.iter().enumerate() {
            assert!(
                t2.abs() < EPS_LOOSE,
                "torque2[{}] = {} (expected ~0)",
                i,
                t2
            );
        }
    }

    /// Linearised check: for small e_x, small e_v, hover attitude, the thrust
    /// should be approximately m*g + correction terms.
    #[test]
    fn small_angle_matches_linearized() {
        let cfg = GeometricConfig::<f64>::standard();
        let ctrl = GeometricController::new(cfg);

        let dz = 0.01_f64; // 1 cm offset
        let state = QuadRotorGeomState {
            position: [0.0, 0.0, dz],
            velocity: [0.0; 3],
            attitude: SO3::identity(),
            omega: [0.0; 3],
        };
        let ref_state = GeometricRef::<f64>::hover_at_origin();

        let (thrust, torque) = ctrl.update(&state, &ref_state);

        let expected_thrust = cfg.mass * cfg.gravity - cfg.k_x * dz;
        assert!(
            (thrust - expected_thrust).abs() < 1e-8,
            "linearized thrust: got {}, expected {}",
            thrust,
            expected_thrust
        );
        for (i, &t) in torque.iter().enumerate() {
            assert!(t.abs() < EPS, "torque[{}] = {}", i, t);
        }
    }

    #[test]
    fn config_standard_has_positive_params() {
        let cfg = GeometricConfig::<f64>::standard();
        assert!(cfg.k_r > 0.0);
        assert!(cfg.k_omega > 0.0);
        assert!(cfg.k_x > 0.0);
        assert!(cfg.k_v > 0.0);
        assert!(cfg.mass > 0.0);
        assert!(cfg.gravity > 0.0);
        for j in cfg.inertia.iter() {
            assert!(*j > 0.0);
        }
    }
}
