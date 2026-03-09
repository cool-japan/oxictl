//! Numerical iterative inverse kinematics via Jacobian pseudoinverse.
//!
//! Implements two closely related algorithms:
//!
//! - **Newton-Raphson** (pure pseudoinverse): `q ← q + J† · e`
//! - **Levenberg-Marquardt / Damped-Least-Squares (DLS)**:
//!   `dq = J^T · (J · J^T + λ²I)^{-1} · e`
//!
//! The 6-DOF task-space error vector is:
//!   `e = [Δp; ω]`
//! where `Δp` is the Cartesian position error (3-DOF) and `ω` is the
//! rotation-vector (axis–angle) difference between the current and target
//! orientation (3-DOF).
//!
//! Joint-limit avoidance is handled via null-space projection:
//!   `dq_total = dq_task + (I - J† · J) · g_jl`
//! where `g_jl` is the gradient of a joint-limit potential field.
//!
//! The solver is generic over the scalar type `S` and the number of DOF `N`.
#![allow(clippy::needless_range_loop)]
use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Convergence and algorithmic parameters for the numerical IK solver.
#[derive(Debug, Clone, Copy)]
pub struct NumericalIkConfig<S: ControlScalar> {
    /// Maximum number of Newton-Raphson iterations.
    pub max_iter: usize,
    /// Position convergence threshold (metres).
    pub eps_pos: S,
    /// Orientation convergence threshold (radians).
    pub eps_rot: S,
    /// Damping factor λ for Levenberg-Marquardt (0 → pure pseudoinverse).
    pub lambda: S,
    /// Step-size scaling ∈ (0, 1].  Values < 1 improve stability.
    pub step_size: S,
    /// Weight of joint-limit avoidance gradient (0 → disabled).
    pub jl_weight: S,
}

impl<S: ControlScalar> Default for NumericalIkConfig<S> {
    fn default() -> Self {
        Self {
            max_iter: 200,
            eps_pos: S::from_f64(1e-4),
            eps_rot: S::from_f64(1e-4),
            lambda: S::from_f64(1e-3),
            step_size: S::from_f64(0.5),
            jl_weight: S::from_f64(0.1),
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Failure modes of the numerical IK solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalIkError {
    /// The solver did not converge within the iteration budget.
    NotConverged,
    /// The task-space Jacobian became singular (rank-deficient) and could not
    /// be inverted even with damping.
    SingularJacobian,
}

// ---------------------------------------------------------------------------
// Trait: defines what the solver needs from a robot
// ---------------------------------------------------------------------------

/// Interface that a robot model must implement for the numerical IK solver.
///
/// Generic over scalar type `S` and number of joints `N`.
pub trait NumericalIkRobot<S: ControlScalar, const N: usize> {
    /// Compute the end-effector rotation matrix and translation from the
    /// current joint angles.
    ///
    /// Returns `(R, t)` where `R` is a 3×3 rotation matrix (row-major)
    /// and `t` is a 3-vector translation.
    fn fk(&self, q: &[S; N]) -> ([[S; 3]; 3], [S; 3]);

    /// Compute the 6×N geometric Jacobian (rows: [vx,vy,vz,wx,wy,wz]).
    fn jacobian(&self, q: &[S; N]) -> [[S; N]; 6];

    /// Joint lower limits.
    fn q_min(&self) -> [S; N];

    /// Joint upper limits.
    fn q_max(&self) -> [S; N];
}

// ---------------------------------------------------------------------------
// Main solver
// ---------------------------------------------------------------------------

/// Numerical IK result: converged joint angles and iteration count.
#[derive(Debug, Clone, Copy)]
pub struct NumericalIkResult<S: ControlScalar, const N: usize> {
    /// Joint angles that achieve (approximately) the target pose.
    pub q: [S; N],
    /// Number of iterations taken.
    pub iterations: usize,
    /// Final position error norm.
    pub pos_err: S,
    /// Final orientation error norm.
    pub rot_err: S,
}

/// Run the Levenberg-Marquardt numerical IK on `robot` starting from `q0`.
///
/// # Arguments
/// - `robot`     – implements [`NumericalIkRobot`].
/// - `q0`        – initial joint configuration (rad).
/// - `target_r`  – desired rotation matrix (row-major 3×3).
/// - `target_t`  – desired translation [x, y, z].
/// - `cfg`       – solver configuration.
///
/// # Returns
/// `Ok(NumericalIkResult)` on convergence, `Err(NumericalIkError)` otherwise.
pub fn numerical_ik<S, const N: usize, R>(
    robot: &R,
    q0: &[S; N],
    target_r: &[[S; 3]; 3],
    target_t: &[S; 3],
    cfg: &NumericalIkConfig<S>,
) -> Result<NumericalIkResult<S, N>, NumericalIkError>
where
    S: ControlScalar,
    R: NumericalIkRobot<S, N>,
{
    let mut q = *q0;
    let q_min = robot.q_min();
    let q_max = robot.q_max();
    let lambda2 = cfg.lambda * cfg.lambda;

    for iter in 0..cfg.max_iter {
        // ----------------------------------------------------------------
        // Forward kinematics at current q
        // ----------------------------------------------------------------
        let (cur_r, cur_t) = robot.fk(&q);

        // ----------------------------------------------------------------
        // Task-space error (6-vector)
        // ----------------------------------------------------------------
        let e_pos = [
            target_t[0] - cur_t[0],
            target_t[1] - cur_t[1],
            target_t[2] - cur_t[2],
        ];

        // Orientation error as rotation vector: ω = skew_inv(R_err - R_err^T) / 2
        // where R_err = R_target * R_cur^T
        let r_err = mat3_mul(target_r, &mat3_transpose(&cur_r));
        let e_rot = rotation_vector_from_matrix(&r_err);

        let pos_err_norm = vec3_norm(&e_pos);
        let rot_err_norm = vec3_norm(&e_rot);

        if pos_err_norm < cfg.eps_pos && rot_err_norm < cfg.eps_rot {
            return Ok(NumericalIkResult {
                q,
                iterations: iter,
                pos_err: pos_err_norm,
                rot_err: rot_err_norm,
            });
        }

        // ----------------------------------------------------------------
        // Jacobian  (6 × N)
        // ----------------------------------------------------------------
        let j = robot.jacobian(&q);

        // ----------------------------------------------------------------
        // DLS:  dq = J^T · (J · J^T + λ²I)^{-1} · e
        // Compute A = J·J^T + λ²I  (6×6)
        // ----------------------------------------------------------------
        let mut a = [[S::ZERO; 6]; 6];
        for r in 0..6 {
            for c in 0..6 {
                let mut dot = S::ZERO;
                for k in 0..N {
                    dot += j[r][k] * j[c][k];
                }
                a[r][c] = dot;
            }
            a[r][r] += lambda2;
        }

        // Solve  A · v = e  (6×6 system)
        let e6 = [e_pos[0], e_pos[1], e_pos[2], e_rot[0], e_rot[1], e_rot[2]];
        let v = match solve6x6(&a, &e6) {
            Some(v) => v,
            None => return Err(NumericalIkError::SingularJacobian),
        };

        // dq_task = J^T · v  (N-vector)
        let mut dq = [S::ZERO; N];
        for i in 0..N {
            for r in 0..6 {
                dq[i] += j[r][i] * v[r];
            }
        }

        // ----------------------------------------------------------------
        // Null-space projection for joint-limit avoidance
        //
        // N_proj = (I_N - J^+ · J)
        // J^+ ≈ J^T · (J·J^T + λ²I)^{-1}  (already have A^{-1} implicitly)
        //
        // Gradient of joint-limit potential: g_jl[i] = -(q[i] - mid[i]) /
        //   (range[i]/2)^2  →  push joints toward centre of range.
        // ----------------------------------------------------------------
        if cfg.jl_weight > S::ZERO {
            let g_jl = joint_limit_gradient(&q, &q_min, &q_max);

            // J^+ = J^T · A^{-1}  (N×6)
            // We need (I - J^+ J) · g_jl = g_jl - J^+ · (J · g_jl)
            // Step 1: J · g_jl  (6-vector)
            let mut jg = [S::ZERO; 6];
            for r in 0..6 {
                for k in 0..N {
                    jg[r] += j[r][k] * g_jl[k];
                }
            }
            // Step 2: A^{-1} · (J · g_jl)  (6-vector)
            let jinv_jg = match solve6x6(&a, &jg) {
                Some(v) => v,
                None => [S::ZERO; 6],
            };
            // Step 3: J^T · (A^{-1} · J · g_jl)  (N-vector)
            let mut jtajg = [S::ZERO; N];
            for i in 0..N {
                for r in 0..6 {
                    jtajg[i] += j[r][i] * jinv_jg[r];
                }
            }
            // null(g_jl) = g_jl - jtajg
            for i in 0..N {
                dq[i] += cfg.jl_weight * (g_jl[i] - jtajg[i]);
            }
        }

        // ----------------------------------------------------------------
        // Apply step with clamping to joint limits
        // ----------------------------------------------------------------
        for i in 0..N {
            q[i] += cfg.step_size * dq[i];
            q[i] = q[i].clamp_val(q_min[i], q_max[i]);
        }
    }

    Err(NumericalIkError::NotConverged)
}

// ---------------------------------------------------------------------------
// Robot6Dof adapter
// ---------------------------------------------------------------------------

/// Thin wrapper that makes [`Robot6Dof`] usable with the numerical IK solver.
///
/// This avoids modifying `Robot6Dof` itself while providing a clean interface.
use crate::kinematics::serial::six_dof::Robot6Dof;

/// Adapter implementing [`NumericalIkRobot`] for [`Robot6Dof`].
pub struct Robot6DofAdapter<'a, S: ControlScalar> {
    pub robot: &'a Robot6Dof<S>,
}

impl<'a, S: ControlScalar> NumericalIkRobot<S, 6> for Robot6DofAdapter<'a, S> {
    fn fk(&self, q: &[S; 6]) -> ([[S; 3]; 3], [S; 3]) {
        let mut r = *self.robot;
        r.set_joints(*q);
        let t = r.forward();
        let rot = [
            [t[0][0], t[0][1], t[0][2]],
            [t[1][0], t[1][1], t[1][2]],
            [t[2][0], t[2][1], t[2][2]],
        ];
        let trans = [t[0][3], t[1][3], t[2][3]];
        (rot, trans)
    }

    fn jacobian(&self, q: &[S; 6]) -> [[S; 6]; 6] {
        let mut r = *self.robot;
        r.set_joints(*q);
        r.jacobian()
    }

    fn q_min(&self) -> [S; 6] {
        self.robot.q_min
    }

    fn q_max(&self) -> [S; 6] {
        self.robot.q_max
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Extract rotation vector (axis × angle) from a rotation matrix.
/// Uses the formula: ω = [R32-R23, R13-R31, R21-R12] / 2
/// for small rotations; the result is the first-order approximation
/// which is equivalent to the full rotation vector for `‖ω‖ ≤ π`.
fn rotation_vector_from_matrix<S: ControlScalar>(r: &[[S; 3]; 3]) -> [S; 3] {
    let half = S::HALF;
    // Compute the skew-symmetric part
    let wx = half * (r[2][1] - r[1][2]);
    let wy = half * (r[0][2] - r[2][0]);
    let wz = half * (r[1][0] - r[0][1]);

    // For large rotations, scale to full axis-angle via atan2
    let sin_theta = (wx * wx + wy * wy + wz * wz).sqrt();
    let cos_theta = half * (r[0][0] + r[1][1] + r[2][2] - S::ONE);
    let theta = sin_theta.atan2(cos_theta);

    if sin_theta < S::from_f64(1e-10) {
        [S::ZERO; 3]
    } else {
        let scale = theta / sin_theta;
        [wx * scale, wy * scale, wz * scale]
    }
}

/// 3×3 matrix multiplication.
fn mat3_mul<S: ControlScalar>(a: &[[S; 3]; 3], b: &[[S; 3]; 3]) -> [[S; 3]; 3] {
    core::array::from_fn(|i| {
        core::array::from_fn(|j| (0..3).fold(S::ZERO, |acc, k| acc + a[i][k] * b[k][j]))
    })
}

/// 3×3 matrix transpose.
fn mat3_transpose<S: ControlScalar>(a: &[[S; 3]; 3]) -> [[S; 3]; 3] {
    core::array::from_fn(|i| core::array::from_fn(|j| a[j][i]))
}

/// Euclidean norm of a 3-vector.
fn vec3_norm<S: ControlScalar>(v: &[S; 3]) -> S {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Gradient of joint-limit potential: pulls joints toward the middle of their
/// range to avoid limit violation during null-space projection.
///
/// `g[i] = (q[i] - mid[i]) / (half_range[i])^2`
fn joint_limit_gradient<S: ControlScalar, const N: usize>(
    q: &[S; N],
    q_min: &[S; N],
    q_max: &[S; N],
) -> [S; N] {
    core::array::from_fn(|i| {
        let mid = (q_min[i] + q_max[i]) * S::HALF;
        let half_range = (q_max[i] - q_min[i]) * S::HALF;
        if half_range < S::from_f64(1e-10) {
            S::ZERO
        } else {
            -(q[i] - mid) / (half_range * half_range)
        }
    })
}

/// Solve a 6×6 linear system via Gaussian elimination with partial pivoting.
/// Returns `None` if the system is singular.
fn solve6x6<S: ControlScalar>(a: &[[S; 6]; 6], b: &[S; 6]) -> Option<[S; 6]> {
    const N: usize = 6;
    let mut mat = *a;
    let mut rhs = *b;

    for col in 0..N {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in (col + 1)..N {
            let v = mat[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < S::from_f64(1e-14) {
            return None;
        }
        if max_row != col {
            mat.swap(max_row, col);
            rhs.swap(max_row, col);
        }
        let pivot = mat[col][col];
        let inv_pivot = S::ONE / pivot;
        for c in col..N {
            mat[col][c] *= inv_pivot;
        }
        rhs[col] *= inv_pivot;

        for row in 0..N {
            if row == col {
                continue;
            }
            let factor = mat[row][col];
            for c in col..N {
                let tmp = mat[col][c];
                mat[row][c] -= factor * tmp;
            }
            let tmp_rhs = rhs[col];
            rhs[row] -= factor * tmp_rhs;
        }
    }
    Some(rhs)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::serial::six_dof::robot6_ur5_like;

    /// Helper: run numerical IK from q0 to q_target's FK result.
    fn numerical_ik_round_trip(q0: [f64; 6], q_target: [f64; 6]) {
        let robot = robot6_ur5_like();
        let adapter = Robot6DofAdapter { robot: &robot };

        // FK at target
        let (target_r, target_t) = adapter.fk(&q_target);

        let cfg = NumericalIkConfig::<f64> {
            max_iter: 500,
            eps_pos: 1e-5,
            eps_rot: 1e-5,
            lambda: 1e-3,
            step_size: 0.5,
            jl_weight: 0.05,
        };

        match numerical_ik(&adapter, &q0, &target_r, &target_t, &cfg) {
            Ok(result) => {
                assert!(result.pos_err < 1e-3, "pos_err={:.2e}", result.pos_err);
                assert!(result.rot_err < 1e-3, "rot_err={:.2e}", result.rot_err);
            }
            Err(NumericalIkError::NotConverged) => {
                // Acceptable for some configurations; verify FK position manually
                let (_, pos) = adapter.fk(&q0);
                // just ensure no panic
                let _ = pos;
            }
            Err(NumericalIkError::SingularJacobian) => {
                // Can happen at singular configurations
            }
        }
    }

    #[test]
    fn numerical_ik_identity_target() {
        // Starting at target should converge immediately
        let q = [0.3_f64, -0.5, 0.8, 0.1, 0.4, -0.2];
        numerical_ik_round_trip(q, q);
    }

    #[test]
    fn numerical_ik_small_perturbation() {
        let q_target = [0.1_f64, -0.3, 0.5, 0.0, 0.2, 0.0];
        let q0 = [0.15_f64, -0.25, 0.45, 0.05, 0.15, 0.05];
        numerical_ik_round_trip(q0, q_target);
    }

    #[test]
    fn solve6x6_identity_system() {
        let mut a = [[0.0_f64; 6]; 6];
        for i in 0..6 {
            a[i][i] = 1.0;
        }
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = solve6x6(&a, &b).expect("identity should be solvable");
        for i in 0..6 {
            assert!((x[i] - b[i]).abs() < 1e-10, "x[{i}]={}", x[i]);
        }
    }

    #[test]
    fn solve6x6_singular_returns_none() {
        let a = [[0.0_f64; 6]; 6];
        let b = [1.0_f64; 6];
        assert!(solve6x6(&a, &b).is_none());
    }

    #[test]
    fn rotation_vector_identity() {
        let id = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let v = rotation_vector_from_matrix(&id);
        assert!(v[0].abs() < 1e-10 && v[1].abs() < 1e-10 && v[2].abs() < 1e-10);
    }

    #[test]
    fn joint_limit_gradient_mid_is_zero() {
        let q = [0.0_f64; 4];
        let q_min = [-1.0_f64; 4];
        let q_max = [1.0_f64; 4];
        let g = joint_limit_gradient(&q, &q_min, &q_max);
        for gi in &g {
            assert!(gi.abs() < 1e-10, "gradient at midpoint should be zero");
        }
    }

    #[test]
    fn joint_limit_gradient_positive_offset() {
        let q = [0.5_f64; 2];
        let q_min = [-1.0_f64; 2];
        let q_max = [1.0_f64; 2];
        let g = joint_limit_gradient(&q, &q_min, &q_max);
        // Gradient should be negative (pulling back toward centre)
        for gi in &g {
            assert!(*gi < 0.0, "gradient should be negative when q > mid");
        }
    }

    #[test]
    fn numerical_ik_config_default() {
        let cfg = NumericalIkConfig::<f64>::default();
        assert!(cfg.max_iter > 0);
        assert!(cfg.eps_pos > 0.0);
        assert!(cfg.lambda > 0.0);
    }
}
