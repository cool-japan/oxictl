//! Differential flatness for a 2-DOF planar revolute manipulator.
//!
//! For a two-link planar arm with link lengths `l1` and `l2`, the end-effector
//! position [x_ee, y_ee] is a flat output: all joint positions, velocities, and
//! accelerations can be recovered from (x_ee, y_ee) and their derivatives.
//!
//! ## Geometric inverse kinematics
//!
//! ```text
//!   cos q2 = (x² + y² − l1² − l2²) / (2·l1·l2)   →  q2 = ±acos(·)
//!   q1 = atan2(y, x) − atan2(l2·sin q2, l1 + l2·cos q2)
//! ```
//!
//! The `±` selects the *elbow-down* configuration (q2 ≥ 0).
//!
//! ## Jacobian and its time derivative
//!
//! ```text
//!   J(q) = [−l1·sin q1 − l2·sin(q1+q2),  −l2·sin(q1+q2)]
//!           [ l1·cos q1 + l2·cos(q1+q2),   l2·cos(q1+q2)]
//!
//!   q̇ = J⁻¹ · ẋ_ee
//!   q̈ = J⁻¹ · (ẍ_ee − J̇·q̇)
//! ```
//!
//! ## Singularities
//! The arm is singular when `sin q2 ≈ 0` (det J = l1·l2·sin q2 → 0),
//! i.e., fully extended or fully folded.

use crate::core::scalar::ControlScalar;
use crate::flatness::FlatnessError;

// ────────────────────────────────────────────────────────────────────────────
// ManipulatorParams
// ────────────────────────────────────────────────────────────────────────────

/// Physical parameters for a 2-DOF planar revolute manipulator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManipulatorParams<S: ControlScalar> {
    /// Length of link 1 (m), must be > 0.
    pub l1: S,
    /// Length of link 2 (m), must be > 0.
    pub l2: S,
}

impl<S: ControlScalar> ManipulatorParams<S> {
    /// Create parameters with validation.
    pub fn new(l1: S, l2: S) -> Result<Self, FlatnessError> {
        if l1 <= S::ZERO {
            return Err(FlatnessError::InvalidParameter("l1 must be positive"));
        }
        if l2 <= S::ZERO {
            return Err(FlatnessError::InvalidParameter("l2 must be positive"));
        }
        Ok(Self { l1, l2 })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ManipulatorFlatMap
// ────────────────────────────────────────────────────────────────────────────

/// Inverse flat map for a 2-DOF planar revolute manipulator.
///
/// Converts end-effector position/velocity/acceleration [x, y, ẋ, ẏ, ẍ, ÿ]
/// into joint positions q = [q1, q2], joint velocities q̇, and joint accelerations q̈.
///
/// **Elbow-down convention**: selects the solution with `q2 ≥ 0` (or the
/// nearest feasible solution).
#[derive(Debug, Clone, Copy)]
pub struct ManipulatorFlatMap<S: ControlScalar> {
    params: ManipulatorParams<S>,
    /// Threshold |sin q2| below which the Jacobian is considered singular.
    pub singularity_threshold: S,
}

impl<S: ControlScalar> ManipulatorFlatMap<S> {
    /// Create a flat map with explicit singularity threshold.
    pub fn new(
        params: ManipulatorParams<S>,
        singularity_threshold: S,
    ) -> Result<Self, FlatnessError> {
        if singularity_threshold < S::ZERO {
            return Err(FlatnessError::InvalidParameter(
                "singularity_threshold must be non-negative",
            ));
        }
        Ok(Self {
            params,
            singularity_threshold,
        })
    }

    /// Compute inverse kinematics: joint angle q2 (elbow-down, q2 ≥ 0).
    ///
    /// Returns `FlatnessError::Singular` if the target is outside the workspace
    /// (|cos q2| > 1) or at a singular configuration.
    fn ik_q2(&self, x: S, y: S) -> Result<S, FlatnessError> {
        let l1 = self.params.l1;
        let l2 = self.params.l2;

        let r2 = x * x + y * y;
        let numer = r2 - l1 * l1 - l2 * l2;
        let denom = S::TWO * l1 * l2;

        if denom.abs() < S::EPSILON {
            return Err(FlatnessError::Singular);
        }

        let cos_q2 = numer / denom;

        // Check workspace
        if cos_q2 < -S::ONE || cos_q2 > S::ONE {
            return Err(FlatnessError::Singular);
        }

        // Clamp floating-point noise near ±1
        let cos_q2 = cos_q2.clamp_val(-S::ONE, S::ONE);
        // Elbow-down: q2 ≥ 0 → use +acos
        let q2 = cos_q2.acos();

        Ok(q2)
    }

    /// Compute inverse kinematics: joint angle q1 given q2.
    fn ik_q1(&self, x: S, y: S, q2: S) -> S {
        let l1 = self.params.l1;
        let l2 = self.params.l2;

        let s2 = q2.sin();
        let c2 = q2.cos();

        // atan2(y, x) - atan2(l2*s2, l1 + l2*c2)
        let alpha = y.atan2(x);
        let beta = (l2 * s2).atan2(l1 + l2 * c2);
        alpha - beta
    }

    /// Compute the 2×2 Jacobian and its determinant.
    ///
    /// ```text
    ///   J = [−l1·s1 − l2·s12,  −l2·s12]
    ///       [ l1·c1 + l2·c12,   l2·c12]
    ///
    ///   det J = l1·l2·sin(q2)
    /// ```
    fn jacobian(&self, q1: S, q2: S) -> ([[S; 2]; 2], S) {
        let l1 = self.params.l1;
        let l2 = self.params.l2;

        let q12 = q1 + q2;
        let s1 = q1.sin();
        let c1 = q1.cos();
        let s12 = q12.sin();
        let c12 = q12.cos();

        let j = [
            [-(l1 * s1 + l2 * s12), -l2 * s12],
            [l1 * c1 + l2 * c12, l2 * c12],
        ];

        // det(J) = J[0][0]*J[1][1] - J[0][1]*J[1][0]
        //        = (−l1s1 − l2s12)(l2c12) − (−l2s12)(l1c1 + l2c12)
        //        = −l1l2s1c12 − l2²s12c12 + l1l2c1s12 + l2²s12c12
        //        = l1l2(c1s12 − s1c12) = l1l2·sin(q2)
        let det = l1 * l2 * q2.sin();
        (j, det)
    }

    /// Compute the time derivative of the Jacobian Ṫ = J̇, given q = (q1, q2) and q̇.
    ///
    /// ```text
    ///   J̇[0][0] = −l1·q̇1·c1 − l2·(q̇1+q̇2)·c12
    ///   J̇[0][1] = −l2·(q̇1+q̇2)·c12
    ///   J̇[1][0] = −l1·q̇1·s1 − l2·(q̇1+q̇2)·s12
    ///   J̇[1][1] = −l2·(q̇1+q̇2)·s12
    /// ```
    fn jacobian_dot(&self, q1: S, q2: S, qd1: S, qd2: S) -> [[S; 2]; 2] {
        let l1 = self.params.l1;
        let l2 = self.params.l2;

        let q12 = q1 + q2;
        let s1 = q1.sin();
        let c1 = q1.cos();
        let s12 = q12.sin();
        let c12 = q12.cos();

        let qd12 = qd1 + qd2;

        [
            [-(l1 * qd1 * c1 + l2 * qd12 * c12), -(l2 * qd12 * c12)],
            [-(l1 * qd1 * s1 + l2 * qd12 * s12), -(l2 * qd12 * s12)],
        ]
    }

    /// Solve 2×2 system A·x = b via Cramer's rule.
    ///
    /// Returns `FlatnessError::Singular` if |det A| < singularity_threshold.
    fn solve2x2(&self, a: [[S; 2]; 2], b: [S; 2]) -> Result<[S; 2], FlatnessError> {
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        if det.abs() < self.singularity_threshold {
            return Err(FlatnessError::Singular);
        }
        let x0 = (b[0] * a[1][1] - b[1] * a[0][1]) / det;
        let x1 = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
        Ok([x0, x1])
    }

    /// Full inverse flat map: recover [q, q̇, q̈] from end-effector position,
    /// velocity, and acceleration.
    ///
    /// # Arguments
    /// - `x`, `y`: end-effector position (m).
    /// - `xd`, `yd`: end-effector velocity (m/s).
    /// - `xdd`, `ydd`: end-effector acceleration (m/s²).
    ///
    /// # Returns
    /// `(q, q_dot, q_ddot)` where each is a `[S; 2]` array [joint1, joint2].
    ///
    /// # Errors
    /// - `FlatnessError::Singular` at a singular configuration or outside the
    ///   workspace reachability envelope.
    #[allow(clippy::type_complexity)]
    pub fn flat_to_joints(
        &self,
        x: S,
        y: S,
        xd: S,
        yd: S,
        xdd: S,
        ydd: S,
    ) -> Result<([S; 2], [S; 2], [S; 2]), FlatnessError> {
        // ── IK: joint positions ─────────────────────────────────────────────
        let q2 = self.ik_q2(x, y)?;
        let q1 = self.ik_q1(x, y, q2);

        // Check singularity threshold on |sin q2|
        let (jac, det) = self.jacobian(q1, q2);
        if det.abs() < self.singularity_threshold {
            return Err(FlatnessError::Singular);
        }

        // ── Joint velocities: q̇ = J⁻¹ · ẋ_ee ─────────────────────────────
        let qd = self.solve2x2(jac, [xd, yd])?;

        // ── Joint accelerations: q̈ = J⁻¹ · (ẍ_ee − J̇·q̇) ─────────────────
        let jdot = self.jacobian_dot(q1, q2, qd[0], qd[1]);

        // J̇·q̇
        let jdot_qd = [
            jdot[0][0] * qd[0] + jdot[0][1] * qd[1],
            jdot[1][0] * qd[0] + jdot[1][1] * qd[1],
        ];

        let rhs = [xdd - jdot_qd[0], ydd - jdot_qd[1]];
        let qdd = self.solve2x2(jac, rhs)?;

        Ok(([q1, q2], qd, qdd))
    }

    /// Forward kinematics: compute end-effector position from joint angles.
    ///
    /// Useful for round-trip verification.
    pub fn forward_kinematics(&self, q1: S, q2: S) -> (S, S) {
        let l1 = self.params.l1;
        let l2 = self.params.l2;
        let q12 = q1 + q2;
        let x = l1 * q1.cos() + l2 * q12.cos();
        let y = l1 * q1.sin() + l2 * q12.sin();
        (x, y)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_map(l1: f64, l2: f64) -> ManipulatorFlatMap<f64> {
        let params = ManipulatorParams::new(l1, l2).expect("valid params");
        ManipulatorFlatMap::new(params, 1e-6).expect("valid map")
    }

    /// At a singular configuration (arm fully extended in +x), the inverse map
    /// should return `FlatnessError::Singular`.
    #[test]
    fn singular_arm_straight_returns_error() {
        let l1 = 1.0_f64;
        let l2 = 1.0_f64;
        let map = make_map(l1, l2);

        // Fully extended: x = l1+l2, y = 0 → q2 = 0 → sin(q2)=0 → singular
        let result = map.flat_to_joints(l1 + l2, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            matches!(result, Err(FlatnessError::Singular)),
            "Expected Singular at fully-extended arm, got {:?}",
            result
        );
    }

    /// Outside workspace returns Singular.
    #[test]
    fn out_of_workspace_returns_error() {
        let l1 = 1.0_f64;
        let l2 = 0.5_f64;
        let map = make_map(l1, l2);

        // Target beyond reach: r = l1+l2+1 > l1+l2
        let result = map.flat_to_joints(l1 + l2 + 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            matches!(result, Err(FlatnessError::Singular)),
            "Expected Singular for out-of-workspace target, got {:?}",
            result
        );
    }

    /// Round-trip FK/IK: given a reachable point, IK should return joint angles
    /// that FK maps back to the original point.
    #[test]
    fn roundtrip_fk_ik() {
        let l1 = 1.0_f64;
        let l2 = 0.8_f64;
        let map = make_map(l1, l2);

        // A reachable interior point
        let x_target = 0.8_f64;
        let y_target = 0.9_f64;

        let (q, _qd, _qdd) = map
            .flat_to_joints(x_target, y_target, 0.0, 0.0, 0.0, 0.0)
            .expect("IK should succeed for interior point");

        let (x_fk, y_fk) = map.forward_kinematics(q[0], q[1]);

        assert!(
            (x_fk - x_target).abs() < 1e-9,
            "FK x={:.8} ≠ target x={:.8}",
            x_fk,
            x_target
        );
        assert!(
            (y_fk - y_target).abs() < 1e-9,
            "FK y={:.8} ≠ target y={:.8}",
            y_fk,
            y_target
        );
    }

    /// With zero velocity and acceleration, joint rates should be zero.
    #[test]
    fn zero_velocity_gives_zero_joint_rates() {
        let l1 = 1.0_f64;
        let l2 = 0.8_f64;
        let map = make_map(l1, l2);

        let (_, qd, qdd) = map
            .flat_to_joints(0.8, 0.9, 0.0, 0.0, 0.0, 0.0)
            .expect("IK");

        assert!(qd[0].abs() < 1e-12, "qd1={:.2e}", qd[0]);
        assert!(qd[1].abs() < 1e-12, "qd2={:.2e}", qd[1]);
        assert!(qdd[0].abs() < 1e-12, "qdd1={:.2e}", qdd[0]);
        assert!(qdd[1].abs() < 1e-12, "qdd2={:.2e}", qdd[1]);
    }

    /// Verify Jacobian consistency: J·q̇ ≈ ẋ_ee for a known configuration.
    #[test]
    fn jacobian_velocity_consistency() {
        let l1 = 1.0_f64;
        let l2 = 0.8_f64;
        let map = make_map(l1, l2);

        let x = 0.8_f64;
        let y = 0.9_f64;
        let xd = 0.2_f64;
        let yd = -0.3_f64;

        let (q, qd, _) = map.flat_to_joints(x, y, xd, yd, 0.0, 0.0).expect("IK");

        // Recompute J·q̇ and check it equals [xd, yd]
        let (j, _) = map.jacobian(q[0], q[1]);
        let ee_vel_x = j[0][0] * qd[0] + j[0][1] * qd[1];
        let ee_vel_y = j[1][0] * qd[0] + j[1][1] * qd[1];

        assert!(
            (ee_vel_x - xd).abs() < 1e-10,
            "J·q̇ x={:.8} ≠ xd={:.8}",
            ee_vel_x,
            xd
        );
        assert!(
            (ee_vel_y - yd).abs() < 1e-10,
            "J·q̇ y={:.8} ≠ yd={:.8}",
            ee_vel_y,
            yd
        );
    }

    /// Elbow-down convention: q2 should be non-negative.
    #[test]
    fn elbow_down_q2_nonnegative() {
        let l1 = 1.0_f64;
        let l2 = 0.8_f64;
        let map = make_map(l1, l2);

        // Several interior targets
        let targets = [(0.5_f64, 0.5_f64), (1.0, 0.3), (0.2, 1.1), (-0.4, 0.8)];
        for (x, y) in targets {
            if let Ok((q, _, _)) = map.flat_to_joints(x, y, 0.0, 0.0, 0.0, 0.0) {
                assert!(
                    q[1] >= 0.0,
                    "q2={:.6} should be ≥ 0 for ({},{})",
                    q[1],
                    x,
                    y
                );
            }
        }
    }

    /// Acceleration recovery: J·q̈ + J̇·q̇ ≈ ẍ_ee.
    #[test]
    fn jacobian_acceleration_consistency() {
        let l1 = 1.0_f64;
        let l2 = 0.8_f64;
        let map = make_map(l1, l2);

        let x = 0.8_f64;
        let y = 0.9_f64;
        let xd = 0.1_f64;
        let yd = 0.2_f64;
        let xdd = -0.05_f64;
        let ydd = 0.1_f64;

        let (q, qd, qdd) = map.flat_to_joints(x, y, xd, yd, xdd, ydd).expect("IK");

        let (j, _) = map.jacobian(q[0], q[1]);
        let jdot = map.jacobian_dot(q[0], q[1], qd[0], qd[1]);

        // ẍ_ee = J·q̈ + J̇·q̇
        let ax_rec = j[0][0] * qdd[0] + j[0][1] * qdd[1] + jdot[0][0] * qd[0] + jdot[0][1] * qd[1];
        let ay_rec = j[1][0] * qdd[0] + j[1][1] * qdd[1] + jdot[1][0] * qd[0] + jdot[1][1] * qd[1];

        assert!(
            (ax_rec - xdd).abs() < 1e-9,
            "ax_rec={:.8} ≠ xdd={:.8}",
            ax_rec,
            xdd
        );
        assert!(
            (ay_rec - ydd).abs() < 1e-9,
            "ay_rec={:.8} ≠ ydd={:.8}",
            ay_rec,
            ydd
        );
    }
}
