use crate::core::scalar::ControlScalar;
use crate::kinematics::forward::Transform3D;

/// SCARA robot configuration parameters.
///
/// SCARA (Selective Compliance Assembly Robot Arm):
///   - 2 revolute joints in horizontal plane (q1, q2)
///   - 1 prismatic joint for vertical motion (d3)
///   - 1 revolute joint for end-effector rotation (q4)
///
/// DH parameters (simplified):
///   Joint 1: a1, α=0, d=0, θ=q1
///   Joint 2: a2, α=π, d=0, θ=q2
///   Joint 3: a=0, α=0, d=d3, θ=0
///   Joint 4: a=0, α=0, d=d4, θ=q4
#[derive(Debug, Clone, Copy)]
pub struct ScaraConfig<S: ControlScalar> {
    /// Link 1 length (m).
    pub a1: S,
    /// Link 2 length (m).
    pub a2: S,
    /// Offset from joint 4 to end-effector (m).
    pub d4: S,
    /// Base height (m).
    pub d1: S,
    /// Joint limits: (min, max) for each joint.
    pub q_min: [S; 4],
    pub q_max: [S; 4],
}

impl<S: ControlScalar> ScaraConfig<S> {
    /// Typical desktop SCARA: 200mm + 200mm reach.
    pub fn desktop() -> Self {
        let two_pi = S::TWO * S::PI;
        Self {
            a1: S::from_f64(0.2),
            a2: S::from_f64(0.2),
            d4: S::from_f64(0.05),
            d1: S::from_f64(0.35),
            q_min: [-two_pi; 4],
            q_max: [two_pi; 4],
        }
    }
}

/// SCARA robot kinematics.
pub struct ScaraRobot<S: ControlScalar> {
    pub config: ScaraConfig<S>,
    /// Current joint values: [q1, q2, d3, q4].
    pub q: [S; 4],
}

impl<S: ControlScalar> ScaraRobot<S> {
    pub fn new(config: ScaraConfig<S>) -> Self {
        Self {
            config,
            q: [S::ZERO; 4],
        }
    }

    /// Set joint values (clamped to limits).
    pub fn set_joints(&mut self, q: [S; 4]) {
        for (i, &qi) in q.iter().enumerate() {
            self.q[i] = qi.clamp_val(self.config.q_min[i], self.config.q_max[i]);
        }
    }

    /// Forward kinematics: compute end-effector position and Z-rotation.
    ///
    /// Returns (x, y, z, psi) where:
    ///   x, y: horizontal position
    ///   z: vertical position (height = d1 - d3 + d4)
    ///   psi: end-effector rotation angle
    pub fn forward(&self) -> (S, S, S, S) {
        self.fk_impl(&self.q)
    }

    fn fk_impl(&self, q: &[S; 4]) -> (S, S, S, S) {
        let q1 = q[0];
        let q2 = q[1];
        let d3 = q[2];
        let q4 = q[3];

        let c1 = q1.cos();
        let s1 = q1.sin();
        let c12 = (q1 + q2).cos();
        let s12 = (q1 + q2).sin();

        let x = self.config.a1 * c1 + self.config.a2 * c12;
        let y = self.config.a1 * s1 + self.config.a2 * s12;
        let z = self.config.d1 - d3 + self.config.d4;
        let psi = q1 + q2 + q4;

        (x, y, z, psi)
    }

    /// Inverse kinematics: compute joint angles for desired (x, y, z, psi).
    ///
    /// SCARA IK has two solutions (elbow-up, elbow-down).
    /// Returns `[q1, q2, d3, q4]` for the elbow-down solution.
    /// Returns `None` if target is unreachable.
    pub fn inverse(&self, x: S, y: S, z: S, psi: S) -> Option<[S; 4]> {
        let r2 = x * x + y * y;
        let r = r2.sqrt();
        let a1 = self.config.a1;
        let a2 = self.config.a2;

        // Check reachability
        let r_max = a1 + a2;
        let r_min = (a1 - a2).abs();
        if r > r_max || r < r_min {
            return None;
        }

        // Solve for q2 using cosine rule: r² = a1² + a2² + 2*a1*a2*cos(q2)
        let cos_q2 = (r2 - a1 * a1 - a2 * a2) / (S::TWO * a1 * a2);
        let cos_q2_clamped = cos_q2.clamp_val(-S::ONE, S::ONE);
        let q2 = -cos_q2_clamped.acos(); // elbow-down: q2 negative

        // Solve for q1: atan2(y, x) - atan2(a2*sin(q2), a1 + a2*cos(q2))
        let sin_q2 = q2.sin();
        let k1 = a1 + a2 * cos_q2_clamped;
        let k2 = a2 * sin_q2;
        let q1 = y.atan2(x) - k2.atan2(k1);

        // Prismatic joint: d3 = d1 + d4 - z
        let d3 = self.config.d1 + self.config.d4 - z;

        // End-effector rotation
        let q4 = psi - q1 - q2;

        // Check joint limits
        let q = [q1, q2, d3, q4];
        for (i, &qi) in q.iter().enumerate() {
            if qi < self.config.q_min[i] || qi > self.config.q_max[i] {
                return None;
            }
        }

        Some(q)
    }

    /// Compute the Jacobian (4×4: dx, dy, dz, dpsi vs dq1, dq2, dd3, dq4).
    pub fn jacobian(&self) -> [[S; 4]; 4] {
        let q1 = self.q[0];
        let q2 = self.q[1];
        let s1 = q1.sin();
        let c1 = q1.cos();
        let s12 = (q1 + q2).sin();
        let c12 = (q1 + q2).cos();
        let a1 = self.config.a1;
        let a2 = self.config.a2;

        [
            // dx/dq1, dx/dq2, dx/dd3, dx/dq4
            [-a1 * s1 - a2 * s12, -a2 * s12, S::ZERO, S::ZERO],
            // dy/dq1, dy/dq2, dy/dd3, dy/dq4
            [a1 * c1 + a2 * c12, a2 * c12, S::ZERO, S::ZERO],
            // dz/dq1, dz/dq2, dz/dd3, dz/dq4
            [S::ZERO, S::ZERO, -S::ONE, S::ZERO],
            // dpsi/dq1, dpsi/dq2, dpsi/dd3, dpsi/dq4
            [S::ONE, S::ONE, S::ZERO, S::ONE],
        ]
    }

    /// FK using Transform3D composition (for visualization or chaining).
    pub fn forward_transforms(&self) -> [Transform3D<S>; 4] {
        let q = &self.q;
        let t1 = Transform3D::rot_z(q[0]).compose(&Transform3D::translate(
            self.config.a1,
            S::ZERO,
            S::ZERO,
        ));
        let t2 = t1
            .compose(&Transform3D::rot_z(q[1]))
            .compose(&Transform3D::translate(self.config.a2, S::ZERO, S::ZERO));
        let t3 = t2.compose(&Transform3D::translate(S::ZERO, S::ZERO, -(q[2])));
        let t4 = t3.compose(&Transform3D::rot_z(q[3]));
        [t1, t2, t3, t4]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_scara() -> ScaraRobot<f64> {
        ScaraRobot::new(ScaraConfig {
            a1: 0.3,
            a2: 0.2,
            d4: 0.05,
            d1: 0.4,
            q_min: [-core::f64::consts::PI * 2.0; 4],
            q_max: [core::f64::consts::PI * 2.0; 4],
        })
    }

    #[test]
    fn fk_at_zero_config() {
        let mut robot = build_scara();
        robot.set_joints([0.0, 0.0, 0.0, 0.0]);
        let (x, y, z, _psi) = robot.forward();
        // At q=[0,0,0,0]: arm along x, x = a1+a2 = 0.5
        assert!((x - 0.5).abs() < 1e-10, "x={}", x);
        assert!(y.abs() < 1e-10, "y={}", y);
        // z = d1 - 0 + d4 = 0.4 + 0.05 = 0.45
        assert!((z - 0.45).abs() < 1e-10, "z={}", z);
    }

    #[test]
    fn ik_fk_roundtrip() {
        let mut robot = build_scara();
        let q_orig = [0.3_f64, -0.5, 0.05, 0.2];
        robot.set_joints(q_orig);
        let (x, y, z, psi) = robot.forward();
        let q_sol = robot.inverse(x, y, z, psi).expect("IK should succeed");
        // Re-run FK with solved joints
        robot.set_joints(q_sol);
        let (x2, y2, z2, psi2) = robot.forward();
        assert!((x2 - x).abs() < 1e-6, "x error: {} vs {}", x2, x);
        assert!((y2 - y).abs() < 1e-6, "y error: {} vs {}", y2, y);
        assert!((z2 - z).abs() < 1e-6, "z error: {} vs {}", z2, z);
        assert!((psi2 - psi).abs() < 1e-6, "psi error: {} vs {}", psi2, psi);
    }

    #[test]
    fn unreachable_target_returns_none() {
        let robot = build_scara();
        // Target beyond arm reach (a1+a2=0.5)
        let result = robot.inverse(0.6, 0.0, 0.45, 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn jacobian_shape() {
        let mut robot = build_scara();
        robot.set_joints([0.3, 0.5, 0.0, 0.0]);
        let j = robot.jacobian();
        // z row should be [0, 0, -1, 0]
        assert!((j[2][2] - (-1.0)).abs() < 1e-10);
        assert!(j[2][0].abs() < 1e-10);
    }
}
