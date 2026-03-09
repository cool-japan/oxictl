use crate::core::scalar::ControlScalar;

/// Geometric Jacobian for a 2-DOF planar revolute robot.
///
/// Maps joint velocities to end-effector velocities:
///   [ẋ]   = J(q) * [q̇1]
///   [ẏ]             [q̇2]
///
/// J = [[-L1*sin(q1) - L2*sin(q1+q2),  -L2*sin(q1+q2)],
///      [ L1*cos(q1) + L2*cos(q1+q2),   L2*cos(q1+q2)]]
pub struct Jacobian2R<S: ControlScalar> {
    /// Link 1 length.
    pub l1: S,
    /// Link 2 length.
    pub l2: S,
}

impl<S: ControlScalar> Jacobian2R<S> {
    pub fn new(l1: S, l2: S) -> Self {
        Self { l1, l2 }
    }

    /// Compute Jacobian matrix at joint angles [q1, q2].
    ///
    /// Returns [[J11, J12], [J21, J22]] (row-major).
    pub fn compute(&self, q1: S, q2: S) -> [[S; 2]; 2] {
        let s1 = q1.sin();
        let c1 = q1.cos();
        let s12 = (q1 + q2).sin();
        let c12 = (q1 + q2).cos();

        [
            [-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
            [self.l1 * c1 + self.l2 * c12, self.l2 * c12],
        ]
    }

    /// Determinant of J (manipulability measure).
    ///
    /// det(J) = L1*L2*sin(q2). Singularity when q2 = 0 or π.
    pub fn determinant(&self, _q1: S, q2: S) -> S {
        self.l1 * self.l2 * q2.sin()
    }

    /// Map joint velocities to Cartesian velocities.
    pub fn apply(&self, q1: S, q2: S, dq1: S, dq2: S) -> (S, S) {
        let j = self.compute(q1, q2);
        let vx = j[0][0] * dq1 + j[0][1] * dq2;
        let vy = j[1][0] * dq1 + j[1][1] * dq2;
        (vx, vy)
    }

    /// Pseudo-inverse Jacobian: J^+ = J^T * (J * J^T)^{-1} for a 2×2 full-rank J.
    ///
    /// Returns `None` if near singular (|det| < threshold).
    pub fn pseudo_inverse(&self, q1: S, q2: S) -> Option<[[S; 2]; 2]> {
        let j = self.compute(q1, q2);
        // det = j[0][0]*j[1][1] - j[0][1]*j[1][0]
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det.abs() < S::from_f64(1e-6) {
            return None;
        }
        let inv_det = S::ONE / det;
        Some([
            [j[1][1] * inv_det, -j[0][1] * inv_det],
            [-j[1][0] * inv_det, j[0][0] * inv_det],
        ])
    }

    /// Differential IK step: given Cartesian error (dx, dy), compute joint corrections.
    ///
    /// Uses J^+ * [dx, dy]. Returns `None` if near singular.
    pub fn ik_step(&self, q1: S, q2: S, dx: S, dy: S) -> Option<(S, S)> {
        let jp = self.pseudo_inverse(q1, q2)?;
        let dq1 = jp[0][0] * dx + jp[0][1] * dy;
        let dq2 = jp[1][0] * dx + jp[1][1] * dy;
        Some((dq1, dq2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobian_2r_at_zero_config() {
        let j = Jacobian2R::new(1.0_f64, 1.0);
        // At q1=0, q2=0: arm is fully extended along x
        let jac = j.compute(0.0, 0.0);
        // J = [[0, 0], [L1+L2, L2]] = [[0,0],[2,1]]
        assert!(jac[0][0].abs() < 1e-10);
        assert!(jac[0][1].abs() < 1e-10);
        assert!((jac[1][0] - 2.0).abs() < 1e-10);
        assert!((jac[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn determinant_formula() {
        let j = Jacobian2R::new(1.0_f64, 1.0);
        // det = L1*L2*sin(q2)
        let det = j.determinant(0.5, core::f64::consts::PI / 2.0);
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn singularity_at_zero_q2() {
        let j = Jacobian2R::new(1.0_f64, 1.0);
        // q2=0 → det=0 → pseudo_inverse returns None
        let result = j.pseudo_inverse(0.5, 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn pseudo_inverse_roundtrip() {
        let j = Jacobian2R::new(1.0_f64, 0.5);
        let q2 = core::f64::consts::PI / 3.0;
        let jp = j.pseudo_inverse(0.3, q2).unwrap();
        let jac = j.compute(0.3, q2);
        // J * J^+ should be ≈ I for 2×2 full-rank
        let ij00 = jac[0][0] * jp[0][0] + jac[0][1] * jp[1][0];
        let ij11 = jac[1][0] * jp[0][1] + jac[1][1] * jp[1][1];
        assert!((ij00 - 1.0).abs() < 1e-10, "ij00={}", ij00);
        assert!((ij11 - 1.0).abs() < 1e-10, "ij11={}", ij11);
    }
}
