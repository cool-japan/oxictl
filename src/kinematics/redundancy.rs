//! Redundancy resolution via null-space projection.
//!
//! For redundant manipulators (N joints > M task-space DOF), the pseudo-inverse
//! maps task-space velocities to a particular joint-space solution. The null-space
//! projector (I - J†J) allows secondary objectives (joint limit avoidance, etc.)
//! to be pursued without affecting the primary task.
#![allow(clippy::needless_range_loop)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Null-space projector for redundant manipulators (N > M DOF).
///
/// N: joint-space dimension (> M), M: task-space dimension.
pub struct NullSpaceProjector<S: ControlScalar, const N: usize, const M: usize> {
    /// Jacobian J (M×N).
    pub j: Matrix<S, M, N>,
    /// Moore-Penrose pseudo-inverse J† (N×M).
    pub j_pinv: Matrix<S, N, M>,
}

impl<S: ControlScalar, const N: usize, const M: usize> NullSpaceProjector<S, N, M> {
    /// Create a new null-space projector from Jacobian J.
    pub fn new(j: Matrix<S, M, N>) -> Self {
        let j_pinv = Self::compute_pinv(&j);
        Self { j, j_pinv }
    }

    /// Update Jacobian and recompute pseudo-inverse.
    pub fn update_jacobian(&mut self, j: Matrix<S, M, N>) {
        self.j = j;
        self.j_pinv = Self::compute_pinv(&j);
    }

    /// Primary task joint velocity: dq_primary = J† · dx
    pub fn dq_primary(&self, dx: &Matrix<S, M, 1>) -> Matrix<S, N, 1> {
        matmul(&self.j_pinv, dx)
    }

    /// Null-space component: dq_null = (I - J†·J) · w
    pub fn dq_null(&self, w: &Matrix<S, N, 1>) -> Matrix<S, N, 1> {
        let eye = Matrix::<S, N, N>::identity();
        let jpinv_j = matmul(&self.j_pinv, &self.j); // N×N
        let null_proj = eye.sub_mat(&jpinv_j); // (I - J†J)
        matmul(&null_proj, w)
    }

    /// Combined joint velocity: dq = dq_primary + dq_null
    pub fn dq(&self, dx: &Matrix<S, M, 1>, w: &Matrix<S, N, 1>) -> Matrix<S, N, 1> {
        let primary = self.dq_primary(dx);
        let null = self.dq_null(w);
        primary.add_mat(&null)
    }

    /// Gradient of joint-limit avoidance cost: ∂H/∂q_i = (q_i - q_mid_i) / range_i^2.
    ///
    /// Used as the secondary objective `w` in null-space projection.
    pub fn joint_limit_gradient(q: &[S; N], q_min: &[S; N], q_max: &[S; N]) -> Matrix<S, N, 1> {
        let mut grad = Matrix::<S, N, 1>::zeros();
        for i in 0..N {
            let range = q_max[i] - q_min[i];
            if range < S::EPSILON {
                continue;
            }
            let mid = (q_max[i] + q_min[i]) * S::HALF;
            // Negative gradient (want to minimise cost, so move toward centre)
            let g = (q[i] - mid) / (range * range);
            grad.data[i][0] = -g;
        }
        grad
    }

    /// Compute Moore-Penrose pseudo-inverse: J† = J^T · (J·J^T)^{-1}
    /// for a full row-rank M×N Jacobian (M ≤ N).
    fn compute_pinv(j: &Matrix<S, M, N>) -> Matrix<S, N, M> {
        let jt = j.transpose(); // N×M
        let jjt = matmul(j, &jt); // M×M
        match jjt.inv() {
            Some(jjt_inv) => matmul(&jt, &jjt_inv), // N×M × M×M = N×M
            None => {
                // Fallback: return zero pseudo-inverse (singular)
                Matrix::<S, N, M>::zeros()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 2×3 Jacobian for a planar 3R robot at a generic config.
    fn planar_3r_jacobian() -> Matrix<f64, 2, 3> {
        // Rows: [dx/dq1, dx/dq2, dx/dq3], [dy/dq1, dy/dq2, dy/dq3]
        // For links of length 1 at q = [0,0,0]:
        //   J = [[3, 2, 1], [0, 0, 0]]  (degenerate; use non-zero config)
        // At q = [0.1, 0.2, 0.3]:
        //   approximate via numeric — use a hard-coded realistic value
        Matrix::<f64, 2, 3> {
            data: [[-0.5, -0.3, -0.1], [0.8, 0.6, 0.2]],
        }
    }

    #[test]
    fn primary_task_reconstructs_dx() {
        let j = planar_3r_jacobian();
        let proj = NullSpaceProjector::new(j);
        let dx = Matrix::<f64, 2, 1> {
            data: [[0.1], [0.2]],
        };
        let dq = proj.dq_primary(&dx);
        // J * dq should ≈ dx
        let recovered = matmul(&proj.j, &dq);
        assert!(
            (recovered.data[0][0] - dx.data[0][0]).abs() < 1e-10,
            "recovered[0]={}",
            recovered.data[0][0]
        );
        assert!(
            (recovered.data[1][0] - dx.data[1][0]).abs() < 1e-10,
            "recovered[1]={}",
            recovered.data[1][0]
        );
    }

    #[test]
    fn null_space_does_not_affect_task() {
        let j = planar_3r_jacobian();
        let proj = NullSpaceProjector::new(j);
        let dx = Matrix::<f64, 2, 1> {
            data: [[0.05], [0.1]],
        };
        let w = Matrix::<f64, 3, 1> {
            data: [[1.0], [-1.0], [0.5]],
        };
        let dq = proj.dq(&dx, &w);
        // J * dq should still ≈ dx (null-space doesn't contribute to J*dq)
        let recovered = matmul(&proj.j, &dq);
        assert!(
            (recovered.data[0][0] - dx.data[0][0]).abs() < 1e-10,
            "task error x: {}",
            (recovered.data[0][0] - dx.data[0][0]).abs()
        );
        assert!(
            (recovered.data[1][0] - dx.data[1][0]).abs() < 1e-10,
            "task error y: {}",
            (recovered.data[1][0] - dx.data[1][0]).abs()
        );
    }

    #[test]
    fn joint_limit_gradient_points_toward_centre() {
        let q = [0.8_f64; 3];
        let q_min = [0.0_f64; 3];
        let q_max = [1.0_f64; 3];
        let grad = NullSpaceProjector::<f64, 3, 2>::joint_limit_gradient(&q, &q_min, &q_max);
        // q > mid → gradient should be negative (pushing back toward centre)
        for i in 0..3 {
            assert!(grad.data[i][0] < 0.0, "grad[{i}]={}", grad.data[i][0]);
        }
    }

    #[test]
    fn update_jacobian_recomputes_pinv() {
        let j_init = planar_3r_jacobian();
        let mut proj = NullSpaceProjector::new(j_init);
        let pinv_before = proj.j_pinv;
        // Update with a scaled Jacobian
        let j_new = Matrix::<f64, 2, 3> {
            data: [[-1.0, -0.6, -0.2], [1.6, 1.2, 0.4]],
        };
        proj.update_jacobian(j_new);
        // Pinv should have changed
        let changed = pinv_before.data[0][0] != proj.j_pinv.data[0][0]
            || pinv_before.data[1][0] != proj.j_pinv.data[1][0];
        assert!(changed, "j_pinv did not change after update");
    }
}
