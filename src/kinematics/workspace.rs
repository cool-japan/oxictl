//! Workspace analysis: reachability, dexterity, manipulability.
//!
//! Provides tools for analysing the reachable workspace and kinematic
//! performance of serial robot arms, including Yoshikawa manipulability
//! and condition number estimation.
#![allow(clippy::needless_range_loop)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Workspace analyser for serial robot arms with up to 3 links.
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceAnalyzer<S: ControlScalar> {
    /// Link lengths [L1, L2, L3].
    pub link_lengths: [S; 3],
    /// Joint lower limits [q1_min, q2_min, q3_min].
    pub q_min: [S; 3],
    /// Joint upper limits [q1_max, q2_max, q3_max].
    pub q_max: [S; 3],
}

impl<S: ControlScalar> WorkspaceAnalyzer<S> {
    /// Create a new workspace analyser.
    pub fn new(link_lengths: [S; 3], q_min: [S; 3], q_max: [S; 3]) -> Self {
        Self {
            link_lengths,
            q_min,
            q_max,
        }
    }

    /// Check if point (x, y, z) is geometrically reachable by a serial 3R robot.
    ///
    /// Reachability check: the distance r from the base must satisfy
    ///   |L1 - L2 - L3| ≤ r ≤ L1 + L2 + L3
    /// (ignores joint limits for a conservative check).
    pub fn is_reachable(&self, x: S, y: S, z: S) -> bool {
        let r_sq = x * x + y * y + z * z;
        let r = r_sq.sqrt();
        let max_r = self.max_reach();
        let min_r = self.min_reach();
        r >= min_r && r <= max_r
    }

    /// Yoshikawa manipulability index: w = sqrt(det(J · J^T)).
    ///
    /// A value near zero indicates a singular configuration.
    /// Works for any M×N Jacobian with M ≤ N.
    pub fn dexterity<const N: usize, const M: usize>(&self, j: &Matrix<S, M, N>) -> S {
        // Compute J * J^T (M×M)
        let jt = j.transpose();
        let jjt = matmul(j, &jt);
        // det via Gaussian elimination (reuse inv which returns None at singularity)
        let det = mat_det(&jjt);
        if det < S::ZERO {
            S::ZERO
        } else {
            det.sqrt()
        }
    }

    /// Condition number estimate σ_max / σ_min for a square N×N Jacobian.
    ///
    /// Uses power iteration to estimate the largest singular value, and
    /// inverse power iteration for the smallest.
    pub fn condition_number<const N: usize>(&self, j: &Matrix<S, N, N>) -> S {
        let jt = j.transpose();
        let jtj = matmul(&jt, j); // J^T * J (N×N), eigenvalues = σ_i^2

        let sigma_max = power_iteration_largest(&jtj, 30).sqrt();
        let sigma_min = inverse_power_iteration_smallest(&jtj, 30).sqrt();

        if sigma_min < S::EPSILON * S::from_f64(1e6) {
            return S::from_f64(1e12); // near-singular
        }
        sigma_max / sigma_min
    }

    /// Maximum reach = L1 + L2 + L3.
    pub fn max_reach(&self) -> S {
        self.link_lengths[0] + self.link_lengths[1] + self.link_lengths[2]
    }

    /// Minimum reach = |L1 - L2 - L3|.
    pub fn min_reach(&self) -> S {
        let inner = self.link_lengths[0] - self.link_lengths[1] - self.link_lengths[2];
        inner.abs()
    }
}

// ---------------------------------------------------------------------------
// Helper: determinant of M×M matrix via expansion / Gaussian elimination
// ---------------------------------------------------------------------------

/// Compute determinant of an M×M matrix via Gaussian elimination.
fn mat_det<S: ControlScalar, const M: usize>(m: &Matrix<S, M, M>) -> S {
    // Copy data into a mutable array
    let mut a = m.data;
    let mut det = S::ONE;
    for col in 0..M {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..M {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < S::EPSILON * S::from_f64(1e6) {
            return S::ZERO;
        }
        if max_row != col {
            a.swap(max_row, col);
            det = det.neg(); // swap negates determinant
        }
        let pivot = a[col][col];
        det *= pivot;
        let inv_pivot = S::ONE / pivot;
        for row in (col + 1)..M {
            let factor = a[row][col] * inv_pivot;
            for c in col..M {
                let tmp = a[col][c];
                a[row][c] -= factor * tmp;
            }
        }
    }
    det
}

/// Power iteration to find largest eigenvalue of a symmetric positive semi-definite matrix.
fn power_iteration_largest<S: ControlScalar, const N: usize>(
    m: &Matrix<S, N, N>,
    iters: usize,
) -> S {
    if N == 0 {
        return S::ZERO;
    }
    // Start with all-ones vector
    let mut v: [S; N] = core::array::from_fn(|_| S::ONE);
    let mut lambda = S::ONE;
    for _ in 0..iters {
        // w = M * v
        let mut w: [S; N] = [S::ZERO; N];
        for i in 0..N {
            for j in 0..N {
                w[i] += m.data[i][j] * v[j];
            }
        }
        // norm
        let mut norm_sq = S::ZERO;
        for i in 0..N {
            norm_sq += w[i] * w[i];
        }
        let norm = norm_sq.sqrt();
        if norm < S::EPSILON {
            return S::ZERO;
        }
        lambda = norm;
        for i in 0..N {
            v[i] = w[i] / norm;
        }
    }
    lambda
}

/// Inverse power iteration to find smallest eigenvalue of a symmetric PSD matrix.
/// Uses shift + inverse; falls back to a small value if singular.
fn inverse_power_iteration_smallest<S: ControlScalar, const N: usize>(
    m: &Matrix<S, N, N>,
    iters: usize,
) -> S {
    if N == 0 {
        return S::ZERO;
    }
    let m_inv = match m.inv() {
        Some(inv) => inv,
        None => return S::ZERO,
    };
    // Largest eigenvalue of M^{-1} = 1 / (smallest eigenvalue of M)
    let lambda_inv_max = power_iteration_largest(&m_inv, iters);
    if lambda_inv_max < S::EPSILON {
        return S::ZERO;
    }
    S::ONE / lambda_inv_max
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyzer() -> WorkspaceAnalyzer<f64> {
        WorkspaceAnalyzer::new(
            [1.0, 1.0, 1.0],
            [
                -core::f64::consts::PI,
                -core::f64::consts::PI,
                -core::f64::consts::PI,
            ],
            [
                core::f64::consts::PI,
                core::f64::consts::PI,
                core::f64::consts::PI,
            ],
        )
    }

    #[test]
    fn max_reach_is_sum() {
        let a = analyzer();
        assert!((a.max_reach() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn min_reach_is_abs() {
        let a = WorkspaceAnalyzer::new(
            [2.0_f64, 1.0, 0.5],
            [-core::f64::consts::PI; 3],
            [core::f64::consts::PI; 3],
        );
        // |2 - 1 - 0.5| = 0.5
        assert!((a.min_reach() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reachable_point_origin_is_not_reachable_for_equal_links() {
        let a = analyzer(); // all links = 1 → min_reach = |1-1-1| = 1
                            // Origin distance = 0 < min_reach = 1 → not reachable
        assert!(!a.is_reachable(0.0, 0.0, 0.0));
    }

    #[test]
    fn reachable_point_at_max_reach() {
        let a = analyzer(); // max_reach = 3
        assert!(a.is_reachable(3.0, 0.0, 0.0));
    }

    #[test]
    fn dexterity_identity_jacobian() {
        let a = analyzer();
        let j = Matrix::<f64, 2, 2>::identity();
        let w = a.dexterity::<2, 2>(&j);
        // det(I*I^T) = det(I) = 1; sqrt = 1
        assert!((w - 1.0).abs() < 1e-9, "w={w}");
    }

    #[test]
    fn condition_number_identity_is_one() {
        let a = analyzer();
        let j = Matrix::<f64, 2, 2>::identity();
        let cond = a.condition_number::<2>(&j);
        assert!((cond - 1.0).abs() < 1e-6, "cond={cond}");
    }
}

// =============================================================================
// DH-based workspace analysis
// =============================================================================

use crate::kinematics::forward::Transform3D;
use crate::kinematics::serial::six_dof::DhParam;

/// DH parameter set for a single joint — re-exported convenience alias.
pub type DhParams<S> = DhParam<S>;

/// N-joint DH configuration.
#[derive(Debug, Clone, Copy)]
pub struct DhConfig<S: ControlScalar, const N: usize> {
    /// DH parameters for each joint.
    pub params: [DhParams<S>; N],
}

impl<S: ControlScalar, const N: usize> DhConfig<S, N> {
    /// Create a new DH configuration.
    pub fn new(params: [DhParams<S>; N]) -> Self {
        Self { params }
    }
}

/// Axis-aligned bounding box representing the approximate reachable workspace.
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceBounds<S: ControlScalar> {
    /// Minimum coordinate along each axis [x_min, y_min, z_min].
    pub min: [S; 3],
    /// Maximum coordinate along each axis [x_max, y_max, z_max].
    pub max: [S; 3],
}

impl<S: ControlScalar> WorkspaceBounds<S> {
    fn empty() -> Self {
        let big = S::from_f64(f64::MAX / 2.0);
        Self {
            min: [big; 3],
            max: [-big; 3],
        }
    }

    /// Expand bounds to include point `p`.
    fn include(&mut self, p: &[S; 3]) {
        for i in 0..3 {
            if p[i] < self.min[i] {
                self.min[i] = p[i];
            }
            if p[i] > self.max[i] {
                self.max[i] = p[i];
            }
        }
    }

    /// Diagonal extent of the workspace bounding box.
    pub fn diagonal(&self) -> S {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Standard DH homogeneous transform  (replicates Robot6Dof::dh_matrix but as
// Transform3D for compatibility with the kinematics::forward module).
// ---------------------------------------------------------------------------

/// Compute the forward kinematics [`Transform3D`] from a DH configuration at
/// joint angles `q`.
///
/// Uses the **standard DH** convention:
///   `T_i = Rz(θ_i + offset) · Tz(d) · Tx(a) · Rx(α)`
///
/// # Panics
/// Never panics.  All trig computed via `num_traits::Float`.
pub fn fk_from_dh<S: ControlScalar, const N: usize>(
    config: &DhConfig<S, N>,
    q: &[S; N],
) -> Transform3D<S> {
    let mut result = Transform3D::identity();
    for i in 0..N {
        let p = &config.params[i];
        let theta = q[i] + p.theta_offset;
        let ct = theta.cos();
        let st = theta.sin();
        let ca = p.alpha.cos();
        let sa = p.alpha.sin();

        // Rotation sub-matrix of the DH transform
        let r = [
            [ct, -st * ca, st * sa],
            [st, ct * ca, -ct * sa],
            [S::ZERO, sa, ca],
        ];
        // Translation column
        let t = [p.a * ct, p.a * st, p.d];

        let step = Transform3D { r, t };
        result = result.compose(&step);
    }
    result
}

/// Approximate reachable workspace by sampling joint angles on a uniform grid.
///
/// For each joint `i`, `samples` equally-spaced angles are drawn from
/// `[q_limits[i].0, q_limits[i].1]` (no random numbers — fully deterministic).
///
/// **Warning**: the total number of forward-kinematics evaluations is
/// `samples^N`, so keep `samples` small (e.g. 5–10) for large `N`.
///
/// # Arguments
/// - `config`    – DH parameter set.
/// - `q_limits`  – `(q_min, q_max)` for each joint.
/// - `samples`   – number of samples per joint axis.
///
/// # Returns
/// Axis-aligned [`WorkspaceBounds`] enclosing all sampled end-effector
/// positions, or `None` if `samples == 0` or all limits are degenerate.
pub fn workspace_reachability<S: ControlScalar, const N: usize>(
    config: &DhConfig<S, N>,
    q_limits: &[(S, S); N],
    samples: usize,
) -> Option<WorkspaceBounds<S>> {
    if samples == 0 || N == 0 {
        return None;
    }

    // Pre-compute sample grids for each joint
    let mut grids: [[S; 32]; N] = [[S::ZERO; 32]; N];
    let actual_samples = samples.min(32);
    for i in 0..N {
        let (lo, hi) = q_limits[i];
        if actual_samples == 1 {
            grids[i][0] = (lo + hi) * S::HALF;
        } else {
            let step = (hi - lo) / S::from_f64((actual_samples - 1) as f64);
            for k in 0..actual_samples {
                grids[i][k] = lo + step * S::from_f64(k as f64);
            }
        }
    }

    let mut bounds = WorkspaceBounds::empty();
    let total: usize = actual_samples.pow(N as u32);
    let mut found_any = false;

    for idx in 0..total {
        // Decode multi-index from flat index
        let mut q = [S::ZERO; N];
        let mut remaining = idx;
        for i in (0..N).rev() {
            let digit = remaining % actual_samples;
            remaining /= actual_samples;
            q[i] = grids[i][digit];
        }

        let tf = fk_from_dh(config, &q);
        bounds.include(&tf.t);
        found_any = true;
    }

    if found_any {
        Some(bounds)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests for DH workspace utilities
// ---------------------------------------------------------------------------

#[cfg(test)]
mod dh_workspace_tests {
    use super::*;
    use crate::kinematics::serial::six_dof::robot6_ur5_like;

    fn ur5_dh_config() -> DhConfig<f64, 6> {
        let robot = robot6_ur5_like();
        DhConfig::new(robot.links)
    }

    #[test]
    fn fk_from_dh_zero_matches_robot6dof() {
        use crate::kinematics::serial::six_dof::robot6_ur5_like;
        let robot = robot6_ur5_like();
        let t_robot = robot.forward();

        let config = ur5_dh_config();
        let q = [0.0_f64; 6];
        let tf = fk_from_dh(&config, &q);

        // Compare translation
        assert!(
            (tf.t[0] - t_robot[0][3]).abs() < 1e-10,
            "x: {} vs {}",
            tf.t[0],
            t_robot[0][3]
        );
        assert!(
            (tf.t[1] - t_robot[1][3]).abs() < 1e-10,
            "y: {} vs {}",
            tf.t[1],
            t_robot[1][3]
        );
        assert!(
            (tf.t[2] - t_robot[2][3]).abs() < 1e-10,
            "z: {} vs {}",
            tf.t[2],
            t_robot[2][3]
        );
    }

    #[test]
    fn workspace_reachability_samples_zero_returns_none() {
        let config = ur5_dh_config();
        let limits = [(-core::f64::consts::PI, core::f64::consts::PI); 6];
        assert!(workspace_reachability(&config, &limits, 0).is_none());
    }

    #[test]
    fn workspace_reachability_single_sample() {
        let config = ur5_dh_config();
        let limits = [(0.0_f64, 0.0_f64); 6];
        let bounds = workspace_reachability(&config, &limits, 1)
            .expect("single sample should produce bounds");
        // With q all zero, min == max (single point)
        for i in 0..3 {
            assert!(
                (bounds.min[i] - bounds.max[i]).abs() < 1e-9,
                "min[{i}]={} max[{i}]={}",
                bounds.min[i],
                bounds.max[i]
            );
        }
    }

    #[test]
    fn workspace_reachability_nonzero_extent() {
        let config = ur5_dh_config();
        let limits = [(-core::f64::consts::PI, core::f64::consts::PI); 6];
        let bounds = workspace_reachability(&config, &limits, 4).expect("should find workspace");
        let diag = bounds.diagonal();
        assert!(diag > 0.1, "workspace diagonal={diag:.4} should be nonzero");
    }

    #[test]
    fn workspace_bounds_diagonal_correct() {
        let mut b = WorkspaceBounds::<f64>::empty();
        b.include(&[0.0, 0.0, 0.0]);
        b.include(&[1.0, 1.0, 1.0]);
        let d = b.diagonal();
        assert!((d - 3.0_f64.sqrt()).abs() < 1e-10, "diag={d}");
    }

    #[test]
    fn dh_config_new_roundtrip() {
        let robot = robot6_ur5_like();
        let config = DhConfig::new(robot.links);
        assert!((config.params[0].d - robot.links[0].d).abs() < 1e-12);
        assert!((config.params[1].a - robot.links[1].a).abs() < 1e-12);
    }
}
