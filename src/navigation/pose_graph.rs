//! Pose Graph Optimisation — linear 2D least squares.
//!
//! A pose graph represents robot poses (nodes) and relative-pose measurements
//! between them (edges). This implementation uses a simple sequential forward
//! pass for the main chain and distributes loop-closure errors linearly.
//!
//! # Nodes
//! Each node stores a 2D pose `[x, y, theta]`. Nodes are numbered `0..NODES`.
//!
//! # Edges
//! Edges carry a relative measurement `[dx, dy, dtheta]` and a scalar
//! information weight. The graph supports up to 256 edges (heapless::Vec).
//!
//! # Optimisation
//! `optimize_sequential` performs:
//! 1. Forward integration of consecutive-node edges to propagate poses.
//! 2. Loop-closure correction: if an edge from node `i` to node `j` with
//!    `j < i` exists (a loop), the accumulated error is spread evenly across
//!    all intermediate nodes.
//!
//! Returns the total weighted residual.

use crate::core::scalar::ControlScalar;
use crate::navigation::dead_reckoning::NavigationError;
use heapless::Vec as HVec;

/// A pose graph for `NODES` robot poses in 2D.
///
/// # Type parameters
/// - `S`     — scalar type implementing [`ControlScalar`].
/// - `NODES` — compile-time number of pose nodes (must be ≥ 1).
pub struct PoseGraph<S: ControlScalar, const NODES: usize> {
    /// Robot poses; each is `[x, y, theta]`.
    poses: [[S; 3]; NODES],
    /// Edges: `(from_node, to_node, measurement [dx, dy, dtheta], info_weight)`.
    edges: HVec<(usize, usize, [S; 3], S), 256>,
}

/// Errors produced by the pose graph module.
pub use crate::navigation::dead_reckoning::NavigationError as PoseGraphError;

impl<S: ControlScalar, const NODES: usize> PoseGraph<S, NODES> {
    /// Create a new pose graph with all poses initialised to the origin.
    pub fn new() -> Self {
        Self {
            poses: [[S::ZERO; 3]; NODES],
            edges: HVec::new(),
        }
    }

    /// Set the pose of node `node` to `pose = [x, y, theta]`.
    ///
    /// # Errors
    /// Returns [`NavigationError::InvalidNode`] if `node >= NODES`.
    pub fn set_pose(&mut self, node: usize, pose: [S; 3]) -> Result<(), NavigationError> {
        if node >= NODES {
            return Err(NavigationError::InvalidNode);
        }
        self.poses[node] = pose;
        Ok(())
    }

    /// Add an edge `from → to` with relative measurement `meas` and information
    /// weight `info` (higher = more trustworthy).
    ///
    /// # Errors
    /// - [`NavigationError::InvalidNode`] if either node index is out of range.
    /// - [`NavigationError::TooManyEdges`] if the edge buffer is full (> 256).
    pub fn add_edge(
        &mut self,
        from: usize,
        to: usize,
        meas: [S; 3],
        info: S,
    ) -> Result<(), NavigationError> {
        if from >= NODES || to >= NODES {
            return Err(NavigationError::InvalidNode);
        }
        self.edges
            .push((from, to, meas, info))
            .map_err(|_| NavigationError::TooManyEdges)
    }

    /// Optimise the pose graph (sequential + loop-closure correction).
    ///
    /// **Algorithm:**
    /// 1. Collect sequential forward edges (`i → i+1`). Integrate them in
    ///    order starting from node 0 to propagate poses.
    /// 2. For any loop-closure edge (`from → to` where `to ≤ from`), compute
    ///    the error between the predicted pose of `to` and the measurement.
    ///    Distribute the error linearly across nodes `to..=from`.
    /// 3. Compute total weighted residual: `sum_edges info * ||residual||^2`.
    ///
    /// Returns the total weighted squared residual.
    ///
    /// # Errors
    /// Returns [`NavigationError::SingularSystem`] if the graph has no edges.
    pub fn optimize_sequential(&mut self) -> Result<S, NavigationError> {
        if self.edges.is_empty() {
            return Err(NavigationError::SingularSystem);
        }

        // ----------------------------------------------------------------
        // Pass 1: integrate sequential edges (from+1 == to) in order.
        // ----------------------------------------------------------------
        // Collect sequential edges sorted by `from` index.
        let mut seq_edges: HVec<(usize, [S; 3], S), 256> = HVec::new();
        for &(from, to, meas, info) in &self.edges {
            if to == from + 1 {
                // Safe push — same capacity bound.
                let _ = seq_edges.push((from, meas, info));
            }
        }

        // Sort by `from` (insertion sort — no_std, small N).
        let n = seq_edges.len();
        for i in 1..n {
            let mut j = i;
            while j > 0 && seq_edges[j - 1].0 > seq_edges[j].0 {
                seq_edges.swap(j - 1, j);
                j -= 1;
            }
        }

        // Forward integration.
        for &(from, meas, _info) in &seq_edges {
            if from + 1 < NODES {
                self.poses[from + 1] = Self::compose_poses(self.poses[from], meas);
            }
        }

        // ----------------------------------------------------------------
        // Pass 2: loop-closure correction.
        // ----------------------------------------------------------------
        for &(from, to, meas, _info) in &self.edges {
            if to < from {
                // This is a loop-closure edge from `from` back to `to`.
                // Predicted pose of `to` from integrating forward from `from`.
                let predicted = Self::compose_poses(self.poses[from], meas);
                let actual = self.poses[to];

                // Error = actual - predicted.
                let err_x = actual[0] - predicted[0];
                let err_y = actual[1] - predicted[1];
                let err_theta = Self::wrap_angle(actual[2] - predicted[2]);

                // Distribute error evenly over nodes `to..=from` (exclusive of anchor 0).
                let span = (from - to) as f64;
                if span < 1.0 {
                    continue;
                }
                let span_s = S::from_f64(span + 1.0);
                for node in to..=from {
                    let weight = S::from_f64((node - to) as f64) / span_s;
                    self.poses[node][0] += err_x * weight;
                    self.poses[node][1] += err_y * weight;
                    self.poses[node][2] =
                        Self::wrap_angle(self.poses[node][2] + err_theta * weight);
                }
            }
        }

        // ----------------------------------------------------------------
        // Compute total weighted residual.
        // ----------------------------------------------------------------
        let mut total = S::ZERO;
        for &(from, to, meas, info) in &self.edges {
            let predicted = Self::compose_poses(self.poses[from], meas);
            let actual = self.poses[to];
            let ex = actual[0] - predicted[0];
            let ey = actual[1] - predicted[1];
            let eth = Self::wrap_angle(actual[2] - predicted[2]);
            total += info * (ex * ex + ey * ey + eth * eth);
        }

        Ok(total)
    }

    /// Return the pose of node `node` as `[x, y, theta]`.
    ///
    /// # Errors
    /// Returns [`NavigationError::InvalidNode`] if `node >= NODES`.
    pub fn pose(&self, node: usize) -> Result<[S; 3], NavigationError> {
        if node >= NODES {
            return Err(NavigationError::InvalidNode);
        }
        Ok(self.poses[node])
    }

    // -----------------------------------------------------------------------
    // Internal helpers.
    // -----------------------------------------------------------------------

    /// Compose two poses: `world_pose ⊕ relative_meas → result_pose`.
    ///
    /// The relative measurement is expressed in the frame of `base`.
    fn compose_poses(base: [S; 3], rel: [S; 3]) -> [S; 3] {
        let th = base[2];
        let cos_th = S::from_f64(libm::cos(th.to_f64()));
        let sin_th = S::from_f64(libm::sin(th.to_f64()));
        let x = base[0] + cos_th * rel[0] - sin_th * rel[1];
        let y = base[1] + sin_th * rel[0] + cos_th * rel[1];
        let theta = Self::wrap_angle(base[2] + rel[2]);
        [x, y, theta]
    }

    /// Wrap angle to `(-pi, pi]`.
    fn wrap_angle(a: S) -> S {
        let mut v = a.to_f64();
        use core::f64::consts::PI;
        while v > PI {
            v -= 2.0 * PI;
        }
        while v <= -PI {
            v += 2.0 * PI;
        }
        S::from_f64(v)
    }
}

impl<S: ControlScalar, const NODES: usize> Default for PoseGraph<S, NODES> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sequential chain integrates measurements correctly.
    #[test]
    fn sequential_chain_integrates() {
        let mut pg = PoseGraph::<f64, 4>::new();
        // All edges: move 1 m in x, no rotation.
        pg.add_edge(0, 1, [1.0, 0.0, 0.0], 1.0).unwrap();
        pg.add_edge(1, 2, [1.0, 0.0, 0.0], 1.0).unwrap();
        pg.add_edge(2, 3, [1.0, 0.0, 0.0], 1.0).unwrap();
        pg.optimize_sequential().unwrap();

        let p1 = pg.pose(1).unwrap();
        let p2 = pg.pose(2).unwrap();
        let p3 = pg.pose(3).unwrap();
        assert!((p1[0] - 1.0).abs() < 1e-9, "p1.x: {}", p1[0]);
        assert!((p2[0] - 2.0).abs() < 1e-9, "p2.x: {}", p2[0]);
        assert!((p3[0] - 3.0).abs() < 1e-9, "p3.x: {}", p3[0]);
    }

    /// Loop closure: error is distributed across nodes.
    ///
    /// Set up: three odometry edges that perfectly integrate to node 3 at (3,0).
    /// The loop-closure edge claims node 3 → node 0 with measurement [-2.5,0,0],
    /// i.e. the loop says the total path is only 2.5 m, not 3 m.
    /// The optimiser must distribute the 0.5 m discrepancy across nodes 0..=3.
    #[test]
    fn loop_closure_distributes_error() {
        let mut pg = PoseGraph::<f64, 4>::new();
        // Three forward edges (perfect odometry: each +1 m in x).
        pg.add_edge(0, 1, [1.0, 0.0, 0.0], 1.0).unwrap();
        pg.add_edge(1, 2, [1.0, 0.0, 0.0], 1.0).unwrap();
        pg.add_edge(2, 3, [1.0, 0.0, 0.0], 1.0).unwrap();
        // Loop-closure: from node 3 back to node 0 with measurement [-2.5, 0, 0].
        // After integration, poses[3] = [3, 0, 0].
        // predicted = compose_poses([3,0,0], [-2.5,0,0]) = [0.5, 0, 0].
        // actual    = poses[0]                           = [0.0, 0, 0].
        // err_x = 0.0 - 0.5 = -0.5. Distributed over nodes 0..=3 with
        // weights 0/4, 1/4, 2/4, 3/4 → nodes shift by 0, -0.125, -0.25, -0.375.
        // Node 3 ends at 3.0 - 0.375 = 2.625 < 3.0.
        pg.add_edge(3, 0, [-2.5, 0.0, 0.0], 1.0).unwrap();
        pg.optimize_sequential().unwrap();

        // After correction, node 3 should be pulled below 3.0.
        let p3 = pg.pose(3).unwrap();
        assert!(
            p3[0] < 3.0,
            "loop closure should pull node 3 below 3.0: got {}",
            p3[0]
        );
        // Verify the shift direction is toward the consistent closure.
        assert!(
            p3[0] > 1.0,
            "node 3 should not move past node 1: got {}",
            p3[0]
        );
    }

    /// Invalid node index returns error.
    #[test]
    fn invalid_node_returns_error() {
        let mut pg = PoseGraph::<f64, 3>::new();
        let result = pg.set_pose(5, [0.0, 0.0, 0.0]);
        assert_eq!(result.unwrap_err(), NavigationError::InvalidNode);
    }

    /// Consistent graph (edges match poses) → residual near zero.
    #[test]
    fn consistent_graph_zero_residual() {
        let mut pg = PoseGraph::<f64, 3>::new();
        pg.set_pose(0, [0.0, 0.0, 0.0]).unwrap();
        pg.set_pose(1, [1.0, 0.0, 0.0]).unwrap();
        pg.set_pose(2, [2.0, 0.0, 0.0]).unwrap();
        pg.add_edge(0, 1, [1.0, 0.0, 0.0], 1.0).unwrap();
        pg.add_edge(1, 2, [1.0, 0.0, 0.0], 1.0).unwrap();
        let residual = pg.optimize_sequential().unwrap();
        assert!(
            residual < 1e-9,
            "consistent graph residual should be ~0: {}",
            residual
        );
    }

    /// No edges → SingularSystem error.
    #[test]
    fn no_edges_returns_singular_error() {
        let mut pg = PoseGraph::<f64, 3>::new();
        let result = pg.optimize_sequential();
        assert_eq!(result.unwrap_err(), NavigationError::SingularSystem);
    }

    /// Pose query for out-of-range node returns error.
    #[test]
    fn pose_query_invalid_node() {
        let pg = PoseGraph::<f64, 3>::new();
        assert_eq!(pg.pose(10).unwrap_err(), NavigationError::InvalidNode);
    }

    /// Edge with out-of-range node returns error.
    #[test]
    fn add_edge_invalid_node_returns_error() {
        let mut pg = PoseGraph::<f64, 3>::new();
        let result = pg.add_edge(0, 10, [1.0, 0.0, 0.0], 1.0);
        assert_eq!(result.unwrap_err(), NavigationError::InvalidNode);
    }
}
