//! Multi-agent consensus protocols over fixed weighted graphs.
//!
//! Consensus algorithms drive a network of agents to agree on a common value
//! by exchanging information only with local neighbours.  This module
//! provides three families of discrete-time consensus protocols:
//!
//! - **Leaderless average consensus** — converges to the mean of the initial
//!   agent states.
//! - **Leader-following consensus** — drives all agents to track an exogenous
//!   leader state via pinning gains.
//! - **Distributed gradient descent (ADMM-lite)** — each agent minimises a
//!   private objective while the consensus constraint is enforced via a
//!   gradient-plus-consensus step.
//!
//! All protocols are parameterised by a static weighted graph stored as a
//! symmetric adjacency matrix.  No heap allocation is used.
#![allow(clippy::needless_range_loop)]
use crate::core::matrix::Matrix;
use crate::core::scalar::ControlScalar;
use crate::networked::NetworkedError;

// ──────────────────────────────────────────────────────────────────────────────
// AgentGraph — weighted graph topology
// ──────────────────────────────────────────────────────────────────────────────

/// Fixed-size weighted undirected graph with `N` agents.
///
/// The graph is stored as a symmetric adjacency weight matrix `W` where
/// `W[i][j] = a_ij` is the weight of the edge between agents `i` and `j`.
/// Diagonal entries must be zero (no self-loops).
///
/// # Invariants
/// - `W[i][j] = W[j][i]` (symmetry).
/// - `W[i][i] = 0` (no self-loops).
/// - `W[i][j] ≥ 0` (non-negative weights).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentGraph<S: ControlScalar, const N: usize> {
    /// Adjacency weight matrix.
    weights: Matrix<S, N, N>,
}

impl<S: ControlScalar, const N: usize> AgentGraph<S, N> {
    /// Construct a graph from a symmetric adjacency weight matrix.
    ///
    /// # Errors
    /// - [`NetworkedError::InsufficientAgents`] if `N < 2`.
    /// - [`NetworkedError::InvalidTopology`] if the matrix has non-zero
    ///   diagonal entries, negative weights, or is non-symmetric.
    pub fn new(weights: Matrix<S, N, N>) -> Result<Self, NetworkedError> {
        if N < 2 {
            return Err(NetworkedError::InsufficientAgents);
        }
        let tol = S::from_f64(1e-9);
        for i in 0..N {
            // No self-loops
            if weights.get(i, i).abs() > tol {
                return Err(NetworkedError::InvalidTopology);
            }
            for j in 0..N {
                // Non-negative weights
                if weights.get(i, j) < -tol {
                    return Err(NetworkedError::InvalidTopology);
                }
                // Symmetry check: |W[i][j] - W[j][i]| ≤ tol
                let diff = (weights.get(i, j) - weights.get(j, i)).abs();
                if diff > tol {
                    return Err(NetworkedError::InvalidTopology);
                }
            }
        }
        Ok(Self { weights })
    }

    /// Compute the graph Laplacian L = D − A, where D is the degree matrix.
    ///
    /// L is symmetric positive semi-definite.  The number of zero eigenvalues
    /// equals the number of connected components.
    pub fn laplacian(&self) -> Matrix<S, N, N> {
        let mut l = Matrix::<S, N, N>::zeros();
        for i in 0..N {
            let mut degree = S::ZERO;
            for j in 0..N {
                let w = self.weights.get(i, j);
                l.set(i, j, -w);
                degree += w;
            }
            l.set(i, i, degree);
        }
        l
    }

    /// Check whether the graph is connected.
    ///
    /// Uses the rank of the Laplacian: the graph is connected if and only if
    /// rank(L) = N − 1 (i.e., exactly one zero eigenvalue).
    ///
    /// Implementation: Gaussian elimination to count the rank of L.
    pub fn is_connected(&self) -> bool {
        let l = self.laplacian();
        matrix_rank(&l) == N - 1
    }

    /// Number of agents.
    pub fn num_agents(&self) -> usize {
        N
    }

    /// Weight of the edge between agents `i` and `j`.
    pub fn weight(&self, i: usize, j: usize) -> S {
        self.weights.get(i, j)
    }

    /// Build a complete graph with uniform weight `w` on all edges.
    pub fn complete(w: S) -> Result<Self, NetworkedError> {
        let mut weights = Matrix::<S, N, N>::zeros();
        for i in 0..N {
            for j in 0..N {
                if i != j {
                    weights.set(i, j, w);
                }
            }
        }
        Self::new(weights)
    }
}

/// Compute the numerical rank of a square matrix via Gaussian elimination.
/// A diagonal element is considered zero if its absolute value < threshold.
fn matrix_rank<S: ControlScalar, const N: usize>(m: &Matrix<S, N, N>) -> usize {
    let threshold = S::from_f64(1e-9);
    // Work on a mutable copy row-by-row
    let mut a: [[S; N]; N] = m.data;
    let mut rank = 0usize;
    let mut row = 0usize;

    for col in 0..N {
        // Find pivot in rows [row..N]
        let mut pivot_row = None;
        let mut max_val = threshold;
        for r in row..N {
            let v = a[r][col].abs();
            if v > max_val {
                max_val = v;
                pivot_row = Some(r);
            }
        }
        let pivot_row = match pivot_row {
            Some(r) => r,
            None => continue, // no pivot in this column
        };
        a.swap(row, pivot_row);
        rank += 1;

        let inv_pivot = S::ONE / a[row][col];
        for c in col..N {
            a[row][c] *= inv_pivot;
        }
        for r in 0..N {
            if r == row {
                continue;
            }
            let factor = a[r][col];
            if factor.abs() <= threshold {
                continue;
            }
            for c in col..N {
                let sub = factor * a[row][c];
                a[r][c] -= sub;
            }
        }
        row += 1;
    }
    rank
}

// ──────────────────────────────────────────────────────────────────────────────
// AverageConsensus — discrete-time leaderless average consensus
// ──────────────────────────────────────────────────────────────────────────────

/// Discrete-time leaderless average consensus protocol.
///
/// Update rule (for agent i):
///   x_i[k+1] = x_i[k] − ε · Σ_j a_ij · (x_i[k] − x_j[k])
/// which is equivalent to the matrix iteration x[k+1] = (I − ε·L)·x[k].
///
/// The step-size ε must satisfy  0 < ε < 1/λ_max(L)  for convergence.
/// A sufficient condition (for connected graphs with maximum node degree d_max)
/// is ε < 1/(2·d_max).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AverageConsensus<S: ControlScalar, const N: usize> {
    graph: AgentGraph<S, N>,
    /// Consensus step-size ε.
    step_size: S,
    /// Convergence tolerance: consensus is declared when max(x) − min(x) < tol.
    tol: S,
}

impl<S: ControlScalar, const N: usize> AverageConsensus<S, N> {
    /// Construct an average consensus protocol.
    ///
    /// # Errors
    /// - [`NetworkedError::InvalidTopology`] if the graph is not connected.
    /// - [`NetworkedError::NumericalError`] if `step_size` ≤ 0 or `tol` ≤ 0.
    pub fn new(graph: AgentGraph<S, N>, step_size: S, tol: S) -> Result<Self, NetworkedError> {
        if step_size <= S::ZERO {
            return Err(NetworkedError::NumericalError);
        }
        if tol <= S::ZERO {
            return Err(NetworkedError::NumericalError);
        }
        if !graph.is_connected() {
            return Err(NetworkedError::InvalidTopology);
        }
        Ok(Self {
            graph,
            step_size,
            tol,
        })
    }

    /// Perform one consensus update step.
    ///
    /// Updates `states` in place and returns the new state array.
    pub fn step(&self, states: &mut [S; N]) -> [S; N] {
        let old = *states;
        for i in 0..N {
            let mut sum = S::ZERO;
            for j in 0..N {
                let a_ij = self.graph.weight(i, j);
                if a_ij > S::ZERO {
                    sum += a_ij * (old[i] - old[j]);
                }
            }
            states[i] = old[i] - self.step_size * sum;
        }
        *states
    }

    /// Check whether consensus has been reached to within tolerance.
    pub fn has_converged(&self, states: &[S; N]) -> bool {
        let (min_v, max_v) = min_max(states);
        max_v - min_v < self.tol
    }

    /// Return the current consensus error (max − min of states).
    pub fn consensus_error(&self, states: &[S; N]) -> S {
        let (min_v, max_v) = min_max(states);
        max_v - min_v
    }

    /// Run the protocol until convergence or `max_steps`.
    ///
    /// Returns `Ok(steps)` on convergence, `Err(NetworkedError::NumericalError)`
    /// if max steps is reached without convergence.
    pub fn run_until_convergence(
        &self,
        states: &mut [S; N],
        max_steps: usize,
    ) -> Result<usize, NetworkedError> {
        for step in 0..max_steps {
            self.step(states);
            if self.has_converged(states) {
                return Ok(step + 1);
            }
        }
        Err(NetworkedError::NumericalError)
    }

    /// Theoretical consensus value: average of initial states.
    pub fn average(states: &[S; N]) -> S {
        let mut sum = S::ZERO;
        for &x in states.iter() {
            sum += x;
        }
        sum / S::from_f64(N as f64)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// LeaderFollowingConsensus
// ──────────────────────────────────────────────────────────────────────────────

/// Discrete-time leader-following consensus protocol.
///
/// Update rule for agent i:
///   x_i[k+1] = x_i[k] − ε · [Σ_j a_ij·(x_i−x_j) + g_i·(x_i−x_0)]
/// where x_0 is the leader state, and g_i ≥ 0 is the pinning gain of agent i
/// (g_i > 0 only for agents directly connected to the leader).
///
/// All agents converge to the leader state x_0 provided that the graph
/// augmented with the pinning matrix is connected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeaderFollowingConsensus<S: ControlScalar, const N: usize> {
    graph: AgentGraph<S, N>,
    /// Pinning gains g_i for each agent.
    pinning: [S; N],
    /// Consensus step-size ε.
    step_size: S,
    /// Convergence tolerance.
    tol: S,
}

impl<S: ControlScalar, const N: usize> LeaderFollowingConsensus<S, N> {
    /// Construct a leader-following consensus protocol.
    ///
    /// At least one agent must have a positive pinning gain (connectivity to
    /// the leader); otherwise the network cannot track the leader.
    ///
    /// # Errors
    /// - [`NetworkedError::InvalidTopology`] if no agent has g_i > 0.
    /// - [`NetworkedError::NumericalError`] if `step_size` ≤ 0 or `tol` ≤ 0.
    pub fn new(
        graph: AgentGraph<S, N>,
        pinning: [S; N],
        step_size: S,
        tol: S,
    ) -> Result<Self, NetworkedError> {
        if step_size <= S::ZERO || tol <= S::ZERO {
            return Err(NetworkedError::NumericalError);
        }
        let any_pinned = pinning.iter().any(|&g| g > S::ZERO);
        if !any_pinned {
            return Err(NetworkedError::InvalidTopology);
        }
        Ok(Self {
            graph,
            pinning,
            step_size,
            tol,
        })
    }

    /// Perform one consensus update step, driving followers toward `leader`.
    pub fn step(&self, leader: S, states: &mut [S; N]) -> [S; N] {
        let old = *states;
        for i in 0..N {
            let mut sum = S::ZERO;
            for j in 0..N {
                let a_ij = self.graph.weight(i, j);
                if a_ij > S::ZERO {
                    sum += a_ij * (old[i] - old[j]);
                }
            }
            // Pinning term
            sum += self.pinning[i] * (old[i] - leader);
            states[i] = old[i] - self.step_size * sum;
        }
        *states
    }

    /// Check whether all agents have converged to within `tol` of the leader.
    pub fn has_converged(&self, leader: S, states: &[S; N]) -> bool {
        for &x in states.iter() {
            if (x - leader).abs() >= self.tol {
                return false;
            }
        }
        true
    }

    /// Maximum tracking error across all agents.
    pub fn tracking_error(&self, leader: S, states: &[S; N]) -> S {
        let mut max_err = S::ZERO;
        for &x in states.iter() {
            let err = (x - leader).abs();
            if err > max_err {
                max_err = err;
            }
        }
        max_err
    }

    /// Run until convergence or `max_steps`.
    pub fn run_until_convergence(
        &self,
        leader: S,
        states: &mut [S; N],
        max_steps: usize,
    ) -> Result<usize, NetworkedError> {
        for step in 0..max_steps {
            self.step(leader, states);
            if self.has_converged(leader, states) {
                return Ok(step + 1);
            }
        }
        Err(NetworkedError::NumericalError)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DistributedGradientDescent — ADMM-lite
// ──────────────────────────────────────────────────────────────────────────────

/// Distributed gradient descent with a consensus regularisation term
/// (ADMM-lite / primal decomposition).
///
/// Each agent i minimises a private convex function f_i(x_i) subject to the
/// consensus constraint x_i = x_j for all neighbours j.  The update rule is:
///
///   x_i[k+1] = x_i[k] − α · ∇f_i(x_i[k]) − ρ · Σ_j a_ij · (x_i[k] − x_j[k])
///
/// where α > 0 is the gradient step-size and ρ > 0 is the consensus coupling
/// weight (analogous to the ADMM penalty parameter).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistributedGradientDescent<S: ControlScalar, const N: usize> {
    graph: AgentGraph<S, N>,
    /// Gradient step-size α.
    grad_step: S,
    /// Consensus coupling weight ρ.
    consensus_weight: S,
    /// Convergence tolerance.
    tol: S,
}

impl<S: ControlScalar, const N: usize> DistributedGradientDescent<S, N> {
    /// Construct a distributed gradient descent protocol.
    ///
    /// # Errors
    /// Returns [`NetworkedError::NumericalError`] if any parameter is ≤ 0.
    pub fn new(
        graph: AgentGraph<S, N>,
        grad_step: S,
        consensus_weight: S,
        tol: S,
    ) -> Result<Self, NetworkedError> {
        if grad_step <= S::ZERO || consensus_weight <= S::ZERO || tol <= S::ZERO {
            return Err(NetworkedError::NumericalError);
        }
        Ok(Self {
            graph,
            grad_step,
            consensus_weight,
            tol,
        })
    }

    /// Perform one distributed gradient + consensus step.
    ///
    /// # Arguments
    /// - `states`:    current agent states (updated in-place).
    /// - `gradients`: ∇f_i(x_i) for each agent, evaluated at the current state.
    /// - `step_size`: per-call override for the gradient step-size (pass
    ///   `self.grad_step` for the default; an override is useful for step-size
    ///   schedules without re-construction).
    ///
    /// Returns the new states.
    pub fn step(&self, states: &mut [S; N], gradients: &[S; N], step_size: S) -> [S; N] {
        let old = *states;
        for i in 0..N {
            // Consensus term: ρ · L_i · x
            let mut consensus = S::ZERO;
            for j in 0..N {
                let a_ij = self.graph.weight(i, j);
                if a_ij > S::ZERO {
                    consensus += a_ij * (old[i] - old[j]);
                }
            }
            states[i] = old[i] - step_size * gradients[i] - self.consensus_weight * consensus;
        }
        *states
    }

    /// Default gradient step-size α (as supplied at construction).
    pub fn grad_step(&self) -> S {
        self.grad_step
    }

    /// Check whether all agents have reached approximate consensus.
    pub fn has_converged(&self, states: &[S; N]) -> bool {
        let (min_v, max_v) = min_max(states);
        max_v - min_v < self.tol
    }

    /// Consensus error (spread of agent states).
    pub fn consensus_error(&self, states: &[S; N]) -> S {
        let (min_v, max_v) = min_max(states);
        max_v - min_v
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper
// ──────────────────────────────────────────────────────────────────────────────

fn min_max<S: ControlScalar, const N: usize>(v: &[S; N]) -> (S, S) {
    let mut min_v = v[0];
    let mut max_v = v[0];
    for &x in v.iter().skip(1) {
        if x < min_v {
            min_v = x;
        }
        if x > max_v {
            max_v = x;
        }
    }
    (min_v, max_v)
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AgentGraph ─────────────────────────────────────────────────────────────

    fn ring_graph_4() -> AgentGraph<f64, 4> {
        // 0 — 1 — 2 — 3 — 0
        let mut w = Matrix::<f64, 4, 4>::zeros();
        let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
        for (i, j) in edges {
            w.set(i, j, 1.0);
            w.set(j, i, 1.0);
        }
        AgentGraph::new(w).expect("valid ring graph")
    }

    #[test]
    fn laplacian_row_sums_to_zero() {
        let g = ring_graph_4();
        let l = g.laplacian();
        for i in 0..4 {
            let mut row_sum = 0.0_f64;
            for j in 0..4 {
                row_sum += l.get(i, j);
            }
            assert!(row_sum.abs() < 1e-10, "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn ring_graph_is_connected() {
        let g = ring_graph_4();
        assert!(g.is_connected());
    }

    #[test]
    fn disconnected_graph_not_connected() {
        // Two isolated pairs: {0,1} and {2,3}
        let mut w = Matrix::<f64, 4, 4>::zeros();
        w.set(0, 1, 1.0);
        w.set(1, 0, 1.0);
        w.set(2, 3, 1.0);
        w.set(3, 2, 1.0);
        let g = AgentGraph::new(w).expect("valid weights");
        assert!(!g.is_connected());
    }

    #[test]
    fn self_loop_rejected() {
        let mut w = Matrix::<f64, 4, 4>::zeros();
        w.set(0, 0, 1.0); // self-loop
        w.set(0, 1, 1.0);
        w.set(1, 0, 1.0);
        assert_eq!(
            AgentGraph::<f64, 4>::new(w),
            Err(NetworkedError::InvalidTopology)
        );
    }

    #[test]
    fn asymmetric_weights_rejected() {
        let mut w = Matrix::<f64, 4, 4>::zeros();
        w.set(0, 1, 1.0);
        w.set(1, 0, 2.0); // asymmetric
        assert_eq!(
            AgentGraph::<f64, 4>::new(w),
            Err(NetworkedError::InvalidTopology)
        );
    }

    #[test]
    fn complete_graph_is_connected() {
        let g = AgentGraph::<f64, 4>::complete(1.0).expect("valid");
        assert!(g.is_connected());
    }

    // ── AverageConsensus ───────────────────────────────────────────────────────

    #[test]
    fn average_consensus_converges_4_agents() {
        let g = ring_graph_4();
        // For ring-4, λ_max(L) = 4.  Use ε = 0.2 < 1/4 for convergence.
        let proto = AverageConsensus::new(g, 0.2, 1e-6).expect("valid protocol");

        let mut states = [1.0_f64, 3.0, 5.0, 7.0];
        let expected_avg = 4.0_f64;

        let steps = proto
            .run_until_convergence(&mut states, 500)
            .expect("should converge");

        assert!(steps <= 500, "converged in {steps} steps");
        for (i, &x) in states.iter().enumerate() {
            assert!(
                (x - expected_avg).abs() < 1e-4,
                "agent {i}: x={x:.6} expected={expected_avg}"
            );
        }
        // Verify using the helper
        let avg_ref = AverageConsensus::<f64, 4>::average(&[1.0, 3.0, 5.0, 7.0]);
        assert!((avg_ref - expected_avg).abs() < 1e-10);

        let _ = proto; // silence unused warning
    }

    #[test]
    fn average_consensus_conserves_sum() {
        // The sum of agent states is invariant under x[k+1] = (I − ε·L)·x[k]
        // since L·1 = 0.
        let g = ring_graph_4();
        let proto = AverageConsensus::new(g, 0.2, 1e-8).expect("valid");
        let mut states = [2.0_f64, -1.0, 4.0, 3.0];
        let initial_sum: f64 = states.iter().sum();

        for _ in 0..50 {
            proto.step(&mut states);
        }
        let final_sum: f64 = states.iter().sum();
        assert!(
            (final_sum - initial_sum).abs() < 1e-8,
            "sum changed: {initial_sum} → {final_sum}"
        );
    }

    #[test]
    fn average_consensus_converges_within_50_steps_complete_graph() {
        // Complete graph has faster convergence.
        let g = AgentGraph::<f64, 4>::complete(1.0).expect("valid");
        // λ_max(L) = 4 for complete-4 with unit weights.  Use ε = 0.1.
        let proto = AverageConsensus::new(g, 0.1, 1e-4).expect("valid");
        let mut states = [0.0_f64, 10.0, 5.0, -5.0];
        let expected_avg = 2.5_f64;

        let steps = proto
            .run_until_convergence(&mut states, 50)
            .expect("should converge in 50 steps");

        assert!(steps <= 50, "took {steps} steps");
        for &x in states.iter() {
            assert!((x - expected_avg).abs() < 1e-3, "x={x}");
        }
    }

    // ── LeaderFollowingConsensus ────────────────────────────────────────────────

    #[test]
    fn leader_following_tracks_leader() {
        // Use a complete graph for faster convergence than ring.
        // For complete-4 with uniform weight w=1, L has eigenvalues {0,4,4,4}.
        // Pinning g=[1,0,0,0] adds 1 to agent 0's diagonal; the augmented
        // matrix eigenvalues satisfy λ ∈ (0, 4], so ε < 0.25 guarantees
        // convergence.  Use ε = 0.2 for faster convergence.
        let g = AgentGraph::<f64, 4>::complete(1.0).expect("valid graph");
        let pinning = [1.0_f64, 0.0, 0.0, 0.0];
        let proto = LeaderFollowingConsensus::new(g, pinning, 0.2, 1e-4).expect("valid protocol");

        let leader = 5.0_f64;
        let mut states = [0.0_f64; 4];

        let steps = proto
            .run_until_convergence(leader, &mut states, 500)
            .expect("should converge within 500 steps");

        assert!(steps <= 500, "converged in {steps} steps");
        for (i, &x) in states.iter().enumerate() {
            assert!(
                (x - leader).abs() < 1e-3,
                "agent {i}: x={x:.4} leader={leader}"
            );
        }
    }

    #[test]
    fn leader_following_no_pinning_rejected() {
        let g = ring_graph_4();
        let pinning = [0.0_f64; 4]; // no pinning
        assert_eq!(
            LeaderFollowingConsensus::<f64, 4>::new(g, pinning, 0.1, 1e-4),
            Err(NetworkedError::InvalidTopology)
        );
    }

    // ── DistributedGradientDescent ─────────────────────────────────────────────

    #[test]
    fn distributed_gd_consensus_constant_gradient() {
        // With zero gradients (f_i(x) = 0), DGD should converge to consensus.
        // For complete-4 with unit weights, λ_max(L) = 4.
        // Stability requires ρ < 1/λ_max = 0.25.  Use ρ = 0.1.
        let g = AgentGraph::<f64, 4>::complete(1.0).expect("valid");
        let proto = DistributedGradientDescent::new(g, 0.01, 0.1, 1e-5).expect("valid");

        let mut states = [1.0_f64, 2.0, 3.0, 4.0];
        let gradients = [0.0_f64; 4];

        for _ in 0..200 {
            proto.step(&mut states, &gradients, 0.01);
            if proto.has_converged(&states) {
                break;
            }
        }
        assert!(
            proto.has_converged(&states),
            "DGD should reach consensus with zero gradients"
        );
    }

    #[test]
    fn distributed_gd_quadratic_minimisation() {
        // Each agent i minimises f_i(x) = (x − c_i)²  → ∇f_i = 2(x − c_i).
        // DGD with constant step-size converges to a biased neighbourhood of
        // the consensus optimum x* = mean(c_i) = 2.5.  The bias is O(α), so
        // with a small step-size α the states reach approximate consensus
        // (all agreeing on nearly the same value near the optimum).
        //
        // For complete-4, ρ < 1/λ_max = 0.25 and α small enough for stability.
        let g = AgentGraph::<f64, 4>::complete(1.0).expect("valid");
        // Use small α=0.005 and ρ=0.1 (both within stability margin).
        let proto = DistributedGradientDescent::new(g, 0.005, 0.1, 0.1).expect("valid");

        let targets = [1.0_f64, 2.0, 3.0, 4.0]; // c_i values
        let expected = 2.5_f64; // approximate consensus point

        let mut states = [0.0_f64; 4];

        for _ in 0..2000 {
            let gradients: [f64; 4] = core::array::from_fn(|i| 2.0 * (states[i] - targets[i]));
            proto.step(&mut states, &gradients, 0.005);
        }
        // States should have reached consensus (small spread) near the optimum.
        assert!(
            proto.has_converged(&states),
            "DGD should achieve approximate consensus; spread={:.4}",
            proto.consensus_error(&states)
        );
        // The consensus value should be within O(alpha) of the true optimum.
        let consensus_val = states.iter().copied().fold(0.0_f64, |s, x| s + x) / 4.0;
        assert!(
            (consensus_val - expected).abs() < 0.5,
            "consensus value {consensus_val:.4} should be near {expected}"
        );
    }

    #[test]
    fn distributed_gd_invalid_params_rejected() {
        let g = AgentGraph::<f64, 4>::complete(1.0).expect("valid");
        assert_eq!(
            DistributedGradientDescent::<f64, 4>::new(g, 0.0, 0.5, 1e-5),
            Err(NetworkedError::NumericalError)
        );
    }
}
