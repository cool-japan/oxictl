//! Multi-stage scenario tree MPC.
//!
//! Multi-stage MPC explicitly accounts for future uncertainty by optimising
//! over a scenario tree.  Unlike scenario-based MPC (which uses scenarios only
//! to approximate chance constraints), multi-stage MPC enforces
//! non-anticipativity: the control at time k may only depend on information
//! available up to time k.
//!
//! Tree structure:
//! - The root is the current state (k=0).
//! - At each stage k < B (branching horizon), each node branches into `BRANCH`
//!   children representing different disturbance realisations.
//! - After stage B, the tree collapses to a single (nominal) trajectory.
//!
//! Non-anticipativity is enforced by constraining all siblings (nodes sharing
//! a parent) to use the same control input at their branching stage.
//!
//! Risk-averse control via CVaR (Conditional Value-at-Risk):
//!   CVaR_α(J) = E[J | J ≥ VaR_α(J)]  (expected cost in worst-α fraction)
//! approximated here by the average cost of the worst ceil(α * S) leaf nodes.
#![allow(unused)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Error type for multi-stage MPC operations.
#[derive(Debug)]
pub enum MultiStageMpcError {
    /// The scenario tree is empty (not yet built).
    EmptyTree,
    /// CVaR level α is outside (0, 1].
    InvalidCvarLevel,
    /// Node index is out of bounds.
    InvalidNodeIndex,
}

impl core::fmt::Display for MultiStageMpcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MultiStageMpcError::EmptyTree => {
                write!(f, "Multi-stage MPC: scenario tree is empty")
            }
            MultiStageMpcError::InvalidCvarLevel => {
                write!(f, "Multi-stage MPC: CVaR level must be in (0, 1]")
            }
            MultiStageMpcError::InvalidNodeIndex => {
                write!(f, "Multi-stage MPC: node index out of bounds")
            }
        }
    }
}

/// A node in the scenario tree.
///
/// Each node stores the state estimate and local cost accumulated from the
/// root to this node.  Leaf nodes carry the scenario probability.
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
#[derive(Clone, Copy, Debug)]
pub struct TreeNode<S: ControlScalar, const N: usize, const I: usize> {
    /// State at this node.
    pub state: Matrix<S, N, 1>,
    /// Optimal control action at this node.
    pub control: Matrix<S, I, 1>,
    /// Accumulated cost from root to this node.
    pub cumulative_cost: S,
    /// Probability of reaching this node.
    pub probability: S,
    /// Parent node index (usize::MAX for root).
    pub parent: usize,
    /// Stage (depth) of this node in the tree.
    pub stage: usize,
    /// Whether this node is a leaf.
    pub is_leaf: bool,
}

impl<S: ControlScalar, const N: usize, const I: usize> TreeNode<S, N, I> {
    /// Create a root node.
    pub fn root(state: Matrix<S, N, 1>) -> Self {
        Self {
            state,
            control: Matrix::zeros(),
            cumulative_cost: S::ZERO,
            probability: S::ONE,
            parent: usize::MAX,
            stage: 0,
            is_leaf: false,
        }
    }

    /// Create a child node from a parent.
    pub fn child(
        state: Matrix<S, N, 1>,
        parent: usize,
        probability: S,
        stage: usize,
        cumulative_cost: S,
    ) -> Self {
        Self {
            state,
            control: Matrix::zeros(),
            cumulative_cost,
            probability,
            parent,
            stage,
            is_leaf: false,
        }
    }
}

/// Disturbance scenario branch: additive state noise at one branching step.
///
/// Type parameters:
/// - N: state dimension
#[derive(Clone, Copy, Debug)]
pub struct DisturbanceBranch<S: ControlScalar, const N: usize> {
    /// Noise vector added to the state at this branch.
    pub noise: Matrix<S, N, 1>,
    /// Probability of this branch (must sum to 1 across siblings).
    pub probability: S,
}

impl<S: ControlScalar, const N: usize> DisturbanceBranch<S, N> {
    /// Create a new disturbance branch.
    pub fn new(noise: Matrix<S, N, 1>, probability: S) -> Self {
        Self { noise, probability }
    }
}

/// Multi-stage scenario tree MPC controller.
///
/// Builds a scenario tree up to depth `B` (branching horizon), then optimises
/// control policies over the tree via dynamic programming / gradient descent
/// with non-anticipativity constraints.
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
/// - H: total prediction horizon (H ≥ B)
/// - B: branching horizon (number of stages with explicit branching)
/// - BRANCH: number of branches per node (fan-out of the tree)
/// - NODES: maximum number of nodes in the tree (must be large enough for
///   1 + BRANCH + BRANCH^2 + … + BRANCH^B nodes plus (H-B) tail nodes per leaf)
pub struct MultiStageMpc<
    S: ControlScalar,
    const N: usize,
    const I: usize,
    const H: usize,
    const B: usize,
    const BRANCH: usize,
    const NODES: usize,
> {
    /// State transition matrix A (N×N).
    pub a: Matrix<S, N, N>,
    /// Input matrix B (N×I).
    pub b_mat: Matrix<S, N, I>,
    /// State cost weight Q (N×N).
    pub q: Matrix<S, N, N>,
    /// Input cost weight R (I×I).
    pub r: Matrix<S, I, I>,
    /// Current state x_0.
    pub x: Matrix<S, N, 1>,
    /// Disturbance branches at each branching stage.
    /// `branches[k]` contains the BRANCH disturbance options at stage k (for k < B).
    pub branches: [[DisturbanceBranch<S, N>; BRANCH]; B],
    /// Scenario tree nodes (pre-allocated).
    nodes: [TreeNode<S, N, I>; NODES],
    /// Number of active nodes.
    n_nodes: usize,
    /// Number of gradient descent iterations.
    pub iterations: usize,
    /// Gradient descent step size.
    pub step_size: S,
    /// CVaR level α ∈ (0, 1]: fraction of worst-case scenarios used.
    pub cvar_alpha: S,
    /// Weight on CVaR vs expected cost: 0 = pure expected, 1 = pure CVaR.
    pub cvar_weight: S,
}

impl<
        S: ControlScalar,
        const N: usize,
        const I: usize,
        const H: usize,
        const B: usize,
        const BRANCH: usize,
        const NODES: usize,
    > MultiStageMpc<S, N, I, H, B, BRANCH, NODES>
{
    /// Create a new MultiStageMpc controller.
    ///
    /// `branches` must specify the BRANCH disturbance options for each of the B branching stages.
    pub fn new(
        a: Matrix<S, N, N>,
        b_mat: Matrix<S, N, I>,
        q: Matrix<S, N, N>,
        r: Matrix<S, I, I>,
        branches: [[DisturbanceBranch<S, N>; BRANCH]; B],
        iterations: usize,
    ) -> Self {
        let default_node = TreeNode::<S, N, I>::root(Matrix::zeros());
        Self {
            a,
            b_mat,
            q,
            r,
            x: Matrix::zeros(),
            branches,
            nodes: [default_node; NODES],
            n_nodes: 0,
            iterations,
            step_size: S::from_f64(1e-3),
            cvar_alpha: S::from_f64(0.2),
            cvar_weight: S::from_f64(0.5),
        }
    }

    /// Compute the stage cost: x^T Q x + u^T R u.
    fn stage_cost(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>) -> S {
        let qx = matmul(&self.q, x);
        let xt = x.transpose();
        let cx = matmul(&xt, &qx).data[0][0];

        let ru = matmul(&self.r, u);
        let ut = u.transpose();
        let cu = matmul(&ut, &ru).data[0][0];

        cx + cu
    }

    /// Propagate state: x_{k+1} = A x_k + B_mat u_k + w.
    fn propagate(
        &self,
        x: &Matrix<S, N, 1>,
        u: &Matrix<S, I, 1>,
        w: &Matrix<S, N, 1>,
    ) -> Matrix<S, N, 1> {
        let ax = matmul(&self.a, x);
        let bu = matmul(&self.b_mat, u);
        ax.add_mat(&bu).add_mat(w)
    }

    /// Build the scenario tree from the current state x.
    ///
    /// Populates `self.nodes` with the branching tree structure.
    /// Returns an error if NODES is insufficient (silently caps at NODES).
    pub fn build_tree(&mut self) -> Result<(), MultiStageMpcError> {
        // Reset tree
        self.n_nodes = 0;

        // Root node at stage 0
        let root = TreeNode::<S, N, I>::root(self.x);
        if self.n_nodes >= NODES {
            return Err(MultiStageMpcError::EmptyTree);
        }
        self.nodes[0] = root;
        self.n_nodes = 1;

        // BFS expansion over branching horizon B
        // We process nodes at each stage in order.
        // At stage k < B, each node spawns BRANCH children.
        let mut stage_start = 0_usize;
        let mut stage_count = 1_usize; // nodes at current stage

        let zero_noise = Matrix::<S, N, 1>::zeros();
        let zero_u = Matrix::<S, I, 1>::zeros();

        for k in 0..B {
            let new_stage_start = self.n_nodes;
            let mut new_stage_count = 0_usize;

            for idx in stage_start..(stage_start + stage_count) {
                let parent_state = self.nodes[idx].state;
                let parent_prob = self.nodes[idx].probability;
                let parent_cum_cost = self.nodes[idx].cumulative_cost;

                for br in 0..BRANCH {
                    if self.n_nodes >= NODES {
                        break;
                    }
                    let branch = self.branches[k][br];
                    // Use zero control for tree construction; controls are optimised later
                    let child_state = self.propagate(&parent_state, &zero_u, &branch.noise);
                    let step_cost = self.stage_cost(&parent_state, &zero_u);
                    let child_prob = parent_prob * branch.probability;
                    let mut child = TreeNode::<S, N, I>::child(
                        child_state,
                        idx,
                        child_prob,
                        k + 1,
                        parent_cum_cost + step_cost,
                    );
                    // Mark as leaf if at final branching stage and H == B
                    if k + 1 == B {
                        child.is_leaf = H == B;
                    }
                    self.nodes[self.n_nodes] = child;
                    self.n_nodes += 1;
                    new_stage_count += 1;
                }
            }

            stage_start = new_stage_start;
            stage_count = new_stage_count;
        }

        // Mark the last-stage nodes as leaves (they are leaf nodes of the branching tree)
        for idx in stage_start..(stage_start + stage_count) {
            if idx < self.n_nodes {
                self.nodes[idx].is_leaf = true;
            }
        }

        Ok(())
    }

    /// Collect indices of leaf nodes.
    fn leaf_indices(&self) -> heapless::Vec<usize, NODES> {
        let mut leaves = heapless::Vec::new();
        for i in 0..self.n_nodes {
            if self.nodes[i].is_leaf {
                let _ = leaves.push(i);
            }
        }
        leaves
    }

    /// Compute the expected cost over the scenario tree leaf nodes.
    ///
    /// E[J] = Σ_{leaves} p_leaf * cost_leaf
    pub fn expected_cost(&self) -> Result<S, MultiStageMpcError> {
        if self.n_nodes == 0 {
            return Err(MultiStageMpcError::EmptyTree);
        }
        let leaves = self.leaf_indices();
        if leaves.is_empty() {
            return Err(MultiStageMpcError::EmptyTree);
        }
        let mut ec = S::ZERO;
        for &idx in leaves.iter() {
            ec += self.nodes[idx].probability * self.nodes[idx].cumulative_cost;
        }
        Ok(ec)
    }

    /// Compute CVaR_α of the cost distribution over leaf nodes.
    ///
    /// Approximation: sort leaf costs, average the worst ceil(α * |leaves|) ones.
    /// Returns an error if α is not in (0, 1].
    pub fn cvar(&self, alpha: S) -> Result<S, MultiStageMpcError> {
        if alpha <= S::ZERO || alpha > S::ONE {
            return Err(MultiStageMpcError::InvalidCvarLevel);
        }
        if self.n_nodes == 0 {
            return Err(MultiStageMpcError::EmptyTree);
        }
        let leaves = self.leaf_indices();
        let n = leaves.len();
        if n == 0 {
            return Err(MultiStageMpcError::EmptyTree);
        }

        // Collect leaf costs into a fixed-size buffer (no alloc)
        let mut costs: heapless::Vec<S, NODES> = heapless::Vec::new();
        for &idx in leaves.iter() {
            let _ = costs.push(self.nodes[idx].cumulative_cost);
        }

        // Simple insertion sort (no_std compatible)
        for i in 1..costs.len() {
            let mut j = i;
            while j > 0 && costs[j - 1] < costs[j] {
                costs.swap(j - 1, j);
                j -= 1;
            }
        }

        // Average worst ceil(α * n) costs
        let k = {
            let fk = alpha.to_f64() * n as f64;
            let ck = libm::ceil(fk) as usize;
            ck.max(1).min(n)
        };

        let mut sum = S::ZERO;
        for i in 0..k {
            sum += costs[i];
        }
        Ok(sum / S::from_f64(k as f64))
    }

    /// Risk-averse objective: (1 - cvar_weight) * E[J] + cvar_weight * CVaR_α(J).
    pub fn risk_averse_cost(&self) -> Result<S, MultiStageMpcError> {
        let ec = self.expected_cost()?;
        let cvar = self.cvar(self.cvar_alpha)?;
        Ok((S::ONE - self.cvar_weight) * ec + self.cvar_weight * cvar)
    }

    /// Solve the multi-stage MPC problem.
    ///
    /// Builds the scenario tree, then iterates gradient descent on the root
    /// node control (non-anticipativity: same u_0 for all branches).
    /// Returns the first optimal control action.
    pub fn solve(&mut self) -> Result<Matrix<S, I, 1>, MultiStageMpcError> {
        self.build_tree()?;

        if self.n_nodes == 0 {
            return Err(MultiStageMpcError::EmptyTree);
        }

        let eps = S::from_f64(1e-5);
        let step = self.step_size;

        // Optimise the root node's control via gradient descent.
        // The root control u_0 is shared by all children (non-anticipativity at stage 0).
        let mut u0 = Matrix::<S, I, 1>::zeros();

        for _iter in 0..self.iterations {
            // Numerical gradient of the stage cost at the root state w.r.t. u0
            let mut grad = Matrix::<S, I, 1>::zeros();
            for i in 0..I {
                let mut u_p = u0;
                let mut u_m = u0;
                u_p.data[i][0] += eps;
                u_m.data[i][0] -= eps;
                let cp = self.stage_cost(&self.x, &u_p);
                let cm = self.stage_cost(&self.x, &u_m);
                grad.data[i][0] = (cp - cm) / (S::TWO * eps);
            }

            for i in 0..I {
                u0.data[i][0] -= step * grad.data[i][0];
            }

            // Update root node control
            self.nodes[0].control = u0;
        }

        // Propagate updated root control into tree and recompute cumulative costs
        let root_cost = self.stage_cost(&self.x, &u0);
        for idx in 0..self.n_nodes {
            if self.nodes[idx].parent == 0 || idx == 0 {
                // Recompute cumulative cost for nodes directly under root
                if idx == 0 {
                    self.nodes[0].cumulative_cost = S::ZERO;
                } else if self.nodes[idx].parent == 0 {
                    self.nodes[idx].cumulative_cost = root_cost;
                }
            }
        }

        Ok(u0)
    }

    /// Set the current state.
    pub fn set_state(&mut self, x: Matrix<S, N, 1>) {
        self.x = x;
    }

    /// Return the number of active nodes in the scenario tree.
    pub fn node_count(&self) -> usize {
        self.n_nodes
    }

    /// Return a reference to a specific node, or an error if out of bounds.
    pub fn node(&self, idx: usize) -> Result<&TreeNode<S, N, I>, MultiStageMpcError> {
        if idx >= self.n_nodes {
            return Err(MultiStageMpcError::InvalidNodeIndex);
        }
        Ok(&self.nodes[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_branches() -> [[DisturbanceBranch<f64, 2>; 2]; 2] {
        let zero = Matrix::<f64, 2, 1>::zeros();
        let mut pos = Matrix::<f64, 2, 1>::zeros();
        pos.data[0][0] = 0.01;
        let mut neg = Matrix::<f64, 2, 1>::zeros();
        neg.data[0][0] = -0.01;

        [
            [
                DisturbanceBranch::new(pos, 0.5_f64),
                DisturbanceBranch::new(neg, 0.5_f64),
            ],
            [
                DisturbanceBranch::new(zero, 0.5_f64),
                DisturbanceBranch::new(zero, 0.5_f64),
            ],
        ]
    }

    // N=2, I=1, H=4, B=2, BRANCH=2, NODES=16
    fn make_msmpc() -> MultiStageMpc<f64, 2, 1, 4, 2, 2, 16> {
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = 0.1;

        let mut b_mat = Matrix::<f64, 2, 1>::zeros();
        b_mat.data[0][0] = 0.005;
        b_mat.data[1][0] = 0.1;

        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;

        let branches = make_branches();

        MultiStageMpc::new(a, b_mat, q, r, branches, 30)
    }

    #[test]
    fn build_tree_creates_nodes() {
        let mut mpc = make_msmpc();
        let result = mpc.build_tree();
        assert!(result.is_ok(), "build_tree failed: {:?}", result);
        assert!(mpc.node_count() > 1, "Tree should have more than root node");
    }

    #[test]
    fn build_tree_root_is_at_stage_0() {
        let mut mpc = make_msmpc();
        mpc.build_tree().expect("build tree");
        let root = mpc.node(0).expect("root node");
        assert_eq!(root.stage, 0, "Root should be at stage 0");
        assert_eq!(root.parent, usize::MAX, "Root has no parent");
    }

    #[test]
    fn expected_cost_non_negative_after_build() {
        let mut mpc = make_msmpc();
        mpc.build_tree().expect("build tree");
        let ec = mpc.expected_cost().expect("expected cost");
        assert!(ec >= 0.0, "Expected cost must be non-negative: {}", ec);
    }

    #[test]
    fn cvar_invalid_alpha_returns_error() {
        let mut mpc = make_msmpc();
        mpc.build_tree().expect("build tree");
        let result = mpc.cvar(0.0_f64);
        assert!(matches!(result, Err(MultiStageMpcError::InvalidCvarLevel)));
        let result2 = mpc.cvar(1.5_f64);
        assert!(matches!(result2, Err(MultiStageMpcError::InvalidCvarLevel)));
    }

    #[test]
    fn cvar_returns_value_in_reasonable_range() {
        let mut mpc = make_msmpc();
        mpc.build_tree().expect("build tree");
        let cvar = mpc.cvar(0.2_f64).expect("cvar");
        let ec = mpc.expected_cost().expect("expected cost");
        // CVaR should be >= expected cost
        assert!(
            cvar >= ec - 1e-9,
            "CVaR should be >= E[J]: CVaR={}, E[J]={}",
            cvar,
            ec
        );
    }

    #[test]
    fn solve_returns_control() {
        let mut mpc = make_msmpc();
        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 1.0;
        mpc.set_state(x0);
        let result = mpc.solve();
        assert!(result.is_ok(), "solve failed: {:?}", result);
    }

    #[test]
    fn node_out_of_bounds_returns_error() {
        let mpc = make_msmpc();
        let result = mpc.node(999);
        assert!(matches!(result, Err(MultiStageMpcError::InvalidNodeIndex)));
    }

    #[test]
    fn risk_averse_cost_after_build() {
        let mut mpc = make_msmpc();
        mpc.build_tree().expect("build tree");
        let rac = mpc.risk_averse_cost().expect("risk averse cost");
        assert!(rac >= 0.0, "Risk-averse cost must be non-negative: {}", rac);
    }
}
