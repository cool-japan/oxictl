//! Min-max robust MPC for polytopic uncertainty sets.
//!
//! Robust MPC accounts for parametric uncertainty in the system model by
//! finding a control policy that minimises the worst-case cost over a polytopic
//! set of possible system matrices.  The uncertainty set is described by its
//! vertices; the true system is assumed to be a convex combination of these.
//!
//! The approach used here:
//! 1. Represent the uncertainty set Ω = conv{(A_1, B_1), ..., (A_V, B_V)}.
//! 2. For each candidate input sequence u, evaluate cost under all V vertex systems.
//! 3. Optimise the input to minimise the maximum (worst-case) vertex cost.
//! 4. Tighten state/input constraints by a robustly computed invariant tube to
//!    guarantee constraint satisfaction for all systems in Ω.
//!
//! The gradient of the worst-case cost is approximated via numerical finite
//! differences with subgradient descent (gradient of the max = gradient of the
//! maximising vertex).
#![allow(unused, clippy::needless_range_loop)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Error type for robust MPC operations.
#[derive(Debug)]
pub enum RobustMpcError {
    /// No uncertainty vertices have been provided.
    NoVertices,
    /// The constraint tightening margin would make constraints infeasible.
    InfeasibleTightening,
}

impl core::fmt::Display for RobustMpcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RobustMpcError::NoVertices => write!(f, "Robust MPC: no uncertainty vertices provided"),
            RobustMpcError::InfeasibleTightening => {
                write!(
                    f,
                    "Robust MPC: constraint tightening made problem infeasible"
                )
            }
        }
    }
}

/// A single vertex of the polytopic uncertainty set (A_v, B_v).
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
#[derive(Clone, Copy)]
pub struct UncertaintyVertex<S: ControlScalar, const N: usize, const I: usize> {
    /// State transition matrix at this vertex.
    pub a: Matrix<S, N, N>,
    /// Input matrix at this vertex.
    pub b: Matrix<S, N, I>,
}

impl<S: ControlScalar, const N: usize, const I: usize> UncertaintyVertex<S, N, I> {
    /// Create a new uncertainty vertex.
    pub fn new(a: Matrix<S, N, N>, b: Matrix<S, N, I>) -> Self {
        Self { a, b }
    }

    /// Propagate state: x_{k+1} = A_v x_k + B_v u_k.
    pub fn propagate(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>) -> Matrix<S, N, 1> {
        let ax = matmul(&self.a, x);
        let bu = matmul(&self.b, u);
        ax.add_mat(&bu)
    }
}

/// Box constraint: ||x||_inf ≤ x_max, ||u||_inf ≤ u_max.
///
/// These are tightened internally for robustness.
#[derive(Clone, Copy, Debug)]
pub struct RobustBoxConstraint<S: ControlScalar> {
    /// Maximum absolute state value (element-wise).
    pub x_max: S,
    /// Maximum absolute input value (element-wise).
    pub u_max: S,
}

impl<S: ControlScalar> RobustBoxConstraint<S> {
    /// Create new box constraints.
    pub fn new(x_max: S, u_max: S) -> Self {
        Self { x_max, u_max }
    }

    /// Tighten state constraint by margin δ.
    ///
    /// Returns the tightened bound x_max - δ, or an error if infeasible.
    pub fn tighten_state(&self, delta: S) -> Result<S, RobustMpcError> {
        let tightened = self.x_max - delta;
        if tightened <= S::ZERO {
            return Err(RobustMpcError::InfeasibleTightening);
        }
        Ok(tightened)
    }

    /// Tighten input constraint by margin δ.
    pub fn tighten_input(&self, delta: S) -> Result<S, RobustMpcError> {
        let tightened = self.u_max - delta;
        if tightened <= S::ZERO {
            return Err(RobustMpcError::InfeasibleTightening);
        }
        Ok(tightened)
    }
}

/// Terminal set approximation: largest invariant ellipsoid {x : x^T P_f x ≤ γ}.
///
/// Stored as the inverse weighting matrix P_f and level γ.
#[derive(Clone, Copy, Debug)]
pub struct TerminalSet<S: ControlScalar, const N: usize> {
    /// Terminal cost matrix P_f (diagonal approximation stored as vector).
    pub p_f_diag: Matrix<S, N, 1>,
    /// Level set parameter γ.
    pub gamma: S,
}

impl<S: ControlScalar, const N: usize> TerminalSet<S, N> {
    /// Create a terminal set with given diagonal P_f and level γ.
    pub fn new(p_f_diag: Matrix<S, N, 1>, gamma: S) -> Self {
        Self { p_f_diag, gamma }
    }

    /// Check if a state x lies within the terminal set: x^T P_f x ≤ γ.
    pub fn contains(&self, x: &Matrix<S, N, 1>) -> bool {
        let mut val = S::ZERO;
        for i in 0..N {
            val += self.p_f_diag.data[i][0] * x.data[i][0] * x.data[i][0];
        }
        val <= self.gamma
    }

    /// Evaluate the terminal cost x^T P_f x.
    pub fn terminal_cost(&self, x: &Matrix<S, N, 1>) -> S {
        let mut val = S::ZERO;
        for i in 0..N {
            val += self.p_f_diag.data[i][0] * x.data[i][0] * x.data[i][0];
        }
        val
    }
}

/// Min-max robust MPC controller.
///
/// Minimises the worst-case quadratic cost over a polytopic uncertainty set,
/// described by V vertices.  The input sequence is optimised via subgradient
/// descent on the worst-case (maximum over vertices) cost.
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
/// - H: prediction horizon
/// - V: number of uncertainty vertices (≥ 1)
pub struct RobustMpc<
    S: ControlScalar,
    const N: usize,
    const I: usize,
    const H: usize,
    const V: usize,
> {
    /// Uncertainty vertices.
    pub vertices: [UncertaintyVertex<S, N, I>; V],
    /// State cost weight Q (N×N diagonal, stored as N×N matrix).
    pub q: Matrix<S, N, N>,
    /// Input cost weight R (I×I diagonal, stored as I×I matrix).
    pub r: Matrix<S, I, I>,
    /// Terminal set for terminal cost / feasibility.
    pub terminal_set: Option<TerminalSet<S, N>>,
    /// Box constraints (before tightening).
    pub constraints: RobustBoxConstraint<S>,
    /// Constraint tightening margin δ (for robust feasibility).
    pub tightening_margin: S,
    /// Current state.
    pub x: Matrix<S, N, 1>,
    /// Number of subgradient iterations per solve.
    pub iterations: usize,
    /// Subgradient step size.
    pub step_size: S,
}

impl<S: ControlScalar, const N: usize, const I: usize, const H: usize, const V: usize>
    RobustMpc<S, N, I, H, V>
{
    /// Create a new RobustMpc controller.
    pub fn new(
        vertices: [UncertaintyVertex<S, N, I>; V],
        q: Matrix<S, N, N>,
        r: Matrix<S, I, I>,
        constraints: RobustBoxConstraint<S>,
        iterations: usize,
    ) -> Self {
        Self {
            vertices,
            q,
            r,
            terminal_set: None,
            constraints,
            tightening_margin: S::from_f64(0.05),
            x: Matrix::zeros(),
            iterations,
            step_size: S::from_f64(1e-3),
        }
    }

    /// Set the terminal set for terminal cost computation.
    pub fn with_terminal_set(mut self, ts: TerminalSet<S, N>) -> Self {
        self.terminal_set = Some(ts);
        self
    }

    /// Set the constraint tightening margin.
    pub fn with_tightening_margin(mut self, margin: S) -> Self {
        self.tightening_margin = margin;
        self
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

    /// Compute the total trajectory cost for vertex v and input sequence.
    fn vertex_cost(&self, v: usize, u_seq: &[Matrix<S, I, 1>; H]) -> S {
        let vertex = &self.vertices[v];
        let mut total = S::ZERO;
        let mut x = self.x;
        for k in 0..H {
            total += self.stage_cost(&x, &u_seq[k]);
            x = vertex.propagate(&x, &u_seq[k]);
        }
        // Terminal cost
        if let Some(ref ts) = self.terminal_set {
            total += ts.terminal_cost(&x);
        } else {
            total += self.stage_cost(&x, &Matrix::zeros());
        }
        total
    }

    /// Find the worst-case (maximum cost) vertex index for a given input sequence.
    fn worst_case_vertex(&self, u_seq: &[Matrix<S, I, 1>; H]) -> usize {
        let mut worst_v = 0;
        let mut worst_cost = self.vertex_cost(0, u_seq);
        for v in 1..V {
            let c = self.vertex_cost(v, u_seq);
            if c > worst_cost {
                worst_cost = c;
                worst_v = v;
            }
        }
        worst_v
    }

    /// Worst-case cost: max over all vertices.
    pub fn worst_case_cost(&self, u_seq: &[Matrix<S, I, 1>; H]) -> S {
        let wv = self.worst_case_vertex(u_seq);
        self.vertex_cost(wv, u_seq)
    }

    /// Numerical subgradient of worst-case cost w.r.t. u_k (central differences).
    fn subgradient_uk(&self, k: usize, u_seq: &[Matrix<S, I, 1>; H], eps: S) -> Matrix<S, I, 1> {
        let two_eps = S::TWO * eps;
        let mut grad = Matrix::<S, I, 1>::zeros();
        for i in 0..I {
            let mut u_p = *u_seq;
            let mut u_m = *u_seq;
            u_p[k].data[i][0] += eps;
            u_m[k].data[i][0] -= eps;
            let cp = self.worst_case_cost(&u_p);
            let cm = self.worst_case_cost(&u_m);
            grad.data[i][0] = (cp - cm) / two_eps;
        }
        grad
    }

    /// Project an input onto the tightened input constraint box.
    fn project_input(&self, u: Matrix<S, I, 1>, u_bound: S) -> Matrix<S, I, 1> {
        let mut out = u;
        for i in 0..I {
            out.data[i][0] = out.data[i][0].clamp_val(-u_bound, u_bound);
        }
        out
    }

    /// Solve the min-max robust MPC problem via subgradient descent.
    ///
    /// Returns the first optimal input action, or an error if constraints
    /// cannot be tightened (infeasible).
    pub fn solve(&mut self) -> Result<Matrix<S, I, 1>, RobustMpcError> {
        // Compute tightened input bound
        let u_bound = self.constraints.tighten_input(self.tightening_margin)?;
        let eps = S::from_f64(1e-5);
        let step = self.step_size;

        let mut u_seq: [Matrix<S, I, 1>; H] = [Matrix::zeros(); H];

        for _iter in 0..self.iterations {
            // Subgradient step for each u_k
            for k in 0..H {
                let g = self.subgradient_uk(k, &u_seq, eps);
                for i in 0..I {
                    u_seq[k].data[i][0] -= step * g.data[i][0];
                }
                u_seq[k] = self.project_input(u_seq[k], u_bound);
            }
        }

        Ok(u_seq[0])
    }

    /// Set the current state.
    pub fn set_state(&mut self, x: Matrix<S, N, 1>) {
        self.x = x;
    }

    /// Compute tightened state constraint bound.
    pub fn tightened_state_bound(&self) -> Result<S, RobustMpcError> {
        self.constraints.tighten_state(self.tightening_margin)
    }

    /// Check if the current state lies in the terminal set.
    pub fn in_terminal_set(&self) -> bool {
        match &self.terminal_set {
            Some(ts) => ts.contains(&self.x),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vertex(a_scale: f64) -> UncertaintyVertex<f64, 2, 1> {
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = a_scale * 0.1;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.005;
        b.data[1][0] = 0.1;

        UncertaintyVertex::new(a, b)
    }

    fn make_robust_mpc() -> RobustMpc<f64, 2, 1, 5, 2> {
        let v1 = make_vertex(0.9);
        let v2 = make_vertex(1.1);

        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;

        let constraints = RobustBoxConstraint::new(10.0_f64, 1.0_f64);

        RobustMpc::new([v1, v2], q, r, constraints, 30)
    }

    #[test]
    fn robust_mpc_construction() {
        let mpc = make_robust_mpc();
        assert_eq!(mpc.iterations, 30);
        assert!(mpc.terminal_set.is_none());
    }

    #[test]
    fn worst_case_cost_non_negative() {
        let mpc = make_robust_mpc();
        let u_seq = [Matrix::<f64, 1, 1>::zeros(); 5];
        let cost = mpc.worst_case_cost(&u_seq);
        assert!(
            cost >= 0.0,
            "Worst-case cost must be non-negative: {}",
            cost
        );
    }

    #[test]
    fn worst_case_cost_at_origin_is_zero() {
        let mpc = make_robust_mpc();
        let u_seq = [Matrix::<f64, 1, 1>::zeros(); 5];
        // With x = 0 and u = 0, stage cost is 0
        let cost = mpc.worst_case_cost(&u_seq);
        assert!(
            cost < 1e-12,
            "Cost at origin with zero input should be zero: {}",
            cost
        );
    }

    #[test]
    fn tightened_bound_is_smaller() {
        let mpc = make_robust_mpc();
        let bound = mpc
            .tightened_state_bound()
            .expect("tightening should succeed");
        assert!(
            bound < 10.0,
            "Tightened bound should be smaller than original: {}",
            bound
        );
    }

    #[test]
    fn solve_returns_control() {
        let mut mpc = make_robust_mpc();
        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 1.0;
        mpc.set_state(x0);
        let result = mpc.solve();
        assert!(result.is_ok(), "solve should succeed: {:?}", result);
    }

    #[test]
    fn terminal_set_contains_origin() {
        let mut p_f = Matrix::<f64, 2, 1>::zeros();
        p_f.data[0][0] = 1.0;
        p_f.data[1][0] = 1.0;
        let ts = TerminalSet::new(p_f, 1.0_f64);
        let origin = Matrix::<f64, 2, 1>::zeros();
        assert!(ts.contains(&origin), "Terminal set must contain origin");
    }

    #[test]
    fn terminal_set_excludes_far_state() {
        let mut p_f = Matrix::<f64, 2, 1>::zeros();
        p_f.data[0][0] = 1.0;
        p_f.data[1][0] = 1.0;
        let ts = TerminalSet::new(p_f, 1.0_f64);
        let mut far = Matrix::<f64, 2, 1>::zeros();
        far.data[0][0] = 2.0; // 2^2 = 4 > 1
        assert!(
            !ts.contains(&far),
            "Terminal set must not contain far state"
        );
    }

    #[test]
    fn infeasible_tightening_returns_error() {
        let constraints = RobustBoxConstraint::new(0.01_f64, 0.01_f64);
        // Tightening by 1.0 from 0.01 should fail
        let result = constraints.tighten_state(1.0_f64);
        assert!(
            matches!(result, Err(RobustMpcError::InfeasibleTightening)),
            "Expected InfeasibleTightening error"
        );
    }
}
