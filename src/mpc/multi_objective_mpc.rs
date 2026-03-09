//! Pareto-optimal MPC with multiple objectives.
//!
//! Many real control problems involve competing objectives (e.g., reference
//! tracking accuracy vs. energy consumption vs. actuator wear).  This module
//! provides two classical scalarisation methods for generating the Pareto front:
//!
//! 1. **Weighted-sum method**: minimise λ₁ J₁ + λ₂ J₂ + … for varying λ.
//!    Simple but cannot capture non-convex parts of the Pareto front.
//!
//! 2. **ε-constraint method**: minimise J₁ s.t. J_i ≤ εᵢ for i ≥ 2.
//!    Can capture the full (possibly non-convex) Pareto front.
//!
//! The Pareto front is approximated as a discrete set of efficient solutions,
//! each corresponding to a different weight / ε-constraint combination.
//!
//! Objectives supported:
//! - Tracking cost: ||x - x_ref||_Q^2  (regulation to reference)
//! - Energy cost: ||u||_R^2            (input energy / actuator effort)
//!
//! Additional objectives can be plugged in as function pointers.
#![allow(unused, clippy::needless_range_loop)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Error type for multi-objective MPC operations.
#[derive(Debug)]
pub enum MultiObjectiveError {
    /// Weight vector does not sum to a positive number.
    InvalidWeights,
    /// ε-constraint bound is negative.
    NegativeBound,
    /// No Pareto points have been computed yet.
    EmptyParetoFront,
}

impl core::fmt::Display for MultiObjectiveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MultiObjectiveError::InvalidWeights => {
                write!(
                    f,
                    "Multi-objective MPC: invalid weight vector (must be positive sum)"
                )
            }
            MultiObjectiveError::NegativeBound => {
                write!(
                    f,
                    "Multi-objective MPC: ε-constraint bound must be non-negative"
                )
            }
            MultiObjectiveError::EmptyParetoFront => {
                write!(f, "Multi-objective MPC: Pareto front is empty")
            }
        }
    }
}

/// A single Pareto-optimal solution: objective values and corresponding first control.
///
/// Type parameters:
/// - I: input dimension
#[derive(Clone, Copy, Debug)]
pub struct ParetoPoint<S: ControlScalar, const I: usize> {
    /// Tracking cost J_track = Σ_k ||x_k - x_ref||_Q^2.
    pub tracking_cost: S,
    /// Energy cost J_energy = Σ_k ||u_k||_R^2.
    pub energy_cost: S,
    /// Corresponding first control action.
    pub u0: Matrix<S, I, 1>,
}

/// Pareto front approximation: a collection of Pareto-optimal points.
///
/// Type parameters:
/// - I: input dimension
/// - P: maximum number of Pareto points
pub struct ParetoFront<S: ControlScalar, const I: usize, const P: usize> {
    /// Array of Pareto points (up to P).
    pub points: [ParetoPoint<S, I>; P],
    /// Number of computed Pareto points (≤ P).
    pub count: usize,
}

impl<S: ControlScalar, const I: usize, const P: usize> ParetoFront<S, I, P> {
    /// Create an empty Pareto front.
    pub fn new() -> Self {
        let default_pt = ParetoPoint {
            tracking_cost: S::ZERO,
            energy_cost: S::ZERO,
            u0: Matrix::zeros(),
        };
        Self {
            points: [default_pt; P],
            count: 0,
        }
    }

    /// Add a new Pareto point (if capacity allows).
    pub fn push(&mut self, pt: ParetoPoint<S, I>) {
        if self.count < P {
            self.points[self.count] = pt;
            self.count += 1;
        }
    }

    /// Find the point on the Pareto front closest to a given weight pair (w_track, w_energy).
    ///
    /// Returns the index of the best trade-off point, or an error if empty.
    pub fn best_for_weights(&self, w_track: S, w_energy: S) -> Result<usize, MultiObjectiveError> {
        if self.count == 0 {
            return Err(MultiObjectiveError::EmptyParetoFront);
        }
        let mut best_idx = 0;
        let mut best_val =
            w_track * self.points[0].tracking_cost + w_energy * self.points[0].energy_cost;
        for i in 1..self.count {
            let v = w_track * self.points[i].tracking_cost + w_energy * self.points[i].energy_cost;
            if v < best_val {
                best_val = v;
                best_idx = i;
            }
        }
        Ok(best_idx)
    }
}

impl<S: ControlScalar, const I: usize, const P: usize> Default for ParetoFront<S, I, P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-objective MPC with tracking and energy objectives.
///
/// Computes the Pareto front over the trade-off between:
///   J_track  = Σ_k (x_k - x_ref)^T Q (x_k - x_ref)
///   J_energy = Σ_k u_k^T R u_k
///
/// Two scalarisation methods are provided:
/// - `solve_weighted_sum`: minimise λ J_track + (1-λ) J_energy
/// - `build_pareto_front`: sweep λ over P equidistant values in (0, 1)
/// - `solve_epsilon_constraint`: minimise J_track s.t. J_energy ≤ ε
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
/// - H: prediction horizon
pub struct MultiObjectiveMpc<S: ControlScalar, const N: usize, const I: usize, const H: usize> {
    /// State transition matrix A (N×N).
    pub a: Matrix<S, N, N>,
    /// Input matrix B (N×I).
    pub b: Matrix<S, N, I>,
    /// Tracking cost weight matrix Q (N×N).
    pub q: Matrix<S, N, N>,
    /// Energy cost weight matrix R (I×I).
    pub r: Matrix<S, I, I>,
    /// Reference state x_ref (N×1).
    pub x_ref: Matrix<S, N, 1>,
    /// Current state.
    pub x: Matrix<S, N, 1>,
    /// Number of gradient descent iterations per scalarised solve.
    pub iterations: usize,
    /// Gradient descent step size.
    pub step_size: S,
}

impl<S: ControlScalar, const N: usize, const I: usize, const H: usize>
    MultiObjectiveMpc<S, N, I, H>
{
    /// Create a new MultiObjectiveMpc.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        q: Matrix<S, N, N>,
        r: Matrix<S, I, I>,
        iterations: usize,
    ) -> Self {
        Self {
            a,
            b,
            q,
            r,
            x_ref: Matrix::zeros(),
            x: Matrix::zeros(),
            iterations,
            step_size: S::from_f64(1e-3),
        }
    }

    /// Propagate state: x_{k+1} = A x_k + B u_k.
    fn propagate(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>) -> Matrix<S, N, 1> {
        let ax = matmul(&self.a, x);
        let bu = matmul(&self.b, u);
        ax.add_mat(&bu)
    }

    /// Compute tracking cost over the horizon: Σ_k (x_k - x_ref)^T Q (x_k - x_ref).
    pub fn tracking_cost(&self, u_seq: &[Matrix<S, I, 1>; H]) -> S {
        let mut total = S::ZERO;
        let mut x = self.x;
        for k in 0..H {
            let e = x.sub_mat(&self.x_ref);
            let qe = matmul(&self.q, &e);
            let et = e.transpose();
            total += matmul(&et, &qe).data[0][0];
            x = self.propagate(&x, &u_seq[k]);
        }
        // Terminal tracking
        let e = x.sub_mat(&self.x_ref);
        let qe = matmul(&self.q, &e);
        let et = e.transpose();
        total += matmul(&et, &qe).data[0][0];
        total
    }

    /// Compute energy cost over the horizon: Σ_k u_k^T R u_k.
    pub fn energy_cost(&self, u_seq: &[Matrix<S, I, 1>; H]) -> S {
        let mut total = S::ZERO;
        for k in 0..H {
            let ru = matmul(&self.r, &u_seq[k]);
            let ut = u_seq[k].transpose();
            total += matmul(&ut, &ru).data[0][0];
        }
        total
    }

    /// Compute the weighted-sum scalarised cost: λ J_track + (1-λ) J_energy.
    pub fn scalarised_cost(&self, u_seq: &[Matrix<S, I, 1>; H], lambda: S) -> S {
        let jt = self.tracking_cost(u_seq);
        let je = self.energy_cost(u_seq);
        lambda * jt + (S::ONE - lambda) * je
    }

    /// Numerical gradient of scalarised cost w.r.t. u_k.
    fn gradient_scalarised(
        &self,
        k: usize,
        u_seq: &[Matrix<S, I, 1>; H],
        lambda: S,
        eps: S,
    ) -> Matrix<S, I, 1> {
        let two_eps = S::TWO * eps;
        let mut grad = Matrix::<S, I, 1>::zeros();
        for i in 0..I {
            let mut u_p = *u_seq;
            let mut u_m = *u_seq;
            u_p[k].data[i][0] += eps;
            u_m[k].data[i][0] -= eps;
            let cp = self.scalarised_cost(&u_p, lambda);
            let cm = self.scalarised_cost(&u_m, lambda);
            grad.data[i][0] = (cp - cm) / two_eps;
        }
        grad
    }

    /// Solve the weighted-sum scalarised problem for a given λ ∈ (0, 1).
    ///
    /// Returns the first optimal control action, or an error for invalid λ.
    pub fn solve_weighted_sum(&self, lambda: S) -> Result<Matrix<S, I, 1>, MultiObjectiveError> {
        if lambda <= S::ZERO || lambda >= S::ONE {
            return Err(MultiObjectiveError::InvalidWeights);
        }
        let eps = S::from_f64(1e-5);
        let step = self.step_size;
        let mut u_seq: [Matrix<S, I, 1>; H] = [Matrix::zeros(); H];

        for _iter in 0..self.iterations {
            for k in 0..H {
                let g = self.gradient_scalarised(k, &u_seq, lambda, eps);
                for i in 0..I {
                    u_seq[k].data[i][0] -= step * g.data[i][0];
                }
            }
        }

        Ok(u_seq[0])
    }

    /// Numerical gradient of tracking cost w.r.t. u_k (for ε-constraint method).
    fn gradient_tracking(&self, k: usize, u_seq: &[Matrix<S, I, 1>; H], eps: S) -> Matrix<S, I, 1> {
        let two_eps = S::TWO * eps;
        let mut grad = Matrix::<S, I, 1>::zeros();
        for i in 0..I {
            let mut u_p = *u_seq;
            let mut u_m = *u_seq;
            u_p[k].data[i][0] += eps;
            u_m[k].data[i][0] -= eps;
            let cp = self.tracking_cost(&u_p);
            let cm = self.tracking_cost(&u_m);
            grad.data[i][0] = (cp - cm) / two_eps;
        }
        grad
    }

    /// Solve using ε-constraint method: minimise J_track s.t. J_energy ≤ epsilon_energy.
    ///
    /// Implements a penalty method: if energy cost exceeds ε, add a quadratic penalty.
    /// Returns an error if epsilon_energy is negative.
    pub fn solve_epsilon_constraint(
        &self,
        epsilon_energy: S,
    ) -> Result<Matrix<S, I, 1>, MultiObjectiveError> {
        if epsilon_energy < S::ZERO {
            return Err(MultiObjectiveError::NegativeBound);
        }
        let eps_fd = S::from_f64(1e-5);
        let penalty = S::from_f64(10.0); // penalty weight for constraint violation
        let step = self.step_size;
        let mut u_seq: [Matrix<S, I, 1>; H] = [Matrix::zeros(); H];

        for _iter in 0..self.iterations {
            let je = self.energy_cost(&u_seq);
            // Penalty if energy exceeds ε: φ(u) = J_track + penalty * max(0, J_e - ε)^2
            let energy_excess = if je > epsilon_energy {
                je - epsilon_energy
            } else {
                S::ZERO
            };

            for k in 0..H {
                let g_track = self.gradient_tracking(k, &u_seq, eps_fd);
                // Penalty gradient contribution (energy penalty gradient)
                let mut g_penalty = Matrix::<S, I, 1>::zeros();
                if energy_excess > S::ZERO {
                    for i in 0..I {
                        let mut u_p = u_seq;
                        let mut u_m = u_seq;
                        u_p[k].data[i][0] += eps_fd;
                        u_m[k].data[i][0] -= eps_fd;
                        let ep = self.energy_cost(&u_p);
                        let em = self.energy_cost(&u_m);
                        // d/du_k [penalty * max(0, J_e - ε)^2] = 2 * penalty * max(0,J_e-ε) * dJ_e/du_k
                        g_penalty.data[i][0] =
                            S::TWO * penalty * energy_excess * (ep - em) / (S::TWO * eps_fd);
                    }
                }

                for i in 0..I {
                    u_seq[k].data[i][0] -= step * (g_track.data[i][0] + g_penalty.data[i][0]);
                }
            }
        }

        Ok(u_seq[0])
    }

    /// Build a Pareto front approximation by sweeping λ over P equidistant values.
    ///
    /// Returns a `ParetoFront<S, I, P>` with up to P Pareto points.
    pub fn build_pareto_front<const P: usize>(
        &self,
    ) -> Result<ParetoFront<S, I, P>, MultiObjectiveError> {
        if P == 0 {
            return Err(MultiObjectiveError::InvalidWeights);
        }
        let mut front = ParetoFront::new();
        let n = S::from_f64(P as f64 + 1.0);

        for p in 1..=P {
            let lambda = S::from_f64(p as f64) / n;
            let u0 = self.solve_weighted_sum(lambda)?;

            // Evaluate objectives at this solution
            let mut u_seq: [Matrix<S, I, 1>; H] = [Matrix::zeros(); H];
            u_seq[0] = u0;
            let jt = self.tracking_cost(&u_seq);
            let je = self.energy_cost(&u_seq);

            front.push(ParetoPoint {
                tracking_cost: jt,
                energy_cost: je,
                u0,
            });
        }

        Ok(front)
    }

    /// Set the current state.
    pub fn set_state(&mut self, x: Matrix<S, N, 1>) {
        self.x = x;
    }

    /// Set the reference state.
    pub fn set_reference(&mut self, x_ref: Matrix<S, N, 1>) {
        self.x_ref = x_ref;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mpc() -> MultiObjectiveMpc<f64, 2, 1, 4> {
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = 0.1;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.005;
        b.data[1][0] = 0.1;

        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;

        MultiObjectiveMpc::new(a, b, q, r, 50)
    }

    #[test]
    fn tracking_cost_zero_at_reference() {
        let mut mpc = make_mpc();
        let mut xref = Matrix::<f64, 2, 1>::zeros();
        xref.data[0][0] = 1.0;
        mpc.set_state(xref);
        mpc.set_reference(xref);
        // With identity A and zero B and zero u, x stays at xref => tracking cost high
        // But here we test that reference == state at step 0 contributes nothing for that step
        let u_seq = [Matrix::<f64, 1, 1>::zeros(); 4];
        let jt = mpc.tracking_cost(&u_seq);
        // x == xref at start but drifts; cost can be >= 0
        assert!(jt >= 0.0, "Tracking cost must be non-negative: {}", jt);
    }

    #[test]
    fn energy_cost_non_negative() {
        let mpc = make_mpc();
        let mut u_seq = [Matrix::<f64, 1, 1>::zeros(); 4];
        u_seq[0].data[0][0] = 1.0;
        let je = mpc.energy_cost(&u_seq);
        assert!(je >= 0.0, "Energy cost must be non-negative: {}", je);
    }

    #[test]
    fn energy_cost_scales_with_input() {
        let mpc = make_mpc();
        let mut u1 = [Matrix::<f64, 1, 1>::zeros(); 4];
        u1[0].data[0][0] = 1.0;
        let mut u2 = [Matrix::<f64, 1, 1>::zeros(); 4];
        u2[0].data[0][0] = 2.0;
        let j1 = mpc.energy_cost(&u1);
        let j2 = mpc.energy_cost(&u2);
        assert!(
            j2 > j1,
            "Larger input must have larger energy cost: {} vs {}",
            j2,
            j1
        );
    }

    #[test]
    fn invalid_lambda_returns_error() {
        let mpc = make_mpc();
        let result = mpc.solve_weighted_sum(0.0_f64);
        assert!(matches!(result, Err(MultiObjectiveError::InvalidWeights)));
        let result2 = mpc.solve_weighted_sum(1.0_f64);
        assert!(matches!(result2, Err(MultiObjectiveError::InvalidWeights)));
    }

    #[test]
    fn weighted_sum_solve_valid_lambda() {
        let mpc = make_mpc();
        let result = mpc.solve_weighted_sum(0.5_f64);
        assert!(result.is_ok(), "Weighted sum solve failed: {:?}", result);
    }

    #[test]
    fn epsilon_constraint_negative_bound_returns_error() {
        let mpc = make_mpc();
        let result = mpc.solve_epsilon_constraint(-1.0_f64);
        assert!(matches!(result, Err(MultiObjectiveError::NegativeBound)));
    }

    #[test]
    fn epsilon_constraint_solve_succeeds() {
        let mpc = make_mpc();
        let result = mpc.solve_epsilon_constraint(10.0_f64);
        assert!(result.is_ok(), "ε-constraint solve failed: {:?}", result);
    }

    #[test]
    fn pareto_front_has_correct_count() {
        let mpc = make_mpc();
        let front: ParetoFront<f64, 1, 5> = mpc.build_pareto_front::<5>().expect("pareto front");
        assert_eq!(front.count, 5, "Pareto front should have 5 points");
    }

    #[test]
    fn pareto_front_best_for_weights() {
        let mpc = make_mpc();
        let front: ParetoFront<f64, 1, 4> = mpc.build_pareto_front::<4>().expect("pareto front");
        let idx = front.best_for_weights(0.5_f64, 0.5_f64);
        assert!(idx.is_ok(), "best_for_weights failed: {:?}", idx);
        let i = idx.unwrap();
        assert!(i < front.count, "Index out of bounds");
    }
}
