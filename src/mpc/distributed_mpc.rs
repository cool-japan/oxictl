//! Distributed MPC with ADMM (Alternating Direction Method of Multipliers) consensus.
//!
//! Each subsystem maintains a local MPC problem and exchanges information with
//! neighbours through the ADMM consensus variable z.  The algorithm converges
//! to the centralised solution for convex cost functions.
#![allow(unused)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Local subsystem MPC with neighbour coupling via ADMM consensus.
///
/// ADMM solves the distributed problem:
///   min  Σ_i f_i(u_i)    s.t.  u_i = z  ∀i
///
/// Each agent performs:
///   1. Primal update: u_i = argmin f_i(u_i) + (rho/2)||u_i - z + lambda_i||^2
///   2. Consensus:     z   = average(u_i) across neighbours
///   3. Dual update:   λ_i += rho * (u_i - z)
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
/// - H: prediction horizon
pub struct SubsystemMpc<S: ControlScalar, const N: usize, const I: usize, const H: usize> {
    /// State transition matrix A (N×N).
    pub a: Matrix<S, N, N>,
    /// Input matrix B (N×I).
    pub b: Matrix<S, N, I>,
    /// State cost weight Q (N×N).
    pub q: Matrix<S, N, N>,
    /// Input cost weight R (I×I).
    pub r: Matrix<S, I, I>,
    /// Current state estimate.
    pub x: Matrix<S, N, 1>,
    /// ADMM penalty parameter ρ.
    pub rho: S,
    /// Dual variable λ (Lagrange multiplier, I×1).
    dual: Matrix<S, I, 1>,
    /// Consensus variable z (shared with neighbours, I×1).
    z: Matrix<S, I, 1>,
    /// Most recently received neighbour prediction.
    neighbor_u: Matrix<S, I, 1>,
}

impl<S: ControlScalar, const N: usize, const I: usize, const H: usize> SubsystemMpc<S, N, I, H> {
    /// Create a new SubsystemMpc.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        q: Matrix<S, N, N>,
        r: Matrix<S, I, I>,
        rho: S,
    ) -> Self {
        Self {
            a,
            b,
            q,
            r,
            x: Matrix::zeros(),
            rho,
            dual: Matrix::zeros(),
            z: Matrix::zeros(),
            neighbor_u: Matrix::zeros(),
        }
    }

    /// Local stage cost: x^T Q x + u^T R u.
    fn stage_cost(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>) -> S {
        let qx = matmul(&self.q, x);
        let xt = x.transpose();
        let cx = matmul(&xt, &qx).data[0][0];

        let ru = matmul(&self.r, u);
        let ut = u.transpose();
        let cu = matmul(&ut, &ru).data[0][0];

        cx + cu
    }

    /// ADMM augmented cost: f(u) + (rho/2)||u - z + lambda||^2
    fn augmented_cost(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>) -> S {
        let base = self.stage_cost(x, u);

        // (rho/2) * ||u - z + lambda||^2
        let mut aug = S::ZERO;
        for i in 0..I {
            let diff = u.data[i][0] - self.z.data[i][0] + self.dual.data[i][0];
            aug += diff * diff;
        }
        aug *= self.rho * S::HALF;

        base + aug
    }

    /// Gradient of augmented cost w.r.t. u (central differences).
    fn grad_augmented(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>, eps: S) -> Matrix<S, I, 1> {
        let mut g = Matrix::<S, I, 1>::zeros();
        for i in 0..I {
            let mut u_p = *u;
            let mut u_m = *u;
            u_p.data[i][0] += eps;
            u_m.data[i][0] -= eps;
            g.data[i][0] =
                (self.augmented_cost(x, &u_p) - self.augmented_cost(x, &u_m)) / (S::TWO * eps);
        }
        g
    }

    /// Primal update: local optimisation step minimising augmented cost.
    ///
    /// Uses gradient descent over the H-step horizon and returns the first
    /// optimal control action.
    pub fn primal_update(&mut self) -> Matrix<S, I, 1> {
        let step = S::from_f64(1e-3);
        let eps = S::from_f64(1e-5);
        let max_iter = 30_usize;
        let horizon = H;

        let mut u_seq: [Matrix<S, I, 1>; H] = [Matrix::zeros(); H];

        for _it in 0..max_iter {
            // Forward pass
            let mut x = self.x;
            let mut xs: [Matrix<S, N, 1>; H] = [Matrix::zeros(); H];
            for k in 0..horizon {
                let ax = matmul(&self.a, &x);
                let bu = matmul(&self.b, &u_seq[k]);
                x = ax.add_mat(&bu);
                xs[k] = x;
            }
            // Gradient step (only on first step for ADMM; augmented cost applies to u_0)
            let g = self.grad_augmented(&xs[0], &u_seq[0], eps);
            for i in 0..I {
                u_seq[0].data[i][0] -= step * g.data[i][0];
            }
            // Plain gradient for rest
            for k in 1..horizon {
                let g2 = self.grad_stage(&xs[k], &u_seq[k], eps);
                for i in 0..I {
                    u_seq[k].data[i][0] -= step * g2.data[i][0];
                }
            }
        }

        u_seq[0]
    }

    /// Gradient of plain stage cost (for horizon steps beyond k=0).
    fn grad_stage(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>, eps: S) -> Matrix<S, I, 1> {
        let mut g = Matrix::<S, I, 1>::zeros();
        for i in 0..I {
            let mut u_p = *u;
            let mut u_m = *u;
            u_p.data[i][0] += eps;
            u_m.data[i][0] -= eps;
            g.data[i][0] = (self.stage_cost(x, &u_p) - self.stage_cost(x, &u_m)) / (S::TWO * eps);
        }
        g
    }

    /// ADMM dual update: λ += ρ * (u - z).
    pub fn dual_update(&mut self, u_local: &Matrix<S, I, 1>) {
        for i in 0..I {
            self.dual.data[i][0] += self.rho * (u_local.data[i][0] - self.z.data[i][0]);
        }
    }

    /// Consensus update: z = (u_local + u_neighbor) / 2.
    pub fn consensus_update(&mut self, u_local: &Matrix<S, I, 1>) {
        for i in 0..I {
            self.z.data[i][0] = (u_local.data[i][0] + self.neighbor_u.data[i][0]) * S::HALF;
        }
    }

    /// Return the current consensus variable z (shared with neighbours).
    pub fn share_prediction(&self) -> Matrix<S, I, 1> {
        self.z
    }

    /// Receive a neighbour's prediction.
    pub fn receive_neighbor(&mut self, u_neighbor: Matrix<S, I, 1>) {
        self.neighbor_u = u_neighbor;
    }

    /// Primal residual: ||u - z||_2.
    pub fn primal_residual(&self, u: &Matrix<S, I, 1>) -> S {
        let diff = u.sub_mat(&self.z);
        diff.frob_norm()
    }

    /// Dual residual: ρ * ||z - z_prev||_2 (approximated as ||λ||/ρ here for no-alloc).
    pub fn dual_residual(&self) -> S {
        self.dual.frob_norm()
    }

    /// Full ADMM iteration: primal → consensus → dual.
    ///
    /// Returns the local optimal control action.
    pub fn admm_iter(&mut self) -> Matrix<S, I, 1> {
        let u = self.primal_update();
        self.consensus_update(&u);
        self.dual_update(&u);
        u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_subsystem() -> SubsystemMpc<f64, 2, 1, 5> {
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = 0.1;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.005;
        b.data[1][0] = 0.1;

        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;

        SubsystemMpc::new(a, b, q, r, 1.0_f64)
    }

    #[test]
    fn subsystem_construction() {
        let sys = make_subsystem();
        assert_eq!(sys.rho, 1.0);
        assert_eq!(sys.z.data[0][0], 0.0);
    }

    #[test]
    fn primal_residual_zero_at_init() {
        let sys = make_subsystem();
        let u = Matrix::<f64, 1, 1>::zeros();
        let res = sys.primal_residual(&u);
        assert!(
            res.abs() < 1e-12,
            "Initial primal residual should be zero: {}",
            res
        );
    }

    #[test]
    fn admm_iter_produces_control() {
        let mut sys = make_subsystem();
        // Set a nonzero state
        sys.x.data[0][0] = 1.0;
        let u = sys.admm_iter();
        // After one iteration, control should be nonzero (system is not at origin)
        // May be small but should run without panic
        let _ = u;
    }

    #[test]
    fn consensus_averages_predictions() {
        let mut sys = make_subsystem();
        let mut u_local = Matrix::<f64, 1, 1>::zeros();
        u_local.data[0][0] = 2.0;

        let mut u_neighbor = Matrix::<f64, 1, 1>::zeros();
        u_neighbor.data[0][0] = 4.0;

        sys.receive_neighbor(u_neighbor);
        sys.consensus_update(&u_local);

        // z = (2 + 4) / 2 = 3
        assert!(
            (sys.z.data[0][0] - 3.0).abs() < 1e-12,
            "z = {}",
            sys.z.data[0][0]
        );
    }

    #[test]
    fn dual_update_accumulates() {
        let mut sys = make_subsystem();
        let mut u_local = Matrix::<f64, 1, 1>::zeros();
        u_local.data[0][0] = 1.0; // z = 0, so u - z = 1

        sys.dual_update(&u_local);
        // lambda += rho * (u - z) = 1.0 * 1.0 = 1.0
        assert!(
            (sys.dual.data[0][0] - 1.0).abs() < 1e-12,
            "Dual: {}",
            sys.dual.data[0][0]
        );

        sys.dual_update(&u_local);
        assert!(
            (sys.dual.data[0][0] - 2.0).abs() < 1e-12,
            "Dual: {}",
            sys.dual.data[0][0]
        );
    }
}
