//! Tube MPC: nominal MPC trajectory with ancillary LQR feedback for robustness.
//!
//! The tube approach separates the control into:
//! 1. Nominal MPC: plans a trajectory for the undisturbed system.
//! 2. Ancillary LQR: corrects errors between the nominal and actual state.
//!
//! The "tube" is the invariant set around the nominal trajectory within which
//! the actual trajectory remains despite bounded disturbances.
#![allow(unused)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Tube MPC combining nominal MPC trajectory with LQR error feedback.
///
/// Control law:
///   u = u_nom + K_lqr * (x_nominal - x_actual)
///
/// The nominal MPC uses tightened constraints (reduced by `tube_size`) to ensure
/// the actual trajectory remains feasible despite disturbances.
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
/// - H: prediction horizon
pub struct TubeMpc<S: ControlScalar, const N: usize, const I: usize, const H: usize> {
    /// State transition matrix A (N×N).
    pub a: Matrix<S, N, N>,
    /// Input matrix B (N×I).
    pub b: Matrix<S, N, I>,
    /// State cost weight Q (N×N).
    pub q: Matrix<S, N, N>,
    /// Input cost weight R (I×I).
    pub r: Matrix<S, I, I>,
    /// Ancillary LQR gain K_lqr (I×N): corrects nominal-actual error.
    pub k_lqr: Matrix<S, I, N>,
    /// Tube size used for constraint tightening.
    pub tube_size: S,
    /// Nominal state (tube centre).
    pub x_nominal: Matrix<S, N, 1>,
    /// Actual (measured) state.
    pub x_actual: Matrix<S, N, 1>,
    /// Prediction horizon (≤ H).
    pub horizon: usize,
}

impl<S: ControlScalar, const N: usize, const I: usize, const H: usize> TubeMpc<S, N, I, H> {
    /// Create a new TubeMpc.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        q: Matrix<S, N, N>,
        r: Matrix<S, I, I>,
        k_lqr: Matrix<S, I, N>,
        tube_size: S,
    ) -> Self {
        Self {
            a,
            b,
            q,
            r,
            k_lqr,
            tube_size,
            x_nominal: Matrix::zeros(),
            x_actual: Matrix::zeros(),
            horizon: H,
        }
    }

    /// Compute nominal stage cost: x^T Q x + u^T R u.
    fn stage_cost(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>) -> S {
        let qx = matmul(&self.q, x);
        let xt = x.transpose();
        let cost_x = matmul(&xt, &qx).data[0][0];

        let ru = matmul(&self.r, u);
        let ut = u.transpose();
        let cost_u = matmul(&ut, &ru).data[0][0];

        cost_x + cost_u
    }

    /// Gradient of cost w.r.t. u (central differences).
    fn grad_u_cost(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>, eps: S) -> Matrix<S, I, 1> {
        let mut grad = Matrix::<S, I, 1>::zeros();
        for i in 0..I {
            let mut u_p = *u;
            let mut u_m = *u;
            u_p.data[i][0] += eps;
            u_m.data[i][0] -= eps;
            grad.data[i][0] =
                (self.stage_cost(x, &u_p) - self.stage_cost(x, &u_m)) / (S::TWO * eps);
        }
        grad
    }

    /// Solve the nominal MPC problem via gradient descent.
    ///
    /// Returns the first element of the optimal input sequence.
    fn solve_nominal(&self) -> Matrix<S, I, 1> {
        let step = S::from_f64(5e-4);
        let eps = S::from_f64(1e-4);
        let max_iter = 60_usize;
        let horizon = self.horizon.min(H);

        let mut u_seq: [Matrix<S, I, 1>; H] = [Matrix::zeros(); H];

        for _it in 0..max_iter {
            let mut x = self.x_nominal;
            let mut xs: [Matrix<S, N, 1>; H] = [Matrix::zeros(); H];
            for k in 0..horizon {
                let ax = matmul(&self.a, &x);
                let bu = matmul(&self.b, &u_seq[k]);
                x = ax.add_mat(&bu);
                xs[k] = x;
            }

            for k in 0..horizon {
                let g = self.grad_u_cost(&xs[k], &u_seq[k], eps);
                for i in 0..I {
                    u_seq[k].data[i][0] -= step * g.data[i][0];
                }
            }
        }

        u_seq[0]
    }

    /// Compute total nominal MPC cost for a given input sequence.
    pub fn nominal_cost(&self, u_seq: &[Matrix<S, I, 1>; H]) -> S {
        let mut total = S::ZERO;
        let mut x = self.x_nominal;
        for u_k in u_seq.iter().take(self.horizon.min(H)) {
            let ax = matmul(&self.a, &x);
            let bu = matmul(&self.b, u_k);
            x = ax.add_mat(&bu);
            total += self.stage_cost(&x, u_k);
        }
        total
    }

    /// Compute nominal + ancillary control, and update the nominal state.
    ///
    /// The total control applied is:
    ///   u = u_nom + K_lqr * (x_nominal - x_actual)
    pub fn step(&mut self, x_actual: Matrix<S, N, 1>) -> Matrix<S, I, 1> {
        self.x_actual = x_actual;

        // Solve nominal MPC
        let u_nom = self.solve_nominal();

        // Error between nominal and actual state
        let e = self.x_nominal.sub_mat(&self.x_actual);

        // Ancillary correction: K_lqr * e
        let u_correction = matmul(&self.k_lqr, &e);

        // Total control
        let u_total = u_nom.add_mat(&u_correction);

        // Propagate nominal state
        let ax = matmul(&self.a, &self.x_nominal);
        let bu = matmul(&self.b, &u_nom);
        self.x_nominal = ax.add_mat(&bu);

        u_total
    }

    /// Tighten a scalar constraint by subtracting the tube size.
    ///
    /// Returns `constraint - tube_size` (the tightened bound).
    pub fn tighten_constraint(&self, constraint: S) -> S {
        constraint - self.tube_size
    }

    /// Compute the tube error (||x_actual - x_nominal||_2).
    pub fn tube_error(&self) -> S {
        let diff = self.x_actual.sub_mat(&self.x_nominal);
        diff.frob_norm()
    }

    /// Reset nominal state to the given actual state.
    pub fn reset_nominal(&mut self, x: Matrix<S, N, 1>) {
        self.x_nominal = x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Mat2x2 = Matrix<f64, 2, 2>;
    type Mat2x1 = Matrix<f64, 2, 1>;
    type Mat1x1 = Matrix<f64, 1, 1>;
    type Mat1x2 = Matrix<f64, 1, 2>;

    fn simple_system() -> (Mat2x2, Mat2x1, Mat2x2, Mat1x1, Mat1x2) {
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = 0.1;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.005;
        b.data[1][0] = 0.1;

        let q = Matrix::<f64, 2, 2>::identity();

        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;

        let mut k = Matrix::<f64, 1, 2>::zeros();
        k.data[0][0] = 3.0;
        k.data[0][1] = 1.5;

        (a, b, q, r, k)
    }

    #[test]
    fn tube_mpc_construction() {
        let (a, b, q, r, k) = simple_system();
        let mpc = TubeMpc::<f64, 2, 1, 5>::new(a, b, q, r, k, 0.1_f64);
        assert_eq!(mpc.tube_size, 0.1);
        assert_eq!(mpc.horizon, 5);
    }

    #[test]
    fn tighten_constraint() {
        let (a, b, q, r, k) = simple_system();
        let mpc = TubeMpc::<f64, 2, 1, 5>::new(a, b, q, r, k, 0.1_f64);
        let tightened = mpc.tighten_constraint(1.0_f64);
        assert!((tightened - 0.9).abs() < 1e-12, "Tightened: {}", tightened);
    }

    #[test]
    fn tube_mpc_step_produces_control() {
        let (a, b, q, r, k) = simple_system();
        let mut mpc = TubeMpc::<f64, 2, 1, 5>::new(a, b, q, r, k, 0.05_f64);

        // Set nominal state at 1.0
        let mut xn = Matrix::<f64, 2, 1>::zeros();
        xn.data[0][0] = 1.0;
        mpc.reset_nominal(xn);

        // Actual state at 0.0 (error = 1.0)
        let xa = Matrix::<f64, 2, 1>::zeros();
        let u = mpc.step(xa);

        // With nonzero error, control should be nonzero
        assert!(
            u.data[0][0].abs() > 0.0,
            "Control should be nonzero for nonzero tube error: {}",
            u.data[0][0]
        );
    }

    #[test]
    fn tube_error_is_zero_when_states_equal() {
        let (a, b, q, r, k) = simple_system();
        let mut mpc = TubeMpc::<f64, 2, 1, 5>::new(a, b, q, r, k, 0.05_f64);
        let x = Matrix::<f64, 2, 1>::zeros();
        mpc.x_actual = x;
        mpc.x_nominal = x;
        assert!(
            mpc.tube_error() < 1e-12,
            "Error should be zero: {}",
            mpc.tube_error()
        );
    }
}
