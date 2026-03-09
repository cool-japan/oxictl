//! H2 optimal control via DARE.
//!
//! The H2 optimal controller minimizes the H2 norm of the closed-loop
//! transfer function, equivalent to minimizing the LQR cost with specific
//! disturbance input structure.
#![allow(unused)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;
use crate::state_feedback::lqr::solve_dare;

/// Solve H2 optimal control problem using DARE.
///
/// Returns optimal gain K (I×N) such that u = -K*x minimizes the H2 cost.
/// A: N×N state matrix, B: N×I input matrix, Q: N×N state cost, R: I×I input cost.
/// Returns None if DARE does not converge or R+B^T P B is singular.
pub fn solve_h2_dare<S: ControlScalar, const N: usize, const I: usize>(
    a: &Matrix<S, N, N>,
    b: &Matrix<S, N, I>,
    q: &Matrix<S, N, N>,
    r: &Matrix<S, I, I>,
) -> Option<(Matrix<S, I, N>, Matrix<S, N, N>)> {
    let sol = solve_dare(a, b, q, r, 2000, S::from_f64(1e-10))?;
    Some((sol.k, sol.p))
}

/// H2 norm bound: trace(B_d^T P B_d) where P is the cost matrix.
///
/// Gives an upper bound on the H2 norm of the closed-loop system
/// from disturbance input B_d to the regulated output.
pub fn h2_norm_bound<S: ControlScalar, const N: usize, const I: usize>(
    p: &Matrix<S, N, N>,
    b_disturbance: &Matrix<S, N, I>,
) -> S {
    // B_d^T P B_d (I×I)
    let bt = b_disturbance.transpose();
    let pb = matmul(p, b_disturbance);
    let btpb = matmul(&bt, &pb);
    btpb.trace()
}

/// H2 Controller: u = -K*x with H2-optimal gain.
///
/// Derived from DARE solution with the given state and input weights.
/// The controller minimises H2 performance criterion.
#[derive(Debug, Clone, Copy)]
pub struct H2Controller<S: ControlScalar, const N: usize, const I: usize> {
    /// Optimal gain matrix K (I×N).
    pub gain: Matrix<S, I, N>,
    /// H2 norm bound (scalar).
    pub h2_norm: S,
    /// DARE solution matrix P (N×N).
    pub p_matrix: Matrix<S, N, N>,
}

impl<S: ControlScalar, const N: usize, const I: usize> H2Controller<S, N, I> {
    /// Construct directly from pre-computed gain and H2 norm.
    pub fn new(gain: Matrix<S, I, N>, h2_norm: S) -> Self {
        Self {
            gain,
            h2_norm,
            p_matrix: Matrix::zeros(),
        }
    }

    /// Design H2 controller from system matrices and cost weights.
    ///
    /// Solves DARE and computes H2 norm bound using the disturbance input matrix.
    /// Returns None if DARE fails.
    pub fn design(
        a: &Matrix<S, N, N>,
        b: &Matrix<S, N, I>,
        q: &Matrix<S, N, N>,
        r: &Matrix<S, I, I>,
        b_disturbance: &Matrix<S, N, I>,
    ) -> Option<Self> {
        let (gain, p) = solve_h2_dare(a, b, q, r)?;
        let norm = h2_norm_bound(&p, b_disturbance);
        Some(Self {
            gain,
            h2_norm: norm,
            p_matrix: p,
        })
    }

    /// Compute control input: u = -K*x.
    pub fn control(&self, x: &Matrix<S, N, 1>) -> Matrix<S, I, 1> {
        let kx = matmul(&self.gain, x);
        kx.neg()
    }

    /// Compute control for state given as array slice.
    pub fn control_arr(&self, x: &[S; N]) -> [S; I] {
        let xm = Matrix {
            data: core::array::from_fn(|r| [x[r]]),
        };
        let u = self.control(&xm);
        core::array::from_fn(|i| u.data[i][0])
    }

    /// Return the steady-state cost for a given initial state.
    ///
    /// The H2 cost from state x is x^T P x.
    pub fn cost_from_state(&self, x: &Matrix<S, N, 1>) -> S {
        let px = matmul(&self.p_matrix, x);
        let xt = x.transpose();
        let cost = matmul(&xt, &px);
        cost.data[0][0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn double_integrator() -> (Matrix<f64, 2, 2>, Matrix<f64, 2, 1>) {
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = 0.1;
        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.005;
        b.data[1][0] = 0.1;
        (a, b)
    }

    #[test]
    fn h2_dare_converges() {
        let (a, b) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;
        let result = solve_h2_dare(&a, &b, &q, &r);
        assert!(result.is_some(), "H2 DARE should converge");
        let (k, _p) = result.unwrap();
        // Gain should be nonzero
        assert!(k.data[0][0].abs() > 0.0 || k.data[0][1].abs() > 0.0);
    }

    #[test]
    fn h2_controller_stabilizes() {
        let (a, b) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;
        let ctrl = H2Controller::design(&a, &b, &q, &r, &b).unwrap();

        // Simulate: x0 = [1, 0], should converge to 0
        let mut x = Matrix::<f64, 2, 1>::zeros();
        x.data[0][0] = 1.0;
        for _ in 0..300 {
            let u = ctrl.control(&x);
            let ax = matmul(&a, &x);
            let bu = matmul(&b, &u);
            x = ax.add_mat(&bu);
        }
        assert!(
            x.data[0][0].abs() < 0.01,
            "Position should converge: {}",
            x.data[0][0]
        );
        assert!(
            x.data[1][0].abs() < 0.01,
            "Velocity should converge: {}",
            x.data[1][0]
        );
    }

    #[test]
    fn h2_norm_bound_nonneg() {
        let (a, b) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;
        let (_k, p) = solve_h2_dare(&a, &b, &q, &r).unwrap();
        let norm = h2_norm_bound(&p, &b);
        assert!(norm >= 0.0, "H2 norm bound must be non-negative: {}", norm);
    }

    #[test]
    fn h2_cost_from_state() {
        let (a, b) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;
        let ctrl = H2Controller::design(&a, &b, &q, &r, &b).unwrap();

        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 1.0;
        let cost = ctrl.cost_from_state(&x0);
        assert!(
            cost > 0.0,
            "Cost from non-zero state should be positive: {}",
            cost
        );

        let x_zero = Matrix::<f64, 2, 1>::zeros();
        let cost_zero = ctrl.cost_from_state(&x_zero);
        assert!(
            cost_zero.abs() < 1e-12,
            "Cost from zero state should be zero: {}",
            cost_zero
        );
    }
}
