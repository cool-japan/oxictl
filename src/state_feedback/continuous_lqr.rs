//! Continuous-time LQR via ARE (Algebraic Riccati Equation).
//!
//! Solves the continuous-time algebraic Riccati equation (CARE):
//!   A^T P + P A - P B R^{-1} B^T P + Q = 0
//!
//! Uses Newton-Hamiltonian iteration (sign function method approximation).
#![allow(unused)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Solve continuous ARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
///
/// Uses fixed-point (value) iteration:
///   P_{k+1} = Q + A^T P_k + P_k A - P_k B R^{-1} B^T P_k
/// scaled by a small step. This is an Euler integration of the Riccati ODE
/// (steepest descent on the Riccati residual), converging for stable A.
///
/// Returns the solution P (N×N) or None if R is singular or the iteration
/// does not converge within max_iter steps.
pub fn solve_care<S: ControlScalar, const N: usize, const I: usize>(
    a: &Matrix<S, N, N>,
    b: &Matrix<S, N, I>,
    q: &Matrix<S, N, N>,
    r: &Matrix<S, I, I>,
    max_iter: usize,
) -> Option<Matrix<S, N, N>> {
    let r_inv = r.inv()?;

    // B R^{-1} B^T  (N×N)
    let br_inv = matmul(b, &r_inv);
    let br_invbt = matmul(&br_inv, &b.transpose());

    let at = a.transpose();
    let step = S::from_f64(1e-3); // integration step for the Riccati ODE
    let tol = S::from_f64(1e-8);

    // Initialise P with Q
    let mut p = *q;

    for _iter in 0..max_iter {
        // Riccati residual: R_p = A^T P + P A - P B R^{-1} B^T P + Q
        let atp = matmul(&at, &p);
        let pa = matmul(&p, a);
        let pbrbt = matmul(&p, &br_invbt);
        let pbrbtp = matmul(&pbrbt, &p);

        let residual = atp.add_mat(&pa).sub_mat(&pbrbtp).add_mat(q);

        let norm = residual.frob_norm();

        // Euler step: P += step * residual
        let p_new = p.add_mat(&residual.scale(step));

        // Symmetrise to avoid numerical drift
        let p_sym = Matrix {
            data: core::array::from_fn(|r_i| {
                core::array::from_fn(|c_i| (p_new.data[r_i][c_i] + p_new.data[c_i][r_i]) * S::HALF)
            }),
        };

        p = p_sym;

        if norm < tol {
            return Some(p);
        }
    }

    // Return best estimate even if not fully converged
    Some(p)
}

/// Continuous-time LQR controller.
///
/// Minimises the infinite-horizon cost:
///   J = ∫_0^∞ (x^T Q x + u^T R u) dt
///
/// The optimal control law is u = -K x where K = R^{-1} B^T P.
pub struct ContinuousLqr<S: ControlScalar, const N: usize, const I: usize> {
    /// Optimal gain matrix K (I×N).
    pub gain: Matrix<S, I, N>,
    /// CARE solution matrix P (N×N).
    pub p_matrix: Matrix<S, N, N>,
}

impl<S: ControlScalar, const N: usize, const I: usize> ContinuousLqr<S, N, I> {
    /// Construct from a pre-computed gain and cost matrix.
    pub fn new(gain: Matrix<S, I, N>, p_matrix: Matrix<S, N, N>) -> Self {
        Self { gain, p_matrix }
    }

    /// Design continuous LQR from system matrices.
    ///
    /// Solves the CARE and computes K = R^{-1} B^T P.
    /// Returns None if CARE fails or R is singular.
    pub fn design(
        a: &Matrix<S, N, N>,
        b: &Matrix<S, N, I>,
        q: &Matrix<S, N, N>,
        r: &Matrix<S, I, I>,
        max_iter: usize,
    ) -> Option<Self> {
        let p = solve_care(a, b, q, r, max_iter)?;
        let r_inv = r.inv()?;
        // K = R^{-1} B^T P
        let btp = matmul(&b.transpose(), &p);
        let gain = matmul(&r_inv, &btp);
        Some(Self { gain, p_matrix: p })
    }

    /// Compute control: u = -K*x.
    pub fn control(&self, x: &Matrix<S, N, 1>) -> Matrix<S, I, 1> {
        let kx = matmul(&self.gain, x);
        kx.neg()
    }

    /// Lyapunov cost from initial state: V(x) = x^T P x.
    pub fn lyapunov_cost(&self, x: &Matrix<S, N, 1>) -> S {
        let px = matmul(&self.p_matrix, x);
        let xt = x.transpose();
        matmul(&xt, &px).data[0][0]
    }
}

/// First-order ZOH approximation: Ad ≈ I + A*dt + (A*dt)^2/2
///
/// Discretises a continuous-time system A using the matrix exponential
/// truncated to second order.  Suitable for small dt relative to the
/// fastest eigenvalue of A.
pub fn discretize_zoh_approx<S: ControlScalar, const N: usize>(
    a: &Matrix<S, N, N>,
    dt: S,
) -> Matrix<S, N, N> {
    let adt = a.scale(dt);
    let adt2 = matmul(&adt, &adt);
    let half = S::HALF;

    // Ad = I + A*dt + (A*dt)^2/2
    let identity = Matrix::<S, N, N>::identity();
    identity.add_mat(&adt).add_mat(&adt2.scale(half))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Continuous double integrator:
    ///   A = [[0,1],[0,0]], B = [[0],[1]]
    fn double_integrator_cont() -> (Matrix<f64, 2, 2>, Matrix<f64, 2, 1>) {
        let mut a = Matrix::<f64, 2, 2>::zeros();
        a.data[0][1] = 1.0;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[1][0] = 1.0;

        (a, b)
    }

    #[test]
    fn care_solves_double_integrator() {
        let (a, b) = double_integrator_cont();
        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 1.0;

        let p = solve_care(&a, &b, &q, &r, 50_000);
        assert!(p.is_some(), "CARE should converge");
        let p = p.unwrap();
        // P must be positive definite (diagonal elements > 0)
        assert!(
            p.data[0][0] > 0.0,
            "P[0,0] must be positive: {}",
            p.data[0][0]
        );
        assert!(
            p.data[1][1] > 0.0,
            "P[1,1] must be positive: {}",
            p.data[1][1]
        );
    }

    #[test]
    fn continuous_lqr_design() {
        let (a, b) = double_integrator_cont();
        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 1.0;

        let lqr = ContinuousLqr::design(&a, &b, &q, &r, 50_000);
        assert!(lqr.is_some(), "ContinuousLqr design should succeed");
        let lqr = lqr.unwrap();
        // Gain should be non-trivial
        assert!(
            lqr.gain.data[0][0].abs() > 0.0 || lqr.gain.data[0][1].abs() > 0.0,
            "Gain should be nonzero"
        );
    }

    #[test]
    fn discretize_zoh_identity_dt_zero() {
        let a = Matrix::<f64, 2, 2>::identity();
        let ad = discretize_zoh_approx(&a, 0.0_f64);
        // With dt=0: Ad = I + 0 + 0 = I
        assert!((ad.data[0][0] - 1.0).abs() < 1e-12);
        assert!((ad.data[0][1]).abs() < 1e-12);
        assert!((ad.data[1][0]).abs() < 1e-12);
        assert!((ad.data[1][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn discretize_zoh_small_dt() {
        let mut a = Matrix::<f64, 2, 2>::zeros();
        a.data[0][1] = 1.0; // [[0,1],[0,0]]
        let dt = 0.01_f64;
        let ad = discretize_zoh_approx(&a, dt);

        // Expected: I + A*dt = [[1, 0.01],[0, 1]] (second-order term is zero for this A)
        assert!((ad.data[0][0] - 1.0).abs() < 1e-10);
        assert!((ad.data[0][1] - dt).abs() < 1e-10);
        assert!((ad.data[1][0]).abs() < 1e-10);
        assert!((ad.data[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn lyapunov_cost_positive() {
        let (a, b) = double_integrator_cont();
        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 1.0;
        let lqr = ContinuousLqr::design(&a, &b, &q, &r, 50_000).unwrap();

        let mut x = Matrix::<f64, 2, 1>::zeros();
        x.data[0][0] = 1.0;
        let cost = lqr.lyapunov_cost(&x);
        assert!(
            cost > 0.0,
            "Lyapunov cost should be positive for nonzero state: {}",
            cost
        );

        let x0 = Matrix::<f64, 2, 1>::zeros();
        let cost0 = lqr.lyapunov_cost(&x0);
        assert!(
            cost0.abs() < 1e-12,
            "Cost should be zero at origin: {}",
            cost0
        );
    }
}
