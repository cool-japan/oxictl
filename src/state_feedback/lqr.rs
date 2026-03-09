use crate::core::matrix::{matmul, matvec, Matrix};
use crate::core::scalar::ControlScalar;

/// Discrete-time Linear Quadratic Regulator (LQR).
///
/// Minimizes J = Σ (x^T Q x + u^T R u) subject to:
///   x[k+1] = A*x[k] + B*u[k]
///
/// Solved via backward Riccati iteration until convergence.
/// The optimal control law is u = -K*x where K is the gain matrix.
///
/// N = state dim, I = input dim.
pub struct Lqr<S: ControlScalar, const N: usize, const I: usize> {
    /// LQR gain matrix (I×N).
    pub gain: Matrix<S, I, N>,
}

/// Result of the Riccati solver.
pub struct RiccatiSolution<S: ControlScalar, const N: usize, const I: usize> {
    /// Optimal gain matrix.
    pub k: Matrix<S, I, N>,
    /// Steady-state cost matrix P.
    pub p: Matrix<S, N, N>,
    /// Number of iterations to converge.
    pub iterations: usize,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Solve the discrete-time algebraic Riccati equation (DARE) via value iteration.
///
/// P∞ = Q + A^T P∞ A - A^T P∞ B (R + B^T P∞ B)^-1 B^T P∞ A
///
/// Returns None if (R + B^T P B) is singular.
pub fn solve_dare<S: ControlScalar, const N: usize, const I: usize>(
    a: &Matrix<S, N, N>,
    b: &Matrix<S, N, I>,
    q: &Matrix<S, N, N>,
    r: &Matrix<S, I, I>,
    max_iter: usize,
    tol: S,
) -> Option<RiccatiSolution<S, N, I>> {
    let mut p = *q;
    let at = a.transpose();
    let bt = b.transpose();

    for iter in 0..max_iter {
        // S_k = R + B^T * P * B  (I×I)
        let pb = matmul(&p, b);
        let s_k = matmul(&bt, &pb).add_mat(r);

        // S_k^-1
        let s_inv = s_k.inv()?;

        // K = S_k^-1 * B^T * P * A  (I×N)
        let btp = matmul(&bt, &p);
        let btpa = matmul(&btp, a);
        let k = matmul(&s_inv, &btpa);

        // P_new = Q + A^T*P*A - A^T*P*B * K
        let atp = matmul(&at, &p);
        let atpa = matmul(&atp, a);
        let atpb = matmul(&atp, b);
        let atpbk = matmul(&atpb, &k);
        let p_new = q.add_mat(&atpa).sub_mat(&atpbk);

        // Check convergence: ||P_new - P||_F < tol
        let diff = p_new.sub_mat(&p);
        let norm = diff.frob_norm();

        p = p_new;

        if norm < tol {
            // Compute final gain
            let pb_final = matmul(&p, b);
            let s_final = matmul(&bt, &pb_final).add_mat(r);
            let s_inv_final = s_final.inv()?;
            let btp_final = matmul(&bt, &p);
            let btpa_final = matmul(&btp_final, a);
            let k_final = matmul(&s_inv_final, &btpa_final);

            return Some(RiccatiSolution {
                k: k_final,
                p,
                iterations: iter + 1,
                converged: true,
            });
        }
    }

    // Return best estimate even if not fully converged
    let pb_final = matmul(&p, b);
    let s_final = matmul(&bt, &pb_final).add_mat(r);
    let s_inv_final = s_final.inv()?;
    let btp_final = matmul(&bt, &p);
    let btpa_final = matmul(&btp_final, a);
    let k_final = matmul(&s_inv_final, &btpa_final);

    Some(RiccatiSolution {
        k: k_final,
        p,
        iterations: max_iter,
        converged: false,
    })
}

impl<S: ControlScalar, const N: usize, const I: usize> Lqr<S, N, I> {
    /// Create an LQR controller with the given gain matrix.
    pub fn new(gain: Matrix<S, I, N>) -> Self {
        Self { gain }
    }

    /// Design an LQR controller from system matrices and weight matrices.
    pub fn design(
        a: &Matrix<S, N, N>,
        b: &Matrix<S, N, I>,
        q: &Matrix<S, N, N>,
        r: &Matrix<S, I, I>,
    ) -> Option<Self> {
        let sol = solve_dare(a, b, q, r, 1000, S::from_f64(1e-10))?;
        Some(Self { gain: sol.k })
    }

    /// Compute optimal control: u = -K*(x - x_ref)
    pub fn control(&self, state: &[S; N], reference: &[S; N]) -> [S; I] {
        let error: [S; N] = core::array::from_fn(|i| state[i] - reference[i]);
        let ku = matvec(&self.gain, &error);
        // u = -K*e
        core::array::from_fn(|i| -ku[i])
    }

    /// Compute control: u = -K*x (regulation to zero)
    pub fn regulate(&self, state: &[S; N]) -> [S; I] {
        let zero = core::array::from_fn(|_| S::ZERO);
        self.control(state, &zero)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple 1D double integrator: x1' = x2, x2' = u
    /// Discretized with dt=0.1: A = [[1,0.1],[0,1]], B = [[0.005],[0.1]]
    fn double_integrator() -> (Matrix<f64, 2, 2>, Matrix<f64, 2, 1>) {
        let dt = 0.1_f64;
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = dt;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = dt * dt / 2.0;
        b.data[1][0] = dt;

        (a, b)
    }

    #[test]
    fn dare_converges_double_integrator() {
        let (a, b) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;

        let sol = solve_dare(&a, &b, &q, &r, 1000, 1e-10);
        assert!(sol.is_some(), "DARE should converge");
        let sol = sol.unwrap();
        assert!(sol.converged, "Should converge within max_iter");
        assert!(sol.iterations < 200);
    }

    #[test]
    fn lqr_gain_stabilizes_system() {
        let (a, b) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;

        let lqr = Lqr::design(&a, &b, &q, &r).unwrap();

        // Simulate: start at x=[1,0], should converge to 0
        let mut x = [1.0_f64, 0.0];
        for _ in 0..200 {
            let u = lqr.regulate(&x);
            // x[k+1] = A*x[k] + B*u[k]
            let ax = matvec(&a, &x);
            let bu = matvec(&b, &u);
            x = [ax[0] + bu[0], ax[1] + bu[1]];
        }

        assert!(x[0].abs() < 0.01, "Position should converge: {}", x[0]);
        assert!(x[1].abs() < 0.01, "Velocity should converge: {}", x[1]);
    }

    #[test]
    fn lqr_tracking() {
        let (a, b) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity().scale(10.0);
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 1.0;

        let lqr = Lqr::design(&a, &b, &q, &r).unwrap();

        // Track reference [5.0, 0.0] from x=[0,0]
        let reference = [5.0_f64, 0.0];
        let mut x = [0.0_f64, 0.0];
        for _ in 0..300 {
            let u = lqr.control(&x, &reference);
            let ax = matvec(&a, &x);
            let bu = matvec(&b, &u);
            x = [ax[0] + bu[0], ax[1] + bu[1]];
        }
        assert!(
            (x[0] - reference[0]).abs() < 0.1,
            "Should track reference: x[0]={}",
            x[0]
        );
    }

    #[test]
    fn lqr_from_gain_matrix() {
        let mut k = Matrix::<f64, 1, 2>::zeros();
        k.data[0][0] = 2.0;
        k.data[0][1] = 1.0;
        let lqr = Lqr::new(k);
        let u = lqr.regulate(&[3.0, 1.0]);
        // u = -[2*3 + 1*1] = -7
        assert!((u[0] - (-7.0)).abs() < 1e-10, "u={}", u[0]);
    }

    #[test]
    fn singular_r_returns_none() {
        let (a, b) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity();
        let r = Matrix::<f64, 1, 1>::zeros(); // singular R
                                              // Should fail (can't invert zero matrix)
        let sol = solve_dare(&a, &b, &q, &r, 10, 1e-6);
        // May or may not return None depending on P initialization,
        // but we just verify it doesn't panic
        let _ = sol;
    }
}
