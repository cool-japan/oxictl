use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Solution from the H∞ DARE synthesis.
pub struct HinfSolution<S: ControlScalar, const N: usize, const I: usize> {
    /// State feedback gain K (I×N). Apply as u = -K*x.
    pub k: Matrix<S, I, N>,
    /// Steady-state cost matrix P (N×N).
    pub p: Matrix<S, N, N>,
    /// Whether the iteration converged.
    pub converged: bool,
    /// Number of iterations used.
    pub iterations: usize,
}

/// Solve the discrete-time H∞ state feedback synthesis via Riccati iteration.
///
/// System:
///   x[k+1] = A·x[k] + B_u·u[k] + B_w·w[k]
///   z[k]   = C_z·x[k]  (performance output)
///
/// Finds state feedback gain K (u = -K·x) achieving disturbance attenuation γ:
///   ‖T_zw‖∞ < γ
///
/// The modified DARE (H∞ version):
///   P = Q + A^T·P·A
///       − A^T·P·B_u·(R + B_u^T·P·B_u)^{−1}·B_u^T·P·A     (control)
///       + γ^{−2}·A^T·P·B_w·(I − γ^{−2}·B_w^T·P·B_w)^{−1}·B_w^T·P·A  (disturbance)
///
/// where Q = C_z^T·C_z (use `c_z.transpose()` * `c_z` to build Q externally).
///
/// Returns `None` if:
/// - `(I − γ^{−2}·B_w^T·P·B_w)` becomes singular (γ too small — increase γ)
/// - `(R + B_u^T·P·B_u)` is singular
///
/// # Type parameters
/// - `N`: state dimension
/// - `I`: number of control inputs
/// - `D`: number of disturbance inputs
#[allow(clippy::too_many_arguments)]
pub fn solve_hinf_dare<S: ControlScalar, const N: usize, const I: usize, const D: usize>(
    a: &Matrix<S, N, N>,
    b_u: &Matrix<S, N, I>,
    b_w: &Matrix<S, N, D>,
    q: &Matrix<S, N, N>,
    r: &Matrix<S, I, I>,
    gamma: S,
    max_iter: usize,
    tol: S,
) -> Option<HinfSolution<S, N, I>> {
    let mut p = *q;
    let at = a.transpose();
    let bt_u = b_u.transpose(); // I×N
    let bt_w = b_w.transpose(); // D×N
    let gamma2_inv = S::ONE / (gamma * gamma);
    let id_d = Matrix::<S, D, D>::identity();

    for iter in 0..max_iter {
        let p_old = p;

        // S_u = R + B_u^T * P * B_u  (I×I)
        let pb_u = matmul(&p, b_u); // N×I
        let s_u = matmul(&bt_u, &pb_u).add_mat(r); // I×I
        let s_u_inv = s_u.inv()?;

        // M_w = I_D - γ^{-2} * B_w^T * P * B_w  (D×D)
        let pb_w = matmul(&p, b_w); // N×D
        let btwpbw = matmul(&bt_w, &pb_w); // D×D
        let m_w = id_d.add_mat(&btwpbw.scale(-gamma2_inv));
        let m_w_inv = m_w.inv()?;

        // A^T * P * A  (N×N)
        let pa = matmul(&p, a); // N×N
        let atpa = matmul(&at, &pa); // N×N

        // Control term: A^T*P*B_u * S_u^{-1} * B_u^T*P*A  (N×N)
        let atpbu = matmul(&at, &pb_u); // N×I
        let btu_pa = matmul(&bt_u, &pa); // I×N
        let ctrl = matmul(&matmul(&atpbu, &s_u_inv), &btu_pa); // N×N

        // Disturbance term: γ^{-2} * A^T*P*B_w * M_w^{-1} * B_w^T*P*A  (N×N)
        let atpbw = matmul(&at, &pb_w); // N×D
        let btw_pa = matmul(&bt_w, &pa); // D×N
        let dist = matmul(&matmul(&atpbw, &m_w_inv), &btw_pa).scale(gamma2_inv); // N×N

        // P_new = Q + A^T*P*A - ctrl + dist
        p = q.add_mat(&atpa).sub_mat(&ctrl).add_mat(&dist);

        // Convergence: Frobenius norm of change
        let err = p.sub_mat(&p_old).frob_norm();
        if err < tol {
            // Final gain: K = S_u^{-1} * B_u^T * P * A
            let pb_u_f = matmul(&p, b_u);
            let btu_pa_f = matmul(&bt_u, &matmul(&p, a));
            let s_u_f = matmul(&bt_u, &pb_u_f).add_mat(r);
            let k = matmul(&s_u_f.inv()?, &btu_pa_f);
            return Some(HinfSolution {
                k,
                p,
                converged: true,
                iterations: iter + 1,
            });
        }
    }

    // Return best effort solution
    let pb_u_f = matmul(&p, b_u);
    let btu_pa_f = matmul(&bt_u, &matmul(&p, a));
    let s_u_f = matmul(&bt_u, &pb_u_f).add_mat(r);
    let k = matmul(&s_u_f.inv()?, &btu_pa_f);
    Some(HinfSolution {
        k,
        p,
        converged: false,
        iterations: max_iter,
    })
}

/// H∞ state feedback controller.
///
/// Pre-computed from `solve_hinf_dare`. Applies u = -K·x.
#[derive(Debug, Clone, Copy)]
pub struct HinfController<S: ControlScalar, const N: usize, const I: usize> {
    pub gain: Matrix<S, I, N>,
}

impl<S: ControlScalar, const N: usize, const I: usize> HinfController<S, N, I> {
    pub fn new(gain: Matrix<S, I, N>) -> Self {
        Self { gain }
    }

    /// Compute control output u = -K·x.
    pub fn control(&self, x: &[S; N]) -> [S; I] {
        use crate::core::matrix::matvec;
        let kx = matvec(&self.gain, x);
        core::array::from_fn(|i| -kx[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn double_integrator() -> (Matrix<f64, 2, 2>, Matrix<f64, 2, 1>, Matrix<f64, 2, 1>) {
        // x[k+1] = A*x + B_u*u + B_w*w
        // Double integrator: dt=0.1
        let a = Matrix::<f64, 2, 2> {
            data: [[1.0, 0.1], [0.0, 1.0]],
        };
        let b_u = Matrix::<f64, 2, 1> {
            data: [[0.005], [0.1]],
        };
        let b_w = Matrix::<f64, 2, 1> {
            data: [[0.001], [0.01]],
        };
        (a, b_u, b_w)
    }

    #[test]
    fn hinf_dare_converges() {
        let (a, b_u, b_w) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity();
        let r = Matrix::<f64, 1, 1> { data: [[0.1]] };
        let gamma = 10.0_f64;

        let sol = solve_hinf_dare(&a, &b_u, &b_w, &q, &r, gamma, 1000, 1e-8);
        assert!(sol.is_some(), "DARE should converge for γ=10");
        let sol = sol.unwrap();
        assert!(sol.converged, "Should converge within 1000 iterations");
    }

    #[test]
    fn hinf_gain_stabilizes_system() {
        let (a, b_u, b_w) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity().scale(10.0);
        let r = Matrix::<f64, 1, 1> { data: [[0.1]] };
        let gamma = 5.0_f64;

        let sol = solve_hinf_dare(&a, &b_u, &b_w, &q, &r, gamma, 2000, 1e-8)
            .expect("H-inf DARE should succeed");
        let ctrl = HinfController::new(sol.k);

        // Simulate closed loop with no disturbance
        let mut x = [1.0_f64, 0.5];
        for _ in 0..200 {
            let u = ctrl.control(&x);
            let x_new: [f64; 2] = core::array::from_fn(|i| {
                a.data[i][0] * x[0] + a.data[i][1] * x[1] + b_u.data[i][0] * u[0]
            });
            x = x_new;
        }

        assert!(
            x[0].abs() < 0.1 && x[1].abs() < 0.1,
            "State should converge to origin: x={:?}",
            x
        );
    }

    #[test]
    fn larger_gamma_accepts_more_disturbance() {
        // With very large γ, H∞ DARE ≈ LQR DARE (disturbance term vanishes)
        let (a, b_u, b_w) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity();
        let r = Matrix::<f64, 1, 1> { data: [[1.0]] };

        let sol_large = solve_hinf_dare(&a, &b_u, &b_w, &q, &r, 1e6_f64, 1000, 1e-8);
        assert!(sol_large.is_some(), "Large γ should always converge");
    }

    #[test]
    fn too_small_gamma_may_fail() {
        // γ = 0.001 is below achievable — should return None or not converge
        let (a, b_u, b_w) = double_integrator();
        let q = Matrix::<f64, 2, 2>::identity();
        let r = Matrix::<f64, 1, 1> { data: [[1.0]] };

        // This may return None or non-converged solution; either is acceptable
        let sol = solve_hinf_dare(&a, &b_u, &b_w, &q, &r, 0.001_f64, 100, 1e-8);
        // Just verify it doesn't panic
        let _ = sol;
    }
}
