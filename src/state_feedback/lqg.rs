use crate::core::matrix::{matmul, matvec, Matrix};
use crate::core::scalar::ControlScalar;
use crate::state_feedback::lqr::{solve_dare, Lqr};

/// Linear Quadratic Gaussian (LQG) controller.
///
/// Combines an LQR state-feedback controller with a Kalman filter observer.
/// Optimal for linear systems with Gaussian process and measurement noise.
///
/// Architecture:
///   Observer: x̂[k+1] = A*x̂[k] + B*u[k] + L*(y[k] - C*x̂[k])
///   Control:  u[k] = -K*x̂[k]
///
/// - N: state dimension
/// - M: measurement dimension
/// - I: input dimension
pub struct Lqg<S: ControlScalar, const N: usize, const M: usize, const I: usize> {
    /// LQR gain matrix K (I×N).
    pub k: Matrix<S, I, N>,
    /// Kalman observer gain L (N×M).
    pub l: Matrix<S, N, M>,
    /// State transition matrix.
    a: Matrix<S, N, N>,
    /// Input matrix.
    b: Matrix<S, N, I>,
    /// Output matrix.
    c: Matrix<S, M, N>,
    /// State estimate.
    x_hat: [S; N],
}

impl<S: ControlScalar, const N: usize, const M: usize, const I: usize> Lqg<S, N, M, I> {
    /// Construct LQG from system matrices and pre-designed gains.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        c: Matrix<S, M, N>,
        k: Matrix<S, I, N>,
        l: Matrix<S, N, M>,
    ) -> Self {
        Self {
            k,
            l,
            a,
            b,
            c,
            x_hat: [S::ZERO; N],
        }
    }

    /// Design LQG: solve DARE for both LQR (control) and Kalman (observer).
    ///
    /// - `q_lqr`, `r_lqr`: state and input cost matrices for LQR
    /// - `q_kf`: process noise covariance (Q_w)
    /// - `r_kf`: measurement noise covariance (R_v)
    pub fn design(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        c: Matrix<S, M, N>,
        q_lqr: &Matrix<S, N, N>,
        r_lqr: &Matrix<S, I, I>,
        q_kf: &Matrix<S, N, N>,
        r_kf: &Matrix<S, M, M>,
    ) -> Option<Self> {
        // LQR: solve DARE for control gain K
        let lqr = Lqr::design(&a, &b, q_lqr, r_lqr)?;

        // Kalman observer: DARE on dual system (A^T, C^T, Q_kf, R_kf)
        let at = a.transpose();
        let ct = c.transpose(); // ct is C^T: N×M
        let kf_sol = solve_dare(&at, &ct, q_kf, r_kf, 1000, S::from_f64(1e-8))?;
        // Kalman gain L = P_inf * C^T * R^{-1}
        let r_inv = r_kf.inv()?;
        let p_ct = matmul(&kf_sol.p, &ct); // P * C^T  (N×N × N×M = N×M)
        let l = matmul(&p_ct, &r_inv); // N×M

        Some(Self::new(a, b, c, lqr.gain, l))
    }

    /// Update: correct observer with measurement, compute control output.
    ///
    /// Returns control input u = -K*x̂ (after observer update).
    pub fn update(&mut self, y: &[S; M], reference: &[S; N]) -> [S; I] {
        // Observer update: x̂_new = A*x̂ + B*u_prev + L*(y - C*x̂)
        // Note: we use the state-feedback control from previous estimate
        let u_prev = self.control_from_estimate(reference);

        let y_hat = matvec(&self.c, &self.x_hat);
        let innov: [S; M] = core::array::from_fn(|i| y[i] - y_hat[i]);

        let ax = matvec(&self.a, &self.x_hat);
        let bu = matvec(&self.b, &u_prev);
        let le = matvec(&self.l, &innov);

        self.x_hat = core::array::from_fn(|i| ax[i] + bu[i] + le[i]);

        // Compute control from updated estimate
        self.control_from_estimate(reference)
    }

    fn control_from_estimate(&self, reference: &[S; N]) -> [S; I] {
        let error: [S; N] = core::array::from_fn(|i| self.x_hat[i] - reference[i]);
        let ku = matvec(&self.k, &error);
        core::array::from_fn(|i| -ku[i])
    }

    pub fn state_estimate(&self) -> &[S; N] {
        &self.x_hat
    }

    pub fn reset(&mut self) {
        self.x_hat = [S::ZERO; N];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lqg_basic_construction() {
        // Simple integrator: x[k+1] = x[k] + u, y = x
        let a = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let b = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let c = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let k = Matrix::<f64, 1, 1> { data: [[0.5]] };
        let l = Matrix::<f64, 1, 1> { data: [[0.7]] };
        let lqg = Lqg::new(a, b, c, k, l);
        assert_eq!(lqg.state_estimate()[0], 0.0);
    }

    #[test]
    fn lqg_design_works() {
        let a = Matrix::<f64, 2, 2> {
            data: [[1.0, 0.01], [0.0, 1.0]],
        };
        let b = Matrix::<f64, 2, 1> {
            data: [[0.0], [0.01]],
        };
        let c = Matrix::<f64, 1, 2> { data: [[1.0, 0.0]] };
        let q_lqr = Matrix::<f64, 2, 2>::identity().scale(1.0);
        let r_lqr = Matrix::<f64, 1, 1> { data: [[0.1]] };
        let q_kf = Matrix::<f64, 2, 2>::identity().scale(0.01);
        let r_kf = Matrix::<f64, 1, 1> { data: [[0.1]] };
        let lqg = Lqg::design(a, b, c, &q_lqr, &r_lqr, &q_kf, &r_kf);
        assert!(lqg.is_some(), "LQG design should succeed");
    }

    #[test]
    fn lqg_stabilizes_integrator() {
        // Discrete integrator + observer
        let a = Matrix::<f64, 1, 1> { data: [[0.9]] }; // stable open loop
        let b = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let c = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let k = Matrix::<f64, 1, 1> { data: [[0.5]] }; // control gain
        let l = Matrix::<f64, 1, 1> { data: [[0.8]] }; // observer gain
        let mut lqg = Lqg::new(a, b, c, k, l);

        let mut x_true = 5.0_f64;
        for _ in 0..200 {
            let u = lqg.update(&[x_true], &[0.0]);
            x_true = 0.9 * x_true + u[0];
        }
        assert!(x_true.abs() < 0.5, "Should converge: x={:.4}", x_true);
    }
}
