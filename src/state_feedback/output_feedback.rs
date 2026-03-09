//! Output feedback control using Luenberger observer + LQR (separation principle).
//!
//! The observer estimates full state from measured outputs, while the LQR
//! gain computes the control input from the estimated state.
#![allow(unused)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Output feedback controller: Luenberger observer + LQR (separation principle).
///
/// The observer update is:
///   x_hat[k+1] = A*x_hat[k] + B*u[k] + L*(y[k] - C*x_hat[k])
///
/// The control law is:
///   u[k] = -K*x_hat[k]
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
/// - M: output dimension
pub struct OutputFeedback<S: ControlScalar, const N: usize, const I: usize, const M: usize> {
    /// LQR gain K (I×N): u = -K*x_hat
    pub k_gain: Matrix<S, I, N>,
    /// Observer (Luenberger) gain L (N×M): correction = L*(y - C*x_hat)
    pub l_gain: Matrix<S, N, M>,
    /// State transition matrix A (N×N).
    pub a: Matrix<S, N, N>,
    /// Input matrix B (N×I).
    pub b: Matrix<S, N, I>,
    /// Output matrix C (M×N).
    pub c: Matrix<S, M, N>,
    /// Current state estimate x_hat (N×1).
    x_hat: Matrix<S, N, 1>,
}

impl<S: ControlScalar, const N: usize, const I: usize, const M: usize> OutputFeedback<S, N, I, M> {
    /// Create a new output feedback controller.
    ///
    /// Gains K and L must be designed externally (e.g., via LQR and pole placement).
    pub fn new(
        k_gain: Matrix<S, I, N>,
        l_gain: Matrix<S, N, M>,
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        c: Matrix<S, M, N>,
    ) -> Self {
        Self {
            k_gain,
            l_gain,
            a,
            b,
            c,
            x_hat: Matrix::zeros(),
        }
    }

    /// Update observer and compute control input.
    ///
    /// Performs one step of the combined observer-controller:
    ///   1. Compute control: u = -K*x_hat
    ///   2. Update observer: x_hat = A*x_hat + B*u_prev + L*(y - C*x_hat)
    ///
    /// Note: `u_prev` is the control applied at the previous time step (used
    /// to propagate the observer model). Pass the output of the previous call.
    pub fn update(
        &mut self,
        y: &Matrix<S, M, 1>,
        u_prev: &Matrix<S, I, 1>,
        _dt: S,
    ) -> Matrix<S, I, 1> {
        // Compute control: u = -K * x_hat
        let kx = matmul(&self.k_gain, &self.x_hat);
        let u = kx.neg();

        // Observer output prediction: y_hat = C * x_hat
        let y_hat = matmul(&self.c, &self.x_hat);

        // Innovation: e = y - y_hat
        let innov = y.sub_mat(&y_hat);

        // Observer correction: L * e
        let l_innov = matmul(&self.l_gain, &innov);

        // Observer prediction: A * x_hat + B * u_prev
        let ax = matmul(&self.a, &self.x_hat);
        let bu = matmul(&self.b, u_prev);

        // Update state estimate
        self.x_hat = ax.add_mat(&bu).add_mat(&l_innov);

        u
    }

    /// Return the current state estimate.
    pub fn state_estimate(&self) -> &Matrix<S, N, 1> {
        &self.x_hat
    }

    /// Reset the state estimate to zero.
    pub fn reset(&mut self) {
        self.x_hat = Matrix::zeros();
    }

    /// Set the initial state estimate.
    pub fn set_state_estimate(&mut self, x_hat: Matrix<S, N, 1>) {
        self.x_hat = x_hat;
    }

    /// Compute observer eigenvalue convergence rate (L2 norm of L*C as proxy).
    pub fn observer_gain_norm(&self) -> S {
        let lc = matmul(&self.l_gain, &self.c);
        lc.frob_norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type SystemMatrices = (
        Matrix<f64, 2, 2>,
        Matrix<f64, 2, 1>,
        Matrix<f64, 1, 2>,
        Matrix<f64, 1, 2>,
        Matrix<f64, 2, 1>,
    );

    /// Build a 2-state, 1-input, 1-output system.
    /// A = [[0.9, 0.1],[0, 0.8]], B = [[0],[1]], C = [[1, 0]]
    fn build_system() -> SystemMatrices {
        let mut a = Matrix::<f64, 2, 2>::zeros();
        a.data[0][0] = 0.9;
        a.data[0][1] = 0.1;
        a.data[1][1] = 0.8;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[1][0] = 1.0;

        let mut c = Matrix::<f64, 1, 2>::zeros();
        c.data[0][0] = 1.0;

        // K: LQR gain (hand-tuned)
        let mut k = Matrix::<f64, 1, 2>::zeros();
        k.data[0][0] = 0.5;
        k.data[0][1] = 0.3;

        // L: observer gain (hand-tuned for fast convergence)
        let mut l = Matrix::<f64, 2, 1>::zeros();
        l.data[0][0] = 0.6;
        l.data[1][0] = 0.3;

        (a, b, c, k, l)
    }

    #[test]
    fn output_feedback_construction() {
        let (a, b, c, k, l) = build_system();
        let ctrl = OutputFeedback::<f64, 2, 1, 1>::new(k, l, a, b, c);
        let est = ctrl.state_estimate();
        assert_eq!(est.data[0][0], 0.0);
        assert_eq!(est.data[1][0], 0.0);
    }

    #[test]
    fn output_feedback_reset() {
        let (a, b, c, k, l) = build_system();
        let mut ctrl = OutputFeedback::<f64, 2, 1, 1>::new(k, l, a, b, c);

        let mut x_init = Matrix::<f64, 2, 1>::zeros();
        x_init.data[0][0] = 5.0;
        ctrl.set_state_estimate(x_init);
        assert!((ctrl.state_estimate().data[0][0] - 5.0).abs() < 1e-12);

        ctrl.reset();
        assert_eq!(ctrl.state_estimate().data[0][0], 0.0);
    }

    #[test]
    fn output_feedback_stabilizes() {
        let (a, b, c, k, l) = build_system();
        let mut ctrl = OutputFeedback::<f64, 2, 1, 1>::new(k, l, a, b, c);

        // True state starts at [1, 0]
        let mut x_true = Matrix::<f64, 2, 1>::zeros();
        x_true.data[0][0] = 1.0;

        let mut u_prev = Matrix::<f64, 1, 1>::zeros();

        for _ in 0..200 {
            // Measure output y = C * x_true
            let y = matmul(&c, &x_true);

            // Controller update
            let u = ctrl.update(&y, &u_prev, 0.0_f64);

            // Plant dynamics: x_true = A*x_true + B*u
            let ax = matmul(&a, &x_true);
            let bu = matmul(&b, &u);
            x_true = ax.add_mat(&bu);

            u_prev = u;
        }

        assert!(
            x_true.data[0][0].abs() < 0.05,
            "State should converge: x[0] = {}",
            x_true.data[0][0]
        );
    }

    #[test]
    fn observer_gain_norm_nonzero() {
        let (a, b, c, k, l) = build_system();
        let ctrl = OutputFeedback::<f64, 2, 1, 1>::new(k, l, a, b, c);
        let norm = ctrl.observer_gain_norm();
        assert!(
            norm > 0.0,
            "Observer gain norm should be positive: {}",
            norm
        );
    }
}
