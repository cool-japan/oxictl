//! Tracking MPC with offset-free disturbance rejection.
//!
//! Augments the plant model with an integrating disturbance channel to achieve
//! zero steady-state error even when the nominal model does not match the plant.
#![allow(unused)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Offset-free tracking MPC with integrating disturbance model.
///
/// The augmented model is:
///   x[k+1]  = A*x[k] + B*u[k] + Bd*d[k]   (d enters through B column direction)
///   d[k+1]  = d[k]                           (random-walk disturbance)
///   y[k]    = C*x[k] + d[k]
///
/// The controller uses gradient-descent optimisation over the H-step horizon
/// with a tracking cost that penalises deviation from the reference.
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
/// - H: prediction horizon (compile-time constant)
pub struct TrackingMpc<S: ControlScalar, const N: usize, const I: usize, const H: usize> {
    /// State transition matrix A (N×N).
    pub a: Matrix<S, N, N>,
    /// Input matrix B (N×I).
    pub b: Matrix<S, N, I>,
    /// Output matrix C (1×N).
    pub c: Matrix<S, 1, N>,
    /// State cost weight Q (N×N).
    pub q: Matrix<S, N, N>,
    /// Input cost weight R (I×I).
    pub r: Matrix<S, I, I>,
    /// Prediction horizon (≤ H).
    pub horizon: usize,
    /// Current state estimate.
    pub x: Matrix<S, N, 1>,
    /// Estimated output disturbance.
    pub d_hat: S,
    /// Scalar reference (setpoint).
    pub reference: S,
    /// Previous control input.
    u_prev: Matrix<S, I, 1>,
}

impl<S: ControlScalar, const N: usize, const I: usize, const H: usize> TrackingMpc<S, N, I, H> {
    /// Create a new TrackingMpc.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        c: Matrix<S, 1, N>,
        q: Matrix<S, N, N>,
        r: Matrix<S, I, I>,
    ) -> Self {
        Self {
            a,
            b,
            c,
            q,
            r,
            horizon: H,
            x: Matrix::zeros(),
            d_hat: S::ZERO,
            reference: S::ZERO,
            u_prev: Matrix::zeros(),
        }
    }

    /// Set the reference (setpoint).
    pub fn set_reference(&mut self, r: S) {
        self.reference = r;
    }

    /// Set the current state.
    pub fn set_state(&mut self, x: Matrix<S, N, 1>) {
        self.x = x;
    }

    /// Compute stage cost for a predicted state and input.
    ///
    /// Stage cost = (y - ref)^2 * q_scalar + u^T R u
    /// where y = C*x + d_hat.
    fn stage_cost(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>) -> S {
        // Output prediction
        let y_mat = matmul(&self.c, x);
        let y = y_mat.data[0][0] + self.d_hat;
        let e = y - self.reference;

        // Output cost: e^T Q_out e — use trace(Q) as scalar weight
        let q_scalar = self.q.trace();
        let mut cost = e * e * q_scalar;

        // Input cost: u^T R u
        let ru = matmul(&self.r, u);
        let ut = u.transpose();
        let utu = matmul(&ut, &ru);
        cost += utu.data[0][0];

        cost
    }

    /// Compute gradient of cost w.r.t. u at step k (numerical, central differences).
    fn grad_u(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>, eps: S) -> Matrix<S, I, 1> {
        let mut grad = Matrix::<S, I, 1>::zeros();
        for i in 0..I {
            let mut u_plus = *u;
            let mut u_minus = *u;
            u_plus.data[i][0] += eps;
            u_minus.data[i][0] -= eps;
            let c_plus = self.stage_cost(x, &u_plus);
            let c_minus = self.stage_cost(x, &u_minus);
            grad.data[i][0] = (c_plus - c_minus) / (S::TWO * eps);
        }
        grad
    }

    /// Predict state H steps ahead under a constant input sequence.
    fn predict(&self, u_seq: &[Matrix<S, I, 1>; H]) -> [Matrix<S, N, 1>; H] {
        let mut xs: [Matrix<S, N, 1>; H] = [Matrix::zeros(); H];
        let mut x = self.x;
        for k in 0..H {
            let ax = matmul(&self.a, &x);
            let bu = matmul(&self.b, &u_seq[k]);
            x = ax.add_mat(&bu);
            xs[k] = x;
        }
        xs
    }

    /// Compute total cost for a given input sequence.
    pub fn total_cost(&self, u_seq: &[Matrix<S, I, 1>; H]) -> S {
        let mut total = S::ZERO;
        let mut x = self.x;
        for u_k in u_seq.iter().take(H.min(self.horizon)) {
            let ax = matmul(&self.a, &x);
            let bu = matmul(&self.b, u_k);
            x = ax.add_mat(&bu);
            total += self.stage_cost(&x, u_k);
        }
        total
    }

    /// Compute the optimal control input via projected gradient descent.
    ///
    /// Optimises over the H-step horizon and returns the first control action.
    /// Gradients are computed via central differences on the full trajectory cost
    /// so that the effect of each u_k on all subsequent predicted states is
    /// captured correctly.
    pub fn step(&mut self, _dt: S) -> Matrix<S, I, 1> {
        let step_size = S::from_f64(0.01);
        let eps = S::from_f64(1e-4);
        let max_iter = 100_usize;
        let horizon = self.horizon.min(H);

        // Warm-start: constant u = u_prev
        let mut u_seq: [Matrix<S, I, 1>; H] = [self.u_prev; H];

        for _iter in 0..max_iter {
            // Gradient descent: perturb each u_k and measure effect on total cost
            for k in 0..horizon {
                for i in 0..I {
                    let mut u_plus = u_seq;
                    let mut u_minus = u_seq;
                    u_plus[k].data[i][0] += eps;
                    u_minus[k].data[i][0] -= eps;
                    let c_plus = self.total_cost(&u_plus);
                    let c_minus = self.total_cost(&u_minus);
                    let grad = (c_plus - c_minus) / (S::TWO * eps);
                    u_seq[k].data[i][0] -= step_size * grad;
                }
            }
        }

        let u_opt = u_seq[0];
        self.u_prev = u_opt;
        u_opt
    }

    /// Update disturbance estimate using output measurement.
    ///
    /// d_hat += alpha * (y_meas - C*x_hat - d_hat)
    /// Simple first-order disturbance observer.
    pub fn update_disturbance(&mut self, y_meas: S) {
        let alpha = S::from_f64(0.1); // observer gain
        let y_pred = matmul(&self.c, &self.x).data[0][0] + self.d_hat;
        let innov = y_meas - y_pred;
        self.d_hat += alpha * innov;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Mat1x1 = Matrix<f64, 1, 1>;

    fn integrator_system() -> (Mat1x1, Mat1x1, Mat1x1, Mat1x1, Mat1x1) {
        let mut a = Matrix::<f64, 1, 1>::zeros();
        a.data[0][0] = 1.0; // pure integrator

        let mut b = Matrix::<f64, 1, 1>::zeros();
        b.data[0][0] = 1.0;

        let mut c = Matrix::<f64, 1, 1>::zeros();
        c.data[0][0] = 1.0;

        let mut q = Matrix::<f64, 1, 1>::zeros();
        q.data[0][0] = 1.0;

        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;

        (a, b, c, q, r)
    }

    #[test]
    fn tracking_mpc_construction() {
        let (a, b, c, q, r) = integrator_system();
        let mpc = TrackingMpc::<f64, 1, 1, 5>::new(a, b, c, q, r);
        assert_eq!(mpc.reference, 0.0);
        assert_eq!(mpc.d_hat, 0.0);
    }

    #[test]
    fn tracking_mpc_setpoint() {
        let (a, b, c, q, r) = integrator_system();
        let mut mpc = TrackingMpc::<f64, 1, 1, 5>::new(a, b, c, q, r);
        mpc.set_reference(2.0);
        assert_eq!(mpc.reference, 2.0);
    }

    #[test]
    fn tracking_mpc_step_returns_nonzero_for_nonzero_ref() {
        let (a, b, c, q, r) = integrator_system();
        let mut mpc = TrackingMpc::<f64, 1, 1, 5>::new(a, b, c, q, r);
        mpc.set_reference(1.0);
        let u = mpc.step(0.1_f64);
        // Should apply positive control to drive state toward reference
        assert!(
            u.data[0][0].abs() > 1e-6,
            "Control should be nonzero: {}",
            u.data[0][0]
        );
    }

    #[test]
    fn disturbance_update_tracks_offset() {
        let (a, b, c, q, r) = integrator_system();
        let mut mpc = TrackingMpc::<f64, 1, 1, 5>::new(a, b, c, q, r);

        // True output has a constant disturbance of 0.5
        for _ in 0..100 {
            mpc.update_disturbance(0.5_f64); // y_meas = 0 (state) + 0.5 (disturbance)
        }

        assert!(
            (mpc.d_hat - 0.5).abs() < 0.05,
            "Disturbance estimate should converge: d_hat={}",
            mpc.d_hat
        );
    }
}
