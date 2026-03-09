//! Moving Horizon Estimator (MHE) combining state estimation with MPC-like optimization.
//!
//! MHE estimates the state of a (possibly nonlinear) system by solving a
//! constrained optimisation problem over a sliding window of past measurements.
//! Unlike the Kalman filter, MHE can explicitly handle constraints on states
//! and noise, and naturally supports nonlinear dynamics via linearisation.
//!
//! The MHE problem over a window of length W:
//!
//!   min_{x_0,...,x_W, w_0,...,w_{W-1}} Σ_k [ ||y_k - h(x_k)||_R^2 + ||w_k||_Q^2 ]
//!                                           + ||x_0 - x̄_0||_P_0^2  (arrival cost)
//!   s.t.  x_{k+1} = f(x_k) + w_k
//!
//! where the arrival cost approximates the cost of past data not in the window.
//!
//! The Gauss-Newton method is used for iterative refinement of the estimate.
#![allow(unused, clippy::needless_range_loop)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Error type for MHE operations.
#[derive(Debug)]
pub enum MheError {
    /// The window is too small to perform estimation.
    WindowTooSmall,
    /// Linearisation produced a degenerate Jacobian.
    DegenerateJacobian,
    /// The measurement dimension is incompatible.
    DimensionMismatch,
}

impl core::fmt::Display for MheError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MheError::WindowTooSmall => write!(f, "MHE window too small"),
            MheError::DegenerateJacobian => write!(f, "Degenerate Jacobian in MHE linearisation"),
            MheError::DimensionMismatch => write!(f, "Dimension mismatch in MHE"),
        }
    }
}

/// Nonlinear dynamics function: x_{k+1} = f(x_k, u_k).
pub type DynamicsFn<S, const N: usize, const I: usize> =
    fn(&Matrix<S, N, 1>, &Matrix<S, I, 1>) -> Matrix<S, N, 1>;

/// Nonlinear measurement function: y_k = h(x_k).
pub type MeasurementFn<S, const N: usize, const M: usize> = fn(&Matrix<S, N, 1>) -> Matrix<S, M, 1>;

/// Sliding window of past measurements for the MHE.
///
/// Stores the W most recent (measurement, input) pairs.
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
/// - M: measurement dimension
/// - W: window length (number of past time steps)
pub struct MheWindow<
    S: ControlScalar,
    const N: usize,
    const I: usize,
    const M: usize,
    const W: usize,
> {
    /// Past measurements y_k (M×1 each), oldest first.
    pub measurements: [Matrix<S, M, 1>; W],
    /// Past inputs u_k (I×1 each), oldest first.
    pub inputs: [Matrix<S, I, 1>; W],
    /// Number of valid entries currently stored (≤ W).
    pub count: usize,
}

impl<S: ControlScalar, const N: usize, const I: usize, const M: usize, const W: usize>
    MheWindow<S, N, I, M, W>
{
    /// Create an empty measurement window.
    pub fn new() -> Self {
        Self {
            measurements: [Matrix::zeros(); W],
            inputs: [Matrix::zeros(); W],
            count: 0,
        }
    }

    /// Push a new (measurement, input) pair into the sliding window.
    ///
    /// Older entries are shifted out when the window is full.
    pub fn push(&mut self, y: Matrix<S, M, 1>, u: Matrix<S, I, 1>) {
        if self.count < W {
            self.measurements[self.count] = y;
            self.inputs[self.count] = u;
            self.count += 1;
        } else {
            // Shift left (drop oldest)
            for k in 0..(W - 1) {
                self.measurements[k] = self.measurements[k + 1];
                self.inputs[k] = self.inputs[k + 1];
            }
            self.measurements[W - 1] = y;
            self.inputs[W - 1] = u;
        }
    }

    /// Returns the number of valid entries in the window.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the window is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl<S: ControlScalar, const N: usize, const I: usize, const M: usize, const W: usize> Default
    for MheWindow<S, N, I, M, W>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Moving Horizon Estimator.
///
/// Estimates the state by minimising a weighted sum of:
/// - Measurement residuals over the window
/// - Process noise penalty
/// - Arrival cost (penalises deviation from prior estimate x̄_0)
///
/// Uses iterative Gauss-Newton with numerical Jacobians.
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
/// - M: measurement dimension
/// - W: window length
pub struct MovingHorizonEstimator<
    S: ControlScalar,
    const N: usize,
    const I: usize,
    const M: usize,
    const W: usize,
> {
    /// Nonlinear dynamics f(x, u).
    pub dynamics: DynamicsFn<S, N, I>,
    /// Nonlinear measurement function h(x).
    pub measurement: MeasurementFn<S, N, M>,
    /// Process noise covariance weight Q^{-1} (N×N diagonal approximation stored as vector).
    pub q_inv_diag: Matrix<S, N, 1>,
    /// Measurement noise covariance weight R^{-1} (M×M diagonal approximation stored as vector).
    pub r_inv_diag: Matrix<S, M, 1>,
    /// Arrival cost weight P_0^{-1} (N×N diagonal approximation stored as vector).
    pub p0_inv_diag: Matrix<S, N, 1>,
    /// Prior state estimate (arrival cost centre) x̄_0.
    pub x_prior: Matrix<S, N, 1>,
    /// Current best state estimate at the end of the window.
    pub x_estimate: Matrix<S, N, 1>,
    /// Sliding measurement window.
    pub window: MheWindow<S, N, I, M, W>,
    /// Number of Gauss-Newton iterations per solve call.
    pub iterations: usize,
    /// Finite difference step for numerical Jacobians.
    pub fd_eps: S,
    /// Gauss-Newton step size (learning rate).
    pub step_size: S,
}

impl<S: ControlScalar, const N: usize, const I: usize, const M: usize, const W: usize>
    MovingHorizonEstimator<S, N, I, M, W>
{
    /// Create a new Moving Horizon Estimator.
    pub fn new(
        dynamics: DynamicsFn<S, N, I>,
        measurement: MeasurementFn<S, N, M>,
        q_inv_diag: Matrix<S, N, 1>,
        r_inv_diag: Matrix<S, M, 1>,
        p0_inv_diag: Matrix<S, N, 1>,
        iterations: usize,
    ) -> Self {
        Self {
            dynamics,
            measurement,
            q_inv_diag,
            r_inv_diag,
            p0_inv_diag,
            x_prior: Matrix::zeros(),
            x_estimate: Matrix::zeros(),
            window: MheWindow::new(),
            iterations,
            fd_eps: S::from_f64(1e-5),
            step_size: S::from_f64(0.1),
        }
    }

    /// Push a new measurement and input into the sliding window.
    pub fn push_measurement(&mut self, y: Matrix<S, M, 1>, u: Matrix<S, I, 1>) {
        self.window.push(y, u);
    }

    /// Roll out the dynamics from x0 over the window using stored inputs.
    ///
    /// Returns an array of predicted states [x_1, ..., x_W].
    fn rollout(&self, x0: &Matrix<S, N, 1>, n: usize) -> [Matrix<S, N, 1>; W] {
        let mut states = [Matrix::<S, N, 1>::zeros(); W];
        let mut x = *x0;
        for k in 0..n {
            let u = self.window.inputs[k];
            let x_next = (self.dynamics)(&x, &u);
            states[k] = x_next;
            x = x_next;
        }
        states
    }

    /// Compute the total MHE cost for a given initial state x0.
    ///
    /// J = ||x0 - x_prior||_{P0_inv}^2
    ///   + Σ_k [ ||y_k - h(x_k)||_{R_inv}^2 + ||w_k||_{Q_inv}^2 ]
    ///
    /// where w_k = x_{k+1} - f(x_k, u_k) is the process noise.
    pub fn cost(&self, x0: &Matrix<S, N, 1>) -> S {
        let n = self.window.len().min(W);
        if n == 0 {
            return S::ZERO;
        }

        // Arrival cost: (x0 - x_prior)^T P0_inv (x0 - x_prior)
        let mut arrival = S::ZERO;
        for i in 0..N {
            let diff = x0.data[i][0] - self.x_prior.data[i][0];
            arrival += self.p0_inv_diag.data[i][0] * diff * diff;
        }

        // Roll out states
        let states = self.rollout(x0, n);

        let mut meas_cost = S::ZERO;
        let mut proc_cost = S::ZERO;

        // Measurement residuals (states[k] corresponds to x_{k+1} from x0)
        // For k=0..n-1: measurement at window[k] is y_k, prediction is h(x_k)
        // We use x0 for measurement at k=0 in the window
        // y_0: measurement of x0
        {
            let y0 = self.window.measurements[0];
            let h0 = (self.measurement)(x0);
            for j in 0..M {
                let r = y0.data[j][0] - h0.data[j][0];
                meas_cost += self.r_inv_diag.data[j][0] * r * r;
            }
        }

        // Measurements for predicted states
        for k in 0..(n - 1) {
            let y = self.window.measurements[k + 1];
            let h = (self.measurement)(&states[k]);
            for j in 0..M {
                let r = y.data[j][0] - h.data[j][0];
                meas_cost += self.r_inv_diag.data[j][0] * r * r;
            }
        }

        // Process noise: w_k = x_{k+1} - f(x_k, u_k)
        // For k=0: w_0 = states[0] - f(x0, u_0)
        {
            let u0 = self.window.inputs[0];
            let fx = (self.dynamics)(x0, &u0);
            for i in 0..N {
                let w = states[0].data[i][0] - fx.data[i][0];
                proc_cost += self.q_inv_diag.data[i][0] * w * w;
            }
        }

        // For k=1..n-1: w_k = states[k] - f(states[k-1], u_k)
        for k in 1..(n - 1) {
            let u = self.window.inputs[k];
            let fx = (self.dynamics)(&states[k - 1], &u);
            for i in 0..N {
                let w = states[k].data[i][0] - fx.data[i][0];
                proc_cost += self.q_inv_diag.data[i][0] * w * w;
            }
        }

        arrival + meas_cost + proc_cost
    }

    /// Numerical gradient of MHE cost w.r.t. x0 (central differences).
    fn gradient_x0(&self, x0: &Matrix<S, N, 1>) -> Matrix<S, N, 1> {
        let eps = self.fd_eps;
        let two_eps = S::TWO * eps;
        let mut grad = Matrix::<S, N, 1>::zeros();
        for i in 0..N {
            let mut xp = *x0;
            let mut xm = *x0;
            xp.data[i][0] += eps;
            xm.data[i][0] -= eps;
            grad.data[i][0] = (self.cost(&xp) - self.cost(&xm)) / two_eps;
        }
        grad
    }

    /// Solve the MHE problem using gradient descent (Gauss-Newton approximation).
    ///
    /// Returns the updated state estimate at the end of the window, or an error
    /// if the window is empty.
    pub fn solve(&mut self) -> Result<Matrix<S, N, 1>, MheError> {
        let n = self.window.len();
        if n == 0 {
            return Err(MheError::WindowTooSmall);
        }

        // Optimise x0 (the initial state of the window) via gradient descent
        let mut x0 = self.x_prior;

        for _iter in 0..self.iterations {
            let grad = self.gradient_x0(&x0);
            for i in 0..N {
                x0.data[i][0] -= self.step_size * grad.data[i][0];
            }
        }

        // Roll out to obtain final state estimate
        let states = self.rollout(&x0, n.min(W));
        let x_final = if n > 0 {
            states[(n - 1).min(W - 1)]
        } else {
            x0
        };

        self.x_estimate = x_final;

        // Update arrival cost centre: shift prior to x0 (Gauss-Newton update)
        self.x_prior = x0;

        Ok(x_final)
    }

    /// Return the current state estimate.
    pub fn estimate(&self) -> Matrix<S, N, 1> {
        self.x_estimate
    }

    /// Reset the estimator to a new initial state.
    pub fn reset(&mut self, x0: Matrix<S, N, 1>) {
        self.x_estimate = x0;
        self.x_prior = x0;
        self.window = MheWindow::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Linear dynamics: x_{k+1} = A x_k + B u_k (identity A, zero B for simplicity)
    fn linear_dynamics(x: &Matrix<f64, 2, 1>, _u: &Matrix<f64, 1, 1>) -> Matrix<f64, 2, 1> {
        let mut xn = Matrix::<f64, 2, 1>::zeros();
        xn.data[0][0] = x.data[0][0];
        xn.data[1][0] = x.data[1][0];
        xn
    }

    /// Identity measurement: y = x (first two components)
    fn identity_measurement(x: &Matrix<f64, 2, 1>) -> Matrix<f64, 2, 1> {
        *x
    }

    fn make_mhe() -> MovingHorizonEstimator<f64, 2, 1, 2, 4> {
        let q_inv = {
            let mut m = Matrix::<f64, 2, 1>::zeros();
            m.data[0][0] = 1.0;
            m.data[1][0] = 1.0;
            m
        };
        let r_inv = {
            let mut m = Matrix::<f64, 2, 1>::zeros();
            m.data[0][0] = 10.0;
            m.data[1][0] = 10.0;
            m
        };
        let p0_inv = {
            let mut m = Matrix::<f64, 2, 1>::zeros();
            m.data[0][0] = 1.0;
            m.data[1][0] = 1.0;
            m
        };
        MovingHorizonEstimator::new(
            linear_dynamics,
            identity_measurement,
            q_inv,
            r_inv,
            p0_inv,
            50,
        )
    }

    #[test]
    fn window_push_fills_correctly() {
        let mut w = MheWindow::<f64, 2, 1, 2, 4>::new();
        assert_eq!(w.len(), 0);
        let y = Matrix::<f64, 2, 1>::zeros();
        let u = Matrix::<f64, 1, 1>::zeros();
        w.push(y, u);
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn window_push_slides_when_full() {
        let mut w = MheWindow::<f64, 2, 1, 2, 3>::new();
        for k in 0..5u32 {
            let mut y = Matrix::<f64, 2, 1>::zeros();
            y.data[0][0] = k as f64;
            w.push(y, Matrix::zeros());
        }
        // Window should hold last 3 values: 2, 3, 4
        assert_eq!(w.len(), 3);
        assert!((w.measurements[0].data[0][0] - 2.0).abs() < 1e-12);
        assert!((w.measurements[2].data[0][0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn mhe_empty_window_returns_error() {
        let mut mhe = make_mhe();
        let result = mhe.solve();
        assert!(matches!(result, Err(MheError::WindowTooSmall)));
    }

    #[test]
    fn mhe_cost_zero_at_prior_with_consistent_measurements() {
        let mut mhe = make_mhe();
        // Place prior at (1, 0)
        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 1.0;
        mhe.x_prior = x0;

        // Measurement consistent with prior (identity measurement)
        let y = x0;
        let u = Matrix::<f64, 1, 1>::zeros();
        mhe.window.push(y, u);

        // Cost should be 0 at the correct state
        let c = mhe.cost(&x0);
        assert!(
            c < 1e-10,
            "Cost at consistent state should be near zero, got {}",
            c
        );
    }

    #[test]
    fn mhe_solve_with_one_measurement() {
        let mut mhe = make_mhe();

        // Measurement at (2, 0)
        let mut y = Matrix::<f64, 2, 1>::zeros();
        y.data[0][0] = 2.0;
        mhe.push_measurement(y, Matrix::zeros());

        let result = mhe.solve();
        assert!(
            result.is_ok(),
            "MHE solve should succeed with one measurement"
        );
    }

    #[test]
    fn mhe_reset_clears_window() {
        let mut mhe = make_mhe();
        mhe.push_measurement(Matrix::zeros(), Matrix::zeros());
        assert_eq!(mhe.window.len(), 1);

        let x_new = Matrix::<f64, 2, 1>::zeros();
        mhe.reset(x_new);
        assert_eq!(mhe.window.len(), 0);
    }
}
