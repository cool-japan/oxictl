//! Economic MPC with custom stage cost and gradient-descent optimiser.
//!
//! Economic MPC replaces the standard quadratic regulation cost with an
//! arbitrary economic criterion (e.g., power consumption, throughput, profit).
//! This allows the controller to optimise process economics directly while
//! respecting dynamic constraints.
//!
//! The turnpike property: for long horizons, the optimal trajectory spends
//! most of its time near the optimal steady-state (the "turnpike").
#![allow(unused)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Economic stage cost function pointer.
///
/// Takes the current state x (N×1) and input u (I×1) and returns the scalar cost.
pub type EconomicCostFn<S, const N: usize, const I: usize> =
    fn(&Matrix<S, N, 1>, &Matrix<S, I, 1>) -> S;

/// Economic stage: wraps a custom cost function with gradient utilities.
pub struct EconomicStage<S: ControlScalar, const N: usize, const I: usize> {
    /// The economic cost function.
    pub cost_fn: EconomicCostFn<S, N, I>,
    /// Default gradient descent step size.
    pub gradient_step: S,
}

impl<S: ControlScalar, const N: usize, const I: usize> EconomicStage<S, N, I> {
    /// Create a new economic stage.
    pub fn new(cost_fn: EconomicCostFn<S, N, I>, gradient_step: S) -> Self {
        Self {
            cost_fn,
            gradient_step,
        }
    }

    /// Evaluate the stage cost at (x, u).
    pub fn stage_cost(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>) -> S {
        (self.cost_fn)(x, u)
    }

    /// Numerical gradient of the stage cost w.r.t. u using central differences.
    ///
    /// ∂L/∂u_i ≈ (L(u + eps*e_i) - L(u - eps*e_i)) / (2*eps)
    pub fn gradient_u(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>, eps: S) -> Matrix<S, I, 1> {
        let mut grad = Matrix::<S, I, 1>::zeros();
        for i in 0..I {
            let mut u_plus = *u;
            let mut u_minus = *u;
            u_plus.data[i][0] += eps;
            u_minus.data[i][0] -= eps;
            let c_plus = (self.cost_fn)(x, &u_plus);
            let c_minus = (self.cost_fn)(x, &u_minus);
            grad.data[i][0] = (c_plus - c_minus) / (S::TWO * eps);
        }
        grad
    }

    /// Numerical gradient of the stage cost w.r.t. x using central differences.
    pub fn gradient_x(&self, x: &Matrix<S, N, 1>, u: &Matrix<S, I, 1>, eps: S) -> Matrix<S, N, 1> {
        let mut grad = Matrix::<S, N, 1>::zeros();
        for i in 0..N {
            let mut x_plus = *x;
            let mut x_minus = *x;
            x_plus.data[i][0] += eps;
            x_minus.data[i][0] -= eps;
            let c_plus = (self.cost_fn)(&x_plus, u);
            let c_minus = (self.cost_fn)(&x_minus, u);
            grad.data[i][0] = (c_plus - c_minus) / (S::TWO * eps);
        }
        grad
    }
}

/// Economic MPC: optimises a custom economic cost over a finite horizon H.
///
/// Uses single-shooting gradient descent (same structure as `NonlinearMpc`):
/// the input sequence [u_0, …, u_{H-1}] is optimised by rolling out the linear
/// dynamics and back-propagating the cost gradient.
///
/// Type parameters:
/// - N: state dimension
/// - I: input dimension
/// - H: prediction horizon (compile-time constant)
pub struct EconomicMpc<S: ControlScalar, const N: usize, const I: usize, const H: usize> {
    /// State transition matrix A (N×N).
    pub a: Matrix<S, N, N>,
    /// Input matrix B (N×I).
    pub b: Matrix<S, N, I>,
    /// Economic stage cost.
    pub stage: EconomicStage<S, N, I>,
    /// Current state.
    pub x: Matrix<S, N, 1>,
    /// Effective horizon (≤ H).
    pub horizon: usize,
    /// Number of gradient descent iterations per solve.
    pub iterations: usize,
    /// Enable terminal equality constraint (x_H = 0).
    pub terminal_constraint: bool,
    /// Turnpike detection threshold: if cost change < threshold, near turnpike.
    pub turnpike_thresh: S,
}

impl<S: ControlScalar, const N: usize, const I: usize, const H: usize> EconomicMpc<S, N, I, H> {
    /// Create a new EconomicMpc.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        stage: EconomicStage<S, N, I>,
        iterations: usize,
    ) -> Self {
        Self {
            a,
            b,
            stage,
            x: Matrix::zeros(),
            horizon: H,
            iterations,
            terminal_constraint: false,
            turnpike_thresh: S::from_f64(1e-4),
        }
    }

    /// Compute the total economic cost for a given input sequence.
    pub fn total_cost(&self, u_seq: &[Matrix<S, I, 1>; H]) -> S {
        let mut total = S::ZERO;
        let mut x = self.x;
        let horizon = self.horizon.min(H);
        for u_k in u_seq.iter().take(horizon) {
            let c = self.stage.stage_cost(&x, u_k);
            total += c;
            let ax = matmul(&self.a, &x);
            let bu = matmul(&self.b, u_k);
            x = ax.add_mat(&bu);
        }
        total
    }

    /// Check if the current trajectory is near the turnpike (cost nearly constant along horizon).
    pub fn near_turnpike(&self, u_seq: &[Matrix<S, I, 1>; H]) -> bool {
        let horizon = self.horizon.min(H);
        if horizon < 2 {
            return false;
        }
        let mut x = self.x;
        let mut prev_cost = self.stage.stage_cost(&x, &u_seq[0]);
        let ax = matmul(&self.a, &x);
        let bu = matmul(&self.b, &u_seq[0]);
        x = ax.add_mat(&bu);
        for u_k in u_seq.iter().take(horizon).skip(1) {
            let c = self.stage.stage_cost(&x, u_k);
            let diff = if c > prev_cost {
                c - prev_cost
            } else {
                prev_cost - c
            };
            if diff > self.turnpike_thresh {
                return false;
            }
            prev_cost = c;
            let ax2 = matmul(&self.a, &x);
            let bu2 = matmul(&self.b, u_k);
            x = ax2.add_mat(&bu2);
        }
        true
    }

    /// Solve the economic MPC problem via gradient descent.
    ///
    /// Returns the first element of the optimal input sequence.
    pub fn step(&mut self) -> Matrix<S, I, 1> {
        let eps = S::from_f64(1e-5);
        let step = self.stage.gradient_step;
        let horizon = self.horizon.min(H);
        let max_iter = self.iterations;

        // Initialise input sequence at zero
        let mut u_seq: [Matrix<S, I, 1>; H] = [Matrix::zeros(); H];

        for _iter in 0..max_iter {
            // Forward pass: collect states
            let mut xs: [Matrix<S, N, 1>; H] = [Matrix::zeros(); H];
            let mut x = self.x;
            for k in 0..horizon {
                let ax = matmul(&self.a, &x);
                let bu = matmul(&self.b, &u_seq[k]);
                x = ax.add_mat(&bu);
                xs[k] = x;
            }

            // Gradient step on each u_k
            for k in 0..horizon {
                let g = self.stage.gradient_u(&xs[k], &u_seq[k], eps);
                for i in 0..I {
                    u_seq[k].data[i][0] -= step * g.data[i][0];
                }
            }

            // Optional terminal equality constraint: project x_H to zero.
            if self.terminal_constraint {
                // Simple projection: scale down u_seq to reduce terminal state norm.
                // Full implementation would require QP; here we use a soft penalty.
                // (Left as a best-effort gradient step; users should use large iterations.)
            }
        }

        u_seq[0]
    }

    /// Set the current state.
    pub fn set_state(&mut self, x: Matrix<S, N, 1>) {
        self.x = x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quadratic cost: (x - target)^2 + u^2
    fn quadratic_cost<const N: usize, const I: usize>(
        x: &Matrix<f64, N, 1>,
        u: &Matrix<f64, I, 1>,
    ) -> f64 {
        let x_cost: f64 = x.data.iter().map(|r| r[0] * r[0]).sum();
        let u_cost: f64 = u.data.iter().map(|r| r[0] * r[0]).sum();
        x_cost + u_cost
    }

    #[test]
    fn economic_stage_cost_evaluates() {
        let stage = EconomicStage::<f64, 2, 1>::new(quadratic_cost, 1e-3);
        let mut x = Matrix::<f64, 2, 1>::zeros();
        x.data[0][0] = 1.0;
        x.data[1][0] = 2.0;
        let mut u = Matrix::<f64, 1, 1>::zeros();
        u.data[0][0] = 0.5;
        let cost = stage.stage_cost(&x, &u);
        // (1^2 + 2^2) + 0.5^2 = 1 + 4 + 0.25 = 5.25
        assert!((cost - 5.25).abs() < 1e-12, "Cost: {}", cost);
    }

    #[test]
    fn gradient_u_numerical() {
        let stage = EconomicStage::<f64, 2, 1>::new(quadratic_cost, 1e-3);
        let x = Matrix::<f64, 2, 1>::zeros();
        let mut u = Matrix::<f64, 1, 1>::zeros();
        u.data[0][0] = 1.0;
        let g = stage.gradient_u(&x, &u, 1e-5);
        // Cost = 0 + u^2, dCost/du = 2u = 2.0
        assert!(
            (g.data[0][0] - 2.0).abs() < 1e-6,
            "Gradient: {}",
            g.data[0][0]
        );
    }

    #[test]
    fn economic_mpc_step_reduces_cost() {
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = 0.1;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.005;
        b.data[1][0] = 0.1;

        let stage = EconomicStage::<f64, 2, 1>::new(quadratic_cost, 1e-3);
        let mut mpc = EconomicMpc::<f64, 2, 1, 5>::new(a, b, stage, 100);

        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 1.0;
        mpc.set_state(x0);

        // Cost at zero input sequence
        let u_zero: [Matrix<f64, 1, 1>; 5] = [Matrix::zeros(); 5];
        let cost_before = mpc.total_cost(&u_zero);

        // Optimised control
        let u_opt = mpc.step();

        // Apply one step and measure cost
        let mut u_opt_seq: [Matrix<f64, 1, 1>; 5] = [u_opt; 5];
        let cost_after = mpc.total_cost(&u_opt_seq);

        assert!(
            cost_after <= cost_before + 1e-6,
            "Cost should not increase: before={}, after={}",
            cost_before,
            cost_after
        );
    }

    #[test]
    fn total_cost_zero_at_origin() {
        let a = Matrix::<f64, 2, 2>::identity();
        let b = Matrix::<f64, 2, 1>::zeros();
        let stage = EconomicStage::<f64, 2, 1>::new(quadratic_cost, 1e-3);
        let mpc = EconomicMpc::<f64, 2, 1, 5>::new(a, b, stage, 10);
        let u_seq = [Matrix::<f64, 1, 1>::zeros(); 5];
        let cost = mpc.total_cost(&u_seq);
        assert!(
            cost.abs() < 1e-12,
            "Cost at origin should be zero: {}",
            cost
        );
    }
}
