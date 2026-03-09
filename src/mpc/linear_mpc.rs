use crate::core::matrix::{matvec, Matrix};
use crate::core::scalar::ControlScalar;

/// Status returned by the MPC solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MpcStatus {
    /// Optimal solution found.
    Optimal,
    /// Reached maximum iterations without full convergence.
    MaxIter,
    /// Constraints are infeasible or system matrix is ill-conditioned.
    Infeasible,
}

/// Input/output constraints for MPC.
///
/// - `I`: number of inputs
#[derive(Debug, Clone, Copy)]
pub struct MpcConstraints<S: ControlScalar, const I: usize> {
    /// Minimum input.
    pub u_min: [S; I],
    /// Maximum input.
    pub u_max: [S; I],
    /// Maximum input rate of change per step.
    pub du_max: [S; I],
}

impl<S: ControlScalar, const I: usize> MpcConstraints<S, I> {
    /// Unconstrained (very large bounds).
    pub fn unconstrained() -> Self {
        let big = S::from_f64(1e9);
        Self {
            u_min: [-big; I],
            u_max: [big; I],
            du_max: [big; I],
        }
    }

    /// Box constraints on input only.
    pub fn box_input(u_min: [S; I], u_max: [S; I]) -> Self {
        Self {
            u_min,
            u_max,
            du_max: [S::from_f64(1e9); I],
        }
    }
}

/// Linear Model Predictive Controller.
///
/// Solves the finite-horizon constrained optimal control problem:
///   min Σ_{k=0}^{H-1} [ x^T Q x + u^T R u ] + x_H^T Q x_H
///   s.t.  x_{k+1} = A x_k + B u_k
///         u_min ≤ u_k ≤ u_max
///         |Δu_k| ≤ du_max
///
/// Uses gradient projection (projected gradient descent) to handle
/// input constraints without a full QP solver.
///
/// - `N`: state dimension
/// - `I`: input dimension
/// - `H`: prediction horizon (const, no alloc)
pub struct LinearMpc<S: ControlScalar, const N: usize, const I: usize, const H: usize> {
    /// State transition matrix A (N×N).
    pub a: Matrix<S, N, N>,
    /// Input matrix B (N×I).
    pub b: Matrix<S, N, I>,
    /// State cost matrix Q (N×N).
    pub q: Matrix<S, N, N>,
    /// Terminal state cost Q_f (N×N).
    pub q_f: Matrix<S, N, N>,
    /// Input cost matrix R (I×I).
    pub r: Matrix<S, I, I>,
    /// Constraints.
    pub constraints: MpcConstraints<S, I>,
    /// Maximum gradient projection iterations.
    pub max_iter: usize,
    /// Gradient step size (learning rate).
    pub step_size: S,
    /// Convergence tolerance.
    pub tol: S,
    /// Last solved control sequence.
    u_seq: [[S; I]; H],
    /// Last applied control (for rate constraint at k=0).
    last_u: [S; I],
}

impl<S: ControlScalar, const N: usize, const I: usize, const H: usize> LinearMpc<S, N, I, H> {
    /// Create a new Linear MPC.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        q: Matrix<S, N, N>,
        q_f: Matrix<S, N, N>,
        r: Matrix<S, I, I>,
        constraints: MpcConstraints<S, I>,
    ) -> Self {
        Self {
            a,
            b,
            q,
            q_f,
            r,
            constraints,
            max_iter: 100,
            step_size: S::from_f64(0.01),
            tol: S::from_f64(1e-6),
            u_seq: [[S::ZERO; I]; H],
            last_u: [S::ZERO; I],
        }
    }

    /// Set optimizer parameters.
    pub fn with_optimizer(mut self, step_size: S, max_iter: usize, tol: S) -> Self {
        self.step_size = step_size;
        self.max_iter = max_iter;
        self.tol = tol;
        self
    }

    /// Solve the MPC problem for current state `x0`.
    ///
    /// Returns `(u_optimal, status)` where `u_optimal` is the first
    /// control input to apply.
    pub fn solve(&mut self, x0: &[S; N], reference: &[S; N]) -> ([S; I], MpcStatus) {
        // Warm-start from previous solution (shift by one, zero-pad last)
        let mut u_seq = self.shift_warm_start();
        let last_u = self.last_u;

        let status = self.projected_gradient(&mut u_seq, x0, reference, &last_u);

        self.last_u = u_seq[0];
        self.u_seq = u_seq;
        (u_seq[0], status)
    }

    /// Shift control sequence for warm-starting.
    fn shift_warm_start(&self) -> [[S; I]; H] {
        let mut u = [[S::ZERO; I]; H];
        let end = H.saturating_sub(1);
        u[..end].copy_from_slice(&self.u_seq[1..end + 1]);
        u[end] = self.u_seq[end];
        u
    }

    /// Projected gradient descent on the input sequence.
    fn projected_gradient(
        &self,
        u_seq: &mut [[S; I]; H],
        x0: &[S; N],
        reference: &[S; N],
        last_u: &[S; I],
    ) -> MpcStatus {
        let mut cost_prev = S::from_f64(1e30);

        for iter in 0..self.max_iter {
            // Compute gradient via forward rollout + backpropagation
            let grad = self.compute_gradient(u_seq, x0, reference);

            // Gradient step
            let mut u_new = [[S::ZERO; I]; H];
            for k in 0..H {
                for j in 0..I {
                    let u_unconstrained = u_seq[k][j] - self.step_size * grad[k][j];
                    // Project onto box constraints
                    u_new[k][j] = u_unconstrained
                        .clamp_val(self.constraints.u_min[j], self.constraints.u_max[j]);
                }
                // Rate constraint (|Δu| ≤ du_max)
                let u_prev: [S; I] = if k == 0 { *last_u } else { u_new[k - 1] };
                for (j, (&up, &du_lim)) in u_prev
                    .iter()
                    .zip(self.constraints.du_max.iter())
                    .enumerate()
                {
                    let du = u_new[k][j] - up;
                    if du.abs() > du_lim {
                        let dir = if du > S::ZERO { S::ONE } else { -S::ONE };
                        u_new[k][j] = up + dir * du_lim;
                    }
                }
            }

            *u_seq = u_new;

            // Check convergence
            let cost = self.compute_cost(u_seq, x0, reference);
            let improvement = (cost_prev - cost).abs();
            cost_prev = cost;

            if iter > 0 && improvement < self.tol {
                return MpcStatus::Optimal;
            }
        }

        MpcStatus::MaxIter
    }

    /// Compute total cost J = Σ(x^T Q x + u^T R u) + x_H^T Q_f x_H.
    fn compute_cost(&self, u_seq: &[[S; I]; H], x0: &[S; N], reference: &[S; N]) -> S {
        let mut x = *x0;
        let mut cost = S::ZERO;

        for uk in u_seq.iter() {
            // Error w.r.t. reference
            let e: [S; N] = core::array::from_fn(|i| x[i] - reference[i]);
            let xe_qe = self.quad_form_n(&self.q, &e);
            let ur = matvec(&self.r, uk);
            let u_ru: S = uk
                .iter()
                .zip(ur.iter())
                .map(|(&u, &ru)| u * ru)
                .fold(S::ZERO, |a, b| a + b);
            cost += xe_qe + u_ru;
            // Propagate
            let ax = matvec(&self.a, &x);
            let bu = matvec(&self.b, uk);
            x = core::array::from_fn(|i| ax[i] + bu[i]);
        }

        // Terminal cost
        let e: [S; N] = core::array::from_fn(|i| x[i] - reference[i]);
        cost += self.quad_form_n(&self.q_f, &e);
        cost
    }

    /// Compute gradient of cost w.r.t. u_seq via backpropagation.
    fn compute_gradient(
        &self,
        u_seq: &[[S; I]; H],
        x0: &[S; N],
        reference: &[S; N],
    ) -> [[S; I]; H] {
        // Forward pass: collect states x_0..x_{H-1} and terminal state x_H separately
        // (avoids H+1 as const generic expression which is unstable)
        let mut states = [[S::ZERO; N]; H];
        states[0] = *x0;
        for k in 0..H.saturating_sub(1) {
            let ax = matvec(&self.a, &states[k]);
            let bu = matvec(&self.b, &u_seq[k]);
            states[k + 1] = core::array::from_fn(|i| ax[i] + bu[i]);
        }
        // Compute x_H = A*x_{H-1} + B*u_{H-1}
        let ax_last = matvec(&self.a, &states[H.saturating_sub(1)]);
        let bu_last = matvec(&self.b, &u_seq[H.saturating_sub(1)]);
        let x_terminal: [S; N] = core::array::from_fn(|i| ax_last[i] + bu_last[i]);

        // Backward pass: costate (lambda) propagation
        // Terminal gradient: λ_H = 2*Q_f*(x_H - ref)
        let e_h: [S; N] = core::array::from_fn(|i| x_terminal[i] - reference[i]);
        let q_f_e = matvec(&self.q_f, &e_h);
        let mut lambda: [S; N] = core::array::from_fn(|i| S::TWO * q_f_e[i]);

        let mut grad = [[S::ZERO; I]; H];
        let at = self.a.transpose();
        let bt = self.b.transpose();

        for k in (0..H).rev() {
            // grad_u[k] = 2*R*u[k] + B^T * lambda
            let r_u = matvec(&self.r, &u_seq[k]);
            let bt_lambda = matvec(&bt, &lambda);
            grad[k] = core::array::from_fn(|j| S::TWO * r_u[j] + bt_lambda[j]);

            // Update lambda: λ_{k} = 2*Q*(x_k - ref) + A^T * λ_{k+1}
            let e_k: [S; N] = core::array::from_fn(|i| states[k][i] - reference[i]);
            let q_e = matvec(&self.q, &e_k);
            let at_lambda = matvec(&at, &lambda);
            lambda = core::array::from_fn(|i| S::TWO * q_e[i] + at_lambda[i]);
        }

        grad
    }

    /// Quadratic form x^T M x.
    fn quad_form_n(&self, m: &Matrix<S, N, N>, x: &[S; N]) -> S {
        let mx = matvec(m, x);
        x.iter()
            .zip(mx.iter())
            .map(|(&xi, &mxi)| xi * mxi)
            .fold(S::ZERO, |a, b| a + b)
    }

    /// Return the last solved control sequence.
    pub fn control_sequence(&self) -> &[[S; I]; H] {
        &self.u_seq
    }

    pub fn reset(&mut self) {
        self.u_seq = [[S::ZERO; I]; H];
    }
}

/// Compute prediction matrices for unconstrained MPC (for offline analysis).
///
/// Returns (Phi, Gamma) where:
///   X_pred = Phi * x0 + Gamma * U
///
/// Phi: (H*N) × N  — not constructed here (would require alloc)
/// Instead, inline rollout is used in the solver above.
#[allow(dead_code)]
fn build_prediction_unused<S: ControlScalar, const N: usize, const I: usize>(
    a: &Matrix<S, N, N>,
    _b: &Matrix<S, N, I>,
) -> Matrix<S, N, N> {
    // A^1
    *a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_integrator_mpc() -> LinearMpc<f64, 2, 1, 10> {
        // Double integrator: x[k+1] = A*x + B*u
        // x = [position, velocity], u = acceleration
        let a = Matrix::<f64, 2, 2> {
            data: [[1.0, 0.1], [0.0, 1.0]],
        };
        let b = Matrix::<f64, 2, 1> {
            data: [[0.005], [0.1]],
        };
        let q = Matrix::<f64, 2, 2>::identity().scale(1.0);
        let r = Matrix::<f64, 1, 1> { data: [[0.01]] };
        let constraints = MpcConstraints::box_input([-10.0], [10.0]);

        LinearMpc::new(a, b, q, q.scale(2.0), r, constraints).with_optimizer(0.001, 200, 1e-8)
    }

    #[test]
    fn solves_double_integrator() {
        let mut mpc = build_integrator_mpc();
        let x0 = [1.0_f64, 0.0]; // position=1, velocity=0
        let reference = [0.0_f64, 0.0]; // drive to origin

        let (u, status) = mpc.solve(&x0, &reference);
        // u should be negative to push toward reference
        assert!(u[0] <= 0.0, "u={:.4} should drive toward ref", u[0]);
        assert!(status == MpcStatus::Optimal || status == MpcStatus::MaxIter);
    }

    #[test]
    fn respects_input_constraints() {
        let mut mpc = build_integrator_mpc();
        let x0 = [100.0_f64, 0.0]; // far from origin
        let reference = [0.0_f64, 0.0];
        let (u, _) = mpc.solve(&x0, &reference);
        assert!(u[0] >= -10.0 - 1e-10, "u={}", u[0]);
        assert!(u[0] <= 10.0 + 1e-10, "u={}", u[0]);
    }

    #[test]
    fn closed_loop_converges() {
        let mut mpc = build_integrator_mpc();
        let mut x = [2.0_f64, 0.0];
        let reference = [0.0_f64, 0.0];
        let a = [[1.0_f64, 0.1], [0.0, 1.0]];
        let b = [0.005_f64, 0.1];

        for _ in 0..100 {
            let (u, _) = mpc.solve(&x, &reference);
            // Propagate plant
            x[0] = a[0][0] * x[0] + a[0][1] * x[1] + b[0] * u[0];
            x[1] = a[1][0] * x[0] + a[1][1] * x[1] + b[1] * u[0];
        }
        assert!(
            (x[0]).abs() < 1.0,
            "Should converge toward origin: x={:.3}",
            x[0]
        );
    }

    #[test]
    fn unconstrained_status() {
        let mut mpc: LinearMpc<f64, 1, 1, 5> = {
            let a = Matrix::<f64, 1, 1> { data: [[0.9]] };
            let b = Matrix::<f64, 1, 1> { data: [[1.0]] };
            let q = Matrix::<f64, 1, 1> { data: [[1.0]] };
            let r = Matrix::<f64, 1, 1> { data: [[0.1]] };
            LinearMpc::new(a, b, q, q, r, MpcConstraints::unconstrained())
                .with_optimizer(0.01, 500, 1e-10)
        };
        let (_, status) = mpc.solve(&[1.0], &[0.0]);
        assert!(status == MpcStatus::Optimal || status == MpcStatus::MaxIter);
    }
}
