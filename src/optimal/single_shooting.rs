//! Direct single shooting method for nonlinear optimal control.
//!
//! Parameterises the control input `u(t)` as a piecewise-constant sequence
//! `{u_0, u_1, …, u_{M-1}}`, each applied over one shooting interval `dt`.
//! The ODE is integrated forward once per candidate control sequence, and the
//! resulting cost is
//!
//! ```text
//! J = Σ_{k=0}^{M-1} stage_cost(x_k, u_k) + terminal_cost(x_M)
//! ```
//!
//! Gradient-based optimisation (gradient descent with Armijo backtracking
//! line search) is applied in the space of `M·I` control parameters.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
use crate::core::scalar::ControlScalar;

use super::{
    ode_solver::{OdeSolver, RungeKutta4},
    OptimalError,
};

// ──────────────────────────────────── problem definition ─────────────────────

/// Box constraints on the control input.
#[derive(Debug, Clone, Copy)]
pub struct ControlConstraints<S: ControlScalar, const I: usize> {
    /// Element-wise lower bound.
    pub u_min: [S; I],
    /// Element-wise upper bound.
    pub u_max: [S; I],
}

impl<S: ControlScalar, const I: usize> ControlConstraints<S, I> {
    /// Create unconstrained (very large bounds).
    pub fn unconstrained() -> Self {
        let big = S::from_f64(1e9);
        Self {
            u_min: [-big; I],
            u_max: [big; I],
        }
    }

    /// Box constraints on each input channel.
    pub fn box_input(u_min: [S; I], u_max: [S; I]) -> Self {
        Self { u_min, u_max }
    }

    /// Project a control vector onto the feasible box.
    #[inline]
    pub fn project(&self, u: &[S; I]) -> [S; I] {
        core::array::from_fn(|j| u[j].clamp_val(self.u_min[j], self.u_max[j]))
    }
}

// ─────────────────────────────────────── single-shooting problem ──────────────

/// Optimal control problem transcribed by direct single shooting.
///
/// # Type parameters
/// - `S`: scalar type
/// - `N`: state dimension
/// - `I`: control dimension
/// - `M`: number of shooting intervals (horizon length)
pub struct SingleShootingProblem<S, const N: usize, const I: usize, const M: usize>
where
    S: ControlScalar,
{
    /// Continuous-time vector field `f(x, u) = ẋ`.
    /// The ODE solver integrates `ẋ = dynamics(x, u)` with `u` held constant
    /// over each interval.
    pub dynamics: fn(&[S; N], &[S; I]) -> [S; N],

    /// Running (stage) cost `l(x_k, u_k)` evaluated at the start of each
    /// interval *before* integration.
    pub stage_cost: fn(&[S; N], &[S; I]) -> S,

    /// Terminal cost `φ(x_M)` evaluated at the final state.
    pub terminal_cost: fn(&[S; N]) -> S,

    /// Step size for ODE integration within each shooting interval.
    /// The interval duration equals `dt * ode_steps`.
    pub dt: S,

    /// Number of ODE sub-steps per shooting interval.
    pub ode_steps: usize,

    /// Input box constraints.
    pub constraints: ControlConstraints<S, I>,

    /// Finite-difference step size for gradient computation.
    pub fd_eps: S,

    /// Maximum gradient-descent iterations.
    pub max_iter: usize,

    /// Armijo line-search parameter c₁ ∈ (0,1).
    pub armijo_c1: S,

    /// Line-search step-size decrease factor β ∈ (0,1).
    pub armijo_beta: S,

    /// Initial gradient step size (learning rate).
    pub step_size: S,

    /// Convergence tolerance on ‖∇J‖∞.
    pub tol: S,
}

impl<S, const N: usize, const I: usize, const M: usize> SingleShootingProblem<S, N, I, M>
where
    S: ControlScalar,
{
    /// Create a problem with sensible defaults.
    pub fn new(
        dynamics: fn(&[S; N], &[S; I]) -> [S; N],
        stage_cost: fn(&[S; N], &[S; I]) -> S,
        terminal_cost: fn(&[S; N]) -> S,
        dt: S,
        constraints: ControlConstraints<S, I>,
    ) -> Self {
        Self {
            dynamics,
            stage_cost,
            terminal_cost,
            dt,
            ode_steps: 4,
            constraints,
            fd_eps: S::from_f64(1e-6),
            max_iter: 200,
            armijo_c1: S::from_f64(1e-4),
            armijo_beta: S::from_f64(0.5),
            step_size: S::from_f64(1e-3),
            tol: S::from_f64(1e-6),
        }
    }

    // ── forward simulation ────────────────────────────────────────────────────

    /// Simulate the state trajectory given initial state and control sequence.
    ///
    /// Returns the sequence of states `[x_0, x_1, …, x_M]`.
    fn simulate(&self, x0: &[S; N], u_seq: &[[S; I]; M]) -> Result<[[S; N]; M], OptimalError> {
        // We store x_1 … x_M (M states after the initial one).
        // x_0 is x0 itself.
        let solver = RungeKutta4::<S, N>::new();
        let mut states = [[S::ZERO; N]; M];
        let mut x = *x0;

        let sub_dt = self.dt / S::from_f64(self.ode_steps as f64);

        for k in 0..M {
            // Integrate from x with constant u_seq[k]
            let u = &u_seq[k];
            let dyn_fn = self.dynamics;
            let f = |xv: &[S; N], _t: S| -> [S; N] { dyn_fn(xv, u) };

            for _ in 0..self.ode_steps {
                x = solver.step(f, &x, S::ZERO, sub_dt);
            }
            states[k] = x;
        }

        Ok(states)
    }

    // ── cost evaluation ───────────────────────────────────────────────────────

    /// Evaluate total cost `J` for a given `x0` and control sequence.
    pub fn cost(&self, x0: &[S; N], u_seq: &[[S; I]; M]) -> Result<S, OptimalError> {
        let solver = RungeKutta4::<S, N>::new();
        let mut x = *x0;
        let mut j = S::ZERO;

        let sub_dt = self.dt / S::from_f64(self.ode_steps as f64);

        for k in 0..M {
            j += (self.stage_cost)(&x, &u_seq[k]);

            let u = &u_seq[k];
            let dyn_fn = self.dynamics;
            let f = |xv: &[S; N], _t: S| -> [S; N] { dyn_fn(xv, u) };

            for _ in 0..self.ode_steps {
                x = solver.step(f, &x, S::ZERO, sub_dt);
            }
        }
        j += (self.terminal_cost)(&x);
        Ok(j)
    }

    // ── finite-difference gradient ────────────────────────────────────────────

    /// Compute ∂J/∂u via central finite differences.
    ///
    /// Returns a `[[S; I]; M]` gradient array in the same layout as `u_seq`.
    fn gradient(&self, x0: &[S; N], u_seq: &[[S; I]; M]) -> Result<[[S; I]; M], OptimalError> {
        let eps = self.fd_eps;
        let two_eps = eps + eps;
        let mut grad = [[S::ZERO; I]; M];

        let j0 = self.cost(x0, u_seq)?;
        let _ = j0; // We use forward differences; central is used below

        for k in 0..M {
            for j in 0..I {
                // Forward perturbation
                let mut u_plus = *u_seq;
                u_plus[k][j] += eps;
                let j_plus = self.cost(x0, &u_plus)?;

                // Backward perturbation
                let mut u_minus = *u_seq;
                u_minus[k][j] -= eps;
                let j_minus = self.cost(x0, &u_minus)?;

                grad[k][j] = (j_plus - j_minus) / two_eps;
            }
        }

        Ok(grad)
    }

    // ── Armijo backtracking line search ───────────────────────────────────────

    /// Perform an Armijo backtracking line search along `-grad` direction.
    ///
    /// Returns `(u_new, alpha)` satisfying the sufficient decrease condition,
    /// or the minimum step if the search fails.
    fn line_search(
        &self,
        x0: &[S; N],
        u_seq: &[[S; I]; M],
        grad: &[[S; I]; M],
        j_curr: S,
    ) -> Result<([[S; I]; M], S), OptimalError> {
        let mut alpha = self.step_size;
        let min_alpha = S::from_f64(1e-15);

        // Directional derivative ∇J · d = -‖∇J‖² (steepest descent)
        let mut dir_deriv = S::ZERO;
        for k in 0..M {
            for j in 0..I {
                dir_deriv += grad[k][j] * grad[k][j];
            }
        }
        let dir_deriv = -dir_deriv; // negative because descent direction is -grad

        loop {
            // Candidate: u_new = project(u - alpha * grad)
            let u_candidate: [[S; I]; M] = core::array::from_fn(|k| {
                self.constraints
                    .project(&core::array::from_fn(|j| u_seq[k][j] - alpha * grad[k][j]))
            });

            let j_new = self.cost(x0, &u_candidate)?;

            // Armijo condition: J(u_new) ≤ J(u) + c1 * alpha * ∇J·d
            if j_new <= j_curr + self.armijo_c1 * alpha * dir_deriv {
                return Ok((u_candidate, alpha));
            }

            alpha *= self.armijo_beta;
            if alpha < min_alpha {
                // Give up — use the current candidate anyway
                let u_fallback: [[S; I]; M] = core::array::from_fn(|k| {
                    self.constraints
                        .project(&core::array::from_fn(|j| u_seq[k][j] - alpha * grad[k][j]))
                });
                return Ok((u_fallback, alpha));
            }
        }
    }

    // ── main solver ───────────────────────────────────────────────────────────

    /// Solve the single-shooting optimal control problem.
    ///
    /// # Parameters
    /// - `x0`     — initial state
    /// - `u_init` — initial control sequence (warm start)
    ///
    /// # Returns
    /// `(u_optimal, cost_optimal)` — the optimal control sequence and the
    /// corresponding total cost.
    pub fn solve(
        &self,
        x0: &[S; N],
        u_init: &[[S; I]; M],
    ) -> Result<([[S; I]; M], S), OptimalError> {
        // Project the initial guess onto feasible region
        let mut u_seq: [[S; I]; M] = core::array::from_fn(|k| self.constraints.project(&u_init[k]));

        for _iter in 0..self.max_iter {
            let j_curr = self.cost(x0, &u_seq)?;
            let grad = self.gradient(x0, &u_seq)?;

            // Convergence check on ‖∇J‖∞
            let mut grad_norm = S::ZERO;
            for k in 0..M {
                for j in 0..I {
                    let g = grad[k][j].abs();
                    if g > grad_norm {
                        grad_norm = g;
                    }
                }
            }
            if grad_norm < self.tol {
                let j_final = self.cost(x0, &u_seq)?;
                return Ok((u_seq, j_final));
            }

            let (u_new, _alpha) = self.line_search(x0, &u_seq, &grad, j_curr)?;
            u_seq = u_new;
        }

        // Return best solution found
        let j_final = self.cost(x0, &u_seq)?;
        Ok((u_seq, j_final))
    }

    /// Compute the state trajectory for the optimal solution.
    ///
    /// Returns `[x_1, …, x_M]` (states *after* each shooting interval).
    pub fn trajectory(
        &self,
        x0: &[S; N],
        u_seq: &[[S; I]; M],
    ) -> Result<[[S; N]; M], OptimalError> {
        self.simulate(x0, u_seq)
    }
}

// ──────────────────────────────────────────────── tests ───────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── minimum-energy control of the double integrator ───────────────────────
    //
    // Continuous-time double integrator:
    //   ẋ₁ = x₂
    //   ẋ₂ = u
    //
    // Cost: J = Σ u_k² * dt  (minimum energy)
    // Task: drive [1, 0] → near [0, 0] in M=20 steps.

    fn double_integrator_dynamics(x: &[f64; 2], u: &[f64; 1]) -> [f64; 2] {
        [x[1], u[0]]
    }

    fn min_energy_stage(x: &[f64; 2], u: &[f64; 1]) -> f64 {
        let _ = x;
        u[0] * u[0]
    }

    fn quadratic_terminal(x: &[f64; 2]) -> f64 {
        10.0 * (x[0] * x[0] + x[1] * x[1])
    }

    fn build_problem() -> SingleShootingProblem<f64, 2, 1, 20> {
        let constraints = ControlConstraints::box_input([-5.0], [5.0]);
        let mut prob = SingleShootingProblem::new(
            double_integrator_dynamics,
            min_energy_stage,
            quadratic_terminal,
            0.1,
            constraints,
        );
        prob.max_iter = 500;
        prob.step_size = 0.05;
        prob.tol = 1e-5;
        prob.ode_steps = 2;
        prob
    }

    #[test]
    fn single_shooting_double_integrator_reduces_cost() {
        let prob = build_problem();
        let x0 = [1.0_f64, 0.0];
        let u_init = [[0.0_f64]; 20];

        let j_init = prob.cost(&x0, &u_init).expect("cost should not fail");
        let (u_opt, j_opt) = prob.solve(&x0, &u_init).expect("solve should not fail");

        // The optimised cost must be strictly less than the trivial cost
        assert!(
            j_opt < j_init,
            "j_opt={:.4} should be < j_init={:.4}",
            j_opt,
            j_init
        );

        // The optimal control must respect bounds
        for k in 0..20 {
            assert!(u_opt[k][0] >= -5.0 - 1e-10);
            assert!(u_opt[k][0] <= 5.0 + 1e-10);
        }
    }

    #[test]
    fn single_shooting_trajectory_reduces_state_norm() {
        let prob = build_problem();
        let x0 = [1.0_f64, 0.0];
        let u_init = [[0.0_f64]; 20];
        let (u_opt, _) = prob.solve(&x0, &u_init).expect("solve should not fail");

        let traj = prob
            .trajectory(&x0, &u_opt)
            .expect("trajectory should not fail");
        let final_state = traj[19];
        let norm_sq = final_state[0] * final_state[0] + final_state[1] * final_state[1];

        // Starting norm_sq = 1.0; the controller should reduce it significantly
        assert!(
            norm_sq < 0.5,
            "Final state norm² = {:.4} should be < 0.5",
            norm_sq
        );
    }

    #[test]
    fn control_constraints_project() {
        let c = ControlConstraints::box_input([-1.0_f64], [1.0]);
        let u = [5.0_f64];
        let p = c.project(&u);
        assert!((p[0] - 1.0).abs() < 1e-12);

        let u2 = [-5.0_f64];
        let p2 = c.project(&u2);
        assert!((p2[0] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn cost_zero_at_equilibrium() {
        // If x=0 and u=0, stage cost is 0 and terminal cost is 0
        let prob = build_problem();
        let x0 = [0.0_f64, 0.0];
        let u_seq = [[0.0_f64]; 20];
        let j = prob.cost(&x0, &u_seq).expect("cost should not fail");
        assert!(j.abs() < 1e-12, "j={:.2e}", j);
    }
}
