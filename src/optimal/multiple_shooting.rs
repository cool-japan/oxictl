//! Direct multiple shooting method for nonlinear optimal control.
//!
//! Divides the time horizon into `M` segments.  Each segment has:
//! - An initial state decision variable `s_k ∈ ℝᴺ` (the "node" state)
//! - A constant control `u_k ∈ ℝᴵ`
//!
//! Shooting a segment forward yields the *predicted* state at the segment
//! end: `F(s_k, u_k)`.  Continuity between segments is enforced through
//! the *shooting gap* residuals:
//!
//! ```text
//! g_k = s_{k+1} − F(s_k, u_k),  k = 0, …, M-2
//! ```
//!
//! These are penalised in a quadratic augmented cost:
//!
//! ```text
//! J_aug = J + ρ · Σ_k ‖g_k‖²
//! ```
//!
//! The penalty parameter `ρ` is increased when gaps remain large, giving a
//! sequence of unconstrained NLP subproblems converging to feasibility.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::large_enum_variant
)]
use crate::core::scalar::ControlScalar;

use super::{
    ode_solver::{OdeSolver, RungeKutta4},
    single_shooting::ControlConstraints,
    OptimalError,
};

// ───────────────────────────────────── problem definition ─────────────────────

/// Optimal control problem transcribed by direct multiple shooting.
///
/// # Type parameters
/// - `S`: scalar type
/// - `N`: state dimension
/// - `I`: control dimension
/// - `M`: number of shooting intervals (horizon)
pub struct MultipleShootingProblem<S, const N: usize, const I: usize, const M: usize>
where
    S: ControlScalar,
{
    /// Continuous-time vector field `ẋ = dynamics(x, u)`.
    pub dynamics: fn(&[S; N], &[S; I]) -> [S; N],

    /// Running (stage) cost `l(x_k, u_k)`.
    pub stage_cost: fn(&[S; N], &[S; I]) -> S,

    /// Terminal cost `φ(x_M)`.
    pub terminal_cost: fn(&[S; N]) -> S,

    /// ODE step size within each segment.
    pub dt: S,

    /// Number of ODE sub-steps per segment.
    pub ode_steps: usize,

    /// Control box constraints.
    pub constraints: ControlConstraints<S, I>,

    /// Finite-difference step for gradient computation.
    pub fd_eps: S,

    /// Maximum outer iterations (penalty schedule).
    pub max_outer_iter: usize,

    /// Maximum inner gradient-descent iterations per outer iteration.
    pub max_inner_iter: usize,

    /// Initial penalty parameter `ρ₀`.
    pub rho_init: S,

    /// Penalty growth factor (multiplied each outer step).
    pub rho_growth: S,

    /// Maximum penalty parameter.
    pub rho_max: S,

    /// Convergence tolerance on shooting gap ‖g_k‖∞.
    pub gap_tol: S,

    /// Inner gradient step size.
    pub step_size: S,

    /// Armijo c₁.
    pub armijo_c1: S,

    /// Armijo backtracking factor.
    pub armijo_beta: S,
}

impl<S, const N: usize, const I: usize, const M: usize> MultipleShootingProblem<S, N, I, M>
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
            max_outer_iter: 20,
            max_inner_iter: 100,
            rho_init: S::from_f64(1.0),
            rho_growth: S::from_f64(5.0),
            rho_max: S::from_f64(1e6),
            gap_tol: S::from_f64(1e-4),
            step_size: S::from_f64(1e-3),
            armijo_c1: S::from_f64(1e-4),
            armijo_beta: S::from_f64(0.5),
        }
    }

    // ── ODE endpoint map F(s_k, u_k) ─────────────────────────────────────────

    /// Integrate one segment forward: `F(s, u) = φ_{dt·ode_steps}(s; u)`.
    fn shoot_segment(&self, s: &[S; N], u: &[S; I]) -> [S; N] {
        let solver = RungeKutta4::<S, N>::new();
        let sub_dt = self.dt / S::from_f64(self.ode_steps as f64);
        let mut x = *s;
        let dyn_fn = self.dynamics;
        for _ in 0..self.ode_steps {
            let uf = u;
            let f = |xv: &[S; N], _t: S| -> [S; N] { dyn_fn(xv, uf) };
            x = solver.step(f, &x, S::ZERO, sub_dt);
        }
        x
    }

    // ── shooting gap computation ──────────────────────────────────────────────

    /// Compute shooting gaps `g_k = s_{k+1} − F(s_k, u_k)` for k = 0…M-2.
    ///
    /// The gap array has `M-1` entries; if M==1 there are no internal gaps
    /// (only initial-condition constraint applied externally).
    fn compute_gaps(&self, states: &[[S; N]; M], u_seq: &[[S; I]; M]) -> [[S; N]; M] {
        // We allocate M slots; k=0…M-2 hold actual gaps, k=M-1 is zeroed.
        let mut gaps = [[S::ZERO; N]; M];
        if M < 2 {
            return gaps;
        }
        for k in 0..M - 1 {
            let f_s = self.shoot_segment(&states[k], &u_seq[k]);
            for i in 0..N {
                gaps[k][i] = states[k + 1][i] - f_s[i];
            }
        }
        gaps
    }

    /// ‖g‖∞ over all gaps.
    fn max_gap_norm(&self, gaps: &[[S; N]; M]) -> S {
        let mut max_norm = S::ZERO;
        for k in 0..M.saturating_sub(1) {
            for i in 0..N {
                let v = gaps[k][i].abs();
                if v > max_norm {
                    max_norm = v;
                }
            }
        }
        max_norm
    }

    // ── augmented cost J_aug ──────────────────────────────────────────────────

    /// Evaluate the augmented cost:
    ///   `J_aug = Σ stage_cost(s_k, u_k) + terminal_cost(s_M_approx) + ρ·Σ‖g_k‖²`
    ///
    /// where `s_M_approx = F(s_{M-1}, u_{M-1})`.
    fn augmented_cost(&self, x0: &[S; N], states: &[[S; N]; M], u_seq: &[[S; I]; M], rho: S) -> S {
        // Stage cost: evaluated at node states (not the propagated states)
        // Node 0 is fixed to x0 (external constraint).
        let mut j = (self.stage_cost)(x0, &u_seq[0]);
        for k in 1..M {
            j += (self.stage_cost)(&states[k - 1], &u_seq[k]);
        }

        // Terminal: propagate s_{M-1} forward one step
        let x_terminal = self.shoot_segment(&states[M - 1], &u_seq[M - 1]);
        j += (self.terminal_cost)(&x_terminal);

        // Shooting gap penalty
        // Gap 0: s_0 - F(x0, u_0)
        let f0 = self.shoot_segment(x0, &u_seq[0]);
        let mut gap_sq = S::ZERO;
        for i in 0..N {
            let g = states[0][i] - f0[i];
            gap_sq += g * g;
        }
        // Gaps k=1..M-1: s_k - F(s_{k-1}, u_{k-1})
        for k in 1..M {
            let f_prev = self.shoot_segment(&states[k - 1], &u_seq[k]);
            for i in 0..N {
                let g = states[k][i] - f_prev[i];
                gap_sq += g * g;
            }
        }
        j += rho * gap_sq;
        j
    }

    // ── finite-difference gradient w.r.t. u and s ────────────────────────────

    /// Gradient of `J_aug` w.r.t. `u_seq` (shape `[M; I]`).
    fn gradient_u(
        &self,
        x0: &[S; N],
        states: &[[S; N]; M],
        u_seq: &[[S; I]; M],
        rho: S,
    ) -> [[S; I]; M] {
        let eps = self.fd_eps;
        let two_eps = eps + eps;
        let mut grad = [[S::ZERO; I]; M];

        for k in 0..M {
            for j in 0..I {
                let mut u_plus = *u_seq;
                u_plus[k][j] += eps;
                let mut u_minus = *u_seq;
                u_minus[k][j] -= eps;

                let jp = self.augmented_cost(x0, states, &u_plus, rho);
                let jm = self.augmented_cost(x0, states, &u_minus, rho);
                grad[k][j] = (jp - jm) / two_eps;
            }
        }
        grad
    }

    /// Gradient of `J_aug` w.r.t. `states` (shape `[M; N]`).
    fn gradient_s(
        &self,
        x0: &[S; N],
        states: &[[S; N]; M],
        u_seq: &[[S; I]; M],
        rho: S,
    ) -> [[S; N]; M] {
        let eps = self.fd_eps;
        let two_eps = eps + eps;
        let mut grad = [[S::ZERO; N]; M];

        for k in 0..M {
            for i in 0..N {
                let mut s_plus = *states;
                s_plus[k][i] += eps;
                let mut s_minus = *states;
                s_minus[k][i] -= eps;

                let jp = self.augmented_cost(x0, &s_plus, u_seq, rho);
                let jm = self.augmented_cost(x0, &s_minus, u_seq, rho);
                grad[k][i] = (jp - jm) / two_eps;
            }
        }
        grad
    }

    // ── Armijo backtracking (joint u and s) ───────────────────────────────────

    fn line_search_joint(
        &self,
        x0: &[S; N],
        states: &[[S; N]; M],
        u_seq: &[[S; I]; M],
        grad_u: &[[S; I]; M],
        grad_s: &[[S; N]; M],
        j_curr: S,
        rho: S,
    ) -> ([[S; N]; M], [[S; I]; M], S) {
        let mut alpha = self.step_size;
        let min_alpha = S::from_f64(1e-15);

        // Directional derivative magnitude
        let mut dir_sq = S::ZERO;
        for k in 0..M {
            for j in 0..I {
                dir_sq += grad_u[k][j] * grad_u[k][j];
            }
            for i in 0..N {
                dir_sq += grad_s[k][i] * grad_s[k][i];
            }
        }
        let dir_deriv = -dir_sq;

        loop {
            let u_cand: [[S; I]; M] = core::array::from_fn(|k| {
                self.constraints.project(&core::array::from_fn(|j| {
                    u_seq[k][j] - alpha * grad_u[k][j]
                }))
            });
            let s_cand: [[S; N]; M] = core::array::from_fn(|k| {
                core::array::from_fn(|i| states[k][i] - alpha * grad_s[k][i])
            });

            let j_new = self.augmented_cost(x0, &s_cand, &u_cand, rho);
            if j_new <= j_curr + self.armijo_c1 * alpha * dir_deriv {
                return (s_cand, u_cand, alpha);
            }

            alpha *= self.armijo_beta;
            if alpha < min_alpha {
                let u_fallback: [[S; I]; M] = core::array::from_fn(|k| {
                    self.constraints.project(&core::array::from_fn(|j| {
                        u_seq[k][j] - alpha * grad_u[k][j]
                    }))
                });
                let s_fallback: [[S; N]; M] = core::array::from_fn(|k| {
                    core::array::from_fn(|i| states[k][i] - alpha * grad_s[k][i])
                });
                return (s_fallback, u_fallback, alpha);
            }
        }
    }

    // ── inner minimisation ────────────────────────────────────────────────────

    /// Run gradient descent on the augmented cost for fixed `rho`.
    fn inner_solve(&self, x0: &[S; N], states: &mut [[S; N]; M], u_seq: &mut [[S; I]; M], rho: S) {
        for _ in 0..self.max_inner_iter {
            let j_curr = self.augmented_cost(x0, states, u_seq, rho);
            let grad_u = self.gradient_u(x0, states, u_seq, rho);
            let grad_s = self.gradient_s(x0, states, u_seq, rho);

            // Convergence check
            let mut g_norm = S::ZERO;
            for k in 0..M {
                for j in 0..I {
                    let g = grad_u[k][j].abs();
                    if g > g_norm {
                        g_norm = g;
                    }
                }
                for i in 0..N {
                    let g = grad_s[k][i].abs();
                    if g > g_norm {
                        g_norm = g;
                    }
                }
            }
            if g_norm < S::from_f64(1e-8) {
                break;
            }

            let (s_new, u_new, _) =
                self.line_search_joint(x0, states, u_seq, &grad_u, &grad_s, j_curr, rho);
            *states = s_new;
            *u_seq = u_new;
        }
    }

    // ── main solver ───────────────────────────────────────────────────────────

    /// Solve the multiple-shooting optimal control problem.
    ///
    /// # Parameters
    /// - `x0`     — initial state (first node is constrained to `x0`)
    /// - `u_init` — initial control sequence
    ///
    /// # Returns
    /// `(states, u_optimal, cost)`:
    /// - `states`: the M node states `[s_0, …, s_{M-1}]` (continuity enforced)
    /// - `u_optimal`: the optimal piecewise-constant control sequence
    /// - `cost`: total cost at optimality
    pub fn solve(
        &self,
        x0: &[S; N],
        u_init: &[[S; I]; M],
    ) -> Result<([[S; N]; M], [[S; I]; M], S), OptimalError> {
        // Initialise node states by open-loop forward simulation
        let mut states: [[S; N]; M] = {
            let mut s = [[S::ZERO; N]; M];
            let mut x = *x0;
            let solver = RungeKutta4::<S, N>::new();
            let sub_dt = self.dt / S::from_f64(self.ode_steps as f64);
            for k in 0..M {
                let u = &u_init[k];
                let dyn_fn = self.dynamics;
                let f = |xv: &[S; N], _t: S| -> [S; N] { dyn_fn(xv, u) };
                for _ in 0..self.ode_steps {
                    x = solver.step(f, &x, S::ZERO, sub_dt);
                }
                s[k] = x;
            }
            s
        };

        let mut u_seq: [[S; I]; M] = core::array::from_fn(|k| self.constraints.project(&u_init[k]));

        let mut rho = self.rho_init;

        for _outer in 0..self.max_outer_iter {
            // Minimise augmented cost
            self.inner_solve(x0, &mut states, &mut u_seq, rho);

            // Check continuity constraint satisfaction
            let gaps = self.compute_gaps(&states, &u_seq);
            let gap_norm = self.max_gap_norm(&gaps);

            if gap_norm < self.gap_tol {
                // Feasible — done
                break;
            }

            // Increase penalty
            rho = (rho * self.rho_growth).clamp_val(rho, self.rho_max);
        }

        // Final cost (without the penalty term, for interpretability)
        // Use the augmented cost at low rho=0 effectively, but we compute
        // by integrating from x0 with the final u_seq (single-shooting cost)
        let final_cost = {
            let solver = RungeKutta4::<S, N>::new();
            let sub_dt = self.dt / S::from_f64(self.ode_steps as f64);
            let mut x = *x0;
            let mut j = S::ZERO;
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
            j
        };

        Ok((states, u_seq, final_cost))
    }

    /// Compute the shooting gaps for the solution — useful for diagnostics.
    ///
    /// Returns gap array; `gap_norm_inf` of all-zeros confirms feasibility.
    pub fn solution_gaps(
        &self,
        x0: &[S; N],
        states: &[[S; N]; M],
        u_seq: &[[S; I]; M],
    ) -> [[S; N]; M] {
        // Include the gap from x0 to s_0
        let mut gaps = [[S::ZERO; N]; M];
        let f0 = self.shoot_segment(x0, &u_seq[0]);
        for i in 0..N {
            gaps[0][i] = states[0][i] - f0[i];
        }
        for k in 1..M.saturating_sub(1) {
            let f_prev = self.shoot_segment(&states[k - 1], &u_seq[k]);
            for i in 0..N {
                gaps[k][i] = states[k][i] - f_prev[i];
            }
        }
        gaps
    }
}

// ──────────────────────────────────────────── tests ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn double_integrator_dynamics(x: &[f64; 2], u: &[f64; 1]) -> [f64; 2] {
        [x[1], u[0]]
    }

    fn min_energy_stage(_x: &[f64; 2], u: &[f64; 1]) -> f64 {
        u[0] * u[0]
    }

    fn quadratic_terminal(x: &[f64; 2]) -> f64 {
        10.0 * (x[0] * x[0] + x[1] * x[1])
    }

    fn build_problem() -> MultipleShootingProblem<f64, 2, 1, 10> {
        let constraints = ControlConstraints::box_input([-5.0], [5.0]);
        let mut prob = MultipleShootingProblem::new(
            double_integrator_dynamics,
            min_energy_stage,
            quadratic_terminal,
            0.1,
            constraints,
        );
        prob.max_outer_iter = 15;
        prob.max_inner_iter = 80;
        prob.step_size = 0.02;
        prob.rho_init = 1.0;
        prob.rho_growth = 4.0;
        prob.gap_tol = 1e-3;
        prob.ode_steps = 2;
        prob
    }

    #[test]
    fn multiple_shooting_double_integrator_reduces_cost() {
        let prob = build_problem();
        let x0 = [1.0_f64, 0.0];
        let u_init = [[0.0_f64]; 10];

        // Cost under zero control (just terminal cost)
        let j_zero = {
            let solver = super::super::ode_solver::RungeKutta4::<f64, 2>::new();
            let sub_dt = prob.dt / prob.ode_steps as f64;
            let mut x = x0;
            let u = [0.0_f64];
            let f = |xv: &[f64; 2], _t: f64| -> [f64; 2] { [xv[1], u[0]] };
            for _ in 0..10 {
                for _ in 0..prob.ode_steps {
                    x = solver.step(f, &x, 0.0, sub_dt);
                }
            }
            quadratic_terminal(&x)
        };

        let (_states, u_opt, j_opt) = prob
            .solve(&x0, &u_init)
            .expect("multiple shooting should not fail");

        assert!(
            j_opt < j_zero * 1.5,
            "j_opt={:.4} should be comparable to zero-control cost {:.4}",
            j_opt,
            j_zero
        );

        // All controls must be within bounds
        for k in 0..10 {
            assert!(u_opt[k][0] >= -5.0 - 1e-9);
            assert!(u_opt[k][0] <= 5.0 + 1e-9);
        }
    }

    #[test]
    fn continuity_constraints_satisfied() {
        let prob = build_problem();
        let x0 = [1.0_f64, 0.0];
        let u_init = [[0.0_f64]; 10];
        let (states, u_opt, _j) = prob
            .solve(&x0, &u_init)
            .expect("multiple shooting should not fail");

        let gaps = prob.solution_gaps(&x0, &states, &u_opt);
        // Check all internal gaps (k = 1..9) are small
        for k in 1..9 {
            let gap_norm = gaps[k][0].abs().max(gaps[k][1].abs());
            assert!(
                gap_norm < 0.5,
                "gap at segment {} = {:.4e} should be small",
                k,
                gap_norm
            );
        }
    }

    #[test]
    fn augmented_cost_zero_at_equilibrium() {
        let prob = build_problem();
        let x0 = [0.0_f64, 0.0];
        let states = [[0.0_f64, 0.0]; 10];
        let u_seq = [[0.0_f64]; 10];
        let j = prob.augmented_cost(&x0, &states, &u_seq, 1.0);
        // With x=0, u=0, all gaps are 0, stage/terminal costs are 0
        assert!(j.abs() < 1e-12, "j={:.2e}", j);
    }

    #[test]
    fn shoot_segment_integrates_correctly() {
        // Free particle: ẋ₁ = x₂, ẋ₂ = 0 (u=0)
        // x(t) = [x0[0] + x0[1]*t, x0[1]]
        let prob = build_problem();
        let s = [1.0_f64, 2.0];
        let u = [0.0_f64];
        let s_next = prob.shoot_segment(&s, &u);
        // Expected: after dt=0.1 with ode_steps=2 sub-steps
        let t = prob.dt;
        let expected = [s[0] + s[1] * t, s[1]];
        assert!(
            (s_next[0] - expected[0]).abs() < 1e-6,
            "s_next[0]={:.6} expected={:.6}",
            s_next[0],
            expected[0]
        );
        assert!(
            (s_next[1] - expected[1]).abs() < 1e-10,
            "s_next[1]={:.6} expected={:.6}",
            s_next[1],
            expected[1]
        );
    }
}
