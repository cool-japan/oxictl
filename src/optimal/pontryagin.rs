//! Pontryagin Minimum Principle (PMP) utilities.
//!
//! The Pontryagin Minimum Principle provides necessary conditions for
//! optimality in continuous-time optimal control via the *Hamiltonian*:
//!
//! ```text
//! H(x, u, λ) = l(x, u) + λᵀ f(x, u)
//! ```
//!
//! where:
//! - `l(x, u)` — running cost (Lagrangian)
//! - `f(x, u)` — system dynamics (vector field)
//! - `λ`       — co-state (adjoint) vector
//!
//! Optimality conditions:
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
/// - **State dynamics**: `ẋ = ∂H/∂λ = f(x, u)`
/// - **Co-state dynamics**: `λ̇ = −∂H/∂x`
/// - **Stationarity**: `∂H/∂u = 0` (unconstrained) or `u* = argmin_u H`
///
/// This module provides computational utilities implementing these conditions
/// via finite-difference approximations, plus a specialised bang-bang law.
use crate::core::scalar::ControlScalar;

// ─────────────────────────────────────── Hamiltonian ─────────────────────────

/// Compute the Hamiltonian `H(x, u, λ) = l(x, u) + λᵀ·f(x, u)`.
///
/// # Arguments
/// - `x`      — state vector (length `N`)
/// - `u`      — control vector (length `I`)
/// - `lambda` — co-state vector (length `N`)
/// - `f`      — dynamics: `ẋ = f(x, u)`
/// - `l`      — running cost: `l(x, u)`
///
/// # Returns
/// The scalar Hamiltonian value.
pub fn compute_hamiltonian<S, const N: usize, const I: usize>(
    x: &[S; N],
    u: &[S; I],
    lambda: &[S; N],
    f: impl Fn(&[S; N], &[S; I]) -> [S; N],
    l: impl Fn(&[S; N], &[S; I]) -> S,
) -> S
where
    S: ControlScalar,
{
    let running = l(x, u);
    let fx = f(x, u);
    // λᵀ·f(x,u)
    let costate_inner: S = lambda
        .iter()
        .zip(fx.iter())
        .fold(S::ZERO, |acc, (&lam_i, &fi)| acc + lam_i * fi);
    running + costate_inner
}

// ─────────────────────────────── co-state derivative ─────────────────────────

/// Compute the co-state derivative `λ̇ = −∂H/∂x` via central finite differences.
///
/// The partial derivative `∂H/∂x_i` is approximated as:
///
/// ```text
/// ∂H/∂x_i ≈ (H(x + ε·eᵢ, u, λ) − H(x − ε·eᵢ, u, λ)) / (2ε)
/// ```
///
/// Then `λ̇_i = −∂H/∂x_i`.
///
/// # Arguments
/// - `x`, `u`, `lambda` — current state, control, co-state
/// - `f` — system dynamics
/// - `l` — running cost
/// - `eps` — finite-difference step size (recommend: ~1e-6)
///
/// # Returns
/// Co-state derivative vector `λ̇` of length `N`.
pub fn compute_costate_derivative<S, const N: usize, const I: usize>(
    x: &[S; N],
    u: &[S; I],
    lambda: &[S; N],
    f: impl Fn(&[S; N], &[S; I]) -> [S; N] + Copy,
    l: impl Fn(&[S; N], &[S; I]) -> S + Copy,
    eps: S,
) -> [S; N]
where
    S: ControlScalar,
{
    let two_eps = eps + eps;
    core::array::from_fn(|i| {
        let mut x_plus = *x;
        x_plus[i] += eps;
        let mut x_minus = *x;
        x_minus[i] -= eps;

        let h_plus = compute_hamiltonian(&x_plus, u, lambda, f, l);
        let h_minus = compute_hamiltonian(&x_minus, u, lambda, f, l);

        // λ̇_i = −∂H/∂x_i
        -(h_plus - h_minus) / two_eps
    })
}

// ──────────────────────────────────── bang-bang control ───────────────────────

/// Compute the optimal bang-bang control law for a scalar affine system.
///
/// For systems linear in the control: `f(x, u) = f₀(x) + B·u`, the
/// Pontryagin minimum principle yields:
///
/// ```text
/// u*(t) = u_min  if  λᵀ·B > 0
/// u*(t) = u_max  if  λᵀ·B < 0
/// u*(t) = 0      if  λᵀ·B = 0  (singular arc — use mid-point)
/// ```
///
/// This function computes `u*` for a *single* scalar control channel given
/// the switching function `σ = λᵀ·b`, where `b` is the `N`-dimensional
/// column of `B` for that channel.
///
/// # Arguments
/// - `u_min`, `u_max` — scalar control bounds
/// - `costate`        — adjoint vector `λ` (length `N`)
/// - `b_column`       — column of `B` corresponding to this control channel
///
/// # Returns
/// Optimal bang-bang control value `u*`.
pub fn bang_bang_control<S, const N: usize>(
    u_min: S,
    u_max: S,
    costate: &[S; N],
    b_column: &[S; N],
) -> S
where
    S: ControlScalar,
{
    // Switching function σ = λᵀ·b
    let sigma: S = costate
        .iter()
        .zip(b_column.iter())
        .fold(S::ZERO, |acc, (&lam_i, &bi)| acc + lam_i * bi);

    if sigma > S::ZERO {
        u_min
    } else if sigma < S::ZERO {
        u_max
    } else {
        // Singular arc — use midpoint as a neutral choice
        S::HALF * (u_min + u_max)
    }
}

// ─────────────────────────────────── shooting gradient via adjoint ───────────

/// Gradient of the total cost `J` w.r.t. the control sequence via the
/// adjoint (backward sweep) method.
///
/// This is more efficient than blind finite-differences on `J` directly
/// because it re-uses the backward pass that is already implicit in the
/// Pontryagin co-state equations.
///
/// **Algorithm** (discrete-time approximation):
/// 1. Forward pass: simulate `x_{k+1} = F(x_k, u_k)` via RK4, collect states.
/// 2. Backward pass: initialise `λ_M = ∂φ/∂x_M` (terminal gradient via FD),
///    then propagate `λ_k = −dt·(∂H/∂x)|_{x_k,u_k,λ_{k+1}} + λ_{k+1}`.
/// 3. Gradient: `∂J/∂u_k = ∂H/∂u|_{x_k,u_k,λ_{k+1}}` via central FD.
///
/// # Type parameters
/// - `N`: state dimension
/// - `I`: control dimension
/// - `MI`: total parameter count `M*I`
///
/// # Arguments
/// - `x0`         — initial state
/// - `u_seq`      — control sequence (M steps of I-vectors, flattened to MI)
/// - `dynamics`   — system dynamics `f(x, u)`
/// - `stage_cost` — running cost `l(x, u)`
/// - `terminal_cost` — terminal cost `φ(x)`
/// - `dt`         — time step per interval
/// - `ode_steps`  — ODE sub-steps per interval
/// - `eps`        — finite-difference perturbation size
///
/// # Returns
/// Gradient vector of length `M*I` (flattened row-major: [∂J/∂u_0, ∂J/∂u_1, …]).
pub fn shooting_gradient<S, const N: usize, const I: usize, const M: usize, const MI: usize>(
    x0: &[S; N],
    u_seq: &[S; MI],
    dynamics: impl Fn(&[S; N], &[S; I]) -> [S; N] + Copy,
    stage_cost: impl Fn(&[S; N], &[S; I]) -> S + Copy,
    terminal_cost: impl Fn(&[S; N]) -> S + Copy,
    dt: S,
    ode_steps: usize,
    eps: S,
) -> [S; MI]
where
    S: ControlScalar,
{
    use crate::optimal::ode_solver::{OdeSolver, RungeKutta4};

    let solver = RungeKutta4::<S, N>::new();
    let sub_dt = dt / S::from_f64(ode_steps as f64);
    let two_eps = eps + eps;

    // ── Forward pass: collect states x_0 … x_M ──────────────────────────────
    // Store M+1 states; index 0 is x0, indices 1…M are the propagated states.
    // We avoid [S;N; M+1] (unstable const-expr), so we use two arrays:
    //   x_nodes[k] = x after k intervals (k=0 is x0).
    // We store only M node states (x_1 … x_M); x0 is separate.
    let mut x_nodes = [[S::ZERO; N]; M];
    {
        let mut x = *x0;
        for k in 0..M {
            let u: [S; I] = core::array::from_fn(|j| u_seq[k * I + j]);
            let f = |xv: &[S; N], _t: S| -> [S; N] { dynamics(xv, &u) };
            for _ in 0..ode_steps {
                x = solver.step(f, &x, S::ZERO, sub_dt);
            }
            x_nodes[k] = x;
        }
    }

    // ── Backward pass: propagate co-state ────────────────────────────────────
    // Terminal: λ_M = ∂φ/∂x_M (central FD)
    let x_terminal = x_nodes[M - 1];
    let mut lambda: [S; N] = core::array::from_fn(|i| {
        let mut xp = x_terminal;
        xp[i] += eps;
        let mut xm = x_terminal;
        xm[i] -= eps;
        (terminal_cost(&xp) - terminal_cost(&xm)) / two_eps
    });

    let mut grad = [S::ZERO; MI];

    for k in (0..M).rev() {
        let x_k = if k == 0 { *x0 } else { x_nodes[k - 1] };
        let u: [S; I] = core::array::from_fn(|j| u_seq[k * I + j]);

        // ∂J/∂u_k = ∂H/∂u|_{x_k, u_k, λ_{k+1}}  via central FD
        for j in 0..I {
            let mut u_plus = u;
            u_plus[j] += eps;
            let mut u_minus = u;
            u_minus[j] -= eps;

            let h_plus = compute_hamiltonian(&x_k, &u_plus, &lambda, dynamics, stage_cost);
            let h_minus = compute_hamiltonian(&x_k, &u_minus, &lambda, dynamics, stage_cost);
            grad[k * I + j] = (h_plus - h_minus) / two_eps;
        }

        // Update co-state: λ_k = λ_{k+1} − dt · ∂H/∂x
        let dlambda = compute_costate_derivative(&x_k, &u, &lambda, dynamics, stage_cost, eps);
        lambda = core::array::from_fn(|i| lambda[i] + dt * dlambda[i]);
    }

    grad
}

// ──────────────────────────────────────────── tests ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Linear Quadratic problem ──────────────────────────────────────────────
    // System: ẋ = -x + u   (scalar, N=1, I=1)
    // Cost:   l(x,u) = x² + u²
    // H(x, u, λ) = x² + u² + λ·(-x + u)
    // ∂H/∂x = 2x - λ  =>  λ̇ = -(2x - λ) = λ - 2x
    // ∂H/∂u = 2u + λ  =>  u* = -λ/2  (unconstrained minimum)

    fn lq_dynamics(x: &[f64; 1], u: &[f64; 1]) -> [f64; 1] {
        [-x[0] + u[0]]
    }

    fn lq_stage(x: &[f64; 1], u: &[f64; 1]) -> f64 {
        x[0] * x[0] + u[0] * u[0]
    }

    #[test]
    fn hamiltonian_lq_manual() {
        // x=1, u=0, λ=1
        // H = 1² + 0² + 1·(-1+0) = 1 - 1 = 0
        let x = [1.0_f64];
        let u = [0.0_f64];
        let lambda = [1.0_f64];
        let h = compute_hamiltonian(&x, &u, &lambda, lq_dynamics, lq_stage);
        assert!((h - 0.0).abs() < 1e-12, "H={:.6}", h);
    }

    #[test]
    fn hamiltonian_lq_with_control() {
        // x=2, u=1, λ=3
        // H = 4 + 1 + 3·(-2+1) = 5 - 3 = 2
        let x = [2.0_f64];
        let u = [1.0_f64];
        let lambda = [3.0_f64];
        let h = compute_hamiltonian(&x, &u, &lambda, lq_dynamics, lq_stage);
        assert!((h - 2.0).abs() < 1e-12, "H={:.6}", h);
    }

    #[test]
    fn costate_derivative_lq() {
        // λ̇ = −∂H/∂x = −(2x − λ) = λ − 2x
        // At x=1, u=0, λ=1: λ̇ = 1 - 2 = -1
        let x = [1.0_f64];
        let u = [0.0_f64];
        let lambda = [1.0_f64];
        let dlambda = compute_costate_derivative(&x, &u, &lambda, lq_dynamics, lq_stage, 1e-7);
        let expected = lambda[0] - 2.0 * x[0]; // -1.0
        assert!(
            (dlambda[0] - expected).abs() < 1e-5,
            "λ̇={:.8} expected={:.8}",
            dlambda[0],
            expected
        );
    }

    #[test]
    fn costate_derivative_matches_sign() {
        // At x=3, u=1, λ=0: λ̇ = 0 - 6 = -6
        let x = [3.0_f64];
        let u = [1.0_f64];
        let lambda = [0.0_f64];
        let dlambda = compute_costate_derivative(&x, &u, &lambda, lq_dynamics, lq_stage, 1e-7);
        let expected = -2.0 * x[0]; // -6.0
        assert!(
            (dlambda[0] - expected).abs() < 1e-5,
            "λ̇={:.8} expected={:.8}",
            dlambda[0],
            expected
        );
    }

    #[test]
    fn bang_bang_selects_min_when_sigma_positive() {
        // σ = λᵀ·b > 0  =>  u* = u_min
        let costate = [2.0_f64, 1.0];
        let b = [1.0_f64, 0.0];
        let u = bang_bang_control(-1.0_f64, 1.0_f64, &costate, &b);
        assert!((u - (-1.0)).abs() < 1e-12, "u={:.4}", u);
    }

    #[test]
    fn bang_bang_selects_max_when_sigma_negative() {
        // σ = λᵀ·b < 0  =>  u* = u_max
        let costate = [-3.0_f64, 0.0];
        let b = [1.0_f64, 0.0];
        let u = bang_bang_control(-1.0_f64, 1.0_f64, &costate, &b);
        assert!((u - 1.0).abs() < 1e-12, "u={:.4}", u);
    }

    #[test]
    fn bang_bang_singular_arc_is_midpoint() {
        // σ = 0  =>  u* = (u_min + u_max)/2
        let costate = [0.0_f64];
        let b = [1.0_f64];
        let u = bang_bang_control(-2.0_f64, 4.0_f64, &costate, &b);
        assert!((u - 1.0).abs() < 1e-12, "u={:.4}", u);
    }

    #[test]
    fn shooting_gradient_lq_direction() {
        // For the LQ problem starting at x=1 with u=0 over M=5 steps,
        // the gradient should be non-zero (pointing toward decreasing cost).
        fn lq_terminal(x: &[f64; 1]) -> f64 {
            x[0] * x[0]
        }

        // MI = M*I = 5*1 = 5
        let x0 = [1.0_f64];
        let u_seq = [0.0_f64; 5]; // M=5, I=1 => MI=5
        let grad = shooting_gradient::<f64, 1, 1, 5, 5>(
            &x0,
            &u_seq,
            lq_dynamics,
            lq_stage,
            lq_terminal,
            0.1,
            2,
            1e-6,
        );

        // At u=0, the gradient w.r.t. u_k should reflect the cost structure
        // (at least some components should be non-zero)
        let grad_norm: f64 = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
        assert!(grad_norm > 1e-6, "Gradient should be non-zero: {:?}", grad);
    }

    #[test]
    fn shooting_gradient_zero_at_equilibrium() {
        // At x=0, u=0 (equilibrium of ẋ = -x + u), cost = 0, gradient should be ~0
        fn lq_terminal(x: &[f64; 1]) -> f64 {
            x[0] * x[0]
        }

        let x0 = [0.0_f64];
        let u_seq = [0.0_f64; 3]; // M=3, I=1 => MI=3
        let grad = shooting_gradient::<f64, 1, 1, 3, 3>(
            &x0,
            &u_seq,
            lq_dynamics,
            lq_stage,
            lq_terminal,
            0.1,
            2,
            1e-6,
        );

        for (k, &g) in grad.iter().enumerate() {
            assert!(
                g.abs() < 1e-6,
                "grad[{}]={:.2e} should be ~0 at equilibrium",
                k,
                g
            );
        }
    }
}
