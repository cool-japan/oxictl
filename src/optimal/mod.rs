/// Optimal control via direct shooting methods.
///
/// This module provides no-std, allocation-free implementations of:
///
/// - **ODE solvers** — Euler, RK4, and adaptive RK45 (Fehlberg) integrators
///   used to transcribe the continuous-time dynamics into the shooting NLP.
/// - **Single shooting** — parameterises the entire control sequence `u(t)` and
///   minimises cost via projected gradient descent with Armijo line search.
/// - **Multiple shooting** — introduces auxiliary node states `{s_k}` as
///   decision variables and enforces continuity via a penalty method.
/// - **Pontryagin utilities** — Hamiltonian computation, adjoint co-state
///   dynamics, bang-bang control law, and adjoint-based shooting gradient.
///
/// All implementations are generic over the scalar type `S: ControlScalar`
/// and use const-generic arrays for zero heap allocation.
pub mod multiple_shooting;
pub mod ode_solver;
pub mod pontryagin;
pub mod single_shooting;

// ── re-exports ────────────────────────────────────────────────────────────────

pub use multiple_shooting::MultipleShootingProblem;
pub use ode_solver::{integrate, Euler, OdeSolver, RungeKutta4, RungeKuttaFehlberg};
pub use pontryagin::{
    bang_bang_control, compute_costate_derivative, compute_hamiltonian, shooting_gradient,
};
pub use single_shooting::{ControlConstraints, SingleShootingProblem};

// ── error type ────────────────────────────────────────────────────────────────

/// Errors that can arise during optimal control computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimalError {
    /// The ODE integrator encountered an invalid configuration (e.g. tf ≤ t0).
    IntegrationFailed(&'static str),

    /// The NLP solver exceeded the maximum iteration count without convergence.
    MaxIterationsExceeded,

    /// A numerical issue (NaN / Inf) was detected during computation.
    NumericalFailure,

    /// The control constraints are infeasible (e.g. u_min > u_max).
    InfeasibleConstraints,
}
