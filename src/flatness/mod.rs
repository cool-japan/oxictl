//! Differential flatness theory and flat trajectory planning.
//!
//! A system is differentially flat if there exists a *flat output* y such that
//! all states and inputs can be expressed as algebraic functions of y and its
//! time derivatives up to some finite order r:
//!
//! ```text
//!   x = Φ_x(y, ẏ, ÿ, ..., y^(r))
//!   u = Φ_u(y, ẏ, ÿ, ..., y^(r))
//! ```
//!
//! This module provides inverse maps (flat output → state/input) for:
//! - **Quadrotor UAV** — flat output [x, y, z, ψ]; uses minimum-snap polynomials.
//! - **Unicycle** — flat output [x, y]; path-tracking with lookahead.
//! - **2-DOF planar manipulator** — flat output [x_ee, y_ee]; geometric IK.

pub mod manipulator_flat;
pub mod quadrotor_flat;
pub mod unicycle_flat;

pub use manipulator_flat::{ManipulatorFlatMap, ManipulatorParams};
pub use quadrotor_flat::{FlatState, FlatTrajectory, QuadrotorFlatMap, QuadrotorFlatParams};
pub use unicycle_flat::{FlatPathTracker, UnicycleFlatMap};

use core::fmt;

/// Errors produced by differential-flatness inverse maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatnessError {
    /// A derivative index or order that is not supported was requested.
    InvalidDerivative,
    /// The flat output or its derivatives correspond to a singular configuration
    /// (e.g., zero velocity for unicycle heading, arm straight for manipulator).
    Singular,
    /// Evaluation time is outside the trajectory's valid range.
    OutOfRange,
    /// A required parameter (mass, link length, …) has an invalid value.
    InvalidParameter(&'static str),
    /// The polynomial solver failed to converge or produced a degenerate system.
    PolynomialSolver,
}

impl fmt::Display for FlatnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDerivative => write!(f, "Invalid derivative order requested"),
            Self::Singular => write!(f, "Singular configuration in flat map"),
            Self::OutOfRange => write!(f, "Time is outside the trajectory range"),
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {msg}"),
            Self::PolynomialSolver => write!(f, "Polynomial solver failed or degenerate system"),
        }
    }
}
