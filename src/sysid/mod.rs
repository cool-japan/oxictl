//! System Identification (`sysid`) module.
//!
//! Provides offline and online algorithms for identifying dynamic system models
//! from input/output data. All algorithms operate in `no_std` environments and
//! use fixed-size const-generic arrays (no heap allocation).
//!
//! # Modules
//! - [`arx`]: ARX model identification (batch LS and recursive RLS).
//! - [`armax`]: ARMAX model identification via Extended Least Squares (ELS).
//! - [`instrumental_variables`]: Instrumental Variables (IV) for bias-consistent estimation.
//! - [`validation`]: Model validation tools — FIT%, residual analysis, whiteness test.
//! - [`subspace`]: Simplified N4SID-inspired subspace identification.
//!
//! # Error type
//! All fallible operations return [`SysIdError`].
//!
//! # Notes on const generics
//! Because Rust stable does not yet support const arithmetic on generic parameters
//! in array-size positions (e.g. `[S; NA + NB]`), identifiers that need the
//! combined regressor dimension require an explicit `P` const parameter that the
//! caller must set to `NA + NB` (or `NA + NB + NC` for ARMAX).  The subspace
//! `identify` function similarly requires `NP1 = N + 1`.

pub mod armax;
pub mod arx;
pub mod instrumental_variables;
pub mod subspace;
pub mod validation;

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use armax::{ArmaxModel, ELSIdentifier};
pub use arx::{fit_percent as arx_fit_percent, ArxIdentifier, ArxModel, RecursiveArx};
pub use instrumental_variables::{IvIdentifier, RefIvIdentifier};
pub use subspace::{SubspaceIdConfig, SubspaceModel};
// Note: `subspace::identify` takes 5 const generics (N, I, PI, NP1) and must be
// called directly from user code as `subspace::identify::<S, N, I, PI, NP1>(...)`.
// A type-alias re-export is not possible since monomorphisation is caller-side.
pub use validation::{
    autocorrelation, cross_correlation, fit_percent, residual_analysis, whiteness_test,
    ResidualStats,
};

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors returned by system identification routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SysIdError {
    /// One or more inputs contained NaN or infinite values.
    InvalidData,
    /// A matrix required for inversion was (near-)singular.
    SingularMatrix,
    /// Not enough data samples were provided to form the regression matrix.
    InsufficientData,
    /// An iterative algorithm did not converge within the allowed iterations.
    NotConverged,
}

impl core::fmt::Display for SysIdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SysIdError::InvalidData => write!(f, "sysid: invalid data (NaN/Inf)"),
            SysIdError::SingularMatrix => write!(f, "sysid: singular matrix"),
            SysIdError::InsufficientData => write!(f, "sysid: insufficient data"),
            SysIdError::NotConverged => write!(f, "sysid: algorithm did not converge"),
        }
    }
}
