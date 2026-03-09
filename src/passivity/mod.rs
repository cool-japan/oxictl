//! Port-Hamiltonian systems and passivity-based control.
//!
//! This module provides:
//!   - [`port_hamiltonian`][]: Port-Hamiltonian system representation and validation
//!   - [`ida_pbc`][]: IDA-PBC (Interconnection and Damping Assignment Passivity-Based Control)
//!   - [`storage_function`][]: Lyapunov / storage function analysis
//!
//! ## Quick Start
//! ```rust,ignore
//! use oxictl::passivity::port_hamiltonian::LinearPh;
//!
//! // Mass-spring-damper (m=k=1, b=0.5): N=2 states, I=1 input
//! let ph = LinearPh::<f64, 2, 1>::new(
//!     [[0.0, 1.0], [-1.0, 0.0]],   // J (skew-symmetric)
//!     [[0.0, 0.0], [0.0, 0.5]],    // R (PSD damping)
//!     [[1.0, 0.0], [0.0, 1.0]],    // Q (energy weighting)
//!     [[0.0], [1.0]],              // g (input matrix)
//! ).unwrap();
//!
//! assert!(ph.is_passive());
//! ```

pub mod ida_pbc;
pub mod port_hamiltonian;
pub mod storage_function;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise in passivity-based control design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassivityError {
    /// The system is not passive (J not skew-symmetric or R not PSD).
    NotPassive,
    /// A matrix inversion failed (singular or near-singular matrix).
    SingularMatrix,
    /// The IDA-PBC matching equations could not be solved.
    MatchingFailed,
    /// The Hamiltonian is not a valid storage function (Q not positive definite).
    InvalidHamiltonian,
}

impl core::fmt::Display for PassivityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PassivityError::NotPassive => {
                f.write_str("system is not passive (J skew-symmetry or R PSD violated)")
            }
            PassivityError::SingularMatrix => {
                f.write_str("singular matrix encountered (cannot invert)")
            }
            PassivityError::MatchingFailed => {
                f.write_str("IDA-PBC matching equations could not be solved")
            }
            PassivityError::InvalidHamiltonian => {
                f.write_str("Hamiltonian is invalid (Q must be positive definite)")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use ida_pbc::{IdaPbcConfig, IdaPbcController, MechanicalIdaPbc};
pub use port_hamiltonian::{LinearPh, PortHamiltonian};
pub use storage_function::{LyapunovStabilityCheck, PassivityVerifier, StorageFunction};
