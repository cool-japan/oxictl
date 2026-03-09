// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright COOLJAPAN OU (Team Kitasan)
//
// ILC module: Iterative Learning Control algorithms.
//
// Provides P-type, D-type, and Norm-Optimal ILC controllers for SISO systems
// that perform the same task repeatedly.  All controllers operate on fixed-size
// trial buffers managed via const generics and are fully no_std compatible.

pub mod d_type_ilc;
pub mod norm_optimal_ilc;
pub mod p_type_ilc;

// ─────────────────────────────────────────────────────────────────────────────
// Shared error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by ILC controllers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IlcError {
    /// The supplied trial-error slice length does not match `TRIAL_LEN`.
    TrialLengthMismatch,
    /// A gain, weight, or time-step parameter is invalid (zero, negative,
    /// non-finite, or otherwise out of range).
    InvalidGain,
    /// The ILC feedforward has diverged (non-finite value detected).
    NotConverged,
}

impl core::fmt::Display for IlcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IlcError::TrialLengthMismatch => {
                write!(f, "IlcError: trial length mismatch")
            }
            IlcError::InvalidGain => {
                write!(
                    f,
                    "IlcError: invalid gain or parameter (zero/negative/non-finite)"
                )
            }
            IlcError::NotConverged => {
                write!(f, "IlcError: feedforward diverged (non-finite value)")
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports
// ─────────────────────────────────────────────────────────────────────────────

pub use d_type_ilc::DTypeIlc;
pub use norm_optimal_ilc::NormOptimalIlc;
pub use p_type_ilc::PTypeIlc;
