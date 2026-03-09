// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright COOLJAPAN OU (Team Kitasan)
//
// IMC module: Internal Model Control and Predictive Functional Control.

pub mod imc_controller;
pub mod pfc;
pub mod smith_predictor;

// ──────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────

/// Errors produced by the IMC / Smith / PFC controllers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImcError {
    /// A configuration parameter is outside its valid range.
    InvalidParameter(&'static str),
    /// A numerical issue (near-singular matrix, degenerate coefficient, …).
    NumericalError(&'static str),
}

impl core::fmt::Display for ImcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ImcError::InvalidParameter(msg) => write!(f, "ImcError::InvalidParameter: {}", msg),
            ImcError::NumericalError(msg) => write!(f, "ImcError::NumericalError: {}", msg),
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Re-exports
// ──────────────────────────────────────────────────────────────

pub use imc_controller::{ImcConfig, ImcController};
pub use pfc::{PfcConfig, PfcController, MAX_HORIZON};
pub use smith_predictor::{SmithPredictor, SmithPredictorConfig};
