//! Fault Detection & Isolation (FDI)
//!
//! This module provides three complementary FDI techniques:
//!
//! - [`parity_space`]: Parity-space residual generation using an open-loop
//!   model predictor. Suitable for discrete-time systems with known model.
//!
//! - [`observer_fdi`]: Luenberger observer-based residual generator with
//!   per-channel isolation. Enables identification of which output is faulty.
//!
//! - [`hypothesis_test`]: Statistical decision logic — χ² test and SPRT —
//!   applied to FDI residuals for principled fault declaration.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod hypothesis_test;
pub mod observer_fdi;
pub mod parity_space;

pub use hypothesis_test::{ChiSquareTest, Sprt, SprtDecision};
pub use observer_fdi::{FaultIsolationResult, ObserverFdi};
pub use parity_space::{FaultStatus, FdiError, ParitySpaceDetector};
