#![cfg_attr(not(feature = "std"), no_std)]

//! Data-driven control design methods.
//!
//! Provides algorithms that tune controllers directly from experimental
//! input/output data, without requiring an explicit plant model:
//!
//! - [`vrft`]: Virtual Reference Feedback Tuning (VRFT) — Campi & Savaresi 2002.
//! - [`correlation_tuning`]: Correlation-Based Tuning (CbT) — Karimi et al.
//! - [`frit`]: Fictitious Reference Iterative Tuning (FRIT) — Soma et al.

pub mod correlation_tuning;
pub mod frit;
pub mod vrft;

pub use correlation_tuning::CorrelationTuner;
pub use frit::FritTuner;
pub use vrft::{DataDrivenError, VrftPid};
