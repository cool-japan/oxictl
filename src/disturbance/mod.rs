//! Disturbance estimation and rejection module.
//!
//! This module provides several classical and nonlinear disturbance observer
//! architectures suitable for real-time embedded control:
//!
//! | Sub-module | Algorithm | Key reference |
//! |------------|-----------|---------------|
//! | [`dob`]    | Q-filter Disturbance Observer | Ohnishi (1987) |
//! | [`ndob`]   | Nonlinear Disturbance Observer | Chen et al. (2000) |
//! | [`ude`]    | Uncertainty & Disturbance Estimator | Zhong & Rees (2004) |
//!
//! All implementations are:
//! - `no_std` compatible (no heap allocation beyond what the caller provides).
//! - Generic over [`crate::core::scalar::ControlScalar`] (f32 / f64).
//! - Free of `unwrap()` — all fallible operations return `Result`.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod dob;
pub mod ndob;
pub mod ude;

pub use dob::{DisturbanceObserver, DisturbanceObserverConfig, DobError};
pub use ndob::{NdobError, NonlinearDob};
pub use ude::{UdeController, UdeError};
