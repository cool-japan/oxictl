#![cfg_attr(not(feature = "std"), no_std)]
//! Control Allocation module.
//!
//! Provides allocation strategies for over-actuated systems where M actuators
//! produce N < M control objectives (or M = N for exact allocation).
//!
//! # Sub-modules
//!
//! - [`weighted_pseudo`]: Weighted pseudo-inverse allocation with constraint
//!   handling via a single-pass re-allocation algorithm.
//! - [`prioritized`]: Priority-based cascaded allocation for multiple control
//!   tasks with ordered importance levels.
//! - [`linear_programming`]: Bounded least-squares allocation via projected
//!   gradient descent.
//!
//! # Common Error Type
//!
//! All solvers use [`AllocationError`] (re-exported from `weighted_pseudo`).

pub mod linear_programming;
pub mod prioritized;
pub mod weighted_pseudo;

pub use linear_programming::BoundedLsAllocator;
pub use prioritized::{AllocationTask, PriorityAllocator};
pub use weighted_pseudo::{AllocationError, WeightedPseudoInverse};
