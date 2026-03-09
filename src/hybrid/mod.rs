//! Hybrid systems module for OxiCtl.
//!
//! Provides implementations for:
//! - [`automaton`]: Hybrid automaton with M modes and N-dimensional continuous state
//! - [`switched_lti`]: Switched linear time-invariant systems with dwell-time constraints
//! - [`piecewise_affine`]: Piecewise affine (PWA) systems and controllers
#![cfg_attr(not(feature = "std"), no_std)]

pub mod automaton;
pub mod piecewise_affine;
pub mod switched_lti;

pub use automaton::{HybridAutomaton, HybridError};
pub use piecewise_affine::{PwaController, PwaError, PwaSystem};
pub use switched_lti::{SwitchedError, SwitchedLti};
