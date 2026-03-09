//! Repetitive control, 2-DOF control, and feedforward controllers.
//!
//! This module provides advanced control structures for periodic disturbance
//! rejection and decoupled reference tracking:
//!
//! - [`RepetitiveController`]: Plug-in repetitive controller for periodic
//!   disturbance rejection based on the internal model principle
//! - [`ModifiedRepetitiveController`]: Modified repetitive controller with
//!   zero-phase 3-tap FIR robustness filter
//! - [`TwoDofController`]: ISA standard 2-DOF PID controller separating
//!   reference tracking from disturbance rejection
//! - [`ReferencePrefilter`]: First-order reference prefilter for step smoothing
//! - [`InversionFeedforward`]: Dynamic inversion feedforward for first-order plants
//! - [`PolynomialFeedforward`]: Velocity/acceleration feedforward for smooth trajectories

pub mod feedforward;
pub mod repetitive_controller;
pub mod two_dof_controller;

pub use feedforward::{FeedforwardError, InversionFeedforward, PolynomialFeedforward};
pub use repetitive_controller::{
    ModifiedRepetitiveController, RepetitiveController, RepetitiveError,
};
pub use two_dof_controller::{ReferencePrefilter, TwoDofController, TwoDofError};
