#![cfg_attr(not(feature = "std"), no_std)]
//! Anti-Windup control structures.
//!
//! This module provides several anti-windup (AW) strategies for controllers
//! operating with actuator saturation:
//!
//! - [`aw_compensator`]: General linear AW compensator (Teel-Praly) and a
//!   simple scalar PI with AW correction.
//! - [`conditioning_technique`]: Back-calculation (Hanus conditioning) PI and
//!   tracking-mode PID anti-windup.
//! - [`observer_aw`]: Observer-based AW where the observer uses the actual
//!   saturated input `v` instead of the commanded input `u_lin`.

pub mod aw_compensator;
pub mod conditioning_technique;
pub mod observer_aw;

pub use aw_compensator::{AntiWindupError, LinearAntiWindup, SimpleAntiWindup};
pub use conditioning_technique::{ConditioningController, TrackingAntiWindup};
pub use observer_aw::ObserverAntiWindup;
