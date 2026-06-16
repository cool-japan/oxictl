//! Fixed-point arithmetic support for PID-style control on MCUs without FPU.
//!
//! Provides Q-format newtype wrappers around [`fixed::FixedI32`] types.
//! All types implement [`crate::core::scalar::PidScalar`], enabling use with
//! `pid::standard::Pid` and related algorithms without floating-point hardware.
//!
//! # Scope
//! Only algorithms that do NOT require transcendental functions (sin, cos, exp,
//! sqrt, etc.) are compatible. Currently: `Pid`, `DerivativeFilter`, `AntiWindupMethod`.
//! Algorithms using `ControlScalar` (KF, EKF, Butterworth, FOC, flatness, etc.)
//! are NOT compatible with fixed-point types.
//!
//! # Available Q-Format Types
//! - [`Q15_16`]: ±32768 range, ~0.0000153 resolution (general-purpose PID)
//! - [`Q3_29`]: ±4 range, ~1.86e-9 resolution (high-precision normalized signals)
//! - [`Q7_24`]: ±128 range, ~5.96e-8 resolution (medium-range precision)
//!
//! Note: `Q1_31` (I1F31) is excluded because the signed fixed-point type with
//! 1 integer bit cannot represent `ONE` (range is [-1, 1)), which would corrupt
//! the multiplicative identity required by `PidScalar`.
//!
//! # Example
//! ```rust,ignore
//! use oxictl::core::fixed_point::{Q15_16, fixed_from_f32_saturating};
//! ```

pub mod convert;
pub mod error;
pub mod ops;
pub mod scalar_impl;
pub mod types;

#[cfg(test)]
mod tests;

pub use convert::{fixed_from_f32_saturating, fixed_to_f32};
pub use error::FixedError;
pub use types::{Q15_16, Q3_29, Q7_24};
