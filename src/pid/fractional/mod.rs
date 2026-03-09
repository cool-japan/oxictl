/// Fractional-order PID control (PI^λ D^μ).
///
/// This module provides three core components:
///
/// 1. **`grunwald`** — Grünwald-Letnikov (GL) fractional derivative/integral
///    approximation via a sliding window of N samples.
/// 2. **`fopid`** — PI^λ D^μ controller built on GL operators, with
///    anti-windup and a grid-search auto-tuner.
/// 3. **`tustin_approx`** — Tustin (bilinear) IIR approximation of s^α.
pub mod fopid;
pub mod grunwald;
pub mod tustin_approx;

pub use fopid::{Fopid, FopidAutoTune, FopidAutoTuneResult, FopidConfig};
pub use grunwald::{FracDifferentiator, FracIntegrator, GrunwaldLeibniz};
pub use tustin_approx::{design_tustin_frac, TustinFrac};

/// Errors arising from fractional-order control operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FracError {
    /// Fractional order is out of range, NaN, or infinite.
    InvalidOrder,
    /// Sample time is zero, negative, or non-finite.
    InvalidSampleTime,
    /// Window or truncation order is too small (must be ≥ 2).
    WindowTooSmall,
    /// A configuration field is invalid (carries a descriptive string).
    InvalidConfig(&'static str),
    /// Auto-tuning grid search failed to find any valid parameter point.
    TuningFailed,
}

impl core::fmt::Display for FracError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FracError::InvalidOrder => write!(f, "fractional order is invalid or out of range"),
            FracError::InvalidSampleTime => write!(f, "sample time must be positive and finite"),
            FracError::WindowTooSmall => write!(f, "window/order must be at least 2"),
            FracError::InvalidConfig(msg) => write!(f, "invalid configuration: {}", msg),
            FracError::TuningFailed => write!(f, "auto-tuning failed to find a valid solution"),
        }
    }
}
