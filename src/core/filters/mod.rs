//! Signal processing filters for control systems.
//!
//! This module provides a collection of `no_std`-compatible digital filters:
//!
//! - **Butterworth IIR** (`butterworth`) — maximally flat IIR filters (LP, HP, BP)
//! - **Chebyshev IIR** (`chebyshev`) — equiripple passband (Type I) or stopband (Type II)
//! - **Moving average** (`moving_average`) — sliding window mean, EMA, RMS, variance
//! - **Median filter** (`median_filter`) — nonlinear impulse rejection
//! - **FIR** (`fir`) — windowed-sinc lowpass FIR filters
//!
//! All filters implement `update(&mut self, x: S) -> S` for sample-by-sample processing
//! and are generic over the scalar type `S: ControlScalar` (i.e. `f32` or `f64`).

pub mod butterworth;
pub mod chebyshev;
pub mod fir;
pub mod median_filter;
pub mod moving_average;

// ─────────────────────────────────────────────────────────────
//  FilterError
// ─────────────────────────────────────────────────────────────

/// Error type for filter design functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterError {
    /// Filter order is zero or exceeds the supported maximum.
    InvalidOrder,
    /// Cutoff or band-edge frequency is outside the valid range (0, fs/2).
    InvalidFrequency,
    /// Sample rate is zero or negative.
    InvalidSampleRate,
    /// Ripple or attenuation parameter is invalid (must be positive).
    InvalidRipple,
}

impl core::fmt::Display for FilterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidOrder => write!(f, "filter order must be between 1 and 8"),
            Self::InvalidFrequency => write!(f, "frequency must be in (0, fs/2)"),
            Self::InvalidSampleRate => write!(f, "sample rate must be positive"),
            Self::InvalidRipple => write!(f, "ripple/attenuation must be positive"),
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Re-exports
// ─────────────────────────────────────────────────────────────

pub use butterworth::{
    design_butterworth_bp, design_butterworth_hp, design_butterworth_lp, ButterworthBiquad,
    ButterworthBp, ButterworthHp, ButterworthLp,
};

pub use chebyshev::{design_chebyshev1_lp, design_chebyshev2_lp, ChebyshevI, ChebyshevII};

pub use moving_average::{ExponentialMovingAverage, MovingAverage, MovingRms, MovingVariance};

pub use median_filter::{median3, MedianFilter, MedianOf3};

pub use fir::{design_fir_lp, FirFilter, WindowType};
