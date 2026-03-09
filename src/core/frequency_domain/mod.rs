//! Frequency-domain analysis tools for discrete-time control systems.
//!
//! This module provides:
//! - **Bode plots**: magnitude and phase frequency response, gain/phase margins
//! - **Nyquist analysis**: Nyquist curve, stability criterion, distance to critical point
//! - **Sensitivity analysis**: loop-shaping, sensitivity/complementary sensitivity functions
//! - **Root locus**: closed-loop pole trajectories as a function of gain (z-plane)

pub mod bode;
pub mod nyquist;
pub mod root_locus;
pub mod sensitivity;

pub use bode::{
    compute_bode, gain_crossover_frequency, gain_margin, phase_crossover_frequency, phase_margin,
    BodeData, BodePoint,
};

pub use nyquist::{
    compute_nyquist, distance_to_critical, encirclement_count, is_stable_nyquist, NyquistData,
    NyquistPoint,
};

pub use sensitivity::{
    bandwidth, peak_sensitivity, sensitivity_crossover, Complex, LoopShaping, SensitivityData,
    SensitivityPoint,
};

pub use root_locus::{compute_root_locus, stability_region, RootLocusData, RootLocusPoint};

/// Errors arising from frequency-domain analysis operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreqError {
    /// The requested number of frequency points is too small (minimum 2).
    InsufficientPoints,
    /// The frequency range is invalid (e.g., `omega_min >= omega_max` or non-positive).
    InvalidFrequencyRange,
    /// A parameter value is outside the valid range (e.g., negative gain limit).
    InvalidParameter,
    /// The transfer function order exceeds the supported maximum.
    OrderTooLarge,
}
