//! Fuzzy logic control module for `oxictl`.
//!
//! Provides:
//! - **Membership functions** ([`membership`]): triangular, trapezoidal,
//!   Gaussian, sigmoid, singleton, generalized bell.
//! - **Rule base** ([`rule_base`]): linguistic variables, fuzzy rules,
//!   T-norm operators, and rule firing.
//! - **Inference engines** ([`inference`]): Mamdani and Sugeno (TSK) engines.
//! - **Defuzzification** ([`defuzzify`]): CoG, MoM, BoA, LoM, SoM.
//! - **Fuzzy-PID hybrid** ([`fuzzy_pid`]): gain-scheduled PID via Sugeno engine.

pub mod defuzzify;
pub mod fuzzy_pid;
pub mod inference;
pub mod membership;
pub mod rule_base;

// ────────────────────────────────────────────────────────────────────────────
// FuzzyError
// ────────────────────────────────────────────────────────────────────────────

/// Errors produced by the fuzzy logic subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuzzyError {
    /// A construction parameter is invalid (e.g. width ≤ 0, unsorted breakpoints).
    InvalidParameter(&'static str),
    /// A fixed-capacity heapless collection is full.
    CapacityExceeded,
    /// A defuzzification denominator is zero (all membership values are zero).
    DivisionByZero,
    /// The sample slice is too short to perform the requested computation.
    InsufficientSamples,
}

impl core::fmt::Display for FuzzyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FuzzyError::InvalidParameter(msg) => write!(f, "InvalidParameter: {msg}"),
            FuzzyError::CapacityExceeded => write!(f, "CapacityExceeded"),
            FuzzyError::DivisionByZero => write!(f, "DivisionByZero"),
            FuzzyError::InsufficientSamples => write!(f, "InsufficientSamples"),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Re-exports
// ────────────────────────────────────────────────────────────────────────────

// Membership
pub use membership::{
    BellShaped, Gaussian, MembershipFn, Sigmoid, Singleton, Trapezoidal, Triangular,
};

// Rule base
pub use rule_base::{
    fnv1a_hash, Antecedent, Condition, Consequent, FuzzyRule, FuzzySet, LinguisticVar, RuleBase,
    TNorm, MAX_ANTECEDENTS,
};

// Inference
pub use inference::{
    single_condition_antecedent, MamdaniEngine, SugenoEngine, SugenoRule, MAMDANI_SAMPLE_COUNT,
};

// Defuzzification
pub use defuzzify::{
    bisector_of_area, centroid_of_gravity, largest_of_maxima, mean_of_maxima, smallest_of_maxima,
};

// Fuzzy-PID
pub use fuzzy_pid::{FuzzyPid, FuzzyPidConfig};
