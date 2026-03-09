//! Extended redundancy comparators for safety-critical signal voting.
//!
//! Provides:
//! - [`DualChannelComparatorExt`] — 1oo2 (one-out-of-two) voting with tolerance
//!   tracking and configurable divergence trip threshold.
//! - [`TripleModularRedundancy`] — 2oo3 voting with outlier channel identification.
//!
//! All structures are `no_std` compatible and perform no heap allocation.

#![allow(dead_code)]

use crate::core::scalar::ControlScalar;

// ─────────────────────────────────────────────────────────────────────────────
// ComparatorConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration shared by both redundancy comparators.
///
/// Two tolerances are provided to handle measurements at different scales:
/// - `absolute_tolerance` — maximum permitted |ch_a − ch_b|.
/// - `relative_tolerance` — maximum permitted |ch_a − ch_b| / |ch_a + ch_b| * 2,
///   evaluated only when both channels are non-zero.
/// - `max_divergence_count` — number of consecutive divergent comparisons
///   required before a fault is declared.
#[derive(Debug, Clone, Copy)]
pub struct ComparatorConfig<S: ControlScalar> {
    /// Absolute tolerance (engineering-unit difference).
    pub absolute_tolerance: S,
    /// Relative tolerance (dimensionless fraction of the signal magnitude).
    pub relative_tolerance: S,
    /// Number of consecutive divergences required to declare a fault.
    pub max_divergence_count: u32,
}

impl<S: ControlScalar> ComparatorConfig<S> {
    /// Construct a new configuration.
    ///
    /// Panics in debug mode if `max_divergence_count` is zero; in production
    /// a zero count is treated as one (immediate trip on first divergence).
    pub fn new(absolute_tolerance: S, relative_tolerance: S, max_divergence_count: u32) -> Self {
        Self {
            absolute_tolerance,
            relative_tolerance,
            max_divergence_count: max_divergence_count.max(1),
        }
    }

    /// Returns `true` if `a` and `b` agree within both absolute and relative
    /// tolerances.
    ///
    /// Either tolerance alone is sufficient to declare agreement.
    pub fn within_tolerance(&self, a: S, b: S) -> bool {
        let abs_diff = (a - b).abs();
        if abs_diff <= self.absolute_tolerance {
            return true;
        }
        // Relative check — guard against division by near-zero
        let magnitude = (a.abs() + b.abs()) * S::HALF;
        if magnitude > S::ZERO {
            let rel_diff = abs_diff / magnitude;
            if rel_diff <= self.relative_tolerance {
                return true;
            }
        }
        false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompareResult
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a single comparison cycle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompareResult<S: Copy> {
    /// Both channels agree; the inner value is the voted/average output.
    Agree(S),
    /// Channels diverge but the trip threshold has not yet been reached.
    /// The values of channel A and channel B are included for logging.
    Diverged(S, S),
    /// Fault declared: the comparator has tripped.
    /// `channel` is 0 = A, 1 = B, 2 = unknown.
    Faulty(u8),
}

// ─────────────────────────────────────────────────────────────────────────────
// DualChannelComparatorExt  (1oo2)
// ─────────────────────────────────────────────────────────────────────────────

/// 1oo2 (one-out-of-two) comparator for safety-critical redundant channels.
///
/// Accepts two independent channel readings and produces an agreed output or
/// a fault signal if the channels diverge beyond configured tolerances for
/// more than `max_divergence_count` consecutive comparisons.
///
/// Once tripped, the comparator stays faulted until explicitly reset.
///
/// # Type parameter
/// `S` — any [`ControlScalar`] (typically `f64` for safety calculations).
#[derive(Debug, Clone, Copy)]
pub struct DualChannelComparatorExt<S: ControlScalar> {
    config: ComparatorConfig<S>,
    /// Running count of consecutive divergent comparisons.
    divergence_count: u32,
    /// Whether a fault has been declared.
    tripped: bool,
    /// Index of the suspected faulty channel (0=A, 1=B, 2=unknown).
    suspect_channel: u8,
}

impl<S: ControlScalar> DualChannelComparatorExt<S> {
    /// Create a new 1oo2 comparator with the given configuration.
    pub fn new(config: ComparatorConfig<S>) -> Self {
        Self {
            config,
            divergence_count: 0,
            tripped: false,
            suspect_channel: 2,
        }
    }

    /// Compare channel A and channel B.
    ///
    /// Returns:
    /// - [`CompareResult::Agree`] — channels are within tolerance; the
    ///   output value is the average of the two channels.
    /// - [`CompareResult::Diverged`] — channels differ but trip count not yet
    ///   reached; the comparator remains un-tripped.
    /// - [`CompareResult::Faulty`] — trip count reached or already tripped;
    ///   the suspect channel index is included.
    pub fn compare(&mut self, ch_a: S, ch_b: S) -> CompareResult<S> {
        if self.tripped {
            return CompareResult::Faulty(self.suspect_channel);
        }

        if self.config.within_tolerance(ch_a, ch_b) {
            self.divergence_count = 0;
            let voted = (ch_a + ch_b) * S::HALF;
            CompareResult::Agree(voted)
        } else {
            self.divergence_count += 1;
            if self.divergence_count >= self.config.max_divergence_count {
                self.tripped = true;
                // Without a reference we cannot identify the faulty channel
                self.suspect_channel = 2;
                CompareResult::Faulty(self.suspect_channel)
            } else {
                CompareResult::Diverged(ch_a, ch_b)
            }
        }
    }

    /// Compare channel A and channel B with an external reference value.
    ///
    /// If the comparator trips, the channel that is *farther* from `reference`
    /// is identified as suspect.
    pub fn compare_with_ref(&mut self, ch_a: S, ch_b: S, reference: S) -> CompareResult<S> {
        let result = self.compare(ch_a, ch_b);
        if self.tripped {
            let da = (ch_a - reference).abs();
            let db = (ch_b - reference).abs();
            self.suspect_channel = if da > db { 0 } else { 1 };
            CompareResult::Faulty(self.suspect_channel)
        } else {
            result
        }
    }

    /// Whether the comparator has tripped (fault declared).
    pub fn is_tripped(&self) -> bool {
        self.tripped
    }

    /// Current consecutive divergence count.
    pub fn divergence_count(&self) -> u32 {
        self.divergence_count
    }

    /// Index of the suspected faulty channel (0=A, 1=B, 2=unknown).
    pub fn suspect_channel(&self) -> u8 {
        self.suspect_channel
    }

    /// Reset the comparator, clearing the trip flag and counters.
    pub fn reset(&mut self) {
        self.divergence_count = 0;
        self.tripped = false;
        self.suspect_channel = 2;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VoteResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a 2oo3 majority vote.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoteResult<S: Copy> {
    /// The majority-voted value (median of the three channels).
    pub voted_value: S,
    /// Index of the channel that differed from the majority, if any.
    /// - `None` — all three channels agree within tolerance.
    /// - `Some(0)` — channel A is the outlier.
    /// - `Some(1)` — channel B is the outlier.
    /// - `Some(2)` — channel C is the outlier.
    pub outlier_channel: Option<u8>,
}

// ─────────────────────────────────────────────────────────────────────────────
// TripleModularRedundancy  (2oo3)
// ─────────────────────────────────────────────────────────────────────────────

/// 2oo3 (two-out-of-three) Triple Modular Redundancy voter.
///
/// Compares three independent channel readings and returns the majority-voted
/// value.  The channel whose reading deviates from the majority is identified
/// as the outlier.
///
/// Consecutive outlier events on a single channel are counted; if they exceed
/// `config.max_divergence_count`, that channel is declared permanently faulty
/// until `reset()` is called.
#[derive(Debug, Clone, Copy)]
pub struct TripleModularRedundancy<S: ControlScalar> {
    config: ComparatorConfig<S>,
    /// Consecutive outlier count per channel [A, B, C].
    outlier_counts: [u32; 3],
    /// Whether each channel has been declared faulty.
    channel_faulted: [bool; 3],
}

impl<S: ControlScalar> TripleModularRedundancy<S> {
    /// Create a new TMR voter with the given configuration.
    pub fn new(config: ComparatorConfig<S>) -> Self {
        Self {
            config,
            outlier_counts: [0; 3],
            channel_faulted: [false; 3],
        }
    }

    /// Perform a 2oo3 majority vote over channels A, B, and C.
    ///
    /// The voted value is the median of the three readings.  The channel
    /// farthest from the median (if it exceeds the configured tolerance) is
    /// flagged as the outlier.
    ///
    /// Returns `Err` if two or more channels have been permanently faulted
    /// (majority cannot be established).
    pub fn vote(&mut self, ch_a: S, ch_b: S, ch_c: S) -> Result<VoteResult<S>, TmrError> {
        let faulted_count = self.channel_faulted.iter().filter(|&&f| f).count();
        if faulted_count >= 2 {
            return Err(TmrError::MajorityLost);
        }

        let values = [ch_a, ch_b, ch_c];
        let voted = median3(ch_a, ch_b, ch_c);

        // Identify the channel with the largest deviation from the median
        let deviations = [
            (ch_a - voted).abs(),
            (ch_b - voted).abs(),
            (ch_c - voted).abs(),
        ];

        // Find the channel with the maximum deviation
        let mut max_dev = S::ZERO;
        let mut max_idx: u8 = 0;
        for (i, &dev) in deviations.iter().enumerate() {
            if dev > max_dev {
                max_dev = dev;
                max_idx = i as u8;
            }
        }

        let outlier_channel: Option<u8> = if !self
            .config
            .within_tolerance(values[max_idx as usize], voted)
        {
            // Update outlier count for the suspect channel
            let idx = max_idx as usize;
            self.outlier_counts[idx] += 1;
            // Reset counts for the other two channels
            for j in 0..3usize {
                if j != idx {
                    self.outlier_counts[j] = 0;
                }
            }
            if self.outlier_counts[idx] >= self.config.max_divergence_count {
                self.channel_faulted[idx] = true;
            }
            Some(max_idx)
        } else {
            // All three channels agree: reset all outlier counts
            self.outlier_counts = [0; 3];
            None
        };

        Ok(VoteResult {
            voted_value: voted,
            outlier_channel,
        })
    }

    /// Whether a specific channel index (0=A, 1=B, 2=C) has been declared faulted.
    pub fn is_channel_faulted(&self, channel: u8) -> bool {
        let idx = channel as usize;
        if idx < 3 {
            self.channel_faulted[idx]
        } else {
            false
        }
    }

    /// Fault flags for all three channels [A, B, C].
    pub fn channel_faults(&self) -> [bool; 3] {
        self.channel_faulted
    }

    /// Consecutive outlier counts for all three channels [A, B, C].
    pub fn outlier_counts(&self) -> [u32; 3] {
        self.outlier_counts
    }

    /// Reset all state (clear faults, counters).
    pub fn reset(&mut self) {
        self.outlier_counts = [0; 3];
        self.channel_faulted = [false; 3];
    }
}

/// Errors from TMR voting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TmrError {
    /// Two or more channels are permanently faulted; majority cannot be established.
    MajorityLost,
}

impl core::fmt::Display for TmrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TmrError::MajorityLost => {
                write!(f, "TMR: majority lost — two or more channels faulted")
            }
        }
    }
}

/// Compute the median of three values without sorting.
fn median3<S: ControlScalar>(a: S, b: S, c: S) -> S {
    // Branchless median: max(min(a,b), min(max(a,b),c))
    let lo = if a < b { a } else { b };
    let hi = if a > b { a } else { b };
    let mid = if hi < c { hi } else { c };
    if lo > mid {
        lo
    } else {
        mid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ComparatorConfig::within_tolerance ──────────────────────────────────

    #[test]
    fn within_absolute_tolerance() {
        // abs_tol = 0.1, rel_tol = 0.001 (very tight)
        // diff = 0.05 < 0.1 → agrees by absolute tolerance
        let cfg = ComparatorConfig::new(0.1_f64, 0.001_f64, 3);
        assert!(cfg.within_tolerance(10.0, 10.05));
        // diff = 0.2 > 0.1; rel = 0.2/10.1 ≈ 0.0198 > 0.001 → both fail → diverged
        assert!(!cfg.within_tolerance(10.0, 10.2));
    }

    #[test]
    fn within_relative_tolerance() {
        // abs_tol = 0.05, rel_tol = 0.05
        // abs diff = 0.2 > 0.05 → absolute fails
        // magnitude = 10.1, rel = 0.2/10.1 ≈ 0.0198 < 0.05 → relative passes
        let cfg = ComparatorConfig::new(0.05_f64, 0.05_f64, 3);
        assert!(cfg.within_tolerance(10.0, 10.2));
        // Very large difference: rel = 10.0/15.0 ≈ 0.667 > 0.05 → both fail
        assert!(!cfg.within_tolerance(10.0, 20.0));
    }

    // ── DualChannelComparatorExt ────────────────────────────────────────────

    #[test]
    fn dual_channels_agree() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 3);
        let mut cmp = DualChannelComparatorExt::new(cfg);
        let result = cmp.compare(5.0, 5.05);
        assert!(matches!(result, CompareResult::Agree(_)));
        if let CompareResult::Agree(v) = result {
            assert!((v - 5.025).abs() < 1e-10);
        }
        assert!(!cmp.is_tripped());
    }

    #[test]
    fn dual_channels_diverge_below_threshold() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 3);
        let mut cmp = DualChannelComparatorExt::new(cfg);
        // Two divergences — trip threshold is 3
        let r1 = cmp.compare(0.0, 10.0);
        let r2 = cmp.compare(0.0, 10.0);
        assert!(matches!(r1, CompareResult::Diverged(_, _)));
        assert!(matches!(r2, CompareResult::Diverged(_, _)));
        assert!(!cmp.is_tripped());
        assert_eq!(cmp.divergence_count(), 2);
    }

    #[test]
    fn dual_channels_trip_after_threshold() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 3);
        let mut cmp = DualChannelComparatorExt::new(cfg);
        for _ in 0..3 {
            cmp.compare(0.0, 10.0);
        }
        assert!(cmp.is_tripped());
        // Subsequent calls return Faulty
        assert!(matches!(cmp.compare(0.0, 10.0), CompareResult::Faulty(_)));
    }

    #[test]
    fn dual_channels_trip_identifies_suspect_with_ref() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 1);
        let mut cmp = DualChannelComparatorExt::new(cfg);
        // ch_a = 10.0 (correct), ch_b = 20.0 (faulty), ref = 10.0
        let result = cmp.compare_with_ref(10.0_f64, 20.0, 10.0);
        assert!(cmp.is_tripped());
        assert_eq!(cmp.suspect_channel(), 1); // ch_b is farther from reference
        assert!(matches!(result, CompareResult::Faulty(1)));
    }

    #[test]
    fn dual_reset_clears_state() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 1);
        let mut cmp = DualChannelComparatorExt::new(cfg);
        cmp.compare(0.0, 10.0); // trip immediately
        assert!(cmp.is_tripped());
        cmp.reset();
        assert!(!cmp.is_tripped());
        assert_eq!(cmp.divergence_count(), 0);
        assert!(matches!(cmp.compare(5.0, 5.0), CompareResult::Agree(_)));
    }

    #[test]
    fn divergence_count_resets_on_agreement() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 5);
        let mut cmp = DualChannelComparatorExt::new(cfg);
        cmp.compare(0.0, 10.0); // diverge: count=1
        cmp.compare(0.0, 10.0); // diverge: count=2
        cmp.compare(5.0, 5.0); // agree:   count=0
        assert_eq!(cmp.divergence_count(), 0);
        assert!(!cmp.is_tripped());
    }

    // ── TripleModularRedundancy ─────────────────────────────────────────────

    #[test]
    fn tmr_all_channels_agree() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 3);
        let mut tmr = TripleModularRedundancy::new(cfg);
        let result = tmr.vote(5.0, 5.05, 4.98).unwrap();
        assert!(result.outlier_channel.is_none());
        // Median of [5.0, 5.05, 4.98] = 5.0
        assert!((result.voted_value - 5.0).abs() < 1e-9);
    }

    #[test]
    fn tmr_identifies_outlier_channel() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 3);
        let mut tmr = TripleModularRedundancy::new(cfg);
        // Channel C = 100.0 is the outlier
        let result = tmr.vote(5.0_f64, 5.05, 100.0).unwrap();
        assert_eq!(result.outlier_channel, Some(2)); // channel C (index 2)
                                                     // Median = 5.05
        assert!((result.voted_value - 5.05).abs() < 1e-9);
    }

    #[test]
    fn tmr_outlier_a_identified() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 3);
        let mut tmr = TripleModularRedundancy::new(cfg);
        // Channel A = 100.0 is the outlier
        let result = tmr.vote(100.0_f64, 5.0, 5.05).unwrap();
        assert_eq!(result.outlier_channel, Some(0)); // channel A
    }

    #[test]
    fn tmr_channel_faulted_after_repeated_outliers() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 3);
        let mut tmr = TripleModularRedundancy::new(cfg);
        // Channel C consistently outlying
        for _ in 0..3 {
            let _ = tmr.vote(5.0_f64, 5.05, 100.0);
        }
        assert!(tmr.is_channel_faulted(2));
        assert!(!tmr.is_channel_faulted(0));
        assert!(!tmr.is_channel_faulted(1));
    }

    #[test]
    fn tmr_majority_lost_when_two_channels_faulted() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 1);
        let mut tmr = TripleModularRedundancy::new(cfg);
        // Fault channel A
        let _ = tmr.vote(100.0_f64, 5.0, 5.0);
        assert!(tmr.is_channel_faulted(0));
        // Fault channel B by making it the outlier
        let _ = tmr.vote(5.0_f64, 100.0, 5.0);
        assert!(tmr.is_channel_faulted(1));
        // Now two channels faulted — majority lost
        let result = tmr.vote(5.0_f64, 5.0, 5.0);
        assert_eq!(result, Err(TmrError::MajorityLost));
    }

    #[test]
    fn tmr_reset_clears_faults() {
        let cfg = ComparatorConfig::new(0.1_f64, 0.01_f64, 1);
        let mut tmr = TripleModularRedundancy::new(cfg);
        let _ = tmr.vote(100.0_f64, 5.0, 5.0); // fault A
        assert!(tmr.is_channel_faulted(0));
        tmr.reset();
        assert!(!tmr.is_channel_faulted(0));
        assert_eq!(tmr.outlier_counts(), [0; 3]);
    }

    #[test]
    fn median3_correctness() {
        assert_eq!(median3(1.0_f64, 2.0, 3.0), 2.0);
        assert_eq!(median3(3.0_f64, 1.0, 2.0), 2.0);
        assert_eq!(median3(2.0_f64, 3.0, 1.0), 2.0);
        assert_eq!(median3(5.0_f64, 5.0, 5.0), 5.0);
        assert_eq!(median3(1.0_f64, 1.0, 100.0), 1.0);
    }
}
