use crate::core::scalar::ControlScalar;

/// Dual-channel comparator for safety-critical redundant systems.
///
/// Compares two independently computed channels (e.g., two CPUs running
/// the same algorithm). A significant mismatch indicates a channel fault.
///
/// Common in SIL-2/SIL-3 architectures (ISO 13849, IEC 62061).
#[derive(Debug, Clone, Copy)]
pub struct DualChannelComparator<S: ControlScalar> {
    /// Maximum allowed absolute difference between channels.
    pub tolerance: S,
    /// Number of consecutive mismatches before tripping.
    pub trip_count: u32,
    consecutive_faults: u32,
    tripped: bool,
    /// Which channel is diagnosed as faulty (0 = A, 1 = B, 2 = unknown).
    suspect_channel: u8,
}

impl<S: ControlScalar> DualChannelComparator<S> {
    pub fn new(tolerance: S, trip_count: u32) -> Self {
        Self {
            tolerance,
            trip_count,
            consecutive_faults: 0,
            tripped: false,
            suspect_channel: 2,
        }
    }

    /// Compare two channel outputs. Returns `true` if channels agree.
    ///
    /// When a mismatch occurs `trip_count` times consecutively, the
    /// comparator trips and marks a safety fault.
    pub fn check(&mut self, channel_a: S, channel_b: S) -> bool {
        if self.tripped {
            return false;
        }
        let diff = (channel_a - channel_b).abs();
        if diff > self.tolerance {
            self.consecutive_faults += 1;
            if self.consecutive_faults >= self.trip_count {
                self.tripped = true;
                // Cannot determine which channel is faulty without reference
                self.suspect_channel = 2;
            }
            return !self.tripped;
        }
        self.consecutive_faults = 0;
        true
    }

    /// Check with an external reference to identify the faulty channel.
    ///
    /// The channel closest to `reference` is considered healthy.
    pub fn check_with_ref(&mut self, channel_a: S, channel_b: S, reference: S) -> bool {
        let ok = self.check(channel_a, channel_b);
        if !ok && self.tripped {
            let da = (channel_a - reference).abs();
            let db = (channel_b - reference).abs();
            self.suspect_channel = if da > db { 0 } else { 1 };
        }
        ok
    }

    pub fn is_tripped(&self) -> bool {
        self.tripped
    }

    /// Index of the suspect channel (0=A, 1=B, 2=unknown).
    pub fn suspect_channel(&self) -> u8 {
        self.suspect_channel
    }

    pub fn reset(&mut self) {
        self.consecutive_faults = 0;
        self.tripped = false;
        self.suspect_channel = 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channels_agree_returns_true() {
        let mut cmp = DualChannelComparator::new(0.1_f64, 3);
        assert!(cmp.check(10.0, 10.05));
        assert!(!cmp.is_tripped());
    }

    #[test]
    fn mismatch_trips_after_count() {
        let mut cmp = DualChannelComparator::new(0.1_f64, 3);
        for _ in 0..3 {
            cmp.check(10.0, 15.0);
        }
        assert!(cmp.is_tripped());
    }

    #[test]
    fn intermittent_mismatch_resets_counter() {
        let mut cmp = DualChannelComparator::new(0.1_f64, 3);
        cmp.check(10.0, 15.0); // mismatch: count=1
        cmp.check(10.0, 10.05); // ok: count=0
        cmp.check(10.0, 15.0); // mismatch: count=1
        cmp.check(10.0, 15.0); // mismatch: count=2
        assert!(!cmp.is_tripped()); // need 3
        cmp.check(10.0, 15.0); // mismatch: count=3 → trip
        assert!(cmp.is_tripped());
    }

    #[test]
    fn channel_id_from_reference() {
        let mut cmp = DualChannelComparator::new(0.1_f64, 1);
        // Channel A = 10.0 (correct), Channel B = 20.0 (faulty), ref = 10.0
        cmp.check_with_ref(10.0_f64, 20.0, 10.0);
        assert!(cmp.is_tripped());
        // Channel B (index 1) is farther from reference
        assert_eq!(cmp.suspect_channel(), 1);
    }

    #[test]
    fn reset_clears_trip() {
        let mut cmp = DualChannelComparator::new(0.1_f64, 1);
        cmp.check(0.0_f64, 10.0);
        assert!(cmp.is_tripped());
        cmp.reset();
        assert!(!cmp.is_tripped());
        assert!(cmp.check(1.0_f64, 1.05));
    }
}
