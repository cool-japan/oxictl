//! Byzantine fault tolerant voting using median.
#![allow(dead_code)]

use crate::core::scalar::ControlScalar;

/// Byzantine voter: tolerates f = (N-1)/3 faults using median selection.
///
/// The median is computed by partially sorting a copy of the input array
/// (selection sort on a fixed-size array — no alloc required).
pub struct ByzantineVoter<S: ControlScalar, const N: usize> {
    _phantom: core::marker::PhantomData<S>,
}

impl<S: ControlScalar, const N: usize> ByzantineVoter<S, N> {
    /// Create a new Byzantine voter.
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }

    /// Vote on N values using median (resilient to f Byzantine faults).
    ///
    /// For an even number of channels the lower-middle element is returned.
    pub fn vote(&self, values: [S; N]) -> S {
        let mut buf = values;
        Self::sort_ascending(&mut buf);
        buf[N / 2]
    }

    /// Number of Byzantine faults this configuration can tolerate: ⌊(N-1)/3⌋.
    pub const fn fault_tolerance_level() -> usize {
        (N.saturating_sub(1)) / 3
    }

    /// Minimum channels needed to tolerate `f` Byzantine faults: 3f + 1.
    pub const fn min_channels_for_faults(f: usize) -> usize {
        3 * f + 1
    }

    /// Return `true` if any value deviates from the median by more than `threshold`.
    pub fn has_outliers(&self, values: [S; N], threshold: S) -> bool {
        let median = self.vote(values);
        for v in values {
            let diff = if v > median { v - median } else { median - v };
            if diff > threshold {
                return true;
            }
        }
        false
    }

    // Insertion sort (stable, O(N²) — fine for small const N).
    fn sort_ascending(buf: &mut [S; N]) {
        for i in 1..N {
            let key = buf[i];
            let mut j = i;
            while j > 0 && buf[j - 1] > key {
                buf[j] = buf[j - 1];
                j -= 1;
            }
            buf[j] = key;
        }
    }
}

impl<S: ControlScalar, const N: usize> Default for ByzantineVoter<S, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_five_values() {
        let voter = ByzantineVoter::<f64, 5>::new();
        // Median of [3,1,5,2,4] = 3.0
        let result = voter.vote([3.0, 1.0, 5.0, 2.0, 4.0]);
        assert!((result - 3.0).abs() < 1e-12, "median={result}");
    }

    #[test]
    fn fault_tolerance_level() {
        // N=7 → f=(7-1)/3 = 2
        assert_eq!(ByzantineVoter::<f64, 7>::fault_tolerance_level(), 2);
        // N=4 → f=(4-1)/3 = 1
        assert_eq!(ByzantineVoter::<f64, 4>::fault_tolerance_level(), 1);
        // N=1 → f=0
        assert_eq!(ByzantineVoter::<f64, 1>::fault_tolerance_level(), 0);
    }

    #[test]
    fn min_channels_for_faults() {
        // f=2 → need 3*2+1 = 7 channels
        assert_eq!(ByzantineVoter::<f64, 1>::min_channels_for_faults(2), 7);
    }

    #[test]
    fn outlier_detection() {
        let voter = ByzantineVoter::<f64, 5>::new();
        // Normal: all values close together
        assert!(!voter.has_outliers([1.0, 1.1, 0.9, 1.05, 0.95], 0.5));
        // One Byzantine fault: value of 100 is far from median ~1.0
        assert!(voter.has_outliers([1.0, 1.0, 100.0, 1.0, 1.0], 0.5));
    }

    #[test]
    fn vote_with_one_byzantine_fault_in_four_channels() {
        // N=4 → f=1, Byzantine fault-tolerant for 1 traitor
        let voter = ByzantineVoter::<f64, 4>::new();
        // Three honest values ≈ 5.0, one faulty = 100.0
        // Sorted: [5.0, 5.0, 5.0, 100.0] → median at index 2 = 5.0
        let result = voter.vote([5.0, 5.0, 100.0, 5.0]);
        assert!((result - 5.0).abs() < 1e-12, "result={result}");
    }
}
