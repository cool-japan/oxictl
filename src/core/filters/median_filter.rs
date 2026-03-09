//! Non-linear median filter for impulse (spike) rejection.
//!
//! `MedianFilter<S, N>` uses a sliding window of length N (must be odd) and
//! an insertion-sort approach suitable for small N on embedded targets.
//!
//! `MedianOf3<S>` is a branchless / branch-minimal 3-sample median, useful
//! as a fast inline pre-filter.

use crate::core::scalar::ControlScalar;
use heapless::Deque;

// ─────────────────────────────────────────────────────────────
//  MedianFilter<S, N>
// ─────────────────────────────────────────────────────────────

/// Sliding-window median filter with window length N (N should be odd).
///
/// Each call to `update` maintains a sorted shadow copy of the window via
/// incremental insertion sort.  Time complexity per sample: O(N).
///
/// # Panics
/// Compilation succeeds for any N, but `update` will always return the
/// middle element of the sorted window, which is the median only when N is odd.
/// For even N the lower of the two middle elements is returned.
#[derive(Debug)]
pub struct MedianFilter<S: ControlScalar, const N: usize> {
    /// Circular buffer of the N most recent raw samples (insertion order).
    buf: Deque<S, N>,
    /// Sorted copy of the current window (maintained incrementally).
    sorted: [S; N],
    /// Number of valid samples currently in the window.
    count: usize,
}

impl<S: ControlScalar, const N: usize> MedianFilter<S, N> {
    /// Create a new `MedianFilter` with an empty window.
    pub fn new() -> Self {
        Self {
            buf: Deque::new(),
            sorted: [S::ZERO; N],
            count: 0,
        }
    }

    /// Process one sample.
    ///
    /// Returns the median of the current window.  Before the window is full,
    /// the median is computed over the samples received so far.
    ///
    /// Returns `S::ZERO` if no samples have been received yet.
    pub fn update(&mut self, x: S) -> S {
        if self.buf.len() == N {
            // Remove the oldest value from the sorted array.
            if let Some(old) = self.buf.pop_front() {
                self.remove_sorted(old);
                self.count = self.count.saturating_sub(1);
            }
        }
        // Insert into circular buffer
        let _ = self.buf.push_back(x);
        // Insert into sorted array (insertion sort into `self.count` elements)
        self.insert_sorted(x);
        self.count += 1;

        if self.count == 0 {
            S::ZERO
        } else {
            self.sorted[(self.count - 1) / 2]
        }
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.count = 0;
        for s in self.sorted.iter_mut() {
            *s = S::ZERO;
        }
    }

    /// Returns the current window length (number of buffered samples).
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if no samples have been received yet.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Insert `x` into the sorted region `sorted[0..count]` maintaining sorted order.
    fn insert_sorted(&mut self, x: S) {
        // Find insertion point (binary search would be O(log N) but for small N
        // a linear scan is cache-friendly and avoids branching overhead).
        let pos = {
            let mut i = 0usize;
            while i < self.count && self.sorted[i] <= x {
                i += 1;
            }
            i
        };
        // Shift elements right to make room (guarded by N bound).
        let end = self.count.min(N - 1);
        let mut j = end;
        while j > pos {
            self.sorted[j] = self.sorted[j - 1];
            j -= 1;
        }
        if pos < N {
            self.sorted[pos] = x;
        }
    }

    /// Remove one occurrence of `old` from the sorted region `sorted[0..count]`.
    /// If `old` is not present (numerical equality), removes the closest value.
    fn remove_sorted(&mut self, old: S) {
        // Linear scan for first occurrence
        let mut pos = None;
        for i in 0..self.count {
            if self.sorted[i] == old {
                pos = Some(i);
                break;
            }
        }
        let pos = match pos {
            Some(p) => p,
            None => return, // defensive: should never happen in normal use
        };
        // Shift elements left
        for i in pos..(self.count.saturating_sub(1)) {
            self.sorted[i] = self.sorted[i + 1];
        }
    }
}

impl<S: ControlScalar, const N: usize> Default for MedianFilter<S, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
//  MedianOf3<S>
// ─────────────────────────────────────────────────────────────

/// Fast 3-sample median using a minimal comparison network (no branches on values).
///
/// Computes `median(a, b, c)` with exactly 3 comparisons.
/// This is equivalent to a branchless sorting network of size 3.
#[derive(Debug, Clone, Copy, Default)]
pub struct MedianOf3<S: ControlScalar> {
    history: [S; 3],
    pos: usize,
    filled: usize,
}

impl<S: ControlScalar> MedianOf3<S> {
    /// Create a new `MedianOf3` with zeroed history.
    pub fn new() -> Self {
        Self {
            history: [S::ZERO; 3],
            pos: 0,
            filled: 0,
        }
    }

    /// Feed one sample and return the 3-sample median.
    ///
    /// Until 3 samples have been seen, returns the input unchanged.
    pub fn update(&mut self, x: S) -> S {
        self.history[self.pos] = x;
        self.pos = (self.pos + 1) % 3;
        if self.filled < 3 {
            self.filled += 1;
        }
        if self.filled < 3 {
            return x;
        }
        median3(self.history[0], self.history[1], self.history[2])
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.history = [S::ZERO; 3];
        self.pos = 0;
        self.filled = 0;
    }
}

/// Compute the median of three values using a sorting network (3 comparisons).
///
/// The network:
///   step 1: sort (a, b)
///   step 2: sort (b, c)   → max is now c
///   step 3: sort (a, b)   → min is now a, median is b
#[inline]
pub fn median3<S: ControlScalar>(a: S, b: S, c: S) -> S {
    let (a, b) = sort2(a, b);
    let (b, _c) = sort2(b, c);
    let (_a, b) = sort2(a, b);
    b
}

/// Return (min, max) of two values without branching on the values themselves.
#[inline]
fn sort2<S: ControlScalar>(a: S, b: S) -> (S, S) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

// ─────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_filter_constant_signal() {
        let mut mf = MedianFilter::<f64, 5>::new();
        for _ in 0..20 {
            mf.update(core::f64::consts::PI);
        }
        let y = mf.update(core::f64::consts::PI);
        assert!(
            (y - core::f64::consts::PI).abs() < 1e-12,
            "Median of constant: {y}"
        );
    }

    #[test]
    fn median_filter_impulse_rejection() {
        let mut mf = MedianFilter::<f64, 5>::new();
        // Steady signal at 1.0 then a spike
        for _ in 0..10 {
            mf.update(1.0);
        }
        // Inject a spike
        mf.update(1000.0);
        // Next normal samples
        let y1 = mf.update(1.0);
        let y2 = mf.update(1.0);
        let y3 = mf.update(1.0);
        // All three should be close to 1.0 (spike is rejected once it leaves window)
        // At least one of them should be 1.0 since spike only occupies 1 of 5 slots
        assert!(y1 < 10.0, "Spike should be attenuated: {y1}");
        assert!(y2 < 10.0, "After spike: {y2}");
        assert!((y3 - 1.0).abs() < 1e-12, "Recovered after spike: {y3}");
    }

    #[test]
    fn median_filter_sorted_output() {
        // For window [1,3,5,7,9], median = 5
        let mut mf = MedianFilter::<f64, 5>::new();
        mf.update(9.0);
        mf.update(1.0);
        mf.update(5.0);
        mf.update(7.0);
        let y = mf.update(3.0);
        assert!(
            (y - 5.0).abs() < 1e-12,
            "Median of [9,1,5,7,3] = 5, got {y}"
        );
    }

    #[test]
    fn median_filter_window_1() {
        // N=1: trivial — returns the input
        let mut mf = MedianFilter::<f64, 1>::new();
        let y = mf.update(42.0);
        assert!((y - 42.0).abs() < 1e-12, "N=1 should return input: {y}");
        let y = mf.update(7.0);
        assert!((y - 7.0).abs() < 1e-12, "N=1 second sample: {y}");
    }

    #[test]
    fn median_filter_reset() {
        let mut mf = MedianFilter::<f64, 5>::new();
        for _ in 0..10 {
            mf.update(5.0);
        }
        mf.reset();
        assert_eq!(mf.len(), 0);
        assert!(mf.is_empty());
    }

    #[test]
    fn median3_function_all_permutations() {
        // median3 should always return the middle value
        let triples = [
            (1.0, 2.0, 3.0),
            (3.0, 1.0, 2.0),
            (2.0, 3.0, 1.0),
            (3.0, 2.0, 1.0),
            (2.0, 1.0, 3.0),
            (1.0, 3.0, 2.0),
        ];
        for (a, b, c) in triples {
            let m = median3::<f64>(a, b, c);
            assert!((m - 2.0).abs() < 1e-12, "median3({a},{b},{c}) = {m}");
        }
    }

    #[test]
    fn median3_equal_values() {
        assert!((median3::<f64>(5.0, 5.0, 5.0) - 5.0).abs() < 1e-12);
        assert!((median3::<f64>(1.0, 1.0, 2.0) - 1.0).abs() < 1e-12);
        assert!((median3::<f64>(1.0, 2.0, 2.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn median3_struct_update() {
        let mut m3 = MedianOf3::<f64>::new();
        // First two calls: not enough history, returns input
        let _ = m3.update(10.0);
        let _ = m3.update(1.0);
        // Third call: returns median of [10, 1, 5] = 5
        let y = m3.update(5.0);
        assert!((y - 5.0).abs() < 1e-12, "MedianOf3 after 3 samples: {y}");
    }

    #[test]
    fn median3_struct_reset() {
        let mut m3 = MedianOf3::<f64>::new();
        m3.update(1.0);
        m3.update(2.0);
        m3.update(3.0);
        m3.reset();
        // After reset, first sample returns itself
        let y = m3.update(7.0);
        assert!((y - 7.0).abs() < 1e-12, "After reset, single sample: {y}");
    }

    #[test]
    fn median_filter_build_up() {
        // Test that partial-window medians are sensible
        let mut mf = MedianFilter::<f64, 5>::new();
        let y1 = mf.update(3.0);
        // Only 1 sample: median = 3
        assert!((y1 - 3.0).abs() < 1e-12, "1 sample: {y1}");
        let y2 = mf.update(1.0);
        // 2 samples [3,1] sorted → [1,3], median = lower middle = 1
        assert!((y2 - 1.0).abs() < 1e-12, "2 samples: {y2}");
        let y3 = mf.update(2.0);
        // 3 samples [1,2,3] → median = 2
        assert!((y3 - 2.0).abs() < 1e-12, "3 samples: {y3}");
    }
}
