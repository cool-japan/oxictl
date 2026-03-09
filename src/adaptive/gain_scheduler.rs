//! Gain scheduler: operating-point based gain interpolation.
//!
//! Supports N breakpoints each with a G-element gain vector.
//! Linear interpolation is used between adjacent breakpoints.
//! The breakpoint table is maintained in sorted ascending order.

use crate::core::scalar::ControlScalar;

/// N-breakpoint gain scheduler with linear interpolation.
///
/// Each breakpoint maps an operating point (e.g. vehicle speed, rotor speed)
/// to a gain vector of length G (e.g. [Kp, Ki, Kd]).
pub struct GainScheduler<S: ControlScalar, const N: usize, const G: usize> {
    /// Operating point breakpoints, sorted ascending. Valid entries are [0..count].
    pub breakpoints: [S; N],
    /// Gain vectors corresponding to each breakpoint.
    pub gains: [[S; G]; N],
    /// Number of valid breakpoints currently stored.
    count: usize,
}

impl<S: ControlScalar, const N: usize, const G: usize> GainScheduler<S, N, G> {
    /// Create an empty gain scheduler.
    pub fn new() -> Self {
        Self {
            breakpoints: [S::ZERO; N],
            gains: [[S::ZERO; G]; N],
            count: 0,
        }
    }

    /// Add a breakpoint at `op_point` with gain vector `gains`.
    ///
    /// Inserts in sorted order. Returns `true` on success, `false` if table is full.
    pub fn add_breakpoint(&mut self, op_point: S, gains: [S; G]) -> bool {
        if self.count >= N {
            return false;
        }
        // Find insertion index (sorted ascending)
        let mut idx = self.count;
        for i in 0..self.count {
            if op_point < self.breakpoints[i] {
                idx = i;
                break;
            }
        }
        // Shift elements right to make room
        if idx < self.count {
            for j in (idx..self.count).rev() {
                self.breakpoints[j + 1] = self.breakpoints[j];
                self.gains[j + 1] = self.gains[j];
            }
        }
        self.breakpoints[idx] = op_point;
        self.gains[idx] = gains;
        self.count += 1;
        true
    }

    /// Interpolate the gain vector at operating point `op_point`.
    ///
    /// - Below the first breakpoint: returns the first gain vector.
    /// - Above the last breakpoint: returns the last gain vector.
    /// - Between breakpoints: linear interpolation.
    pub fn schedule(&self, op_point: S) -> [S; G] {
        if self.count == 0 {
            return [S::ZERO; G];
        }
        if self.count == 1 || op_point <= self.breakpoints[0] {
            return self.gains[0];
        }
        if op_point >= self.breakpoints[self.count - 1] {
            return self.gains[self.count - 1];
        }
        // Binary search for bracketing interval
        let mut lo = 0usize;
        let mut hi = self.count - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if op_point >= self.breakpoints[mid] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let x0 = self.breakpoints[lo];
        let x1 = self.breakpoints[hi];
        let span = x1 - x0;
        let t = if span > S::EPSILON {
            (op_point - x0) / span
        } else {
            S::ZERO
        };
        self.lerp_gains(self.gains[lo], self.gains[hi], t)
    }

    /// Linear interpolation between two gain vectors.
    ///
    /// Returns g0 * (1 - t) + g1 * t.
    fn lerp_gains(&self, g0: [S; G], g1: [S; G], t: S) -> [S; G] {
        core::array::from_fn(|i| g0[i] + t * (g1[i] - g0[i]))
    }

    /// Number of active breakpoints.
    pub fn len(&self) -> usize {
        self.count
    }

    /// True if no breakpoints are defined.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl<S: ControlScalar, const N: usize, const G: usize> Default for GainScheduler<S, N, G> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_schedule_single() {
        let mut sched = GainScheduler::<f64, 8, 3>::new();
        sched.add_breakpoint(10.0, [1.0, 0.1, 0.01]);
        let g = sched.schedule(10.0);
        assert_eq!(g, [1.0, 0.1, 0.01]);
        // Extrapolation: below
        let g_low = sched.schedule(0.0);
        assert_eq!(g_low, [1.0, 0.1, 0.01]);
        // Extrapolation: above
        let g_high = sched.schedule(100.0);
        assert_eq!(g_high, [1.0, 0.1, 0.01]);
    }

    #[test]
    fn test_linear_interpolation() {
        let mut sched = GainScheduler::<f64, 4, 2>::new();
        sched.add_breakpoint(0.0, [0.0, 0.0]);
        sched.add_breakpoint(10.0, [10.0, 20.0]);
        let g = sched.schedule(5.0);
        assert!((g[0] - 5.0).abs() < 1e-10, "g[0]={}", g[0]);
        assert!((g[1] - 10.0).abs() < 1e-10, "g[1]={}", g[1]);
    }

    #[test]
    fn test_sorted_insertion() {
        let mut sched = GainScheduler::<f64, 8, 1>::new();
        // Insert out of order
        sched.add_breakpoint(30.0, [3.0]);
        sched.add_breakpoint(10.0, [1.0]);
        sched.add_breakpoint(20.0, [2.0]);
        assert_eq!(sched.len(), 3);
        // Check sorted order
        assert!((sched.breakpoints[0] - 10.0).abs() < 1e-10);
        assert!((sched.breakpoints[1] - 20.0).abs() < 1e-10);
        assert!((sched.breakpoints[2] - 30.0).abs() < 1e-10);
        // Interpolate at midpoint
        let g = sched.schedule(15.0);
        assert!((g[0] - 1.5).abs() < 1e-10, "g[0]={}", g[0]);
    }

    #[test]
    fn test_table_full_returns_false() {
        let mut sched = GainScheduler::<f64, 2, 1>::new();
        assert!(sched.add_breakpoint(0.0, [1.0]));
        assert!(sched.add_breakpoint(1.0, [2.0]));
        // Table now full
        assert!(!sched.add_breakpoint(2.0, [3.0]));
        assert_eq!(sched.len(), 2);
    }

    #[test]
    fn test_empty_scheduler_returns_zeros() {
        let sched = GainScheduler::<f64, 4, 3>::new();
        assert!(sched.is_empty());
        let g = sched.schedule(5.0);
        assert_eq!(g, [0.0, 0.0, 0.0]);
    }
}
