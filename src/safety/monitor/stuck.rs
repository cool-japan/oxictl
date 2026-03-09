use crate::core::scalar::ControlScalar;

/// Stuck (stagnation) monitor.
///
/// Detects when a signal stops changing, which may indicate a sensor failure,
/// a frozen actuator, or a software hang.
///
/// A fault is raised when the signal changes by less than `threshold`
/// for a continuous period of `timeout` seconds.
#[derive(Debug, Clone, Copy)]
pub struct StuckMonitor<S: ControlScalar> {
    /// Minimum absolute change required to reset the stagnation timer.
    pub threshold: S,
    /// Duration of stagnation before a fault is declared (s).
    pub timeout: S,
    /// Previous value for change detection.
    prev_value: S,
    /// Accumulated stagnation time (s).
    stagnant_time: S,
    /// Whether the monitor has detected a stuck condition.
    stuck: bool,
    /// Whether the monitor has been initialized.
    initialized: bool,
}

impl<S: ControlScalar> StuckMonitor<S> {
    pub fn new(threshold: S, timeout: S) -> Self {
        Self {
            threshold,
            timeout,
            prev_value: S::ZERO,
            stagnant_time: S::ZERO,
            stuck: false,
            initialized: false,
        }
    }

    /// Update with a new value. Returns `true` if signal is healthy (not stuck).
    pub fn check(&mut self, value: S, dt: S) -> bool {
        if !self.initialized {
            self.prev_value = value;
            self.initialized = true;
            return true;
        }

        let change = (value - self.prev_value).abs();
        self.prev_value = value;

        if change >= self.threshold {
            // Signal is moving — reset stagnation timer
            self.stagnant_time = S::ZERO;
        } else {
            self.stagnant_time += dt;
            if self.stagnant_time >= self.timeout {
                self.stuck = true;
            }
        }

        !self.stuck
    }

    pub fn is_stuck(&self) -> bool {
        self.stuck
    }

    pub fn reset(&mut self) {
        self.stagnant_time = S::ZERO;
        self.stuck = false;
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moving_signal_does_not_trip() {
        let mut mon = StuckMonitor::new(0.01_f64, 1.0);
        let mut v = 0.0_f64;
        for _ in 0..100 {
            v += 0.1;
            assert!(mon.check(v, 0.01));
        }
        assert!(!mon.is_stuck());
    }

    #[test]
    fn stuck_signal_trips_after_timeout() {
        let mut mon = StuckMonitor::new(0.01_f64, 1.0);
        // Signal frozen at 5.0
        for _ in 0..200 {
            mon.check(5.0, 0.01); // 2s total
        }
        assert!(mon.is_stuck());
    }

    #[test]
    fn small_changes_below_threshold_count_as_stuck() {
        let mut mon = StuckMonitor::new(0.1_f64, 0.5); // 50 steps to trip
        for i in 0..200 {
            // Change by only 0.001 per step — below threshold 0.1
            mon.check(i as f64 * 0.001, 0.01);
        }
        assert!(mon.is_stuck());
    }

    #[test]
    fn reset_clears_stuck_state() {
        let mut mon = StuckMonitor::new(0.01_f64, 0.1);
        for _ in 0..20 {
            mon.check(1.0, 0.01);
        }
        assert!(mon.is_stuck());
        mon.reset();
        assert!(!mon.is_stuck());
        // After reset, first check re-initializes
        assert!(mon.check(2.0, 0.01));
    }
}
