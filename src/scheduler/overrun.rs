use crate::core::scalar::ControlScalar;

/// Task execution overrun monitor.
///
/// Detects when a task exceeds its execution time budget.
/// Consecutive overruns beyond `max_consecutive` trip the monitor.
#[derive(Debug, Clone, Copy)]
pub struct OverrunMonitor<S: ControlScalar> {
    /// Nominal execution time budget (s or ticks).
    pub budget: S,
    /// Maximum consecutive overruns before tripping.
    pub max_consecutive: u32,
    consecutive_overruns: u32,
    total_overruns: u32,
    tripped: bool,
    /// Maximum observed execution time.
    max_observed: S,
}

impl<S: ControlScalar> OverrunMonitor<S> {
    pub fn new(budget: S, max_consecutive: u32) -> Self {
        Self {
            budget,
            max_consecutive,
            consecutive_overruns: 0,
            total_overruns: 0,
            tripped: false,
            max_observed: S::ZERO,
        }
    }

    /// Report an actual execution time. Returns `true` if within budget.
    pub fn report(&mut self, actual_time: S) -> bool {
        if actual_time > self.max_observed {
            self.max_observed = actual_time;
        }
        if actual_time > self.budget {
            self.consecutive_overruns += 1;
            self.total_overruns += 1;
            if self.consecutive_overruns > self.max_consecutive {
                self.tripped = true;
            }
            return false;
        }
        self.consecutive_overruns = 0;
        true
    }

    pub fn is_tripped(&self) -> bool {
        self.tripped
    }
    pub fn total_overruns(&self) -> u32 {
        self.total_overruns
    }
    pub fn max_observed(&self) -> S {
        self.max_observed
    }
    pub fn consecutive_overruns(&self) -> u32 {
        self.consecutive_overruns
    }

    pub fn reset(&mut self) {
        self.consecutive_overruns = 0;
        self.total_overruns = 0;
        self.tripped = false;
        self.max_observed = S::ZERO;
    }
}

/// Jitter monitor: tracks execution time variation (max - min).
#[derive(Debug, Clone, Copy)]
pub struct JitterMonitor<S: ControlScalar> {
    /// Maximum allowed jitter (execution time variation).
    pub max_jitter: S,
    min_time: S,
    max_time: S,
    initialized: bool,
    jitter_exceeded: bool,
}

impl<S: ControlScalar> JitterMonitor<S> {
    pub fn new(max_jitter: S) -> Self {
        Self {
            max_jitter,
            min_time: S::ZERO,
            max_time: S::ZERO,
            initialized: false,
            jitter_exceeded: false,
        }
    }

    /// Record a timing sample. Returns `true` if jitter is within limit.
    pub fn record(&mut self, time: S) -> bool {
        if !self.initialized {
            self.min_time = time;
            self.max_time = time;
            self.initialized = true;
            return true;
        }
        if time < self.min_time {
            self.min_time = time;
        }
        if time > self.max_time {
            self.max_time = time;
        }

        let jitter = self.max_time - self.min_time;
        if jitter > self.max_jitter {
            self.jitter_exceeded = true;
        }
        !self.jitter_exceeded
    }

    pub fn jitter(&self) -> S {
        if self.initialized {
            self.max_time - self.min_time
        } else {
            S::ZERO
        }
    }

    pub fn is_exceeded(&self) -> bool {
        self.jitter_exceeded
    }

    pub fn reset(&mut self) {
        self.initialized = false;
        self.jitter_exceeded = false;
        self.min_time = S::ZERO;
        self.max_time = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn within_budget_ok() {
        let mut mon = OverrunMonitor::new(1.0_f64, 3);
        assert!(mon.report(0.8));
        assert!(mon.report(0.9));
        assert!(!mon.is_tripped());
    }

    #[test]
    fn overrun_trips_after_consecutive() {
        let mut mon = OverrunMonitor::new(1.0_f64, 2);
        mon.report(1.5); // overrun 1
        mon.report(1.5); // overrun 2
        mon.report(1.5); // overrun 3 → trip
        assert!(mon.is_tripped());
    }

    #[test]
    fn single_overrun_allowed() {
        let mut mon = OverrunMonitor::new(1.0_f64, 2);
        mon.report(1.5); // overrun 1 (allowed up to 2)
        assert!(!mon.is_tripped());
        mon.report(0.8); // ok → resets counter
        mon.report(1.5); // overrun 1 again
        assert!(!mon.is_tripped());
    }

    #[test]
    fn max_observed_tracked() {
        let mut mon = OverrunMonitor::new(1.0_f64, 5);
        mon.report(0.5);
        mon.report(1.2);
        mon.report(0.9);
        assert!((mon.max_observed() - 1.2).abs() < 1e-10);
    }

    #[test]
    fn jitter_monitor_tracks_variation() {
        let mut jm = JitterMonitor::new(0.5_f64);
        jm.record(1.0);
        jm.record(1.2);
        jm.record(0.9);
        assert!((jm.jitter() - 0.3).abs() < 1e-10);
        assert!(!jm.is_exceeded());
    }

    #[test]
    fn jitter_exceeds_limit() {
        let mut jm = JitterMonitor::new(0.1_f64);
        jm.record(1.0);
        jm.record(1.5); // jitter = 0.5 > 0.1
        assert!(jm.is_exceeded());
    }
}
