use crate::core::scalar::ControlScalar;

/// Real-time task timing statistics.
///
/// Tracks execution time statistics for a periodic task:
/// min/max/mean execution time, jitter, and overrun count.
/// All times are in the same units as the input (typically seconds or microseconds).
#[derive(Debug, Clone, Copy)]
pub struct TaskTiming<S: ControlScalar> {
    /// Minimum observed execution time.
    pub min_exec: S,
    /// Maximum observed execution time.
    pub max_exec: S,
    /// Running mean execution time (exponential filter).
    pub mean_exec: S,
    /// Target period.
    pub period: S,
    /// Number of overruns (execution time > period).
    pub overrun_count: u32,
    /// Total execution count.
    pub exec_count: u32,
    /// Filter coefficient for mean computation.
    alpha: S,
}

impl<S: ControlScalar> TaskTiming<S> {
    /// Create timing tracker.
    ///
    /// - `period`: expected task period (for overrun detection)
    /// - `alpha`: EMA filter weight (0 < α ≤ 1; smaller = more smoothing)
    pub fn new(period: S, alpha: S) -> Self {
        let big = S::from_f64(1e30);
        Self {
            min_exec: big,
            max_exec: S::ZERO,
            mean_exec: S::ZERO,
            period,
            overrun_count: 0,
            exec_count: 0,
            alpha,
        }
    }

    /// Record one execution measurement.
    ///
    /// - `exec_time`: measured execution time for this invocation
    pub fn record(&mut self, exec_time: S) {
        self.exec_count += 1;

        if exec_time < self.min_exec {
            self.min_exec = exec_time;
        }
        if exec_time > self.max_exec {
            self.max_exec = exec_time;
        }

        // Exponential moving average
        if self.exec_count == 1 {
            self.mean_exec = exec_time;
        } else {
            self.mean_exec = self.alpha * exec_time + (S::ONE - self.alpha) * self.mean_exec;
        }

        if exec_time > self.period {
            self.overrun_count = self.overrun_count.saturating_add(1);
        }
    }

    /// Utilization ratio: mean_exec / period (0 to 1 is normal).
    pub fn utilization(&self) -> S {
        if self.period > S::ZERO {
            self.mean_exec / self.period
        } else {
            S::ZERO
        }
    }

    /// Jitter: max_exec - min_exec.
    pub fn jitter(&self) -> S {
        if self.exec_count > 1 {
            self.max_exec - self.min_exec
        } else {
            S::ZERO
        }
    }

    /// True if any overrun has occurred.
    pub fn has_overrun(&self) -> bool {
        self.overrun_count > 0
    }

    pub fn reset(&mut self) {
        let big = S::from_f64(1e30);
        self.min_exec = big;
        self.max_exec = S::ZERO;
        self.mean_exec = S::ZERO;
        self.overrun_count = 0;
        self.exec_count = 0;
    }
}

/// Deadline overrun monitor with configurable action.
///
/// Raises an overrun flag when the elapsed time since task start
/// exceeds the deadline. Intended for cooperative real-time systems
/// where the task checks for overruns at checkpoints.
#[derive(Debug, Clone, Copy)]
pub struct DeadlineMonitor<S: ControlScalar> {
    /// Deadline duration.
    pub deadline: S,
    /// Elapsed time since last `start()`.
    elapsed: S,
    /// Whether the deadline was exceeded.
    overrun: bool,
    /// Total overrun count.
    pub overrun_count: u32,
}

impl<S: ControlScalar> DeadlineMonitor<S> {
    pub fn new(deadline: S) -> Self {
        Self {
            deadline,
            elapsed: S::ZERO,
            overrun: false,
            overrun_count: 0,
        }
    }

    /// Mark the start of a task execution.
    pub fn start(&mut self) {
        self.elapsed = S::ZERO;
        self.overrun = false;
    }

    /// Advance time by `dt` and check for deadline violation.
    ///
    /// Returns `true` if deadline exceeded.
    pub fn tick(&mut self, dt: S) -> bool {
        self.elapsed += dt;
        if !self.overrun && self.elapsed > self.deadline {
            self.overrun = true;
            self.overrun_count = self.overrun_count.saturating_add(1);
        }
        self.overrun
    }

    pub fn is_overrun(&self) -> bool {
        self.overrun
    }

    pub fn elapsed(&self) -> S {
        self.elapsed
    }

    pub fn reset(&mut self) {
        self.elapsed = S::ZERO;
        self.overrun = false;
        self.overrun_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_min_max_mean() {
        let mut timing = TaskTiming::new(0.001_f64, 0.5);
        timing.record(0.0005);
        timing.record(0.0008);
        timing.record(0.0003);
        assert!((timing.min_exec - 0.0003).abs() < 1e-12);
        assert!((timing.max_exec - 0.0008).abs() < 1e-12);
        assert!(timing.mean_exec > 0.0);
        assert_eq!(timing.overrun_count, 0);
    }

    #[test]
    fn detects_overrun() {
        let mut timing = TaskTiming::new(0.001_f64, 0.5);
        timing.record(0.0005); // ok
        timing.record(0.0015); // overrun
        timing.record(0.0020); // overrun
        assert_eq!(timing.overrun_count, 2);
        assert!(timing.has_overrun());
    }

    #[test]
    fn utilization_ratio() {
        let mut timing = TaskTiming::new(1.0_f64, 1.0);
        timing.record(0.5);
        assert!((timing.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn deadline_monitor_no_overrun() {
        let mut mon = DeadlineMonitor::new(1.0_f64);
        mon.start();
        assert!(!mon.tick(0.5));
        assert!(!mon.tick(0.4));
        assert!(!mon.is_overrun());
    }

    #[test]
    fn deadline_monitor_overrun() {
        let mut mon = DeadlineMonitor::new(1.0_f64);
        mon.start();
        mon.tick(0.5);
        assert!(mon.tick(0.6)); // total > 1.0
        assert!(mon.is_overrun());
        assert_eq!(mon.overrun_count, 1);
    }

    #[test]
    fn deadline_monitor_resets_between_starts() {
        let mut mon = DeadlineMonitor::new(1.0_f64);
        mon.start();
        mon.tick(2.0); // overrun
        assert!(mon.is_overrun());
        mon.start(); // reset for new period
        assert!(!mon.is_overrun());
        assert!(!mon.tick(0.5));
    }
}
