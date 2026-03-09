use crate::core::scalar::ControlScalar;

/// Fixed-rate task execution tracker.
///
/// Tracks whether a control task is due to run based on elapsed time.
/// Used in a superloop to decide when to execute a periodic task.
#[derive(Debug, Clone, Copy)]
pub struct FixedRateTask<S: ControlScalar> {
    /// Target period in seconds.
    period: S,
    /// Time accumulated since last execution.
    accumulated: S,
    /// Execution count.
    count: u64,
    /// Whether the task is active.
    active: bool,
}

impl<S: ControlScalar> FixedRateTask<S> {
    pub fn new(period: S) -> Self {
        Self {
            period,
            accumulated: S::ZERO,
            count: 0,
            active: true,
        }
    }

    /// Advance the timer by dt. Returns true if the task should execute.
    pub fn tick(&mut self, dt: S) -> bool {
        if !self.active || self.period <= S::ZERO {
            return false;
        }
        self.accumulated += dt;
        if self.accumulated >= self.period {
            self.accumulated -= self.period;
            self.count += 1;
            true
        } else {
            false
        }
    }

    /// Force the task to be due on the next tick.
    pub fn trigger_now(&mut self) {
        self.accumulated = self.period;
    }

    /// Reset the accumulator.
    pub fn reset(&mut self) {
        self.accumulated = S::ZERO;
    }

    pub fn period(&self) -> S {
        self.period
    }

    pub fn execution_count(&self) -> u64 {
        self.count
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Frequency in Hz.
    pub fn frequency(&self) -> S {
        if self.period > S::ZERO {
            S::ONE / self.period
        } else {
            S::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_at_period() {
        let mut task = FixedRateTask::<f64>::new(0.01); // 100 Hz
        assert!(!task.tick(0.005)); // not yet
        assert!(task.tick(0.006)); // fires at 0.011 >= 0.01
    }

    #[test]
    fn fires_exactly_every_n_steps() {
        let mut task = FixedRateTask::<f64>::new(0.01);
        let mut fires = 0;
        for _ in 0..1000 {
            if task.tick(0.001) {
                fires += 1;
            }
        }
        assert_eq!(fires, 100, "Should fire 100 times in 1s at 100Hz");
    }

    #[test]
    fn count_increments() {
        let mut task = FixedRateTask::<f64>::new(10.0);
        for _ in 0..100 {
            task.tick(1.0);
        }
        assert_eq!(task.execution_count(), 10);
    }

    #[test]
    fn inactive_task_never_fires() {
        let mut task = FixedRateTask::<f64>::new(0.01);
        task.set_active(false);
        for _ in 0..1000 {
            assert!(!task.tick(0.1));
        }
    }

    #[test]
    fn trigger_now_forces_fire() {
        let mut task = FixedRateTask::<f64>::new(1.0); // 1 Hz
        task.trigger_now();
        assert!(task.tick(0.001));
    }

    #[test]
    fn reset_resets_accumulator() {
        let mut task = FixedRateTask::<f64>::new(0.1);
        task.tick(0.09);
        task.reset();
        assert!(!task.tick(0.05)); // should not fire after reset
    }

    #[test]
    fn frequency() {
        let task = FixedRateTask::<f64>::new(0.01);
        assert!((task.frequency() - 100.0).abs() < 1e-10);
    }
}
