use crate::core::scalar::ControlScalar;
use crate::scheduler::fixed_rate::FixedRateTask;

/// Multi-rate scheduler with N tasks at different frequencies.
///
/// All tasks are driven from a single "base tick" (fastest rate).
/// Higher-rate tasks fire every tick, lower-rate tasks fire every N ticks.
///
/// N = number of tasks.
pub struct MultiRateScheduler<S: ControlScalar, const N: usize> {
    tasks: [FixedRateTask<S>; N],
}

impl<S: ControlScalar, const N: usize> MultiRateScheduler<S, N> {
    /// Create scheduler with N tasks, each with the given period.
    pub fn new(periods: [S; N]) -> Self {
        Self {
            tasks: core::array::from_fn(|i| FixedRateTask::new(periods[i])),
        }
    }

    /// Advance all tasks by dt. Returns a bitmask of which tasks fired.
    pub fn tick(&mut self, dt: S) -> [bool; N] {
        core::array::from_fn(|i| self.tasks[i].tick(dt))
    }

    /// Advance and call a closure for each fired task.
    pub fn tick_with<F: FnMut(usize)>(&mut self, dt: S, mut callback: F) {
        for i in 0..N {
            if self.tasks[i].tick(dt) {
                callback(i);
            }
        }
    }

    pub fn task(&self, index: usize) -> Option<&FixedRateTask<S>> {
        self.tasks.get(index)
    }

    pub fn task_mut(&mut self, index: usize) -> Option<&mut FixedRateTask<S>> {
        self.tasks.get_mut(index)
    }

    pub fn reset_all(&mut self) {
        for task in &mut self.tasks {
            task.reset();
        }
    }
}

/// Task priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Named task descriptor for multi-rate scheduling.
#[derive(Debug, Clone)]
pub struct TaskDescriptor {
    pub name: &'static str,
    pub priority: TaskPriority,
}

/// Multi-rate scheduler with priority and overrun detection.
pub struct PriorityScheduler<S: ControlScalar, const N: usize> {
    pub scheduler: MultiRateScheduler<S, N>,
    pub descriptors: [TaskDescriptor; N],
    overrun_counts: [u32; N],
    last_execution_time: [S; N],
    budget: [S; N],
}

impl<S: ControlScalar, const N: usize> PriorityScheduler<S, N> {
    pub fn new(periods: [S; N], descriptors: [TaskDescriptor; N], budgets: [S; N]) -> Self {
        Self {
            scheduler: MultiRateScheduler::new(periods),
            descriptors,
            overrun_counts: [0; N],
            last_execution_time: core::array::from_fn(|_| S::ZERO),
            budget: budgets,
        }
    }

    /// Report task execution time. Returns true if overrun occurred.
    pub fn report_execution(&mut self, task_index: usize, execution_time: S) -> bool {
        if task_index >= N {
            return false;
        }
        self.last_execution_time[task_index] = execution_time;
        let overrun = execution_time > self.budget[task_index];
        if overrun {
            self.overrun_counts[task_index] += 1;
        }
        overrun
    }

    pub fn overrun_count(&self, task_index: usize) -> u32 {
        if task_index < N {
            self.overrun_counts[task_index]
        } else {
            0
        }
    }

    pub fn last_execution_time(&self, task_index: usize) -> S {
        if task_index < N {
            self.last_execution_time[task_index]
        } else {
            S::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_rate_fires_at_correct_rates() {
        let mut sched = MultiRateScheduler::<f64, 3>::new([0.001, 0.01, 0.1]);
        let mut counts = [0u32; 3];

        for _ in 0..10000 {
            let fired = sched.tick(0.001);
            for i in 0..3 {
                if fired[i] {
                    counts[i] += 1;
                }
            }
        }

        // 10s simulation at 1ms tick:
        assert!(
            counts[0] >= 9990 && counts[0] <= 10010,
            "1kHz: {}",
            counts[0]
        );
        assert!(
            counts[1] >= 990 && counts[1] <= 1010,
            "100Hz: {}",
            counts[1]
        );
        assert!(counts[2] >= 99 && counts[2] <= 101, "10Hz: {}", counts[2]);
    }

    #[test]
    fn tick_with_calls_callback() {
        let mut sched = MultiRateScheduler::<f64, 2>::new([0.01, 0.1]);
        let mut task0_count = 0u32;
        let mut task1_count = 0u32;

        for _ in 0..1000 {
            sched.tick_with(0.001, |i| {
                if i == 0 {
                    task0_count += 1;
                } else {
                    task1_count += 1;
                }
            });
        }

        assert!((99..=101).contains(&task0_count), "100Hz: {}", task0_count);
        assert!((9..=11).contains(&task1_count), "10Hz: {}", task1_count);
    }

    #[test]
    fn priority_scheduler_overrun_detection() {
        let descriptors = [
            TaskDescriptor {
                name: "fast",
                priority: TaskPriority::Critical,
            },
            TaskDescriptor {
                name: "slow",
                priority: TaskPriority::Low,
            },
        ];
        let mut ps = PriorityScheduler::<f64, 2>::new(
            [0.001, 0.01],
            descriptors,
            [0.0005, 0.005], // budgets
        );

        // Report overrun for task 0
        let overrun = ps.report_execution(0, 0.001); // twice the budget
        assert!(overrun);
        assert_eq!(ps.overrun_count(0), 1);

        // No overrun for task 1
        let no_overrun = ps.report_execution(1, 0.001);
        assert!(!no_overrun);
        assert_eq!(ps.overrun_count(1), 0);
    }

    #[test]
    fn reset_all() {
        let mut sched = MultiRateScheduler::<f64, 2>::new([0.1, 0.2]);
        sched.tick(0.09);
        sched.reset_all();
        let fired = sched.tick(0.05);
        assert!(!fired[0]); // should not fire immediately after reset
    }
}
