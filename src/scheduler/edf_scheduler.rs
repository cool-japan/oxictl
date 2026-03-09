//! Earliest Deadline First (EDF) scheduler for real-time tasks.
//!
//! EDF is optimal for preemptive real-time scheduling: a task set is schedulable
//! iff the total utilization ≤ 1. At each tick the ready task with the earliest
//! absolute deadline is dispatched first.
use crate::core::scalar::ControlScalar;

/// EDF task descriptor.
#[derive(Debug, Clone, Copy)]
pub struct EdfTask<S: ControlScalar> {
    /// Unique task identifier.
    pub id: u8,
    /// Task period (seconds).
    pub period: S,
    /// Relative deadline (seconds, ≤ period).
    pub deadline: S,
    /// Execution budget (seconds).
    pub budget: S,
    /// Absolute next deadline (seconds from time-origin).
    pub next_abs_deadline: S,
    /// Next release time (seconds from time-origin).
    pub next_release: S,
}

impl<S: ControlScalar> EdfTask<S> {
    /// Create a new EDF task. Initial release and deadline start at 0.
    pub fn new(id: u8, period: S, deadline: S, budget: S) -> Self {
        Self {
            id,
            period,
            deadline,
            budget,
            next_abs_deadline: deadline,
            next_release: S::ZERO,
        }
    }

    /// Advance release and absolute deadline by one period after execution.
    pub fn advance(&mut self) {
        self.next_release += self.period;
        self.next_abs_deadline += self.period;
    }

    /// CPU utilization fraction for this task.
    pub fn utilization(&self) -> S {
        self.budget / self.period
    }
}

/// EDF scheduler holding up to N tasks.
pub struct EdfScheduler<S: ControlScalar, const N: usize> {
    pub tasks: [Option<EdfTask<S>>; N],
    pub current_time: S,
    pub deadline_miss_count: u32,
}

impl<S: ControlScalar, const N: usize> EdfScheduler<S, N> {
    /// Create an empty scheduler.
    pub fn new() -> Self {
        Self {
            tasks: core::array::from_fn(|_| None),
            current_time: S::ZERO,
            deadline_miss_count: 0,
        }
    }

    /// Add a task. Returns `false` if the task table is full.
    pub fn add_task(&mut self, task: EdfTask<S>) -> bool {
        for slot in self.tasks.iter_mut() {
            if slot.is_none() {
                *slot = Some(task);
                return true;
            }
        }
        false
    }

    /// Advance time by `dt` seconds.
    ///
    /// Returns a `heapless::Vec` of task IDs that are ready (released and not
    /// yet past their deadline), sorted by ascending absolute deadline (EDF order).
    pub fn tick(&mut self, dt: S) -> heapless::Vec<u8, N> {
        self.current_time += dt;

        // Collect ready tasks: released and not yet past deadline.
        let mut ready: heapless::Vec<(S, u8), N> = heapless::Vec::new();
        for slot in self.tasks.iter_mut().flatten() {
            if self.current_time >= slot.next_release {
                if self.current_time > slot.next_abs_deadline {
                    self.deadline_miss_count += 1;
                }
                let _ = ready.push((slot.next_abs_deadline, slot.id));
                slot.advance();
            }
        }

        // Sort by deadline (bubble sort — N is small, no alloc needed).
        let len = ready.len();
        for i in 0..len {
            for j in 0..len - 1 - i {
                if ready[j].0 > ready[j + 1].0 {
                    ready.swap(j, j + 1);
                }
            }
        }

        let mut ids: heapless::Vec<u8, N> = heapless::Vec::new();
        for (_, id) in ready {
            let _ = ids.push(id);
        }
        ids
    }

    /// Schedulability test: total utilization ≤ 1.
    pub fn is_schedulable(&self) -> bool {
        self.total_utilization() <= S::ONE
    }

    /// Sum of per-task utilizations.
    pub fn total_utilization(&self) -> S {
        self.tasks
            .iter()
            .flatten()
            .fold(S::ZERO, |acc, t| acc + t.utilization())
    }
}

impl<S: ControlScalar, const N: usize> Default for EdfScheduler<S, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedulability_check() {
        let mut sched = EdfScheduler::<f64, 4>::new();
        // Tasks: util = 0.3 + 0.4 = 0.7 → schedulable
        sched.add_task(EdfTask::new(1, 1.0, 1.0, 0.3));
        sched.add_task(EdfTask::new(2, 1.0, 1.0, 0.4));
        assert!(sched.is_schedulable());
        assert!((sched.total_utilization() - 0.7).abs() < 1e-10);

        // Push over 1.0
        sched.add_task(EdfTask::new(3, 1.0, 1.0, 0.4));
        assert!(!sched.is_schedulable());
    }

    #[test]
    fn edf_ordering_by_deadline() {
        let mut sched = EdfScheduler::<f64, 4>::new();
        // Task A: period 1.0, deadline 1.0
        // Task B: period 2.0, deadline 0.5 (earlier deadline)
        sched.add_task(EdfTask::new(10, 1.0, 1.0, 0.1));
        sched.add_task(EdfTask::new(20, 2.0, 0.5, 0.1));
        let ready = sched.tick(0.1);
        // Both released at t=0; task 20 has abs_deadline=0.5, task 10 has 1.0
        assert_eq!(ready.len(), 2);
        assert_eq!(ready[0], 20, "task with earlier deadline should be first");
        assert_eq!(ready[1], 10);
    }

    #[test]
    fn add_task_full_returns_false() {
        let mut sched = EdfScheduler::<f64, 2>::new();
        assert!(sched.add_task(EdfTask::new(1, 1.0, 1.0, 0.1)));
        assert!(sched.add_task(EdfTask::new(2, 1.0, 1.0, 0.1)));
        assert!(!sched.add_task(EdfTask::new(3, 1.0, 1.0, 0.1)));
    }

    #[test]
    fn periodic_release_advances_correctly() {
        let mut sched = EdfScheduler::<f64, 2>::new();
        sched.add_task(EdfTask::new(1, 1.0, 1.0, 0.1));
        // Tick past first period — task fires at t=0 and t=1.
        let r0 = sched.tick(0.5); // t=0.5: released at t=0
        assert_eq!(r0.len(), 1);
        let r1 = sched.tick(0.5); // t=1.0: next release
        assert_eq!(r1.len(), 1);
    }
}
