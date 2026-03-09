use crate::core::scalar::ControlScalar;

/// Monitors for stale/missing data by checking if updates arrive within a timeout.
#[derive(Debug, Clone, Copy)]
pub struct TimeoutMonitor<S: ControlScalar> {
    timeout: S,
    elapsed: S,
    in_violation: bool,
}

impl<S: ControlScalar> TimeoutMonitor<S> {
    pub fn new(timeout: S) -> Self {
        Self {
            timeout,
            elapsed: S::ZERO,
            in_violation: false,
        }
    }

    /// Call this when new data arrives (resets the timer).
    pub fn feed(&mut self) {
        self.elapsed = S::ZERO;
        self.in_violation = false;
    }

    /// Advance time. Returns true if within timeout, false if timed out.
    pub fn check(&mut self, dt: S) -> bool {
        self.elapsed += dt;
        if self.elapsed >= self.timeout {
            self.in_violation = true;
            false
        } else {
            true
        }
    }

    pub fn in_violation(&self) -> bool {
        self.in_violation
    }

    pub fn reset(&mut self) {
        self.elapsed = S::ZERO;
        self.in_violation = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_timeout_when_fed() {
        let mut m = TimeoutMonitor::<f64>::new(0.1);
        assert!(m.check(0.05));
        m.feed();
        assert!(m.check(0.05));
        m.feed();
        assert!(m.check(0.05));
        assert!(!m.in_violation());
    }

    #[test]
    fn timeout_when_not_fed() {
        let mut m = TimeoutMonitor::<f64>::new(0.1);
        assert!(m.check(0.05));
        assert!(!m.check(0.06));
        assert!(m.in_violation());
    }

    #[test]
    fn feed_clears_violation() {
        let mut m = TimeoutMonitor::<f64>::new(0.1);
        m.check(0.2);
        assert!(m.in_violation());
        m.feed();
        assert!(!m.in_violation());
        assert!(m.check(0.05));
    }

    #[test]
    fn reset_clears() {
        let mut m = TimeoutMonitor::<f64>::new(0.1);
        m.check(0.2);
        m.reset();
        assert!(!m.in_violation());
    }
}
