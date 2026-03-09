use crate::core::scalar::ControlScalar;

/// Monitors whether a value stays within acceptable bounds.
#[derive(Debug, Clone, Copy)]
pub struct RangeMonitor<S: ControlScalar> {
    low: S,
    high: S,
    in_violation: bool,
}

impl<S: ControlScalar> RangeMonitor<S> {
    pub fn new(low: S, high: S) -> Self {
        debug_assert!(low <= high);
        Self {
            low,
            high,
            in_violation: false,
        }
    }

    /// Check a value. Returns true if within range, false if violation.
    pub fn check(&mut self, value: S) -> bool {
        if value < self.low || value > self.high {
            self.in_violation = true;
            false
        } else {
            self.in_violation = false;
            true
        }
    }

    pub fn in_violation(&self) -> bool {
        self.in_violation
    }

    pub fn reset(&mut self) {
        self.in_violation = false;
    }

    pub fn low(&self) -> S {
        self.low
    }

    pub fn high(&self) -> S {
        self.high
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn within_range() {
        let mut m = RangeMonitor::<f64>::new(0.0, 100.0);
        assert!(m.check(50.0));
        assert!(!m.in_violation());
    }

    #[test]
    fn above_range() {
        let mut m = RangeMonitor::<f64>::new(0.0, 100.0);
        assert!(!m.check(101.0));
        assert!(m.in_violation());
    }

    #[test]
    fn below_range() {
        let mut m = RangeMonitor::<f64>::new(0.0, 100.0);
        assert!(!m.check(-1.0));
        assert!(m.in_violation());
    }

    #[test]
    fn boundary_values() {
        let mut m = RangeMonitor::<f64>::new(0.0, 100.0);
        assert!(m.check(0.0));
        assert!(m.check(100.0));
    }

    #[test]
    fn violation_clears_when_back_in_range() {
        let mut m = RangeMonitor::<f64>::new(0.0, 100.0);
        m.check(200.0);
        assert!(m.in_violation());
        m.check(50.0);
        assert!(!m.in_violation());
    }
}
