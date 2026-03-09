use crate::core::scalar::ControlScalar;

/// Monitors the rate of change of a signal.
/// Triggers a violation if |dv/dt| exceeds the threshold.
#[derive(Debug, Clone, Copy)]
pub struct RateMonitor<S: ControlScalar> {
    max_rate: S,
    prev_value: Option<S>,
    in_violation: bool,
    last_rate: S,
}

impl<S: ControlScalar> RateMonitor<S> {
    pub fn new(max_rate: S) -> Self {
        Self {
            max_rate,
            prev_value: None,
            in_violation: false,
            last_rate: S::ZERO,
        }
    }

    /// Check a new value with time step dt.
    /// Returns true if rate is acceptable, false if violation.
    pub fn check(&mut self, value: S, dt: S) -> bool {
        match self.prev_value {
            None => {
                self.prev_value = Some(value);
                self.in_violation = false;
                true
            }
            Some(prev) => {
                if dt <= S::ZERO {
                    self.prev_value = Some(value);
                    return true;
                }
                let rate = (value - prev) / dt;
                self.last_rate = rate;
                self.prev_value = Some(value);

                if rate.abs() > self.max_rate {
                    self.in_violation = true;
                    false
                } else {
                    self.in_violation = false;
                    true
                }
            }
        }
    }

    pub fn in_violation(&self) -> bool {
        self.in_violation
    }

    pub fn last_rate(&self) -> S {
        self.last_rate
    }

    pub fn reset(&mut self) {
        self.prev_value = None;
        self.in_violation = false;
        self.last_rate = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_value_ok() {
        let mut m = RateMonitor::<f64>::new(100.0);
        assert!(m.check(50.0, 0.01));
    }

    #[test]
    fn slow_change_ok() {
        let mut m = RateMonitor::<f64>::new(100.0);
        m.check(0.0, 0.01);
        assert!(m.check(0.5, 0.01)); // rate = 50/s < 100
    }

    #[test]
    fn fast_change_violation() {
        let mut m = RateMonitor::<f64>::new(100.0);
        m.check(0.0, 0.01);
        assert!(!m.check(10.0, 0.01)); // rate = 1000/s > 100
        assert!(m.in_violation());
    }

    #[test]
    fn negative_rate_violation() {
        let mut m = RateMonitor::<f64>::new(100.0);
        m.check(10.0, 0.01);
        assert!(!m.check(0.0, 0.01)); // rate = -1000/s, |rate| > 100
    }

    #[test]
    fn violation_clears() {
        let mut m = RateMonitor::<f64>::new(100.0);
        m.check(0.0, 0.01);
        m.check(10.0, 0.01); // violation
        assert!(m.in_violation());
        m.check(10.1, 0.01); // rate = 10/s, ok
        assert!(!m.in_violation());
    }

    #[test]
    fn reset_clears_state() {
        let mut m = RateMonitor::<f64>::new(100.0);
        m.check(0.0, 0.01);
        m.check(100.0, 0.01);
        m.reset();
        assert!(!m.in_violation());
        // After reset, next value is treated as first
        assert!(m.check(200.0, 0.01));
    }
}
