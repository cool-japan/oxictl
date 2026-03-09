use crate::core::scalar::ControlScalar;

/// Plausibility monitor: checks consistency between redundant sensors.
///
/// When two sensors measure the same physical quantity, their values
/// should agree within a tolerance. A large discrepancy may indicate
/// a failed sensor or wiring fault.
#[derive(Debug, Clone, Copy)]
pub struct PlausibilityMonitor<S: ControlScalar> {
    /// Maximum allowed absolute difference between two sensors.
    pub max_diff: S,
    /// Number of consecutive failures before declaring fault.
    pub trip_count: u32,
    consecutive_fail: u32,
    tripped: bool,
}

impl<S: ControlScalar> PlausibilityMonitor<S> {
    pub fn new(max_diff: S, trip_count: u32) -> Self {
        Self {
            max_diff,
            trip_count,
            consecutive_fail: 0,
            tripped: false,
        }
    }

    /// Check if two sensor readings are plausible.
    ///
    /// Returns `true` if readings agree within tolerance.
    pub fn check(&mut self, a: S, b: S) -> bool {
        if self.tripped {
            return false;
        }
        let diff = (a - b).abs();
        if diff > self.max_diff {
            self.consecutive_fail += 1;
            if self.consecutive_fail >= self.trip_count {
                self.tripped = true;
            }
            return !self.tripped;
        }
        self.consecutive_fail = 0;
        true
    }

    pub fn is_tripped(&self) -> bool {
        self.tripped
    }

    pub fn reset(&mut self) {
        self.consecutive_fail = 0;
        self.tripped = false;
    }
}

/// Three-sensor plausibility checker (2oo3 logic with range check).
///
/// Validates that at least two of three sensors agree within `max_diff`.
/// Returns the median (most-agreed) value when plausible.
#[derive(Debug, Clone, Copy)]
pub struct TripleSensorPlausibility<S: ControlScalar> {
    pub max_diff: S,
}

impl<S: ControlScalar> TripleSensorPlausibility<S> {
    pub fn new(max_diff: S) -> Self {
        Self { max_diff }
    }

    /// Check three sensors. Returns `Some(best_value)` if at least two agree,
    /// or `None` if all three disagree (sensor system fault).
    pub fn check(&self, a: S, b: S, c: S) -> Option<S> {
        let ab = (a - b).abs() <= self.max_diff;
        let ac = (a - c).abs() <= self.max_diff;
        let bc = (b - c).abs() <= self.max_diff;

        match (ab, ac, bc) {
            (true, true, true) => {
                // All agree: return median
                Some(median3(a, b, c))
            }
            (true, _, _) => Some((a + b) / S::TWO), // a & b agree
            (_, true, _) => Some((a + c) / S::TWO), // a & c agree
            (_, _, true) => Some((b + c) / S::TWO), // b & c agree
            _ => None,                              // no pair agrees
        }
    }
}

fn median3<S: ControlScalar>(a: S, b: S, c: S) -> S {
    if (a >= b && a <= c) || (a >= c && a <= b) {
        a
    } else if (b >= a && b <= c) || (b >= c && b <= a) {
        b
    } else {
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensors_agree_returns_true() {
        let mut mon = PlausibilityMonitor::new(0.5_f64, 3);
        assert!(mon.check(10.0, 10.2));
        assert!(!mon.is_tripped());
    }

    #[test]
    fn large_discrepancy_trips_after_count() {
        let mut mon = PlausibilityMonitor::new(0.5_f64, 3);
        // 3 consecutive failures
        for _ in 0..3 {
            mon.check(10.0, 15.0);
        }
        assert!(mon.is_tripped());
    }

    #[test]
    fn intermittent_fault_does_not_trip() {
        let mut mon = PlausibilityMonitor::new(0.5_f64, 3);
        mon.check(10.0, 15.0); // fail
        mon.check(10.0, 10.1); // ok — reset counter
        mon.check(10.0, 15.0); // fail
        mon.check(10.0, 10.1); // ok
        assert!(!mon.is_tripped());
    }

    #[test]
    fn triple_sensor_median() {
        let mon = TripleSensorPlausibility::new(0.5_f64);
        let result = mon.check(10.0, 10.1, 10.2);
        assert!(result.is_some());
        let v = result.unwrap();
        assert!((v - 10.1).abs() < 0.01, "v={v:.4}");
    }

    #[test]
    fn triple_sensor_faulty_one() {
        let mon = TripleSensorPlausibility::new(0.5_f64);
        // Sensor C is faulty: 10.0, 10.1 agree; 20.0 disagrees
        let result = mon.check(10.0, 10.1, 20.0);
        assert!(result.is_some());
        let v = result.unwrap();
        assert!((v - 10.05).abs() < 0.01, "v={v:.4}");
    }

    #[test]
    fn triple_sensor_all_disagree_returns_none() {
        let mon = TripleSensorPlausibility::new(0.5_f64);
        let result = mon.check(10.0, 15.0, 20.0);
        assert!(result.is_none());
    }
}
