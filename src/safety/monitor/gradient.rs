//! Gradient (2nd derivative) monitor for detecting abrupt signal changes.
#![allow(dead_code)]

use crate::core::scalar::ControlScalar;

/// Monitors the rate-of-change of rate-of-change (acceleration / 2nd derivative) of a signal.
///
/// On the first call to [`GradientMonitor::check`] the monitor initialises its internal state
/// and returns `true` (no violation can be detected yet).  On the second call the first
/// derivative is established.  Only from the third call onwards can a 2nd-derivative violation
/// be reported.
#[derive(Debug, Clone, Copy)]
pub struct GradientMonitor<S: ControlScalar> {
    /// Maximum allowed absolute acceleration (2nd derivative magnitude).
    pub max_gradient: S,
    prev_value: S,
    prev_derivative: S,
    initialized: bool,
    /// True once we have at least one valid derivative estimate.
    has_derivative: bool,
}

impl<S: ControlScalar> GradientMonitor<S> {
    /// Create a new gradient monitor with the given acceleration limit.
    pub fn new(max_gradient: S) -> Self {
        Self {
            max_gradient,
            prev_value: S::ZERO,
            prev_derivative: S::ZERO,
            initialized: false,
            has_derivative: false,
        }
    }

    /// Check a new sample.
    ///
    /// Returns `true` if the 2nd derivative is within the allowed limit (or not yet
    /// computable), `false` if the limit is exceeded.
    pub fn check(&mut self, value: S, dt: S) -> bool {
        if dt <= S::ZERO {
            return true;
        }

        if !self.initialized {
            // First sample — just store value.
            self.prev_value = value;
            self.initialized = true;
            return true;
        }

        // Estimate first derivative.
        let derivative = (value - self.prev_value) / dt;

        if !self.has_derivative {
            // Second sample — first derivative now available but no 2nd derivative yet.
            self.prev_derivative = derivative;
            self.prev_value = value;
            self.has_derivative = true;
            return true;
        }

        // Estimate second derivative (gradient of the derivative).
        let gradient = (derivative - self.prev_derivative) / dt;

        self.prev_derivative = derivative;
        self.prev_value = value;

        let abs_gradient = if gradient >= S::ZERO {
            gradient
        } else {
            S::ZERO - gradient
        };

        abs_gradient <= self.max_gradient
    }

    /// Current estimated first derivative (velocity).
    pub fn derivative(&self) -> S {
        self.prev_derivative
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.prev_value = S::ZERO;
        self.prev_derivative = S::ZERO;
        self.initialized = false;
        self.has_derivative = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_signal_no_violation() {
        let mut mon = GradientMonitor::new(10.0_f64);
        // All samples at the same value → 2nd derivative = 0.
        for _ in 0..10 {
            assert!(mon.check(5.0, 0.01));
        }
    }

    #[test]
    fn linear_ramp_no_violation() {
        let mut mon = GradientMonitor::new(1.0_f64);
        let dt = 0.1_f64;
        // Linear ramp: y = k*t → dy/dt = k, d²y/dt² = 0.
        for i in 0..20 {
            let v = 2.0 * (i as f64) * dt;
            assert!(mon.check(v, dt), "step {i}");
        }
    }

    #[test]
    fn step_change_triggers_violation() {
        let mut mon = GradientMonitor::new(1.0_f64);
        let dt = 0.01_f64;
        // Prime with two samples.
        mon.check(0.0, dt);
        mon.check(0.0, dt);
        // Sudden large step → 2nd derivative is enormous.
        let ok = mon.check(100.0, dt);
        assert!(!ok, "should detect large gradient");
    }

    #[test]
    fn reset_clears_state() {
        let mut mon = GradientMonitor::new(1.0_f64);
        mon.check(0.0, 0.01);
        mon.check(1.0, 0.01);
        mon.check(100.0, 0.01);
        mon.reset();
        // After reset, first two samples should pass without violation.
        assert!(mon.check(0.0, 0.01));
        assert!(mon.check(50.0, 0.01)); // only derivative set, no 2nd yet
        assert_eq!(mon.derivative(), 50.0 / 0.01);
    }
}
