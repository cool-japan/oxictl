use crate::core::scalar::{ControlScalar, PidScalar};

/// Clamps a value between min and max limits.
#[derive(Debug, Clone, Copy)]
pub struct OutputLimiter<S: PidScalar> {
    min: S,
    max: S,
}

impl<S: PidScalar> OutputLimiter<S> {
    pub fn new(min: S, max: S) -> Self {
        debug_assert!(min <= max, "OutputLimiter: min must be <= max");
        Self { min, max }
    }

    /// Symmetric limiter: -limit to +limit.
    pub fn symmetric(limit: S) -> Self {
        Self::new(-limit, limit)
    }

    /// Apply the limit. Returns (clamped_value, was_saturated).
    pub fn apply(&self, value: S) -> (S, bool) {
        if value > self.max {
            (self.max, true)
        } else if value < self.min {
            (self.min, true)
        } else {
            (value, false)
        }
    }

    pub fn min(&self) -> S {
        self.min
    }

    pub fn max(&self) -> S {
        self.max
    }
}

/// Limits the rate of change of a signal.
#[derive(Debug, Clone, Copy)]
pub struct RateLimiter<S: ControlScalar> {
    max_rate: S,
    prev_value: Option<S>,
}

impl<S: ControlScalar> RateLimiter<S> {
    pub fn new(max_rate: S) -> Self {
        Self {
            max_rate,
            prev_value: None,
        }
    }

    /// Apply rate limiting. dt must be > 0.
    pub fn apply(&mut self, value: S, dt: S) -> S {
        match self.prev_value {
            None => {
                self.prev_value = Some(value);
                value
            }
            Some(prev) => {
                if dt <= S::ZERO {
                    self.prev_value = Some(value);
                    return value;
                }
                let delta = value - prev;
                let max_delta = self.max_rate * dt;
                let limited = if delta > max_delta {
                    prev + max_delta
                } else if delta < -max_delta {
                    prev - max_delta
                } else {
                    value
                };
                self.prev_value = Some(limited);
                limited
            }
        }
    }

    pub fn reset(&mut self) {
        self.prev_value = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_limiter_clamps_high() {
        let lim = OutputLimiter::new(0.0_f64, 10.0);
        let (v, sat) = lim.apply(15.0);
        assert_eq!(v, 10.0);
        assert!(sat);
    }

    #[test]
    fn output_limiter_clamps_low() {
        let lim = OutputLimiter::new(0.0_f64, 10.0);
        let (v, sat) = lim.apply(-5.0);
        assert_eq!(v, 0.0);
        assert!(sat);
    }

    #[test]
    fn output_limiter_passthrough() {
        let lim = OutputLimiter::new(0.0_f64, 10.0);
        let (v, sat) = lim.apply(5.0);
        assert_eq!(v, 5.0);
        assert!(!sat);
    }

    #[test]
    fn output_limiter_symmetric() {
        let lim = OutputLimiter::<f64>::symmetric(5.0);
        assert_eq!(lim.min(), -5.0);
        assert_eq!(lim.max(), 5.0);
    }

    #[test]
    fn output_limiter_boundary_values() {
        let lim = OutputLimiter::new(-1.0_f64, 1.0);
        let (v, sat) = lim.apply(1.0);
        assert_eq!(v, 1.0);
        assert!(!sat);

        let (v, sat) = lim.apply(-1.0);
        assert_eq!(v, -1.0);
        assert!(!sat);
    }

    #[test]
    fn rate_limiter_first_value_passthrough() {
        let mut rl = RateLimiter::new(10.0_f64);
        let v = rl.apply(100.0, 0.01);
        assert_eq!(v, 100.0);
    }

    #[test]
    fn rate_limiter_limits_increase() {
        let mut rl = RateLimiter::new(10.0_f64);
        rl.apply(0.0, 0.01);
        let v = rl.apply(100.0, 0.01);
        // max change = 10.0 * 0.01 = 0.1
        assert!((v - 0.1).abs() < 1e-10);
    }

    #[test]
    fn rate_limiter_limits_decrease() {
        let mut rl = RateLimiter::new(10.0_f64);
        rl.apply(100.0, 0.01);
        let v = rl.apply(0.0, 0.01);
        assert!((v - 99.9).abs() < 1e-10);
    }

    #[test]
    fn rate_limiter_within_rate() {
        let mut rl = RateLimiter::new(1000.0_f64);
        rl.apply(0.0, 0.01);
        let v = rl.apply(1.0, 0.01);
        assert_eq!(v, 1.0);
    }

    #[test]
    fn rate_limiter_reset() {
        let mut rl = RateLimiter::new(10.0_f64);
        rl.apply(100.0, 0.01);
        rl.reset();
        let v = rl.apply(200.0, 0.01);
        assert_eq!(v, 200.0);
    }
}
