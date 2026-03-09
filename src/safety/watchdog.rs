use crate::core::scalar::ControlScalar;

/// A software watchdog timer. Must be "kicked" periodically or it trips.
#[derive(Debug, Clone)]
pub struct Watchdog<S: ControlScalar> {
    timeout: S,
    elapsed: S,
    tripped: bool,
}

impl<S: ControlScalar> Watchdog<S> {
    /// Create a new watchdog with the given timeout period (seconds).
    pub fn new(timeout: S) -> Self {
        Self {
            timeout,
            elapsed: S::ZERO,
            tripped: false,
        }
    }

    /// Reset the watchdog timer (kick it).
    pub fn kick(&mut self) {
        self.elapsed = S::ZERO;
    }

    /// Advance the watchdog by dt seconds. Returns true if the watchdog just tripped.
    pub fn check(&mut self, dt: S) -> bool {
        if self.tripped {
            return true;
        }
        self.elapsed += dt;
        if self.elapsed >= self.timeout {
            self.tripped = true;
            true
        } else {
            false
        }
    }

    /// Whether the watchdog has tripped.
    pub fn is_tripped(&self) -> bool {
        self.tripped
    }

    /// Reset the watchdog completely (clear tripped state).
    pub fn reset(&mut self) {
        self.elapsed = S::ZERO;
        self.tripped = false;
    }

    pub fn timeout(&self) -> S {
        self.timeout
    }

    pub fn elapsed(&self) -> S {
        self.elapsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_watchdog_not_tripped() {
        let wd = Watchdog::<f64>::new(1.0);
        assert!(!wd.is_tripped());
    }

    #[test]
    fn trips_after_timeout() {
        let mut wd = Watchdog::<f64>::new(0.1);
        assert!(!wd.check(0.05));
        assert!(!wd.is_tripped());
        assert!(wd.check(0.06));
        assert!(wd.is_tripped());
    }

    #[test]
    fn kick_resets_timer() {
        let mut wd = Watchdog::<f64>::new(0.1);
        wd.check(0.08);
        assert!(!wd.is_tripped());
        wd.kick();
        assert!(!wd.check(0.08));
        assert!(!wd.is_tripped());
    }

    #[test]
    fn stays_tripped() {
        let mut wd = Watchdog::<f64>::new(0.1);
        wd.check(0.2);
        assert!(wd.is_tripped());
        // Even kicking doesn't clear tripped state (must call reset)
        wd.kick();
        assert!(wd.is_tripped());
    }

    #[test]
    fn reset_clears_trip() {
        let mut wd = Watchdog::<f64>::new(0.1);
        wd.check(0.2);
        assert!(wd.is_tripped());
        wd.reset();
        assert!(!wd.is_tripped());
        assert_eq!(wd.elapsed(), 0.0);
    }

    #[test]
    fn f32_watchdog() {
        let mut wd = Watchdog::<f32>::new(0.5);
        for _ in 0..4 {
            assert!(!wd.check(0.1));
        }
        assert!(wd.check(0.15));
        assert!(wd.is_tripped());
    }
}
