use crate::core::scalar::ControlScalar;

/// Heartbeat monitor for periodic signal health checking.
///
/// Expects a signal to "beat" at least once every `period` time units.
/// If `max_missed` consecutive beats are missed, the monitor trips.
#[derive(Debug, Clone, Copy)]
pub struct Heartbeat<S: ControlScalar> {
    /// Expected beat period (s).
    pub period: S,
    /// Maximum consecutive missed beats before tripping.
    pub max_missed: u32,
    /// Elapsed time since last beat (s).
    elapsed: S,
    /// Number of consecutive missed beats.
    missed_count: u32,
    /// Whether the monitor has tripped.
    tripped: bool,
}

impl<S: ControlScalar> Heartbeat<S> {
    pub fn new(period: S, max_missed: u32) -> Self {
        Self {
            period,
            max_missed,
            elapsed: S::ZERO,
            missed_count: 0,
            tripped: false,
        }
    }

    /// Register a heartbeat signal. Resets the elapsed timer.
    pub fn beat(&mut self) {
        self.elapsed = S::ZERO;
        self.missed_count = 0;
    }

    /// Advance time by `dt`. Returns `true` if heartbeat is healthy.
    ///
    /// Should be called every control cycle with `dt` = cycle time.
    pub fn tick(&mut self, dt: S) -> bool {
        if self.tripped {
            return false;
        }
        self.elapsed += dt;
        if self.elapsed >= self.period {
            self.elapsed = S::ZERO;
            self.missed_count += 1;
            if self.missed_count > self.max_missed {
                self.tripped = true;
                return false;
            }
        }
        true
    }

    /// Whether the monitor has detected a missed-beat fault.
    pub fn is_tripped(&self) -> bool {
        self.tripped
    }

    /// Number of consecutive missed beats.
    pub fn missed_count(&self) -> u32 {
        self.missed_count
    }

    pub fn reset(&mut self) {
        self.elapsed = S::ZERO;
        self.missed_count = 0;
        self.tripped = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_heartbeat_does_not_trip() {
        let mut hb = Heartbeat::new(0.1_f64, 2);
        // Beat regularly every 50ms
        for _ in 0..10 {
            for _ in 0..5 {
                assert!(hb.tick(0.01));
            }
            hb.beat();
        }
        assert!(!hb.is_tripped());
    }

    #[test]
    fn missed_beats_trip_monitor() {
        let mut hb = Heartbeat::new(0.1_f64, 1); // allow 1 missed beat
                                                 // Advance 300ms without beating (3 periods → 3 missed)
        for _ in 0..300 {
            hb.tick(0.001);
        }
        assert!(hb.is_tripped());
    }

    #[test]
    fn single_miss_allowed_with_max_2() {
        let mut hb = Heartbeat::new(0.1_f64, 2);
        // One full period passes without beat (1 miss, allowed)
        for _ in 0..100 {
            hb.tick(0.001);
        }
        assert!(!hb.is_tripped(), "one miss should be tolerated");
        assert_eq!(hb.missed_count(), 1);

        // Beat arrives
        hb.beat();
        assert!(!hb.is_tripped());
    }

    #[test]
    fn reset_clears_trip() {
        let mut hb = Heartbeat::new(0.1_f64, 0);
        for _ in 0..200 {
            hb.tick(0.001);
        }
        assert!(hb.is_tripped());
        hb.reset();
        assert!(!hb.is_tripped());
        assert!(hb.tick(0.001));
    }
}
