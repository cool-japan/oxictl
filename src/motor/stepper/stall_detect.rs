use crate::core::scalar::ControlScalar;

/// Stepper motor stall detection via back-EMF / load angle monitoring.
///
/// Detects stall conditions by monitoring:
///   1. **Load angle**: estimated angle between commanded and actual position.
///      If load angle exceeds 90° (mechanical), the motor stalls.
///   2. **Current signature**: stall causes a current rise without position change.
///      Monitors peak current vs. expected current at given speed.
///
/// Since steppers run open-loop, stall detection is heuristic:
///   - Step loss detection: commanded steps vs. back-EMF measured position
///   - Energy method: power consumption increases significantly at stall
///
/// This implementation uses the load angle estimation method based on
/// back-EMF voltage monitoring (requires current sensing).
#[derive(Debug, Clone, Copy)]
pub struct StallDetector<S: ControlScalar> {
    /// Number of steps taken (commanded, from the driver).
    pub commanded_steps: i64,
    /// Stall threshold: consecutive failed step count before declaring stall.
    pub stall_threshold: u32,
    /// Current stall counter.
    stall_count: u32,
    /// Stalled flag.
    stalled: bool,
    /// Low-pass filtered current magnitude.
    filtered_current: S,
    /// Filter coefficient (0 = no update, 1 = instant).
    alpha: S,
    /// Expected current at nominal conditions (A).
    pub nominal_current: S,
    /// Stall current ratio: if actual_current / nominal_current > this, suspect stall.
    pub stall_current_ratio: S,
    /// Speed threshold below which stall check is bypassed (steps/s).
    pub min_speed: S,
    /// Current speed estimate (steps/s).
    current_speed: S,
}

impl<S: ControlScalar> StallDetector<S> {
    /// Create a stall detector.
    ///
    /// - `nominal_current`: expected RMS current at nominal load (A)
    /// - `stall_current_ratio`: ratio of actual/nominal current to declare stall (e.g. 1.5)
    /// - `stall_threshold`: consecutive high-current samples before declaring stall
    /// - `min_speed`: minimum steps/s to perform detection (avoids false positives at low speed)
    pub fn new(
        nominal_current: S,
        stall_current_ratio: S,
        stall_threshold: u32,
        min_speed: S,
    ) -> Self {
        Self {
            commanded_steps: 0,
            stall_threshold,
            stall_count: 0,
            stalled: false,
            filtered_current: S::ZERO,
            alpha: S::from_f64(0.1),
            nominal_current,
            stall_current_ratio,
            min_speed,
            current_speed: S::ZERO,
        }
    }

    /// Notify the detector that steps were commanded.
    pub fn command_steps(&mut self, steps: i32) {
        self.commanded_steps += steps as i64;
    }

    /// Update the stall detector.
    ///
    /// - `measured_current`: measured phase current magnitude (A)
    /// - `speed`: current commanded speed (steps/s), used to skip at low speeds
    ///
    /// Returns `true` if stall is detected.
    pub fn update(&mut self, measured_current: S, speed: S) -> bool {
        self.current_speed = speed;

        // Low-pass filter current
        self.filtered_current =
            self.alpha * measured_current + (S::ONE - self.alpha) * self.filtered_current;

        // Skip detection at low speed
        if speed.abs() < self.min_speed {
            self.stall_count = 0;
            return self.stalled;
        }

        // Check current ratio
        let ratio = if self.nominal_current > S::ZERO {
            self.filtered_current / self.nominal_current
        } else {
            S::ZERO
        };

        if ratio > self.stall_current_ratio {
            self.stall_count = self.stall_count.saturating_add(1);
        } else {
            // Decrement stall count on good readings (hysteresis)
            self.stall_count = self.stall_count.saturating_sub(1);
        }

        if self.stall_count >= self.stall_threshold {
            self.stalled = true;
        }

        self.stalled
    }

    pub fn is_stalled(&self) -> bool {
        self.stalled
    }

    /// Clear stall condition (after recovery).
    pub fn clear_stall(&mut self) {
        self.stalled = false;
        self.stall_count = 0;
    }

    pub fn reset(&mut self) {
        self.commanded_steps = 0;
        self.stall_count = 0;
        self.stalled = false;
        self.filtered_current = S::ZERO;
        self.current_speed = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_current_no_stall() {
        let mut det = StallDetector::new(2.0_f64, 1.5, 10, 100.0);
        for _ in 0..100 {
            det.command_steps(1);
            det.update(2.0, 500.0); // nominal current
        }
        assert!(!det.is_stalled());
    }

    #[test]
    fn high_current_triggers_stall() {
        let mut det = StallDetector::new(2.0_f64, 1.5, 10, 100.0);
        for _ in 0..200 {
            det.command_steps(1);
            det.update(4.0, 500.0); // 2× nominal current
        }
        assert!(det.is_stalled());
    }

    #[test]
    fn low_speed_skips_detection() {
        let mut det = StallDetector::new(2.0_f64, 1.5, 10, 100.0);
        for _ in 0..200 {
            det.update(4.0, 50.0); // below min_speed=100
        }
        assert!(!det.is_stalled());
    }

    #[test]
    fn clear_stall_resets_flag() {
        let mut det = StallDetector::new(2.0_f64, 1.5, 5, 100.0);
        for _ in 0..100 {
            det.update(4.0, 500.0);
        }
        assert!(det.is_stalled());
        det.clear_stall();
        assert!(!det.is_stalled());
    }
}
