use crate::core::scalar::ControlScalar;

/// Full-step stepper motor driver.
///
/// Generates the coil excitation sequence for a 4-phase stepper motor.
/// Supports full-step (4 steps/cycle) and wave-drive (1 coil at a time).
///
/// Phase sequence (full-step, both coils energized):
///   Step 0: A+, B+  → [1, 0, 1, 0]
///   Step 1: A-, B+  → [0, 1, 1, 0]
///   Step 2: A-, B-  → [0, 1, 0, 1]
///   Step 3: A+, B-  → [1, 0, 0, 1]
#[derive(Debug, Clone, Copy)]
pub struct FullStepDriver<S: ControlScalar> {
    /// Steps per electrical revolution.
    steps_per_rev: u32,
    /// Current step position (counts).
    step_count: i64,
    /// Step angle in radians.
    step_angle: S,
}

/// Coil activation pattern [A+, A-, B+, B-].
pub type PhasePattern = [bool; 4];

// Full-step sequence (2-phase energized)
const FULL_STEP_SEQ: [PhasePattern; 4] = [
    [true, false, true, false], // 0: A+B+
    [false, true, true, false], // 1: A-B+
    [false, true, false, true], // 2: A-B-
    [true, false, false, true], // 3: A+B-
];

// Wave-drive sequence (1-phase energized, lower torque, less heat)
const WAVE_DRIVE_SEQ: [PhasePattern; 4] = [
    [true, false, false, false], // 0: A+
    [false, false, true, false], // 1: B+
    [false, true, false, false], // 2: A-
    [false, false, false, true], // 3: B-
];

impl<S: ControlScalar> FullStepDriver<S> {
    /// Create a full-step driver.
    ///
    /// `steps_per_rev`: motor steps per mechanical revolution (typical: 200).
    pub fn new(steps_per_rev: u32) -> Self {
        let step_angle = S::TWO * S::PI / S::from_f64(steps_per_rev as f64);
        Self {
            steps_per_rev,
            step_count: 0,
            step_angle,
        }
    }

    /// Advance by `steps` (positive = forward, negative = reverse).
    pub fn step(&mut self, steps: i32) {
        self.step_count += i64::from(steps);
    }

    /// Get current full-step phase pattern.
    pub fn phase_pattern(&self) -> PhasePattern {
        let idx = self.step_count.rem_euclid(4) as usize;
        FULL_STEP_SEQ[idx]
    }

    /// Get wave-drive phase pattern.
    pub fn wave_drive_pattern(&self) -> PhasePattern {
        let idx = self.step_count.rem_euclid(4) as usize;
        WAVE_DRIVE_SEQ[idx]
    }

    /// Current position in steps.
    pub fn position_steps(&self) -> i64 {
        self.step_count
    }

    /// Current position in radians.
    pub fn position_rad(&self) -> S {
        S::from_f64(self.step_count as f64) * self.step_angle
    }

    /// Current position in degrees.
    pub fn position_deg(&self) -> S {
        self.position_rad() * S::from_f64(180.0 / core::f64::consts::PI)
    }

    pub fn reset(&mut self) {
        self.step_count = 0;
    }

    pub fn steps_per_rev(&self) -> u32 {
        self.steps_per_rev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_pattern() {
        let d = FullStepDriver::<f64>::new(200);
        let p = d.phase_pattern();
        assert_eq!(p, FULL_STEP_SEQ[0]);
    }

    #[test]
    fn step_advances_pattern() {
        let mut d = FullStepDriver::<f64>::new(200);
        d.step(1);
        assert_eq!(d.phase_pattern(), FULL_STEP_SEQ[1]);
        d.step(1);
        assert_eq!(d.phase_pattern(), FULL_STEP_SEQ[2]);
    }

    #[test]
    fn pattern_wraps_after_4_steps() {
        let mut d = FullStepDriver::<f64>::new(200);
        d.step(4);
        assert_eq!(d.phase_pattern(), FULL_STEP_SEQ[0]);
    }

    #[test]
    fn reverse_direction() {
        let mut d = FullStepDriver::<f64>::new(200);
        d.step(-1);
        assert_eq!(d.phase_pattern(), FULL_STEP_SEQ[3]);
    }

    #[test]
    fn position_tracking() {
        let mut d = FullStepDriver::<f64>::new(200);
        d.step(100);
        assert_eq!(d.position_steps(), 100);
        let expected_deg = 100.0 * 360.0 / 200.0;
        assert!((d.position_deg() - expected_deg).abs() < 1e-6);
    }

    #[test]
    fn reset_zeroes_position() {
        let mut d = FullStepDriver::<f64>::new(200);
        d.step(50);
        d.reset();
        assert_eq!(d.position_steps(), 0);
    }
}
