use crate::core::scalar::ControlScalar;

/// Micro-step stepper motor driver.
///
/// Generates sinusoidal coil currents for smooth micro-stepping.
/// Reduces vibration and noise at the cost of higher computation.
///
/// For N micro-steps per full step:
///   Phase A current: I_peak * cos(θ)
///   Phase B current: I_peak * sin(θ)
/// where θ advances by 2π/(4*N) per micro-step.
#[derive(Debug, Clone, Copy)]
pub struct MicroStepDriver<S: ControlScalar> {
    /// Steps per mechanical revolution (full steps).
    full_steps_per_rev: u32,
    /// Micro-steps per full step.
    micro_per_full: u32,
    /// Peak coil current (normalized, 0..1).
    peak_current: S,
    /// Current electrical angle (accumulates continuously).
    electrical_angle: S,
    /// Micro-step count (signed).
    micro_count: i64,
}

impl<S: ControlScalar> MicroStepDriver<S> {
    /// Create a micro-step driver.
    ///
    /// `full_steps_per_rev`: full steps per mechanical revolution (e.g. 200).
    /// `micro_per_full`: micro-steps per full step (e.g. 8, 16, 32, 64).
    /// `peak_current`: normalized peak coil current (0..1).
    pub fn new(full_steps_per_rev: u32, micro_per_full: u32, peak_current: S) -> Self {
        Self {
            full_steps_per_rev,
            micro_per_full,
            peak_current,
            electrical_angle: S::ZERO,
            micro_count: 0,
        }
    }

    /// Advance by `micro_steps` (positive = forward, negative = reverse).
    pub fn step(&mut self, micro_steps: i32) {
        self.micro_count += i64::from(micro_steps);
        let total_micro_per_rev =
            S::from_f64((self.full_steps_per_rev * self.micro_per_full) as f64);
        let delta = S::TWO * S::PI / total_micro_per_rev;
        self.electrical_angle += delta * S::from_f64(f64::from(micro_steps));
    }

    /// Get coil currents [phase_a, phase_b].
    ///
    /// Returns normalized currents in [-peak_current, +peak_current].
    pub fn coil_currents(&self) -> (S, S) {
        let ia = self.peak_current * self.electrical_angle.cos();
        let ib = self.peak_current * self.electrical_angle.sin();
        (ia, ib)
    }

    /// Current position in micro-steps.
    pub fn position_microsteps(&self) -> i64 {
        self.micro_count
    }

    /// Current position in full steps (rounded).
    pub fn position_full_steps(&self) -> i64 {
        let m = self.micro_per_full as i64;
        if m == 0 {
            return 0;
        }
        self.micro_count / m
    }

    /// Current electrical angle (radians).
    pub fn electrical_angle(&self) -> S {
        self.electrical_angle
    }

    /// Current position in radians (mechanical).
    pub fn position_rad(&self) -> S {
        let total = S::from_f64((self.full_steps_per_rev * self.micro_per_full) as f64);
        S::TWO * S::PI * S::from_f64(self.micro_count as f64) / total
    }

    pub fn reset(&mut self) {
        self.micro_count = 0;
        self.electrical_angle = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_currents_at_zero_angle() {
        let d = MicroStepDriver::<f64>::new(200, 16, 1.0);
        let (ia, ib) = d.coil_currents();
        assert!((ia - 1.0).abs() < 1e-10); // cos(0) = 1
        assert!(ib.abs() < 1e-10); // sin(0) = 0
    }

    #[test]
    fn currents_bounded_by_peak() {
        let mut d = MicroStepDriver::<f64>::new(200, 16, 0.8);
        for step in 0..1000 {
            d.step(if step % 2 == 0 { 1 } else { -1 });
            let (ia, ib) = d.coil_currents();
            assert!(ia.abs() <= 0.8 + 1e-10);
            assert!(ib.abs() <= 0.8 + 1e-10);
        }
    }

    #[test]
    fn position_tracking() {
        let mut d = MicroStepDriver::<f64>::new(200, 16, 1.0);
        d.step(32); // 2 full steps
        assert_eq!(d.position_microsteps(), 32);
        assert_eq!(d.position_full_steps(), 2);
    }

    #[test]
    fn reset_zeroes_state() {
        let mut d = MicroStepDriver::<f64>::new(200, 16, 1.0);
        d.step(50);
        d.reset();
        assert_eq!(d.position_microsteps(), 0);
        let (ia, ib) = d.coil_currents();
        assert!((ia - 1.0).abs() < 1e-10);
        assert!(ib.abs() < 1e-10);
    }

    #[test]
    fn one_full_step_quarter_cycle() {
        let mut d = MicroStepDriver::<f64>::new(200, 4, 1.0);
        // 4 micro-steps = 1 full step = 90° electrical
        d.step(4);
        let (ia, ib) = d.coil_currents();
        let theta = d.electrical_angle();
        let expected_theta = 2.0 * core::f64::consts::PI / 200.0; // one full step angle
        assert!((theta - expected_theta).abs() < 1e-10);
        assert!((ia - theta.cos()).abs() < 1e-10);
        assert!((ib - theta.sin()).abs() < 1e-10);
    }
}
