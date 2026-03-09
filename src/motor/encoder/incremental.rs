use crate::core::scalar::ControlScalar;

/// Incremental encoder state decoder and velocity estimator.
///
/// Supports quadrature encoding (A + B channels) and velocity estimation
/// via a derivative filter on the position.
#[derive(Debug, Clone)]
pub struct IncrementalEncoder<S: ControlScalar> {
    /// Encoder counts per revolution.
    counts_per_rev: i32,
    /// Current count.
    count: i32,
    /// Previous count for velocity estimation.
    prev_count: i32,
    /// Estimated velocity (revolutions/second).
    velocity: S,
    /// Velocity filter alpha (0..1, higher = more filtering).
    vel_alpha: S,
}

impl<S: ControlScalar> IncrementalEncoder<S> {
    pub fn new(counts_per_rev: i32) -> Self {
        Self {
            counts_per_rev,
            count: 0,
            prev_count: 0,
            velocity: S::ZERO,
            vel_alpha: S::from_f64(0.1), // default: slight smoothing
        }
    }

    pub fn with_velocity_filter(mut self, alpha: S) -> Self {
        self.vel_alpha = alpha;
        self
    }

    /// Update with new encoder count. dt = time since last update (seconds).
    pub fn update(&mut self, new_count: i32, dt: S) {
        self.count = new_count;
        if dt > S::ZERO {
            let delta = self.count - self.prev_count;
            let cpr = S::from_f64(self.counts_per_rev as f64);
            let raw_vel = S::from_f64(delta as f64) / (cpr * dt);
            // Low-pass filter velocity
            self.velocity = self.vel_alpha * self.velocity + (S::ONE - self.vel_alpha) * raw_vel;
        }
        self.prev_count = self.count;
    }

    /// Update from pulse direction: +1 or -1 per pulse.
    pub fn pulse(&mut self, direction: i32, dt: S) {
        self.update(self.count + direction, dt);
    }

    /// Position in revolutions.
    pub fn position_rev(&self) -> S {
        S::from_f64(self.count as f64) / S::from_f64(self.counts_per_rev as f64)
    }

    /// Position in radians.
    pub fn position_rad(&self) -> S {
        self.position_rev() * S::TWO * S::PI
    }

    /// Position in degrees.
    pub fn position_deg(&self) -> S {
        self.position_rev() * S::from_f64(360.0)
    }

    /// Estimated velocity (rev/s).
    pub fn velocity_rev_s(&self) -> S {
        self.velocity
    }

    /// Estimated velocity (rad/s).
    pub fn velocity_rad_s(&self) -> S {
        self.velocity * S::TWO * S::PI
    }

    /// Raw count.
    pub fn count(&self) -> i32 {
        self.count
    }

    /// Reset encoder.
    pub fn reset(&mut self) {
        self.count = 0;
        self.prev_count = 0;
        self.velocity = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_at_start() {
        let enc = IncrementalEncoder::<f64>::new(1000);
        assert_eq!(enc.count(), 0);
        assert_eq!(enc.position_rev(), 0.0);
    }

    #[test]
    fn one_revolution() {
        let mut enc = IncrementalEncoder::<f64>::new(1000);
        enc.update(1000, 0.01);
        assert!((enc.position_rev() - 1.0).abs() < 1e-10);
        assert!((enc.position_deg() - 360.0).abs() < 1e-10);
    }

    #[test]
    fn velocity_estimation() {
        let mut enc = IncrementalEncoder::<f64>::new(1000).with_velocity_filter(0.0);
        // 500 counts in 0.5s = 1 rev/s
        enc.update(0, 0.001);
        enc.update(500, 0.5);
        // Raw velocity = 500 / (1000 * 0.5) = 1.0 rev/s
        assert!(
            (enc.velocity_rev_s() - 1.0).abs() < 0.01,
            "vel={}",
            enc.velocity_rev_s()
        );
    }

    #[test]
    fn pulse_increments() {
        let mut enc = IncrementalEncoder::<f64>::new(200);
        enc.pulse(1, 0.001);
        enc.pulse(1, 0.001);
        enc.pulse(1, 0.001);
        assert_eq!(enc.count(), 3);
    }

    #[test]
    fn negative_direction() {
        let mut enc = IncrementalEncoder::<f64>::new(1000);
        enc.update(-500, 0.1);
        assert!(enc.position_rev() < 0.0);
        assert!(enc.velocity_rev_s() < 0.0);
    }

    #[test]
    fn reset_clears() {
        let mut enc = IncrementalEncoder::<f64>::new(1000);
        enc.update(5000, 1.0);
        enc.reset();
        assert_eq!(enc.count(), 0);
        assert_eq!(enc.velocity_rev_s(), 0.0);
    }
}
