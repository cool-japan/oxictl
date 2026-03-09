use crate::core::scalar::ControlScalar;

/// Absolute encoder with multi-turn tracking.
///
/// Models an absolute encoder (e.g. magnetic, optical, or SIN/COS based)
/// that provides a raw position value in [0, resolution).
///
/// Multi-turn tracking: wraps are detected when the raw position jumps by
/// more than half the resolution, and a turn counter is updated accordingly.
///
/// Velocity is estimated via finite backward difference.
#[derive(Debug, Clone, Copy)]
pub struct AbsoluteEncoder<S: ControlScalar> {
    /// Encoder resolution (counts per revolution), e.g. 8192 for 13-bit.
    pub resolution: u32,
    /// Raw position (0..resolution).
    position: u32,
    /// Full-turn counter (positive = CCW if convention).
    turns: i32,
    /// Estimated angular velocity (rad/s), low-pass filtered.
    omega: S,
    /// Previous full-turn angle for velocity estimation.
    theta_prev: S,
    /// Velocity filter coefficient (EMA, 0 < α ≤ 1).
    alpha: S,
    /// First-sample flag.
    initialized: bool,
}

impl<S: ControlScalar> AbsoluteEncoder<S> {
    /// Create an absolute encoder.
    ///
    /// - `resolution`: counts per revolution (e.g. 4096, 8192, 16384)
    /// - `alpha`: velocity EMA filter coefficient (1.0 = raw, 0.1 = heavy smoothing)
    pub fn new(resolution: u32, alpha: S) -> Self {
        Self {
            resolution,
            position: 0,
            turns: 0,
            omega: S::ZERO,
            theta_prev: S::ZERO,
            alpha,
            initialized: false,
        }
    }

    /// Update encoder with new raw reading.
    ///
    /// - `raw`: raw encoder count in [0, resolution)
    /// - `dt`: time since last update (s)
    pub fn update(&mut self, raw: u32, dt: S) {
        let raw = raw % self.resolution;
        let half = self.resolution / 2;

        if !self.initialized {
            self.position = raw;
            self.theta_prev = self.theta();
            self.initialized = true;
            return;
        }

        let prev = self.position;

        // Detect wrap-around
        if raw > prev + half {
            // Position decreased past 0 → turns--
            self.turns -= 1;
        } else if prev > raw + half {
            // Position increased past resolution → turns++
            self.turns += 1;
        }

        self.position = raw;

        // Velocity estimation
        if dt > S::ZERO {
            let theta_new = self.theta();
            let dtheta = theta_new - self.theta_prev;

            // Unwrap dtheta to [-π, π]
            let pi = S::PI;
            let two_pi = S::TWO * pi;
            let dtheta = if dtheta > pi {
                dtheta - two_pi
            } else if dtheta < -pi {
                dtheta + two_pi
            } else {
                dtheta
            };

            let omega_raw = dtheta / dt;
            // EMA filter
            self.omega += self.alpha * (omega_raw - self.omega);
            self.theta_prev = theta_new;
        }
    }

    /// Current angular position (rad), monotonic across full turns.
    ///
    /// theta = 2π·turns + 2π·position/resolution
    pub fn theta(&self) -> S {
        let two_pi = S::TWO * S::PI;
        let frac = S::from_f64(self.position as f64 / self.resolution as f64);
        S::from_f64(self.turns as f64) * two_pi + frac * two_pi
    }

    /// Angle within current revolution [0, 2π).
    pub fn angle_within_turn(&self) -> S {
        S::from_f64(self.position as f64 / self.resolution as f64) * S::TWO * S::PI
    }

    /// Estimated angular velocity (rad/s).
    pub fn omega(&self) -> S {
        self.omega
    }

    /// Current turn count (full revolutions from reset).
    pub fn turns(&self) -> i32 {
        self.turns
    }

    /// Raw position count.
    pub fn raw_position(&self) -> u32 {
        self.position
    }

    pub fn reset(&mut self) {
        self.position = 0;
        self.turns = 0;
        self.omega = S::ZERO;
        self.theta_prev = S::ZERO;
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn angle_within_turn_correct() {
        let mut enc = AbsoluteEncoder::new(4096_u32, 1.0_f64);
        enc.update(1024, 0.001); // 1/4 turn
        let expected = 2.0 * PI / 4.0;
        assert!(
            (enc.angle_within_turn() - expected).abs() < 0.01,
            "angle={:.4}",
            enc.angle_within_turn()
        );
    }

    #[test]
    fn multi_turn_tracking_forward() {
        let mut enc = AbsoluteEncoder::new(4096_u32, 1.0_f64);
        enc.update(3000, 0.001); // initialize at 3000
        enc.update(100, 0.001); // wrap forward (3000 → 100: big backward jump → turns++)
        assert_eq!(enc.turns(), 1, "Should count one forward turn");
    }

    #[test]
    fn multi_turn_tracking_backward() {
        let mut enc = AbsoluteEncoder::new(4096_u32, 1.0_f64);
        enc.update(100, 0.001); // initialize at 100
        enc.update(3900, 0.001); // wrap backward (100 → 3900: big forward jump → turns--)
        assert_eq!(enc.turns(), -1, "Should count one backward turn");
    }

    #[test]
    fn velocity_estimation() {
        let mut enc = AbsoluteEncoder::new(4096_u32, 1.0_f64);
        let dt = 0.001_f64;
        let omega_true = 2.0 * PI * 10.0; // 10 Hz = ~62.8 rad/s
        let counts_per_step = (omega_true * dt / (2.0 * PI) * 4096.0) as u32;

        let mut pos = 0u32;
        for _ in 0..200 {
            pos = (pos + counts_per_step) % 4096;
            enc.update(pos, dt);
        }

        // After 200 steps, velocity should be close to true omega
        assert!(
            (enc.omega() - omega_true).abs() < 5.0,
            "omega={:.2}, true={:.2}",
            enc.omega(),
            omega_true
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut enc = AbsoluteEncoder::new(4096_u32, 1.0_f64);
        enc.update(2000, 0.001);
        enc.update(100, 0.001);
        enc.reset();
        assert_eq!(enc.turns(), 0);
        assert!((enc.omega()).abs() < 1e-10);
    }
}
