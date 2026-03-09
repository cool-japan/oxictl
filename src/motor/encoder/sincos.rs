//! Analog sin/cos encoder interpolation for high-resolution position.
//!
//! A sin/cos encoder outputs two analog signals:
//!   S = A·sin(θ) + V_os    (sin channel)
//!   C = A·cos(θ) + V_oc    (cos channel)
//!
//! After removing DC offsets and normalizing gain, the angle is computed via:
//!   θ = atan2(sin_cal, cos_cal)
//!
//! An online min/max tracker performs automatic offset and gain calibration
//! as the encoder rotates through a full cycle.
#![allow(clippy::excessive_precision)]

use crate::core::scalar::ControlScalar;

/// Analog sin/cos encoder with online offset and gain calibration.
///
/// The calibration derives:
///   offset = (max + min) / 2
///   gain   = 2 / (max - min)
///
/// These are updated continuously from running min/max statistics.
#[derive(Debug, Clone, Copy)]
pub struct SincosEncoder<S: ControlScalar> {
    /// DC offset for sin channel (online calibration).
    pub sin_offset: S,
    /// DC offset for cos channel.
    pub cos_offset: S,
    /// Sin channel gain normalization factor (multiplied after offset subtraction).
    pub sin_gain: S,
    /// Cos channel gain normalization factor.
    pub cos_gain: S,
    // Online min/max tracking for auto-calibration
    sin_min: S,
    sin_max: S,
    cos_min: S,
    cos_max: S,
    /// Current angle estimate (rad), range (-π, π].
    pub angle: S,
    /// Angular velocity estimate (rad/s).
    pub velocity: S,
    /// Tracking loop bandwidth (rad/s); higher = faster response, more noise.
    pub bandwidth: S,
    // Previous raw angle for velocity estimation
    prev_angle: S,
}

impl<S: ControlScalar> SincosEncoder<S> {
    /// Create a new sin/cos encoder with given tracking bandwidth.
    ///
    /// Initial offsets are 0, gains are 1 (assume pre-calibrated or
    /// rely on online calibration to converge).
    pub fn new(bandwidth: S) -> Self {
        // Initialize min/max to extreme values so first sample sets them
        let big = S::from_f64(1e10);
        Self {
            sin_offset: S::ZERO,
            cos_offset: S::ZERO,
            sin_gain: S::ONE,
            cos_gain: S::ONE,
            sin_min: big,
            sin_max: -big,
            cos_min: big,
            cos_max: -big,
            angle: S::ZERO,
            velocity: S::ZERO,
            bandwidth,
            prev_angle: S::ZERO,
        }
    }

    /// Process raw sin/cos samples; updates angle and velocity.
    ///
    /// Algorithm:
    /// 1. Apply calibration (offset subtraction and gain normalization).
    /// 2. Update min/max trackers for online calibration.
    /// 3. Compute angle via atan2.
    /// 4. Estimate velocity from angle difference (with unwrapping).
    pub fn update(&mut self, sin_raw: S, cos_raw: S, dt: S) {
        // Apply current calibration
        let (sin_cal, cos_cal) = self.calibrate_sample(sin_raw, cos_raw);

        // Update min/max for next calibration cycle
        self.update_minmax(sin_cal, cos_cal);

        // Compute angle
        let new_angle = sin_cal.atan2(cos_cal);

        // Unwrap angle delta for velocity computation
        let two_pi = S::TWO * S::PI;
        let mut delta = new_angle - self.prev_angle;
        // Wrap to (-π, π)
        if delta > S::PI {
            delta -= two_pi;
        } else if delta < -S::PI {
            delta += two_pi;
        }

        self.angle = new_angle;
        self.prev_angle = new_angle;

        // Velocity estimate: differentiate angle
        if dt > S::EPSILON {
            // First-order low-pass filter on velocity:
            // v_new = v_old + bw * dt * (delta/dt - v_old)
            // Simplified: one-step Euler
            let raw_vel = delta / dt;
            let alpha = (self.bandwidth * dt).min(S::ONE);
            self.velocity = self.velocity + alpha * (raw_vel - self.velocity);
        }
    }

    /// Apply calibration: remove DC offset and normalize gain.
    ///
    /// sin_cal = (sin_raw - sin_offset) * sin_gain
    /// cos_cal = (cos_raw - cos_offset) * cos_gain
    fn calibrate_sample(&self, sin_raw: S, cos_raw: S) -> (S, S) {
        let sin_cal = (sin_raw - self.sin_offset) * self.sin_gain;
        let cos_cal = (cos_raw - self.cos_offset) * self.cos_gain;
        (sin_cal, cos_cal)
    }

    /// Update running min/max and recompute offsets and gains.
    ///
    /// Called after calibrate_sample with the calibrated values.
    /// The calibration updates gently so that initialization converges
    /// over several full rotations.
    fn update_minmax(&mut self, sin_cal: S, cos_cal: S) {
        // Track min/max of calibrated signal
        if sin_cal < self.sin_min {
            self.sin_min = sin_cal;
        }
        if sin_cal > self.sin_max {
            self.sin_max = sin_cal;
        }
        if cos_cal < self.cos_min {
            self.cos_min = cos_cal;
        }
        if cos_cal > self.cos_max {
            self.cos_max = cos_cal;
        }

        // Recompute offset and gain from current min/max (in raw terms)
        // sin_offset (adjustment from current calibration):
        let sin_range = self.sin_max - self.sin_min;
        let cos_range = self.cos_max - self.cos_min;

        // Only update calibration once we have enough excitation
        if sin_range > S::from_f64(0.1) {
            let sin_center = (self.sin_max + self.sin_min) * S::HALF;
            // Adjust offset: current calibration gives sin_cal = (raw - off)*gain
            // Center of calibrated signal should be 0; adjust offset accordingly
            // off_new = off + center/gain
            self.sin_offset += sin_center / self.sin_gain;
            // Adjust gain so that range becomes 2.0 (i.e., -1..+1)
            if sin_range > S::EPSILON {
                let target_range = S::TWO;
                self.sin_gain *= target_range / sin_range;
            }
            // Reset min/max after update
            let big = S::from_f64(1e10);
            self.sin_min = big;
            self.sin_max = -big;
        }

        if cos_range > S::from_f64(0.1) {
            let cos_center = (self.cos_max + self.cos_min) * S::HALF;
            self.cos_offset += cos_center / self.cos_gain;
            if cos_range > S::EPSILON {
                let target_range = S::TWO;
                self.cos_gain *= target_range / cos_range;
            }
            let big = S::from_f64(1e10);
            self.cos_min = big;
            self.cos_max = -big;
        }
    }

    /// Return current angle in degrees.
    pub fn angle_deg(&self) -> S {
        self.angle * S::from_f64(180.0 / core::f64::consts::PI)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_at_zero() {
        let mut enc = SincosEncoder::<f32>::new(100.0);
        // sin(0)=0, cos(0)=1 → atan2(0,1) = 0
        enc.update(0.0, 1.0, 0.001);
        assert!(enc.angle.abs() < 1e-5, "angle={}", enc.angle);
    }

    #[test]
    fn test_angle_at_90_degrees() {
        let mut enc = SincosEncoder::<f32>::new(100.0);
        // sin(π/2)=1, cos(π/2)=0 → atan2(1,0) = π/2
        enc.update(1.0, 0.0, 0.001);
        let expected = core::f32::consts::PI / 2.0;
        assert!(
            (enc.angle - expected).abs() < 1e-5,
            "angle={}, expected={expected}",
            enc.angle
        );
    }

    #[test]
    fn test_velocity_estimation() {
        let mut enc = SincosEncoder::<f32>::new(10.0);
        let dt = 0.001_f32;
        // Simulate constant rotation: θ increases by 0.1 rad/step
        let omega = 100.0_f32; // rad/s
        let mut theta = 0.0_f32;
        for _ in 0..10000 {
            theta += omega * dt;
            enc.update(theta.sin(), theta.cos(), dt);
        }
        // After many steps, velocity should converge toward omega
        let v = enc.velocity;
        assert!((v - omega).abs() < 5.0, "velocity={v}, expected≈{omega}");
    }

    #[test]
    fn test_angle_at_180_degrees() {
        let mut enc = SincosEncoder::<f32>::new(100.0);
        // sin(π)≈0, cos(π)=-1 → atan2(0,-1) = π
        enc.update(0.0, -1.0, 0.001);
        assert!(
            (enc.angle.abs() - core::f32::consts::PI).abs() < 1e-4,
            "angle={}",
            enc.angle
        );
    }
}
