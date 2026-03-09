use crate::core::scalar::ControlScalar;

/// Complementary filter for IMU-based attitude estimation.
///
/// Combines a high-pass filtered gyroscope signal with a low-pass filtered
/// accelerometer/magnetometer signal:
///
///   θ[k] = α * (θ[k-1] + ω[k]*dt) + (1-α) * θ_acc[k]
///
/// where α = tau/(tau + dt) controls the crossover frequency.
#[derive(Debug, Clone, Copy)]
pub struct ComplementaryFilter<S: ControlScalar> {
    /// Filter time constant (seconds). Larger → trust gyro more.
    tau: S,
    /// Current angle estimate (radians or degrees, matching input units).
    angle: S,
    initialized: bool,
}

impl<S: ControlScalar> ComplementaryFilter<S> {
    pub fn new(tau: S) -> Self {
        Self {
            tau,
            angle: S::ZERO,
            initialized: false,
        }
    }

    /// Initialize with a known angle.
    pub fn with_initial(tau: S, initial_angle: S) -> Self {
        Self {
            tau,
            angle: initial_angle,
            initialized: true,
        }
    }

    /// Update the filter.
    /// - `gyro_rate`: angular rate from gyroscope (rad/s or deg/s)
    /// - `accel_angle`: angle from accelerometer/reference (same units as gyro_rate * s)
    /// - `dt`: time step (seconds)
    pub fn update(&mut self, gyro_rate: S, accel_angle: S, dt: S) -> S {
        if !self.initialized {
            self.angle = accel_angle;
            self.initialized = true;
            return self.angle;
        }
        let alpha = self.tau / (self.tau + dt);
        // High-pass filtered gyro integration + low-pass filtered accelerometer
        self.angle = alpha * (self.angle + gyro_rate * dt) + (S::ONE - alpha) * accel_angle;
        self.angle
    }

    pub fn angle(&self) -> S {
        self.angle
    }

    pub fn reset(&mut self) {
        self.angle = S::ZERO;
        self.initialized = false;
    }

    pub fn reset_to(&mut self, angle: S) {
        self.angle = angle;
        self.initialized = true;
    }
}

/// Two-axis complementary filter (roll + pitch).
#[derive(Debug, Clone, Copy)]
pub struct ComplementaryFilter2D<S: ControlScalar> {
    pub roll: ComplementaryFilter<S>,
    pub pitch: ComplementaryFilter<S>,
}

impl<S: ControlScalar> ComplementaryFilter2D<S> {
    pub fn new(tau: S) -> Self {
        Self {
            roll: ComplementaryFilter::new(tau),
            pitch: ComplementaryFilter::new(tau),
        }
    }

    /// Update both axes.
    /// - `gx`, `gy`: gyro roll/pitch rates (rad/s)
    /// - `roll_acc`, `pitch_acc`: angles from accelerometer
    pub fn update(&mut self, gx: S, gy: S, roll_acc: S, pitch_acc: S, dt: S) -> (S, S) {
        let roll = self.roll.update(gx, roll_acc, dt);
        let pitch = self.pitch.update(gy, pitch_acc, dt);
        (roll, pitch)
    }

    pub fn reset(&mut self) {
        self.roll.reset();
        self.pitch.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sample_initializes_to_accel() {
        let mut f = ComplementaryFilter::<f64>::new(0.1);
        let angle = f.update(0.0, 0.5, 0.01);
        assert_eq!(angle, 0.5);
    }

    #[test]
    fn gyro_integration_dominates_with_high_alpha() {
        // High tau → trust gyro more
        let mut f = ComplementaryFilter::<f64>::with_initial(10.0, 0.0);
        let angle = f.update(1.0, 0.0, 0.01); // gyro says rotate 1 rad/s, accel says 0
                                              // alpha = 10/(10+0.01) ≈ 0.999, so result ≈ 0.999 * (0 + 1*0.01) + 0.001*0 ≈ 0.00999
        assert!(angle > 0.009 && angle < 0.011, "got {}", angle);
    }

    #[test]
    fn accel_dominates_with_low_alpha() {
        // Low tau → trust accelerometer more
        let mut f = ComplementaryFilter::<f64>::with_initial(0.001, 0.0);
        let angle = f.update(1.0, 1.0, 0.01); // gyro: 1rad/s, accel: 1rad
                                              // alpha = 0.001/(0.001+0.01) ≈ 0.0909
                                              // result ≈ 0.0909*(0+0.01) + 0.9091*1.0 ≈ 0.9100
        assert!(angle > 0.90 && angle < 0.92, "got {}", angle);
    }

    #[test]
    fn gyro_drift_rejected() {
        // Pure gyro (alpha=1) would drift, but with accel correction it converges
        let mut f = ComplementaryFilter::<f64>::with_initial(0.5, 0.0);
        // True angle is 0, gyro has drift of 0.1 rad/s, accel reads 0.0
        for _ in 0..1000 {
            f.update(0.1, 0.0, 0.01); // gyro drifts, accel says 0
        }
        // After convergence, angle should be bounded (not unbounded drift)
        assert!(
            f.angle().abs() < 1.0,
            "Gyro drift should be bounded: {}",
            f.angle()
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut f = ComplementaryFilter::<f64>::with_initial(0.5, 1.0);
        f.update(1.0, 1.0, 0.01);
        f.reset();
        let angle = f.update(0.0, 0.5, 0.01);
        assert_eq!(angle, 0.5); // re-initializes from accel
    }

    #[test]
    fn two_axis_filter() {
        let mut f = ComplementaryFilter2D::<f64>::new(0.1);
        let (roll, pitch) = f.update(0.0, 0.0, 0.3, 0.2, 0.01);
        assert_eq!(roll, 0.3); // initialized to accel
        assert_eq!(pitch, 0.2);
    }
}
