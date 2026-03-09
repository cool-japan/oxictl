//! Dead Reckoning for ground vehicles.
//!
//! Fuses wheel odometry and IMU (gyroscope) measurements via a complementary
//! filter to maintain an estimate of the vehicle pose [x, y, theta] and
//! velocity [v, omega].
//!
//! # Model
//! State: `[x, y, theta, v, omega]`
//!
//! From differential-drive wheel odometry:
//! ```text
//! v     = (v_r + v_l) / 2
//! omega = (v_r - v_l) / b        (b = wheelbase)
//! ```
//!
//! Euler integration:
//! ```text
//! x     += v * cos(theta) * dt
//! y     += v * sin(theta) * dt
//! theta += omega * dt
//! ```
//!
//! Complementary filter on heading:
//! ```text
//! theta_fused = alpha * theta_odom + (1 - alpha) * theta_imu_integrated
//! ```

use crate::core::scalar::ControlScalar;

/// Errors produced by the navigation module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationError {
    /// A parameter value is outside its valid range.
    InvalidParameter,
    /// A measurement value is invalid (e.g. NaN / Inf).
    InvalidMeasurement,
    /// The requested landmark index is out of range.
    InvalidLandmarkId,
    /// The requested node index is out of range.
    InvalidNode,
    /// The edge buffer is full.
    TooManyEdges,
    /// The system is singular (no unique solution).
    SingularSystem,
}

/// Dead-reckoning estimator for a differential-drive ground vehicle.
///
/// # State layout
/// `[x, y, theta, v, omega]`
/// - `x`, `y`    — position in the world frame (m)
/// - `theta`     — heading angle (rad), relative to the +x axis
/// - `v`         — forward linear velocity (m/s)
/// - `omega`     — angular velocity (rad/s)
///
/// # Usage
/// Call [`update_odometry`] each control tick with left/right wheel speeds,
/// and optionally [`update_imu`] with a gyroscope reading. Query pose with
/// [`pose`] and velocities with [`velocity`].
#[derive(Debug)]
pub struct DeadReckoning<S: ControlScalar> {
    /// Full state: [x, y, theta, v, omega].
    state: [S; 5],
    /// Wheelbase (distance between left and right wheels), in metres.
    wheelbase: S,
    /// Complementary filter weight for odometry heading (0 ≤ alpha ≤ 1).
    /// `alpha = 1.0` → pure odometry; `alpha = 0.0` → pure IMU integration.
    alpha: S,
    /// Fixed time step (s).
    dt: S,
    /// IMU-integrated heading angle (rad).
    theta_imu: S,
}

impl<S: ControlScalar> DeadReckoning<S> {
    /// Create a new dead-reckoning estimator.
    ///
    /// # Parameters
    /// - `wheelbase` — distance between left and right wheels (m), must be > 0.
    /// - `alpha`     — complementary filter weight for odometry (0 ≤ alpha ≤ 1).
    /// - `dt`        — fixed integration time step (s), must be > 0.
    ///
    /// # Errors
    /// Returns [`NavigationError::InvalidParameter`] if any parameter is
    /// non-positive or out of range.
    pub fn new(wheelbase: S, alpha: S, dt: S) -> Result<Self, NavigationError> {
        if wheelbase <= S::ZERO {
            return Err(NavigationError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(NavigationError::InvalidParameter);
        }
        if alpha < S::ZERO || alpha > S::ONE {
            return Err(NavigationError::InvalidParameter);
        }
        Ok(Self {
            state: [S::ZERO; 5],
            wheelbase,
            alpha,
            dt,
            theta_imu: S::ZERO,
        })
    }

    /// Update pose from wheel odometry.
    ///
    /// Computes linear velocity `v` and angular velocity `omega` from
    /// left/right wheel speeds, then performs an Euler integration step.
    /// The heading is the pure odometry-integrated heading; call
    /// [`update_imu`] afterwards to apply complementary fusion.
    ///
    /// # Parameters
    /// - `v_l` — left wheel speed (m/s).
    /// - `v_r` — right wheel speed (m/s).
    ///
    /// # Errors
    /// Returns [`NavigationError::InvalidMeasurement`] if either value is
    /// non-finite.
    pub fn update_odometry(&mut self, v_l: S, v_r: S) -> Result<(), NavigationError> {
        if !v_l.is_finite() || !v_r.is_finite() {
            return Err(NavigationError::InvalidMeasurement);
        }

        let two = S::TWO;
        let v = (v_r + v_l) / two;
        let omega = (v_r - v_l) / self.wheelbase;

        // Store velocities.
        self.state[3] = v;
        self.state[4] = omega;

        let theta = self.state[2];

        // Euler integration of position and heading.
        let cos_theta = S::from_f64(libm::cos(theta.to_f64()));
        let sin_theta = S::from_f64(libm::sin(theta.to_f64()));

        self.state[0] += v * cos_theta * self.dt;
        self.state[1] += v * sin_theta * self.dt;
        self.state[2] = theta + omega * self.dt;

        Ok(())
    }

    /// Update heading using a gyroscope reading and apply complementary fusion.
    ///
    /// Integrates `gyro_z` (angular rate about the z axis, rad/s) to obtain
    /// an IMU-derived heading estimate, then blends it with the odometry
    /// heading via the complementary filter weight `alpha`.
    ///
    /// # Parameters
    /// - `gyro_z` — z-axis angular rate from the IMU (rad/s).
    ///
    /// # Errors
    /// Returns [`NavigationError::InvalidMeasurement`] if the value is
    /// non-finite.
    pub fn update_imu(&mut self, gyro_z: S) -> Result<(), NavigationError> {
        if !gyro_z.is_finite() {
            return Err(NavigationError::InvalidMeasurement);
        }

        // Integrate IMU heading.
        self.theta_imu += gyro_z * self.dt;

        // Complementary filter fusion.
        let theta_odom = self.state[2];
        self.state[2] = self.alpha * theta_odom + (S::ONE - self.alpha) * self.theta_imu;

        Ok(())
    }

    /// Return the current pose estimate `[x, y, theta]`.
    #[inline]
    pub fn pose(&self) -> [S; 3] {
        [self.state[0], self.state[1], self.state[2]]
    }

    /// Return the current velocity estimate `[v, omega]`.
    #[inline]
    pub fn velocity(&self) -> [S; 2] {
        [self.state[3], self.state[4]]
    }

    /// Reset all state to zero (including the IMU integrator).
    pub fn reset(&mut self) {
        self.state = [S::ZERO; 5];
        self.theta_imu = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight-line motion at heading 0 must increase x only.
    #[test]
    fn straight_line_increases_x() {
        let mut dr = DeadReckoning::<f64>::new(0.5, 1.0, 0.1).unwrap();
        // Both wheels at 1 m/s → forward at 1 m/s, no rotation.
        for _ in 0..10 {
            dr.update_odometry(1.0, 1.0).unwrap();
        }
        let pose = dr.pose();
        assert!(pose[0] > 0.9, "x should be ~1.0, got {}", pose[0]);
        assert!(pose[1].abs() < 1e-10, "y should be 0, got {}", pose[1]);
        assert!(pose[2].abs() < 1e-10, "theta should be 0");
    }

    /// Pure rotation (no forward motion) must change only heading.
    #[test]
    fn pure_rotation_changes_heading() {
        let wheelbase = 0.5_f64;
        let dt = 0.1_f64;
        let mut dr = DeadReckoning::<f64>::new(wheelbase, 1.0, dt).unwrap();
        // v_r = -v_l → pure spin, no translation.
        let v_spin = 0.5_f64; // m/s
        for _ in 0..10 {
            dr.update_odometry(-v_spin, v_spin).unwrap();
        }
        let pose = dr.pose();
        // omega = (v_r - v_l) / b = 1.0 / 0.5 = 2.0 rad/s → 10 steps × 0.1 s × 2.0 = 2.0 rad
        let expected_theta = 2.0_f64;
        assert!(
            (pose[2] - expected_theta).abs() < 1e-9,
            "theta mismatch: got {}, expected {}",
            pose[2],
            expected_theta
        );
        // Position should remain at (0, 0) for pure rotation.
        assert!(pose[0].abs() < 1e-10, "x should be 0 for pure rotation");
        assert!(pose[1].abs() < 1e-10, "y should be 0 for pure rotation");
    }

    /// IMU fusion with alpha=0 uses pure IMU heading.
    #[test]
    fn imu_fusion_alpha_zero_uses_imu() {
        let mut dr = DeadReckoning::<f64>::new(0.5, 0.0, 0.1).unwrap();
        // One odometry step (no rotation).
        dr.update_odometry(1.0, 1.0).unwrap();
        // IMU says we rotated by 0.5 rad/s.
        dr.update_imu(5.0).unwrap(); // 5.0 rad/s * 0.1 s = 0.5 rad added
        let pose = dr.pose();
        // alpha=0 → purely IMU integrated. theta_imu = 0.5 rad.
        assert!(
            (pose[2] - 0.5).abs() < 1e-10,
            "with alpha=0 theta should equal imu-integrated heading: {}",
            pose[2]
        );
    }

    /// IMU fusion with alpha=1 uses pure odometry heading.
    #[test]
    fn imu_fusion_alpha_one_uses_odometry() {
        let mut dr = DeadReckoning::<f64>::new(0.5, 1.0, 0.1).unwrap();
        // Drive straight — theta_odom stays 0.
        dr.update_odometry(1.0, 1.0).unwrap();
        // IMU claims a large rotation — should be ignored.
        dr.update_imu(100.0).unwrap();
        let pose = dr.pose();
        assert!(
            pose[2].abs() < 1e-10,
            "with alpha=1 theta should equal odometry heading: {}",
            pose[2]
        );
    }

    /// Zero motion must leave state unchanged.
    #[test]
    fn zero_motion_state_unchanged() {
        let mut dr = DeadReckoning::<f64>::new(0.5, 0.8, 0.01).unwrap();
        for _ in 0..100 {
            dr.update_odometry(0.0, 0.0).unwrap();
            dr.update_imu(0.0).unwrap();
        }
        let pose = dr.pose();
        assert!(pose[0].abs() < 1e-12);
        assert!(pose[1].abs() < 1e-12);
        assert!(pose[2].abs() < 1e-12);
    }

    /// Complementary filter produces correct weighted average.
    #[test]
    fn complementary_filter_weighted_average() {
        let alpha = 0.7_f64;
        let dt = 0.1_f64;
        let mut dr = DeadReckoning::<f64>::new(0.5, alpha, dt).unwrap();
        // One odometry step with pure rotation.
        let omega_odom = 2.0_f64;
        let v_r = omega_odom * 0.5 / 2.0; // (v_r - v_l) / 0.5 = 2.0 → v_r = 0.5, v_l = -0.5
        let v_l = -v_r;
        dr.update_odometry(v_l, v_r).unwrap();
        let theta_odom = omega_odom * dt; // 0.2 rad

        // IMU at different rate.
        let gyro_z = 3.0_f64;
        dr.update_imu(gyro_z).unwrap();
        let theta_imu = gyro_z * dt; // 0.3 rad

        let expected = alpha * theta_odom + (1.0 - alpha) * theta_imu;
        let pose = dr.pose();
        assert!(
            (pose[2] - expected).abs() < 1e-10,
            "complementary filter: got {}, expected {}",
            pose[2],
            expected
        );
    }

    /// Invalid parameter: zero wheelbase.
    #[test]
    fn invalid_wheelbase_returns_error() {
        let result = DeadReckoning::<f64>::new(0.0, 0.5, 0.01);
        assert_eq!(result.unwrap_err(), NavigationError::InvalidParameter);
    }

    /// Invalid parameter: alpha > 1.
    #[test]
    fn invalid_alpha_returns_error() {
        let result = DeadReckoning::<f64>::new(0.5, 1.5, 0.01);
        assert_eq!(result.unwrap_err(), NavigationError::InvalidParameter);
    }

    /// Invalid measurement: NaN in odometry.
    #[test]
    fn nan_odometry_returns_error() {
        let mut dr = DeadReckoning::<f64>::new(0.5, 0.8, 0.01).unwrap();
        let result = dr.update_odometry(f64::NAN, 1.0);
        assert_eq!(result.unwrap_err(), NavigationError::InvalidMeasurement);
    }

    /// Reset clears all state.
    #[test]
    fn reset_clears_state() {
        let mut dr = DeadReckoning::<f64>::new(0.5, 0.8, 0.1).unwrap();
        dr.update_odometry(1.0, 1.0).unwrap();
        dr.update_imu(1.0).unwrap();
        dr.reset();
        let pose = dr.pose();
        assert!(pose[0].abs() < 1e-12);
        assert!(pose[1].abs() < 1e-12);
        assert!(pose[2].abs() < 1e-12);
        let vel = dr.velocity();
        assert!(vel[0].abs() < 1e-12);
        assert!(vel[1].abs() < 1e-12);
    }
}
