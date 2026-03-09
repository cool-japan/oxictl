use crate::core::scalar::ControlScalar;

/// Error type for differential drive robot operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiffDriveError {
    /// A constructor parameter is invalid (e.g. negative wheel radius or dt).
    InvalidParameter,
    /// Wheelbase is zero or negative.
    ZeroWheelbase,
}

impl core::fmt::Display for DiffDriveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DiffDriveError::InvalidParameter => {
                write!(f, "invalid differential drive parameter")
            }
            DiffDriveError::ZeroWheelbase => write!(f, "wheelbase must be positive"),
        }
    }
}

/// Differential drive robot with unicycle kinematics.
///
/// State vector: `[x, y, theta]`
/// - `x`     : position along world X-axis (m)
/// - `y`     : position along world Y-axis (m)
/// - `theta` : heading angle measured counter-clockwise from X-axis (rad)
///
/// Kinematics from individual wheel surface velocities:
/// ```text
///   v     = (v_r + v_l) / 2          linear speed   (m/s)
///   omega = (v_r - v_l) / wheelbase  angular speed  (rad/s)
///   ẋ     = v · cos(θ)
///   ẏ     = v · sin(θ)
///   θ̇     = omega
/// ```
///
/// Integration: first-order Euler with fixed timestep `dt`.
///
/// Encoder accumulators track accumulated wheel rotation in radians:
/// `encoder_left  += v_l · dt / wheel_radius`
/// `encoder_right += v_r · dt / wheel_radius`
#[derive(Debug, Clone, Copy)]
pub struct DifferentialDrive<S: ControlScalar> {
    /// Current state `[x, y, theta]`.
    pub state: [S; 3],
    /// Distance between the contact points of the two wheels (m).
    pub wheelbase: S,
    /// Wheel radius (m).
    pub wheel_radius: S,
    /// Integration timestep (s).
    pub dt: S,
    /// Accumulated left-wheel rotation (rad).
    encoder_left: S,
    /// Accumulated right-wheel rotation (rad).
    encoder_right: S,
}

impl<S: ControlScalar> DifferentialDrive<S> {
    /// Construct a differential drive robot at the origin with zero heading.
    ///
    /// # Parameters
    /// - `wheelbase`    : distance between wheel contact points (m); must be positive.
    /// - `wheel_radius` : wheel radius (m); must be positive.
    /// - `dt`           : integration step (s); must be positive.
    pub fn new(wheelbase: S, wheel_radius: S, dt: S) -> Result<Self, DiffDriveError> {
        if wheelbase <= S::ZERO {
            return Err(DiffDriveError::ZeroWheelbase);
        }
        if wheel_radius <= S::ZERO {
            return Err(DiffDriveError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(DiffDriveError::InvalidParameter);
        }
        Ok(Self {
            state: [S::ZERO; 3],
            wheelbase,
            wheel_radius,
            dt,
            encoder_left: S::ZERO,
            encoder_right: S::ZERO,
        })
    }

    /// Advance the robot by one timestep using individual wheel velocities.
    ///
    /// - `v_l` : left  wheel surface speed (m/s)
    /// - `v_r` : right wheel surface speed (m/s)
    ///
    /// Returns the new state `[x, y, theta]`.
    pub fn step(&mut self, v_l: S, v_r: S) -> Result<[S; 3], DiffDriveError> {
        let dt = self.dt;
        let v = (v_r + v_l) * S::HALF;
        let omega = (v_r - v_l) / self.wheelbase;

        let theta = self.state[2];
        self.state[0] += dt * v * theta.cos();
        self.state[1] += dt * v * theta.sin();
        self.state[2] += dt * omega;

        // Accumulate encoder readings (radians of wheel rotation)
        self.encoder_left += v_l * dt / self.wheel_radius;
        self.encoder_right += v_r * dt / self.wheel_radius;

        Ok(self.state)
    }

    /// Alternative interface: advance using linear and angular velocity directly.
    ///
    /// Internally converts to individual wheel speeds:
    /// ```text
    ///   v_r = v + omega · wheelbase / 2
    ///   v_l = v − omega · wheelbase / 2
    /// ```
    ///
    /// Returns the new state `[x, y, theta]`.
    pub fn step_angular(&mut self, v: S, omega: S) -> Result<[S; 3], DiffDriveError> {
        let half_b = self.wheelbase * S::HALF;
        let v_r = v + omega * half_b;
        let v_l = v - omega * half_b;
        self.step(v_l, v_r)
    }

    /// Return the current odometry estimate (identical to the state `[x, y, theta]`).
    pub fn odometry(&self) -> [S; 3] {
        self.state
    }

    /// Reset position, heading, and encoder accumulators to zero.
    pub fn reset(&mut self) {
        self.state = [S::ZERO; 3];
        self.encoder_left = S::ZERO;
        self.encoder_right = S::ZERO;
    }

    /// Raw encoder readings `[left_rad, right_rad]` (accumulated wheel rotation).
    pub fn encoder_readings(&self) -> [S; 2] {
        [self.encoder_left, self.encoder_right]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_robot() -> DifferentialDrive<f64> {
        DifferentialDrive::<f64>::new(0.3, 0.05, 0.01).expect("valid params")
    }

    #[test]
    fn straight_line() {
        // Equal wheel speeds → straight line along world X-axis
        let mut robot = make_robot();
        for _ in 0..100 {
            robot.step(1.0, 1.0).expect("step ok");
        }
        let s = robot.state;
        // 100 steps × 0.01 s × 1.0 m/s = 1.0 m in X
        assert!(s[0] > 0.9, "x should increase: {}", s[0]);
        assert!(s[1].abs() < 1e-10, "y should stay 0: {}", s[1]);
        assert!(s[2].abs() < 1e-10, "theta should stay 0: {}", s[2]);
    }

    #[test]
    fn pure_rotation() {
        // v_l = −v_r → pure in-place rotation; position must stay near origin
        // and heading must change substantially.
        let mut robot = make_robot();
        let v = 0.3_f64;
        // omega = (v_r - v_l) / b = 2v / wheelbase
        let omega = 2.0 * v / 0.3;
        // Run for exactly 100 steps (1 second) and verify position stays pinned
        // at origin and heading grows proportionally to omega.
        let steps = 100_usize;
        let dt = 0.01_f64;
        for _ in 0..steps {
            robot.step(-v, v).expect("step ok");
        }
        let s = robot.state;
        // Position must stay at the origin (pure rotation has zero CoG displacement)
        let dist = (s[0] * s[0] + s[1] * s[1]).sqrt();
        assert!(
            dist < 1e-9,
            "robot should stay at origin during pure rotation: dist={}",
            dist
        );
        // Heading must be close to omega * elapsed_time (Euler accumulation is exact here)
        let expected_theta = omega * (steps as f64) * dt;
        assert!(
            (s[2] - expected_theta).abs() < 1e-10,
            "heading should match omega*t: expected={:.6}, got={:.6}",
            expected_theta,
            s[2]
        );
    }

    #[test]
    fn circle_arc() {
        // Unequal wheel speeds → circular arc.
        // Expected turning radius: R = b·(v_r + v_l) / (2·(v_r − v_l))
        let wheelbase = 0.3_f64;
        let mut robot =
            DifferentialDrive::<f64>::new(wheelbase, 0.05, 0.001).expect("valid params");
        let v_l = 0.8_f64;
        let v_r = 1.2_f64;
        let r_expected = wheelbase * (v_r + v_l) / (2.0 * (v_r - v_l));
        let v_avg = (v_r + v_l) / 2.0;
        let period = 2.0 * core::f64::consts::PI * r_expected / v_avg;
        let steps = (period / 0.001).ceil() as usize;
        for _ in 0..steps {
            robot.step(v_l, v_r).expect("step ok");
        }
        // After one complete circle the robot should be close to the origin
        let s = robot.state;
        let dist = (s[0] * s[0] + s[1] * s[1]).sqrt();
        assert!(
            dist < r_expected * 0.05,
            "after full arc dist={:.4} should be < {:.4}",
            dist,
            r_expected * 0.05
        );
    }

    #[test]
    fn invalid_params() {
        // Zero wheelbase
        assert!(DifferentialDrive::<f64>::new(0.0, 0.05, 0.01).is_err());
        // Negative wheelbase
        assert!(DifferentialDrive::<f64>::new(-0.3, 0.05, 0.01).is_err());
        // Zero wheel radius
        assert!(DifferentialDrive::<f64>::new(0.3, 0.0, 0.01).is_err());
        // Zero dt
        assert!(DifferentialDrive::<f64>::new(0.3, 0.05, 0.0).is_err());
    }

    #[test]
    fn step_angular_matches_step() {
        // step_angular(v, omega) must produce the same state as step(v_l, v_r)
        let b = 0.3_f64;
        let mut r1 = DifferentialDrive::<f64>::new(b, 0.05, 0.01).expect("ok");
        let mut r2 = DifferentialDrive::<f64>::new(b, 0.05, 0.01).expect("ok");
        let v = 1.0_f64;
        let omega = 0.5_f64;
        let v_r = v + omega * b / 2.0;
        let v_l = v - omega * b / 2.0;
        r1.step(v_l, v_r).expect("ok");
        r2.step_angular(v, omega).expect("ok");
        for i in 0..3 {
            assert!(
                (r1.state[i] - r2.state[i]).abs() < 1e-12,
                "state[{}] mismatch: {} vs {}",
                i,
                r1.state[i],
                r2.state[i]
            );
        }
    }

    #[test]
    fn reset_clears_state_and_encoders() {
        let mut robot = make_robot();
        for _ in 0..100 {
            robot.step(1.0, 0.8).expect("ok");
        }
        robot.reset();
        let s = robot.state;
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 0.0);
        assert_eq!(s[2], 0.0);
        let enc = robot.encoder_readings();
        assert_eq!(enc[0], 0.0);
        assert_eq!(enc[1], 0.0);
    }

    #[test]
    fn encoders_accumulate_correctly() {
        // At constant v_l=1 m/s, v_r=1 m/s for N steps:
        // encoder_left = encoder_right = N * dt * v / r_wheel
        let mut robot = make_robot();
        let v = 1.0_f64;
        let n = 50_usize;
        for _ in 0..n {
            robot.step(v, v).expect("ok");
        }
        let enc = robot.encoder_readings();
        let expected = v * (n as f64) * 0.01 / 0.05; // v*N*dt / wheel_radius
        assert!(
            (enc[0] - expected).abs() < 1e-10,
            "left encoder expected {:.4}, got {:.4}",
            expected,
            enc[0]
        );
        assert!(
            (enc[1] - expected).abs() < 1e-10,
            "right encoder expected {:.4}, got {:.4}",
            expected,
            enc[1]
        );
    }
}
