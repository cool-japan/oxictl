use crate::core::scalar::ControlScalar;

/// FOC position control loop.
///
/// Outer loop that converts a position error into a speed reference
/// for the inner speed/current loops. Implements a P or PD controller
/// on position with configurable limits.
///
/// Typical cascaded FOC structure:
///   position_loop → speed_ref → SpeedLoop → torque_ref → CurrentLoop
pub struct PositionLoop<S: ControlScalar> {
    /// Proportional gain on position error.
    pub kp: S,
    /// Derivative gain on position error (velocity feedforward).
    pub kd: S,
    /// Maximum speed reference output (rad/s or deg/s).
    pub speed_limit: S,
    /// Previous position for derivative calculation.
    prev_pos: S,
    /// Whether prev_pos has been initialized.
    initialized: bool,
}

impl<S: ControlScalar> PositionLoop<S> {
    /// Create a position loop with P gain only (kd = 0).
    pub fn new_p(kp: S, speed_limit: S) -> Self {
        Self {
            kp,
            kd: S::ZERO,
            speed_limit,
            prev_pos: S::ZERO,
            initialized: false,
        }
    }

    /// Create a position loop with PD gains.
    pub fn new_pd(kp: S, kd: S, speed_limit: S) -> Self {
        Self {
            kp,
            kd,
            speed_limit,
            prev_pos: S::ZERO,
            initialized: false,
        }
    }

    /// Update the position loop.
    ///
    /// - `pos_ref`: desired position
    /// - `pos_actual`: measured position
    /// - `dt`: time step
    ///
    /// Returns the speed reference (clamped to ±speed_limit).
    pub fn update(&mut self, pos_ref: S, pos_actual: S, dt: S) -> S {
        let error = pos_ref - pos_actual;

        let d_term = if self.initialized && dt > S::ZERO {
            let pos_rate = (pos_actual - self.prev_pos) / dt;
            -self.kd * pos_rate // derivative of ACTUAL position (no derivative kick)
        } else {
            S::ZERO
        };

        self.prev_pos = pos_actual;
        self.initialized = true;

        let speed_ref = self.kp * error + d_term;
        speed_ref.clamp_val(-self.speed_limit, self.speed_limit)
    }

    pub fn reset(&mut self) {
        self.prev_pos = S::ZERO;
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_controller_converges() {
        let mut loop_ = PositionLoop::new_p(10.0_f64, 100.0);
        let mut pos = 0.0_f64;
        let target = 1.0_f64;
        let dt = 0.01;
        for _ in 0..500 {
            let speed_ref = loop_.update(target, pos, dt);
            pos += speed_ref * dt; // simple integrator plant
        }
        assert!((pos - target).abs() < 0.05, "pos={:.4}", pos);
    }

    #[test]
    fn output_clamped() {
        let mut loop_ = PositionLoop::new_p(1000.0_f64, 10.0);
        let speed = loop_.update(100.0, 0.0, 0.01);
        assert!(speed.abs() <= 10.0 + 1e-10);
    }

    #[test]
    fn reset_clears_state() {
        let mut loop_ = PositionLoop::new_pd(5.0_f64, 1.0, 50.0);
        loop_.update(1.0, 0.0, 0.01);
        loop_.reset();
        assert!(!loop_.initialized);
    }
}
