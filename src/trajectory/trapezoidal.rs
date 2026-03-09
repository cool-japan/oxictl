use crate::core::scalar::ControlScalar;

/// Trapezoidal velocity profile for point-to-point motion.
///
/// Generates a position/velocity/acceleration profile with:
///   - Linear acceleration from 0 to v_max
///   - Constant velocity at v_max
///   - Linear deceleration from v_max to 0
///
/// Used for commanded motion when jerk-limiting is not required.
#[derive(Debug, Clone, Copy)]
pub struct TrapezoidalProfile<S: ControlScalar> {
    /// Maximum velocity (units/s).
    pub v_max: S,
    /// Maximum acceleration/deceleration (units/s²).
    pub a_max: S,
    // Computed plan
    t_accel: S,
    t_const: S,
    t_total: S,
    distance: S,
    sign: S,
}

impl<S: ControlScalar> TrapezoidalProfile<S> {
    pub fn new(v_max: S, a_max: S) -> Self {
        Self {
            v_max,
            a_max,
            t_accel: S::ZERO,
            t_const: S::ZERO,
            t_total: S::ZERO,
            distance: S::ZERO,
            sign: S::ONE,
        }
    }

    /// Plan a move over the given distance (positive or negative).
    pub fn plan(&mut self, distance: S) {
        if distance.abs() <= S::EPSILON {
            self.distance = S::ZERO;
            self.t_accel = S::ZERO;
            self.t_const = S::ZERO;
            self.t_total = S::ZERO;
            return;
        }

        self.sign = if distance >= S::ZERO { S::ONE } else { -S::ONE };
        self.distance = distance.abs();

        let a = self.a_max.max(S::EPSILON);
        let v = self.v_max.max(S::EPSILON);

        // Distance needed to reach v_max and back to 0
        let d_accel = v * v / (S::TWO * a);

        if self.distance >= S::TWO * d_accel {
            // Full trapezoidal profile
            self.t_accel = v / a;
            let d_const = self.distance - S::TWO * d_accel;
            self.t_const = d_const / v;
        } else {
            // Triangular profile (can't reach v_max)
            let v_peak = (a * self.distance).sqrt();
            self.t_accel = v_peak / a;
            self.t_const = S::ZERO;
        }
        self.t_total = S::TWO * self.t_accel + self.t_const;
    }

    /// Query the profile at time `t` after start.
    ///
    /// Returns (position, velocity, acceleration) as signed values.
    pub fn query(&self, t: S) -> (S, S, S) {
        if self.distance <= S::EPSILON || t >= self.t_total {
            return (self.sign * self.distance, S::ZERO, S::ZERO);
        }

        let a = self.a_max;
        let v_peak = self.t_accel * a;

        let (pos_mag, vel_mag, acc_mag) = if t < self.t_accel {
            // Acceleration phase
            let p = S::HALF * a * t * t;
            let v = a * t;
            (p, v, a)
        } else if t < self.t_accel + self.t_const {
            // Constant velocity phase
            let t2 = t - self.t_accel;
            let p_accel = S::HALF * a * self.t_accel * self.t_accel;
            let p = p_accel + v_peak * t2;
            (p, v_peak, S::ZERO)
        } else {
            // Deceleration phase
            let t3 = t - self.t_accel - self.t_const;
            let p_accel = S::HALF * a * self.t_accel * self.t_accel;
            let p_const = v_peak * self.t_const;
            let p = p_accel + p_const + v_peak * t3 - S::HALF * a * t3 * t3;
            let v = (v_peak - a * t3).max(S::ZERO);
            (p, v, -a)
        };

        (
            self.sign * pos_mag,
            self.sign * vel_mag,
            self.sign * acc_mag,
        )
    }

    /// Total profile duration.
    pub fn total_time(&self) -> S {
        self.t_total
    }

    /// Whether the move is complete at time `t`.
    pub fn is_done(&self, t: S) -> bool {
        t >= self.t_total
    }
}

/// Streaming trapezoidal profile: tracks elapsed time and returns commands.
#[derive(Debug, Clone, Copy)]
pub struct TrapezoidalMotion<S: ControlScalar> {
    profile: TrapezoidalProfile<S>,
    elapsed: S,
    start_pos: S,
}

impl<S: ControlScalar> TrapezoidalMotion<S> {
    pub fn new(v_max: S, a_max: S) -> Self {
        Self {
            profile: TrapezoidalProfile::new(v_max, a_max),
            elapsed: S::ZERO,
            start_pos: S::ZERO,
        }
    }

    /// Start a move from `current_pos` to `target_pos`.
    pub fn start_move(&mut self, current_pos: S, target_pos: S) {
        self.start_pos = current_pos;
        self.elapsed = S::ZERO;
        self.profile.plan(target_pos - current_pos);
    }

    /// Update by `dt` seconds. Returns (position_command, velocity_command).
    pub fn update(&mut self, dt: S) -> (S, S) {
        self.elapsed += dt;
        let (rel_pos, vel, _) = self.profile.query(self.elapsed);
        (self.start_pos + rel_pos, vel)
    }

    pub fn is_done(&self) -> bool {
        self.profile.is_done(self.elapsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_distance_is_done() {
        let mut p = TrapezoidalProfile::<f64>::new(1.0, 2.0);
        p.plan(0.0);
        let (pos, vel, _) = p.query(0.0);
        assert_eq!(pos, 0.0);
        assert_eq!(vel, 0.0);
        assert!(p.is_done(0.0));
    }

    #[test]
    fn reaches_target_position() {
        let mut p = TrapezoidalProfile::<f64>::new(2.0, 4.0);
        p.plan(10.0);
        let t = p.total_time();
        let (pos, vel, _) = p.query(t);
        assert!((pos - 10.0).abs() < 1e-6, "pos={}", pos);
        assert!(vel.abs() < 1e-6, "vel={}", vel);
    }

    #[test]
    fn negative_distance() {
        let mut p = TrapezoidalProfile::<f64>::new(2.0, 4.0);
        p.plan(-10.0);
        let t = p.total_time();
        let (pos, vel, _) = p.query(t);
        assert!((pos + 10.0).abs() < 1e-6);
        assert!(vel.abs() < 1e-6);
    }

    #[test]
    fn velocity_peaks_at_v_max() {
        let mut p = TrapezoidalProfile::<f64>::new(3.0, 6.0);
        p.plan(20.0);
        let t_peak = p.t_accel + p.t_const / 2.0;
        let (_, vel, _) = p.query(t_peak);
        assert!((vel - 3.0).abs() < 1e-6, "vel={}", vel);
    }

    #[test]
    fn triangular_profile_short_distance() {
        // Distance too short to reach v_max
        let mut p = TrapezoidalProfile::<f64>::new(10.0, 4.0);
        p.plan(1.0); // Very short
        assert_eq!(p.t_const, 0.0); // No constant velocity phase
        let t = p.total_time();
        let (pos, _, _) = p.query(t);
        assert!((pos - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trapezoidal_motion_streaming() {
        let mut m = TrapezoidalMotion::<f64>::new(2.0, 4.0);
        m.start_move(5.0, 15.0);
        let mut t = 0.0;
        let dt = 0.001;
        while !m.is_done() && t < 100.0 {
            let _ = m.update(dt);
            t += dt;
        }
        let (pos, _) = m.update(0.0);
        assert!((pos - 15.0).abs() < 0.1, "Final pos: {}", pos);
    }
}
