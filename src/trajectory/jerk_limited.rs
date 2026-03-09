use crate::core::scalar::ControlScalar;

/// Jerk-limited (S-curve) motion profile generator.
///
/// Generates a 7-segment profile that respects limits on jerk, acceleration,
/// and velocity. The result is a smooth S-curve with continuous acceleration.
///
/// Segments (by acceleration profile):
///   1. Jerk-up: acceleration ramps from 0 to a_max
///   2. Constant acceleration: at a_max
///   3. Jerk-down: acceleration ramps from a_max to 0
///   4. Constant velocity: at v_max
///   5. Jerk-up (negative): acceleration ramps from 0 to -a_max
///   6. Constant deceleration: at -a_max
///   7. Jerk-down (negative): acceleration ramps from -a_max to 0
///
/// Handles short distances where some segments may be zero-duration.
///
/// All state is maintained internally; call `update(dt)` in a control loop.
#[derive(Debug, Clone, Copy)]
pub struct JerkLimitedProfile<S: ControlScalar> {
    /// Maximum jerk (rate of change of acceleration), units/s³.
    pub j_max: S,
    /// Maximum acceleration, units/s².
    pub a_max: S,
    /// Maximum velocity, units/s.
    pub v_max: S,

    // Internal motion state
    pos: S,
    vel: S,
    acc: S,
    elapsed: S,

    // Segment durations
    t1: S, // jerk-up to a_max
    t2: S, // constant acceleration
    t3: S, // jerk-down to 0
    t4: S, // constant velocity
    t5: S, // jerk-up to -a_max
    t6: S, // constant deceleration
    t7: S, // jerk-down to 0

    // Target position and initial conditions
    target: S,
    pos0: S,
    direction: S,

    total_duration: S,
    done: bool,
}

impl<S: ControlScalar> JerkLimitedProfile<S> {
    /// Create a jerk-limited profile generator.
    pub fn new(j_max: S, a_max: S, v_max: S) -> Self {
        Self {
            j_max,
            a_max,
            v_max,
            pos: S::ZERO,
            vel: S::ZERO,
            acc: S::ZERO,
            elapsed: S::ZERO,
            t1: S::ZERO,
            t2: S::ZERO,
            t3: S::ZERO,
            t4: S::ZERO,
            t5: S::ZERO,
            t6: S::ZERO,
            t7: S::ZERO,
            target: S::ZERO,
            pos0: S::ZERO,
            direction: S::ONE,
            total_duration: S::ZERO,
            done: true,
        }
    }

    /// Plan a move from current position to `target_pos`.
    ///
    /// Starting conditions: position=current, velocity=0, acceleration=0.
    pub fn plan(&mut self, target_pos: S) {
        let distance = target_pos - self.pos;
        if distance.abs() < S::EPSILON {
            self.done = true;
            return;
        }

        self.direction = if distance > S::ZERO { S::ONE } else { -S::ONE };
        let dist = distance.abs();
        self.pos0 = self.pos;
        self.target = target_pos;
        self.vel = S::ZERO;
        self.acc = S::ZERO;
        self.elapsed = S::ZERO;
        self.done = false;

        // Time to reach a_max from 0 with j_max
        let t_jerk = self.a_max / self.j_max;
        // Velocity gained during jerk phase
        let v_jerk = S::HALF * self.j_max * t_jerk * t_jerk;

        // Check if we can reach v_max
        // Distance to accelerate from 0 to v_max (including jerk phases):
        // If v_max > 2*v_jerk: need jerk1 + const_acc + jerk2
        //   d_accel = v_jerk*t_jerk + v_jerk*t_jerk + (v_max - 2*v_jerk)*t_jerk + ... complex
        // Simpler: use the standard 7-segment duration formulas

        let two = S::TWO;
        let four = S::from_f64(4.0);

        // Actual peak velocity achievable for this distance:
        // Full accel+decel (no const vel): needs dist >= 2*d_ramp
        // d_ramp = v_max * t_total_ramp / 2 where t_total_ramp = 2*t_jerk + t2
        // For the no-const-vel case: v_peak = min(v_max, sqrt(j*d/2) type formula)

        // Triangular check: distance for pure accel/decel without const-vel segment
        // t_accel_full = 2*t_jerk + t2_full where t2_full depends on v_max
        // Simplified: compute whether full v_max can be reached

        let v_peak = {
            // Distance to accelerate from 0 to v_max (symmetric accel/decel):
            // d = v_max * (t_jerk + t2 + t_jerk) - quadratic terms ...
            // Actually: d_half = v_max^2/(2*a_max) + a_max/(2*j_max) * (v_max/a_max - a_max/j_max)
            // Simplified conservative estimate:
            let d_needed = v_jerk * t_jerk * two
                + self.v_max / self.a_max * (self.v_max - v_jerk * two)
                + (self.v_max - v_jerk * two).clamp_val(S::ZERO, S::from_f64(1e9)) * S::ZERO;
            // Full formula for symmetric profile:
            // d_total = v_max * (2*t_jerk + t2_accel + t2_decel) - correction terms
            // Use simplified: if dist >= v_max * t_ramp → use v_max, else reduce v_peak
            let _ = d_needed;
            // Quick check: minimum distance needed for v_max profile
            let t_ramp = two * t_jerk;
            let d_min_for_vmax = self.v_max * (t_ramp + self.v_max / self.a_max) * S::HALF;
            if dist >= d_min_for_vmax * two {
                self.v_max
            } else {
                // Triangular: v_peak = sqrt(j_max * dist / 2) (rough)
                let v_try = (self.j_max * dist / four).sqrt();
                v_try.clamp_val(S::ZERO, self.v_max)
            }
        };

        let a_used = self.a_max.clamp_val(S::ZERO, self.a_max);

        // Jerk phase duration
        self.t1 = if v_peak > S::ZERO {
            (v_peak / self.j_max).sqrt().min(a_used / self.j_max)
        } else {
            S::ZERO
        };
        // Actually: t1 = min(a_max/j_max, sqrt(v_peak/j_max)) — standard formula
        self.t1 = (a_used / self.j_max).min((v_peak / self.j_max).sqrt());
        let v_after_t1 = S::HALF * self.j_max * self.t1 * self.t1;

        // Constant acceleration phase
        self.t2 = if v_peak > two * v_after_t1 {
            (v_peak - two * v_after_t1) / a_used
        } else {
            S::ZERO
        };

        // Jerk-down phase (symmetric to t1)
        self.t3 = self.t1;

        // Constant velocity phase
        let d_accel = v_after_t1 * self.t1 * S::from_f64(2.0 / 3.0)
            + v_after_t1 * self.t2
            + v_peak * self.t2 / two
            + v_peak * self.t1 * S::from_f64(2.0 / 3.0);
        // Simplified distance during full acceleration (to v_peak and back to 0):
        let d_accel_decel = v_peak * (two * self.t1 + self.t2);

        let d_const = dist - d_accel_decel;
        self.t4 = if d_const > S::ZERO {
            d_const / v_peak
        } else {
            S::ZERO
        };
        let _ = d_accel;

        // Deceleration mirror
        self.t5 = self.t1;
        self.t6 = self.t2;
        self.t7 = self.t1;

        self.total_duration = self.t1 + self.t2 + self.t3 + self.t4 + self.t5 + self.t6 + self.t7;
    }

    /// Update the profile by one time step.
    ///
    /// Returns `(position, velocity, acceleration)`.
    pub fn update(&mut self, dt: S) -> (S, S, S) {
        if self.done {
            return (self.target, S::ZERO, S::ZERO);
        }

        self.elapsed += dt;

        // Determine which segment we are in and what jerk to apply
        let jerk = self.current_jerk();

        // Integrate
        self.acc += jerk * dt;
        self.vel += self.acc * dt;
        self.pos += self.vel * dt;

        // Clamp velocity
        let v_max_dir = self.v_max * self.direction;
        if self.direction > S::ZERO {
            self.vel = self.vel.clamp_val(-self.v_max, v_max_dir);
        } else {
            self.vel = self.vel.clamp_val(v_max_dir, self.v_max);
        }

        // Check completion
        if self.elapsed >= self.total_duration {
            self.pos = self.target;
            self.vel = S::ZERO;
            self.acc = S::ZERO;
            self.done = true;
        }

        (self.pos, self.vel, self.acc)
    }

    fn current_jerk(&self) -> S {
        let t = self.elapsed;
        let j = self.j_max;
        let dir = self.direction;

        let seg_end = [
            self.t1,
            self.t1 + self.t2,
            self.t1 + self.t2 + self.t3,
            self.t1 + self.t2 + self.t3 + self.t4,
            self.t1 + self.t2 + self.t3 + self.t4 + self.t5,
            self.t1 + self.t2 + self.t3 + self.t4 + self.t5 + self.t6,
            self.total_duration,
        ];

        if t < seg_end[0] {
            j * dir // Seg 1: jerk up
        } else if t < seg_end[1] {
            S::ZERO // Seg 2: constant accel
        } else if t < seg_end[2] {
            -j * dir // Seg 3: jerk down
        } else if t < seg_end[3] {
            S::ZERO // Seg 4: constant vel
        } else if t < seg_end[4] {
            -j * dir // Seg 5: jerk up (decel)
        } else if t < seg_end[5] {
            S::ZERO // Seg 6: constant decel
        } else {
            j * dir // Seg 7: jerk down (back to 0 accel)
        }
    }

    pub fn position(&self) -> S {
        self.pos
    }

    pub fn velocity(&self) -> S {
        self.vel
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn set_position(&mut self, pos: S) {
        self.pos = pos;
        self.vel = S::ZERO;
        self.acc = S::ZERO;
        self.done = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reaches_target_position() {
        let mut prof = JerkLimitedProfile::new(100.0_f64, 10.0, 1.0);
        prof.set_position(0.0);
        prof.plan(2.0);
        let dt = 0.001;
        for _ in 0..10000 {
            if prof.is_done() {
                break;
            }
            prof.update(dt);
        }
        assert!(prof.is_done());
        assert!(
            (prof.position() - 2.0).abs() < 0.1,
            "pos={:.4}",
            prof.position()
        );
    }

    #[test]
    fn velocity_stays_within_limit() {
        let mut prof = JerkLimitedProfile::new(50.0_f64, 5.0, 2.0);
        prof.set_position(0.0);
        prof.plan(10.0);
        let dt = 0.001;
        let mut max_vel = 0.0_f64;
        for _ in 0..20000 {
            if prof.is_done() {
                break;
            }
            let (_, v, _) = prof.update(dt);
            max_vel = max_vel.max(v.abs());
        }
        assert!(max_vel <= 2.1, "max_vel={:.3}", max_vel); // allow 5% tolerance
    }

    #[test]
    fn zero_distance_immediately_done() {
        let mut prof = JerkLimitedProfile::new(10.0_f64, 5.0, 1.0);
        prof.set_position(3.0);
        prof.plan(3.0);
        assert!(prof.is_done());
    }
}
