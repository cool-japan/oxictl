use crate::core::scalar::ControlScalar;

/// S-curve (trapezoidal jerk-limited) velocity profile generator.
///
/// Generates a smooth velocity profile for stepper motor acceleration,
/// avoiding the sharp corners of trapezoidal profiles.
///
/// States: Accel, ConstVel, Decel, Done
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileState {
    Accelerating,
    ConstantVelocity,
    Decelerating,
    Done,
}

/// S-curve profile parameters.
#[derive(Debug, Clone, Copy)]
pub struct SCurveProfile<S: ControlScalar> {
    pub v_max: S,
    pub a_max: S,
    pub j_max: S,
}

impl<S: ControlScalar> SCurveProfile<S> {
    pub fn new(v_max: S, a_max: S, j_max: S) -> Self {
        Self {
            v_max,
            a_max,
            j_max,
        }
    }
}

/// S-curve velocity profile generator.
///
/// Produces a velocity command each timestep.
/// Call `update(dt)` each control cycle.
#[derive(Debug, Clone, Copy)]
pub struct SCurveGenerator<S: ControlScalar> {
    profile: SCurveProfile<S>,
    state: ProfileState,
    velocity: S,
    acceleration: S,
    target_pos: S,
    current_pos: S,
}

impl<S: ControlScalar> SCurveGenerator<S> {
    pub fn new(profile: SCurveProfile<S>) -> Self {
        Self {
            profile,
            state: ProfileState::Done,
            velocity: S::ZERO,
            acceleration: S::ZERO,
            target_pos: S::ZERO,
            current_pos: S::ZERO,
        }
    }

    /// Start a move to target position.
    pub fn start_move(&mut self, current_pos: S, target_pos: S) {
        self.current_pos = current_pos;
        self.target_pos = target_pos;
        self.velocity = S::ZERO;
        self.acceleration = S::ZERO;
        if (target_pos - current_pos).abs() > S::EPSILON {
            self.state = ProfileState::Accelerating;
        } else {
            self.state = ProfileState::Done;
        }
    }

    /// Update profile. Returns (velocity_command, position_estimate).
    pub fn update(&mut self, dt: S) -> (S, S) {
        if self.state == ProfileState::Done {
            return (S::ZERO, self.current_pos);
        }

        let direction = if self.target_pos > self.current_pos {
            S::ONE
        } else {
            -S::ONE
        };
        let remaining = (self.target_pos - self.current_pos).abs();

        // Simple jerk-limited approach:
        // - Increase acceleration up to a_max during acceleration
        // - Maintain v_max during constant velocity
        // - Begin deceleration when remaining distance requires it
        let decel_dist =
            self.velocity * self.velocity / (S::TWO * self.profile.a_max.max(S::EPSILON));

        match self.state {
            ProfileState::Accelerating => {
                // Apply jerk to increase acceleration
                self.acceleration = (self.acceleration + self.profile.j_max * dt)
                    .clamp_val(S::ZERO, self.profile.a_max);
                self.velocity =
                    (self.velocity + self.acceleration * dt).clamp_val(S::ZERO, self.profile.v_max);

                if decel_dist >= remaining || self.velocity >= self.profile.v_max {
                    if self.velocity >= self.profile.v_max {
                        self.state = ProfileState::ConstantVelocity;
                    } else {
                        self.state = ProfileState::Decelerating;
                    }
                }
            }
            ProfileState::ConstantVelocity => {
                if decel_dist >= remaining {
                    self.state = ProfileState::Decelerating;
                }
            }
            ProfileState::Decelerating => {
                self.acceleration = (self.acceleration - self.profile.j_max * dt)
                    .clamp_val(-self.profile.a_max, S::ZERO);
                self.velocity = (self.velocity + self.acceleration * dt).max(S::ZERO);

                if self.velocity <= S::ZERO || remaining <= S::EPSILON {
                    self.velocity = S::ZERO;
                    self.acceleration = S::ZERO;
                    self.state = ProfileState::Done;
                    self.current_pos = self.target_pos;
                    return (S::ZERO, self.current_pos);
                }
            }
            ProfileState::Done => {}
        }

        // Update position
        self.current_pos += direction * self.velocity * dt;

        (direction * self.velocity, self.current_pos)
    }

    pub fn is_done(&self) -> bool {
        self.state == ProfileState::Done
    }

    pub fn velocity(&self) -> S {
        self.velocity
    }

    pub fn position(&self) -> S {
        self.current_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_done() {
        let gen = SCurveGenerator::<f64>::new(SCurveProfile::new(100.0, 1000.0, 10000.0));
        assert!(gen.is_done());
    }

    #[test]
    fn moves_toward_target() {
        let mut gen = SCurveGenerator::<f64>::new(SCurveProfile::new(100.0, 1000.0, 50000.0));
        gen.start_move(0.0, 1.0);
        let mut pos = 0.0_f64;
        for _ in 0..10000 {
            let r = gen.update(0.001);
            pos = r.1;
            if gen.is_done() {
                break;
            }
        }
        assert!(gen.is_done(), "Should complete move");
        assert!((pos - 1.0).abs() < 0.1, "Should reach target: pos={}", pos);
    }

    #[test]
    fn velocity_stays_bounded() {
        let v_max = 100.0_f64;
        let mut gen = SCurveGenerator::<f64>::new(SCurveProfile::new(v_max, 1000.0, 50000.0));
        gen.start_move(0.0, 1000.0);
        for _ in 0..100000 {
            let (v, _) = gen.update(0.001);
            assert!(v.abs() <= v_max + 1e-6, "Velocity exceeded: {}", v);
            if gen.is_done() {
                break;
            }
        }
    }

    #[test]
    fn negative_direction() {
        let mut gen = SCurveGenerator::<f64>::new(SCurveProfile::new(100.0, 1000.0, 50000.0));
        gen.start_move(10.0, 5.0);
        let mut pos = 10.0;
        for _ in 0..50000 {
            let (_, p) = gen.update(0.001);
            pos = p;
            if gen.is_done() {
                break;
            }
        }
        assert!(
            pos <= 10.0,
            "Should move in negative direction: pos={}",
            pos
        );
    }

    #[test]
    fn already_at_target() {
        let mut gen = SCurveGenerator::<f64>::new(SCurveProfile::new(100.0, 1000.0, 50000.0));
        gen.start_move(5.0, 5.0);
        assert!(gen.is_done());
        let (v, _) = gen.update(0.001);
        assert_eq!(v, 0.0);
    }
}
