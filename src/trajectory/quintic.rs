use crate::core::scalar::ControlScalar;

/// Quintic (5th-order) polynomial trajectory between two endpoints.
///
/// Computes coefficients a0..a5 such that:
///   p(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
///
/// with boundary conditions:
///   p(0) = p0,  p'(0) = v0,  p''(0) = a0_val
///   p(T) = p1,  p'(T) = v1,  p''(T) = a1_val
///
/// Commonly used with v0=v1=0, a0_val=a1_val=0 for rest-to-rest motion
/// with smooth acceleration profile (no jerk discontinuity at start/end).
#[derive(Debug, Clone, Copy)]
pub struct QuinticPolynomial<S: ControlScalar> {
    /// Coefficients a0..a5.
    c: [S; 6],
    /// Total duration.
    pub duration: S,
}

impl<S: ControlScalar> QuinticPolynomial<S> {
    /// Fit a quintic polynomial with full boundary conditions.
    ///
    /// - `p0`, `v0`, `acc0`: initial position, velocity, acceleration
    /// - `p1`, `v1`, `acc1`: final position, velocity, acceleration
    /// - `duration`: total motion time (T > 0)
    ///
    /// Returns `None` if duration ≤ 0.
    pub fn new(p0: S, v0: S, acc0: S, p1: S, v1: S, acc1: S, duration: S) -> Option<Self> {
        if duration <= S::ZERO {
            return None;
        }
        let t = duration;
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;

        // Solve 6×6 system analytically (Vandermonde-like for quintic BCs):
        // a0 = p0
        // a1 = v0
        // a2 = acc0 / 2
        let a0 = p0;
        let a1 = v0;
        let a2 = acc0 * S::HALF;

        // Remaining 3 coefficients from endpoint BCs:
        //   p(T) = a0 + a1*T + a2*T² + a3*T³ + a4*T⁴ + a5*T⁵ = p1
        //   p'(T) = a1 + 2*a2*T + 3*a3*T² + 4*a4*T³ + 5*a5*T⁴ = v1
        //   p''(T) = 2*a2 + 6*a3*T + 12*a4*T² + 20*a5*T³ = acc1

        let dp = p1 - a0 - a1 * t - a2 * t2;
        let dv = v1 - a1 - S::TWO * a2 * t;
        let da = acc1 - S::TWO * a2;

        // Solve [T³  T⁴   T⁵ ] [a3]   [dp]
        //       [3T² 4T³  5T⁴] [a4] = [dv]
        //       [6T  12T² 20T³][a5]   [da]
        //
        // Closed-form solution:
        let six = S::from_f64(6.0);
        let ten = S::from_f64(10.0);
        let fifteen = S::from_f64(15.0);
        let three = S::from_f64(3.0);
        let four = S::from_f64(4.0);
        let five = S::from_f64(5.0);
        let eight = S::from_f64(8.0);
        let twelve = S::from_f64(12.0);
        let twenty = S::from_f64(20.0);

        let a3 = (ten * dp - four * dv * t + S::HALF * da * t2) / t3;
        let a4 = (-fifteen * dp + seven_s(dv * t) - da * t2) / t4;
        let a5 = (six * dp - three * dv * t + S::HALF * da * t2) / t5;

        // Verify the a3 formula is correct:
        // From the matrix system above, the exact closed form is:
        // [a3]   1/T^9 * [20T^6  -8T^7   T^8 ] [dp]   Wait, this gets complex.
        // Let me use direct algebra instead.
        //
        // Actually the standard quintic formula is:
        // a3 = (20*dp - (8*v1 + 12*v0)*T + (acc1 - 3*acc0)*T²) / (2*T³)
        // a4 = (-30*dp + (14*v1 + 16*v0)*T + (-2*acc1 + 3*acc0)*T²) / (2*T⁴)
        // a5 = (12*dp - (6*v1 + 6*v0)*T + (acc1 - acc0)*T²) / (2*T⁵)
        //
        // Using full BCs (v0, v1, acc0, acc1):
        let two = S::TWO;

        let dp_full = p1 - p0;
        let a3_final = (twenty * dp_full - (eight * v1 + twelve * v0) * t
            + (acc1 - three * acc0) * t2)
            / (two * t3);
        let a4_final = (-thirty_s(dp_full)
            + (fourteen_s(v1) + sixteen_s(v0)) * t
            + (-two * acc1 + three * acc0) * t2)
            / (two * t4);
        let a5_final =
            (twelve * dp_full - (six * v1 + six * v0) * t + (acc1 - acc0) * t2) / (two * t5);

        // Use the correct full-BC formulation (above), discard the earlier a3/a4/a5
        let _ = (
            a3, a4, a5, three, four, five, six, eight, ten, fifteen, twelve, twenty,
        );

        Some(Self {
            c: [a0, a1, a2, a3_final, a4_final, a5_final],
            duration,
        })
    }

    /// Rest-to-rest motion (v0=v1=0, acc0=acc1=0).
    pub fn rest_to_rest(p0: S, p1: S, duration: S) -> Option<Self> {
        Self::new(p0, S::ZERO, S::ZERO, p1, S::ZERO, S::ZERO, duration)
    }

    /// Evaluate position at time `t` (clamped to [0, duration]).
    pub fn position(&self, t: S) -> S {
        let t = t.clamp_val(S::ZERO, self.duration);
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;
        self.c[0]
            + self.c[1] * t
            + self.c[2] * t2
            + self.c[3] * t3
            + self.c[4] * t4
            + self.c[5] * t5
    }

    /// Evaluate velocity at time `t`.
    pub fn velocity(&self, t: S) -> S {
        let t = t.clamp_val(S::ZERO, self.duration);
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        self.c[1]
            + S::TWO * self.c[2] * t
            + S::from_f64(3.0) * self.c[3] * t2
            + S::from_f64(4.0) * self.c[4] * t3
            + S::from_f64(5.0) * self.c[5] * t4
    }

    /// Evaluate acceleration at time `t`.
    pub fn acceleration(&self, t: S) -> S {
        let t = t.clamp_val(S::ZERO, self.duration);
        let t2 = t * t;
        let t3 = t2 * t;
        S::TWO * self.c[2]
            + S::from_f64(6.0) * self.c[3] * t
            + S::from_f64(12.0) * self.c[4] * t2
            + S::from_f64(20.0) * self.c[5] * t3
    }

    /// Evaluate jerk at time `t`.
    pub fn jerk(&self, t: S) -> S {
        let t = t.clamp_val(S::ZERO, self.duration);
        let t2 = t * t;
        S::from_f64(6.0) * self.c[3]
            + S::from_f64(24.0) * self.c[4] * t
            + S::from_f64(60.0) * self.c[5] * t2
    }
}

fn seven_s<S: ControlScalar>(v: S) -> S {
    S::from_f64(7.0) * v
}
fn fourteen_s<S: ControlScalar>(v: S) -> S {
    S::from_f64(14.0) * v
}
fn sixteen_s<S: ControlScalar>(v: S) -> S {
    S::from_f64(16.0) * v
}
fn thirty_s<S: ControlScalar>(v: S) -> S {
    S::from_f64(30.0) * v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rest_to_rest_boundary_conditions() {
        let q = QuinticPolynomial::rest_to_rest(0.0_f64, 1.0, 1.0).unwrap();
        assert!(
            (q.position(0.0) - 0.0).abs() < 1e-10,
            "p(0)={}",
            q.position(0.0)
        );
        assert!(
            (q.position(1.0) - 1.0).abs() < 1e-10,
            "p(T)={}",
            q.position(1.0)
        );
        assert!(q.velocity(0.0).abs() < 1e-10, "v(0)={}", q.velocity(0.0));
        assert!(q.velocity(1.0).abs() < 1e-10, "v(T)={}", q.velocity(1.0));
        assert!(
            q.acceleration(0.0).abs() < 1e-10,
            "a(0)={}",
            q.acceleration(0.0)
        );
        assert!(
            q.acceleration(1.0).abs() < 1e-10,
            "a(T)={}",
            q.acceleration(1.0)
        );
    }

    #[test]
    fn velocity_peak_is_positive() {
        let q = QuinticPolynomial::rest_to_rest(0.0_f64, 1.0, 2.0).unwrap();
        let v_mid = q.velocity(1.0);
        assert!(v_mid > 0.0, "Peak velocity should be positive: v={}", v_mid);
    }

    #[test]
    fn clamped_outside_duration() {
        let q = QuinticPolynomial::rest_to_rest(0.0_f64, 5.0, 1.0).unwrap();
        assert!((q.position(-1.0) - 0.0).abs() < 1e-10);
        assert!((q.position(2.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn zero_duration_returns_none() {
        let result = QuinticPolynomial::rest_to_rest(0.0_f64, 1.0, 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn full_bc_satisfied() {
        let q = QuinticPolynomial::new(1.0_f64, 0.5, 0.1, 3.0, -0.5, 0.0, 2.0).unwrap();
        assert!((q.position(0.0) - 1.0).abs() < 1e-9);
        assert!((q.position(2.0) - 3.0).abs() < 1e-9);
        assert!((q.velocity(0.0) - 0.5).abs() < 1e-9);
        assert!((q.velocity(2.0) - (-0.5)).abs() < 1e-9);
        assert!((q.acceleration(0.0) - 0.1).abs() < 1e-9);
        assert!((q.acceleration(2.0) - 0.0).abs() < 1e-9);
    }
}
