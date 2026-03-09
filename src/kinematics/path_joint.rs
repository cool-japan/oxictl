//! Joint-space path with cubic Hermite blending between waypoints.
//!
//! K waypoints in N-DOF joint space are connected by cubic Hermite spline
//! segments.  Velocities at internal waypoints are estimated via
//! finite-differences (Catmull-Rom style).  The path can optionally be
//! retimed to respect per-joint velocity limits.
#![allow(clippy::needless_range_loop)]

use crate::core::scalar::ControlScalar;

/// K-waypoint path in N-DOF joint space.
///
/// Waypoints are blended with cubic Hermite splines.
pub struct JointSpacePath<S: ControlScalar, const N: usize, const K: usize> {
    /// Waypoint joint configurations.
    pub waypoints: [[S; N]; K],
    /// Monotonically increasing waypoint timestamps (normalised 0..1).
    pub times: [S; K],
    /// Velocities at each waypoint (finite-difference estimate).
    velocities: [[S; N]; K],
}

impl<S: ControlScalar, const N: usize, const K: usize> JointSpacePath<S, N, K> {
    /// Create a new joint-space path.
    ///
    /// `times` must be strictly increasing with times[0] = 0 and times[K-1] = 1.
    pub fn new(waypoints: [[S; N]; K], times: [S; K]) -> Self {
        let velocities = Self::compute_velocities(&waypoints, &times);
        Self {
            waypoints,
            times,
            velocities,
        }
    }

    /// Evaluate joint angles at normalised time t ∈ [0, 1].
    pub fn evaluate(&self, t: S) -> [S; N] {
        if K == 0 {
            return [S::ZERO; N];
        }
        if K == 1 {
            return self.waypoints[0];
        }
        let t = t.clamp_val(self.times[0], self.times[K - 1]);
        let seg = self.find_segment(t);
        let (t0, t1) = (self.times[seg], self.times[seg + 1]);
        let dt = t1 - t0;
        let tau = if dt > S::EPSILON {
            (t - t0) / dt
        } else {
            S::ZERO
        };
        let (h00, h10, h01, h11) = Self::hermite_basis(tau);
        let mut result = [S::ZERO; N];
        for j in 0..N {
            let p0 = self.waypoints[seg][j];
            let p1 = self.waypoints[seg + 1][j];
            // Scale velocities by segment duration
            let m0 = self.velocities[seg][j] * dt;
            let m1 = self.velocities[seg + 1][j] * dt;
            result[j] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
        }
        result
    }

    /// Evaluate joint velocities (dq/dt) at normalised time t.
    pub fn velocity(&self, t: S) -> [S; N] {
        if K <= 1 {
            return [S::ZERO; N];
        }
        let t = t.clamp_val(self.times[0], self.times[K - 1]);
        let seg = self.find_segment(t);
        let (t0, t1) = (self.times[seg], self.times[seg + 1]);
        let dt = t1 - t0;
        let tau = if dt > S::EPSILON {
            (t - t0) / dt
        } else {
            S::ZERO
        };
        // Derivatives of Hermite basis w.r.t. tau, then divided by dt for dq/dt
        // dh00/dτ = 6τ^2 - 6τ = 6τ(τ-1)
        // dh10/dτ = 3τ^2 - 4τ + 1
        // dh01/dτ = -6τ^2 + 6τ = 6τ(1-τ)... Wait: dh01/dτ = -dh00/dτ
        // dh11/dτ = 3τ^2 - 2τ
        let two = S::TWO;
        let three = S::from_f64(3.0);
        let four = S::from_f64(4.0);
        let six = S::from_f64(6.0);
        let dh00 = six * tau * tau - six * tau;
        let dh10 = three * tau * tau - four * tau + S::ONE;
        let dh01 = -six * tau * tau + six * tau;
        let dh11 = three * tau * tau - two * tau;
        let mut vel = [S::ZERO; N];
        for j in 0..N {
            let p0 = self.waypoints[seg][j];
            let p1 = self.waypoints[seg + 1][j];
            let m0 = self.velocities[seg][j] * dt;
            let m1 = self.velocities[seg + 1][j] * dt;
            // dq/dτ, then chain-rule: dq/dt = dq/dτ * dτ/dt = dq/dτ / dt
            let dq_dtau = dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1;
            vel[j] = if dt > S::EPSILON {
                dq_dtau / dt
            } else {
                S::ZERO
            };
        }
        vel
    }

    /// Retime path segments so that joint velocities remain within `qdot_max`.
    ///
    /// Each segment duration is extended if any joint would exceed its limit.
    pub fn retime_for_velocity_limits(&mut self, qdot_max: [S; N]) {
        if K < 2 {
            return;
        }
        // Compute required scale factors per segment
        let mut new_times = self.times;
        let mut cumulative = S::ZERO;
        new_times[0] = S::ZERO;
        for seg in 0..(K - 1) {
            let dt_old = self.times[seg + 1] - self.times[seg];
            let mut scale = S::ONE;
            for j in 0..N {
                let delta = (self.waypoints[seg + 1][j] - self.waypoints[seg][j]).abs();
                let v_lim = qdot_max[j];
                if v_lim > S::EPSILON {
                    // Approximate: linear segment would need delta/v_lim time
                    let required = delta / v_lim;
                    if required > dt_old * scale {
                        scale = required / dt_old;
                    }
                }
            }
            cumulative += dt_old * scale;
            new_times[seg + 1] = cumulative;
        }
        // Normalise so that times[K-1] = 1
        let total = new_times[K - 1];
        if total > S::EPSILON {
            for k in 0..K {
                new_times[k] = new_times[k] / total;
            }
        }
        self.times = new_times;
        self.velocities = Self::compute_velocities(&self.waypoints, &self.times);
    }

    /// Cubic Hermite basis functions for τ ∈ [0, 1].
    ///
    /// Returns (h00, h10, h01, h11).
    fn hermite_basis(t: S) -> (S, S, S, S) {
        let t2 = t * t;
        let t3 = t2 * t;
        let two = S::TWO;
        let three = S::from_f64(3.0);
        let h00 = two * t3 - three * t2 + S::ONE;
        let h10 = t3 - two * t2 + t;
        let h01 = -two * t3 + three * t2;
        let h11 = t3 - t2;
        (h00, h10, h01, h11)
    }

    /// Compute finite-difference velocities (Catmull-Rom style).
    fn compute_velocities(waypoints: &[[S; N]; K], times: &[S; K]) -> [[S; N]; K] {
        let mut vels: [[S; N]; K] = core::array::from_fn(|_| [S::ZERO; N]);
        if K < 2 {
            return vels;
        }
        for k in 0..K {
            for j in 0..N {
                if k == 0 {
                    // Forward difference
                    let dt = times[1] - times[0];
                    vels[k][j] = if dt > S::EPSILON {
                        (waypoints[1][j] - waypoints[0][j]) / dt
                    } else {
                        S::ZERO
                    };
                } else if k == K - 1 {
                    // Backward difference
                    let dt = times[K - 1] - times[K - 2];
                    vels[k][j] = if dt > S::EPSILON {
                        (waypoints[K - 1][j] - waypoints[K - 2][j]) / dt
                    } else {
                        S::ZERO
                    };
                } else {
                    // Central difference
                    let dt = times[k + 1] - times[k - 1];
                    vels[k][j] = if dt > S::EPSILON {
                        (waypoints[k + 1][j] - waypoints[k - 1][j]) / dt
                    } else {
                        S::ZERO
                    };
                }
            }
        }
        vels
    }

    /// Find the segment index such that times[seg] ≤ t < times[seg+1].
    fn find_segment(&self, t: S) -> usize {
        for seg in 0..(K - 1) {
            if t <= self.times[seg + 1] {
                return seg;
            }
        }
        K - 2 // clamp to last segment
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn two_waypoint_path() -> JointSpacePath<f64, 2, 2> {
        let waypoints = [[0.0, 0.0], [1.0, 1.0]];
        let times = [0.0, 1.0];
        JointSpacePath::new(waypoints, times)
    }

    #[test]
    fn evaluate_at_endpoints() {
        let path = two_waypoint_path();
        let q0 = path.evaluate(0.0);
        let q1 = path.evaluate(1.0);
        assert!((q0[0] - 0.0).abs() < 1e-10, "q0[0]={}", q0[0]);
        assert!((q0[1] - 0.0).abs() < 1e-10, "q0[1]={}", q0[1]);
        assert!((q1[0] - 1.0).abs() < 1e-10, "q1[0]={}", q1[0]);
        assert!((q1[1] - 1.0).abs() < 1e-10, "q1[1]={}", q1[1]);
    }

    #[test]
    fn evaluate_midpoint_is_interpolated() {
        let path = two_waypoint_path();
        let qmid = path.evaluate(0.5);
        // Cubic Hermite at τ=0.5 for p0=0, p1=1 with equal endpoint velocities
        // h00(0.5)=0.5, h01(0.5)=0.5 → q = 0.5 (velocity terms cancel for symmetric case)
        assert!((qmid[0] - 0.5).abs() < 1e-9, "qmid[0]={}", qmid[0]);
    }

    #[test]
    fn three_waypoint_path_continuity() {
        let waypoints = [[0.0_f64], [1.0], [0.0]];
        let times = [0.0, 0.5, 1.0];
        let path = JointSpacePath::<f64, 1, 3>::new(waypoints, times);
        // Sample at many points and check continuity
        let mut prev = path.evaluate(0.0)[0];
        let mut max_jump = 0.0_f64;
        for i in 1..=100 {
            let t = i as f64 * 0.01;
            let q = path.evaluate(t)[0];
            let jump = (q - prev).abs();
            if jump > max_jump {
                max_jump = jump;
            }
            prev = q;
        }
        assert!(max_jump < 0.1, "max_jump={max_jump}");
    }

    #[test]
    fn retime_keeps_endpoints() {
        let waypoints = [[0.0_f64, 0.0], [1.0, 2.0], [2.0, 1.0]];
        let times = [0.0, 0.5, 1.0];
        let mut path = JointSpacePath::<f64, 2, 3>::new(waypoints, times);
        path.retime_for_velocity_limits([0.5, 0.5]);
        assert!((path.times[0] - 0.0).abs() < 1e-10);
        assert!((path.times[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn hermite_basis_partition_of_unity() {
        // h00 + h01 = 1 for all τ (position terms)
        for i in 0..=10 {
            let tau = i as f64 * 0.1;
            let (h00, _h10, h01, _h11) = JointSpacePath::<f64, 1, 2>::hermite_basis(tau);
            assert!(
                (h00 + h01 - 1.0).abs() < 1e-12,
                "tau={tau} sum={}",
                h00 + h01
            );
        }
    }
}
