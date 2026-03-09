//! Time-optimal bang-bang trajectory (minimum-time 1D).
//!
//! Plans a minimum-time trajectory from `(x0, v0)` to `(x1, v1)` subject to
//! velocity and acceleration limits.  The profile consists of:
//!   - Phase 1: acceleration at ±a_max
//!   - Phase 2: optional coast at v_max / v_min (zero duration if not needed)
//!   - Phase 3: deceleration at ∓a_max
//!
//! The algorithm handles both forward (x1 > x0) and backward (x1 < x0) moves,
//! initial velocities, and cases where the maximum velocity is not reachable.
use crate::core::scalar::ControlScalar;

/// Phase durations for a bang-bang trajectory.
#[derive(Debug, Clone, Copy)]
pub struct BangBangPhases<S: ControlScalar> {
    /// Duration of acceleration phase.
    pub t1: S,
    /// Duration of coast phase (may be 0).
    pub t2: S,
    /// Duration of deceleration phase.
    pub t3: S,
    /// Total trajectory duration.
    pub total: S,
}

/// Time-optimal 1D motion profile (bang-bang with optional coast).
#[derive(Debug, Clone, Copy)]
pub struct TimeOptimalProfile<S: ControlScalar> {
    pub x0: S,
    pub v0: S,
    pub x1: S,
    pub v1: S,
    pub v_max: S,
    pub a_max: S,
    pub phases: BangBangPhases<S>,
    /// Sign of the primary acceleration (+1 or -1).
    sign: S,
    /// Peak velocity reached during the profile.
    v_peak: S,
}

impl<S: ControlScalar> TimeOptimalProfile<S> {
    /// Plan a minimum-time trajectory from `(x0, v0)` to `(x1, v1)`.
    ///
    /// `v_max` and `a_max` are both positive limits.
    pub fn plan(x0: S, v0: S, x1: S, v1: S, v_max: S, a_max: S) -> Self {
        // Clamp target velocities to ±v_max.
        let v1 = v1.clamp_val(-v_max, v_max);
        let v0 = v0.clamp_val(-v_max, v_max);
        // Try a positive (forward) bang profile first.
        let profile_pos = Self::try_plan(x0, v0, x1, v1, v_max, a_max, S::ONE);
        // Try a negative (backward) bang profile.
        let profile_neg = Self::try_plan(x0, v0, x1, v1, v_max, a_max, -S::ONE);

        // Choose the valid profile with minimum time.
        match (profile_pos, profile_neg) {
            (Some(p), Some(n)) => {
                if p.phases.total <= n.phases.total {
                    p
                } else {
                    n
                }
            }
            (Some(p), None) => p,
            (None, Some(n)) => n,
            (None, None) => Self::zero_profile(x0, v0, x1, v1, v_max, a_max),
        }
    }

    /// Attempt to build a profile with `sign` direction for the first phase.
    fn try_plan(x0: S, v0: S, x1: S, v1: S, v_max: S, a_max: S, sign: S) -> Option<Self> {
        // Peak velocity: the profile accelerates to v_peak, then decelerates to v1.
        let v_peak_candidate = sign * v_max;

        // Time to accelerate from v0 to v_peak.
        let dv1 = v_peak_candidate - v0;
        let t1 = dv1 / (sign * a_max);
        if t1 < -S::EPSILON {
            return None;
        }
        let t1 = t1.max(S::ZERO);

        // Time to decelerate from v_peak to v1.
        let dv3 = v_peak_candidate - v1;
        let t3 = dv3 / (sign * a_max);
        if t3 < -S::EPSILON {
            return None;
        }
        let t3 = t3.max(S::ZERO);

        // Distance covered during phases 1 and 3.
        let x_total = x1 - x0;
        let x_accel = v0 * t1 + sign * a_max * t1 * t1 * S::HALF;
        let x_decel = v_peak_candidate * t3 - sign * a_max * t3 * t3 * S::HALF;
        let x_remaining = x_total - x_accel - x_decel;

        // Coast duration.
        let t2_raw = x_remaining / v_peak_candidate;
        if t2_raw < -S::EPSILON {
            // Peak velocity is too high; try with the exact achievable peak.
            return Self::try_plan_no_coast(x0, v0, x1, v1, a_max, sign);
        }
        let t2 = t2_raw.max(S::ZERO);
        let total = t1 + t2 + t3;
        if total < -S::EPSILON {
            return None;
        }

        Some(Self {
            x0,
            v0,
            x1,
            v1,
            v_max,
            a_max,
            phases: BangBangPhases {
                t1,
                t2,
                t3,
                total: total.max(S::ZERO),
            },
            sign,
            v_peak: v_peak_candidate,
        })
    }

    /// Build a profile that just touches the required peak (no coast, triangular).
    fn try_plan_no_coast(x0: S, v0: S, x1: S, v1: S, a_max: S, sign: S) -> Option<Self> {
        // Solve for v_peak such that:
        //   (v_peak - v0)^2 / (2*a) + (v_peak - v1)^2 / (2*a) = |dx|
        // where dx is the signed displacement.
        let dx = x1 - x0;
        let a = sign * a_max;
        // From kinematics:
        // t1 = (v_peak - v0)/a, t3 = (v_peak - v1)/a
        // dx = v0*t1 + 0.5*a*t1^2 + v_peak*t3 - 0.5*a*t3^2
        // Expanding and collecting: 2*a*v_peak - v0^2/a - v1^2/a... use quadratic
        // Simpler: v_peak^2*(1/a) - v_peak*(v0+v1)/a + (v0^2+v1^2)/(2*a) - dx/1 = 0
        // Actually solve: v_peak = sqrt(a*dx + (v0^2+v1^2)/2) (for sign=+1, dx>0)
        let discriminant = a * dx + (v0 * v0 + v1 * v1) * S::HALF;
        if discriminant < S::ZERO {
            return None;
        }
        let v_peak = discriminant.sqrt();
        if v_peak.abs() < S::EPSILON {
            return None;
        }

        let t1_raw = (v_peak - v0) / a;
        let t3_raw = (v_peak - v1) / a;
        if t1_raw < -S::EPSILON || t3_raw < -S::EPSILON {
            return None;
        }
        let t1 = t1_raw.max(S::ZERO);
        let t3 = t3_raw.max(S::ZERO);
        let total = t1 + t3;

        Some(Self {
            x0,
            v0,
            x1,
            v1,
            v_max: v_peak,
            a_max,
            phases: BangBangPhases {
                t1,
                t2: S::ZERO,
                t3,
                total,
            },
            sign,
            v_peak,
        })
    }

    /// Fallback zero-displacement profile.
    fn zero_profile(x0: S, v0: S, x1: S, v1: S, v_max: S, a_max: S) -> Self {
        Self {
            x0,
            v0,
            x1,
            v1,
            v_max,
            a_max,
            phases: BangBangPhases {
                t1: S::ZERO,
                t2: S::ZERO,
                t3: S::ZERO,
                total: S::ZERO,
            },
            sign: S::ONE,
            v_peak: S::ZERO,
        }
    }

    /// Query trajectory state at time `t ∈ [0, duration]`.
    ///
    /// Returns `(position, velocity, acceleration)`.
    pub fn query(&self, t: S) -> (S, S, S) {
        let t = t.clamp_val(S::ZERO, self.phases.total);
        let a = self.sign * self.a_max;

        if t <= self.phases.t1 {
            // Acceleration phase.
            let pos = self.x0 + self.v0 * t + S::HALF * a * t * t;
            let vel = self.v0 + a * t;
            (pos, vel, a)
        } else if t <= self.phases.t1 + self.phases.t2 {
            // Coast phase.
            let tc = t - self.phases.t1;
            let x1_end =
                self.x0 + self.v0 * self.phases.t1 + S::HALF * a * self.phases.t1 * self.phases.t1;
            let pos = x1_end + self.v_peak * tc;
            (pos, self.v_peak, S::ZERO)
        } else {
            // Deceleration phase.
            let t3 = t - self.phases.t1 - self.phases.t2;
            let x_accel_end =
                self.x0 + self.v0 * self.phases.t1 + S::HALF * a * self.phases.t1 * self.phases.t1;
            let x_coast_end = x_accel_end + self.v_peak * self.phases.t2;
            let pos = x_coast_end + self.v_peak * t3 - S::HALF * a * t3 * t3;
            let vel = self.v_peak - a * t3;
            (pos, vel, -a)
        }
    }

    /// Total trajectory duration.
    pub fn duration(&self) -> S {
        self.phases.total
    }

    /// Whether the profile has a non-zero duration.
    pub fn is_valid(&self) -> bool {
        self.phases.total > S::ZERO
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TimeOptimalSegment API  (trapezoidal / triangular 1-DOF planner)
// ─────────────────────────────────────────────────────────────────────────────

use crate::trajectory::TrajectoryError;
use heapless::Vec as HVec;

/// A single kinematic segment (constant-acceleration phase).
///
/// Each segment describes one phase of a trapezoidal or triangular velocity
/// profile: acceleration-up, cruise (constant velocity), or deceleration.
#[derive(Debug, Clone, Copy)]
pub struct TimeOptimalSegment<S: ControlScalar> {
    /// Duration of this segment (≥ 0).
    pub duration: S,
    /// Velocity at the start of this segment.
    pub v_start: S,
    /// Velocity at the end of this segment.
    pub v_end: S,
    /// Constant acceleration applied during this segment (may be 0 for cruise).
    pub accel: S,
}

/// Plan a minimum-time 1-DOF profile for a given `distance`.
///
/// The returned list contains up to 3 segments:
/// 1. Acceleration from `v_start` toward `v_max`.
/// 2. Optional cruise at `v_max` (omitted for a triangular profile).
/// 3. Deceleration from peak velocity to `v_end`.
///
/// # Arguments
/// - `distance` — signed displacement (positive only; direction handled by caller)
/// - `v_max`    — maximum speed (> 0)
/// - `a_max`    — maximum acceleration magnitude (> 0)
/// - `v_start`  — initial speed (clamped to [0, v_max])
/// - `v_end`    — final speed (clamped to [0, v_max])
///
/// # Errors
/// Returns [`TrajectoryError::InvalidParameter`] if `v_max ≤ 0` or `a_max ≤ 0`.
pub fn plan_1dof<S: ControlScalar>(
    distance: S,
    v_max: S,
    a_max: S,
    v_start: S,
    v_end: S,
) -> Result<HVec<TimeOptimalSegment<S>, 3>, TrajectoryError> {
    if v_max <= S::ZERO || a_max <= S::ZERO {
        return Err(TrajectoryError::InvalidParameter);
    }

    // Clamp boundary velocities.
    let v0 = v_start.clamp_val(S::ZERO, v_max);
    let v1 = v_end.clamp_val(S::ZERO, v_max);

    let mut segments: HVec<TimeOptimalSegment<S>, 3> = HVec::new();

    // ── Distance covered while accelerating to v_max then decelerating to v1 ──
    // d_accel = (v_max^2 - v0^2) / (2*a_max)
    // d_decel = (v_max^2 - v1^2) / (2*a_max)
    let d_accel_full = (v_max * v_max - v0 * v0) / (S::TWO * a_max);
    let d_decel_full = (v_max * v_max - v1 * v1) / (S::TWO * a_max);
    let d_min = d_accel_full + d_decel_full;

    if d_min.to_f64() <= distance.to_f64() {
        // ── Trapezoidal profile ──────────────────────────────────────────────
        // Phase 1: accelerate v0 → v_max.
        let t1 = (v_max - v0) / a_max;
        if t1 > S::ZERO {
            segments
                .push(TimeOptimalSegment {
                    duration: t1,
                    v_start: v0,
                    v_end: v_max,
                    accel: a_max,
                })
                .map_err(|_| TrajectoryError::BufferFull)?;
        }

        // Phase 2: cruise at v_max.
        let d_cruise = distance - d_min;
        let t2 = d_cruise / v_max;
        if t2 > S::ZERO {
            segments
                .push(TimeOptimalSegment {
                    duration: t2,
                    v_start: v_max,
                    v_end: v_max,
                    accel: S::ZERO,
                })
                .map_err(|_| TrajectoryError::BufferFull)?;
        }

        // Phase 3: decelerate v_max → v1.
        let t3 = (v_max - v1) / a_max;
        if t3 > S::ZERO {
            segments
                .push(TimeOptimalSegment {
                    duration: t3,
                    v_start: v_max,
                    v_end: v1,
                    accel: -a_max,
                })
                .map_err(|_| TrajectoryError::BufferFull)?;
        }
    } else {
        // ── Triangular profile ───────────────────────────────────────────────
        // Peak velocity v_peak < v_max.
        // d = (v_peak^2 - v0^2)/(2*a) + (v_peak^2 - v1^2)/(2*a)
        //   = (2*v_peak^2 - v0^2 - v1^2) / (2*a)
        // → v_peak = sqrt(a*d + (v0^2 + v1^2)/2)
        let disc = a_max * distance + (v0 * v0 + v1 * v1) * S::HALF;
        let v_peak = if disc > S::ZERO {
            disc.sqrt().min(v_max)
        } else {
            v0.max(v1)
        };

        // Phase 1: v0 → v_peak.
        let t1 = if v_peak > v0 {
            (v_peak - v0) / a_max
        } else {
            S::ZERO
        };
        if t1 > S::ZERO {
            segments
                .push(TimeOptimalSegment {
                    duration: t1,
                    v_start: v0,
                    v_end: v_peak,
                    accel: a_max,
                })
                .map_err(|_| TrajectoryError::BufferFull)?;
        }

        // Phase 2 (decel): v_peak → v1.
        let t3 = if v_peak > v1 {
            (v_peak - v1) / a_max
        } else {
            S::ZERO
        };
        if t3 > S::ZERO {
            segments
                .push(TimeOptimalSegment {
                    duration: t3,
                    v_start: v_peak,
                    v_end: v1,
                    accel: -a_max,
                })
                .map_err(|_| TrajectoryError::BufferFull)?;
        }
    }

    Ok(segments)
}

/// Total duration of a segment list.
pub fn total_time<S: ControlScalar>(segments: &HVec<TimeOptimalSegment<S>, 3>) -> S {
    segments.iter().fold(S::ZERO, |acc, seg| acc + seg.duration)
}

/// Sample position and velocity at time `t` within a segment list.
///
/// `t` is clamped to `[0, total_time(segments)]`.
///
/// # Returns
/// `Ok((position, velocity))` where position is integrated from 0.
///
/// # Errors
/// Returns [`TrajectoryError::InvalidParameter`] if `segments` is empty.
pub fn sample_at<S: ControlScalar>(
    segments: &HVec<TimeOptimalSegment<S>, 3>,
    t: S,
) -> Result<(S, S), TrajectoryError> {
    if segments.is_empty() {
        return Err(TrajectoryError::InvalidParameter);
    }

    let t_total = total_time(segments);
    let t = t.clamp_val(S::ZERO, t_total);

    let mut pos = S::ZERO;
    let mut elapsed = S::ZERO;

    for seg in segments.iter() {
        if t <= elapsed + seg.duration {
            // t falls within this segment.
            let dt = t - elapsed;
            let p = seg.v_start * dt + S::HALF * seg.accel * dt * dt;
            let v = seg.v_start + seg.accel * dt;
            return Ok((pos + p, v));
        }
        // Accumulate full segment.
        pos += seg.v_start * seg.duration + S::HALF * seg.accel * seg.duration * seg.duration;
        elapsed += seg.duration;
    }

    // At or beyond end.
    let last = segments[segments.len() - 1];
    Ok((pos, last.v_end))
}

#[cfg(test)]
mod segment_tests {
    use super::*;

    /// Full trapezoidal profile: position at end should equal distance.
    #[test]
    fn trapezoidal_profile_reaches_distance() {
        let segs = plan_1dof(10.0_f64, 5.0, 2.0, 0.0, 0.0).expect("should plan");
        assert!(segs.len() >= 2, "expected ≥2 segments, got {}", segs.len());
        let t_tot = total_time(&segs);
        let (pos, vel) = sample_at(&segs, t_tot).expect("sample ok");
        assert!((pos - 10.0).abs() < 1e-9, "pos={}", pos);
        assert!(vel.abs() < 1e-9, "vel={}", vel);
    }

    /// Triangular profile: v_max not reached.
    #[test]
    fn triangular_profile_short_distance() {
        // For v_max=5, a_max=2, v0=0, v1=0: d_min = (5^2)/(2*2) + (5^2)/(2*2) = 12.5.
        // distance=4 < 12.5, so triangular.
        let segs = plan_1dof(4.0_f64, 5.0, 2.0, 0.0, 0.0).expect("should plan");
        let t_tot = total_time(&segs);
        let (pos, _vel) = sample_at(&segs, t_tot).expect("sample ok");
        assert!((pos - 4.0).abs() < 1e-9, "pos={}", pos);
    }

    /// Non-zero start/end velocities.
    #[test]
    fn non_zero_boundary_velocities() {
        let segs = plan_1dof(8.0_f64, 4.0, 2.0, 1.0, 1.0).expect("should plan");
        let t_tot = total_time(&segs);
        let (pos, vel) = sample_at(&segs, t_tot).expect("sample ok");
        assert!((pos - 8.0).abs() < 1e-9, "pos={}", pos);
        assert!((vel - 1.0).abs() < 1e-6, "vel={}", vel);
    }

    /// sample_at(0) returns zero position.
    #[test]
    fn sample_at_zero_returns_start() {
        let segs = plan_1dof(5.0_f64, 3.0, 1.5, 0.0, 0.0).expect("plan ok");
        let (pos, vel) = sample_at(&segs, 0.0).expect("sample ok");
        assert!(pos.abs() < 1e-12, "pos={}", pos);
        assert!(vel.abs() < 1e-12, "vel={}", vel);
    }

    /// total_time for zero segments returns zero.
    #[test]
    fn total_time_empty_is_zero() {
        let segs: HVec<TimeOptimalSegment<f64>, 3> = HVec::new();
        assert_eq!(total_time(&segs), 0.0);
    }

    /// Invalid parameters return error.
    #[test]
    fn invalid_v_max_returns_error() {
        let r = plan_1dof(5.0_f64, 0.0, 2.0, 0.0, 0.0);
        assert!(r.is_err());
    }

    #[test]
    fn invalid_a_max_returns_error() {
        let r = plan_1dof(5.0_f64, 3.0, 0.0, 0.0, 0.0);
        assert!(r.is_err());
    }

    /// Position is monotonically increasing for a forward move.
    #[test]
    fn position_monotonically_increases() {
        let segs = plan_1dof(10.0_f64, 5.0, 2.0, 0.0, 0.0).expect("plan ok");
        let t_tot = total_time(&segs);
        let n = 50usize;
        let mut prev_pos = -1.0_f64;
        for i in 0..=n {
            let t = t_tot * (i as f64) / (n as f64);
            let (pos, _) = sample_at(&segs, t).expect("sample ok");
            assert!(
                pos >= prev_pos - 1e-10,
                "pos not monotone at t={}: {}",
                t,
                pos
            );
            prev_pos = pos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_move_reaches_target() {
        let p = TimeOptimalProfile::plan(0.0_f64, 0.0, 10.0, 0.0, 5.0, 2.0);
        assert!(p.is_valid());
        let (x_end, v_end, _) = p.query(p.duration());
        assert!((x_end - 10.0).abs() < 1e-6, "x_end={}", x_end);
        assert!(v_end.abs() < 1e-6, "v_end={}", v_end);
    }

    #[test]
    fn backward_move_reaches_target() {
        let p = TimeOptimalProfile::plan(0.0_f64, 0.0, -8.0, 0.0, 4.0, 2.0);
        assert!(p.is_valid());
        let (x_end, v_end, _) = p.query(p.duration());
        assert!((x_end + 8.0).abs() < 1e-4, "x_end={}", x_end);
        assert!(v_end.abs() < 1e-4, "v_end={}", v_end);
    }

    #[test]
    fn zero_displacement_profile() {
        let p = TimeOptimalProfile::plan(5.0_f64, 0.0, 5.0, 0.0, 3.0, 2.0);
        // Zero or near-zero displacement: either not valid or immediately done
        let (x_end, _, _) = p.query(p.duration());
        assert!((x_end - 5.0).abs() < 1e-6, "x_end={}", x_end);
    }

    #[test]
    fn initial_velocity_handled() {
        // Start with v0=2.0, move forward to x=10.
        let p = TimeOptimalProfile::plan(0.0_f64, 2.0, 10.0, 0.0, 5.0, 3.0);
        assert!(p.is_valid());
        let (x_end, v_end, _) = p.query(p.duration());
        assert!((x_end - 10.0).abs() < 1e-4, "x_end={}", x_end);
        assert!(v_end.abs() < 1e-4, "v_end={}", v_end);
    }
}
