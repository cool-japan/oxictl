//! Dubins path planner.
//!
//! A Dubins path is the shortest curve connecting two configurations `(x, y, θ)`
//! in the plane, subject to a minimum turning radius `ρ` (no reversing allowed).
//!
//! The six path types are: `LSL`, `RSR`, `LSR`, `RSL`, `RLR`, `LRL`, where
//! `L` = left turn (counter-clockwise), `R` = right turn (clockwise), and
//! `S` = straight segment.
//!
//! # References
//! Dubins, L. E. (1957). "On curves of minimal length with a constraint on
//! average curvature". *American Journal of Mathematics*, 79(3), 497–516.

use crate::core::scalar::ControlScalar;
use crate::trajectory::TrajectoryError;

// ────────────────────────────────────────────────────────────────────────────
// Path type
// ────────────────────────────────────────────────────────────────────────────

/// The six Dubins path primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DubinsPathType {
    /// Left – Straight – Left
    LSL,
    /// Right – Straight – Right
    RSR,
    /// Left – Straight – Right
    LSR,
    /// Right – Straight – Left
    RSL,
    /// Right – Left – Right
    RLR,
    /// Left – Right – Left
    LRL,
}

// ────────────────────────────────────────────────────────────────────────────
// DubinsPath
// ────────────────────────────────────────────────────────────────────────────

/// A Dubins path between two `(x, y, θ)` configurations.
///
/// Segment durations `t1`, `t2`, `t3` are dimensionless (radians for turns,
/// distance/rho for straight segments).  Total arc length = `(t1+t2+t3)*rho`.
#[derive(Debug, Clone, Copy)]
pub struct DubinsPath<S: ControlScalar> {
    /// Path primitive type.
    pub path_type: DubinsPathType,
    /// Normalised duration of segment 1 (non-negative).
    pub t1: S,
    /// Normalised duration of segment 2 (non-negative).
    pub t2: S,
    /// Normalised duration of segment 3 (non-negative).
    pub t3: S,
    /// Start configuration `[x, y, heading_rad]`.
    pub q0: [S; 3],
    /// Minimum turning radius.
    pub rho: S,
}

// ────────────────────────────────────────────────────────────────────────────
// Angle utilities (operate purely in f64 then convert back)
// ────────────────────────────────────────────────────────────────────────────

/// Wrap angle into [0, 2π).
#[inline]
fn mod2pi(theta: f64) -> f64 {
    let two_pi = core::f64::consts::PI * 2.0;
    let v = theta - (theta / two_pi).floor() * two_pi;
    if v < 0.0 {
        v + two_pi
    } else {
        v
    }
}

/// Wrap angle into (-π, π].
#[inline]
fn wrap_pi(theta: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let two_pi = pi * 2.0;
    let mut w = theta - (theta / two_pi).round() * two_pi;
    if w > pi {
        w -= two_pi;
    } else if w <= -pi {
        w += two_pi;
    }
    w
}

// ────────────────────────────────────────────────────────────────────────────
// Path-type computations (all in f64, normalised to ρ = 1)
// ────────────────────────────────────────────────────────────────────────────

/// Returns `Some((t, p, q))` if LSL is feasible, else `None`.
fn lsl(alpha: f64, beta: f64, d: f64) -> Option<(f64, f64, f64)> {
    let p_sq = 2.0 + d * d - 2.0 * libm::cos(alpha - beta)
        + 2.0 * d * (libm::sin(alpha) - libm::sin(beta));
    if p_sq < 0.0 {
        return None;
    }
    let p = libm::sqrt(p_sq);
    let tmp = libm::atan2(
        libm::cos(beta) - libm::cos(alpha),
        d + libm::sin(alpha) - libm::sin(beta),
    );
    let t = mod2pi(-alpha + tmp);
    let q = mod2pi(beta - tmp);
    Some((t, p, q))
}

/// Returns `Some((t, p, q))` if RSR is feasible, else `None`.
fn rsr(alpha: f64, beta: f64, d: f64) -> Option<(f64, f64, f64)> {
    let p_sq = 2.0 + d * d
        - 2.0 * libm::cos(alpha - beta)
        - 2.0 * d * (libm::sin(alpha) - libm::sin(beta));
    if p_sq < 0.0 {
        return None;
    }
    let p = libm::sqrt(p_sq);
    let tmp = libm::atan2(
        libm::cos(alpha) - libm::cos(beta),
        d - libm::sin(alpha) + libm::sin(beta),
    );
    let t = mod2pi(alpha - tmp);
    let q = mod2pi(mod2pi(-beta) + tmp);
    Some((t, p, q))
}

/// Returns `Some((t, p, q))` if LSR is feasible, else `None`.
fn lsr(alpha: f64, beta: f64, d: f64) -> Option<(f64, f64, f64)> {
    let p_sq = -2.0
        + d * d
        + 2.0 * libm::cos(alpha - beta)
        + 2.0 * d * (libm::sin(alpha) + libm::sin(beta));
    if p_sq < 0.0 {
        return None;
    }
    let p = libm::sqrt(p_sq);
    let tmp = libm::atan2(
        -libm::cos(alpha) - libm::cos(beta),
        d + libm::sin(alpha) + libm::sin(beta),
    ) - libm::atan2(-2.0, p);
    let t = mod2pi(-alpha + tmp);
    let q = mod2pi(mod2pi(-beta) + tmp);
    Some((t, p, q))
}

/// Returns `Some((t, p, q))` if RSL is feasible, else `None`.
fn rsl(alpha: f64, beta: f64, d: f64) -> Option<(f64, f64, f64)> {
    let p_sq = -2.0 + d * d + 2.0 * libm::cos(alpha - beta)
        - 2.0 * d * (libm::sin(alpha) + libm::sin(beta));
    if p_sq < 0.0 {
        return None;
    }
    let p = libm::sqrt(p_sq);
    let tmp = libm::atan2(
        libm::cos(alpha) + libm::cos(beta),
        d - libm::sin(alpha) - libm::sin(beta),
    ) - libm::atan2(2.0, p);
    let t = mod2pi(alpha - tmp);
    let q = mod2pi(beta - tmp);
    Some((t, p, q))
}

/// Returns `Some((t, p, q))` if RLR is feasible, else `None`.
fn rlr(alpha: f64, beta: f64, d: f64) -> Option<(f64, f64, f64)> {
    let tmp = (6.0 - d * d
        + 2.0 * libm::cos(alpha - beta)
        + 2.0 * d * (libm::sin(alpha) - libm::sin(beta)))
        / 8.0;
    if libm::fabs(tmp) > 1.0 {
        return None;
    }
    let p = mod2pi(2.0 * core::f64::consts::PI - libm::acos(tmp));
    let t = mod2pi(
        alpha
            - libm::atan2(
                libm::cos(alpha) - libm::cos(beta),
                d - libm::sin(alpha) + libm::sin(beta),
            )
            + mod2pi(p / 2.0),
    );
    let q = mod2pi(alpha - beta - t + mod2pi(p));
    Some((t, p, q))
}

/// Returns `Some((t, p, q))` if LRL is feasible, else `None`.
fn lrl(alpha: f64, beta: f64, d: f64) -> Option<(f64, f64, f64)> {
    let tmp = (6.0 - d * d + 2.0 * libm::cos(alpha - beta)
        - 2.0 * d * (libm::sin(alpha) - libm::sin(beta)))
        / 8.0;
    if libm::fabs(tmp) > 1.0 {
        return None;
    }
    let p = mod2pi(2.0 * core::f64::consts::PI - libm::acos(tmp));
    let t = mod2pi(
        -alpha
            + libm::atan2(
                -libm::cos(alpha) + libm::cos(beta),
                d + libm::sin(alpha) - libm::sin(beta),
            )
            + mod2pi(p / 2.0),
    );
    let q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p));
    Some((t, p, q))
}

// ────────────────────────────────────────────────────────────────────────────
// Internal segment kind
// ────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum SegmentKind {
    Left,
    Right,
    Straight,
}

// ────────────────────────────────────────────────────────────────────────────
// DubinsPath implementation
// ────────────────────────────────────────────────────────────────────────────

impl<S: ControlScalar> DubinsPath<S> {
    /// Compute the shortest Dubins path from `q0` to `q1` with turning radius `rho`.
    ///
    /// # Arguments
    /// - `q0`  — start configuration `[x, y, heading_rad]`
    /// - `q1`  — goal configuration  `[x, y, heading_rad]`
    /// - `rho` — minimum turning radius (must be > 0)
    ///
    /// # Errors
    /// Returns [`TrajectoryError::InvalidParameter`] if `rho ≤ 0`.
    /// Returns [`TrajectoryError::NoPathFound`] if no feasible path exists.
    pub fn shortest_path(q0: [S; 3], q1: [S; 3], rho: S) -> Result<Self, TrajectoryError> {
        if rho <= S::ZERO {
            return Err(TrajectoryError::InvalidParameter);
        }

        // Work in f64 for normalised calculations.
        let q0x = q0[0].to_f64();
        let q0y = q0[1].to_f64();
        let q0h = q0[2].to_f64();
        let q1x = q1[0].to_f64();
        let q1y = q1[1].to_f64();
        let q1h = q1[2].to_f64();
        let rho_f = rho.to_f64();

        let dx = (q1x - q0x) / rho_f;
        let dy = (q1y - q0y) / rho_f;
        let d = libm::sqrt(dx * dx + dy * dy);
        let theta = libm::atan2(dy, dx);
        let alpha = wrap_pi(q0h - theta);
        let beta = wrap_pi(q1h - theta);

        // Evaluate all six path types; keep the shortest valid one.
        type Candidate = (DubinsPathType, Option<(f64, f64, f64)>);
        let candidates: [Candidate; 6] = [
            (DubinsPathType::LSL, lsl(alpha, beta, d)),
            (DubinsPathType::RSR, rsr(alpha, beta, d)),
            (DubinsPathType::LSR, lsr(alpha, beta, d)),
            (DubinsPathType::RSL, rsl(alpha, beta, d)),
            (DubinsPathType::RLR, rlr(alpha, beta, d)),
            (DubinsPathType::LRL, lrl(alpha, beta, d)),
        ];

        let mut best: Option<(DubinsPathType, f64, f64, f64)> = None;
        for (ptype, maybe) in &candidates {
            if let Some((t, p, q)) = maybe {
                let total = t + p + q;
                if total >= 0.0 {
                    let is_better = match &best {
                        None => true,
                        Some((_, bt, bp, bq)) => total < bt + bp + bq,
                    };
                    if is_better {
                        best = Some((*ptype, *t, *p, *q));
                    }
                }
            }
        }

        match best {
            None => Err(TrajectoryError::NoPathFound),
            Some((ptype, t1, t2, t3)) => Ok(Self {
                path_type: ptype,
                t1: S::from_f64(t1),
                t2: S::from_f64(t2),
                t3: S::from_f64(t3),
                q0,
                rho,
            }),
        }
    }

    /// Total arc length of the path.
    #[inline]
    pub fn length(&self) -> S {
        (self.t1 + self.t2 + self.t3) * self.rho
    }

    /// Sample the path at arc-length `s ∈ [0, length()]`.
    ///
    /// Returns `[x, y, heading_rad]`.
    pub fn sample_at(&self, s: S) -> [S; 3] {
        // Normalised arc length in ρ=1 frame.
        let t_max = self.t1 + self.t2 + self.t3;
        let t = (s / self.rho).clamp_val(S::ZERO, t_max);
        let (x, y, heading) = self.integrate_segments(t);
        [x, y, heading]
    }

    /// Integrate position along the path up to normalised arc length `t`.
    fn integrate_segments(&self, t: S) -> (S, S, S) {
        let (seg1, seg2, seg3) = self.segment_types();

        let mut x = self.q0[0];
        let mut y = self.q0[1];
        let mut h = self.q0[2];

        // Segment 1.
        let dt1 = t.min(self.t1);
        let (dx, dy, dh) = Self::segment_delta(seg1, dt1, self.rho, h);
        x += dx;
        y += dy;
        h += dh;

        if t <= self.t1 {
            return (x, y, h);
        }

        // Segment 2.
        let dt2 = (t - self.t1).min(self.t2);
        let (dx, dy, dh) = Self::segment_delta(seg2, dt2, self.rho, h);
        x += dx;
        y += dy;
        h += dh;

        if t <= self.t1 + self.t2 {
            return (x, y, h);
        }

        // Segment 3.
        let dt3 = (t - self.t1 - self.t2).min(self.t3);
        let (dx, dy, dh) = Self::segment_delta(seg3, dt3, self.rho, h);
        x += dx;
        y += dy;
        h += dh;

        (x, y, h)
    }

    /// Map path type to three segment primitives.
    fn segment_types(&self) -> (SegmentKind, SegmentKind, SegmentKind) {
        match self.path_type {
            DubinsPathType::LSL => (SegmentKind::Left, SegmentKind::Straight, SegmentKind::Left),
            DubinsPathType::RSR => (
                SegmentKind::Right,
                SegmentKind::Straight,
                SegmentKind::Right,
            ),
            DubinsPathType::LSR => (SegmentKind::Left, SegmentKind::Straight, SegmentKind::Right),
            DubinsPathType::RSL => (SegmentKind::Right, SegmentKind::Straight, SegmentKind::Left),
            DubinsPathType::RLR => (SegmentKind::Right, SegmentKind::Left, SegmentKind::Right),
            DubinsPathType::LRL => (SegmentKind::Left, SegmentKind::Right, SegmentKind::Left),
        }
    }

    /// Compute `(Δx, Δy, Δheading)` for one primitive segment.
    ///
    /// `dt`  — normalised duration (radians for turns, distance/rho for straight)
    /// `rho` — turning radius
    /// `h`   — current heading in radians
    fn segment_delta(kind: SegmentKind, dt: S, rho: S, h: S) -> (S, S, S) {
        let h_f = h.to_f64();
        let dt_f = dt.to_f64();
        let rho_f = rho.to_f64();

        match kind {
            SegmentKind::Straight => {
                // Move along current heading for arc-length dt * rho.
                let arc = dt_f * rho_f;
                let dx = arc * libm::cos(h_f);
                let dy = arc * libm::sin(h_f);
                (S::from_f64(dx), S::from_f64(dy), S::ZERO)
            }
            SegmentKind::Left => {
                // CCW turn: heading increases by dt (radians).
                let new_h = h_f + dt_f;
                // Position relative to turn centre:
                //   centre = (x - rho*sin(h), y + rho*cos(h))
                // After sweep angle dt CCW:
                //   Δx = rho*(sin(h + dt) - sin(h))
                //   Δy = rho*(-cos(h + dt) + cos(h))
                let dx = rho_f * (libm::sin(new_h) - libm::sin(h_f));
                let dy = rho_f * (libm::cos(h_f) - libm::cos(new_h));
                (S::from_f64(dx), S::from_f64(dy), S::from_f64(dt_f))
            }
            SegmentKind::Right => {
                // CW turn: heading decreases by dt (radians).
                let new_h = h_f - dt_f;
                let dx = rho_f * (-libm::sin(new_h) + libm::sin(h_f));
                let dy = rho_f * (libm::cos(new_h) - libm::cos(h_f));
                (S::from_f64(dx), S::from_f64(dy), S::from_f64(-dt_f))
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    const TOL: f64 = 1e-9;

    fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
        libm::sqrt((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2))
    }

    /// sample_at(0) must equal the start configuration.
    #[test]
    fn sample_at_zero_equals_start() {
        let q0 = [1.0_f64, 2.0, PI / 4.0];
        let q1 = [5.0_f64, 5.0, PI / 2.0];
        let path = DubinsPath::shortest_path(q0, q1, 0.8_f64).expect("path must exist");
        let start = path.sample_at(0.0_f64);
        assert!(
            (start[0] - q0[0]).abs() < TOL,
            "x: {} vs {}",
            start[0],
            q0[0]
        );
        assert!(
            (start[1] - q0[1]).abs() < TOL,
            "y: {} vs {}",
            start[1],
            q0[1]
        );
        assert!(
            (start[2] - q0[2]).abs() < TOL,
            "h: {} vs {}",
            start[2],
            q0[2]
        );
    }

    /// sample_at(length) must be near the goal position.
    #[test]
    fn sample_at_length_near_goal() {
        let q0 = [0.0_f64, 0.0, 0.0];
        let q1 = [3.0_f64, 3.0, PI / 2.0];
        let path = DubinsPath::shortest_path(q0, q1, 1.0_f64).expect("path must exist");
        let end = path.sample_at(path.length());
        let d = dist2(end, q1);
        assert!(d < 0.02, "end position distance to goal = {}", d);
    }

    /// Invalid turning radius should return an error.
    #[test]
    fn zero_rho_returns_error() {
        let q0 = [0.0_f64, 0.0, 0.0];
        let q1 = [1.0_f64, 0.0, 0.0];
        assert!(DubinsPath::shortest_path(q0, q1, 0.0_f64).is_err());
    }

    #[test]
    fn negative_rho_returns_error() {
        let q0 = [0.0_f64, 0.0, 0.0];
        let q1 = [1.0_f64, 0.0, 0.0];
        assert!(DubinsPath::shortest_path(q0, q1, -1.0_f64).is_err());
    }

    /// Path from q0 to q0 itself should have near-zero length.
    #[test]
    fn path_to_same_config_is_trivial() {
        let q0 = [2.0_f64, -1.0, PI / 3.0];
        let path = DubinsPath::shortest_path(q0, q0, 1.0_f64).expect("path must exist");
        assert!(path.length() < 1e-3, "length={}", path.length());
    }

    /// Dubins length should be ≥ Euclidean distance between positions.
    #[test]
    fn length_ge_euclidean_distance() {
        let q0 = [0.0_f64, 0.0, 0.0];
        let q1 = [4.0_f64, 2.0, PI];
        let path = DubinsPath::shortest_path(q0, q1, 0.5_f64).expect("path must exist");
        let euc = libm::sqrt(16.0 + 4.0);
        assert!(
            path.length() >= euc - 1e-6,
            "length={} < euclid={}",
            path.length(),
            euc
        );
    }

    /// End heading should approximately match the goal heading.
    #[test]
    fn end_heading_matches_goal() {
        let q0 = [0.0_f64, 0.0, 0.0];
        let q1 = [2.0_f64, 2.0, PI / 2.0];
        let path = DubinsPath::shortest_path(q0, q1, 1.0_f64).expect("path must exist");
        let end = path.sample_at(path.length());
        let dh = (end[2] - q1[2]).abs();
        let dh = if dh > PI { (2.0 * PI - dh).abs() } else { dh };
        assert!(dh < 0.05, "heading error={}", dh);
    }

    /// Total length is positive for non-trivial paths.
    #[test]
    fn length_is_positive_for_nontrivial_path() {
        let q0 = [0.0_f64, 0.0, 0.0];
        let q1 = [5.0_f64, 0.0, PI];
        let path = DubinsPath::shortest_path(q0, q1, 1.0_f64).expect("path must exist");
        assert!(path.length() > 0.0);
    }

    /// f32 scalar type compiles and runs correctly.
    #[test]
    fn f32_scalar_type() {
        let q0 = [0.0_f32, 0.0, 0.0];
        let q1 = [3.0_f32, 0.0, 0.0];
        let path = DubinsPath::shortest_path(q0, q1, 1.0_f32).expect("path must exist");
        assert!(path.length() > 0.0);
    }
}
