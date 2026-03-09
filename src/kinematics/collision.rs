//! Bounding volume collision detection for robot links.
//!
//! Provides axis-aligned bounding boxes (AABB) and capsule bounding volumes
//! for fast self-collision detection in robot arms.  All arithmetic uses
//! `libm` via `ControlScalar` — no std required.
#![allow(clippy::needless_range_loop)]

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb<S: ControlScalar> {
    /// Minimum corner [x, y, z].
    pub min: [S; 3],
    /// Maximum corner [x, y, z].
    pub max: [S; 3],
}

impl<S: ControlScalar> Aabb<S> {
    /// Create a new AABB.
    pub fn new(min: [S; 3], max: [S; 3]) -> Self {
        Self { min, max }
    }

    /// Returns true if this AABB overlaps `other` (touching counts as overlap).
    pub fn intersects(&self, other: &Self) -> bool {
        for i in 0..3 {
            if self.max[i] < other.min[i] || other.max[i] < self.min[i] {
                return false;
            }
        }
        true
    }

    /// Returns true if point `p` is inside this AABB.
    pub fn contains_point(&self, p: [S; 3]) -> bool {
        for i in 0..3 {
            if p[i] < self.min[i] || p[i] > self.max[i] {
                return false;
            }
        }
        true
    }

    /// Return a new AABB expanded by `margin` in all directions.
    pub fn expand(&self, margin: S) -> Self {
        Self {
            min: core::array::from_fn(|i| self.min[i] - margin),
            max: core::array::from_fn(|i| self.max[i] + margin),
        }
    }
}

// ---------------------------------------------------------------------------
// Capsule
// ---------------------------------------------------------------------------

/// Capsule: a line segment [p0, p1] swept by `radius`.
#[derive(Debug, Clone, Copy)]
pub struct Capsule<S: ControlScalar> {
    /// Start of the central segment.
    pub p0: [S; 3],
    /// End of the central segment.
    pub p1: [S; 3],
    /// Capsule radius.
    pub radius: S,
}

impl<S: ControlScalar> Capsule<S> {
    /// Create a new capsule.
    pub fn new(p0: [S; 3], p1: [S; 3], radius: S) -> Self {
        Self { p0, p1, radius }
    }

    /// Length of the central segment.
    pub fn length(&self) -> S {
        let dx = self.p1[0] - self.p0[0];
        let dy = self.p1[1] - self.p0[1];
        let dz = self.p1[2] - self.p0[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Segment-to-segment distance
// ---------------------------------------------------------------------------

/// Compute the minimum distance between two line segments [p0,p1] and [q0,q1].
///
/// Uses the GJK-style parametric approach (Ericson 2005, §5.1.9).
pub fn segment_distance<S: ControlScalar>(p0: [S; 3], p1: [S; 3], q0: [S; 3], q1: [S; 3]) -> S {
    let d1 = vec3_sub(p1, p0); // direction of segment P
    let d2 = vec3_sub(q1, q0); // direction of segment Q
    let r = vec3_sub(p0, q0);

    let a = vec3_dot(d1, d1); // squared length of P
    let e = vec3_dot(d2, d2); // squared length of Q
    let f = vec3_dot(d2, r);
    let c = vec3_dot(d1, r);

    let eps = S::EPSILON * S::from_f64(1e6);

    let mut s;
    let mut t;

    if a <= eps && e <= eps {
        // Both segments degenerate into points
        return vec3_len(r);
    }
    if a <= eps {
        // P degenerates into a point
        s = S::ZERO;
        t = (f / e).clamp_val(S::ZERO, S::ONE);
    } else if e <= eps {
        // Q degenerates into a point
        t = S::ZERO;
        s = (-c / a).clamp_val(S::ZERO, S::ONE);
    } else {
        let b = vec3_dot(d1, d2);
        let denom = a * e - b * b;
        if denom.abs() > eps {
            s = ((b * f - c * e) / denom).clamp_val(S::ZERO, S::ONE);
        } else {
            s = S::ZERO; // parallel segments — arbitrary s
        }
        // Compute t for current s, then re-clamp
        t = (b * s + f) / e;
        if t < S::ZERO {
            t = S::ZERO;
            s = (-c / a).clamp_val(S::ZERO, S::ONE);
        } else if t > S::ONE {
            t = S::ONE;
            s = ((b - c) / a).clamp_val(S::ZERO, S::ONE);
        }
    }

    // Closest points: cp = p0 + s*d1,  cq = q0 + t*d2
    let cp = vec3_add(p0, vec3_scale(d1, s));
    let cq = vec3_add(q0, vec3_scale(d2, t));
    vec3_len(vec3_sub(cp, cq))
}

/// Distance between two capsules (surface-to-surface).
///
/// Returns 0.0 if the capsules overlap.
pub fn capsule_distance<S: ControlScalar>(a: &Capsule<S>, b: &Capsule<S>) -> S {
    let seg_dist = segment_distance(a.p0, a.p1, b.p0, b.p1);
    let surface_dist = seg_dist - a.radius - b.radius;
    if surface_dist < S::ZERO {
        S::ZERO
    } else {
        surface_dist
    }
}

// ---------------------------------------------------------------------------
// Self-collision checker
// ---------------------------------------------------------------------------

/// Self-collision checker for a robot with N links, each represented as a capsule.
pub struct SelfCollisionChecker<S: ControlScalar, const N: usize> {
    /// Per-link capsules in world frame.
    pub capsules: [Capsule<S>; N],
    /// Minimum allowed distance between non-adjacent links.
    pub min_distance: S,
}

impl<S: ControlScalar, const N: usize> SelfCollisionChecker<S, N> {
    /// Create a new self-collision checker.
    pub fn new(capsules: [Capsule<S>; N], min_distance: S) -> Self {
        Self {
            capsules,
            min_distance,
        }
    }

    /// Update the capsule for link `i` (after FK update).
    pub fn update_link(&mut self, i: usize, capsule: Capsule<S>) {
        self.capsules[i] = capsule;
    }

    /// Check all non-adjacent link pairs.
    ///
    /// Returns `true` if any pair has distance < `min_distance`.
    /// Adjacent links (|i - j| == 1) are skipped as they share a joint.
    pub fn check_self_collision(&self) -> bool {
        for i in 0..N {
            for j in (i + 2)..N {
                if self.link_distance(i, j) < self.min_distance {
                    return true;
                }
            }
        }
        false
    }

    /// Surface-to-surface distance between link `i` and link `j`.
    pub fn link_distance(&self, i: usize, j: usize) -> S {
        capsule_distance(&self.capsules[i], &self.capsules[j])
    }
}

// ---------------------------------------------------------------------------
// 3-D vector helpers (no_std)
// ---------------------------------------------------------------------------

#[inline]
fn vec3_sub<S: ControlScalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn vec3_add<S: ControlScalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn vec3_scale<S: ControlScalar>(a: [S; 3], s: S) -> [S; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn vec3_dot<S: ControlScalar>(a: [S; 3], b: [S; 3]) -> S {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn vec3_len<S: ControlScalar>(a: [S; 3]) -> S {
    vec3_dot(a, a).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_intersection() {
        let a = Aabb::new([0.0_f64; 3], [1.0; 3]);
        let b = Aabb::new([0.5; 3], [1.5; 3]);
        assert!(a.intersects(&b));
        let c = Aabb::new([2.0; 3], [3.0; 3]);
        assert!(!a.intersects(&c));
    }

    #[test]
    fn aabb_contains_point() {
        let a = Aabb::new([0.0_f64; 3], [1.0; 3]);
        assert!(a.contains_point([0.5; 3]));
        assert!(!a.contains_point([1.5; 3]));
    }

    #[test]
    fn capsule_length() {
        let c = Capsule::new([0.0_f64; 3], [3.0, 4.0, 0.0], 0.1);
        assert!((c.length() - 5.0).abs() < 1e-10, "len={}", c.length());
    }

    #[test]
    fn segment_distance_parallel_segments() {
        // Two parallel segments offset by 1 unit in Y
        let dist = segment_distance(
            [0.0_f64, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0_f64, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        );
        assert!((dist - 1.0).abs() < 1e-9, "dist={dist}");
    }

    #[test]
    fn capsule_distance_no_overlap() {
        let a = Capsule::new([0.0_f64; 3], [1.0, 0.0, 0.0], 0.1);
        let b = Capsule::new([2.0, 0.0, 0.0_f64], [3.0, 0.0, 0.0], 0.1);
        let d = capsule_distance(&a, &b);
        // Gap between surfaces: 2.0 - 1.0 - 0.1 - 0.1 = 0.8
        assert!((d - 0.8).abs() < 1e-9, "d={d}");
    }

    #[test]
    fn self_collision_checker_no_collision() {
        // 3 links in a straight line, well separated
        let caps = [
            Capsule::new([0.0_f64; 3], [1.0, 0.0, 0.0], 0.05),
            Capsule::new([1.0, 0.0, 0.0_f64], [2.0, 0.0, 0.0], 0.05),
            Capsule::new([2.0, 0.0, 0.0_f64], [3.0, 0.0, 0.0], 0.05),
        ];
        let checker = SelfCollisionChecker::new(caps, 0.01);
        // Links 0 and 2 are non-adjacent; distance ~ 1.0 - 0.05 - 0.05 = 0.9
        assert!(!checker.check_self_collision());
    }

    #[test]
    fn self_collision_checker_detects_collision() {
        // Links 0 and 2 overlap (folded arm)
        let caps = [
            Capsule::new([0.0_f64; 3], [1.0, 0.0, 0.0], 0.1),
            Capsule::new([1.0, 0.0, 0.0_f64], [0.5, 0.1, 0.0], 0.1),
            // Link 2 placed on top of link 0
            Capsule::new([0.5, 0.1, 0.0_f64], [0.1, 0.0, 0.0], 0.1),
        ];
        let checker = SelfCollisionChecker::new(caps, 0.01);
        assert!(checker.check_self_collision());
    }
}
