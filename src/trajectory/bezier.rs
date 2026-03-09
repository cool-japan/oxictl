use crate::core::scalar::ControlScalar;

/// Cubic Bézier curve in N-dimensional space.
///
/// Parameterized by t ∈ [0, 1]:
///   B(t) = (1-t)³ P0 + 3(1-t)²t P1 + 3(1-t)t² P2 + t³ P3
///
/// Control points: P0 (start), P1, P2, P3 (end).
/// P0 and P3 are interpolated; P1 and P2 are tangent handles.
///
/// DIM: dimensionality (2 for planar, 3 for spatial)
#[derive(Debug, Clone, Copy)]
pub struct BezierCurve<S: ControlScalar, const DIM: usize> {
    /// Control points [P0, P1, P2, P3].
    pub control: [[S; DIM]; 4],
}

impl<S: ControlScalar, const DIM: usize> BezierCurve<S, DIM> {
    /// Create a cubic Bézier curve from 4 control points.
    pub fn new(p0: [S; DIM], p1: [S; DIM], p2: [S; DIM], p3: [S; DIM]) -> Self {
        Self {
            control: [p0, p1, p2, p3],
        }
    }

    /// Create a straight-line Bézier (P1 and P2 are 1/3 and 2/3 along).
    pub fn line(p0: [S; DIM], p1: [S; DIM]) -> Self {
        let third = S::ONE / S::from_f64(3.0);
        let two_thirds = S::TWO / S::from_f64(3.0);
        let cp1 = core::array::from_fn(|i| p0[i] + third * (p1[i] - p0[i]));
        let cp2 = core::array::from_fn(|i| p0[i] + two_thirds * (p1[i] - p0[i]));
        Self::new(p0, cp1, cp2, p1)
    }

    /// Evaluate the curve at parameter `t` ∈ [0, 1].
    ///
    /// Clamps `t` to [0, 1].
    pub fn evaluate(&self, t: S) -> [S; DIM] {
        let t = t.clamp_val(S::ZERO, S::ONE);
        let mt = S::ONE - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;
        let t2 = t * t;
        let t3 = t2 * t;
        let three = S::from_f64(3.0);

        core::array::from_fn(|i| {
            mt3 * self.control[0][i]
                + three * mt2 * t * self.control[1][i]
                + three * mt * t2 * self.control[2][i]
                + t3 * self.control[3][i]
        })
    }

    /// Evaluate the first derivative (tangent) at `t`.
    ///
    /// B'(t) = 3[(1-t)²(P1-P0) + 2(1-t)t(P2-P1) + t²(P3-P2)]
    pub fn derivative(&self, t: S) -> [S; DIM] {
        let t = t.clamp_val(S::ZERO, S::ONE);
        let mt = S::ONE - t;
        let mt2 = mt * mt;
        let t2 = t * t;
        let three = S::from_f64(3.0);
        let two = S::TWO;

        core::array::from_fn(|i| {
            three
                * (mt2 * (self.control[1][i] - self.control[0][i])
                    + two * mt * t * (self.control[2][i] - self.control[1][i])
                    + t2 * (self.control[3][i] - self.control[2][i]))
        })
    }

    /// Evaluate the second derivative (curvature) at `t`.
    ///
    /// B''(t) = 6[(1-t)(P2-2P1+P0) + t(P3-2P2+P1)]
    pub fn second_derivative(&self, t: S) -> [S; DIM] {
        let t = t.clamp_val(S::ZERO, S::ONE);
        let mt = S::ONE - t;
        let six = S::from_f64(6.0);
        let two = S::TWO;

        core::array::from_fn(|i| {
            six * (mt * (self.control[2][i] - two * self.control[1][i] + self.control[0][i])
                + t * (self.control[3][i] - two * self.control[2][i] + self.control[1][i]))
        })
    }

    /// Split curve at parameter `t` into two sub-curves (de Casteljau).
    pub fn split(&self, t: S) -> (Self, Self) {
        let t = t.clamp_val(S::ZERO, S::ONE);
        let mt = S::ONE - t;

        // de Casteljau algorithm
        let q0: [S; DIM] =
            core::array::from_fn(|i| mt * self.control[0][i] + t * self.control[1][i]);
        let q1: [S; DIM] =
            core::array::from_fn(|i| mt * self.control[1][i] + t * self.control[2][i]);
        let q2: [S; DIM] =
            core::array::from_fn(|i| mt * self.control[2][i] + t * self.control[3][i]);

        let r0: [S; DIM] = core::array::from_fn(|i| mt * q0[i] + t * q1[i]);
        let r1: [S; DIM] = core::array::from_fn(|i| mt * q1[i] + t * q2[i]);

        let s: [S; DIM] = core::array::from_fn(|i| mt * r0[i] + t * r1[i]);

        (
            Self::new(self.control[0], q0, r0, s),
            Self::new(s, r1, q2, self.control[3]),
        )
    }

    /// Approximate arc length using N-segment numerical integration.
    /// Uses N=16 for accuracy without allocation.
    pub fn arc_length(&self) -> S {
        let n = 16usize;
        let dt = S::ONE / S::from_f64(n as f64);
        let mut length = S::ZERO;
        let mut prev = self.evaluate(S::ZERO);

        for i in 1..=n {
            let t = S::from_f64(i as f64) * dt;
            let curr = self.evaluate(t);
            let seg_len: S = (0..DIM)
                .map(|d| (curr[d] - prev[d]) * (curr[d] - prev[d]))
                .fold(S::ZERO, |a, b| a + b)
                .sqrt();
            length += seg_len;
            prev = curr;
        }

        length
    }
}

/// Composite Bézier path through N segments.
///
/// Each segment is a cubic Bézier. Continuity between segments
/// is enforced by the user when setting control points.
#[derive(Debug, Clone, Copy)]
pub struct BezierPath<S: ControlScalar, const DIM: usize, const SEGS: usize> {
    pub segments: [BezierCurve<S, DIM>; SEGS],
}

impl<S: ControlScalar, const DIM: usize, const SEGS: usize> BezierPath<S, DIM, SEGS> {
    pub fn new(segments: [BezierCurve<S, DIM>; SEGS]) -> Self {
        Self { segments }
    }

    /// Evaluate path at global parameter `t` ∈ [0, SEGS].
    ///
    /// `t` in [k, k+1) maps to segment k with local parameter t-k.
    pub fn evaluate(&self, t: S) -> [S; DIM] {
        let t_clamped = t.clamp_val(S::ZERO, S::from_f64(SEGS as f64));
        let seg_f = t_clamped.min(S::from_f64((SEGS - 1) as f64));
        let seg_idx = {
            let mut idx = 0usize;
            for i in 0..SEGS {
                if seg_f >= S::from_f64(i as f64) {
                    idx = i;
                }
            }
            idx
        };
        let local_t = t_clamped - S::from_f64(seg_idx as f64);
        self.segments[seg_idx].evaluate(local_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_interpolated() {
        let p0 = [0.0_f64, 0.0];
        let p1 = [1.0, 2.0];
        let p2 = [2.0, 2.0];
        let p3 = [3.0, 0.0];
        let c = BezierCurve::new(p0, p1, p2, p3);
        let start = c.evaluate(0.0);
        let end = c.evaluate(1.0);
        assert!((start[0] - 0.0).abs() < 1e-10 && (start[1] - 0.0).abs() < 1e-10);
        assert!((end[0] - 3.0).abs() < 1e-10 && (end[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn line_is_straight() {
        let c = BezierCurve::line([0.0_f64, 0.0], [4.0, 0.0]);
        let mid = c.evaluate(0.5);
        assert!((mid[0] - 2.0).abs() < 1e-10, "mid_x={}", mid[0]);
        assert!(mid[1].abs() < 1e-10, "mid_y={}", mid[1]);
    }

    #[test]
    fn derivative_at_start() {
        // At t=0: B'(0) = 3*(P1-P0)
        let p0 = [0.0_f64, 0.0];
        let p1 = [1.0, 0.0];
        let p2 = [2.0, 0.0];
        let p3 = [3.0, 0.0];
        let c = BezierCurve::new(p0, p1, p2, p3);
        let d = c.derivative(0.0);
        assert!((d[0] - 3.0).abs() < 1e-10, "d[0]={}", d[0]);
        assert!(d[1].abs() < 1e-10);
    }

    #[test]
    fn split_reconstructs_curve() {
        let c = BezierCurve::new([0.0_f64, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0]);
        let (c1, c2) = c.split(0.5);
        // At the split point, both curves should give the same value as original at t=0.5
        let orig = c.evaluate(0.5);
        let split_pt1 = c1.evaluate(1.0);
        let split_pt2 = c2.evaluate(0.0);
        for d in 0..2 {
            assert!(
                (split_pt1[d] - orig[d]).abs() < 1e-10,
                "d={} split1={}",
                d,
                split_pt1[d]
            );
            assert!(
                (split_pt2[d] - orig[d]).abs() < 1e-10,
                "d={} split2={}",
                d,
                split_pt2[d]
            );
        }
    }

    #[test]
    fn arc_length_line() {
        // Straight line from (0,0) to (3,4): length = 5.0
        let c = BezierCurve::line([0.0_f64, 0.0], [3.0, 4.0]);
        let len = c.arc_length();
        assert!((len - 5.0).abs() < 0.01, "len={}", len);
    }

    #[test]
    fn clamp_outside_range() {
        let c = BezierCurve::line([0.0_f64], [1.0]);
        assert!((c.evaluate(-1.0)[0] - 0.0).abs() < 1e-10);
        assert!((c.evaluate(2.0)[0] - 1.0).abs() < 1e-10);
    }
}
