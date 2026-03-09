#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]
use crate::core::scalar::ControlScalar;

/// Maximum number of knots for BSpline (M + K + 1 ≤ MAX_KNOTS).
pub const MAX_KNOTS: usize = 64;
/// Maximum B-spline degree.
pub const MAX_DEGREE: usize = 7;

/// B-spline trajectory using the de Boor algorithm.
///
/// A B-spline of degree `K` with `M` control points in `DIM` dimensions,
/// using a runtime-specified knot vector of length M + K + 1.
///
/// The maximum supported degree is `MAX_DEGREE` (7) and maximum total knots is
/// `MAX_KNOTS` (64), covering up to M=57 control points at degree 7.
///
/// de Boor's algorithm evaluates the spline at parameter `t` in O(K²).
///
/// Clamped (endpoint-interpolating) B-splines are supported via
/// `clamped_uniform` and `clamped_normalized` constructors.
///
/// # Type parameters
/// - `S`:   scalar type (f32 or f64)
/// - `DIM`: spatial dimension (e.g. 2 for planar, 3 for 3D)
/// - `M`:   number of control points (const, ≤ MAX_KNOTS - 1)
#[derive(Debug, Clone, Copy)]
pub struct BSpline<S: ControlScalar, const DIM: usize, const M: usize> {
    /// Control points, each of length DIM.
    pub control_points: [[S; DIM]; M],
    /// Knot vector (length = M + degree + 1, stored in fixed array).
    pub knots: [S; MAX_KNOTS],
    /// B-spline degree K.
    pub degree: usize,
    /// Number of valid knots: M + degree + 1.
    pub knot_count: usize,
}

impl<S: ControlScalar, const DIM: usize, const M: usize> BSpline<S, DIM, M> {
    /// Create a B-spline with the given control points, knot vector, and degree.
    ///
    /// `knots_slice`: exactly M + degree + 1 values, non-decreasing.
    ///
    /// Returns `None` if:
    /// - M == 0 or degree == 0
    /// - knot_count > MAX_KNOTS
    /// - knots are not non-decreasing
    pub fn new(control_points: [[S; DIM]; M], knots_slice: &[S], degree: usize) -> Option<Self> {
        if M == 0 || degree == 0 {
            return None;
        }
        let expected_knots = M + degree + 1;
        if knots_slice.len() != expected_knots || expected_knots > MAX_KNOTS {
            return None;
        }
        // Check non-decreasing
        for i in 0..(expected_knots - 1) {
            if knots_slice[i + 1] < knots_slice[i] {
                return None;
            }
        }
        let mut knots = [S::ZERO; MAX_KNOTS];
        for (i, &k) in knots_slice.iter().enumerate() {
            knots[i] = k;
        }
        Some(Self {
            control_points,
            knots,
            degree,
            knot_count: expected_knots,
        })
    }

    /// Create a clamped uniform B-spline (endpoints are interpolated).
    ///
    /// Knot vector (degree `k`):
    ///   [0, 0, …, 0,  (k+1 zeros)
    ///    1, 2, …, M-k-1,   (interior knots)
    ///    M-k, M-k, …, M-k]  (k+1 repeats of last)
    ///
    /// Returns `None` if M ≤ degree.
    pub fn clamped_uniform(control_points: [[S; DIM]; M], degree: usize) -> Option<Self> {
        if M <= degree || degree == 0 {
            return None;
        }
        let n_interior = M - degree - 1;
        let last_knot_val = S::from_f64((n_interior + 1) as f64);
        let n_knots = M + degree + 1;
        if n_knots > MAX_KNOTS {
            return None;
        }
        let mut knots_buf = [S::ZERO; MAX_KNOTS];
        // First degree+1 knots = 0
        for i in 0..=degree {
            knots_buf[i] = S::ZERO;
        }
        // Interior knots: 1, 2, …, n_interior
        for j in 0..n_interior {
            knots_buf[degree + 1 + j] = S::from_f64((j + 1) as f64);
        }
        // Last degree+1 knots = last_knot_val
        for i in 0..=degree {
            knots_buf[M + i] = last_knot_val;
        }
        let knots_slice: Vec<S> = knots_buf[..n_knots].to_vec();
        // We have to use the new() path but the data is already validated
        let mut knots = [S::ZERO; MAX_KNOTS];
        for (i, &k) in knots_slice.iter().enumerate() {
            knots[i] = k;
        }
        Some(Self {
            control_points,
            knots,
            degree,
            knot_count: n_knots,
        })
    }

    /// Normalized clamped uniform B-spline with knots in [0, 1].
    pub fn clamped_normalized(control_points: [[S; DIM]; M], degree: usize) -> Option<Self> {
        if M <= degree || degree == 0 {
            return None;
        }
        let n_interior = M - degree - 1;
        let n_knots = M + degree + 1;
        if n_knots > MAX_KNOTS {
            return None;
        }
        let total_span = S::from_f64((n_interior + 1) as f64);
        let mut knots = [S::ZERO; MAX_KNOTS];
        for i in 0..=degree {
            knots[i] = S::ZERO;
        }
        for j in 0..n_interior {
            knots[degree + 1 + j] = S::from_f64((j + 1) as f64) / total_span;
        }
        for i in 0..=degree {
            knots[M + i] = S::ONE;
        }
        Some(Self {
            control_points,
            knots,
            degree,
            knot_count: n_knots,
        })
    }

    /// Find the knot span index `i` such that `knots[i] <= t < knots[i+1]`.
    fn find_span(&self, t: S) -> usize {
        let k = self.degree;
        let n = M - 1;
        let t_max = self.knots[n + 1];
        let t_min = self.knots[k];

        if t >= t_max {
            let mut span = n;
            while span > k && self.knots[span] == t_max {
                if span == 0 {
                    break;
                }
                span -= 1;
            }
            return span;
        }
        if t <= t_min {
            return k;
        }

        let mut lo = k;
        let mut hi = n + 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if t >= self.knots[mid] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Evaluate the B-spline position at parameter `t`.
    ///
    /// Clamps t to valid parameter range.
    pub fn evaluate(&self, t: S) -> [S; DIM] {
        let k = self.degree;
        let t_clamped = t.clamp_val(self.knots[k], self.knots[M]);
        let span = self.find_span(t_clamped);

        // de Boor working array (max degree+1 points)
        let mut d = [[S::ZERO; DIM]; MAX_DEGREE + 1];
        for j in 0..=k {
            let cp_idx = span + j - k;
            if cp_idx < M {
                d[j] = self.control_points[cp_idx];
            }
        }

        for r in 1..=k {
            for j in (r..=k).rev() {
                let i = span + j - k;
                let knot_lo = self.knots[i];
                let knot_hi = self.knots[i + k + 1 - r];
                let denom = knot_hi - knot_lo;
                let alpha = if denom.abs() > S::EPSILON {
                    (t_clamped - knot_lo) / denom
                } else {
                    S::ZERO
                };
                for dim in 0..DIM {
                    d[j][dim] = (S::ONE - alpha) * d[j - 1][dim] + alpha * d[j][dim];
                }
            }
        }

        d[k]
    }

    /// Evaluate the first derivative (velocity) of the B-spline at `t`.
    ///
    /// Derivative of degree-K spline = degree-(K-1) spline of derivative control points.
    pub fn velocity(&self, t: S) -> [S; DIM] {
        let k = self.degree;
        if k == 0 || M < 2 {
            return [S::ZERO; DIM];
        }

        let t_clamped = t.clamp_val(self.knots[k], self.knots[M]);
        let span = self.find_span(t_clamped);
        let k_s = S::from_f64(k as f64);

        // Derivative control points Q_i = K * (P_{i+1} - P_i) / (t_{i+K+1} - t_{i+1})
        // We need Q_{span-K} .. Q_{span-1} (K points)
        let mut q = [[S::ZERO; DIM]; MAX_DEGREE + 1];
        for j in 0..k {
            let i = if span >= k { span - k + j } else { j };
            if i + 1 < M {
                let denom = self.knots[i + k + 1] - self.knots[i + 1];
                let scale = if denom.abs() > S::EPSILON {
                    k_s / denom
                } else {
                    S::ZERO
                };
                for dim in 0..DIM {
                    q[j][dim] =
                        scale * (self.control_points[i + 1][dim] - self.control_points[i][dim]);
                }
            }
        }

        if k == 1 {
            return q[0];
        }

        // de Boor on q with degree k-1
        let km1 = k - 1;
        for r in 1..=km1 {
            for j in (r..=km1).rev() {
                let i = if span >= k { span - k + 1 + j } else { 1 + j };
                let knot_lo = self.knots[i];
                let knot_hi = self.knots[i + k - r];
                let denom = knot_hi - knot_lo;
                let alpha = if denom.abs() > S::EPSILON {
                    (t_clamped - knot_lo) / denom
                } else {
                    S::ZERO
                };
                for dim in 0..DIM {
                    q[j][dim] = (S::ONE - alpha) * q[j - 1][dim] + alpha * q[j][dim];
                }
            }
        }

        q[km1]
    }

    /// Evaluate the second derivative (acceleration) at `t`.
    pub fn acceleration(&self, t: S) -> [S; DIM] {
        let k = self.degree;
        if k < 2 || M < 3 {
            return [S::ZERO; DIM];
        }

        let t_clamped = t.clamp_val(self.knots[k], self.knots[M]);
        let k_s = S::from_f64(k as f64);
        let km1_s = S::from_f64((k - 1) as f64);

        // First derivative control points Q_i (M-1 of them)
        let mut q_all = [[S::ZERO; DIM]; MAX_KNOTS];
        let m1 = M.saturating_sub(1);
        for i in 0..m1 {
            let denom = self.knots[i + k + 1] - self.knots[i + 1];
            let scale = if denom.abs() > S::EPSILON {
                k_s / denom
            } else {
                S::ZERO
            };
            for dim in 0..DIM {
                q_all[i][dim] =
                    scale * (self.control_points[i + 1][dim] - self.control_points[i][dim]);
            }
        }

        // Second derivative control points R_i (M-2 of them)
        let span = self.find_span(t_clamped);
        let start = span.saturating_sub(k);
        let mut r = [[S::ZERO; DIM]; MAX_DEGREE + 1];
        let km1 = k - 1;
        for j in 0..km1 {
            let i = start + j;
            if i + 1 < m1 {
                let denom = self.knots[i + k] - self.knots[i + 1];
                let scale = if denom.abs() > S::EPSILON {
                    km1_s / denom
                } else {
                    S::ZERO
                };
                for dim in 0..DIM {
                    r[j][dim] = scale * (q_all[i + 1][dim] - q_all[i][dim]);
                }
            }
        }

        if k == 2 {
            return r[0];
        }

        let km2 = k - 2;
        for s in 1..=km2 {
            for j in (s..=km2).rev() {
                let i = start + 1 + j;
                let knot_lo = self.knots[i];
                let knot_hi = self.knots[i + k - 1 - s];
                let denom = knot_hi - knot_lo;
                let alpha = if denom.abs() > S::EPSILON {
                    (t_clamped - knot_lo) / denom
                } else {
                    S::ZERO
                };
                for dim in 0..DIM {
                    r[j][dim] = (S::ONE - alpha) * r[j - 1][dim] + alpha * r[j][dim];
                }
            }
        }

        r[km2]
    }

    /// Parameter range: [t_min, t_max].
    pub fn param_range(&self) -> (S, S) {
        (self.knots[self.degree], self.knots[M])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_bspline_interpolates_endpoints() {
        // Degree-1 B-spline with 2 control points = straight line
        // M=2, K=1, knots=[0,0,1,1] (length 4 = M+K+1)
        let cp = [[0.0_f64, 0.0], [1.0, 1.0]];
        let knots = [0.0_f64, 0.0, 1.0, 1.0];
        let bs = BSpline::<f64, 2, 2>::new(cp, &knots, 1).unwrap();
        let p0 = bs.evaluate(0.0);
        let p1 = bs.evaluate(1.0);
        assert!((p0[0]).abs() < 1e-10, "start x={}", p0[0]);
        assert!((p0[1]).abs() < 1e-10, "start y={}", p0[1]);
        assert!((p1[0] - 1.0).abs() < 1e-10, "end x={}", p1[0]);
        assert!((p1[1] - 1.0).abs() < 1e-10, "end y={}", p1[1]);
    }

    #[test]
    fn quadratic_bspline_midpoint() {
        // Degree-2 with 3 control points: M=3, K=2, knots=[0,0,0,1,1,1] (length 6 = M+K+1)
        let cp = [[0.0_f64, 0.0], [0.5, 1.0], [1.0, 0.0]];
        let knots = [0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bs = BSpline::<f64, 2, 3>::new(cp, &knots, 2).unwrap();
        let pm = bs.evaluate(0.5);
        // Quadratic Bezier at t=0.5: (1-t)²P0 + 2t(1-t)P1 + t²P2
        let expected_x = 0.25 * 0.0 + 2.0 * 0.25 * 0.5 + 0.25 * 1.0;
        let expected_y = 0.25 * 0.0 + 2.0 * 0.25 * 1.0 + 0.25 * 0.0;
        assert!((pm[0] - expected_x).abs() < 1e-9, "x={}", pm[0]);
        assert!((pm[1] - expected_y).abs() < 1e-9, "y={}", pm[1]);
    }

    #[test]
    fn clamped_uniform_endpoints_interpolated() {
        // Cubic B-spline: M=5, K=3
        let cp = [
            [0.0_f64, 0.0],
            [1.0, 2.0],
            [2.0, 0.0],
            [3.0, 2.0],
            [4.0, 0.0],
        ];
        let bs = BSpline::<f64, 2, 5>::clamped_uniform(cp, 3).unwrap();
        let (t0, t1) = bs.param_range();
        let p0 = bs.evaluate(t0);
        let p1 = bs.evaluate(t1);
        assert!((p0[0] - 0.0).abs() < 1e-9, "start x={}", p0[0]);
        assert!((p1[0] - 4.0).abs() < 1e-9, "end x={}", p1[0]);
    }

    #[test]
    fn clamped_normalized_param_range() {
        let cp = [[0.0_f64], [1.0], [2.0], [3.0]];
        let bs = BSpline::<f64, 1, 4>::clamped_normalized(cp, 2).unwrap();
        let (t0, t1) = bs.param_range();
        assert!((t0 - 0.0).abs() < 1e-10);
        assert!((t1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn velocity_nonnull_for_moving_spline() {
        let cp = [[0.0_f64, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]];
        let bs = BSpline::<f64, 2, 4>::clamped_normalized(cp, 2).unwrap();
        let v = bs.velocity(0.5);
        let speed = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!(speed > 0.01, "speed={}", speed);
    }

    #[test]
    fn invalid_knot_count_returns_none() {
        let cp = [[0.0_f64, 0.0], [1.0, 1.0]];
        // Wrong length knot vector for M=2, K=1 (needs 4 knots)
        let knots = [0.0_f64, 0.0, 1.0]; // only 3 knots
        assert!(BSpline::<f64, 2, 2>::new(cp, &knots, 1).is_none());
    }

    #[test]
    fn decreasing_knot_returns_none() {
        let cp = [[0.0_f64, 0.0], [1.0, 1.0]];
        let knots = [0.0_f64, 1.0, 0.5, 1.0]; // decreasing at position 2
        assert!(BSpline::<f64, 2, 2>::new(cp, &knots, 1).is_none());
    }

    #[test]
    fn too_few_control_points_for_degree() {
        // M=2, degree=3: M <= degree → invalid for clamped_uniform
        let cp = [[0.0_f64], [1.0]];
        assert!(BSpline::<f64, 1, 2>::clamped_uniform(cp, 3).is_none());
    }

    #[test]
    fn evaluate_clamped_beyond_range() {
        let cp = [[0.0_f64, 0.0], [1.0, 1.0], [2.0, 0.0]];
        let bs = BSpline::<f64, 2, 3>::clamped_normalized(cp, 2).unwrap();
        let p_before = bs.evaluate(-1.0);
        let p_after = bs.evaluate(2.0);
        // Should clamp to endpoints
        assert!(
            (p_before[0] - 0.0).abs() < 1e-9,
            "before[0]={}",
            p_before[0]
        );
        assert!((p_after[0] - 2.0).abs() < 1e-9, "after[0]={}", p_after[0]);
    }

    #[test]
    fn cubic_bspline_acceleration_finite() {
        let cp = [
            [0.0_f64, 0.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [3.0, 0.0],
            [4.0, 1.0],
        ];
        let bs = BSpline::<f64, 2, 5>::clamped_normalized(cp, 3).unwrap();
        let a = bs.acceleration(0.5);
        assert!(a[0].is_finite() && a[1].is_finite(), "accel={:?}", a);
    }

    #[test]
    fn linear_spline_midpoint_correct() {
        // 1D linear spline from 0 to 2: midpoint should be 1
        let cp = [[0.0_f64], [2.0]];
        let knots = [0.0_f64, 0.0, 1.0, 1.0];
        let bs = BSpline::<f64, 1, 2>::new(cp, &knots, 1).unwrap();
        let mid = bs.evaluate(0.5);
        assert!((mid[0] - 1.0).abs() < 1e-9, "mid={}", mid[0]);
    }
}
