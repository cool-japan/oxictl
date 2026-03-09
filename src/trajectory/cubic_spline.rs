use crate::core::scalar::ControlScalar;

/// Natural cubic spline through N waypoints.
///
/// Given N time/value pairs (t_i, y_i), fits a piecewise cubic polynomial
/// satisfying:
///   - Interpolation: s(t_i) = y_i
///   - C² continuity at interior knots
///   - Natural BC: s''(t_0) = s''(t_{N-1}) = 0
///
/// For each segment i ∈ [0, N-2]:
///   s_i(t) = y[i] + b[i]*h + c[i]*h² + d[i]*h³    where h = t - t[i]
///
/// N must be ≥ 2.
#[derive(Debug, Clone, Copy)]
pub struct CubicSpline<S: ControlScalar, const N: usize> {
    t: [S; N],
    y: [S; N],
    /// Coefficient of h¹ (first derivative at left knot of each segment).
    b: [S; N],
    /// Coefficient of h².
    c: [S; N],
    /// Coefficient of h³.
    d: [S; N],
}

impl<S: ControlScalar, const N: usize> CubicSpline<S, N> {
    /// Fit a natural cubic spline to the given data points.
    ///
    /// `t` must be strictly increasing (t[0] < t[1] < ... < t[N-1]).
    /// Returns `None` if N < 2 or if any interval width is non-positive.
    pub fn natural(t: [S; N], y: [S; N]) -> Option<Self> {
        if N < 2 {
            return None;
        }

        // h[i] = t[i+1] - t[i]
        let mut h = [S::ZERO; N];
        for i in 0..(N - 1) {
            h[i] = t[i + 1] - t[i];
            if h[i] <= S::ZERO {
                return None;
            }
        }

        // Second derivatives m[0..N]: natural BC → m[0]=m[N-1]=0
        // Interior: solve tridiagonal system for m[1..N-2]
        let mut m = [S::ZERO; N];

        if N > 2 {
            // Build tridiagonal system (size N-2 interior points)
            // Lower diagonal: l[i] = h[i]
            // Main diagonal:  diag[i] = 2*(h[i]+h[i+1])
            // Upper diagonal: u[i] = h[i+1]
            // RHS:            r[i] = 6*((y[i+2]-y[i+1])/h[i+1] - (y[i+1]-y[i])/h[i])
            // For i = 1..N-2 (0-indexed interior points)

            // Thomas algorithm: forward sweep then back substitution
            let n_int = N - 2; // number of interior knots
            let mut diag = [S::ZERO; N];
            let mut rhs = [S::ZERO; N];
            let mut lower = [S::ZERO; N];
            let mut upper = [S::ZERO; N];

            for i in 1..=(n_int) {
                let idx = i - 1; // 0-based interior index
                lower[idx] = h[i - 1];
                diag[idx] = S::TWO * (h[i - 1] + h[i]);
                upper[idx] = h[i];
                rhs[idx] =
                    S::from_f64(6.0) * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
            }

            // Forward elimination (Thomas)
            let mut c_prime = [S::ZERO; N];
            let mut d_prime = [S::ZERO; N];

            if diag[0].abs() <= S::EPSILON {
                return None;
            }
            c_prime[0] = upper[0] / diag[0];
            d_prime[0] = rhs[0] / diag[0];

            for i in 1..n_int {
                let denom = diag[i] - lower[i] * c_prime[i - 1];
                if denom.abs() <= S::EPSILON {
                    return None;
                }
                c_prime[i] = upper[i] / denom;
                d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom;
            }

            // Back substitution
            if n_int > 0 {
                m[n_int] = d_prime[n_int - 1];
                for i in (0..n_int.saturating_sub(1)).rev() {
                    m[i + 1] = d_prime[i] - c_prime[i] * m[i + 2];
                }
            }
        }

        // Compute b, c, d coefficients for each segment
        let mut b = [S::ZERO; N];
        let mut c = [S::ZERO; N];
        let mut d_coeff = [S::ZERO; N];

        for i in 0..(N - 1) {
            let hi = h[i];
            c[i] = m[i] * S::HALF;
            d_coeff[i] = (m[i + 1] - m[i]) / (S::from_f64(6.0) * hi);
            b[i] = (y[i + 1] - y[i]) / hi - hi * (S::TWO * m[i] + m[i + 1]) / S::from_f64(6.0);
        }

        Some(Self {
            t,
            y,
            b,
            c,
            d: d_coeff,
        })
    }

    /// Find the segment index for time `t_query`.
    fn segment(&self, t_query: S) -> usize {
        if t_query <= self.t[0] {
            return 0;
        }
        if t_query >= self.t[N - 1] {
            return N - 2;
        }
        // Binary search
        let mut lo = 0usize;
        let mut hi = N - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if t_query >= self.t[mid] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Evaluate the spline at `t_query`.
    ///
    /// Clamps to endpoint values for queries outside [t[0], t[N-1]].
    pub fn evaluate(&self, t_query: S) -> S {
        if t_query <= self.t[0] {
            return self.y[0];
        }
        if t_query >= self.t[N - 1] {
            return self.y[N - 1];
        }
        let i = self.segment(t_query);
        let h = t_query - self.t[i];
        self.y[i] + h * (self.b[i] + h * (self.c[i] + h * self.d[i]))
    }

    /// Evaluate the first derivative at `t_query`.
    pub fn velocity(&self, t_query: S) -> S {
        let i = self.segment(t_query);
        let h = t_query - self.t[i];
        self.b[i] + h * (S::TWO * self.c[i] + S::from_f64(3.0) * h * self.d[i])
    }

    /// Evaluate the second derivative at `t_query`.
    pub fn acceleration(&self, t_query: S) -> S {
        let i = self.segment(t_query);
        let h = t_query - self.t[i];
        S::TWO * self.c[i] + S::from_f64(6.0) * h * self.d[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_knots() {
        let t = [0.0_f64, 1.0, 2.0, 3.0];
        let y = [0.0_f64, 1.0, 0.0, 1.0];
        let s = CubicSpline::natural(t, y).unwrap();
        for i in 0..4 {
            assert!((s.evaluate(t[i]) - y[i]).abs() < 1e-10, "knot {}", i);
        }
    }

    #[test]
    fn straight_line_exact() {
        // Spline through collinear points should be exactly linear
        let t = [0.0_f64, 1.0, 2.0];
        let y = [0.0_f64, 1.0, 2.0];
        let s = CubicSpline::natural(t, y).unwrap();
        for step in 0..=20 {
            let ti = step as f64 * 0.1;
            assert!((s.evaluate(ti) - ti).abs() < 1e-10, "t={}", ti);
        }
    }

    #[test]
    fn clamps_at_endpoints() {
        let t = [0.0_f64, 1.0, 2.0];
        let y = [0.0_f64, 1.0, 0.0];
        let s = CubicSpline::natural(t, y).unwrap();
        assert!((s.evaluate(-1.0) - 0.0).abs() < 1e-10);
        assert!((s.evaluate(3.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn velocity_at_midpoint() {
        let t = [0.0_f64, 1.0, 2.0];
        let y = [0.0_f64, 1.0, 2.0]; // linear
        let s = CubicSpline::natural(t, y).unwrap();
        // Linear spline should have velocity 1.0 everywhere
        assert!((s.velocity(0.5) - 1.0).abs() < 1e-8);
        assert!((s.velocity(1.5) - 1.0).abs() < 1e-8);
    }

    #[test]
    fn natural_bc_zero_second_derivative() {
        let t = [0.0_f64, 1.0, 2.0, 3.0];
        let y = [1.0_f64, 2.0, 1.5, 3.0];
        let s = CubicSpline::natural(t, y).unwrap();
        // Natural boundary conditions: second derivative = 0 at endpoints
        assert!(
            s.acceleration(0.0).abs() < 1e-8,
            "acc at t=0: {}",
            s.acceleration(0.0)
        );
        assert!(
            s.acceleration(3.0).abs() < 1e-8,
            "acc at t=3: {}",
            s.acceleration(3.0)
        );
    }

    #[test]
    fn two_point_spline() {
        let t = [0.0_f64, 1.0];
        let y = [0.0_f64, 1.0];
        let s = CubicSpline::natural(t, y).unwrap();
        assert!((s.evaluate(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn non_increasing_times_returns_none() {
        let t = [0.0_f64, 1.0, 1.0]; // duplicate t
        let y = [0.0_f64, 1.0, 2.0];
        assert!(CubicSpline::natural(t, y).is_none());
    }
}
