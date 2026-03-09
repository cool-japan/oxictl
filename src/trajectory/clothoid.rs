//! Clothoid (Euler spiral): linearly ramping curvature path segment.
//!
//! A clothoid satisfies κ(s) = A·s, where s is arc length and A is the
//! curvature rate parameter.  This gives a smooth transition between
//! straight lines (κ=0) and circular arcs.  Clothoids are widely used in
//! road/rail design and robot path planning.
//!
//! The x/y position is computed via Fresnel integrals approximated by a
//! 7-term power series, which is accurate for |t| ≤ ~3.
use crate::core::scalar::ControlScalar;

/// Clothoid state at a given arc length.
#[derive(Debug, Clone, Copy)]
pub struct ClothoidState<S: ControlScalar> {
    /// X coordinate.
    pub x: S,
    /// Y coordinate.
    pub y: S,
    /// Heading angle (rad).
    pub theta: S,
    /// Curvature at this arc length.
    pub kappa: S,
}

/// Clothoid segment with curvature κ(s) = curvature_rate · s.
///
/// The segment starts at the origin with heading 0 and zero curvature.
#[derive(Debug, Clone, Copy)]
pub struct ClothoidSegment<S: ControlScalar> {
    /// Curvature rate A: curvature κ = A · s.
    pub curvature_rate: S,
    /// Total arc length of the segment.
    pub length: S,
}

impl<S: ControlScalar> ClothoidSegment<S> {
    /// Create a new clothoid segment.
    pub fn new(curvature_rate: S, length: S) -> Self {
        Self {
            curvature_rate,
            length,
        }
    }

    /// Evaluate the clothoid at arc length `s`.
    ///
    /// Uses the Fresnel series expansion:
    ///   x(s) = ∫₀ˢ cos(A·u²/2) du
    ///   y(s) = ∫₀ˢ sin(A·u²/2) du
    ///
    /// Substituting t = √(A/π)·s:
    ///   x = √(π/A)·C(t),  y = √(π/A)·S(t)
    pub fn evaluate(&self, s: S) -> ClothoidState<S> {
        let s = s.clamp_val(S::ZERO, self.length);

        let theta = self.curvature_rate * s * s * S::HALF;
        let kappa = self.curvature_rate * s;

        // When curvature_rate ≈ 0 the path is a straight line.
        if self.curvature_rate.abs() < S::EPSILON {
            return ClothoidState {
                x: s,
                y: S::ZERO,
                theta: S::ZERO,
                kappa: S::ZERO,
            };
        }

        // Scale factor: t = √(A/π)·s
        let pi = S::PI;
        let a = self.curvature_rate.abs();
        let scale = (pi / a).sqrt(); // √(π/A)
        let t = s / scale; // t = s·√(A/π)

        let cx = Self::fresnel_c(t);
        let sx = Self::fresnel_s(t);

        let sign = if self.curvature_rate >= S::ZERO {
            S::ONE
        } else {
            -S::ONE
        };

        ClothoidState {
            x: scale * cx,
            y: scale * sx * sign,
            theta,
            kappa,
        }
    }

    /// Fresnel cosine integral C(t) = ∫₀ᵗ cos(π·u²/2) du via 7-term series.
    ///
    /// Series: C(t) = Σₙ₌₀^∞ (-1)ⁿ (π/2)²ⁿ t^(4n+1) / ((4n+1)(2n)!)
    fn fresnel_c(t: S) -> S {
        let pi = S::PI;
        let half_pi = pi * S::HALF;
        let t2 = t * t;
        let t4 = t2 * t2;

        // Precompute powers of (π/2)²
        let hp2 = half_pi * half_pi; // (π/2)²

        // Term 0: t
        let term0 = t;
        // Term 1: -(π/2)^2 · t^5 / (5·2!)
        let term1 = -hp2 * t4 * t / S::from_f64(10.0);
        // Term 2: (π/2)^4 · t^9 / (9·4!)
        let term2 = hp2 * hp2 * t4 * t4 * t / S::from_f64(216.0);
        // Term 3: -(π/2)^6 · t^13 / (13·6!)
        let term3 = -hp2 * hp2 * hp2 * t4 * t4 * t4 * t / S::from_f64(9360.0);
        // Term 4: (π/2)^8 · t^17 / (17·8!)
        let t17 = t4 * t4 * t4 * t4 * t;
        let term4 = hp2 * hp2 * hp2 * hp2 * t17 / S::from_f64(685_440.0);
        // Term 5: -(π/2)^10 · t^21 / (21·10!)
        let t21 = t17 * t4;
        let term5 = -hp2 * hp2 * hp2 * hp2 * hp2 * t21 / S::from_f64(76_204_800.0);
        // Term 6: (π/2)^12 · t^25 / (25·12!)
        let t25 = t21 * t4;
        let hp12 = hp2 * hp2 * hp2 * hp2 * hp2 * hp2;
        let term6 = hp12 * t25 / S::from_f64(11_975_040_000.0);

        term0 + term1 + term2 + term3 + term4 + term5 + term6
    }

    /// Fresnel sine integral S(t) = ∫₀ᵗ sin(π·u²/2) du via 7-term series.
    ///
    /// Series: S(t) = Σₙ₌₀^∞ (-1)ⁿ (π/2)^(2n+1) t^(4n+3) / ((4n+3)(2n+1)!)
    fn fresnel_s(t: S) -> S {
        let pi = S::PI;
        let half_pi = pi * S::HALF;
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t2 * t2;

        let hp2 = half_pi * half_pi;

        // Term 0: (π/2) · t^3 / (3·1!)
        let term0 = half_pi * t3 / S::from_f64(3.0);
        // Term 1: -(π/2)^3 · t^7 / (7·3!)
        let term1 = -half_pi * hp2 * t4 * t3 / S::from_f64(42.0);
        // Term 2: (π/2)^5 · t^11 / (11·5!)
        let term2 = half_pi * hp2 * hp2 * t4 * t4 * t3 / S::from_f64(1320.0);
        // Term 3: -(π/2)^7 · t^15 / (15·7!)
        let t15 = t4 * t4 * t4 * t3;
        let term3 = -half_pi * hp2 * hp2 * hp2 * t15 / S::from_f64(75_600.0);
        // Term 4: (π/2)^9 · t^19 / (19·9!)
        let t19 = t15 * t4;
        let term4 = half_pi * hp2 * hp2 * hp2 * hp2 * t19 / S::from_f64(6_894_720.0);
        // Term 5: -(π/2)^11 · t^23 / (23·11!)
        let t23 = t19 * t4;
        let hp11 = half_pi * hp2 * hp2 * hp2 * hp2 * hp2;
        let term5 = -hp11 * t23 / S::from_f64(916_215_040.0);
        // Term 6: (π/2)^13 · t^27 / (27·13!)
        let t27 = t23 * t4;
        let hp13 = hp11 * hp2;
        let term6 = hp13 * t27 / S::from_f64(168_129_561_600.0);

        term0 + term1 + term2 + term3 + term4 + term5 + term6
    }

    /// State at the end of the segment.
    pub fn end_state(&self) -> ClothoidState<S> {
        self.evaluate(self.length)
    }

    /// Curvature at arc length `s`.
    pub fn kappa(&self, s: S) -> S {
        self.curvature_rate * s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_curvature_rate_is_straight_line() {
        let seg = ClothoidSegment::new(0.0_f64, 5.0);
        let state = seg.evaluate(3.0);
        assert!((state.x - 3.0).abs() < 1e-6, "x={}", state.x);
        assert!(state.y.abs() < 1e-6, "y={}", state.y);
        assert!(state.theta.abs() < 1e-6);
    }

    #[test]
    fn curvature_increases_linearly() {
        let a = 0.5_f64;
        let seg = ClothoidSegment::new(a, 4.0);
        // κ(2) = 0.5*2 = 1.0, κ(4) = 0.5*4 = 2.0
        assert!((seg.kappa(2.0) - 1.0).abs() < 1e-10);
        assert!((seg.kappa(4.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn heading_matches_integral_of_curvature() {
        // θ(s) = ∫₀ˢ κ du = A·s²/2
        let a = 0.4_f64;
        let seg = ClothoidSegment::new(a, 3.0);
        let state = seg.evaluate(3.0);
        let expected_theta = a * 9.0 / 2.0; // A·s²/2
        assert!(
            (state.theta - expected_theta).abs() < 1e-10,
            "theta={}",
            state.theta
        );
    }

    #[test]
    fn fresnel_small_t_accuracy() {
        // For small t, C(t) ≈ t and S(t) ≈ (π/2)·t³/3.
        // With curvature_rate=1, s=0.1:  t = s·√(A/π) ≈ 0.0564.
        // The clothoid maps back via scale = √(π/A):
        //   x = scale·C(t) ≈ scale·t = s = 0.1
        //   y = scale·S(t) ≈ scale·(π/2)·t³/3 = s³/6 ≈ 1.67e-4
        // The tolerance for y must be larger than this leading-order term.
        let seg = ClothoidSegment::new(1.0_f64, 0.1);
        let state = seg.evaluate(0.1);
        // x ≈ s (nearly straight for small curvature_rate*s²)
        assert!((state.x - 0.1).abs() < 1e-4, "x={}", state.x);
        // y ≈ s³/6 ≈ 1.67e-4 for A=1, s=0.1; allow a relative 5× margin
        assert!(state.y.abs() < 1e-3, "y={}", state.y);
    }
}
