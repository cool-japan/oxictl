//! Membership functions for fuzzy logic systems.
//!
//! All membership functions map a crisp input value `x` to a degree of
//! membership in `[0, 1]`. Construction is validated; invalid parameters
//! return `Err(FuzzyError)`.

use crate::core::scalar::ControlScalar;
use crate::fuzzy::FuzzyError;

// ────────────────────────────────────────────────────────────────────────────
// Trait
// ────────────────────────────────────────────────────────────────────────────

/// A membership function that maps a crisp scalar `x` to `[0, 1]`.
pub trait MembershipFn<S: ControlScalar> {
    fn membership(&self, x: S) -> S;
}

// ────────────────────────────────────────────────────────────────────────────
// Triangular
// ────────────────────────────────────────────────────────────────────────────

/// Triangular membership function defined by three breakpoints.
///
/// `left ≤ center ≤ right`, with `left < right`.
///
/// ```text
///       1 |      /\
///         |     /  \
///       0 |____/    \____
///         left center right
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Triangular<S: ControlScalar> {
    left: S,
    center: S,
    right: S,
}

impl<S: ControlScalar> Triangular<S> {
    /// Construct a triangular MF.
    ///
    /// Requires `left < right` and `left ≤ center ≤ right`.
    pub fn new(left: S, center: S, right: S) -> Result<Self, FuzzyError> {
        if left >= right {
            return Err(FuzzyError::InvalidParameter(
                "Triangular: left must be strictly less than right",
            ));
        }
        if center < left || center > right {
            return Err(FuzzyError::InvalidParameter(
                "Triangular: center must lie in [left, right]",
            ));
        }
        Ok(Self {
            left,
            center,
            right,
        })
    }
}

impl<S: ControlScalar> MembershipFn<S> for Triangular<S> {
    fn membership(&self, x: S) -> S {
        if x <= self.left || x >= self.right {
            return S::ZERO;
        }
        if x <= self.center {
            let denom = self.center - self.left;
            if denom <= S::ZERO {
                return S::ONE;
            }
            (x - self.left) / denom
        } else {
            let denom = self.right - self.center;
            if denom <= S::ZERO {
                return S::ONE;
            }
            (self.right - x) / denom
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Trapezoidal
// ────────────────────────────────────────────────────────────────────────────

/// Trapezoidal membership function defined by four breakpoints.
///
/// `left ≤ left_top ≤ right_top ≤ right`, with `left < right`.
///
/// ```text
///       1 |      ________
///         |     /        \
///       0 |____/          \____
///        left  lt         rt  right
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Trapezoidal<S: ControlScalar> {
    left: S,
    left_top: S,
    right_top: S,
    right: S,
}

impl<S: ControlScalar> Trapezoidal<S> {
    /// Construct a trapezoidal MF.
    ///
    /// Requires `left ≤ left_top ≤ right_top ≤ right` and `left < right`.
    pub fn new(left: S, left_top: S, right_top: S, right: S) -> Result<Self, FuzzyError> {
        if left > left_top || left_top > right_top || right_top > right {
            return Err(FuzzyError::InvalidParameter(
                "Trapezoidal: must satisfy left ≤ left_top ≤ right_top ≤ right",
            ));
        }
        if left >= right {
            return Err(FuzzyError::InvalidParameter(
                "Trapezoidal: left must be strictly less than right",
            ));
        }
        Ok(Self {
            left,
            left_top,
            right_top,
            right,
        })
    }
}

impl<S: ControlScalar> MembershipFn<S> for Trapezoidal<S> {
    fn membership(&self, x: S) -> S {
        if x <= self.left || x >= self.right {
            return S::ZERO;
        }
        if x >= self.left_top && x <= self.right_top {
            return S::ONE;
        }
        if x < self.left_top {
            let denom = self.left_top - self.left;
            if denom <= S::ZERO {
                return S::ONE;
            }
            return (x - self.left) / denom;
        }
        // x > self.right_top
        let denom = self.right - self.right_top;
        if denom <= S::ZERO {
            return S::ONE;
        }
        (self.right - x) / denom
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Gaussian
// ────────────────────────────────────────────────────────────────────────────

/// Gaussian (bell-curve) membership function: `exp(-(x-center)²/(2σ²))`.
#[derive(Debug, Clone, Copy)]
pub struct Gaussian<S: ControlScalar> {
    center: S,
    sigma: S,
}

impl<S: ControlScalar> Gaussian<S> {
    /// Construct a Gaussian MF. Requires `sigma > 0`.
    pub fn new(center: S, sigma: S) -> Result<Self, FuzzyError> {
        if sigma <= S::ZERO {
            return Err(FuzzyError::InvalidParameter(
                "Gaussian: sigma must be positive",
            ));
        }
        Ok(Self { center, sigma })
    }
}

impl<S: ControlScalar> MembershipFn<S> for Gaussian<S> {
    fn membership(&self, x: S) -> S {
        let two = S::TWO;
        let diff = x - self.center;
        let exponent = -(diff * diff) / (two * self.sigma * self.sigma);
        exponent.exp()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Sigmoid
// ────────────────────────────────────────────────────────────────────────────

/// Sigmoid membership function: `1 / (1 + exp(-a * (x - c)))`.
///
/// `a > 0` → rising; `a < 0` → falling. `a = 0` is invalid.
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid<S: ControlScalar> {
    a: S,
    c: S,
}

impl<S: ControlScalar> Sigmoid<S> {
    /// Construct a sigmoid MF. Requires `a ≠ 0`.
    pub fn new(a: S, c: S) -> Result<Self, FuzzyError> {
        if a == S::ZERO {
            return Err(FuzzyError::InvalidParameter(
                "Sigmoid: parameter a must not be zero",
            ));
        }
        Ok(Self { a, c })
    }
}

impl<S: ControlScalar> MembershipFn<S> for Sigmoid<S> {
    fn membership(&self, x: S) -> S {
        let neg_a_xc = -(self.a * (x - self.c));
        S::ONE / (S::ONE + neg_a_xc.exp())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Singleton
// ────────────────────────────────────────────────────────────────────────────

/// Singleton membership function (Kronecker delta).
///
/// Returns `1` when `x == value` (within machine epsilon), `0` otherwise.
/// Primarily used for Sugeno crisp consequents.
#[derive(Debug, Clone, Copy)]
pub struct Singleton<S: ControlScalar> {
    value: S,
}

impl<S: ControlScalar> Singleton<S> {
    /// Construct a singleton MF.
    pub fn new(value: S) -> Self {
        Self { value }
    }
}

impl<S: ControlScalar> MembershipFn<S> for Singleton<S> {
    fn membership(&self, x: S) -> S {
        if (x - self.value).abs() <= S::EPSILON {
            S::ONE
        } else {
            S::ZERO
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BellShaped (generalized bell)
// ────────────────────────────────────────────────────────────────────────────

/// Generalized bell-shaped membership function.
///
/// `gbellmf(x) = 1 / (1 + |(x - c) / a|^{2b})`
///
/// - `a > 0`: half-width at the crossover point.
/// - `b > 0`: controls slope steepness (integer preferred).
/// - `c`: center.
#[derive(Debug, Clone, Copy)]
pub struct BellShaped<S: ControlScalar> {
    a: S,
    b: S,
    c: S,
}

impl<S: ControlScalar> BellShaped<S> {
    /// Construct a generalized bell MF. Requires `a > 0` and `b > 0`.
    pub fn new(a: S, b: S, c: S) -> Result<Self, FuzzyError> {
        if a <= S::ZERO {
            return Err(FuzzyError::InvalidParameter(
                "BellShaped: parameter a must be positive",
            ));
        }
        if b <= S::ZERO {
            return Err(FuzzyError::InvalidParameter(
                "BellShaped: parameter b must be positive",
            ));
        }
        Ok(Self { a, b, c })
    }
}

impl<S: ControlScalar> MembershipFn<S> for BellShaped<S> {
    fn membership(&self, x: S) -> S {
        let ratio = (x - self.c) / self.a;
        let ratio_abs = ratio.abs();
        // pow with real exponent: |ratio|^{2b} = exp(2b * ln(|ratio|))
        let exponent = self.b + self.b; // 2b
        let powered = if ratio_abs == S::ZERO {
            S::ZERO
        } else {
            (ratio_abs.ln() * exponent).exp()
        };
        S::ONE / (S::ONE + powered)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // ── Triangular ───────────────────────────────────────────────────────────

    #[test]
    fn triangular_center_is_one() {
        let mf = Triangular::new(0.0_f64, 5.0, 10.0).unwrap();
        assert!((mf.membership(5.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn triangular_outside_is_zero() {
        let mf = Triangular::new(0.0_f64, 5.0, 10.0).unwrap();
        assert_eq!(mf.membership(-1.0), 0.0);
        assert_eq!(mf.membership(10.0), 0.0);
        assert_eq!(mf.membership(11.0), 0.0);
    }

    #[test]
    fn triangular_midpoints() {
        let mf = Triangular::new(0.0_f64, 4.0, 8.0).unwrap();
        assert!((mf.membership(2.0) - 0.5).abs() < EPS);
        assert!((mf.membership(6.0) - 0.5).abs() < EPS);
    }

    #[test]
    fn triangular_invalid_construction() {
        assert!(Triangular::new(5.0_f64, 3.0, 10.0).is_err()); // center < left
        assert!(Triangular::new(5.0_f64, 5.0, 5.0).is_err()); // left == right
        assert!(Triangular::new(10.0_f64, 5.0, 3.0).is_err()); // left >= right
    }

    // ── Trapezoidal ──────────────────────────────────────────────────────────

    #[test]
    fn trapezoidal_flat_top_is_one() {
        let mf = Trapezoidal::new(0.0_f64, 3.0, 7.0, 10.0).unwrap();
        assert!((mf.membership(3.0) - 1.0).abs() < EPS);
        assert!((mf.membership(5.0) - 1.0).abs() < EPS);
        assert!((mf.membership(7.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn trapezoidal_outside_is_zero() {
        let mf = Trapezoidal::new(0.0_f64, 3.0, 7.0, 10.0).unwrap();
        assert_eq!(mf.membership(-0.1), 0.0);
        assert_eq!(mf.membership(10.0), 0.0);
    }

    #[test]
    fn trapezoidal_slopes() {
        let mf = Trapezoidal::new(0.0_f64, 4.0, 6.0, 10.0).unwrap();
        // Rising slope at x=2: (2-0)/(4-0) = 0.5
        assert!((mf.membership(2.0) - 0.5).abs() < EPS);
        // Falling slope at x=8: (10-8)/(10-6) = 0.5
        assert!((mf.membership(8.0) - 0.5).abs() < EPS);
    }

    #[test]
    fn trapezoidal_invalid_construction() {
        assert!(Trapezoidal::new(5.0_f64, 3.0, 7.0, 10.0).is_err());
    }

    // ── Gaussian ─────────────────────────────────────────────────────────────

    #[test]
    fn gaussian_center_is_one() {
        let mf = Gaussian::new(0.0_f64, 1.0).unwrap();
        assert!((mf.membership(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn gaussian_tails_near_zero() {
        let mf = Gaussian::new(0.0_f64, 1.0).unwrap();
        assert!(mf.membership(10.0) < 1e-20);
        assert!(mf.membership(-10.0) < 1e-20);
    }

    #[test]
    fn gaussian_symmetry() {
        let mf = Gaussian::new(5.0_f64, 2.0).unwrap();
        let left = mf.membership(3.0);
        let right = mf.membership(7.0);
        assert!((left - right).abs() < EPS);
    }

    #[test]
    fn gaussian_invalid_sigma() {
        assert!(Gaussian::new(0.0_f64, 0.0).is_err());
        assert!(Gaussian::new(0.0_f64, -1.0).is_err());
    }

    // ── Sigmoid ──────────────────────────────────────────────────────────────

    #[test]
    fn sigmoid_center_is_half() {
        let mf = Sigmoid::new(1.0_f64, 0.0).unwrap();
        assert!((mf.membership(0.0) - 0.5).abs() < EPS);
    }

    #[test]
    fn sigmoid_rises_left_to_right() {
        let mf = Sigmoid::new(2.0_f64, 0.0).unwrap();
        assert!(mf.membership(-5.0) < 0.1);
        assert!(mf.membership(5.0) > 0.9);
    }

    #[test]
    fn sigmoid_invalid_a() {
        assert!(Sigmoid::new(0.0_f64, 0.0).is_err());
    }

    // ── Singleton ────────────────────────────────────────────────────────────

    #[test]
    fn singleton_at_value_is_one() {
        let mf = Singleton::new(3.0_f64);
        assert_eq!(mf.membership(3.0), 1.0);
    }

    #[test]
    fn singleton_elsewhere_is_zero() {
        let mf = Singleton::new(3.0_f64);
        assert_eq!(mf.membership(3.001), 0.0);
        assert_eq!(mf.membership(0.0), 0.0);
    }

    // ── BellShaped ───────────────────────────────────────────────────────────

    #[test]
    fn bell_center_is_one() {
        let mf = BellShaped::new(2.0_f64, 4.0, 0.0).unwrap();
        assert!((mf.membership(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn bell_crossover_at_a() {
        // At x = c ± a, the value should be exactly 0.5
        let a = 2.0_f64;
        let b = 1.0_f64; // 2b = 2, so |(x-c)/a|^2 = 1 → result = 1/(1+1) = 0.5
        let mf = BellShaped::new(a, b, 0.0).unwrap();
        assert!((mf.membership(a) - 0.5).abs() < EPS);
        assert!((mf.membership(-a) - 0.5).abs() < EPS);
    }

    #[test]
    fn bell_invalid_params() {
        assert!(BellShaped::new(0.0_f64, 2.0, 0.0).is_err());
        assert!(BellShaped::new(2.0_f64, 0.0, 0.0).is_err());
        assert!(BellShaped::new(-1.0_f64, 2.0, 0.0).is_err());
    }
}
