use core::fmt;
use num_traits::Float;

/// Trait abstracting numeric types used in control computations.
/// Supports f32 and f64, enabling compile-time selection of precision.
pub trait ControlScalar:
    Float
    + Copy
    + Default
    + fmt::Debug
    + fmt::Display
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::MulAssign
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const HALF: Self;
    const PI: Self;
    const EPSILON: Self;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;

    fn clamp_val(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    fn saturate(self, limit: Self) -> Self {
        self.clamp_val(-limit, limit)
    }
}

impl ControlScalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f32::consts::PI;
    const EPSILON: Self = f32::EPSILON;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ControlScalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f64::consts::PI;
    const EPSILON: Self = f64::EPSILON;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scalar_basics<S: ControlScalar>() {
        assert_eq!(S::ZERO + S::ONE, S::ONE);
        assert_eq!(S::ONE + S::ONE, S::TWO);
        assert_eq!(S::HALF + S::HALF, S::ONE);
    }

    fn test_clamp<S: ControlScalar>() {
        let val = S::from_f64(5.0);
        let clamped = val.clamp_val(S::from_f64(0.0), S::from_f64(3.0));
        assert_eq!(clamped, S::from_f64(3.0));

        let val = S::from_f64(-2.0);
        let clamped = val.clamp_val(S::from_f64(0.0), S::from_f64(3.0));
        assert_eq!(clamped, S::ZERO);

        let val = S::from_f64(1.5);
        let clamped = val.clamp_val(S::from_f64(0.0), S::from_f64(3.0));
        assert_eq!(clamped, S::from_f64(1.5));
    }

    fn test_saturate<S: ControlScalar>() {
        let val = S::from_f64(10.0);
        assert_eq!(val.saturate(S::from_f64(5.0)), S::from_f64(5.0));

        let val = S::from_f64(-10.0);
        assert_eq!(val.saturate(S::from_f64(5.0)), S::from_f64(-5.0));

        let val = S::from_f64(3.0);
        assert_eq!(val.saturate(S::from_f64(5.0)), S::from_f64(3.0));
    }

    fn test_from_to_f64<S: ControlScalar>() {
        let val = S::from_f64(core::f64::consts::PI);
        let back = val.to_f64();
        assert!((back - core::f64::consts::PI).abs() < 0.01);
    }

    fn test_trig<S: ControlScalar>() {
        let zero = S::ZERO;
        assert!((zero.sin() - S::ZERO).abs() < S::from_f64(1e-6));
        assert!((zero.cos() - S::ONE).abs() < S::from_f64(1e-6));
    }

    #[test]
    fn f32_basics() {
        test_scalar_basics::<f32>();
    }

    #[test]
    fn f64_basics() {
        test_scalar_basics::<f64>();
    }

    #[test]
    fn f32_clamp() {
        test_clamp::<f32>();
    }

    #[test]
    fn f64_clamp() {
        test_clamp::<f64>();
    }

    #[test]
    fn f32_saturate() {
        test_saturate::<f32>();
    }

    #[test]
    fn f64_saturate() {
        test_saturate::<f64>();
    }

    #[test]
    fn f32_from_to_f64() {
        test_from_to_f64::<f32>();
    }

    #[test]
    fn f64_from_to_f64() {
        test_from_to_f64::<f64>();
    }

    #[test]
    fn f32_trig() {
        test_trig::<f32>();
    }

    #[test]
    fn f64_trig() {
        test_trig::<f64>();
    }
}
