use fixed::types::{I16F16, I3F29, I8F24};

use super::types::{Q15_16, Q3_29, Q7_24};

macro_rules! impl_ops {
    ($T:ty, $Inner:ty) => {
        impl core::ops::Add for $T {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self(self.0.saturating_add(rhs.0))
            }
        }

        impl core::ops::Sub for $T {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0.saturating_sub(rhs.0))
            }
        }

        impl core::ops::Mul for $T {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self(self.0.saturating_mul(rhs.0))
            }
        }

        impl core::ops::Div for $T {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                // checked_div returns None on /0 or overflow; saturate using
                // the numerator's sign so that positive/zero → MAX and
                // negative/zero → MIN (matches mathematical intuition).
                Self(self.0.checked_div(rhs.0).unwrap_or_else(|| {
                    if self.0 < <$Inner>::ZERO {
                        <$Inner>::MIN
                    } else {
                        <$Inner>::MAX
                    }
                }))
            }
        }

        impl core::ops::Neg for $T {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self(self.0.saturating_neg())
            }
        }
    };
}

impl_ops!(Q15_16, I16F16);
impl_ops!(Q3_29, I3F29);
impl_ops!(Q7_24, I8F24);
