use crate::core::scalar::PidScalar;
use fixed::types::{I16F16, I3F29, I8F24};

use super::types::{Q15_16, Q3_29, Q7_24};

impl PidScalar for Q15_16 {
    const ZERO: Self = Q15_16(I16F16::ZERO);
    const ONE: Self = Q15_16(I16F16::ONE);
    /// Smallest representable positive value = 2^(-16) for I16F16.
    const EPSILON: Self = Q15_16(I16F16::DELTA);

    #[inline]
    fn from_int(v: i32) -> Self {
        Q15_16(I16F16::from_num(v))
    }

    #[inline]
    fn abs(self) -> Self {
        Q15_16(self.0.abs())
    }
}

impl PidScalar for Q3_29 {
    const ZERO: Self = Q3_29(I3F29::ZERO);
    const ONE: Self = Q3_29(I3F29::ONE);
    /// Smallest representable positive value = 2^(-29) for I3F29.
    const EPSILON: Self = Q3_29(I3F29::DELTA);

    #[inline]
    fn from_int(v: i32) -> Self {
        Q3_29(I3F29::from_num(v))
    }

    #[inline]
    fn abs(self) -> Self {
        Q3_29(self.0.abs())
    }
}

impl PidScalar for Q7_24 {
    const ZERO: Self = Q7_24(I8F24::ZERO);
    const ONE: Self = Q7_24(I8F24::ONE);
    /// Smallest representable positive value = 2^(-24) for I8F24.
    const EPSILON: Self = Q7_24(I8F24::DELTA);

    #[inline]
    fn from_int(v: i32) -> Self {
        Q7_24(I8F24::from_num(v))
    }

    #[inline]
    fn abs(self) -> Self {
        Q7_24(self.0.abs())
    }
}
