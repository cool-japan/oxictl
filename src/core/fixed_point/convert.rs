use fixed::types::{I16F16, I3F29, I8F24};

use super::types::{Q15_16, Q3_29, Q7_24};

/// Convert f32 to Q15_16, saturating on overflow.
pub fn fixed_from_f32_saturating(x: f32) -> Q15_16 {
    Q15_16(I16F16::from_num(x))
}

/// Convert Q15_16 to f32 (may lose precision).
pub fn fixed_to_f32(x: Q15_16) -> f32 {
    x.0.to_num::<f32>()
}

/// Convert i32 to Q15_16 (saturating if out of range).
pub fn q15_16_from_int(v: i32) -> Q15_16 {
    Q15_16(I16F16::from_num(v))
}

/// Convert f32 to Q3_29, saturating on overflow.
pub fn q3_29_from_f32(x: f32) -> Q3_29 {
    Q3_29(I3F29::from_num(x))
}

/// Convert Q3_29 to f32.
pub fn q3_29_to_f32(x: Q3_29) -> f32 {
    x.0.to_num::<f32>()
}

/// Convert f32 to Q7_24, saturating on overflow.
pub fn q7_24_from_f32(x: f32) -> Q7_24 {
    Q7_24(I8F24::from_num(x))
}

/// Convert Q7_24 to f32.
pub fn q7_24_to_f32(x: Q7_24) -> f32 {
    x.0.to_num::<f32>()
}
