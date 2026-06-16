use fixed::types::{I16F16, I3F29, I8F24};

/// Q15.16 signed fixed-point: 16-bit integer (including sign) + 16 fractional bits.
/// Range: [-32768, 32768). Resolution: ~0.0000153.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Q15_16(pub I16F16);

/// Q3.29 signed fixed-point: 3-bit integer (including sign) + 29 fractional bits.
/// Range: [-4, 4). Very high fractional resolution (~1.86e-9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Q3_29(pub I3F29);

/// Q7.24 signed fixed-point: 8-bit integer (including sign) + 24 fractional bits.
/// Range: [-128, 128). Resolution: ~5.96e-8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Q7_24(pub I8F24);
