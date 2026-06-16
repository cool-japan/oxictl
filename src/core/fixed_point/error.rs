/// Errors that can occur in fixed-point arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedError {
    /// Division by zero attempted.
    DivByZero,
    /// Result overflowed the representable range.
    Overflow,
}

impl core::fmt::Display for FixedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FixedError::DivByZero => f.write_str("fixed-point division by zero"),
            FixedError::Overflow => f.write_str("fixed-point arithmetic overflow"),
        }
    }
}
