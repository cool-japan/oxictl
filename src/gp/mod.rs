#![cfg_attr(not(feature = "std"), no_std)]

pub mod cholesky;
pub mod gp_regression;
pub mod kernel;
pub mod sparse_gp;

pub use cholesky::{backward_sub, cholesky, cholesky_solve, forward_sub};
pub use gp_regression::GpRegression;
pub use kernel::{AdditiveKernel, Kernel, LinearKernel, Matern52Kernel, RbfKernel};
pub use sparse_gp::SparseGp;

/// Error type for Gaussian Process operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpError {
    /// The kernel matrix is not positive definite (Cholesky failed).
    NotPositiveDefinite,
    /// GP has not been fitted yet.
    NotTrained,
    /// Input dimensions do not match.
    DimensionMismatch,
    /// A numerical error occurred (e.g. division by zero).
    NumericalError,
}

impl core::fmt::Display for GpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GpError::NotPositiveDefinite => {
                write!(f, "GpError: kernel matrix not positive definite")
            }
            GpError::NotTrained => write!(f, "GpError: GP not trained"),
            GpError::DimensionMismatch => write!(f, "GpError: dimension mismatch"),
            GpError::NumericalError => write!(f, "GpError: numerical error"),
        }
    }
}
