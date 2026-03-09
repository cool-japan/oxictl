//! Neural network based control algorithms.
//!
//! This module provides:
//! * Activation functions and their analytic derivatives (`activations`).
//! * A stack-allocated dense layer with Xavier initialisation (`layer`).
//! * A fixed-topology MLP (input→hidden→output) with mini-batch SGD (`network`).
//! * A Radial Basis Function network with fixed centres and learned output weights
//!   (`rbf_network`).
//! * A neural-network adaptive PID controller (`neural_pid`).
//!
//! All types are `no_std` compatible and use `libm` for transcendental functions.
//! No heap allocation is performed; all state lives in const-sized arrays.

pub mod activations;
pub mod layer;
pub mod network;
pub mod neural_pid;
pub mod rbf_network;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in neural network operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuralError {
    /// A size or dimension argument is invalid (e.g. zero sigma, empty network).
    InvalidDimension,
    /// The training procedure did not converge within the allowed iterations.
    NotConverged,
    /// A computation produced a non-finite value (infinity or NaN).
    NumericalOverflow,
}

impl core::fmt::Display for NeuralError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NeuralError::InvalidDimension => {
                write!(f, "NeuralError: invalid dimension or parameter")
            }
            NeuralError::NotConverged => write!(f, "NeuralError: training did not converge"),
            NeuralError::NumericalOverflow => {
                write!(f, "NeuralError: numerical overflow (non-finite value)")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use activations::{ActivationFn, LeakyRelu, Linear, Relu, Sigmoid, Swish, Tanh};
pub use layer::{DenseLayer, GradDense};
pub use network::{Mlp, MlpRegressor};
pub use neural_pid::{make_rbf_centers, NeuralPid, NeuralPidConfig};
pub use rbf_network::{gaussian_rbf, RbfCenter, RbfNetwork};
