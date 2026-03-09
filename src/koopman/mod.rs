//! Koopman operator theory for data-driven nonlinear control.
//!
//! This module provides tools for approximating the Koopman operator of a
//! nonlinear dynamical system and exploiting the resulting linear representation
//! for prediction and control.
//!
//! # Components
//!
//! - [`lifting_functions`] — Observable maps ψ: ℝᴺ → ℝᴸ (polynomial, RBF, delay embedding).
//! - [`edmd`] — Extended Dynamic Mode Decomposition (EDMD) to fit K from data.
//! - [`koopman_mpc`] — Single-step greedy Koopman MPC in the lifted space.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod edmd;
pub mod koopman_mpc;
pub mod lifting_functions;

pub use edmd::Edmd;
pub use koopman_mpc::KoopmanGreedyMpc;
pub use lifting_functions::{
    DelayEmbedding, KoopmanError, LiftingMap, PolynomialLifting, RbfLifting,
};
