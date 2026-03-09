// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright COOLJAPAN OU (Team Kitasan)
//
// Extremum Seeking Control (ESC) module.
//
// Provides model-free, real-time optimisation of unknown static maps via
// periodic perturbation and demodulation.
//
// # Algorithms
//
// | Type | Struct | Convergence |
// |------|--------|-------------|
// | Gradient ESC (SISO) | [`GradientEsc`] | O(1/k) |
// | Gradient ESC (2-input) | [`GradientEsc2D`] | O(1/k) |
// | Newton ESC (SISO) | [`NewtonEsc`] | curvature-independent |

#![cfg_attr(not(feature = "std"), no_std)]

pub mod gradient_esc;
pub mod newton_esc;

pub use gradient_esc::{ExtremumError, GradientEsc, GradientEsc2D};
pub use newton_esc::NewtonEsc;
