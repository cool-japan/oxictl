//! Adaptive digital filters for system identification and noise cancellation.
//!
//! This module provides a suite of adaptive FIR filters that update their
//! weights online to minimize a cost function without prior knowledge of the
//! unknown system:
//!
//! | Filter | Algorithm | Convergence | Complexity |
//! |--------|-----------|-------------|------------|
//! | [`LmsFilter`] | LMS | Slow, simple | O(N) |
//! | [`NlmsFilter`] | NLMS | Faster, robust to power variation | O(N) |
//! | [`VssLmsFilter`] | VSS-LMS | Adaptive step, lower misadjustment | O(N) |
//! | [`RlsFilter`] | RLS | Very fast (O(N²)) | O(N²) |
//! | [`ApaFilter`] | APA | Tunable via projection order P | O(P²N) |
//!
//! All filters are:
//! - Generic over a [`ControlScalar`] floating-point type (`f32` or `f64`)
//! - `no_std` compatible (no heap allocation, fixed-size const generics)
//! - Free of `unwrap()` — all fallible operations return `Result`
//!
//! [`ControlScalar`]: crate::core::scalar::ControlScalar

pub mod affine_projection;
pub mod lms;
pub mod rls;

pub use affine_projection::ApaFilter;
pub use lms::{AdaptiveFilterError, LmsFilter, NlmsFilter, VssLmsFilter};
pub use rls::RlsFilter;
