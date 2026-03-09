//! Parameter identification submodule for motor parameter estimation.
//!
//! Provides online estimation algorithms for both PMSM and induction motors,
//! enabling adaptive control without offline commissioning procedures.

pub mod induction_id;
pub mod pmsm_id;

pub use induction_id::{InductionIdConfig, InductionParamId, InductionParamIdResult};
pub use pmsm_id::{PmsmIdConfig, PmsmParamId, PmsmParamIdResult, RlsAxis};
