pub mod induction;
pub mod parameters;
pub mod pmsm;
pub mod srm;

pub use induction::InductionMotor;
pub use parameters::{
    DcTestResult, InductionMotorParams, LockedRotorTestResult, NoLoadTestResult, PmsmParams,
};
pub use pmsm::PmsmModel;
pub use srm::{srm_6_4_default, CommutationAngles, SrmError, SrmModel, SrmPhaseParams};
