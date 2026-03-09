pub use crate::core::scalar::ControlScalar;
pub use crate::core::signal::{ControlOutput, Feedback, Setpoint};
pub use crate::core::traits::{Controller, Plant};

#[cfg(feature = "pid")]
pub use crate::pid::{AntiWindupMethod, CascadePid, IncrementalPid, Pid, PidConfig};

#[cfg(feature = "safety")]
pub use crate::safety::{FaultHandler, FaultResponse, FaultSeverity, SafetyMonitor, Watchdog};

#[cfg(feature = "estimator")]
pub use crate::estimator::{ComplementaryFilter, Ekf, KalmanFilter};

#[cfg(feature = "state_feedback")]
pub use crate::state_feedback::Lqr;

#[cfg(feature = "motor")]
pub use crate::motor::{
    clarke, park, svpwm, AlphaBeta, Dq, FocController, IncrementalEncoder, SixStepCommutator,
    SvpwmDuty,
};

#[cfg(feature = "scheduler")]
pub use crate::scheduler::{FixedRateTask, MultiRateScheduler};

#[cfg(feature = "sim")]
pub use crate::sim::{Scope, ThermalPlant};
