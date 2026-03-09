pub mod byzantine_voter;
pub mod diagnostic;
pub mod diagnostics;
pub mod fault;
pub mod fmea;
pub mod handler;
pub mod heartbeat;
pub mod monitor;
pub mod redundancy;
pub mod redundancy_ext;
pub mod reliability;
pub mod safe_state_ext;
pub mod sil;
pub mod watchdog;

pub use byzantine_voter::ByzantineVoter;
pub use diagnostic::{DiagnosticMonitor, FaultTree, ReliabilityBlock};
pub use diagnostics::{
    DiagnosticCoverage, RedundancyCoverage, SafeStateLevel, SafeStateMachine,
    SafetyFunctionCoverage, SilLevel,
};
pub use fault::{FaultDef, FaultEvent, FaultResponse, FaultSeverity};
pub use fmea::{FmeaEntry, FmeaTable};
pub use handler::FaultHandler;
pub use heartbeat::Heartbeat;
pub use monitor::{
    PlausibilityMonitor, RangeMonitor, RateMonitor, StuckMonitor, TimeoutMonitor,
    TripleSensorPlausibility,
};
pub use redundancy::{DualChannelComparator, Voter, VoterStrategy};
pub use redundancy_ext::{
    ComparatorConfig, CompareResult, DualChannelComparatorExt, TripleModularRedundancy, VoteResult,
};
pub use reliability::ReliabilityModel;
pub use reliability::SilLevel as ReliabilitySilLevel;
pub use safe_state_ext::{
    MachineState, SafeError, SafeStateAction, SafeStateConfig, SafeStateMachineExt,
};
pub use sil::SilLevel as Sil508Level;
pub use sil::{PfdRange, PfhRange, SafetyRequirement, SilError};
pub use watchdog::Watchdog;

use crate::core::scalar::ControlScalar;

/// Result of a safety evaluation cycle.
#[derive(Debug, Clone, Copy)]
pub struct SafetyStatus {
    pub any_violation: bool,
    pub response: FaultResponse,
}

/// Aggregator that evaluates multiple safety monitors in a single cycle.
/// Uses const generics for zero-allocation operation.
///
/// N = number of range monitors, M = number of rate monitors, T = number of timeout monitors.
#[derive(Debug)]
pub struct SafetyMonitor<S: ControlScalar, const N: usize, const M: usize, const T: usize> {
    pub range_monitors: [Option<RangeMonitor<S>>; N],
    pub rate_monitors: [Option<RateMonitor<S>>; M],
    pub timeout_monitors: [Option<TimeoutMonitor<S>>; T],
    range_severities: [FaultSeverity; N],
    rate_severities: [FaultSeverity; M],
    timeout_severities: [FaultSeverity; T],
}

impl<S: ControlScalar, const N: usize, const M: usize, const T: usize> SafetyMonitor<S, N, M, T> {
    pub fn new() -> Self {
        Self {
            range_monitors: core::array::from_fn(|_| None),
            rate_monitors: core::array::from_fn(|_| None),
            timeout_monitors: core::array::from_fn(|_| None),
            range_severities: [FaultSeverity::Warning; N],
            rate_severities: [FaultSeverity::Warning; M],
            timeout_severities: [FaultSeverity::Warning; T],
        }
    }

    pub fn set_range(&mut self, slot: usize, min: S, max: S, severity: FaultSeverity) {
        if slot < N {
            self.range_monitors[slot] = Some(RangeMonitor::new(min, max));
            self.range_severities[slot] = severity;
        }
    }

    pub fn set_rate(&mut self, slot: usize, max_rate: S, severity: FaultSeverity) {
        if slot < M {
            self.rate_monitors[slot] = Some(RateMonitor::new(max_rate));
            self.rate_severities[slot] = severity;
        }
    }

    pub fn set_timeout(&mut self, slot: usize, timeout: S, severity: FaultSeverity) {
        if slot < T {
            self.timeout_monitors[slot] = Some(TimeoutMonitor::new(timeout));
            self.timeout_severities[slot] = severity;
        }
    }

    /// Evaluate range monitors with N values, rate monitors with M values,
    /// and advance timeout monitors by `dt`.
    pub fn evaluate(&mut self, range_values: &[S; N], rate_values: &[S; M], dt: S) -> SafetyStatus {
        let mut any_violation = false;
        let mut worst_severity = FaultSeverity::Info;

        for (i, val) in range_values.iter().enumerate() {
            if let Some(ref mut mon) = self.range_monitors[i] {
                if !mon.check(*val) {
                    any_violation = true;
                    if self.range_severities[i] > worst_severity {
                        worst_severity = self.range_severities[i];
                    }
                }
            }
        }

        for (i, val) in rate_values.iter().enumerate() {
            if let Some(ref mut mon) = self.rate_monitors[i] {
                if !mon.check(*val, dt) {
                    any_violation = true;
                    if self.rate_severities[i] > worst_severity {
                        worst_severity = self.rate_severities[i];
                    }
                }
            }
        }

        for i in 0..T {
            if let Some(ref mut mon) = self.timeout_monitors[i] {
                if !mon.check(dt) {
                    any_violation = true;
                    if self.timeout_severities[i] > worst_severity {
                        worst_severity = self.timeout_severities[i];
                    }
                }
            }
        }

        SafetyStatus {
            any_violation,
            response: worst_severity.default_response(),
        }
    }
}

impl<S: ControlScalar, const N: usize, const M: usize, const T: usize> Default
    for SafetyMonitor<S, N, M, T>
{
    fn default() -> Self {
        Self::new()
    }
}
