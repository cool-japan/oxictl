//! Extended safe-state machine for functional safety (IEC 61508 / IEC 62061).
//!
//! Provides [`SafeStateMachineExt`], a four-state machine that manages
//! system-level safe-state transitions driven by fault severity.
//!
//! ## State model
//!
//! ```text
//! Normal ──(warning/degraded fault)──► Degraded
//!   │                                      │
//!   └──(safe-state fault)──► SafeState ◄───┘
//!            │                   │
//!            └──(emergency)──► EmergencyStop (latching)
//! ```
//!
//! Once in `SafeState` or `EmergencyStop` the machine latches.  A keyed reset
//! sequence (matching the configured `reset_key`) is required to return to
//! `Normal`.
//!
//! All state is stored inline — no heap allocation, fully `no_std` compatible.

#![allow(dead_code)]

use crate::safety::fault::FaultSeverity;

// ─────────────────────────────────────────────────────────────────────────────
// MachineState
// ─────────────────────────────────────────────────────────────────────────────

/// The four states of the extended safe-state machine.
///
/// Ordered from least to most restrictive; transitions may only move upward
/// (increasing severity) during automatic fault processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MachineState {
    /// Normal operation — all safety functions active, no faults active.
    Normal,
    /// Degraded operation — reduced functionality; non-critical fault active.
    /// Actuators remain enabled but outputs may be limited.
    Degraded,
    /// Safe state — safety function maintains process in a defined safe condition.
    /// Actuators are inhibited; manual operator action required before reset.
    SafeState,
    /// Emergency stop — immediate power removal / de-energisation.
    /// Latching: requires keyed reset + verification sequence.
    EmergencyStop,
}

impl MachineState {
    /// Returns `true` if the state is latching (requires keyed reset).
    pub fn is_latching(self) -> bool {
        matches!(self, MachineState::SafeState | MachineState::EmergencyStop)
    }

    /// Returns `true` if actuator outputs should be inhibited.
    pub fn inhibits_actuators(self) -> bool {
        matches!(self, MachineState::SafeState | MachineState::EmergencyStop)
    }

    /// Returns `true` if the safety function is fully operational.
    pub fn is_operational(self) -> bool {
        matches!(self, MachineState::Normal | MachineState::Degraded)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SafeStateAction
// ─────────────────────────────────────────────────────────────────────────────

/// The action the safe-state machine prescribes in response to a fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeStateAction {
    /// Continue normal operation; no state change.
    Continue,
    /// Transition to (or remain in) degraded mode.
    EnterDegraded,
    /// Transition to (or remain in) safe state.
    EnterSafeState,
    /// Initiate emergency stop.
    EmergencyStop,
}

// ─────────────────────────────────────────────────────────────────────────────
// SafeStateConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`SafeStateMachineExt`].
///
/// Defines the fault-severity thresholds that drive state transitions and
/// the keyed reset credential.
#[derive(Debug, Clone, Copy)]
pub struct SafeStateConfig {
    /// Faults at this severity or above (but below `safe_state_threshold`) trigger
    /// the `Degraded` state.
    pub degraded_threshold: FaultSeverity,
    /// Faults at this severity or above (but below `emergency_stop_threshold`)
    /// trigger the `SafeState`.
    pub safe_state_threshold: FaultSeverity,
    /// Faults at this severity trigger an `EmergencyStop`.
    pub emergency_stop_threshold: FaultSeverity,
    /// 32-bit key required to authorise a reset from a latching state.
    ///
    /// Chose a value that is hard to guess accidentally (not 0 or 0xFFFF_FFFF).
    pub reset_key: u32,
}

impl SafeStateConfig {
    /// Construct a configuration with explicit thresholds and reset key.
    ///
    /// The caller must ensure that:
    ///   `degraded_threshold ≤ safe_state_threshold ≤ emergency_stop_threshold`
    ///
    /// A violation is allowed but will produce degenerate (potentially unsafe)
    /// behaviour; no panic is raised so that `no_std` targets are supported.
    pub fn new(
        degraded_threshold: FaultSeverity,
        safe_state_threshold: FaultSeverity,
        emergency_stop_threshold: FaultSeverity,
        reset_key: u32,
    ) -> Self {
        Self {
            degraded_threshold,
            safe_state_threshold,
            emergency_stop_threshold,
            reset_key,
        }
    }

    /// A sensible default configuration suitable for SIL-2 systems:
    /// - Warning → Degraded
    /// - Error → SafeState
    /// - Critical → EmergencyStop
    /// - reset_key = 0xDEAD_BEEF
    pub fn default_sil2() -> Self {
        Self::new(
            FaultSeverity::Warning,
            FaultSeverity::Error,
            FaultSeverity::Critical,
            0xDEAD_BEEF,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SafeError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from safe-state machine operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeError {
    /// The supplied reset key does not match the configured key.
    InvalidResetKey,
    /// A reset was attempted from a non-latching state.
    NotInLatchingState,
    /// The machine is in EmergencyStop; safe-state reset is not sufficient.
    EmergencyStopRequiresFullReset,
}

impl core::fmt::Display for SafeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SafeError::InvalidResetKey => write!(f, "safe-state reset rejected: invalid key"),
            SafeError::NotInLatchingState => {
                write!(f, "safe-state reset not needed: machine is not latching")
            }
            SafeError::EmergencyStopRequiresFullReset => write!(
                f,
                "EmergencyStop requires emergency_stop_reset(), not safe_state_reset()"
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SafeStateMachineExt
// ─────────────────────────────────────────────────────────────────────────────

/// Extended safe-state machine for functional safety.
///
/// Manages transitions between `Normal`, `Degraded`, `SafeState`, and
/// `EmergencyStop` driven by fault severity.  `SafeState` and `EmergencyStop`
/// are latching: once entered, an explicit keyed `reset()` call is required
/// before the machine returns to `Normal`.
///
/// The machine tracks:
/// - The current [`MachineState`].
/// - The number of consecutive cycles in the current state.
/// - Whether a safety function is currently active (affects the severity
///   mapping: if the safety function is *not* active, all faults escalate by
///   one tier).
///
/// # Example
/// ```
/// use oxictl::safety::safe_state_ext::{SafeStateMachineExt, SafeStateConfig};
/// use oxictl::safety::fault::FaultSeverity;
///
/// let config = SafeStateConfig::default_sil2();
/// let mut fsm = SafeStateMachineExt::new(config);
/// assert!(fsm.state().is_operational());
///
/// // A critical fault drives an emergency stop
/// fsm.process_fault(FaultSeverity::Critical, true);
/// assert_eq!(fsm.state(), oxictl::safety::safe_state_ext::MachineState::EmergencyStop);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SafeStateMachineExt {
    config: SafeStateConfig,
    state: MachineState,
    /// Monotonic count of process_fault() calls in the current state.
    state_cycles: u32,
    /// Cumulative count of all faults processed since last reset.
    total_faults: u32,
}

impl SafeStateMachineExt {
    /// Create a new machine in the `Normal` state.
    pub fn new(config: SafeStateConfig) -> Self {
        Self {
            config,
            state: MachineState::Normal,
            state_cycles: 0,
            total_faults: 0,
        }
    }

    /// Current machine state.
    pub fn state(&self) -> MachineState {
        self.state
    }

    /// How many consecutive `process_fault` cycles the machine has spent in the
    /// current state.
    pub fn state_cycles(&self) -> u32 {
        self.state_cycles
    }

    /// Total number of fault-processing calls since last `reset`.
    pub fn total_faults(&self) -> u32 {
        self.total_faults
    }

    /// Process an incoming fault and advance the machine state as required.
    ///
    /// # Parameters
    /// - `fault_severity` — severity of the fault being reported.
    /// - `safety_function_active` — if `false`, the safety function is
    ///   unavailable (e.g. a diagnostic channel is offline); all faults
    ///   escalate by one tier.
    ///
    /// # Returns
    /// The [`SafeStateAction`] the caller should execute.
    pub fn process_fault(
        &mut self,
        fault_severity: FaultSeverity,
        safety_function_active: bool,
    ) -> SafeStateAction {
        self.total_faults = self.total_faults.saturating_add(1);

        // If the safety function is not active, escalate by one severity tier.
        let effective_severity = if safety_function_active {
            fault_severity
        } else {
            escalate(fault_severity)
        };

        let required_action = self.severity_to_action(effective_severity);
        let required_state = action_to_state(required_action);

        // Latching check: once in a latching state, we can never automatically
        // downgrade — only an explicit reset() can do that.
        if required_state > self.state {
            let prev = self.state;
            self.state = required_state;
            if prev != self.state {
                self.state_cycles = 0;
            }
        }
        self.state_cycles = self.state_cycles.saturating_add(1);

        // Return the action dictated by the *current* (possibly latched) state.
        state_to_action(self.state)
    }

    /// Map a fault severity to the required action given the configured thresholds.
    fn severity_to_action(&self, severity: FaultSeverity) -> SafeStateAction {
        if severity >= self.config.emergency_stop_threshold {
            SafeStateAction::EmergencyStop
        } else if severity >= self.config.safe_state_threshold {
            SafeStateAction::EnterSafeState
        } else if severity >= self.config.degraded_threshold {
            SafeStateAction::EnterDegraded
        } else {
            SafeStateAction::Continue
        }
    }

    /// Attempt to reset from `SafeState` to `Normal`.
    ///
    /// Requires `key == config.reset_key`.
    ///
    /// # Errors
    /// - [`SafeError::InvalidResetKey`] — wrong key supplied.
    /// - [`SafeError::NotInLatchingState`] — machine is not currently latching.
    /// - [`SafeError::EmergencyStopRequiresFullReset`] — machine is in
    ///   `EmergencyStop`; use `emergency_stop_reset()` instead.
    pub fn reset(&mut self, key: u32) -> Result<(), SafeError> {
        if !self.state.is_latching() {
            return Err(SafeError::NotInLatchingState);
        }
        if self.state == MachineState::EmergencyStop {
            return Err(SafeError::EmergencyStopRequiresFullReset);
        }
        if key != self.config.reset_key {
            return Err(SafeError::InvalidResetKey);
        }
        self.state = MachineState::Normal;
        self.state_cycles = 0;
        Ok(())
    }

    /// Perform a full reset from `EmergencyStop` to `Normal`.
    ///
    /// Requires `key == config.reset_key`.  Two-step authorisation is
    /// deliberately required for the most severe state.
    ///
    /// # Errors
    /// - [`SafeError::InvalidResetKey`] — wrong key supplied.
    /// - [`SafeError::NotInLatchingState`] — machine is not in `EmergencyStop`.
    pub fn emergency_stop_reset(&mut self, key: u32) -> Result<(), SafeError> {
        if self.state != MachineState::EmergencyStop {
            return Err(SafeError::NotInLatchingState);
        }
        if key != self.config.reset_key {
            return Err(SafeError::InvalidResetKey);
        }
        self.state = MachineState::Normal;
        self.state_cycles = 0;
        Ok(())
    }

    /// Force the machine back to `Normal` without key verification.
    ///
    /// **Only for use in test or commissioning contexts.** This method bypasses
    /// all safety latches; in a production system this function should not be
    /// called without additional hardware interlocks.
    #[cfg(test)]
    pub fn force_reset_for_test(&mut self) {
        self.state = MachineState::Normal;
        self.state_cycles = 0;
        self.total_faults = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Escalate a fault severity by one tier (saturates at `Critical`).
fn escalate(s: FaultSeverity) -> FaultSeverity {
    match s {
        FaultSeverity::Info => FaultSeverity::Warning,
        FaultSeverity::Warning => FaultSeverity::Error,
        FaultSeverity::Error | FaultSeverity::Critical => FaultSeverity::Critical,
    }
}

fn action_to_state(a: SafeStateAction) -> MachineState {
    match a {
        SafeStateAction::Continue => MachineState::Normal,
        SafeStateAction::EnterDegraded => MachineState::Degraded,
        SafeStateAction::EnterSafeState => MachineState::SafeState,
        SafeStateAction::EmergencyStop => MachineState::EmergencyStop,
    }
}

fn state_to_action(s: MachineState) -> SafeStateAction {
    match s {
        MachineState::Normal => SafeStateAction::Continue,
        MachineState::Degraded => SafeStateAction::EnterDegraded,
        MachineState::SafeState => SafeStateAction::EnterSafeState,
        MachineState::EmergencyStop => SafeStateAction::EmergencyStop,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_fsm() -> SafeStateMachineExt {
        SafeStateMachineExt::new(SafeStateConfig::default_sil2())
    }

    // ── Basic state transitions ─────────────────────────────────────────────

    #[test]
    fn starts_in_normal() {
        let fsm = default_fsm();
        assert_eq!(fsm.state(), MachineState::Normal);
        assert!(fsm.state().is_operational());
    }

    #[test]
    fn info_fault_stays_normal() {
        let mut fsm = default_fsm();
        let action = fsm.process_fault(FaultSeverity::Info, true);
        assert_eq!(action, SafeStateAction::Continue);
        assert_eq!(fsm.state(), MachineState::Normal);
    }

    #[test]
    fn warning_fault_enters_degraded() {
        let mut fsm = default_fsm();
        let action = fsm.process_fault(FaultSeverity::Warning, true);
        assert_eq!(action, SafeStateAction::EnterDegraded);
        assert_eq!(fsm.state(), MachineState::Degraded);
    }

    #[test]
    fn error_fault_enters_safe_state() {
        let mut fsm = default_fsm();
        let action = fsm.process_fault(FaultSeverity::Error, true);
        assert_eq!(action, SafeStateAction::EnterSafeState);
        assert_eq!(fsm.state(), MachineState::SafeState);
        assert!(fsm.state().inhibits_actuators());
    }

    #[test]
    fn critical_fault_enters_emergency_stop() {
        let mut fsm = default_fsm();
        let action = fsm.process_fault(FaultSeverity::Critical, true);
        assert_eq!(action, SafeStateAction::EmergencyStop);
        assert_eq!(fsm.state(), MachineState::EmergencyStop);
        assert!(fsm.state().inhibits_actuators());
    }

    // ── Latching behaviour ──────────────────────────────────────────────────

    #[test]
    fn safe_state_latches_against_lower_faults() {
        let mut fsm = default_fsm();
        fsm.process_fault(FaultSeverity::Error, true); // → SafeState
        fsm.process_fault(FaultSeverity::Info, true); // lower severity
                                                      // Machine must remain in SafeState
        assert_eq!(fsm.state(), MachineState::SafeState);
    }

    #[test]
    fn emergency_stop_latches_against_all_lower_faults() {
        let mut fsm = default_fsm();
        fsm.process_fault(FaultSeverity::Critical, true); // → EmergencyStop
        fsm.process_fault(FaultSeverity::Warning, true); // lower
        assert_eq!(fsm.state(), MachineState::EmergencyStop);
    }

    #[test]
    fn safe_state_escalates_to_emergency_on_critical() {
        let mut fsm = default_fsm();
        fsm.process_fault(FaultSeverity::Error, true); // → SafeState
        fsm.process_fault(FaultSeverity::Critical, true); // → EmergencyStop
        assert_eq!(fsm.state(), MachineState::EmergencyStop);
    }

    // ── Keyed reset ─────────────────────────────────────────────────────────

    #[test]
    fn keyed_reset_from_safe_state_succeeds() {
        let mut fsm = default_fsm();
        fsm.process_fault(FaultSeverity::Error, true); // → SafeState
        let result = fsm.reset(0xDEAD_BEEF);
        assert!(result.is_ok());
        assert_eq!(fsm.state(), MachineState::Normal);
    }

    #[test]
    fn keyed_reset_from_safe_state_wrong_key() {
        let mut fsm = default_fsm();
        fsm.process_fault(FaultSeverity::Error, true); // → SafeState
        let result = fsm.reset(0x1234_5678);
        assert_eq!(result, Err(SafeError::InvalidResetKey));
        assert_eq!(fsm.state(), MachineState::SafeState); // unchanged
    }

    #[test]
    fn reset_from_safe_state_blocked_in_estop() {
        let mut fsm = default_fsm();
        fsm.process_fault(FaultSeverity::Critical, true); // → EmergencyStop
        let result = fsm.reset(0xDEAD_BEEF);
        assert_eq!(result, Err(SafeError::EmergencyStopRequiresFullReset));
    }

    #[test]
    fn reset_from_normal_is_error() {
        let mut fsm = default_fsm();
        let result = fsm.reset(0xDEAD_BEEF);
        assert_eq!(result, Err(SafeError::NotInLatchingState));
    }

    #[test]
    fn emergency_stop_reset_correct_key() {
        let mut fsm = default_fsm();
        fsm.process_fault(FaultSeverity::Critical, true); // → EmergencyStop
        let result = fsm.emergency_stop_reset(0xDEAD_BEEF);
        assert!(result.is_ok());
        assert_eq!(fsm.state(), MachineState::Normal);
    }

    #[test]
    fn emergency_stop_reset_wrong_key() {
        let mut fsm = default_fsm();
        fsm.process_fault(FaultSeverity::Critical, true);
        let result = fsm.emergency_stop_reset(0x0000_0001);
        assert_eq!(result, Err(SafeError::InvalidResetKey));
        assert_eq!(fsm.state(), MachineState::EmergencyStop);
    }

    #[test]
    fn emergency_stop_reset_when_not_in_estop() {
        let mut fsm = default_fsm();
        let result = fsm.emergency_stop_reset(0xDEAD_BEEF);
        assert_eq!(result, Err(SafeError::NotInLatchingState));
    }

    // ── Safety-function-inactive escalation ─────────────────────────────────

    #[test]
    fn info_fault_escalates_to_warning_when_sf_inactive() {
        let mut fsm = default_fsm();
        // Info → escalates to Warning → Degraded
        let action = fsm.process_fault(FaultSeverity::Info, false);
        assert_eq!(action, SafeStateAction::EnterDegraded);
    }

    #[test]
    fn warning_fault_escalates_to_error_when_sf_inactive() {
        let mut fsm = default_fsm();
        // Warning → escalates to Error → SafeState
        let action = fsm.process_fault(FaultSeverity::Warning, false);
        assert_eq!(action, SafeStateAction::EnterSafeState);
    }

    #[test]
    fn error_fault_escalates_to_critical_when_sf_inactive() {
        let mut fsm = default_fsm();
        // Error → escalates to Critical → EmergencyStop
        let action = fsm.process_fault(FaultSeverity::Error, false);
        assert_eq!(action, SafeStateAction::EmergencyStop);
    }

    // ── Cycle counting ──────────────────────────────────────────────────────

    #[test]
    fn state_cycles_increment_each_call() {
        let mut fsm = default_fsm();
        fsm.process_fault(FaultSeverity::Info, true);
        fsm.process_fault(FaultSeverity::Info, true);
        assert_eq!(fsm.state_cycles(), 2);
    }

    #[test]
    fn state_cycles_reset_on_state_change() {
        let mut fsm = default_fsm();
        fsm.process_fault(FaultSeverity::Info, true); // Normal, cycles=1
        fsm.process_fault(FaultSeverity::Warning, true); // → Degraded, cycles=1
        assert_eq!(fsm.state_cycles(), 1);
    }

    #[test]
    fn total_faults_counts_all_calls() {
        let mut fsm = default_fsm();
        for _ in 0..5 {
            fsm.process_fault(FaultSeverity::Info, true);
        }
        assert_eq!(fsm.total_faults(), 5);
    }

    // ── MachineState helpers ────────────────────────────────────────────────

    #[test]
    fn state_is_latching_flags() {
        assert!(!MachineState::Normal.is_latching());
        assert!(!MachineState::Degraded.is_latching());
        assert!(MachineState::SafeState.is_latching());
        assert!(MachineState::EmergencyStop.is_latching());
    }

    #[test]
    fn state_inhibits_actuators_flags() {
        assert!(!MachineState::Normal.inhibits_actuators());
        assert!(!MachineState::Degraded.inhibits_actuators());
        assert!(MachineState::SafeState.inhibits_actuators());
        assert!(MachineState::EmergencyStop.inhibits_actuators());
    }
}
