use crate::safety::fault::FaultSeverity;

/// Safe state levels for a control system.
///
/// States are ordered by severity. Transitions can only increase severity
/// (degradation is irreversible until an explicit reset/recommission).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SafeStateLevel {
    /// Normal operation — all systems healthy.
    Normal,
    /// Warning — degraded performance expected, monitoring active.
    Warning,
    /// Degraded — reduced functionality, operator notification required.
    Degraded,
    /// Emergency stop — actuators disabled, power may be cut.
    EmergencyStop,
}

/// Safe state machine for a control system.
///
/// Manages transitions between safe states based on detected faults.
/// Once in EmergencyStop, an explicit `reset()` (recommissioning) is required.
#[derive(Debug, Clone, Copy)]
pub struct SafeStateMachine {
    state: SafeStateLevel,
    /// Whether a reset (recommissioning) is required before returning to Normal.
    requires_reset: bool,
}

impl SafeStateMachine {
    pub fn new() -> Self {
        Self {
            state: SafeStateLevel::Normal,
            requires_reset: false,
        }
    }

    /// Current safe state level.
    pub fn state(&self) -> SafeStateLevel {
        self.state
    }

    /// Apply a fault and transition to appropriate state if needed.
    ///
    /// Returns the new (possibly unchanged) state level.
    pub fn apply_fault(&mut self, severity: FaultSeverity) -> SafeStateLevel {
        let new_level = match severity {
            FaultSeverity::Info => SafeStateLevel::Normal,
            FaultSeverity::Warning => SafeStateLevel::Warning,
            FaultSeverity::Error => SafeStateLevel::Degraded,
            FaultSeverity::Critical => SafeStateLevel::EmergencyStop,
        };

        if new_level > self.state {
            self.state = new_level;
            if self.state == SafeStateLevel::EmergencyStop {
                self.requires_reset = true;
            }
        }
        self.state
    }

    /// Returns true if the system is in a safe/normal operating state.
    pub fn is_operational(&self) -> bool {
        matches!(self.state, SafeStateLevel::Normal | SafeStateLevel::Warning)
    }

    /// Whether actuators should be disabled (degraded or worse).
    pub fn should_disable_actuators(&self) -> bool {
        self.state >= SafeStateLevel::Degraded
    }

    /// Attempt to reset to Normal. Only succeeds if not in EmergencyStop
    /// (or if `force` is true, simulating a recommission procedure).
    pub fn reset(&mut self, force: bool) -> bool {
        if self.state == SafeStateLevel::EmergencyStop && !force {
            return false; // Cannot self-reset from E-Stop
        }
        self.state = SafeStateLevel::Normal;
        self.requires_reset = false;
        true
    }

    /// Clear a Warning-level state (auto-recovery when fault clears).
    pub fn clear_warning(&mut self) {
        if self.state == SafeStateLevel::Warning {
            self.state = SafeStateLevel::Normal;
        }
    }

    pub fn requires_reset(&self) -> bool {
        self.requires_reset
    }
}

impl Default for SafeStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_in_normal() {
        let fsm = SafeStateMachine::new();
        assert_eq!(fsm.state(), SafeStateLevel::Normal);
        assert!(fsm.is_operational());
    }

    #[test]
    fn warning_fault_transitions() {
        let mut fsm = SafeStateMachine::new();
        fsm.apply_fault(FaultSeverity::Warning);
        assert_eq!(fsm.state(), SafeStateLevel::Warning);
        assert!(fsm.is_operational());
        assert!(!fsm.should_disable_actuators());
    }

    #[test]
    fn critical_fault_triggers_estop() {
        let mut fsm = SafeStateMachine::new();
        fsm.apply_fault(FaultSeverity::Critical);
        assert_eq!(fsm.state(), SafeStateLevel::EmergencyStop);
        assert!(!fsm.is_operational());
        assert!(fsm.should_disable_actuators());
        assert!(fsm.requires_reset());
    }

    #[test]
    fn cannot_self_reset_from_estop() {
        let mut fsm = SafeStateMachine::new();
        fsm.apply_fault(FaultSeverity::Critical);
        let ok = fsm.reset(false);
        assert!(!ok);
        assert_eq!(fsm.state(), SafeStateLevel::EmergencyStop);
    }

    #[test]
    fn force_reset_from_estop() {
        let mut fsm = SafeStateMachine::new();
        fsm.apply_fault(FaultSeverity::Critical);
        let ok = fsm.reset(true);
        assert!(ok);
        assert_eq!(fsm.state(), SafeStateLevel::Normal);
    }

    #[test]
    fn state_monotonically_increases_on_faults() {
        let mut fsm = SafeStateMachine::new();
        fsm.apply_fault(FaultSeverity::Critical); // → EmergencyStop
                                                  // A less severe fault cannot lower the state
        fsm.apply_fault(FaultSeverity::Warning);
        assert_eq!(fsm.state(), SafeStateLevel::EmergencyStop);
    }

    #[test]
    fn clear_warning_recovers_to_normal() {
        let mut fsm = SafeStateMachine::new();
        fsm.apply_fault(FaultSeverity::Warning);
        fsm.clear_warning();
        assert_eq!(fsm.state(), SafeStateLevel::Normal);
    }
}
