//! CiA DS-402 Drive Profile — standard motion control state machine.
//!
//! DS-402 defines a state machine for variable-speed drives and servo controllers.
//! Used in EtherCAT/CANopen drives (e.g. Beckhoff, Schneider, Yaskawa).
//!
//! States: Not Ready → Switch On Disabled → Ready To Switch On →
//!         Switched On → Operation Enabled ↔ Quick Stop Active
//!                     ↕
//!         Fault Reaction Active → Fault

/// DS-402 control word bits (0x6040).
pub mod cw {
    pub const SWITCH_ON: u16 = 1 << 0;
    pub const ENABLE_VOLTAGE: u16 = 1 << 1;
    pub const QUICK_STOP: u16 = 1 << 2; // active low
    pub const ENABLE_OPERATION: u16 = 1 << 3;
    pub const FAULT_RESET: u16 = 1 << 7;
    pub const HALT: u16 = 1 << 8;
}

/// DS-402 status word bits (0x6041).
pub mod sw {
    pub const READY_TO_SWITCH_ON: u16 = 1 << 0;
    pub const SWITCHED_ON: u16 = 1 << 1;
    pub const OPERATION_ENABLED: u16 = 1 << 2;
    pub const FAULT: u16 = 1 << 3;
    pub const VOLTAGE_ENABLED: u16 = 1 << 4;
    pub const QUICK_STOP: u16 = 1 << 5; // active low in drive
    pub const SWITCH_ON_DISABLED: u16 = 1 << 6;
    pub const WARNING: u16 = 1 << 7;
    pub const TARGET_REACHED: u16 = 1 << 10;
}

/// DS-402 drive state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriveState {
    NotReadyToSwitchOn,
    SwitchOnDisabled,
    ReadyToSwitchOn,
    SwitchedOn,
    OperationEnabled,
    QuickStopActive,
    FaultReactionActive,
    Fault,
}

impl DriveState {
    /// Status word for this state (simplified).
    pub fn status_word(&self) -> u16 {
        match self {
            Self::NotReadyToSwitchOn => 0x0000,
            Self::SwitchOnDisabled => sw::SWITCH_ON_DISABLED,
            Self::ReadyToSwitchOn => sw::READY_TO_SWITCH_ON | sw::QUICK_STOP,
            Self::SwitchedOn => sw::READY_TO_SWITCH_ON | sw::SWITCHED_ON | sw::QUICK_STOP,
            Self::OperationEnabled => {
                sw::READY_TO_SWITCH_ON
                    | sw::SWITCHED_ON
                    | sw::OPERATION_ENABLED
                    | sw::QUICK_STOP
                    | sw::VOLTAGE_ENABLED
            }
            Self::QuickStopActive => sw::VOLTAGE_ENABLED,
            Self::FaultReactionActive => sw::FAULT | sw::VOLTAGE_ENABLED,
            Self::Fault => sw::FAULT | sw::SWITCH_ON_DISABLED,
        }
    }
}

/// DS-402 state machine for one drive axis.
#[derive(Debug, Clone, Copy)]
pub struct Ds402StateMachine {
    pub state: DriveState,
    /// Last applied control word.
    pub control_word: u16,
    /// Active fault code (0 = no fault).
    pub fault_code: u16,
    /// Operation mode (0x6060): 1=PP, 3=PV, 8=CSP, 9=CSV, 10=CST.
    pub op_mode: i8,
}

impl Ds402StateMachine {
    pub fn new() -> Self {
        Self {
            state: DriveState::SwitchOnDisabled,
            control_word: 0,
            fault_code: 0,
            op_mode: 8, // Cyclic Sync Position by default
        }
    }

    /// Apply control word — advance state machine.
    pub fn apply_control_word(&mut self, cw_val: u16) {
        self.control_word = cw_val;

        let sw_on = cw_val & cw::SWITCH_ON != 0;
        let ev = cw_val & cw::ENABLE_VOLTAGE != 0;
        let qs = cw_val & cw::QUICK_STOP != 0; // active-high in control word
        let eo = cw_val & cw::ENABLE_OPERATION != 0;
        let fr = cw_val & cw::FAULT_RESET != 0;

        self.state = match self.state {
            DriveState::SwitchOnDisabled => {
                if ev && qs && !sw_on {
                    DriveState::ReadyToSwitchOn
                } else {
                    self.state
                }
            }
            DriveState::ReadyToSwitchOn => {
                if !ev {
                    DriveState::SwitchOnDisabled
                } else if sw_on && ev && qs {
                    DriveState::SwitchedOn
                } else {
                    self.state
                }
            }
            DriveState::SwitchedOn => {
                if !ev {
                    DriveState::SwitchOnDisabled
                } else if !sw_on {
                    DriveState::ReadyToSwitchOn
                } else if eo {
                    DriveState::OperationEnabled
                } else {
                    self.state
                }
            }
            DriveState::OperationEnabled => {
                if !ev {
                    DriveState::SwitchOnDisabled
                } else if !sw_on {
                    DriveState::ReadyToSwitchOn
                } else if !qs {
                    DriveState::QuickStopActive
                } else if !eo {
                    DriveState::SwitchedOn
                } else {
                    self.state
                }
            }
            DriveState::QuickStopActive => {
                if !qs {
                    self.state
                }
                // stay in QS
                else {
                    DriveState::SwitchOnDisabled
                }
            }
            DriveState::Fault => {
                if fr {
                    DriveState::SwitchOnDisabled
                } else {
                    self.state
                }
            }
            DriveState::FaultReactionActive => DriveState::Fault,
            DriveState::NotReadyToSwitchOn => self.state,
        };
    }

    /// Inject a fault (transitions to FaultReactionActive → Fault).
    pub fn inject_fault(&mut self, fault_code: u16) {
        self.fault_code = fault_code;
        self.state = DriveState::FaultReactionActive;
    }

    /// Current status word.
    pub fn status_word(&self) -> u16 {
        self.state.status_word()
    }

    pub fn is_operation_enabled(&self) -> bool {
        self.state == DriveState::OperationEnabled
    }
}

impl Default for Ds402StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bring_up(sm: &mut Ds402StateMachine) {
        // Standard bring-up sequence
        sm.apply_control_word(cw::ENABLE_VOLTAGE | cw::QUICK_STOP); // → ReadyToSwitchOn
        sm.apply_control_word(cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP); // → SwitchedOn
        sm.apply_control_word(
            cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::QUICK_STOP | cw::ENABLE_OPERATION,
        ); // → OperationEnabled
    }

    #[test]
    fn standard_bring_up() {
        let mut sm = Ds402StateMachine::new();
        bring_up(&mut sm);
        assert_eq!(sm.state, DriveState::OperationEnabled);
    }

    #[test]
    fn fault_injection_and_reset() {
        let mut sm = Ds402StateMachine::new();
        bring_up(&mut sm);
        sm.inject_fault(0x3210);
        // Fault reaction → Fault on next CW
        sm.apply_control_word(0); // triggers FaultReactionActive → Fault
        assert_eq!(sm.state, DriveState::Fault);

        // Reset
        sm.apply_control_word(cw::FAULT_RESET);
        assert_eq!(sm.state, DriveState::SwitchOnDisabled);
    }

    #[test]
    fn quick_stop_from_operation_enabled() {
        let mut sm = Ds402StateMachine::new();
        bring_up(&mut sm);
        // Quick stop: QUICK_STOP bit = 0
        sm.apply_control_word(cw::SWITCH_ON | cw::ENABLE_VOLTAGE | cw::ENABLE_OPERATION);
        assert_eq!(sm.state, DriveState::QuickStopActive);
    }

    #[test]
    fn status_word_op_enabled() {
        let mut sm = Ds402StateMachine::new();
        bring_up(&mut sm);
        let sw_val = sm.status_word();
        assert!(sw_val & sw::OPERATION_ENABLED != 0);
        assert!(sw_val & sw::SWITCHED_ON != 0);
        assert!(sm.is_operation_enabled());
    }
}
