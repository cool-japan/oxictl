//! Fault escalation and handler validation.

use oxictl::safety::fault::{FaultDef, FaultEvent, FaultResponse, FaultSeverity};
use oxictl::safety::handler::FaultHandler;

fn make_fault(id: u16, name: &'static str, severity: FaultSeverity) -> FaultEvent {
    let def = FaultDef {
        id,
        name,
        severity,
        response: severity.default_response(),
    };
    FaultEvent::new(&def, 0.0)
}

/// Info fault → LogAndContinue.
#[test]
fn info_fault_logs_and_continues() {
    let mut handler = FaultHandler::<16>::new();
    let response = handler.report(make_fault(1, "sensor_drift", FaultSeverity::Info));
    assert_eq!(response, FaultResponse::LogAndContinue);
    assert!(!handler.has_critical());
}

/// Warning fault → LogAndContinue.
#[test]
fn warning_fault_logs_and_continues() {
    let mut handler = FaultHandler::<16>::new();
    let response = handler.report(make_fault(2, "high_temp", FaultSeverity::Warning));
    assert_eq!(response, FaultResponse::LogAndContinue);
}

/// Error fault → Degrade.
#[test]
fn error_fault_triggers_degrade() {
    let mut handler = FaultHandler::<16>::new();
    let response = handler.report(make_fault(3, "encoder_error", FaultSeverity::Error));
    assert_eq!(response, FaultResponse::Degrade);
}

/// Critical fault → EmergencyStop.
#[test]
fn critical_fault_triggers_emergency_stop() {
    let mut handler = FaultHandler::<16>::new();
    let response = handler.report(make_fault(4, "overcurrent", FaultSeverity::Critical));
    assert_eq!(response, FaultResponse::EmergencyStop);
    assert!(handler.has_critical());
}

/// Multiple faults accumulate correctly.
#[test]
fn multiple_faults_accumulate() {
    let mut handler = FaultHandler::<16>::new();

    handler.report(make_fault(1, "f1", FaultSeverity::Info));
    handler.report(make_fault(2, "f2", FaultSeverity::Warning));
    handler.report(make_fault(3, "f3", FaultSeverity::Error));

    assert_eq!(handler.fault_count(), 3);
    assert!(!handler.has_critical());

    let resp = handler.report(make_fault(4, "f4", FaultSeverity::Critical));
    assert_eq!(resp, FaultResponse::EmergencyStop);
    assert!(handler.has_critical());
    assert_eq!(handler.fault_count(), 4);
}

/// Fault handler clear resets state.
#[test]
fn fault_handler_clear_resets() {
    let mut handler = FaultHandler::<16>::new();
    handler.report(make_fault(1, "overheat", FaultSeverity::Critical));
    assert!(handler.has_critical());

    handler.clear();
    assert!(!handler.has_critical());
    assert_eq!(handler.fault_count(), 0);
}

/// Current response escalates with severity.
#[test]
fn current_response_reflects_highest_severity() {
    let mut handler = FaultHandler::<16>::new();

    handler.report(make_fault(1, "err", FaultSeverity::Error));
    assert_eq!(handler.current_response(), FaultResponse::Degrade);

    handler.report(make_fault(2, "crit", FaultSeverity::Critical));
    assert_eq!(handler.current_response(), FaultResponse::EmergencyStop);
}

/// FaultSeverity ordering: Critical > Error > Warning > Info.
#[test]
fn fault_severity_ordering() {
    assert!(FaultSeverity::Critical > FaultSeverity::Error);
    assert!(FaultSeverity::Error > FaultSeverity::Warning);
    assert!(FaultSeverity::Warning > FaultSeverity::Info);
}

/// Active faults list contains all reported faults.
#[test]
fn active_faults_list() {
    let mut handler = FaultHandler::<16>::new();
    for i in 0..5u16 {
        handler.report(make_fault(i, "fault", FaultSeverity::Warning));
    }
    assert_eq!(handler.active_faults().len(), 5);
}
