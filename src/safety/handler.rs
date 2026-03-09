use crate::safety::fault::{FaultEvent, FaultResponse, FaultSeverity};
use heapless::Vec as HVec;

/// Fault handler that collects fault events and determines the system response.
/// Uses a fixed-size buffer (no heap allocation).
#[derive(Debug)]
pub struct FaultHandler<const MAX_FAULTS: usize> {
    active_faults: HVec<FaultEvent, MAX_FAULTS>,
    highest_severity: Option<FaultSeverity>,
}

impl<const MAX_FAULTS: usize> FaultHandler<MAX_FAULTS> {
    pub fn new() -> Self {
        Self {
            active_faults: HVec::new(),
            highest_severity: None,
        }
    }

    /// Report a fault. Returns the response action.
    pub fn report(&mut self, event: FaultEvent) -> FaultResponse {
        let response = event.response;

        // Update highest severity
        self.highest_severity = Some(match self.highest_severity {
            Some(current) if current >= event.severity => current,
            _ => event.severity,
        });

        // Store if space available
        let _ = self.active_faults.push(event);

        response
    }

    /// Get the current recommended system response based on all active faults.
    pub fn current_response(&self) -> FaultResponse {
        match self.highest_severity {
            None => FaultResponse::LogAndContinue,
            Some(sev) => sev.default_response(),
        }
    }

    /// Number of active faults.
    pub fn fault_count(&self) -> usize {
        self.active_faults.len()
    }

    /// Whether any critical fault is active.
    pub fn has_critical(&self) -> bool {
        self.highest_severity == Some(FaultSeverity::Critical)
    }

    /// Clear all faults (acknowledge/reset).
    pub fn clear(&mut self) {
        self.active_faults.clear();
        self.highest_severity = None;
    }

    /// Get active faults as a slice.
    pub fn active_faults(&self) -> &[FaultEvent] {
        &self.active_faults
    }
}

impl<const MAX_FAULTS: usize> Default for FaultHandler<MAX_FAULTS> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safety::fault::{FaultDef, FaultEvent};

    fn make_fault(id: u16, severity: FaultSeverity) -> FaultEvent {
        let def = FaultDef {
            id,
            name: "test",
            severity,
            response: severity.default_response(),
        };
        FaultEvent::new(&def, 0.0)
    }

    #[test]
    fn empty_handler_no_faults() {
        let handler = FaultHandler::<16>::new();
        assert_eq!(handler.fault_count(), 0);
        assert!(!handler.has_critical());
        assert_eq!(handler.current_response(), FaultResponse::LogAndContinue);
    }

    #[test]
    fn report_warning() {
        let mut handler = FaultHandler::<16>::new();
        let resp = handler.report(make_fault(1, FaultSeverity::Warning));
        assert_eq!(resp, FaultResponse::LogAndContinue);
        assert_eq!(handler.fault_count(), 1);
    }

    #[test]
    fn report_critical_escalates() {
        let mut handler = FaultHandler::<16>::new();
        handler.report(make_fault(1, FaultSeverity::Warning));
        handler.report(make_fault(2, FaultSeverity::Critical));
        assert!(handler.has_critical());
        assert_eq!(handler.current_response(), FaultResponse::EmergencyStop);
    }

    #[test]
    fn clear_resets_handler() {
        let mut handler = FaultHandler::<16>::new();
        handler.report(make_fault(1, FaultSeverity::Critical));
        assert!(handler.has_critical());
        handler.clear();
        assert_eq!(handler.fault_count(), 0);
        assert!(!handler.has_critical());
    }

    #[test]
    fn buffer_overflow_handled() {
        let mut handler = FaultHandler::<2>::new();
        handler.report(make_fault(1, FaultSeverity::Info));
        handler.report(make_fault(2, FaultSeverity::Info));
        // Third fault: buffer full, but severity still tracked
        handler.report(make_fault(3, FaultSeverity::Critical));
        assert!(handler.has_critical());
        assert_eq!(handler.fault_count(), 2); // buffer maxed
    }
}
