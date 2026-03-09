/// Severity level of a fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FaultSeverity {
    /// Informational only, no action required.
    Info,
    /// Warning: log and continue.
    Warning,
    /// Error: attempt graceful degradation.
    Error,
    /// Critical: immediate safe shutdown.
    Critical,
}

/// What action to take when a fault occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultResponse {
    /// Log the fault and continue normal operation.
    LogAndContinue,
    /// Switch to a degraded/safe output.
    Degrade,
    /// Emergency stop: set output to zero.
    EmergencyStop,
}

impl FaultSeverity {
    /// Default response for a given severity.
    pub fn default_response(self) -> FaultResponse {
        match self {
            FaultSeverity::Info | FaultSeverity::Warning => FaultResponse::LogAndContinue,
            FaultSeverity::Error => FaultResponse::Degrade,
            FaultSeverity::Critical => FaultResponse::EmergencyStop,
        }
    }
}

/// A fault definition: describes what kind of fault can occur.
#[derive(Debug, Clone)]
pub struct FaultDef {
    pub id: u16,
    pub name: &'static str,
    pub severity: FaultSeverity,
    pub response: FaultResponse,
}

/// A recorded fault event with timestamp info.
#[derive(Debug, Clone, Copy)]
pub struct FaultEvent {
    pub fault_id: u16,
    pub severity: FaultSeverity,
    pub response: FaultResponse,
    /// Monotonic timestamp (seconds) when the fault occurred.
    pub timestamp: f64,
}

impl FaultEvent {
    pub fn new(def: &FaultDef, timestamp: f64) -> Self {
        Self {
            fault_id: def.id,
            severity: def.severity,
            response: def.response,
            timestamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_ordering() {
        assert!(FaultSeverity::Info < FaultSeverity::Warning);
        assert!(FaultSeverity::Warning < FaultSeverity::Error);
        assert!(FaultSeverity::Error < FaultSeverity::Critical);
    }

    #[test]
    fn default_responses() {
        assert_eq!(
            FaultSeverity::Info.default_response(),
            FaultResponse::LogAndContinue
        );
        assert_eq!(
            FaultSeverity::Warning.default_response(),
            FaultResponse::LogAndContinue
        );
        assert_eq!(
            FaultSeverity::Error.default_response(),
            FaultResponse::Degrade
        );
        assert_eq!(
            FaultSeverity::Critical.default_response(),
            FaultResponse::EmergencyStop
        );
    }

    #[test]
    fn fault_event_from_def() {
        let def = FaultDef {
            id: 42,
            name: "overtemp",
            severity: FaultSeverity::Critical,
            response: FaultResponse::EmergencyStop,
        };
        let event = FaultEvent::new(&def, 1.5);
        assert_eq!(event.fault_id, 42);
        assert_eq!(event.severity, FaultSeverity::Critical);
        assert_eq!(event.response, FaultResponse::EmergencyStop);
        assert_eq!(event.timestamp, 1.5);
    }
}
