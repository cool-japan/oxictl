//! EtherCAT Distributed Clock (DC) synchronization.
//!
//! DC provides sub-microsecond synchronization across all EtherCAT slaves.
//! The master uses the first DC-capable slave as reference clock.
//!
//! Key concepts:
//!   - System time: 64-bit ns counter from reference slave
//!   - Propagation delay: cable delay compensation per slave
//!   - Sync0 pulse: interrupt generated at SYNC_CYCLE intervals

/// DC synchronization state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcState {
    /// DC not enabled.
    Disabled,
    /// Drift correction in progress.
    Locking,
    /// Locked — synchronized within tolerance.
    Locked,
    /// Lost sync (drift exceeded threshold).
    LostSync,
}

/// Distributed Clock configuration for one slave.
#[derive(Debug, Clone, Copy)]
pub struct DcConfig {
    /// Sync0 cycle time (ns).
    pub sync0_cycle_ns: u32,
    /// Sync0 pulse width (ns). 0 = single-shot.
    pub sync0_pulse_ns: u32,
    /// Sync1 cycle time (ns). 0 = disabled.
    pub sync1_cycle_ns: u32,
    /// Synchronization enabled.
    pub enabled: bool,
}

impl DcConfig {
    /// 1 kHz sync (1 ms cycle).
    pub fn at_1khz() -> Self {
        Self {
            sync0_cycle_ns: 1_000_000,
            sync0_pulse_ns: 500_000,
            sync1_cycle_ns: 0,
            enabled: true,
        }
    }

    /// 4 kHz sync (250 µs cycle).
    pub fn at_4khz() -> Self {
        Self {
            sync0_cycle_ns: 250_000,
            sync0_pulse_ns: 125_000,
            sync1_cycle_ns: 0,
            enabled: true,
        }
    }
}

/// DC time synchronizer for the EtherCAT master.
///
/// Tracks reference time and computes drift correction for each cycle.
#[derive(Debug, Clone, Copy)]
pub struct DcSynchronizer {
    /// Reference clock (system time, ns).
    pub system_time_ns: u64,
    /// Cycle time (ns).
    pub cycle_ns: u64,
    /// Maximum allowed drift (ns) before `LostSync`.
    pub max_drift_ns: u64,
    /// Current drift (ns).
    pub drift_ns: i64,
    /// PI controller for drift correction.
    drift_integrator: i64,
    /// State.
    pub state: DcState,
    /// Cycles since lock.
    locked_cycles: u32,
}

impl DcSynchronizer {
    pub fn new(cycle_ns: u64, max_drift_ns: u64) -> Self {
        Self {
            system_time_ns: 0,
            cycle_ns,
            max_drift_ns,
            drift_ns: 0,
            drift_integrator: 0,
            state: DcState::Disabled,
            locked_cycles: 0,
        }
    }

    /// Enable DC synchronization.
    pub fn enable(&mut self) {
        self.state = DcState::Locking;
        self.locked_cycles = 0;
    }

    /// Update with measured slave reference time.
    ///
    /// `ref_time_ns`: 64-bit system time from reference slave.
    /// Returns the correction offset (ns) to apply.
    pub fn update(&mut self, ref_time_ns: u64) -> i64 {
        let expected = self.system_time_ns + self.cycle_ns;
        self.drift_ns = ref_time_ns as i64 - expected as i64;
        self.system_time_ns = ref_time_ns;

        // PI drift correction
        let kp = 64i64;
        let ki = 1i64;
        self.drift_integrator = self.drift_integrator.saturating_add(self.drift_ns * ki);
        let correction = (self.drift_ns * kp + self.drift_integrator).saturating_div(1024);

        // State machine
        match self.state {
            DcState::Disabled => {}
            DcState::Locking => {
                if self.drift_ns.unsigned_abs() < self.max_drift_ns {
                    self.locked_cycles += 1;
                    if self.locked_cycles >= 10 {
                        self.state = DcState::Locked;
                    }
                } else {
                    self.locked_cycles = 0;
                }
            }
            DcState::Locked => {
                if self.drift_ns.unsigned_abs() > self.max_drift_ns * 4 {
                    self.state = DcState::LostSync;
                    self.locked_cycles = 0;
                }
            }
            DcState::LostSync => {
                if self.drift_ns.unsigned_abs() < self.max_drift_ns {
                    self.state = DcState::Locking;
                    self.locked_cycles = 0;
                }
            }
        }

        -correction
    }

    pub fn is_synchronized(&self) -> bool {
        self.state == DcState::Locked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_config_1khz() {
        let cfg = DcConfig::at_1khz();
        assert_eq!(cfg.sync0_cycle_ns, 1_000_000);
        assert!(cfg.enabled);
    }

    #[test]
    fn dc_synchronizer_locks_on_perfect_timing() {
        let mut dc = DcSynchronizer::new(1_000_000, 1_000);
        dc.enable();

        let mut t = 0u64;
        for _ in 0..20 {
            t += 1_000_000;
            dc.update(t);
        }

        assert_eq!(dc.state, DcState::Locked);
    }

    #[test]
    fn dc_synchronizer_lost_sync_on_large_drift() {
        let mut dc = DcSynchronizer::new(1_000_000, 1_000);
        dc.enable();

        // First lock
        let mut t = 0u64;
        for _ in 0..15 {
            t += 1_000_000;
            dc.update(t);
        }
        assert_eq!(dc.state, DcState::Locked);

        // Inject large drift
        dc.update(t + 1_000_000 + 10_000); // 10µs drift → > 4×1000ns

        assert_ne!(dc.state, DcState::Locked);
    }
}
