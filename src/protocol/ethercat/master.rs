//! EtherCAT Master — bus management and cyclic data exchange.
//!
//! This is a software model/stub of an EtherCAT master.
//! A real implementation requires kernel-space NIC access (IgH EtherLab, SOEM, etc.).
//!
//! This module provides:
//!   - State machine for bus bring-up (INIT → PREOP → SAFEOP → OP)
//!   - Slave registry with PDO configuration
//!   - Cyclic exchange timing model

/// EtherCAT slave state (CoE / EtherCAT state machine).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlState {
    /// Slave booting up.
    Init = 0x01,
    /// Pre-operational: SDO communication allowed.
    PreOp = 0x02,
    /// Safe-operational: TxPDOs active, RxPDOs zeroed.
    SafeOp = 0x04,
    /// Operational: all PDOs active.
    Op = 0x08,
}

/// Slave configuration descriptor.
#[derive(Debug, Clone, Copy)]
pub struct SlaveConfig {
    /// EtherCAT address (position-addressed, 0-indexed).
    pub address: u16,
    /// Vendor ID.
    pub vendor_id: u32,
    /// Product code.
    pub product_code: u32,
    /// Current state.
    pub state: AlState,
    /// PDO input size (bytes) — data from slave to master.
    pub input_bytes: usize,
    /// PDO output size (bytes) — data from master to slave.
    pub output_bytes: usize,
}

impl SlaveConfig {
    pub fn new(address: u16, vendor_id: u32, product_code: u32) -> Self {
        Self {
            address,
            vendor_id,
            product_code,
            state: AlState::Init,
            input_bytes: 0,
            output_bytes: 0,
        }
    }
}

/// EtherCAT master state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MasterState {
    /// Not initialized.
    Idle,
    /// Scanning for slaves.
    Scanning,
    /// All slaves in PreOp, PDOs being configured.
    Configuring,
    /// Running — cyclic exchange active.
    Running,
    /// Bus error detected.
    Error,
}

/// EtherCAT master (software model).
///
/// `MAX_SLAVES` = maximum number of slaves on the bus.
pub struct EtherCatMaster<const MAX_SLAVES: usize> {
    pub slaves: [Option<SlaveConfig>; MAX_SLAVES],
    pub n_slaves: usize,
    pub state: MasterState,
    /// Cyclic counter (increments each exchange cycle).
    pub cycle_count: u64,
    /// Working counter: number of slaves that responded.
    pub working_counter: u16,
}

impl<const MAX_SLAVES: usize> EtherCatMaster<MAX_SLAVES> {
    pub fn new() -> Self {
        Self {
            slaves: core::array::from_fn(|_| None),
            n_slaves: 0,
            state: MasterState::Idle,
            cycle_count: 0,
            working_counter: 0,
        }
    }

    /// Register a slave. Returns slot index or None if full.
    pub fn add_slave(&mut self, config: SlaveConfig) -> Option<usize> {
        if self.n_slaves >= MAX_SLAVES {
            return None;
        }
        let idx = self.n_slaves;
        self.slaves[idx] = Some(config);
        self.n_slaves += 1;
        Some(idx)
    }

    /// Simulate bus bring-up: transition all slaves to target state.
    pub fn request_state(&mut self, target: AlState) -> bool {
        for slave in self.slaves[..self.n_slaves].iter_mut().flatten() {
            slave.state = target;
        }
        self.state = if target == AlState::Op {
            MasterState::Running
        } else {
            MasterState::Configuring
        };
        true
    }

    /// Simulate one cyclic exchange cycle.
    /// Returns working counter (should equal n_slaves for healthy bus).
    pub fn exchange(&mut self) -> u16 {
        if self.state != MasterState::Running {
            return 0;
        }
        self.cycle_count += 1;
        // In a real implementation: send Ethernet frame, receive responses
        // Here: simulate all slaves responding
        self.working_counter = self.n_slaves as u16;
        self.working_counter
    }

    pub fn is_running(&self) -> bool {
        self.state == MasterState::Running
    }

    /// Get slave by address.
    pub fn slave(&self, address: u16) -> Option<&SlaveConfig> {
        self.slaves[..self.n_slaves]
            .iter()
            .filter_map(|s| s.as_ref())
            .find(|s| s.address == address)
    }
}

impl<const MAX_SLAVES: usize> Default for EtherCatMaster<MAX_SLAVES> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn master_add_slaves() {
        let mut master = EtherCatMaster::<8>::new();
        master.add_slave(SlaveConfig::new(0, 0x000022D2, 0x00000001));
        master.add_slave(SlaveConfig::new(1, 0x000022D2, 0x00000002));
        assert_eq!(master.n_slaves, 2);
    }

    #[test]
    fn master_bring_up() {
        let mut master = EtherCatMaster::<4>::new();
        master.add_slave(SlaveConfig::new(0, 0x1234, 0x0001));
        master.add_slave(SlaveConfig::new(1, 0x1234, 0x0002));

        assert!(master.request_state(AlState::Op));
        assert_eq!(master.state, MasterState::Running);
        assert_eq!(master.slaves[0].unwrap().state, AlState::Op);
    }

    #[test]
    fn master_cyclic_exchange() {
        let mut master = EtherCatMaster::<4>::new();
        master.add_slave(SlaveConfig::new(0, 0x1234, 0x0001));
        master.request_state(AlState::Op);

        let wc = master.exchange();
        assert_eq!(wc, 1);
        assert_eq!(master.cycle_count, 1);
    }

    #[test]
    fn master_not_running_exchange_returns_zero() {
        let mut master = EtherCatMaster::<2>::new();
        let wc = master.exchange();
        assert_eq!(wc, 0);
    }

    #[test]
    fn master_find_slave_by_address() {
        let mut master = EtherCatMaster::<4>::new();
        master.add_slave(SlaveConfig::new(3, 0xABCD, 0x1234));
        let s = master.slave(3).unwrap();
        assert_eq!(s.vendor_id, 0xABCD);
    }
}
