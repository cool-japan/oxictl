//! EtherCAT slave profile and FMMU configuration.
//!
//! Slave-side state machine, object dictionary access points,
//! and Fieldbus Memory Management Unit (FMMU) configuration.

use super::master::AlState;

/// FMMU (Fieldbus Memory Management Unit) mapping.
///
/// Maps logical process image addresses to slave's physical memory.
#[derive(Debug, Clone, Copy)]
pub struct FmmuMapping {
    /// Logical start address in the process image.
    pub logical_start: u32,
    /// Byte length of the mapping.
    pub length: u16,
    /// Physical start address in slave memory.
    pub physical_start: u16,
    /// Enable read (input from slave).
    pub read_enable: bool,
    /// Enable write (output to slave).
    pub write_enable: bool,
}

/// EtherCAT slave device model.
///
/// Represents one EtherCAT slave participating in the bus.
#[derive(Debug, Clone, Copy)]
pub struct EtherCatSlave {
    /// Configured station address.
    pub address: u16,
    /// Current AL state.
    pub state: AlState,
    /// Vendor ID (from ESI/EEPROM).
    pub vendor_id: u32,
    /// Product code.
    pub product_code: u32,
    /// Revision number.
    pub revision: u32,
    /// FMMU channels (up to 8 per slave per spec).
    pub fmmus: [Option<FmmuMapping>; 8],
    pub fmmu_count: usize,
    /// Error counter (increments on CRC errors).
    pub error_count: u16,
}

impl EtherCatSlave {
    pub fn new(address: u16, vendor_id: u32, product_code: u32, revision: u32) -> Self {
        Self {
            address,
            state: AlState::Init,
            vendor_id,
            product_code,
            revision,
            fmmus: [None; 8],
            fmmu_count: 0,
            error_count: 0,
        }
    }

    /// Add FMMU mapping. Returns false if all 8 channels used.
    pub fn add_fmmu(&mut self, mapping: FmmuMapping) -> bool {
        if self.fmmu_count >= 8 {
            return false;
        }
        self.fmmus[self.fmmu_count] = Some(mapping);
        self.fmmu_count += 1;
        true
    }

    /// Transition to requested state (simplified — no error checking).
    pub fn request_state(&mut self, target: AlState) -> bool {
        // Simplified: allow all valid forward transitions
        let allowed = matches!(
            (self.state, target),
            (AlState::Init, AlState::PreOp)
                | (AlState::PreOp, AlState::SafeOp)
                | (AlState::SafeOp, AlState::Op)
                | (AlState::Op, AlState::SafeOp)
                | (AlState::SafeOp, AlState::PreOp)
                | (AlState::PreOp, AlState::Init)
        );
        if allowed {
            self.state = target;
        }
        allowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slave_state_transitions() {
        let mut slave = EtherCatSlave::new(0, 0x22D2, 0x0001, 0x0001);
        assert!(slave.request_state(AlState::PreOp));
        assert_eq!(slave.state, AlState::PreOp);
        assert!(slave.request_state(AlState::SafeOp));
        assert!(slave.request_state(AlState::Op));
        assert_eq!(slave.state, AlState::Op);
    }

    #[test]
    fn slave_invalid_transition() {
        let mut slave = EtherCatSlave::new(0, 0x22D2, 0x0001, 0x0001);
        assert!(!slave.request_state(AlState::Op)); // Init → Op not allowed
        assert_eq!(slave.state, AlState::Init);
    }

    #[test]
    fn slave_fmmu_mapping() {
        let mut slave = EtherCatSlave::new(0, 0x22D2, 0x0001, 0x0001);
        let fmmu = FmmuMapping {
            logical_start: 0,
            length: 4,
            physical_start: 0x1000,
            read_enable: true,
            write_enable: false,
        };
        assert!(slave.add_fmmu(fmmu));
        assert_eq!(slave.fmmu_count, 1);
    }

    #[test]
    fn slave_fmmu_max_8() {
        let mut slave = EtherCatSlave::new(0, 0x22D2, 0x0001, 0x0001);
        for _ in 0..8 {
            slave.add_fmmu(FmmuMapping {
                logical_start: 0,
                length: 1,
                physical_start: 0,
                read_enable: true,
                write_enable: false,
            });
        }
        assert!(!slave.add_fmmu(FmmuMapping {
            logical_start: 0,
            length: 1,
            physical_start: 0,
            read_enable: true,
            write_enable: false,
        }));
    }
}
