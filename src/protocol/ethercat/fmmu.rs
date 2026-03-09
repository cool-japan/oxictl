//! EtherCAT FMMU (Fieldbus Memory Management Unit) address translation.
//!
//! The FMMU maps logical addresses (used in the master's process image)
//! to physical addresses in the slave's ESC (EtherCAT Slave Controller) memory.
//!
//! Each slave can have up to 8 FMMU channels.

/// FMMU channel direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmmuDir {
    /// Read: slave → master (inputs/TxPDO).
    Read,
    /// Write: master → slave (outputs/RxPDO).
    Write,
}

/// FMMU translation table for a single bus.
///
/// Given a logical address and direction, resolves to (slave_address, physical_address).
#[derive(Debug, Clone, Copy)]
pub struct FmmuEntry {
    pub logical_start: u32,
    pub logical_end: u32, // exclusive
    pub physical_start: u16,
    pub slave_address: u16,
    pub dir: FmmuDir,
}

impl FmmuEntry {
    pub fn new(
        logical_start: u32,
        byte_length: u16,
        physical_start: u16,
        slave_address: u16,
        dir: FmmuDir,
    ) -> Self {
        Self {
            logical_start,
            logical_end: logical_start + byte_length as u32,
            physical_start,
            slave_address,
            dir,
        }
    }

    pub fn contains(&self, logical_addr: u32) -> bool {
        logical_addr >= self.logical_start && logical_addr < self.logical_end
    }
}

/// FMMU lookup table for the master.
#[derive(Debug)]
pub struct FmmuTable<const N: usize> {
    entries: [Option<FmmuEntry>; N],
    count: usize,
}

impl<const N: usize> FmmuTable<N> {
    pub fn new() -> Self {
        Self {
            entries: core::array::from_fn(|_| None),
            count: 0,
        }
    }

    /// Register an FMMU mapping.
    pub fn add(&mut self, entry: FmmuEntry) -> bool {
        if self.count >= N {
            return false;
        }
        self.entries[self.count] = Some(entry);
        self.count += 1;
        true
    }

    /// Translate logical address → (slave_address, physical_address).
    pub fn translate(&self, logical_addr: u32, dir: FmmuDir) -> Option<(u16, u16)> {
        for entry in self.entries[..self.count].iter().flatten() {
            if entry.dir == dir && entry.contains(logical_addr) {
                let offset = (logical_addr - entry.logical_start) as u16;
                return Some((entry.slave_address, entry.physical_start + offset));
            }
        }
        None
    }
}

impl<const N: usize> Default for FmmuTable<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmmu_translation() {
        let mut table = FmmuTable::<8>::new();
        // Slave 0: logical 0..4 → physical 0x1000 (read)
        table.add(FmmuEntry::new(0, 4, 0x1000, 0, FmmuDir::Read));
        // Slave 1: logical 4..8 → physical 0x1000 (read)
        table.add(FmmuEntry::new(4, 4, 0x1000, 1, FmmuDir::Read));

        let (slave, phys) = table.translate(0, FmmuDir::Read).unwrap();
        assert_eq!(slave, 0);
        assert_eq!(phys, 0x1000);

        let (slave, phys) = table.translate(6, FmmuDir::Read).unwrap();
        assert_eq!(slave, 1);
        assert_eq!(phys, 0x1002); // offset 2 within slave 1's region
    }

    #[test]
    fn fmmu_direction_mismatch() {
        let mut table = FmmuTable::<4>::new();
        table.add(FmmuEntry::new(0, 4, 0x1000, 0, FmmuDir::Read));
        assert!(table.translate(0, FmmuDir::Write).is_none());
    }

    #[test]
    fn fmmu_out_of_range() {
        let mut table = FmmuTable::<4>::new();
        table.add(FmmuEntry::new(0, 4, 0x1000, 0, FmmuDir::Read));
        assert!(table.translate(4, FmmuDir::Read).is_none()); // exclusive end
    }
}
