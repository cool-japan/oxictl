//! EtherCAT Process Data Object (PDO) mapping.
//!
//! PDOs carry real-time cyclic process data between master and slaves.
//! Each PDO maps a set of object dictionary entries to the process image.
//!
//! - RxPDO: slave receives data from master (outputs)
//! - TxPDO: slave transmits data to master (inputs)

/// PDO entry: maps one object dictionary variable to a PDO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdoEntry {
    /// Object index in the object dictionary.
    pub index: u16,
    /// Sub-index within the object.
    pub sub_index: u8,
    /// Bit length of the mapped variable.
    pub bit_length: u8,
}

impl PdoEntry {
    pub const fn new(index: u16, sub_index: u8, bit_length: u8) -> Self {
        Self {
            index,
            sub_index,
            bit_length,
        }
    }

    /// Byte length (rounded up).
    pub const fn byte_length(&self) -> usize {
        self.bit_length.div_ceil(8) as usize
    }
}

/// PDO mapping: a named PDO with up to N entries.
#[derive(Debug, Clone)]
pub struct PdoMapping<const N: usize> {
    /// PDO sync manager assignment index (e.g. 0x1A00 for TxPDO 1).
    pub pdo_index: u16,
    /// Entries in this PDO.
    pub entries: [Option<PdoEntry>; N],
    pub entry_count: usize,
}

impl<const N: usize> PdoMapping<N> {
    pub fn new(pdo_index: u16) -> Self {
        Self {
            pdo_index,
            entries: [None; N],
            entry_count: 0,
        }
    }

    /// Add an entry to the PDO mapping. Returns false if full.
    pub fn add_entry(&mut self, entry: PdoEntry) -> bool {
        if self.entry_count >= N {
            return false;
        }
        self.entries[self.entry_count] = Some(entry);
        self.entry_count += 1;
        true
    }

    /// Total byte length of this PDO.
    pub fn total_bytes(&self) -> usize {
        self.entries[..self.entry_count]
            .iter()
            .filter_map(|e| e.as_ref())
            .map(|e| e.byte_length())
            .sum()
    }
}

/// Process image: the shared memory area for cyclic PDO data.
///
/// Offset in the image is determined by the master's PDO mapping.
#[derive(Debug)]
pub struct ProcessImage<const SIZE: usize> {
    pub data: [u8; SIZE],
}

impl<const SIZE: usize> ProcessImage<SIZE> {
    pub fn new() -> Self {
        Self { data: [0u8; SIZE] }
    }

    /// Read a u16 value at byte offset.
    pub fn read_u16(&self, offset: usize) -> Option<u16> {
        if offset + 1 < SIZE {
            Some(u16::from_le_bytes([
                self.data[offset],
                self.data[offset + 1],
            ]))
        } else {
            None
        }
    }

    /// Write a u16 value at byte offset.
    pub fn write_u16(&mut self, offset: usize, val: u16) -> bool {
        if offset + 1 < SIZE {
            let bytes = val.to_le_bytes();
            self.data[offset] = bytes[0];
            self.data[offset + 1] = bytes[1];
            true
        } else {
            false
        }
    }

    /// Read a u32 value at byte offset.
    pub fn read_u32(&self, offset: usize) -> Option<u32> {
        if offset + 3 < SIZE {
            Some(u32::from_le_bytes([
                self.data[offset],
                self.data[offset + 1],
                self.data[offset + 2],
                self.data[offset + 3],
            ]))
        } else {
            None
        }
    }

    /// Write a u32 value at byte offset.
    pub fn write_u32(&mut self, offset: usize, val: u32) -> bool {
        if offset + 3 < SIZE {
            let bytes = val.to_le_bytes();
            self.data[offset..offset + 4].copy_from_slice(&bytes);
            true
        } else {
            false
        }
    }
}

impl<const SIZE: usize> Default for ProcessImage<SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdo_entry_byte_length() {
        let e = PdoEntry::new(0x6040, 0, 16);
        assert_eq!(e.byte_length(), 2);
        let e2 = PdoEntry::new(0x6040, 0, 1);
        assert_eq!(e2.byte_length(), 1);
    }

    #[test]
    fn pdo_mapping_total_bytes() {
        let mut pdo = PdoMapping::<4>::new(0x1600);
        pdo.add_entry(PdoEntry::new(0x6040, 0, 16)); // 2 bytes
        pdo.add_entry(PdoEntry::new(0x607A, 0, 32)); // 4 bytes
        assert_eq!(pdo.total_bytes(), 6);
    }

    #[test]
    fn pdo_mapping_full() {
        let mut pdo = PdoMapping::<2>::new(0x1600);
        assert!(pdo.add_entry(PdoEntry::new(0x6040, 0, 16)));
        assert!(pdo.add_entry(PdoEntry::new(0x607A, 0, 32)));
        assert!(!pdo.add_entry(PdoEntry::new(0x6060, 0, 8))); // full
    }

    #[test]
    fn process_image_rw_u16() {
        let mut pi = ProcessImage::<8>::new();
        pi.write_u16(0, 0x1234);
        assert_eq!(pi.read_u16(0), Some(0x1234));
    }

    #[test]
    fn process_image_rw_u32() {
        let mut pi = ProcessImage::<8>::new();
        pi.write_u32(0, 0xDEADBEEF);
        assert_eq!(pi.read_u32(0), Some(0xDEADBEEF));
    }
}
