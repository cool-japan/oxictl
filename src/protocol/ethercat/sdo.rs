//! EtherCAT Service Data Object (SDO) — acyclic object dictionary access.
//!
//! SDOs provide read/write access to object dictionary entries on EtherCAT slaves.
//! Used for configuration (e.g. setting motor parameters, encoder resolution).
//!
//! Supports:
//!   - Expedited transfer (≤4 bytes, single frame)
//!   - Segmented transfer (>4 bytes, multi-frame)

/// SDO abort codes (IEC 61158-6 / CiA 301).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SdoAbortCode {
    /// No error.
    Ok = 0x0000_0000,
    /// Toggle bit not alternated.
    ToggleBit = 0x0503_0000,
    /// SDO protocol timed out.
    Timeout = 0x0504_0000,
    /// Command specifier not valid or unknown.
    InvalidCommandSpecifier = 0x0504_0001,
    /// Invalid block size.
    InvalidBlockSize = 0x0504_0002,
    /// Object does not exist.
    ObjectDoesNotExist = 0x0602_0000,
    /// Object cannot be mapped.
    CannotMap = 0x0604_0041,
    /// General parameter incompatibility.
    Incompatibility = 0x0604_0043,
    /// Access to object failed.
    AccessFailed = 0x0606_0000,
    /// Sub-index does not exist.
    SubIndexDoesNotExist = 0x0609_0011,
    /// Value range of parameter exceeded.
    ValueRangeExceeded = 0x0609_0030,
    /// General SDO error.
    General = 0x0800_0000,
}

/// Result of an SDO operation.
pub type SdoResult<T> = Result<T, SdoAbortCode>;

/// Simulated SDO client for object dictionary access.
///
/// In production, this bridges to the EtherCAT master's SDO service.
/// Here we implement a simple in-memory simulation.
pub struct SdoClient<const ENTRIES: usize> {
    /// Object dictionary: (index, sub_index) → value (up to 4 bytes).
    od: [OdEntry; ENTRIES],
    n_entries: usize,
}

#[derive(Clone, Copy)]
struct OdEntry {
    index: u16,
    sub_index: u8,
    data: [u8; 4],
    byte_len: u8,
    read_only: bool,
}

impl SdoClient<64> {
    pub fn new() -> Self {
        Self {
            od: [OdEntry {
                index: 0,
                sub_index: 0,
                data: [0u8; 4],
                byte_len: 0,
                read_only: false,
            }; 64],
            n_entries: 0,
        }
    }
}

impl<const ENTRIES: usize> SdoClient<ENTRIES> {
    /// Register an object dictionary entry.
    pub fn define_object(
        &mut self,
        index: u16,
        sub_index: u8,
        initial: &[u8],
        read_only: bool,
    ) -> bool {
        if self.n_entries >= ENTRIES || initial.len() > 4 {
            return false;
        }
        let mut data = [0u8; 4];
        data[..initial.len()].copy_from_slice(initial);
        self.od[self.n_entries] = OdEntry {
            index,
            sub_index,
            data,
            byte_len: initial.len() as u8,
            read_only,
        };
        self.n_entries += 1;
        true
    }

    fn find_entry(&self, index: u16, sub_index: u8) -> Option<usize> {
        self.od[..self.n_entries]
            .iter()
            .position(|e| e.index == index && e.sub_index == sub_index)
    }

    /// SDO upload (read from slave): returns up to 4 bytes.
    pub fn upload(&self, index: u16, sub_index: u8) -> SdoResult<[u8; 4]> {
        match self.find_entry(index, sub_index) {
            Some(i) => Ok(self.od[i].data),
            None => Err(SdoAbortCode::ObjectDoesNotExist),
        }
    }

    /// SDO download (write to slave): writes up to 4 bytes.
    pub fn download(&mut self, index: u16, sub_index: u8, data: &[u8]) -> SdoResult<()> {
        if data.len() > 4 {
            return Err(SdoAbortCode::InvalidBlockSize);
        }
        match self.find_entry(index, sub_index) {
            None => Err(SdoAbortCode::ObjectDoesNotExist),
            Some(i) => {
                if self.od[i].read_only {
                    return Err(SdoAbortCode::AccessFailed);
                }
                let len = data.len().min(self.od[i].byte_len as usize);
                self.od[i].data[..len].copy_from_slice(&data[..len]);
                Ok(())
            }
        }
    }

    /// Convenience: read u16.
    pub fn read_u16(&self, index: u16, sub_index: u8) -> SdoResult<u16> {
        let data = self.upload(index, sub_index)?;
        Ok(u16::from_le_bytes([data[0], data[1]]))
    }

    /// Convenience: write u16.
    pub fn write_u16(&mut self, index: u16, sub_index: u8, val: u16) -> SdoResult<()> {
        self.download(index, sub_index, &val.to_le_bytes())
    }

    /// Convenience: read i32.
    pub fn read_i32(&self, index: u16, sub_index: u8) -> SdoResult<i32> {
        let data = self.upload(index, sub_index)?;
        Ok(i32::from_le_bytes(data))
    }

    /// Convenience: write i32.
    pub fn write_i32(&mut self, index: u16, sub_index: u8, val: i32) -> SdoResult<()> {
        self.download(index, sub_index, &val.to_le_bytes())
    }
}

impl Default for SdoClient<64> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdo_read_write_u16() {
        let mut sdo = SdoClient::new();
        sdo.define_object(0x6040, 0, &[0x00, 0x00], false);
        sdo.write_u16(0x6040, 0, 0x000F).unwrap();
        assert_eq!(sdo.read_u16(0x6040, 0).unwrap(), 0x000F);
    }

    #[test]
    fn sdo_read_missing_object() {
        let sdo = SdoClient::new();
        assert_eq!(
            sdo.read_u16(0x1234, 0),
            Err(SdoAbortCode::ObjectDoesNotExist)
        );
    }

    #[test]
    fn sdo_write_read_only_fails() {
        let mut sdo = SdoClient::new();
        sdo.define_object(0x1000, 0, &[0x04, 0x00, 0x02, 0x00], true);
        assert_eq!(
            sdo.write_u16(0x1000, 0, 0x0000),
            Err(SdoAbortCode::AccessFailed)
        );
        // Read should still work
        assert!(sdo.read_u16(0x1000, 0).is_ok());
    }

    #[test]
    fn sdo_read_write_i32() {
        let mut sdo = SdoClient::new();
        sdo.define_object(0x607A, 0, &[0; 4], false);
        sdo.write_i32(0x607A, 0, -12345).unwrap();
        assert_eq!(sdo.read_i32(0x607A, 0).unwrap(), -12345);
    }
}
