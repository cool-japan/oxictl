//! CANopen Object Dictionary (OD) implementation.
//!
//! Provides a fixed-capacity key-value store for CANopen object dictionary
//! entries indexed by (index, sub_index) pairs. Pre-populated with DS-301
//! mandatory objects.

use heapless::index_map::FnvIndexMap;

/// Object dictionary key: (index, sub_index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OdIndex(pub u16, pub u8);

/// Object dictionary value variants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OdValue {
    U8(u8),
    U16(u16),
    U32(u32),
    I16(i16),
    I32(i32),
    F32(f32),
    /// Fixed 8-byte opaque blob.
    Bytes([u8; 8]),
}

impl OdValue {
    /// Get as u8 if variant matches.
    pub fn as_u8(&self) -> Option<u8> {
        if let OdValue::U8(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Get as u16 if variant matches.
    pub fn as_u16(&self) -> Option<u16> {
        if let OdValue::U16(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Get as u32 if variant matches.
    pub fn as_u32(&self) -> Option<u32> {
        if let OdValue::U32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Get as i32 if variant matches.
    pub fn as_i32(&self) -> Option<i32> {
        if let OdValue::I32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Get as f32 if variant matches.
    pub fn as_f32(&self) -> Option<f32> {
        if let OdValue::F32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Get as byte array if variant matches.
    pub fn as_bytes(&self) -> Option<[u8; 8]> {
        if let OdValue::Bytes(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Size in bytes for this value type.
    pub fn byte_size(&self) -> usize {
        match self {
            OdValue::U8(_) => 1,
            OdValue::I16(_) | OdValue::U16(_) => 2,
            OdValue::U32(_) | OdValue::I32(_) | OdValue::F32(_) => 4,
            OdValue::Bytes(_) => 8,
        }
    }
}

/// Object dictionary error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdError {
    /// Entry does not exist (index not found).
    NotFound,
    /// The sub-index does not exist for a found index.
    SubindexNotFound,
    /// Dictionary is full (capacity exceeded).
    Full,
    /// Type mismatch for the entry.
    TypeMismatch,
    /// Entry is read-only; write access denied.
    ReadOnly,
    /// Entry is write-only; read access denied.
    WriteOnly,
    /// Access is completely denied.
    AccessDenied,
    /// Index not found (alias kept for compatibility with StaticOd usage).
    IndexNotFound,
}

/// Object dictionary with heapless backing store.
///
/// `N` is the maximum number of entries.
pub struct ObjectDict<const N: usize> {
    map: FnvIndexMap<OdIndex, OdValue, N>,
    /// Read-only flags stored separately (parallel array not practical with heapless map,
    /// so we use a simple scan approach with a fixed-size tracking list).
    ro_list: heapless::Vec<OdIndex, N>,
}

impl<const N: usize> ObjectDict<N> {
    /// Create an empty object dictionary.
    pub fn new() -> Self {
        Self {
            map: FnvIndexMap::new(),
            ro_list: heapless::Vec::new(),
        }
    }

    /// Create an object dictionary pre-populated with DS-301 mandatory objects.
    ///
    /// Mandatory entries:
    /// - 0x1000:00 — Device Type (VAR, UNSIGNED32)
    /// - 0x1001:00 — Error Register (VAR, UNSIGNED8)
    /// - 0x1017:00 — Producer Heartbeat Time (VAR, UNSIGNED16)
    /// - 0x1018:00 — Identity Object (RECORD, sub 0=count)
    /// - 0x1018:01 — Vendor ID
    /// - 0x1018:02 — Product Code
    /// - 0x1018:03 — Revision Number
    /// - 0x1018:04 — Serial Number
    pub fn with_ds301_defaults(
        vendor_id: u32,
        product_code: u32,
        revision: u32,
        serial: u32,
    ) -> Self {
        let mut od = Self::new();
        // 0x1000:00 Device Type
        let _ = od.define_ro(OdIndex(0x1000, 0), OdValue::U32(0x0000_0402));
        // 0x1001:00 Error Register
        let _ = od.define(OdIndex(0x1001, 0), OdValue::U8(0));
        // 0x1017:00 Producer Heartbeat Time (ms, 0=disabled)
        let _ = od.define(OdIndex(0x1017, 0), OdValue::U16(0));
        // 0x1018 Identity Object
        let _ = od.define_ro(OdIndex(0x1018, 0), OdValue::U8(4));
        let _ = od.define_ro(OdIndex(0x1018, 1), OdValue::U32(vendor_id));
        let _ = od.define_ro(OdIndex(0x1018, 2), OdValue::U32(product_code));
        let _ = od.define_ro(OdIndex(0x1018, 3), OdValue::U32(revision));
        let _ = od.define_ro(OdIndex(0x1018, 4), OdValue::U32(serial));
        od
    }

    /// Insert a read-write entry.
    pub fn define(&mut self, key: OdIndex, value: OdValue) -> Result<(), OdError> {
        self.map.insert(key, value).map_err(|_| OdError::Full)?;
        Ok(())
    }

    /// Insert a read-only entry.
    pub fn define_ro(&mut self, key: OdIndex, value: OdValue) -> Result<(), OdError> {
        self.map.insert(key, value).map_err(|_| OdError::Full)?;
        self.ro_list.push(key).map_err(|_| OdError::Full)?;
        Ok(())
    }

    /// Check if an entry exists.
    pub fn has(&self, idx: u16, sub: u8) -> bool {
        self.map.contains_key(&OdIndex(idx, sub))
    }

    /// Get a reference to an entry value.
    pub fn get(&self, idx: u16, sub: u8) -> Result<&OdValue, OdError> {
        self.map.get(&OdIndex(idx, sub)).ok_or(OdError::NotFound)
    }

    /// Set the value of an existing entry.
    pub fn set(&mut self, idx: u16, sub: u8, value: OdValue) -> Result<(), OdError> {
        let key = OdIndex(idx, sub);
        if self.ro_list.iter().any(|k| k == &key) {
            return Err(OdError::ReadOnly);
        }
        let entry = self.map.get_mut(&key).ok_or(OdError::NotFound)?;
        // Check type compatibility
        match (&*entry, &value) {
            (OdValue::U8(_), OdValue::U8(_))
            | (OdValue::U16(_), OdValue::U16(_))
            | (OdValue::U32(_), OdValue::U32(_))
            | (OdValue::I16(_), OdValue::I16(_))
            | (OdValue::I32(_), OdValue::I32(_))
            | (OdValue::F32(_), OdValue::F32(_))
            | (OdValue::Bytes(_), OdValue::Bytes(_)) => {
                *entry = value;
                Ok(())
            }
            _ => Err(OdError::TypeMismatch),
        }
    }

    /// Get u8 value directly.
    pub fn get_u8(&self, idx: u16, sub: u8) -> Result<u8, OdError> {
        self.get(idx, sub)?.as_u8().ok_or(OdError::TypeMismatch)
    }

    /// Get u16 value directly.
    pub fn get_u16(&self, idx: u16, sub: u8) -> Result<u16, OdError> {
        self.get(idx, sub)?.as_u16().ok_or(OdError::TypeMismatch)
    }

    /// Get u32 value directly.
    pub fn get_u32(&self, idx: u16, sub: u8) -> Result<u32, OdError> {
        self.get(idx, sub)?.as_u32().ok_or(OdError::TypeMismatch)
    }

    /// Get i32 value directly.
    pub fn get_i32(&self, idx: u16, sub: u8) -> Result<i32, OdError> {
        self.get(idx, sub)?.as_i32().ok_or(OdError::TypeMismatch)
    }

    /// Get f32 value directly.
    pub fn get_f32(&self, idx: u16, sub: u8) -> Result<f32, OdError> {
        self.get(idx, sub)?.as_f32().ok_or(OdError::TypeMismatch)
    }

    /// Set u8 value.
    pub fn set_u8(&mut self, idx: u16, sub: u8, val: u8) -> Result<(), OdError> {
        self.set(idx, sub, OdValue::U8(val))
    }

    /// Set u16 value.
    pub fn set_u16(&mut self, idx: u16, sub: u8, val: u16) -> Result<(), OdError> {
        self.set(idx, sub, OdValue::U16(val))
    }

    /// Set u32 value.
    pub fn set_u32(&mut self, idx: u16, sub: u8, val: u32) -> Result<(), OdError> {
        self.set(idx, sub, OdValue::U32(val))
    }

    /// Set i32 value.
    pub fn set_i32(&mut self, idx: u16, sub: u8, val: i32) -> Result<(), OdError> {
        self.set(idx, sub, OdValue::I32(val))
    }

    /// Set f32 value.
    pub fn set_f32(&mut self, idx: u16, sub: u8, val: f32) -> Result<(), OdError> {
        self.set(idx, sub, OdValue::F32(val))
    }

    /// Number of entries in the dictionary.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl<const N: usize> Default for ObjectDict<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── CiA 301 Data Types ───────────────────────────────────────────────────────

/// Standard CiA 301 object dictionary data types.
///
/// These match the type codes used in EDS files and the CiA 301 standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataType {
    /// Boolean (1 bit, stored as u8).
    Boolean = 0x01,
    /// 8-bit signed integer.
    Integer8 = 0x02,
    /// 16-bit signed integer.
    Integer16 = 0x03,
    /// 32-bit signed integer.
    Integer32 = 0x04,
    /// 8-bit unsigned integer.
    Unsigned8 = 0x05,
    /// 16-bit unsigned integer.
    Unsigned16 = 0x06,
    /// 32-bit unsigned integer.
    Unsigned32 = 0x07,
    /// Octet string (fixed 8 bytes in this implementation).
    OctetString = 0x0A,
    /// Visible string (ASCII, fixed 8 bytes in this implementation).
    VisibleString = 0x09,
}

// ─── Access Types ─────────────────────────────────────────────────────────────

/// Access permission for an OD entry (per CiA 301).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    /// Read-only.
    RO,
    /// Write-only.
    WO,
    /// Read-write.
    RW,
    /// Read-write on process input (read access, write by device).
    RWR,
    /// Read-write on process output (write access, read by device).
    RWW,
    /// Constant — read-only and semantically immutable.
    Const,
}

impl AccessType {
    /// Returns `true` if SDO read (upload) is permitted.
    pub fn can_read(self) -> bool {
        matches!(
            self,
            Self::RO | Self::RW | Self::RWR | Self::RWW | Self::Const
        )
    }

    /// Returns `true` if SDO write (download) is permitted.
    pub fn can_write(self) -> bool {
        matches!(self, Self::WO | Self::RW | Self::RWR | Self::RWW)
    }
}

// ─── OdEntry — rich, statically-typed OD entry ────────────────────────────────

/// Extended OD value enum, covering all CiA 301 base types.
///
/// This is separate from the lighter-weight `OdValue` used by `ObjectDict`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OdEntryValue {
    /// Boolean stored as u8 (0 = false, non-zero = true).
    Bool(bool),
    /// 8-bit signed integer.
    I8(i8),
    /// 16-bit signed integer.
    I16(i16),
    /// 32-bit signed integer.
    I32(i32),
    /// 8-bit unsigned integer.
    U8(u8),
    /// 16-bit unsigned integer.
    U16(u16),
    /// 32-bit unsigned integer.
    U32(u32),
    /// Fixed 8-byte octet / visible string.
    OctetString([u8; 8]),
}

impl OdEntryValue {
    /// Returns the `DataType` discriminant for this value.
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Bool(_) => DataType::Boolean,
            Self::I8(_) => DataType::Integer8,
            Self::I16(_) => DataType::Integer16,
            Self::I32(_) => DataType::Integer32,
            Self::U8(_) => DataType::Unsigned8,
            Self::U16(_) => DataType::Unsigned16,
            Self::U32(_) => DataType::Unsigned32,
            Self::OctetString(_) => DataType::OctetString,
        }
    }

    /// Size in bytes.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Bool(_) | Self::U8(_) | Self::I8(_) => 1,
            Self::U16(_) | Self::I16(_) => 2,
            Self::U32(_) | Self::I32(_) => 4,
            Self::OctetString(_) => 8,
        }
    }

    /// Encode to little-endian bytes; fills the returned array, zero-padded.
    pub fn to_le_bytes(&self) -> [u8; 8] {
        let mut out = [0u8; 8];
        match self {
            Self::Bool(b) => out[0] = *b as u8,
            Self::U8(v) => out[0] = *v,
            Self::I8(v) => out[0] = *v as u8,
            Self::U16(v) => out[..2].copy_from_slice(&v.to_le_bytes()),
            Self::I16(v) => out[..2].copy_from_slice(&v.to_le_bytes()),
            Self::U32(v) => out[..4].copy_from_slice(&v.to_le_bytes()),
            Self::I32(v) => out[..4].copy_from_slice(&v.to_le_bytes()),
            Self::OctetString(b) => out.copy_from_slice(b),
        }
        out
    }

    /// Check whether two values have compatible types (same discriminant).
    pub fn type_compatible(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}

/// A single, richly-typed Object Dictionary entry with access control.
#[derive(Debug, Clone, Copy)]
pub struct OdEntry {
    /// Object index (0x0000–0xFFFF).
    pub index: u16,
    /// Sub-index (0x00–0xFF).
    pub subindex: u8,
    /// CiA 301 data type.
    pub data_type: DataType,
    /// Access permission.
    pub access_type: AccessType,
    /// Current value.
    pub value: OdEntryValue,
}

impl OdEntry {
    /// Construct a new `OdEntry`.
    pub const fn new(
        index: u16,
        subindex: u8,
        data_type: DataType,
        access_type: AccessType,
        value: OdEntryValue,
    ) -> Self {
        Self {
            index,
            subindex,
            data_type,
            access_type,
            value,
        }
    }
}

// ─── StaticOd — fixed-capacity, heapless Object Dictionary ───────────────────

/// A heapless, fixed-capacity Object Dictionary backed by a const-generic array.
///
/// `N` is the maximum number of entries.  All entries are stored inline —
/// no heap allocation is required, making `StaticOd` suitable for `no_std`
/// embedded targets.
///
/// # Example
/// ```rust,ignore
/// let mut od = StaticOd::<32>::new();
/// od.insert(OdEntry::new(0x1000, 0, DataType::Unsigned32, AccessType::RO,
///                         OdEntryValue::U32(0x0402))).unwrap();
/// let val = od.read(0x1000, 0).unwrap();
/// ```
pub struct StaticOd<const N: usize> {
    entries: [Option<OdEntry>; N],
    count: usize,
}

impl<const N: usize> StaticOd<N> {
    /// Create an empty `StaticOd`.
    pub fn new() -> Self {
        Self {
            entries: [None; N],
            count: 0,
        }
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if no entries are stored.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns `true` if the dictionary is at capacity.
    pub fn is_full(&self) -> bool {
        self.count >= N
    }

    /// Insert (or replace) an OD entry.
    ///
    /// If an entry with the same `(index, subindex)` already exists it is
    /// replaced.  Otherwise a new slot is allocated.  Returns `Err(OdError::Full)`
    /// if the dictionary is at capacity and no existing entry matches.
    pub fn insert(&mut self, entry: OdEntry) -> Result<(), OdError> {
        // Try to update an existing entry first.
        for e in self.entries[..self.count].iter_mut().flatten() {
            if e.index == entry.index && e.subindex == entry.subindex {
                *e = entry;
                return Ok(());
            }
        }
        // Allocate a new slot.
        if self.count >= N {
            return Err(OdError::Full);
        }
        self.entries[self.count] = Some(entry);
        self.count += 1;
        Ok(())
    }

    /// Find the index into `self.entries` for the given `(index, subindex)`.
    fn find(&self, index: u16, subindex: u8) -> Option<usize> {
        self.entries[..self.count].iter().position(|slot| {
            slot.map(|e| e.index == index && e.subindex == subindex)
                .unwrap_or(false)
        })
    }

    /// Returns `true` if any entry exists at `(index, subindex)`.
    pub fn has(&self, index: u16, subindex: u8) -> bool {
        self.find(index, subindex).is_some()
    }

    /// Read the value at `(index, subindex)`.
    ///
    /// # Errors
    /// - `OdError::IndexNotFound` — no entry with this index exists at all.
    /// - `OdError::SubindexNotFound` — index found but sub-index absent.
    /// - `OdError::WriteOnly` — access type is write-only; read denied.
    pub fn read(&self, index: u16, subindex: u8) -> Result<OdEntryValue, OdError> {
        // Check whether the index exists at all (any subindex).
        let index_present = self.entries[..self.count]
            .iter()
            .any(|s| s.map(|e| e.index == index).unwrap_or(false));
        if !index_present {
            return Err(OdError::IndexNotFound);
        }
        let pos = self
            .find(index, subindex)
            .ok_or(OdError::SubindexNotFound)?;
        let entry = self.entries[pos].ok_or(OdError::SubindexNotFound)?;
        if !entry.access_type.can_read() {
            return Err(OdError::WriteOnly);
        }
        Ok(entry.value)
    }

    /// Write `value` to `(index, subindex)`.
    ///
    /// # Errors
    /// - `OdError::IndexNotFound` — no entry with this index exists at all.
    /// - `OdError::SubindexNotFound` — index found but sub-index absent.
    /// - `OdError::ReadOnly` — access type does not permit writes.
    /// - `OdError::TypeMismatch` — `value` variant does not match the stored type.
    pub fn write(&mut self, index: u16, subindex: u8, value: OdEntryValue) -> Result<(), OdError> {
        let index_present = self.entries[..self.count]
            .iter()
            .any(|s| s.map(|e| e.index == index).unwrap_or(false));
        if !index_present {
            return Err(OdError::IndexNotFound);
        }
        let pos = self
            .find(index, subindex)
            .ok_or(OdError::SubindexNotFound)?;
        let entry = self.entries[pos]
            .as_mut()
            .ok_or(OdError::SubindexNotFound)?;
        if !entry.access_type.can_write() {
            return Err(OdError::ReadOnly);
        }
        if !entry.value.type_compatible(&value) {
            return Err(OdError::TypeMismatch);
        }
        entry.value = value;
        Ok(())
    }

    /// Return a reference to the raw `OdEntry` at `(index, subindex)` if it
    /// exists, regardless of access control.
    pub fn entry(&self, index: u16, subindex: u8) -> Option<&OdEntry> {
        let pos = self.find(index, subindex)?;
        self.entries[pos].as_ref()
    }
}

impl<const N: usize> Default for StaticOd<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_get_set() {
        let mut od = ObjectDict::<32>::new();
        od.define(OdIndex(0x6040, 0), OdValue::U16(0)).unwrap();
        od.set_u16(0x6040, 0, 0x000F).unwrap();
        assert_eq!(od.get_u16(0x6040, 0).unwrap(), 0x000F);
    }

    #[test]
    fn test_read_only_protection() {
        let mut od = ObjectDict::<32>::new();
        od.define_ro(OdIndex(0x1000, 0), OdValue::U32(0x0402))
            .unwrap();
        assert_eq!(od.set_u32(0x1000, 0, 0), Err(OdError::ReadOnly));
        assert_eq!(od.get_u32(0x1000, 0).unwrap(), 0x0402);
    }

    #[test]
    fn test_ds301_defaults() {
        let od = ObjectDict::<32>::with_ds301_defaults(0xABCD, 0x1234, 0x0001, 0xDEAD);
        assert!(od.has(0x1000, 0));
        assert!(od.has(0x1001, 0));
        assert!(od.has(0x1018, 0));
        assert_eq!(od.get_u32(0x1018, 1).unwrap(), 0xABCD);
        assert_eq!(od.get_u32(0x1018, 4).unwrap(), 0xDEAD);
    }

    #[test]
    fn test_not_found() {
        let od = ObjectDict::<32>::new();
        assert_eq!(od.get(0x9999, 0), Err(OdError::NotFound));
    }

    #[test]
    fn test_type_mismatch() {
        let mut od = ObjectDict::<32>::new();
        od.define(OdIndex(0x6040, 0), OdValue::U16(0)).unwrap();
        // Try to set with wrong type
        assert_eq!(
            od.set(0x6040, 0, OdValue::U32(0)),
            Err(OdError::TypeMismatch)
        );
    }

    #[test]
    fn test_multiple_types() {
        let mut od = ObjectDict::<64>::new();
        od.define(OdIndex(0x2001, 0), OdValue::I32(-100)).unwrap();
        od.define(OdIndex(0x2002, 0), OdValue::F32(core::f32::consts::PI))
            .unwrap();
        od.define(OdIndex(0x2003, 0), OdValue::Bytes([1, 2, 3, 4, 5, 6, 7, 8]))
            .unwrap();

        od.set_i32(0x2001, 0, -999).unwrap();
        assert_eq!(od.get_i32(0x2001, 0).unwrap(), -999);

        let fb = od.get(0x2003, 0).unwrap().as_bytes().unwrap();
        assert_eq!(fb[0], 1);
    }

    // ── StaticOd tests ───────────────────────────────────────────────────────

    #[test]
    fn static_od_insert_and_read() {
        let mut od = StaticOd::<16>::new();
        od.insert(OdEntry::new(
            0x1000,
            0,
            DataType::Unsigned32,
            AccessType::RO,
            OdEntryValue::U32(0x0402),
        ))
        .unwrap();
        let val = od.read(0x1000, 0).unwrap();
        assert_eq!(val, OdEntryValue::U32(0x0402));
    }

    #[test]
    fn static_od_write_rw() {
        let mut od = StaticOd::<16>::new();
        od.insert(OdEntry::new(
            0x6040,
            0,
            DataType::Unsigned16,
            AccessType::RW,
            OdEntryValue::U16(0),
        ))
        .unwrap();
        od.write(0x6040, 0, OdEntryValue::U16(0x000F)).unwrap();
        assert_eq!(od.read(0x6040, 0).unwrap(), OdEntryValue::U16(0x000F));
    }

    #[test]
    fn static_od_read_only_rejects_write() {
        let mut od = StaticOd::<16>::new();
        od.insert(OdEntry::new(
            0x1000,
            0,
            DataType::Unsigned32,
            AccessType::RO,
            OdEntryValue::U32(0x0402),
        ))
        .unwrap();
        assert_eq!(
            od.write(0x1000, 0, OdEntryValue::U32(0)),
            Err(OdError::ReadOnly)
        );
    }

    #[test]
    fn static_od_write_only_rejects_read() {
        let mut od = StaticOd::<16>::new();
        od.insert(OdEntry::new(
            0x2100,
            0,
            DataType::Unsigned8,
            AccessType::WO,
            OdEntryValue::U8(0),
        ))
        .unwrap();
        assert_eq!(od.read(0x2100, 0), Err(OdError::WriteOnly));
        od.write(0x2100, 0, OdEntryValue::U8(42)).unwrap();
    }

    #[test]
    fn static_od_index_not_found() {
        let od = StaticOd::<16>::new();
        assert_eq!(od.read(0x9999, 0), Err(OdError::IndexNotFound));
    }

    #[test]
    fn static_od_subindex_not_found() {
        let mut od = StaticOd::<16>::new();
        od.insert(OdEntry::new(
            0x1018,
            0,
            DataType::Unsigned8,
            AccessType::RO,
            OdEntryValue::U8(4),
        ))
        .unwrap();
        assert_eq!(od.read(0x1018, 5), Err(OdError::SubindexNotFound));
    }

    #[test]
    fn static_od_type_mismatch_on_write() {
        let mut od = StaticOd::<16>::new();
        od.insert(OdEntry::new(
            0x6040,
            0,
            DataType::Unsigned16,
            AccessType::RW,
            OdEntryValue::U16(0),
        ))
        .unwrap();
        assert_eq!(
            od.write(0x6040, 0, OdEntryValue::U32(0)),
            Err(OdError::TypeMismatch)
        );
    }

    #[test]
    fn static_od_capacity_error() {
        let mut od = StaticOd::<2>::new();
        od.insert(OdEntry::new(
            0x1000,
            0,
            DataType::Unsigned32,
            AccessType::RO,
            OdEntryValue::U32(0),
        ))
        .unwrap();
        od.insert(OdEntry::new(
            0x1001,
            0,
            DataType::Unsigned8,
            AccessType::RW,
            OdEntryValue::U8(0),
        ))
        .unwrap();
        let result = od.insert(OdEntry::new(
            0x1017,
            0,
            DataType::Unsigned16,
            AccessType::RW,
            OdEntryValue::U16(0),
        ));
        assert_eq!(result, Err(OdError::Full));
    }

    #[test]
    fn static_od_replace_existing() {
        let mut od = StaticOd::<4>::new();
        od.insert(OdEntry::new(
            0x1000,
            0,
            DataType::Unsigned32,
            AccessType::RW,
            OdEntryValue::U32(1),
        ))
        .unwrap();
        od.insert(OdEntry::new(
            0x1000,
            0,
            DataType::Unsigned32,
            AccessType::RW,
            OdEntryValue::U32(2),
        ))
        .unwrap();
        assert_eq!(od.len(), 1);
        assert_eq!(od.read(0x1000, 0).unwrap(), OdEntryValue::U32(2));
    }

    #[test]
    fn od_entry_value_to_le_bytes_u32() {
        let v = OdEntryValue::U32(0xDEAD_BEEF);
        let bytes = v.to_le_bytes();
        assert_eq!(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            0xDEAD_BEEF
        );
        assert_eq!(&bytes[4..], &[0u8; 4]);
    }

    #[test]
    fn access_type_permissions() {
        assert!(AccessType::RO.can_read());
        assert!(!AccessType::RO.can_write());
        assert!(!AccessType::WO.can_read());
        assert!(AccessType::WO.can_write());
        assert!(AccessType::RW.can_read());
        assert!(AccessType::RW.can_write());
        assert!(AccessType::Const.can_read());
        assert!(!AccessType::Const.can_write());
    }
}
