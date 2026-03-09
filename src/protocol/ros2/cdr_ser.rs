//! CDR (Common Data Representation) serializer/deserializer for ROS2.
//!
//! CDR is the wire format used by ROS2 DDS implementations. This module
//! provides little-endian CDR serialization with proper 4-byte alignment
//! for structured types, and the standard ROS2 encapsulation header.

use heapless::Vec;

/// CDR encapsulation header for little-endian data.
pub const CDR_ENCAP_LE: [u8; 4] = [0x00, 0x01, 0x00, 0x00];

/// CDR serialization error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdrError {
    /// Buffer capacity exceeded.
    BufferFull,
    /// Not enough data to deserialize.
    InsufficientData,
    /// Invalid encapsulation header.
    InvalidHeader,
    /// Alignment error.
    AlignmentError,
}

/// CDR serializer with a heapless backing buffer.
///
/// `N` is the maximum serialized size in bytes.
pub struct CdrSerializer<const N: usize> {
    buf: Vec<u8, N>,
    header_written: bool,
}

impl<const N: usize> CdrSerializer<N> {
    /// Create a new CDR serializer.
    ///
    /// Automatically writes the encapsulation header (4 bytes).
    pub fn new() -> Result<Self, CdrError> {
        let mut s = Self {
            buf: Vec::new(),
            header_written: false,
        };
        s.write_header()?;
        Ok(s)
    }

    /// Create without writing the encapsulation header (for sub-messages).
    pub fn new_without_header() -> Self {
        Self {
            buf: Vec::new(),
            header_written: false,
        }
    }

    /// Write the CDR little-endian encapsulation header.
    fn write_header(&mut self) -> Result<(), CdrError> {
        for &b in &CDR_ENCAP_LE {
            self.push_byte(b)?;
        }
        self.header_written = true;
        Ok(())
    }

    /// Push a single byte.
    fn push_byte(&mut self, b: u8) -> Result<(), CdrError> {
        self.buf.push(b).map_err(|_| CdrError::BufferFull)
    }

    /// Current byte position (length of written data).
    pub fn position(&self) -> usize {
        self.buf.len()
    }

    /// Pad to the given alignment boundary.
    pub fn pad_to_alignment(&mut self, align: usize) -> Result<(), CdrError> {
        let pos = self.buf.len();
        let rem = pos % align;
        if rem != 0 {
            let pad = align - rem;
            for _ in 0..pad {
                self.push_byte(0)?;
            }
        }
        Ok(())
    }

    /// Serialize a u8 (no alignment needed).
    pub fn serialize_u8(&mut self, val: u8) -> Result<(), CdrError> {
        self.push_byte(val)
    }

    /// Serialize a bool as u8.
    pub fn serialize_bool(&mut self, val: bool) -> Result<(), CdrError> {
        self.push_byte(if val { 1 } else { 0 })
    }

    /// Serialize a u16 (2-byte aligned).
    pub fn serialize_u16(&mut self, val: u16) -> Result<(), CdrError> {
        self.pad_to_alignment(2)?;
        let b = val.to_le_bytes();
        self.push_byte(b[0])?;
        self.push_byte(b[1])
    }

    /// Serialize an i16 (2-byte aligned).
    pub fn serialize_i16(&mut self, val: i16) -> Result<(), CdrError> {
        self.serialize_u16(val as u16)
    }

    /// Serialize a u32 (4-byte aligned).
    pub fn serialize_u32(&mut self, val: u32) -> Result<(), CdrError> {
        self.pad_to_alignment(4)?;
        let b = val.to_le_bytes();
        for byte in b {
            self.push_byte(byte)?;
        }
        Ok(())
    }

    /// Serialize an i32 (4-byte aligned).
    pub fn serialize_i32(&mut self, val: i32) -> Result<(), CdrError> {
        self.serialize_u32(val as u32)
    }

    /// Serialize an f32 (4-byte aligned).
    pub fn serialize_f32(&mut self, val: f32) -> Result<(), CdrError> {
        self.serialize_u32(val.to_bits())
    }

    /// Serialize a u64 (8-byte aligned).
    pub fn serialize_u64(&mut self, val: u64) -> Result<(), CdrError> {
        self.pad_to_alignment(4)?; // CDR uses 4-byte max alignment for primitives
        let b = val.to_le_bytes();
        for byte in b {
            self.push_byte(byte)?;
        }
        Ok(())
    }

    /// Serialize an i64 (4-byte aligned in CDR).
    pub fn serialize_i64(&mut self, val: i64) -> Result<(), CdrError> {
        self.serialize_u64(val as u64)
    }

    /// Serialize an f64 (4-byte aligned in CDR).
    pub fn serialize_f64(&mut self, val: f64) -> Result<(), CdrError> {
        self.serialize_u64(val.to_bits())
    }

    /// Serialize a fixed-length byte array.
    pub fn serialize_bytes(&mut self, data: &[u8]) -> Result<(), CdrError> {
        for &b in data {
            self.push_byte(b)?;
        }
        Ok(())
    }

    /// Serialize a string as CDR sequence (length prefix u32 + bytes + null terminator).
    pub fn serialize_string(&mut self, s: &[u8]) -> Result<(), CdrError> {
        // CDR string: length includes null terminator
        self.serialize_u32((s.len() + 1) as u32)?;
        self.serialize_bytes(s)?;
        self.push_byte(0) // null terminator
    }

    /// Serialize an array length prefix (u32).
    pub fn serialize_sequence_length(&mut self, len: u32) -> Result<(), CdrError> {
        self.serialize_u32(len)
    }

    /// Access the serialized bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Finalize and return the buffer length.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the buffer is empty (no bytes written yet).
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Whether the header was written.
    pub fn has_header(&self) -> bool {
        self.header_written
    }
}

impl<const N: usize> Default for CdrSerializer<N> {
    fn default() -> Self {
        Self::new_without_header()
    }
}

/// CDR deserializer working from a byte slice.
pub struct CdrDeserializer<'a> {
    data: &'a [u8],
    cursor: usize,
    little_endian: bool,
}

impl<'a> CdrDeserializer<'a> {
    /// Create from a complete CDR buffer including encapsulation header.
    pub fn new(data: &'a [u8]) -> Result<Self, CdrError> {
        if data.len() < 4 {
            return Err(CdrError::InvalidHeader);
        }
        // Check encapsulation: byte 0=0x00 (XCDR1), byte 1=0x01 (LE) or 0x00 (BE)
        if data[0] != 0x00 {
            return Err(CdrError::InvalidHeader);
        }
        let little_endian = data[1] == 0x01;
        Ok(Self {
            data,
            cursor: 4,
            little_endian,
        })
    }

    /// Create without encapsulation header (for sub-messages).
    pub fn new_raw(data: &'a [u8]) -> Self {
        Self {
            data,
            cursor: 0,
            little_endian: true,
        }
    }

    /// Current cursor position.
    pub fn position(&self) -> usize {
        self.cursor
    }

    /// Remaining bytes.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.cursor)
    }

    /// Align cursor to given boundary.
    fn align_to(&mut self, align: usize) {
        let rem = self.cursor % align;
        if rem != 0 {
            self.cursor += align - rem;
        }
    }

    /// Read a single byte.
    fn read_byte(&mut self) -> Result<u8, CdrError> {
        if self.cursor >= self.data.len() {
            return Err(CdrError::InsufficientData);
        }
        let b = self.data[self.cursor];
        self.cursor += 1;
        Ok(b)
    }

    /// Deserialize a u8.
    pub fn deserialize_u8(&mut self) -> Result<u8, CdrError> {
        self.read_byte()
    }

    /// Deserialize a bool.
    pub fn deserialize_bool(&mut self) -> Result<bool, CdrError> {
        Ok(self.read_byte()? != 0)
    }

    /// Deserialize a u16 (2-byte aligned).
    pub fn deserialize_u16(&mut self) -> Result<u16, CdrError> {
        self.align_to(2);
        let lo = self.read_byte()?;
        let hi = self.read_byte()?;
        if self.little_endian {
            Ok(u16::from_le_bytes([lo, hi]))
        } else {
            Ok(u16::from_be_bytes([lo, hi]))
        }
    }

    /// Deserialize an i16.
    pub fn deserialize_i16(&mut self) -> Result<i16, CdrError> {
        Ok(self.deserialize_u16()? as i16)
    }

    /// Deserialize a u32 (4-byte aligned).
    pub fn deserialize_u32(&mut self) -> Result<u32, CdrError> {
        self.align_to(4);
        if self.cursor + 4 > self.data.len() {
            return Err(CdrError::InsufficientData);
        }
        let bytes = [
            self.data[self.cursor],
            self.data[self.cursor + 1],
            self.data[self.cursor + 2],
            self.data[self.cursor + 3],
        ];
        self.cursor += 4;
        if self.little_endian {
            Ok(u32::from_le_bytes(bytes))
        } else {
            Ok(u32::from_be_bytes(bytes))
        }
    }

    /// Deserialize an i32.
    pub fn deserialize_i32(&mut self) -> Result<i32, CdrError> {
        Ok(self.deserialize_u32()? as i32)
    }

    /// Deserialize an f32.
    pub fn deserialize_f32(&mut self) -> Result<f32, CdrError> {
        Ok(f32::from_bits(self.deserialize_u32()?))
    }

    /// Deserialize a u64.
    pub fn deserialize_u64(&mut self) -> Result<u64, CdrError> {
        self.align_to(4);
        if self.cursor + 8 > self.data.len() {
            return Err(CdrError::InsufficientData);
        }
        let bytes = [
            self.data[self.cursor],
            self.data[self.cursor + 1],
            self.data[self.cursor + 2],
            self.data[self.cursor + 3],
            self.data[self.cursor + 4],
            self.data[self.cursor + 5],
            self.data[self.cursor + 6],
            self.data[self.cursor + 7],
        ];
        self.cursor += 8;
        if self.little_endian {
            Ok(u64::from_le_bytes(bytes))
        } else {
            Ok(u64::from_be_bytes(bytes))
        }
    }

    /// Deserialize an i64.
    pub fn deserialize_i64(&mut self) -> Result<i64, CdrError> {
        Ok(self.deserialize_u64()? as i64)
    }

    /// Deserialize an f64.
    pub fn deserialize_f64(&mut self) -> Result<f64, CdrError> {
        Ok(f64::from_bits(self.deserialize_u64()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_u32_roundtrip() {
        let mut ser = CdrSerializer::<64>::new().unwrap();
        ser.serialize_u32(0xDEADBEEF).unwrap();
        let bytes = ser.as_bytes();

        let mut de = CdrDeserializer::new(bytes).unwrap();
        let val = de.deserialize_u32().unwrap();
        assert_eq!(val, 0xDEADBEEF);
    }

    #[test]
    fn test_serialize_f32_roundtrip() {
        let mut ser = CdrSerializer::<64>::new().unwrap();
        ser.serialize_f32(core::f32::consts::PI).unwrap();
        let bytes = ser.as_bytes();

        let mut de = CdrDeserializer::new(bytes).unwrap();
        let val = de.deserialize_f32().unwrap();
        assert!((val - core::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn test_serialize_f64_roundtrip() {
        let mut ser = CdrSerializer::<64>::new().unwrap();
        ser.serialize_f64(core::f64::consts::E).unwrap();
        let bytes = ser.as_bytes();

        let mut de = CdrDeserializer::new(bytes).unwrap();
        let val = de.deserialize_f64().unwrap();
        assert!((val - core::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_encapsulation_header() {
        let ser = CdrSerializer::<64>::new().unwrap();
        let bytes = ser.as_bytes();
        assert_eq!(&bytes[0..4], &CDR_ENCAP_LE);
    }

    #[test]
    fn test_alignment_padding() {
        let mut ser = CdrSerializer::<64>::new().unwrap();
        // After 4-byte header, write a u8 (no alignment)
        ser.serialize_u8(0xAB).unwrap();
        // Now a u32 should be padded to 4-byte boundary (3 bytes of padding)
        let pos_before = ser.position();
        ser.serialize_u32(0x12345678).unwrap();
        let pos_after = ser.position();
        // Should have added 3 padding bytes + 4 data bytes = 7 bytes
        assert_eq!(pos_after - pos_before, 7);

        let mut de = CdrDeserializer::new(ser.as_bytes()).unwrap();
        let u = de.deserialize_u8().unwrap();
        assert_eq!(u, 0xAB);
        let v = de.deserialize_u32().unwrap();
        assert_eq!(v, 0x12345678);
    }

    #[test]
    fn test_mixed_types_roundtrip() {
        let mut ser = CdrSerializer::<128>::new().unwrap();
        ser.serialize_u8(42).unwrap();
        ser.serialize_u16(1000).unwrap();
        ser.serialize_i32(-99999).unwrap();
        ser.serialize_f64(1.23456789).unwrap();

        let mut de = CdrDeserializer::new(ser.as_bytes()).unwrap();
        assert_eq!(de.deserialize_u8().unwrap(), 42);
        assert_eq!(de.deserialize_u16().unwrap(), 1000);
        assert_eq!(de.deserialize_i32().unwrap(), -99999);
        let f = de.deserialize_f64().unwrap();
        assert!((f - 1.23456789).abs() < 1e-10);
    }

    #[test]
    fn test_insufficient_data_error() {
        let data = [0x00u8, 0x01, 0x00, 0x00, 0x01]; // only 1 byte of payload
        let mut de = CdrDeserializer::new(&data).unwrap();
        // Reading a u32 with only 1 byte available should fail
        assert_eq!(de.deserialize_u32(), Err(CdrError::InsufficientData));
    }
}
