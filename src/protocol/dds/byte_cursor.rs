use crate::protocol::dds::error::RtpsError;

/// Wire endianness for a submessage (RTPS E flag, bit 0 of flags byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Big,
    Little,
}

impl Endianness {
    /// Decode from the RTPS submessage flags byte (bit 0 = E flag).
    #[inline]
    pub fn from_flags(flags: u8) -> Self {
        if flags & 0x01 != 0 {
            Self::Little
        } else {
            Self::Big
        }
    }

    /// Encode into the RTPS submessage flags byte (sets or clears bit 0).
    #[inline]
    pub fn into_flags(self, flags: u8) -> u8 {
        match self {
            Self::Little => flags | 0x01,
            Self::Big => flags & !0x01,
        }
    }
}

/// Cursor for reading bytes with endianness awareness.
pub struct ByteCursor<'a> {
    buf: &'a [u8],
    pos: usize,
    pub endianness: Endianness,
}

impl<'a> ByteCursor<'a> {
    pub fn new(buf: &'a [u8], endianness: Endianness) -> Self {
        Self {
            buf,
            pos: 0,
            endianness,
        }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    pub fn read_u8(&mut self) -> Result<u8, RtpsError> {
        if self.pos >= self.buf.len() {
            return Err(RtpsError::TruncatedHeader);
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    pub fn read_u16(&mut self) -> Result<u16, RtpsError> {
        let bytes = self.read_fixed::<2>()?;
        Ok(match self.endianness {
            Endianness::Little => u16::from_le_bytes(bytes),
            Endianness::Big => u16::from_be_bytes(bytes),
        })
    }

    pub fn read_i16(&mut self) -> Result<i16, RtpsError> {
        let bytes = self.read_fixed::<2>()?;
        Ok(match self.endianness {
            Endianness::Little => i16::from_le_bytes(bytes),
            Endianness::Big => i16::from_be_bytes(bytes),
        })
    }

    pub fn read_u32(&mut self) -> Result<u32, RtpsError> {
        let bytes = self.read_fixed::<4>()?;
        Ok(match self.endianness {
            Endianness::Little => u32::from_le_bytes(bytes),
            Endianness::Big => u32::from_be_bytes(bytes),
        })
    }

    pub fn read_i32(&mut self) -> Result<i32, RtpsError> {
        let bytes = self.read_fixed::<4>()?;
        Ok(match self.endianness {
            Endianness::Little => i32::from_le_bytes(bytes),
            Endianness::Big => i32::from_be_bytes(bytes),
        })
    }

    pub fn read_u64(&mut self) -> Result<u64, RtpsError> {
        let bytes = self.read_fixed::<8>()?;
        Ok(match self.endianness {
            Endianness::Little => u64::from_le_bytes(bytes),
            Endianness::Big => u64::from_be_bytes(bytes),
        })
    }

    pub fn read_i64(&mut self) -> Result<i64, RtpsError> {
        let bytes = self.read_fixed::<8>()?;
        Ok(match self.endianness {
            Endianness::Little => i64::from_le_bytes(bytes),
            Endianness::Big => i64::from_be_bytes(bytes),
        })
    }

    /// Read a 32-bit IEEE 754 float (via bit-cast from u32).
    pub fn read_f32(&mut self) -> Result<f32, RtpsError> {
        let bytes = self.read_fixed::<4>()?;
        let bits = match self.endianness {
            Endianness::Little => u32::from_le_bytes(bytes),
            Endianness::Big => u32::from_be_bytes(bytes),
        };
        Ok(f32::from_bits(bits))
    }

    /// Read a 64-bit IEEE 754 float (via bit-cast from u64).
    pub fn read_f64(&mut self) -> Result<f64, RtpsError> {
        let bytes = self.read_fixed::<8>()?;
        let bits = match self.endianness {
            Endianness::Little => u64::from_le_bytes(bytes),
            Endianness::Big => u64::from_be_bytes(bytes),
        };
        Ok(f64::from_bits(bits))
    }

    /// Read exactly `n` bytes and return a slice. Zero-copy borrow from input.
    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], RtpsError> {
        let end = self.pos.checked_add(n).ok_or(RtpsError::TruncatedHeader)?;
        if end > self.buf.len() {
            return Err(RtpsError::TruncatedHeader);
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    /// Advance position to the next multiple of `align` (1, 2, or 4).
    /// Used for CDR 4-byte alignment of multi-byte fields.
    pub fn align_to(&mut self, align: usize) -> Result<(), RtpsError> {
        let rem = self.pos % align;
        if rem != 0 {
            let skip = align - rem;
            let new_pos = self
                .pos
                .checked_add(skip)
                .ok_or(RtpsError::AlignmentError)?;
            if new_pos > self.buf.len() {
                return Err(RtpsError::TruncatedHeader);
            }
            self.pos = new_pos;
        }
        Ok(())
    }

    /// Peek at the remaining bytes without advancing.
    pub fn peek_remaining(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    /// Read a CDR string, returning a borrowed `&str` into the input buffer.
    ///
    /// Format: `[u32 length (includes null terminator)][bytes][null][zero padding to 4-byte boundary]`.
    /// The returned slice excludes the trailing NUL.
    pub fn read_cdr_string(&mut self) -> Result<&'a str, RtpsError> {
        let len = self.read_u32()? as usize;
        if len == 0 {
            // Non-standard zero-length: treat as empty string.
            return Ok("");
        }
        // `len` includes the trailing NUL.
        let raw = self.read_bytes(len)?;
        // Strip trailing NUL if present.
        let s_bytes = if raw.last() == Some(&0) {
            &raw[..raw.len() - 1]
        } else {
            raw
        };
        let s = core::str::from_utf8(s_bytes).map_err(|_| RtpsError::InvalidStringEncoding)?;
        // Advance past padding: total written = 4 (length prefix) + len (bytes + NUL).
        // Pad so that (4 + len) is 4-byte-aligned from the start of the field.
        let pad = (4 - (len % 4)) % 4;
        if pad > 0 {
            self.skip(pad)?;
        }
        Ok(s)
    }

    /// Advance position by `n` bytes without reading them.
    pub fn skip(&mut self, n: usize) -> Result<(), RtpsError> {
        let new_pos = self.pos.checked_add(n).ok_or(RtpsError::TruncatedHeader)?;
        if new_pos > self.buf.len() {
            return Err(RtpsError::TruncatedHeader);
        }
        self.pos = new_pos;
        Ok(())
    }

    fn read_fixed<const N: usize>(&mut self) -> Result<[u8; N], RtpsError> {
        let end = self.pos.checked_add(N).ok_or(RtpsError::TruncatedHeader)?;
        if end > self.buf.len() {
            return Err(RtpsError::TruncatedHeader);
        }
        let mut arr = [0u8; N];
        arr.copy_from_slice(&self.buf[self.pos..end]);
        self.pos = end;
        Ok(arr)
    }
}

/// Writer for serializing bytes with endianness awareness.
pub struct ByteWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
    pub endianness: Endianness,
}

impl<'a> ByteWriter<'a> {
    pub fn new(buf: &'a mut [u8], endianness: Endianness) -> Self {
        Self {
            buf,
            pos: 0,
            endianness,
        }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    pub fn write_u8(&mut self, v: u8) -> Result<(), RtpsError> {
        self.write_bytes(&[v])
    }

    pub fn write_u16(&mut self, v: u16) -> Result<(), RtpsError> {
        let bytes = match self.endianness {
            Endianness::Little => v.to_le_bytes(),
            Endianness::Big => v.to_be_bytes(),
        };
        self.write_bytes(&bytes)
    }

    pub fn write_i16(&mut self, v: i16) -> Result<(), RtpsError> {
        let bytes = match self.endianness {
            Endianness::Little => v.to_le_bytes(),
            Endianness::Big => v.to_be_bytes(),
        };
        self.write_bytes(&bytes)
    }

    pub fn write_u32(&mut self, v: u32) -> Result<(), RtpsError> {
        let bytes = match self.endianness {
            Endianness::Little => v.to_le_bytes(),
            Endianness::Big => v.to_be_bytes(),
        };
        self.write_bytes(&bytes)
    }

    pub fn write_i32(&mut self, v: i32) -> Result<(), RtpsError> {
        let bytes = match self.endianness {
            Endianness::Little => v.to_le_bytes(),
            Endianness::Big => v.to_be_bytes(),
        };
        self.write_bytes(&bytes)
    }

    pub fn write_u64(&mut self, v: u64) -> Result<(), RtpsError> {
        let bytes = match self.endianness {
            Endianness::Little => v.to_le_bytes(),
            Endianness::Big => v.to_be_bytes(),
        };
        self.write_bytes(&bytes)
    }

    pub fn write_i64(&mut self, v: i64) -> Result<(), RtpsError> {
        let bytes = match self.endianness {
            Endianness::Little => v.to_le_bytes(),
            Endianness::Big => v.to_be_bytes(),
        };
        self.write_bytes(&bytes)
    }

    /// Write a 32-bit IEEE 754 float (via bit-cast to u32).
    pub fn write_f32(&mut self, v: f32) -> Result<(), RtpsError> {
        let bits = v.to_bits();
        let bytes = match self.endianness {
            Endianness::Little => bits.to_le_bytes(),
            Endianness::Big => bits.to_be_bytes(),
        };
        self.write_bytes(&bytes)
    }

    /// Write a 64-bit IEEE 754 float (via bit-cast to u64).
    pub fn write_f64(&mut self, v: f64) -> Result<(), RtpsError> {
        let bits = v.to_bits();
        let bytes = match self.endianness {
            Endianness::Little => bits.to_le_bytes(),
            Endianness::Big => bits.to_be_bytes(),
        };
        self.write_bytes(&bytes)
    }

    pub fn write_bytes(&mut self, data: &[u8]) -> Result<(), RtpsError> {
        let end = self
            .pos
            .checked_add(data.len())
            .ok_or(RtpsError::BufferTooSmall)?;
        if end > self.buf.len() {
            return Err(RtpsError::BufferTooSmall);
        }
        self.buf[self.pos..end].copy_from_slice(data);
        self.pos = end;
        Ok(())
    }

    /// Write zero padding to align position to `align` bytes.
    pub fn align_to(&mut self, align: usize) -> Result<(), RtpsError> {
        let rem = self.pos % align;
        if rem != 0 {
            let pad = align - rem;
            let zeros = [0u8; 8];
            self.write_bytes(&zeros[..pad])?;
        }
        Ok(())
    }

    /// Write a CDR string into the buffer.
    ///
    /// Format: `[u32 length (includes null terminator)][bytes][null][zero padding to 4-byte boundary]`.
    /// The `u32` length field and padding respect the writer's current endianness for the length
    /// prefix, but the string bytes and NUL are always written as-is.
    pub fn write_cdr_string(&mut self, s: &str) -> Result<(), RtpsError> {
        let with_null = s.len() + 1; // content bytes + NUL
        self.write_u32(with_null as u32)?;
        self.write_bytes(s.as_bytes())?;
        self.write_u8(0)?; // NUL terminator
                           // Pad so that (4 + with_null) is 4-byte-aligned from the start of the CDR string field.
                           // We've written `4 + with_null` bytes; pad to multiple-of-4.
        let pad = (4 - (with_null % 4)) % 4;
        if pad > 0 {
            let zeros = [0u8; 3];
            self.write_bytes(&zeros[..pad])?;
        }
        Ok(())
    }

    /// Overwrite 2 bytes at a previously-written position (for backfilling octetsToNextHeader).
    pub fn patch_u16_at(&mut self, offset: usize, v: u16) -> Result<(), RtpsError> {
        let end = offset.checked_add(2).ok_or(RtpsError::BufferTooSmall)?;
        if end > self.pos {
            return Err(RtpsError::BufferTooSmall);
        }
        let bytes = match self.endianness {
            Endianness::Little => v.to_le_bytes(),
            Endianness::Big => v.to_be_bytes(),
        };
        self.buf[offset..end].copy_from_slice(&bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a CDR string through ByteWriter::write_cdr_string + ByteCursor::read_cdr_string.
    #[test]
    fn cdr_string_roundtrip_ascii() {
        let s = "hello";
        // Format: u32(6) + 5 bytes + NUL + pad(3) = 4 + 8 = 12 bytes
        let mut buf = [0u8; 64];
        let mut w = ByteWriter::new(&mut buf, Endianness::Little);
        w.write_cdr_string(s).unwrap();
        let written = w.position();
        assert_eq!(written, 12); // 4 (len prefix) + 6 (content + NUL) + 2 (pad to 8) = 12

        let mut cur = ByteCursor::new(&buf[..written], Endianness::Little);
        let result = cur.read_cdr_string().unwrap();
        assert_eq!(result, s);
    }

    /// CDR string whose length is already 4-byte aligned (no extra padding).
    #[test]
    fn cdr_string_roundtrip_aligned_length() {
        // "abc" → with_null=4 → 4%4==0 → pad=0 → total = 4+4 = 8 bytes
        let s = "abc";
        let mut buf = [0u8; 64];
        let mut w = ByteWriter::new(&mut buf, Endianness::Little);
        w.write_cdr_string(s).unwrap();
        let written = w.position();
        assert_eq!(written, 8); // 4 (len) + 4 (3 chars + NUL, already aligned)

        let mut cur = ByteCursor::new(&buf[..written], Endianness::Little);
        let result = cur.read_cdr_string().unwrap();
        assert_eq!(result, s);
    }

    /// Empty string round-trip.
    #[test]
    fn cdr_string_roundtrip_empty() {
        // "" → with_null=1 → pad=3 → total = 4+1+3 = 8 bytes (NUL + 3 pad)
        let s = "";
        let mut buf = [0u8; 64];
        let mut w = ByteWriter::new(&mut buf, Endianness::Little);
        w.write_cdr_string(s).unwrap();
        let written = w.position();
        assert_eq!(written, 8); // 4 (len=1) + 1 (NUL) + 3 (pad) = 8

        let mut cur = ByteCursor::new(&buf[..written], Endianness::Little);
        let result = cur.read_cdr_string().unwrap();
        assert_eq!(result, s);
    }
}
