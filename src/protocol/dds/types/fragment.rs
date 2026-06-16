use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};
use crate::protocol::dds::error::RtpsError;

/// 1-based fragment number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FragmentNumber(pub u32);

impl FragmentNumber {
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        Ok(Self(cur.read_u32()?))
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_u32(self.0)
    }
}

/// Bitmap set of fragment numbers. Same encoding as SequenceNumberSet but for u32 base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentNumberSet {
    pub bitmap_base: FragmentNumber,
    pub num_bits: u32,
    pub bitmap: [u32; 8],
}

impl FragmentNumberSet {
    pub fn empty(base: FragmentNumber) -> Self {
        Self {
            bitmap_base: base,
            num_bits: 0,
            bitmap: [0u32; 8],
        }
    }

    pub fn is_set(&self, frag: FragmentNumber) -> bool {
        let diff = frag.0.wrapping_sub(self.bitmap_base.0);
        if diff >= self.num_bits {
            return false;
        }
        let word = (diff / 32) as usize;
        let mask = 1u32 << (31 - (diff % 32));
        word < 8 && (self.bitmap[word] & mask) != 0
    }

    pub fn set(&mut self, frag: FragmentNumber) -> Result<(), RtpsError> {
        let diff = frag.0.wrapping_sub(self.bitmap_base.0);
        if diff > 255 {
            return Err(RtpsError::InvalidParameterLength);
        }
        let word = (diff / 32) as usize;
        let mask = 1u32 << (31 - (diff % 32));
        self.bitmap[word] |= mask;
        if diff >= self.num_bits {
            self.num_bits = diff + 1;
        }
        Ok(())
    }

    fn num_words(&self) -> usize {
        self.num_bits.div_ceil(32) as usize
    }

    pub fn serialized_len(&self) -> usize {
        8 + 4 * self.num_words()
    }

    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let base = FragmentNumber(cur.read_u32()?);
        let num_bits = cur.read_u32()?;
        if num_bits > 256 {
            return Err(RtpsError::InvalidParameterLength);
        }
        let m = num_bits.div_ceil(32) as usize;
        let mut bitmap = [0u32; 8];
        for word in bitmap.iter_mut().take(m) {
            *word = cur.read_u32()?;
        }
        Ok(Self {
            bitmap_base: base,
            num_bits,
            bitmap,
        })
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_u32(self.bitmap_base.0)?;
        w.write_u32(self.num_bits)?;
        let m = self.num_words();
        for i in 0..m {
            w.write_u32(self.bitmap[i])?;
        }
        Ok(())
    }
}
