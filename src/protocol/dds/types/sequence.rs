use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};
use crate::protocol::dds::error::RtpsError;

pub const SEQUENCENUMBER_UNKNOWN: SequenceNumber = SequenceNumber { high: -1, low: 0 };
pub const SEQUENCENUMBER_ZERO: SequenceNumber = SequenceNumber { high: 0, low: 0 };

/// RTPS sequence number. Wire layout: [high i32][low u32], 8 bytes total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceNumber {
    pub high: i32,
    pub low: u32,
}

impl SequenceNumber {
    pub fn new(value: i64) -> Self {
        Self {
            high: (value >> 32) as i32,
            low: value as u32,
        }
    }

    pub fn to_i64(self) -> i64 {
        ((self.high as i64) << 32) | (self.low as u64 as i64)
    }

    pub fn is_unknown(&self) -> bool {
        self == &SEQUENCENUMBER_UNKNOWN
    }

    pub fn increment(&self) -> Self {
        Self::new(self.to_i64().saturating_add(1))
    }

    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let high = cur.read_i32()?;
        let low = cur.read_u32()?;
        Ok(Self { high, low })
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.high)?;
        w.write_u32(self.low)
    }
}

impl PartialOrd for SequenceNumber {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SequenceNumber {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.to_i64().cmp(&other.to_i64())
    }
}

/// Compact bitmap set of sequence numbers. Max 256 elements.
///
/// Wire layout: [bitmap_base: 8 bytes][num_bits: u32][bitmap: M × u32]
/// where M = ceil(num_bits / 32). Empty set (num_bits=0): M=0, total 12 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceNumberSet {
    pub bitmap_base: SequenceNumber,
    pub num_bits: u32,
    pub bitmap: [u32; 8],
}

impl SequenceNumberSet {
    pub fn empty(base: SequenceNumber) -> Self {
        Self {
            bitmap_base: base,
            num_bits: 0,
            bitmap: [0u32; 8],
        }
    }

    /// Check if a sequence number is in this set.
    pub fn is_set(&self, sn: SequenceNumber) -> bool {
        let diff = sn.to_i64() - self.bitmap_base.to_i64();
        if diff < 0 || diff >= self.num_bits as i64 {
            return false;
        }
        let bit = diff as u32;
        let word = (bit / 32) as usize;
        let mask = 1u32 << (31 - (bit % 32));
        word < 8 && (self.bitmap[word] & mask) != 0
    }

    /// Set a sequence number in the bitmap.
    pub fn set(&mut self, sn: SequenceNumber) -> Result<(), RtpsError> {
        let diff = sn.to_i64() - self.bitmap_base.to_i64();
        if !(0..=255).contains(&diff) {
            return Err(RtpsError::InvalidParameterLength);
        }
        let bit = diff as u32;
        let word = (bit / 32) as usize;
        let mask = 1u32 << (31 - (bit % 32));
        self.bitmap[word] |= mask;
        if bit >= self.num_bits {
            self.num_bits = bit + 1;
        }
        Ok(())
    }

    /// Number of bitmap words needed for this set.
    fn num_words(&self) -> usize {
        self.num_bits.div_ceil(32) as usize
    }

    /// Serialized byte length: 8 (base) + 4 (num_bits) + 4*M.
    pub fn serialized_len(&self) -> usize {
        12 + 4 * self.num_words()
    }

    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let bitmap_base = SequenceNumber::parse(cur)?;
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
            bitmap_base,
            num_bits,
            bitmap,
        })
    }

    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        self.bitmap_base.serialize(w)?;
        w.write_u32(self.num_bits)?;
        let m = self.num_words();
        for i in 0..m {
            w.write_u32(self.bitmap[i])?;
        }
        Ok(())
    }

    /// Iterate over sequence numbers that are set in the bitmap.
    pub fn iter(&self) -> SequenceNumberSetIter<'_> {
        SequenceNumberSetIter {
            set: self,
            bit_index: 0,
        }
    }
}

pub struct SequenceNumberSetIter<'a> {
    set: &'a SequenceNumberSet,
    bit_index: u32,
}

impl Iterator for SequenceNumberSetIter<'_> {
    type Item = SequenceNumber;

    fn next(&mut self) -> Option<Self::Item> {
        while self.bit_index < self.set.num_bits {
            let bit = self.bit_index;
            self.bit_index += 1;
            let word = (bit / 32) as usize;
            let mask = 1u32 << (31 - (bit % 32));
            if self.set.bitmap[word] & mask != 0 {
                let sn_value = self.set.bitmap_base.to_i64() + bit as i64;
                return Some(SequenceNumber::new(sn_value));
            }
        }
        None
    }
}
