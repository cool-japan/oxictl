//! Compact binary log: `[timestamp_f32, CH×f32]` records.
//!
//! Each record stores a timestamp and `CH` channel values packed as little-endian
//! `f32`.  Designed for efficient telemetry storage and transmission in embedded
//! HIL (hardware-in-the-loop) testing scenarios.
//!
//! Only available with the `std` feature.
use std::vec::Vec;

/// A single binary log record: timestamp + CH channel values.
#[derive(Debug, Clone, Copy)]
pub struct BinaryRecord<const CH: usize> {
    /// Timestamp in seconds.
    pub timestamp: f32,
    /// Channel values.
    pub values: [f32; CH],
}

/// Binary log with `CH` channels per record.
///
/// Each record serialises to `(CH + 1) * 4` little-endian bytes.
pub struct BinaryLog<const CH: usize> {
    records: Vec<BinaryRecord<CH>>,
}

impl<const CH: usize> BinaryLog<CH> {
    /// Create an empty binary log.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Append a record with `t` as the timestamp and `values` as the channel data.
    pub fn push(&mut self, t: f32, values: [f32; CH]) {
        self.records.push(BinaryRecord {
            timestamp: t,
            values,
        });
    }

    /// Number of records stored.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the log contains no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Access the raw records slice.
    pub fn records(&self) -> &[BinaryRecord<CH>] {
        &self.records
    }

    /// Serialize all records to a little-endian byte stream.
    ///
    /// Layout per record: `[timestamp_f32_LE, ch0_f32_LE, …, ch{CH-1}_f32_LE]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let bytes_per_record = (CH + 1) * 4;
        let mut out = Vec::with_capacity(self.records.len() * bytes_per_record);
        for rec in &self.records {
            out.extend_from_slice(&rec.timestamp.to_le_bytes());
            for &v in rec.values.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Remove all records from the log.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Retrieve the timestamp of record at index `i`, if it exists.
    pub fn timestamp(&self, i: usize) -> Option<f32> {
        self.records.get(i).map(|r| r.timestamp)
    }

    /// Retrieve channel `ch` value of record at index `i`, if valid.
    pub fn channel_value(&self, i: usize, ch: usize) -> Option<f32> {
        if ch >= CH {
            return None;
        }
        self.records.get(i).map(|r| r.values[ch])
    }
}

impl<const CH: usize> Default for BinaryLog<CH> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_retrieve() {
        let mut log = BinaryLog::<3>::new();
        log.push(0.1, [1.0, 2.0, 3.0]);
        assert_eq!(log.len(), 1);
        assert!((log.timestamp(0).unwrap() - 0.1).abs() < 1e-6);
        assert!((log.channel_value(0, 0).unwrap() - 1.0).abs() < 1e-6);
        assert!((log.channel_value(0, 2).unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn to_bytes_length() {
        let mut log = BinaryLog::<2>::new();
        log.push(0.0, [1.0, 2.0]);
        log.push(1.0, [3.0, 4.0]);
        let bytes = log.to_bytes();
        // 2 records × (2+1) channels × 4 bytes = 24 bytes.
        assert_eq!(bytes.len(), 24);
    }

    #[test]
    fn to_bytes_little_endian_roundtrip() {
        let mut log = BinaryLog::<1>::new();
        log.push(1.5, [42.0]);
        let bytes = log.to_bytes();
        let t = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let v = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert!((t - 1.5).abs() < 1e-6, "t={}", t);
        assert!((v - 42.0).abs() < 1e-6, "v={}", v);
    }

    #[test]
    fn clear_empties_log() {
        let mut log = BinaryLog::<2>::new();
        log.push(0.0, [1.0, 2.0]);
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.to_bytes().len(), 0);
    }
}
