//! History cache for best-effort StatelessWriter.

use std::collections::VecDeque;

use crate::protocol::dds::types::sequence::SequenceNumber;
use crate::protocol::dds::types::time::Time;

/// A single cached change (data sample with its sequence number).
pub struct CacheChange {
    /// RTPS sequence number of this change.
    pub sequence_number: SequenceNumber,
    /// Serialized payload bytes.
    pub data: Vec<u8>,
    /// Optional source timestamp.
    pub timestamp: Option<Time>,
}

/// Bounded history cache backed by a [`VecDeque`].
///
/// Oldest entries are evicted when the cache is full and a new entry is added.
pub struct HistoryCache {
    changes: VecDeque<CacheChange>,
    capacity: usize,
}

impl HistoryCache {
    /// Create a new cache with the given maximum `capacity`.
    ///
    /// A capacity of 0 means no changes can ever be stored; [`add`](Self::add)
    /// will always return `false`.
    pub fn new(capacity: usize) -> Self {
        Self {
            changes: VecDeque::with_capacity(capacity.min(1024)),
            capacity,
        }
    }

    /// Add a new change.
    ///
    /// If the cache is already at capacity the oldest entry is evicted first.
    /// Returns `true` on success, `false` if `capacity == 0`.
    pub fn add(&mut self, sn: SequenceNumber, data: Vec<u8>, ts: Option<Time>) -> bool {
        if self.capacity == 0 {
            return false;
        }
        if self.changes.len() >= self.capacity {
            self.changes.pop_front();
        }
        self.changes.push_back(CacheChange {
            sequence_number: sn,
            data,
            timestamp: ts,
        });
        true
    }

    /// Look up a cached change by its sequence number.
    pub fn get(&self, sn: SequenceNumber) -> Option<&CacheChange> {
        self.changes.iter().find(|c| c.sequence_number == sn)
    }

    /// Remove the cached change with the given sequence number.
    ///
    /// Returns `true` if a matching entry was found and removed.
    pub fn remove(&mut self, sn: SequenceNumber) -> bool {
        if let Some(pos) = self.changes.iter().position(|c| c.sequence_number == sn) {
            self.changes.remove(pos);
            true
        } else {
            false
        }
    }

    /// Minimum (oldest) sequence number in the cache, or `None` if empty.
    pub fn min_sn(&self) -> Option<SequenceNumber> {
        self.changes.iter().map(|c| c.sequence_number).min()
    }

    /// Maximum (newest) sequence number in the cache, or `None` if empty.
    pub fn max_sn(&self) -> Option<SequenceNumber> {
        self.changes.iter().map(|c| c.sequence_number).max()
    }

    /// Number of changes currently in the cache.
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    /// Returns `true` if the cache contains no changes.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sn(v: i64) -> SequenceNumber {
        SequenceNumber::new(v)
    }

    #[test]
    fn history_cache_add_get() {
        let mut cache = HistoryCache::new(8);
        assert!(cache.add(sn(1), b"alpha".to_vec(), None));
        assert!(cache.add(sn(2), b"beta".to_vec(), None));
        assert!(cache.add(sn(3), b"gamma".to_vec(), None));

        assert_eq!(
            cache.get(sn(1)).map(|c| c.data.as_slice()),
            Some(b"alpha".as_ref())
        );
        assert_eq!(
            cache.get(sn(2)).map(|c| c.data.as_slice()),
            Some(b"beta".as_ref())
        );
        assert_eq!(
            cache.get(sn(3)).map(|c| c.data.as_slice()),
            Some(b"gamma".as_ref())
        );
        assert!(cache.get(sn(4)).is_none());
    }

    #[test]
    fn history_cache_eviction() {
        let mut cache = HistoryCache::new(2);
        assert!(cache.add(sn(1), b"first".to_vec(), None));
        assert!(cache.add(sn(2), b"second".to_vec(), None));
        // Adding a third entry must evict sn(1).
        assert!(cache.add(sn(3), b"third".to_vec(), None));

        assert_eq!(cache.len(), 2);
        assert!(
            cache.get(sn(1)).is_none(),
            "oldest entry should have been evicted"
        );
        assert!(cache.get(sn(2)).is_some());
        assert!(cache.get(sn(3)).is_some());
    }

    #[test]
    fn history_cache_remove() {
        let mut cache = HistoryCache::new(8);
        cache.add(sn(10), b"a".to_vec(), None);
        cache.add(sn(11), b"b".to_vec(), None);
        cache.add(sn(12), b"c".to_vec(), None);

        assert!(cache.remove(sn(11)));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(sn(11)).is_none());
        assert!(cache.get(sn(10)).is_some());
        assert!(cache.get(sn(12)).is_some());
    }

    #[test]
    fn history_cache_min_max() {
        let mut cache = HistoryCache::new(4);
        assert!(cache.min_sn().is_none());
        assert!(cache.max_sn().is_none());

        cache.add(sn(5), vec![], None);
        cache.add(sn(3), vec![], None);
        cache.add(sn(7), vec![], None);

        assert_eq!(cache.min_sn(), Some(sn(3)));
        assert_eq!(cache.max_sn(), Some(sn(7)));

        // After eviction (capacity=4 → add a 4th then 5th entry)
        cache.add(sn(1), vec![], None); // [5,3,7,1]
        cache.add(sn(9), vec![], None); // evicts 5; [3,7,1,9]
        assert_eq!(cache.min_sn(), Some(sn(1)));
        assert_eq!(cache.max_sn(), Some(sn(9)));
    }
}
