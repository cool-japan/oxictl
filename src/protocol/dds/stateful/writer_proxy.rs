//! WriterProxy — per-writer reception state tracked by a StatefulReader.

use crate::protocol::dds::types::guid::Guid;
use crate::protocol::dds::types::locator::Locator;
use crate::protocol::dds::types::sequence::{SequenceNumber, SequenceNumberSet};

/// Tracks per-writer reception state inside a [`StatefulReader`].
///
/// The proxy records which sequence numbers have been received and
/// can build the `reader_sn_state` field of an ACKNACK.
///
/// [`StatefulReader`]: super::reader::StatefulReader
pub struct WriterProxy {
    /// Full GUID of the remote writer.
    pub remote_writer_guid: Guid,
    /// Unicast locators to which ACKNACK submessages are sent.
    pub unicast_locators: Vec<Locator>,
    /// Highest contiguous SN: everything `<= this` has been received.
    /// `None` means nothing received yet.
    pub highest_contiguous_sn: Option<SequenceNumber>,
    /// SNs received out-of-order (gap exists below them).
    pub received_out_of_order: Vec<SequenceNumber>,
    /// Monotonically increasing count included in each ACKNACK.
    pub acknack_count: i32,
}

impl WriterProxy {
    /// Create a new proxy with no received samples.
    pub fn new(guid: Guid, unicast_locators: Vec<Locator>) -> Self {
        Self {
            remote_writer_guid: guid,
            unicast_locators,
            highest_contiguous_sn: None,
            received_out_of_order: Vec::new(),
            acknack_count: 0,
        }
    }

    /// Record the reception of a DATA sample with the given sequence number.
    ///
    /// If `sn` fills the gap immediately above `highest_contiguous_sn`,
    /// the contiguous window is advanced as far as `received_out_of_order` allows.
    pub fn received(&mut self, sn: SequenceNumber) {
        // Compute what the "next expected" SN is.
        let next_expected = match self.highest_contiguous_sn {
            None => SequenceNumber::new(1),
            Some(h) => h.increment(),
        };

        if sn == next_expected {
            // Advance highest_contiguous_sn.
            self.highest_contiguous_sn = Some(sn);
            // Greedily advance further using the out-of-order buffer.
            let mut current = sn;
            loop {
                let next = current.increment();
                if let Some(pos) = self.received_out_of_order.iter().position(|&x| x == next) {
                    self.received_out_of_order.swap_remove(pos);
                    current = next;
                    self.highest_contiguous_sn = Some(next);
                } else {
                    break;
                }
            }
        } else {
            // Out-of-order or duplicate — add to set if not already present.
            let above_contiguous = match self.highest_contiguous_sn {
                None => sn.to_i64() > 0,
                Some(h) => sn > h,
            };
            if above_contiguous && !self.received_out_of_order.contains(&sn) {
                self.received_out_of_order.push(sn);
            }
        }
    }

    /// Build the `reader_sn_state` bitmap for an ACKNACK.
    ///
    /// The base is `highest_contiguous_sn + 1` (or SN(1) if nothing received).
    /// Bits are set for each SN in `[base, writer_last_sn]` that has NOT yet
    /// been received (i.e. not in `received_out_of_order` and above contiguous).
    /// The window is clamped to 256 SNs per the RTPS SequenceNumberSet constraint.
    pub fn build_missing_sn_set(&self, writer_last_sn: SequenceNumber) -> SequenceNumberSet {
        let base = match self.highest_contiguous_sn {
            None => SequenceNumber::new(1),
            Some(h) => h.increment(),
        };

        let mut set = SequenceNumberSet::empty(base);

        // Window: [base, writer_last_sn], clamped to 256 bits.
        let start_v = base.to_i64();
        let end_v = writer_last_sn.to_i64();
        if end_v < start_v {
            // Nothing to request — writer has nothing beyond what we know.
            return set;
        }

        let window = (end_v - start_v + 1).min(256) as u64;
        for offset in 0..window {
            let sn = SequenceNumber::new(start_v + offset as i64);
            // A SN is "missing" if it is not in received_out_of_order.
            if !self.received_out_of_order.contains(&sn) {
                // set() can only fail if diff is outside 0..=255, but we clamped window.
                let _ = set.set(sn);
            }
        }

        set
    }

    /// Increment and return the new ACKNACK count.
    pub fn next_acknack_count(&mut self) -> i32 {
        self.acknack_count = self.acknack_count.saturating_add(1);
        self.acknack_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::types::guid::{EntityId, GuidPrefix};

    fn sn(v: i64) -> SequenceNumber {
        SequenceNumber::new(v)
    }

    fn make_proxy() -> WriterProxy {
        let guid = Guid {
            prefix: GuidPrefix([0xAAu8; 12]),
            entity_id: EntityId {
                entity_key: [0, 0, 1],
                entity_kind: 0x02,
            },
        };
        WriterProxy::new(guid, Vec::new())
    }

    #[test]
    fn writer_proxy_received_in_order() {
        let mut proxy = make_proxy();
        proxy.received(sn(1));
        proxy.received(sn(2));
        proxy.received(sn(3));
        assert_eq!(proxy.highest_contiguous_sn, Some(sn(3)));
        assert!(proxy.received_out_of_order.is_empty());
    }

    #[test]
    fn writer_proxy_received_gap() {
        let mut proxy = make_proxy();
        // Receive SN(1), then SN(3) out of order.
        proxy.received(sn(1));
        proxy.received(sn(3));
        assert_eq!(proxy.highest_contiguous_sn, Some(sn(1)));
        assert!(proxy.received_out_of_order.contains(&sn(3)));

        // Receive the missing SN(2): should advance to SN(3).
        proxy.received(sn(2));
        assert_eq!(proxy.highest_contiguous_sn, Some(sn(3)));
        assert!(proxy.received_out_of_order.is_empty());
    }

    #[test]
    fn writer_proxy_build_missing_set() {
        let mut proxy = make_proxy();
        proxy.received(sn(1));
        proxy.received(sn(3)); // gap at SN(2)

        let set = proxy.build_missing_sn_set(sn(4));
        // SN(2) is missing, SN(4) is missing (not received at all).
        // SN(1) is below base, SN(3) is in received_out_of_order so NOT missing.
        assert!(set.is_set(sn(2)), "SN(2) should be requested");
        assert!(!set.is_set(sn(1)), "SN(1) already received — not missing");
        assert!(!set.is_set(sn(3)), "SN(3) already received — not missing");
        assert!(set.is_set(sn(4)), "SN(4) should be requested");
    }
}
