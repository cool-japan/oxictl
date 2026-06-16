//! ReaderProxy — per-reader ACK state tracked by a StatefulWriter.

use crate::protocol::dds::message::submessage::AckNack;
use crate::protocol::dds::types::guid::Guid;
use crate::protocol::dds::types::locator::Locator;
use crate::protocol::dds::types::sequence::SequenceNumber;

/// Tracks per-reader ACK state inside a [`StatefulWriter`].
///
/// The proxy records the highest acknowledged SN and any SNs the reader
/// has explicitly requested retransmission of via NACK bitmask.
///
/// [`StatefulWriter`]: super::writer::StatefulWriter
pub struct ReaderProxy {
    /// Full GUID of the remote reader.
    pub remote_reader_guid: Guid,
    /// Unicast locators used to deliver DATA and HEARTBEAT.
    pub unicast_locators: Vec<Locator>,
    /// Highest SN the reader has fully acknowledged (everything below
    /// `reader_sn_state.bitmap_base` in the most recent ACKNACK).
    pub acked_sn: Option<SequenceNumber>,
    /// SNs the reader explicitly requested retransmit via NACK bitmap.
    pub requested_changes: Vec<SequenceNumber>,
}

impl ReaderProxy {
    /// Create a new proxy with no acknowledgements.
    pub fn new(guid: Guid, unicast_locators: Vec<Locator>) -> Self {
        Self {
            remote_reader_guid: guid,
            unicast_locators,
            acked_sn: None,
            requested_changes: Vec::new(),
        }
    }

    /// Process an incoming ACKNACK submessage from this reader.
    ///
    /// Updates `acked_sn` to `bitmap_base - 1` (everything below the base is
    /// acknowledged) and populates `requested_changes` with the SNs whose bits
    /// are SET in the bitmap (those are the gaps the reader wants retransmitted).
    ///
    /// Returns `true` if there are outstanding changes to retransmit.
    pub fn process_acknack(&mut self, acknack: &AckNack) -> bool {
        // The reader has acknowledged everything strictly before bitmap_base.
        let base_v = acknack.reader_sn_state.bitmap_base.to_i64();
        if base_v > 1 {
            let acked = SequenceNumber::new(base_v - 1);
            self.acked_sn = Some(match self.acked_sn {
                None => acked,
                Some(prev) => {
                    if acked > prev {
                        acked
                    } else {
                        prev
                    }
                }
            });
        }

        // Collect all SNs that are SET in the bitmap — those are missing/requested.
        self.requested_changes.clear();
        for sn in acknack.reader_sn_state.iter() {
            if !self.requested_changes.contains(&sn) {
                self.requested_changes.push(sn);
            }
        }

        !self.requested_changes.is_empty()
    }

    /// Drain (remove and return) all currently requested changes.
    ///
    /// Called by the writer before transmitting retransmissions.
    pub fn drain_requested(&mut self) -> Vec<SequenceNumber> {
        let mut out = Vec::new();
        core::mem::swap(&mut self.requested_changes, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::byte_cursor::Endianness;
    use crate::protocol::dds::message::submessage::AckNack;
    use crate::protocol::dds::types::guid::{EntityId, GuidPrefix, ENTITYID_UNKNOWN};
    use crate::protocol::dds::types::sequence::{SequenceNumber, SequenceNumberSet};

    fn sn(v: i64) -> SequenceNumber {
        SequenceNumber::new(v)
    }

    fn make_proxy() -> ReaderProxy {
        let guid = Guid {
            prefix: GuidPrefix([0xBBu8; 12]),
            entity_id: EntityId {
                entity_key: [0, 0, 7],
                entity_kind: 0x04,
            },
        };
        ReaderProxy::new(guid, Vec::new())
    }

    #[test]
    fn reader_proxy_process_acknack() {
        let mut proxy = make_proxy();

        // Build an AckNack: bitmap_base=3, bits for SN(3) and SN(5) are set.
        // SN(3) is at bit-offset 0 (base=3), SN(5) is at bit-offset 2.
        let mut sn_set = SequenceNumberSet::empty(sn(3));
        sn_set.set(sn(3)).unwrap();
        sn_set.set(sn(5)).unwrap();

        let acknack = AckNack {
            endianness: Endianness::Little,
            final_flag: true,
            reader_id: ENTITYID_UNKNOWN,
            writer_id: ENTITYID_UNKNOWN,
            reader_sn_state: sn_set,
            count: 1,
        };

        let has_changes = proxy.process_acknack(&acknack);
        assert!(has_changes);

        // acked_sn = bitmap_base - 1 = 3 - 1 = 2
        assert_eq!(proxy.acked_sn, Some(sn(2)));

        // requested_changes must contain SN(3) and SN(5).
        let mut req = proxy.requested_changes.clone();
        req.sort();
        assert_eq!(req, vec![sn(3), sn(5)]);
    }
}
