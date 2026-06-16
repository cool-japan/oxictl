//! High-level DDS `Participant` — manages discovery and endpoint matching.
//!
//! # Auto-discovery via multicast SPDP
//!
//! `Participant::new(domain_id, guid_prefix, qos)` creates an ephemeral SEDP unicast
//! socket and also attempts to join the RTPS SPDP multicast group for the given domain.
//! When two participants share the same domain and multicast is available (loopback
//! multicast must be enabled by the OS), they will discover each other automatically
//! via `spin_once` without any `add_peer` calls.
//!
//! If multicast is not available (restricted CI, firewalled environment) the
//! participant falls back silently to explicit-peer mode: call `add_peer` with each
//! remote participant's `local_metatraffic_addr()`.

use std::collections::HashSet;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::{Duration, Instant};

use crate::protocol::dds::discovery::endpoint_data::{
    PublicationBuiltinTopicData, SubscriptionBuiltinTopicData,
};
use crate::protocol::dds::discovery::qos_profile::QosProfile;
use crate::protocol::dds::discovery::sedp::SedpParticipant;
use crate::protocol::dds::discovery::spdp::SpdpParticipant;
use crate::protocol::dds::discovery::IncomingResult;
use crate::protocol::dds::transport::{
    bind_multicast_reuse, metatraffic_multicast_port, TransportConfig, UdpTransport,
    SPDP_MULTICAST_IPV4,
};
use crate::protocol::dds::types::guid::{Guid, GuidPrefix, ENTITYID_PARTICIPANT};

use super::dds_type::DdsType;
use super::entity_id::EntityIdAllocator;
use super::error::DdsApiError;
use super::publisher::{addr_to_locator, Publisher, WriterEntry};
use super::subscription::{
    guid_to_bytes, pub_data_locators, sub_data_locator, ReaderEntry, Subscription,
};

/// Interval between SPDP multicast beacons.
const SPDP_BEACON_PERIOD: Duration = Duration::from_millis(1000);

/// High-level DDS participant.
///
/// Owns all `Publisher<T>` and `Subscription<T>` endpoints.  Call
/// `spin_once` in a loop to drive discovery and data exchange.
pub struct Participant {
    domain_id: u16,
    guid_prefix: GuidPrefix,
    /// SPDP multicast instance. `None` if multicast is unavailable on this host.
    spdp: Option<SpdpParticipant>,
    /// Locator for sending SPDP beacons to the multicast group.
    spdp_multicast_locator: Option<crate::protocol::dds::types::locator::Locator>,
    /// Last time we sent an SPDP beacon.
    last_beacon_at: Option<Instant>,
    /// GuidPrefix bytes of peers already promoted to `peer_metatraffic_locators`.
    auto_discovered_peers: HashSet<[u8; 12]>,
    sedp: SedpParticipant,
    peer_metatraffic_locators: Vec<crate::protocol::dds::types::locator::Locator>,
    writers: Vec<WriterEntry>,
    readers: Vec<ReaderEntry>,
    /// GUIDs of remote publications already matched to local readers.
    matched_pubs: HashSet<[u8; 16]>,
    /// GUIDs of remote subscriptions already matched to local writers.
    matched_subs: HashSet<[u8; 16]>,
}

impl Participant {
    /// Create a new participant for the given domain.
    ///
    /// Binds an ephemeral UDP port for SEDP metatraffic and attempts to join
    /// the SPDP multicast group for `domain_id`.  If multicast is unavailable,
    /// falls back to explicit-peer mode: call `add_peer` with remote addresses.
    pub fn new(
        domain_id: u16,
        guid_prefix: GuidPrefix,
        qos: QosProfile,
    ) -> Result<Self, DdsApiError> {
        // SEDP unicast socket (ephemeral).
        let bind = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0));
        let cfg = TransportConfig {
            read_timeout: Some(Duration::from_millis(5)),
            ..TransportConfig::unicast(bind)
        };
        let sedp_transport = UdpTransport::new(cfg)?;
        let sedp_local_addr = sedp_transport.local_addr()?;
        let sedp = SedpParticipant::with_transport_and_qos(guid_prefix, sedp_transport, qos);

        // SPDP multicast socket (best-effort — falls back to None if OS denies).
        let mc_port = metatraffic_multicast_port(domain_id);
        let unicast_locator = addr_to_locator(sedp_local_addr)?;
        let mut unicast_locators: heapless::Vec<crate::protocol::dds::types::locator::Locator, 4> =
            heapless::Vec::new();
        let _ = unicast_locators.push(unicast_locator);

        let spdp_multicast_locator: Option<crate::protocol::dds::types::locator::Locator>;
        let spdp = match bind_multicast_reuse(mc_port, SPDP_MULTICAST_IPV4) {
            Ok(mc_sock) => {
                let mc_transport = UdpTransport::from_socket(mc_sock);
                // Best-effort: ignore error on set_read_timeout for the multicast socket.
                let _ = mc_transport.set_read_timeout(Some(Duration::from_millis(1)));
                // Compute the multicast locator we'll send beacons TO.
                let mc_loc = crate::protocol::dds::types::locator::Locator::udp_v4(
                    mc_port as u32,
                    SPDP_MULTICAST_IPV4,
                );
                spdp_multicast_locator = Some(mc_loc);
                match SpdpParticipant::with_multicast_transport(
                    domain_id,
                    guid_prefix,
                    mc_transport,
                    unicast_locators,
                ) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        eprintln!(
                            "oxictl: SPDP multicast init failed ({e}); falling back to explicit add_peer"
                        );
                        None
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "oxictl: multicast socket bind failed ({e}); SPDP auto-discovery disabled"
                );
                spdp_multicast_locator = None;
                None
            }
        };

        Ok(Self {
            domain_id,
            guid_prefix,
            spdp,
            spdp_multicast_locator,
            last_beacon_at: None,
            auto_discovered_peers: HashSet::new(),
            sedp,
            peer_metatraffic_locators: Vec::new(),
            writers: Vec::new(),
            readers: Vec::new(),
            matched_pubs: HashSet::new(),
            matched_subs: HashSet::new(),
        })
    }

    /// Domain ID this participant operates in.
    pub fn domain_id(&self) -> u16 {
        self.domain_id
    }

    /// Returns true if multicast SPDP auto-discovery is active on this participant.
    pub fn has_multicast(&self) -> bool {
        self.spdp.is_some()
    }

    /// Return the local socket address of the SEDP metatraffic transport.
    ///
    /// Needed for `add_peer` on the remote participant in tests and explicit
    /// peer configurations.
    pub fn local_metatraffic_addr(&self) -> Result<SocketAddr, DdsApiError> {
        Ok(self.sedp.local_addr()?)
    }

    /// Register a remote participant's metatraffic address for SEDP announcements.
    ///
    /// SEDP `announce_publication` / `announce_subscription` will be called
    /// for every locator registered here.
    pub fn add_peer(&mut self, addr: SocketAddr) -> Result<(), DdsApiError> {
        let locator = addr_to_locator(addr)?;
        self.peer_metatraffic_locators.push(locator);
        Ok(())
    }

    /// Create a `Publisher<T>` for the given topic, using the participant's QoS.
    pub fn create_publisher<T: DdsType>(
        &mut self,
        topic_name: &str,
        qos: &QosProfile,
    ) -> Result<Publisher<T>, DdsApiError> {
        let entity_id = EntityIdAllocator::next_writer();
        let entry = WriterEntry::new(self.guid_prefix, entity_id, topic_name, T::TYPE_NAME, qos)?;
        let idx = self.writers.len();
        self.writers.push(entry);
        Ok(Publisher::new(idx))
    }

    /// Create a `Subscription<T>` for the given topic, using the participant's QoS.
    pub fn create_subscription<T: DdsType>(
        &mut self,
        topic_name: &str,
        qos: &QosProfile,
    ) -> Result<Subscription<T>, DdsApiError> {
        let entity_id = EntityIdAllocator::next_reader();
        let entry = ReaderEntry::new(self.guid_prefix, entity_id, topic_name, T::TYPE_NAME, qos)?;
        let idx = self.readers.len();
        self.readers.push(entry);
        Ok(Subscription::new(idx))
    }

    /// Publish a value via the given publisher handle.
    ///
    /// Convenience wrapper so the caller does not need to access the internal
    /// `WriterEntry` directly.
    pub fn publish<T: DdsType>(
        &mut self,
        publisher: &Publisher<T>,
        value: &T,
    ) -> Result<crate::protocol::dds::types::sequence::SequenceNumber, DdsApiError> {
        let entry = self
            .writers
            .get_mut(publisher.entry_idx)
            .ok_or(DdsApiError::Serialization("invalid publisher handle"))?;
        publisher.publish(entry, value)
    }

    /// Take all buffered samples from the given subscription handle.
    pub fn take<T: DdsType>(
        &mut self,
        subscription: &Subscription<T>,
    ) -> Vec<super::dds_type::Sample<T>> {
        match self.readers.get_mut(subscription.entry_idx) {
            Some(entry) => subscription.take(entry),
            None => Vec::new(),
        }
    }

    /// Number of samples buffered in the subscription queue.
    pub fn queue_depth<T: DdsType>(&self, subscription: &Subscription<T>) -> usize {
        self.readers
            .get(subscription.entry_idx)
            .map(|e| e.raw_queue.len())
            .unwrap_or(0)
    }

    /// Drive one iteration of the discovery and data exchange loop.
    ///
    /// Steps:
    /// 0a. Send SPDP multicast beacon if due.
    /// 0b. Process incoming SPDP and auto-promote newly-discovered peers.
    /// 1. Announce unannounced publishers and subscriptions to all peers.
    /// 2. Call `sedp.process_incoming` to receive remote announcements.
    /// 3. Match newly-discovered remote publications to local subscriptions.
    /// 4. Match newly-discovered remote subscriptions to local writers.
    /// 5. Drive writer `process_incoming` (ACKNACK handling) and `send_heartbeat_if_due`.
    /// 6. Drive reader `recv` and buffer any received payloads.
    pub fn spin_once(&mut self) -> Result<IncomingResult, DdsApiError> {
        // 0a. Send SPDP beacon if due.
        if let (Some(spdp), Some(mc_loc)) = (&mut self.spdp, &self.spdp_multicast_locator) {
            let send_now = self
                .last_beacon_at
                .map(|t| t.elapsed() >= SPDP_BEACON_PERIOD)
                .unwrap_or(true);
            if send_now {
                let _ = spdp.send_beacon_to(mc_loc);
                self.last_beacon_at = Some(Instant::now());
            }
        }

        // 0b. Process incoming SPDP and auto-promote newly-discovered peers.
        //     Collect candidates first to avoid borrow-checker conflicts when
        //     mutating `self.peer_metatraffic_locators` and `self.auto_discovered_peers`.
        let mut new_peers: Vec<([u8; 12], crate::protocol::dds::types::locator::Locator)> =
            Vec::new();
        if let Some(spdp) = self.spdp.as_mut() {
            let _ = spdp.process_incoming();
            for disc in spdp.discovered() {
                let prefix_bytes = disc.data.participant_guid.prefix.0;
                if let Some(loc) = disc.data.metatraffic_unicast_locators.first() {
                    new_peers.push((prefix_bytes, *loc));
                }
            }
        }
        for (prefix, loc) in new_peers {
            if self.auto_discovered_peers.insert(prefix) {
                self.peer_metatraffic_locators.push(loc);
            }
        }

        // 1. Announce endpoints that have not yet been announced.
        self.announce_endpoints()?;

        // 2. Receive and store remote endpoint announcements.
        let result = self.sedp.process_incoming()?;

        // 3. Match remote publications → local subscriptions.
        self.match_publications_to_readers()?;

        // 4. Match remote subscriptions → local writers.
        self.match_subscriptions_to_writers()?;

        // 5. Heartbeat cycle for writers.
        for entry in self.writers.iter_mut() {
            let _ = entry.writer.process_incoming();
            let _ = entry.writer.send_heartbeat_if_due();
        }

        // 6. Receive data for readers.
        for entry in self.readers.iter_mut() {
            // Drain multiple packets per spin: loop until WouldBlock / timeout.
            loop {
                match entry.reader.recv() {
                    Ok(Some(sample)) => {
                        let guid_bytes = guid_to_bytes(&sample.writer_guid);
                        // Best-effort: if queue is full, drop the oldest.
                        if entry.raw_queue.is_full() {
                            // Remove the first element to make room.
                            if !entry.raw_queue.is_empty() {
                                entry.raw_queue.swap_remove(0);
                            }
                        }
                        let _ = entry.raw_queue.push((sample.data, guid_bytes));
                    }
                    Ok(None) => break,
                    Err(_) => break,
                }
            }
        }

        Ok(result)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn announce_endpoints(&mut self) -> Result<(), DdsApiError> {
        for locator in self.peer_metatraffic_locators.clone() {
            // Announce writers.
            for entry in self.writers.iter_mut() {
                self.sedp.announce_publication(&entry.pub_data, &locator)?;
                entry.announced = true;
            }
            // Announce readers.
            for entry in self.readers.iter_mut() {
                self.sedp.announce_subscription(&entry.sub_data, &locator)?;
                entry.announced = true;
            }
        }
        Ok(())
    }

    fn match_publications_to_readers(&mut self) -> Result<(), DdsApiError> {
        // Collect discovered pubs and our local sub data to avoid borrow conflicts.
        let discovered: Vec<PublicationBuiltinTopicData> =
            self.sedp.discovered_publications().to_vec();

        for pub_data in &discovered {
            let guid_bytes = guid_to_bytes(&pub_data.endpoint_guid);
            if self.matched_pubs.contains(&guid_bytes) {
                continue;
            }
            // Find all local readers on the same (topic_name, type_name).
            let writer_guid = pub_data.endpoint_guid;
            let writer_locators: Vec<_> = pub_data_locators(pub_data);
            for entry in self.readers.iter_mut() {
                if entry.sub_data.topic_name == pub_data.topic_name
                    && entry.sub_data.type_name == pub_data.type_name
                {
                    entry
                        .reader
                        .add_matched_writer(writer_guid, writer_locators.clone());
                }
            }
            self.matched_pubs.insert(guid_bytes);
        }
        Ok(())
    }

    fn match_subscriptions_to_writers(&mut self) -> Result<(), DdsApiError> {
        let discovered: Vec<SubscriptionBuiltinTopicData> =
            self.sedp.discovered_subscriptions().to_vec();

        for sub_data in &discovered {
            let guid_bytes = guid_to_bytes(&sub_data.endpoint_guid);
            if self.matched_subs.contains(&guid_bytes) {
                continue;
            }
            let reader_guid = sub_data.endpoint_guid;
            let reader_locators: Vec<_> = match sub_data_locator(sub_data) {
                Some(loc) => vec![loc],
                None => continue,
            };
            for entry in self.writers.iter_mut() {
                if entry.pub_data.topic_name == sub_data.topic_name
                    && entry.pub_data.type_name == sub_data.type_name
                {
                    entry
                        .writer
                        .add_matched_reader(reader_guid, reader_locators.clone());
                }
            }
            self.matched_subs.insert(guid_bytes);
        }
        Ok(())
    }

    /// Return the 16-byte GUID of the writer backing `Publisher<T>`.
    ///
    /// Used by `ServiceClient` to capture its request-writer GUID for
    /// request/reply correlation.  Returns `None` if the index is out of range.
    pub fn publisher_guid<T: DdsType>(&self, pubr: &Publisher<T>) -> Option<[u8; 16]> {
        self.writers
            .get(pubr.entry_idx)
            .map(|e| guid_to_bytes(&e.pub_data.endpoint_guid))
    }

    /// Return the GUID of this participant.
    pub fn guid(&self) -> Guid {
        Guid::new(self.guid_prefix, ENTITYID_PARTICIPANT)
    }

    /// Number of registered peer metatraffic locators (explicit + auto-discovered).
    pub fn peer_count(&self) -> usize {
        self.peer_metatraffic_locators.len()
    }
}

// ─── Inline tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn participant_new_with_domain_constructs() {
        let guid = GuidPrefix([0xAA; 12]);
        let p = Participant::new(
            0,
            guid,
            crate::protocol::dds::discovery::qos_profile::QosProfile::ros2_default(),
        )
        .expect("participant construction failed");
        assert_eq!(p.domain_id(), 0);
        assert!(p.local_metatraffic_addr().is_ok());
    }

    #[test]
    fn participant_peer_count_starts_zero() {
        let p = Participant::new(
            1,
            GuidPrefix([0xBB; 12]),
            crate::protocol::dds::discovery::qos_profile::QosProfile::ros2_default(),
        )
        .expect("participant construction failed");
        // auto-discovered peers start at 0 (no SPDP traffic yet)
        assert_eq!(p.peer_count(), 0);
    }
}
