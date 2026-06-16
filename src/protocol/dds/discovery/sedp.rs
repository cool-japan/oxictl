//! SEDP (Simple Endpoint Discovery Protocol) participant.
//!
//! Manages announcement of local endpoint (publication/subscription) metadata
//! and discovery of remote endpoints from other RTPS participants.
//! Builds on the Phase 22.2 UDPv4 transport and Phase 22.1 RTPS wire protocol.

use std::net::SocketAddr;
use std::vec::Vec;

use heapless::Vec as HVec;

use crate::protocol::dds::byte_cursor::Endianness;
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::message::submessage::Data;
use crate::protocol::dds::message::{Message, MessageHeader, Submessage};
use crate::protocol::dds::transport::{TransportError, UdpTransport};
use crate::protocol::dds::types::guid::{
    GuidPrefix, ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER,
    ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_WRITER, ENTITYID_UNKNOWN, PROTOCOL_VERSION_2_3,
    VENDOR_ID_OXICTL,
};
use crate::protocol::dds::types::locator::Locator;
use crate::protocol::dds::types::sequence::SequenceNumber;

use super::endpoint_data::{PublicationBuiltinTopicData, SubscriptionBuiltinTopicData};
use super::error::DiscoveryError;
use super::qos_match::match_endpoint_qos;
use super::qos_profile::QosProfile;

/// Counts of compatible vs. QoS-rejected endpoints from one `process_incoming` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IncomingResult {
    pub compatible_pubs: usize,
    pub rejected_pubs: usize,
    pub compatible_subs: usize,
    pub rejected_subs: usize,
}

/// RTPS SEDP participant — announces and discovers endpoint metadata.
///
/// Sends DATA submessages carrying `PublicationBuiltinTopicData` or
/// `SubscriptionBuiltinTopicData` CDR payloads, and processes incoming DATA
/// submessages on the builtin publications/subscriptions topics.
pub struct SedpParticipant {
    transport: UdpTransport,
    guid_prefix: GuidPrefix,
    local_qos: QosProfile,
    discovered_publications: Vec<PublicationBuiltinTopicData>,
    rejected_publications: Vec<PublicationBuiltinTopicData>,
    discovered_subscriptions: Vec<SubscriptionBuiltinTopicData>,
    rejected_subscriptions: Vec<SubscriptionBuiltinTopicData>,
    pub_sn: i64,
    sub_sn: i64,
}

impl SedpParticipant {
    /// Create an SEDP participant with a pre-built transport.
    ///
    /// Useful for testing with ephemeral ports; for production use, bind to
    /// the metatraffic unicast port derived from `domain_id` and `participant_id`.
    pub fn with_transport(guid_prefix: GuidPrefix, transport: UdpTransport) -> Self {
        Self {
            transport,
            guid_prefix,
            local_qos: QosProfile::ros2_default(),
            discovered_publications: Vec::new(),
            rejected_publications: Vec::new(),
            discovered_subscriptions: Vec::new(),
            rejected_subscriptions: Vec::new(),
            pub_sn: 1,
            sub_sn: 1,
        }
    }

    /// Create an SEDP participant with a pre-built transport and a specific local QoS profile.
    ///
    /// The `local_qos` profile is used for QoS compatibility filtering of discovered endpoints.
    /// Incompatible remote endpoints are stored in `rejected_publications`/`rejected_subscriptions`
    /// rather than in the discovered lists.
    pub fn with_transport_and_qos(
        guid_prefix: GuidPrefix,
        transport: UdpTransport,
        local_qos: QosProfile,
    ) -> Self {
        Self {
            transport,
            guid_prefix,
            local_qos,
            discovered_publications: Vec::new(),
            rejected_publications: Vec::new(),
            discovered_subscriptions: Vec::new(),
            rejected_subscriptions: Vec::new(),
            pub_sn: 1,
            sub_sn: 1,
        }
    }

    /// Announce a local publication endpoint to a remote participant's metatraffic locator.
    ///
    /// Encodes `pub_data` as a CDR PL_CDR_LE payload and sends it as a DATA submessage
    /// with writer ID `ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER`.
    pub fn announce_publication(
        &mut self,
        pub_data: &PublicationBuiltinTopicData,
        locator: &Locator,
    ) -> Result<(), DiscoveryError> {
        let mut payload_buf = [0u8; 1024];
        let payload_len = pub_data
            .serialize_to_payload(&mut payload_buf)
            .map_err(DiscoveryError::Parse)?;
        let payload = &payload_buf[..payload_len];

        let sn = SequenceNumber::new(self.pub_sn);
        self.pub_sn += 1;

        let data_sub = Data {
            endianness: Endianness::Little,
            inline_qos_flag: false,
            data_flag: true,
            key_flag: false,
            non_standard_payload_flag: false,
            extra_flags: 0,
            reader_id: ENTITYID_UNKNOWN,
            writer_id: ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER,
            writer_sn: sn,
            inline_qos: None,
            serialized_payload: payload,
        };

        let mut subs: HVec<Submessage<'_>, 64> = HVec::new();
        subs.push(Submessage::Data(data_sub))
            .map_err(|_| DiscoveryError::Parse(RtpsError::TooManySubmessages))?;

        let msg = Message {
            header: MessageHeader {
                version: PROTOCOL_VERSION_2_3,
                vendor_id: VENDOR_ID_OXICTL,
                guid_prefix: self.guid_prefix,
            },
            submessages: subs,
        };

        self.transport.send_to(&msg, locator)?;
        Ok(())
    }

    /// Announce a local subscription endpoint to a remote participant's metatraffic locator.
    ///
    /// Encodes `sub_data` as a CDR PL_CDR_LE payload and sends it as a DATA submessage
    /// with writer ID `ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_WRITER`.
    pub fn announce_subscription(
        &mut self,
        sub_data: &SubscriptionBuiltinTopicData,
        locator: &Locator,
    ) -> Result<(), DiscoveryError> {
        let mut payload_buf = [0u8; 1024];
        let payload_len = sub_data
            .serialize_to_payload(&mut payload_buf)
            .map_err(DiscoveryError::Parse)?;
        let payload = &payload_buf[..payload_len];

        let sn = SequenceNumber::new(self.sub_sn);
        self.sub_sn += 1;

        let data_sub = Data {
            endianness: Endianness::Little,
            inline_qos_flag: false,
            data_flag: true,
            key_flag: false,
            non_standard_payload_flag: false,
            extra_flags: 0,
            reader_id: ENTITYID_UNKNOWN,
            writer_id: ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_WRITER,
            writer_sn: sn,
            inline_qos: None,
            serialized_payload: payload,
        };

        let mut subs: HVec<Submessage<'_>, 64> = HVec::new();
        subs.push(Submessage::Data(data_sub))
            .map_err(|_| DiscoveryError::Parse(RtpsError::TooManySubmessages))?;

        let msg = Message {
            header: MessageHeader {
                version: PROTOCOL_VERSION_2_3,
                vendor_id: VENDOR_ID_OXICTL,
                guid_prefix: self.guid_prefix,
            },
            submessages: subs,
        };

        self.transport.send_to(&msg, locator)?;
        Ok(())
    }

    /// Receive and process one incoming SEDP message.
    ///
    /// Returns an [`IncomingResult`] counting compatible vs. QoS-rejected endpoints.
    /// Returns `IncomingResult::default()` on socket timeout (no data available).
    pub fn process_incoming(&mut self) -> Result<IncomingResult, DiscoveryError> {
        let mut buf = [0u8; 65535];
        let (msg, _sender) = match self.transport.recv_into(&mut buf) {
            Ok(r) => r,
            Err(TransportError::Io(ref e))
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                return Ok(IncomingResult::default());
            }
            Err(e) => return Err(DiscoveryError::Transport(e)),
        };

        let mut result = IncomingResult::default();

        for sub in msg.iter_submessages() {
            if let Submessage::Data(data) = sub {
                if data.data_flag && !data.serialized_payload.is_empty() {
                    if data.writer_id == ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER {
                        if let Ok(pub_data) =
                            PublicationBuiltinTopicData::parse_from_payload(data.serialized_payload)
                        {
                            if self.upsert_publication(pub_data) {
                                result.compatible_pubs += 1;
                            } else {
                                result.rejected_pubs += 1;
                            }
                        }
                    } else if data.writer_id == ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_WRITER {
                        if let Ok(sub_data) = SubscriptionBuiltinTopicData::parse_from_payload(
                            data.serialized_payload,
                        ) {
                            if self.upsert_subscription(sub_data) {
                                result.compatible_subs += 1;
                            } else {
                                result.rejected_subs += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// All currently discovered publications.
    pub fn discovered_publications(&self) -> &[PublicationBuiltinTopicData] {
        &self.discovered_publications
    }

    /// All currently discovered subscriptions.
    pub fn discovered_subscriptions(&self) -> &[SubscriptionBuiltinTopicData] {
        &self.discovered_subscriptions
    }

    /// All QoS-rejected publications (remote writers incompatible with our local QoS).
    pub fn rejected_publications(&self) -> &[PublicationBuiltinTopicData] {
        &self.rejected_publications
    }

    /// All QoS-rejected subscriptions (remote readers incompatible with our local QoS).
    pub fn rejected_subscriptions(&self) -> &[SubscriptionBuiltinTopicData] {
        &self.rejected_subscriptions
    }

    /// Local socket address this participant is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, DiscoveryError> {
        Ok(self.transport.local_addr()?)
    }

    fn upsert_publication(&mut self, data: PublicationBuiltinTopicData) -> bool {
        let writer_qos = QosProfile {
            reliability: data.reliability,
            history: data.history,
            durability: data.durability,
            deadline: data.deadline,
            liveliness: data.liveliness,
        };

        if match_endpoint_qos(&self.local_qos, &writer_qos).is_err() {
            let guid = data.endpoint_guid;
            if let Some(existing) = self
                .rejected_publications
                .iter_mut()
                .find(|p| p.endpoint_guid == guid)
            {
                *existing = data;
            } else {
                self.rejected_publications.push(data);
            }
            return false;
        }

        let guid = data.endpoint_guid;
        if let Some(existing) = self
            .discovered_publications
            .iter_mut()
            .find(|p| p.endpoint_guid == guid)
        {
            *existing = data;
        } else {
            self.discovered_publications.push(data);
        }
        true
    }

    fn upsert_subscription(&mut self, data: SubscriptionBuiltinTopicData) -> bool {
        let reader_qos = QosProfile {
            reliability: data.reliability,
            history: data.history,
            durability: data.durability,
            deadline: data.deadline,
            liveliness: data.liveliness,
        };

        if match_endpoint_qos(&reader_qos, &self.local_qos).is_err() {
            let guid = data.endpoint_guid;
            if let Some(existing) = self
                .rejected_subscriptions
                .iter_mut()
                .find(|s| s.endpoint_guid == guid)
            {
                *existing = data;
            } else {
                self.rejected_subscriptions.push(data);
            }
            return false;
        }

        let guid = data.endpoint_guid;
        if let Some(existing) = self
            .discovered_subscriptions
            .iter_mut()
            .find(|s| s.endpoint_guid == guid)
        {
            *existing = data;
        } else {
            self.discovered_subscriptions.push(data);
        }
        true
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::time::Duration as StdDuration;

    use super::*;
    use crate::protocol::dds::discovery::endpoint_data::{
        PublicationBuiltinTopicData, SubscriptionBuiltinTopicData,
    };
    use crate::protocol::dds::discovery::qos::{
        DurabilityKind, HistoryKind, HistoryQosPolicy, LivelinessKind, ReliabilityKind,
        ReliabilityQosPolicy,
    };
    use crate::protocol::dds::transport::{TransportConfig, UdpTransport};
    use crate::protocol::dds::types::guid::{
        Guid, GuidPrefix, ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER,
        ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_WRITER,
    };
    use crate::protocol::dds::types::locator::Locator;
    use crate::protocol::dds::types::time::Duration as RtpsDuration;

    // ── test helpers ──────────────────────────────────────────────────────────

    fn make_sedp(guid: [u8; 12]) -> SedpParticipant {
        let bind = SocketAddr::from(([127, 0, 0, 1], 0));
        let cfg = TransportConfig {
            read_timeout: Some(StdDuration::from_millis(500)),
            ..TransportConfig::unicast(bind)
        };
        let transport = UdpTransport::new(cfg).unwrap();
        SedpParticipant::with_transport(GuidPrefix(guid), transport)
    }

    fn make_sedp_with_qos(guid: [u8; 12], qos: QosProfile) -> SedpParticipant {
        let bind = SocketAddr::from(([127, 0, 0, 1], 0));
        let cfg = TransportConfig {
            read_timeout: Some(StdDuration::from_millis(500)),
            ..TransportConfig::unicast(bind)
        };
        let transport = UdpTransport::new(cfg).unwrap();
        SedpParticipant::with_transport_and_qos(GuidPrefix(guid), transport, qos)
    }

    fn chatter_pub(prefix: [u8; 12]) -> PublicationBuiltinTopicData {
        PublicationBuiltinTopicData {
            endpoint_guid: Guid::new(
                GuidPrefix(prefix),
                ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER,
            ),
            topic_name: heapless::String::try_from("/chatter").unwrap(),
            type_name: heapless::String::try_from("std_msgs::String").unwrap(),
            ..PublicationBuiltinTopicData::default()
        }
    }

    fn chatter_sub(prefix: [u8; 12]) -> SubscriptionBuiltinTopicData {
        SubscriptionBuiltinTopicData {
            endpoint_guid: Guid::new(
                GuidPrefix(prefix),
                ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_WRITER,
            ),
            topic_name: heapless::String::try_from("/chatter").unwrap(),
            type_name: heapless::String::try_from("std_msgs::String").unwrap(),
            ..SubscriptionBuiltinTopicData::default()
        }
    }

    // ── loopback tests ────────────────────────────────────────────────────────

    #[test]
    fn sedp_loopback_publication_discovery() {
        let mut a = make_sedp([0x01; 12]);
        let mut b = make_sedp([0x02; 12]);

        let b_addr = b.local_addr().unwrap();
        let b_locator = Locator::udp_v4(b_addr.port() as u32, [127, 0, 0, 1]);

        let pub_data = chatter_pub([0x01; 12]);
        a.announce_publication(&pub_data, &b_locator).unwrap();

        let result = b.process_incoming().unwrap();
        assert_eq!(result.compatible_pubs, 1);
        assert_eq!(result.compatible_subs, 0);
        assert_eq!(b.discovered_publications().len(), 1);
        assert_eq!(
            b.discovered_publications()[0].topic_name.as_str(),
            "/chatter"
        );
        assert_eq!(
            b.discovered_publications()[0].type_name.as_str(),
            "std_msgs::String"
        );
    }

    #[test]
    fn sedp_loopback_subscription_discovery() {
        let mut a = make_sedp([0x03; 12]);
        let mut b = make_sedp([0x04; 12]);

        let b_addr = b.local_addr().unwrap();
        let b_locator = Locator::udp_v4(b_addr.port() as u32, [127, 0, 0, 1]);

        let sub_data = chatter_sub([0x03; 12]);
        a.announce_subscription(&sub_data, &b_locator).unwrap();

        let result = b.process_incoming().unwrap();
        assert_eq!(result.compatible_pubs, 0);
        assert_eq!(result.compatible_subs, 1);
        assert_eq!(b.discovered_subscriptions().len(), 1);
        assert_eq!(
            b.discovered_subscriptions()[0].topic_name.as_str(),
            "/chatter"
        );
        assert_eq!(
            b.discovered_subscriptions()[0].type_name.as_str(),
            "std_msgs::String"
        );
    }

    #[test]
    fn sedp_loopback_pub_and_sub() {
        let mut a = make_sedp([0x05; 12]);
        let mut b = make_sedp([0x06; 12]);

        let a_addr = a.local_addr().unwrap();
        let b_addr = b.local_addr().unwrap();
        let a_locator = Locator::udp_v4(a_addr.port() as u32, [127, 0, 0, 1]);
        let b_locator = Locator::udp_v4(b_addr.port() as u32, [127, 0, 0, 1]);

        // A announces publication → B
        let pub_data = chatter_pub([0x05; 12]);
        a.announce_publication(&pub_data, &b_locator).unwrap();

        // B announces subscription → A
        let sub_data = chatter_sub([0x06; 12]);
        b.announce_subscription(&sub_data, &a_locator).unwrap();

        // Both process incoming
        let b_result = b.process_incoming().unwrap();
        let a_result = a.process_incoming().unwrap();

        assert_eq!(
            b_result.compatible_pubs, 1,
            "B should discover A's publication"
        );
        assert_eq!(b_result.compatible_subs, 0);
        assert_eq!(a_result.compatible_pubs, 0);
        assert_eq!(
            a_result.compatible_subs, 1,
            "A should discover B's subscription"
        );

        assert_eq!(b.discovered_publications().len(), 1);
        assert_eq!(a.discovered_subscriptions().len(), 1);
        assert_eq!(
            b.discovered_publications()[0].topic_name.as_str(),
            "/chatter"
        );
        assert_eq!(
            a.discovered_subscriptions()[0].topic_name.as_str(),
            "/chatter"
        );
    }

    // ── encode/decode roundtrip tests ─────────────────────────────────────────

    #[test]
    fn pub_data_encode_decode_roundtrip() {
        let original = chatter_pub([0xAB; 12]);

        let mut buf = [0u8; 1024];
        let n = original.serialize_to_payload(&mut buf).unwrap();
        assert!(n >= 4, "payload must include CDR header");

        let decoded = PublicationBuiltinTopicData::parse_from_payload(&buf[..n]).unwrap();
        assert_eq!(decoded.endpoint_guid, original.endpoint_guid);
        assert_eq!(decoded.topic_name.as_str(), "/chatter");
        assert_eq!(decoded.type_name.as_str(), "std_msgs::String");
        assert_eq!(decoded.reliability, original.reliability);
        assert_eq!(decoded.history, original.history);
        assert_eq!(decoded.durability, original.durability);
        assert_eq!(decoded.liveliness, original.liveliness);
        assert_eq!(decoded.deadline, original.deadline);
    }

    #[test]
    fn sub_data_encode_decode_roundtrip() {
        let original = chatter_sub([0xCD; 12]);

        let mut buf = [0u8; 1024];
        let n = original.serialize_to_payload(&mut buf).unwrap();
        assert!(n >= 4);

        let decoded = SubscriptionBuiltinTopicData::parse_from_payload(&buf[..n]).unwrap();
        assert_eq!(decoded.endpoint_guid, original.endpoint_guid);
        assert_eq!(decoded.topic_name.as_str(), "/chatter");
        assert_eq!(decoded.type_name.as_str(), "std_msgs::String");
        assert_eq!(decoded.reliability, original.reliability);
        assert_eq!(decoded.history, original.history);
        assert_eq!(decoded.durability, original.durability);
        assert_eq!(decoded.liveliness, original.liveliness);
        assert_eq!(decoded.deadline, original.deadline);
    }

    // ── QoS unit tests ────────────────────────────────────────────────────────

    #[test]
    fn qos_reliability_roundtrip() {
        use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};

        let original = ReliabilityQosPolicy::reliable();
        let mut buf = [0u8; 12];
        {
            let mut w = ByteWriter::new(&mut buf, Endianness::Little);
            original.serialize(&mut w).unwrap();
        }
        let mut cur = ByteCursor::new(&buf, Endianness::Little);
        let decoded = ReliabilityQosPolicy::parse(&mut cur).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.kind, ReliabilityKind::Reliable);
        assert_eq!(
            decoded.max_blocking_time,
            RtpsDuration {
                seconds: 0,
                fraction: 0x1999_9999
            }
        );
    }

    #[test]
    fn qos_history_roundtrip() {
        use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};

        let original = HistoryQosPolicy::keep_last_1();
        let mut buf = [0u8; 8];
        {
            let mut w = ByteWriter::new(&mut buf, Endianness::Little);
            original.serialize(&mut w).unwrap();
        }
        let mut cur = ByteCursor::new(&buf, Endianness::Little);
        let decoded = HistoryQosPolicy::parse(&mut cur).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.kind, HistoryKind::KeepLast);
        assert_eq!(decoded.depth, 1);
    }

    #[test]
    fn qos_default_values() {
        let rel = ReliabilityQosPolicy::default();
        assert_eq!(rel.kind, ReliabilityKind::Reliable);
        assert_eq!(rel.max_blocking_time.seconds, 0);
        assert_eq!(rel.max_blocking_time.fraction, 0x1999_9999);

        let hist = HistoryQosPolicy::default();
        assert_eq!(hist.kind, HistoryKind::KeepLast);
        assert_eq!(hist.depth, 1);

        let dur = crate::protocol::dds::discovery::qos::DurabilityQosPolicy::default();
        assert_eq!(dur.kind, DurabilityKind::Volatile);

        let live = crate::protocol::dds::discovery::qos::LivelinessQosPolicy::default();
        assert_eq!(live.kind, LivelinessKind::Automatic);
        assert_eq!(live.lease_duration.seconds, 0x7FFF_FFFF);
        assert_eq!(live.lease_duration.fraction, 0xFFFF_FFFF);

        let dl = crate::protocol::dds::discovery::qos::DeadlineQosPolicy::default();
        assert_eq!(dl.period.seconds, 0x7FFF_FFFF);
        assert_eq!(dl.period.fraction, 0xFFFF_FFFF);
    }

    // ── QoS filtering tests ───────────────────────────────────────────────────

    #[test]
    fn sedp_qos_incompatible_pub_is_rejected() {
        // Local participant wants Reliable; remote announces BestEffort.
        // Matching: reader(local=Reliable) vs writer(remote=BestEffort) → incompatible.
        use crate::protocol::dds::discovery::qos::{ReliabilityKind, ReliabilityQosPolicy};
        use crate::protocol::dds::discovery::qos_profile::QosProfile;

        let mut a = make_sedp_with_qos([0x10; 12], QosProfile::ros2_default()); // Reliable reader
        let mut b = make_sedp([0x11; 12]);

        let b_addr = b.local_addr().unwrap();
        let b_locator = Locator::udp_v4(b_addr.port() as u32, [127, 0, 0, 1]);

        // Build a BestEffort publication
        let pub_data = PublicationBuiltinTopicData {
            endpoint_guid: Guid::new(
                GuidPrefix([0x10; 12]),
                ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER,
            ),
            topic_name: heapless::String::try_from("/chatter").unwrap(),
            type_name: heapless::String::try_from("std_msgs::String").unwrap(),
            reliability: ReliabilityQosPolicy {
                kind: ReliabilityKind::BestEffort,
                ..ReliabilityQosPolicy::default()
            },
            ..PublicationBuiltinTopicData::default()
        };
        a.announce_publication(&pub_data, &b_locator).unwrap();

        let result = b.process_incoming().unwrap();
        // b has ros2_default local_qos (Reliable) — rejects BestEffort writer
        assert_eq!(
            result.compatible_pubs, 0,
            "BestEffort writer should be rejected by Reliable reader"
        );
        assert_eq!(result.rejected_pubs, 1, "should have 1 rejected pub");
        assert_eq!(b.discovered_publications().len(), 0);
        assert_eq!(b.rejected_publications().len(), 1);
    }

    #[test]
    fn sedp_qos_compatible_pub_is_accepted() {
        // Both Reliable — should be accepted.
        use crate::protocol::dds::discovery::qos_profile::QosProfile;

        let mut a = make_sedp([0x12; 12]);
        let mut b = make_sedp_with_qos([0x13; 12], QosProfile::ros2_default());

        let b_addr = b.local_addr().unwrap();
        let b_locator = Locator::udp_v4(b_addr.port() as u32, [127, 0, 0, 1]);

        let pub_data = chatter_pub([0x12; 12]); // Reliable (default)
        a.announce_publication(&pub_data, &b_locator).unwrap();

        let result = b.process_incoming().unwrap();
        assert_eq!(result.compatible_pubs, 1);
        assert_eq!(result.rejected_pubs, 0);
        assert_eq!(b.discovered_publications().len(), 1);
        assert_eq!(b.rejected_publications().len(), 0);
    }

    #[test]
    fn sedp_qos_incompatible_sub_is_rejected() {
        // Local writer is BestEffort; remote subscriber requires Reliable.
        // Matching: reader(remote=Reliable) vs writer(local=BestEffort) → incompatible.
        use crate::protocol::dds::discovery::qos::{ReliabilityKind, ReliabilityQosPolicy};
        use crate::protocol::dds::discovery::qos_profile::QosProfile;

        let mut a = make_sedp([0x14; 12]);
        // b is the local BestEffort writer
        let be_profile = QosProfile {
            reliability: ReliabilityQosPolicy {
                kind: ReliabilityKind::BestEffort,
                ..ReliabilityQosPolicy::default()
            },
            ..QosProfile::ros2_default()
        };
        let mut b = make_sedp_with_qos([0x15; 12], be_profile);

        let b_addr = b.local_addr().unwrap();
        let b_locator = Locator::udp_v4(b_addr.port() as u32, [127, 0, 0, 1]);

        // Remote subscriber requires Reliable
        let sub_data = SubscriptionBuiltinTopicData {
            endpoint_guid: Guid::new(
                GuidPrefix([0x14; 12]),
                ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_WRITER,
            ),
            topic_name: heapless::String::try_from("/chatter").unwrap(),
            type_name: heapless::String::try_from("std_msgs::String").unwrap(),
            reliability: ReliabilityQosPolicy {
                kind: ReliabilityKind::Reliable,
                ..ReliabilityQosPolicy::default()
            },
            ..SubscriptionBuiltinTopicData::default()
        };
        a.announce_subscription(&sub_data, &b_locator).unwrap();

        let result = b.process_incoming().unwrap();
        assert_eq!(result.compatible_subs, 0);
        assert_eq!(result.rejected_subs, 1);
        assert_eq!(b.discovered_subscriptions().len(), 0);
        assert_eq!(b.rejected_subscriptions().len(), 1);
    }

    #[test]
    fn sedp_sensor_data_pub_compatible_with_best_effort_reader() {
        // BestEffort reader accepts BestEffort writer.
        use crate::protocol::dds::discovery::qos::{ReliabilityKind, ReliabilityQosPolicy};
        use crate::protocol::dds::discovery::qos_profile::QosProfile;

        let mut a = make_sedp([0x16; 12]);
        let be_profile = QosProfile {
            reliability: ReliabilityQosPolicy {
                kind: ReliabilityKind::BestEffort,
                ..ReliabilityQosPolicy::default()
            },
            ..QosProfile::ros2_default()
        };
        let mut b = make_sedp_with_qos([0x17; 12], be_profile);

        let b_addr = b.local_addr().unwrap();
        let b_locator = Locator::udp_v4(b_addr.port() as u32, [127, 0, 0, 1]);

        let pub_data = PublicationBuiltinTopicData {
            endpoint_guid: Guid::new(
                GuidPrefix([0x16; 12]),
                ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER,
            ),
            topic_name: heapless::String::try_from("/scan").unwrap(),
            type_name: heapless::String::try_from("sensor_msgs::LaserScan").unwrap(),
            reliability: ReliabilityQosPolicy {
                kind: ReliabilityKind::BestEffort,
                ..ReliabilityQosPolicy::default()
            },
            ..PublicationBuiltinTopicData::default()
        };
        a.announce_publication(&pub_data, &b_locator).unwrap();

        let result = b.process_incoming().unwrap();
        assert_eq!(result.compatible_pubs, 1);
        assert_eq!(result.rejected_pubs, 0);
    }
}
