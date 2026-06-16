use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::{Duration as StdDuration, Instant};
use std::vec::Vec;

use heapless::Vec as HVec;

use crate::protocol::dds::byte_cursor::Endianness;
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::message::submessage::Data;
use crate::protocol::dds::message::{Message, MessageHeader, Submessage};
use crate::protocol::dds::transport::{
    metatraffic_unicast_port, TransportConfig, TransportError, UdpTransport,
};
use crate::protocol::dds::types::guid::{
    Guid, GuidPrefix, ENTITYID_PARTICIPANT, ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
    ENTITYID_UNKNOWN, PROTOCOL_VERSION_2_3, VENDOR_ID_OXICTL,
};
use crate::protocol::dds::types::locator::Locator;
use crate::protocol::dds::types::sequence::SequenceNumber;

use super::error::DiscoveryError;
use super::participant_data::ParticipantBuiltinTopicData;

/// A remote participant discovered via SPDP.
#[derive(Debug)]
pub struct DiscoveredParticipant {
    pub data: ParticipantBuiltinTopicData,
    pub last_seen: Instant,
}

/// RTPS SPDP participant — sends and receives participant discovery beacons.
///
/// Binds a UDP socket and maintains the list of discovered remote participants.
/// Call [`send_beacon_to`](SpdpParticipant::send_beacon_to) to announce the local
/// participant and [`process_incoming`](SpdpParticipant::process_incoming) to
/// process received beacons.
pub struct SpdpParticipant {
    transport: UdpTransport,
    own_data: ParticipantBuiltinTopicData,
    discovered: Vec<DiscoveredParticipant>,
    domain_id: u16,
    beacon_sn: i64,
}

impl SpdpParticipant {
    /// Create a new SPDP participant.
    ///
    /// Binds a UDP socket on `127.0.0.1:metatraffic_unicast_port(domain_id, participant_id)`.
    /// Use [`with_transport`](SpdpParticipant::with_transport) for custom bind addresses.
    pub fn new(
        domain_id: u16,
        participant_id: u16,
        guid_prefix: GuidPrefix,
    ) -> Result<Self, DiscoveryError> {
        let port = metatraffic_unicast_port(domain_id, participant_id);
        let bind = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
        let cfg = TransportConfig {
            read_timeout: Some(StdDuration::from_millis(100)),
            ..TransportConfig::unicast(bind)
        };
        Self::with_transport(domain_id, guid_prefix, UdpTransport::new(cfg)?)
    }

    /// Create with a pre-built transport (useful for testing with ephemeral ports).
    pub fn with_transport(
        domain_id: u16,
        guid_prefix: GuidPrefix,
        transport: UdpTransport,
    ) -> Result<Self, DiscoveryError> {
        let local_addr = transport.local_addr()?;
        let unicast_locator = Locator::udp_v4(
            local_addr.port() as u32,
            match local_addr {
                SocketAddr::V4(v4) => v4.ip().octets(),
                SocketAddr::V6(_) => {
                    return Err(DiscoveryError::Transport(TransportError::InvalidLocator))
                }
            },
        );

        let participant_guid = Guid::new(guid_prefix, ENTITYID_PARTICIPANT);

        let mut metatraffic_unicast_locators = HVec::new();
        let _ = metatraffic_unicast_locators.push(unicast_locator);
        let own_data = ParticipantBuiltinTopicData {
            participant_guid,
            protocol_version: PROTOCOL_VERSION_2_3,
            vendor_id: VENDOR_ID_OXICTL,
            metatraffic_unicast_locators,
            ..ParticipantBuiltinTopicData::default()
        };

        Ok(Self {
            transport,
            own_data,
            discovered: Vec::new(),
            domain_id,
            beacon_sn: 1,
        })
    }

    /// Create a new SPDP participant using a pre-built multicast transport.
    ///
    /// The `advertised_unicast_locators` are used in the own `ParticipantBuiltinTopicData`
    /// as `metatraffic_unicast_locators`, so remote participants know where to send
    /// SEDP unicast traffic after discovering us via multicast SPDP.
    ///
    /// The `mc_transport` is the multicast-bound socket that listens for SPDP beacons.
    pub fn with_multicast_transport(
        domain_id: u16,
        guid_prefix: GuidPrefix,
        mc_transport: UdpTransport,
        advertised_unicast_locators: HVec<Locator, 4>,
    ) -> Result<Self, DiscoveryError> {
        let participant_guid = Guid::new(guid_prefix, ENTITYID_PARTICIPANT);
        let own_data = ParticipantBuiltinTopicData {
            participant_guid,
            protocol_version: PROTOCOL_VERSION_2_3,
            vendor_id: VENDOR_ID_OXICTL,
            metatraffic_unicast_locators: advertised_unicast_locators,
            ..ParticipantBuiltinTopicData::default()
        };
        Ok(Self {
            transport: mc_transport,
            own_data,
            discovered: Vec::new(),
            domain_id,
            beacon_sn: 1,
        })
    }

    /// Send the participant's own SPDP announcement to the given locator.
    pub fn send_beacon_to(&mut self, locator: &Locator) -> Result<(), DiscoveryError> {
        let mut payload_buf = [0u8; 1024];
        let payload_len = self.own_data.serialize_to_payload(&mut payload_buf)?;
        let payload = &payload_buf[..payload_len];

        let sn = SequenceNumber::new(self.beacon_sn);
        self.beacon_sn += 1;

        let data_sub = Data {
            endianness: Endianness::Little,
            inline_qos_flag: false,
            data_flag: true,
            key_flag: false,
            non_standard_payload_flag: false,
            extra_flags: 0,
            reader_id: ENTITYID_UNKNOWN,
            writer_id: ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
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
                guid_prefix: self.own_data.participant_guid.prefix,
            },
            submessages: subs,
        };

        self.transport.send_to(&msg, locator)?;
        Ok(())
    }

    /// Receive and process one incoming RTPS message.
    ///
    /// Returns the number of participants added or refreshed.
    /// Returns `Ok(0)` if the socket timed out (no data available).
    pub fn process_incoming(&mut self) -> Result<usize, DiscoveryError> {
        let mut buf = [0u8; 65535];
        let (msg, _sender) = match self.transport.recv_into(&mut buf) {
            Ok(r) => r,
            Err(TransportError::Io(ref e))
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                return Ok(0);
            }
            Err(e) => return Err(DiscoveryError::Transport(e)),
        };

        let mut count = 0;
        for sub in msg.iter_submessages() {
            if let Submessage::Data(data) = sub {
                if data.writer_id == ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER
                    && data.data_flag
                    && !data.serialized_payload.is_empty()
                {
                    if let Ok(participant_data) =
                        ParticipantBuiltinTopicData::parse_from_payload(data.serialized_payload)
                    {
                        // Skip our own beacons (would be reflected back on loopback)
                        if participant_data.participant_guid != self.own_data.participant_guid {
                            self.upsert_participant(participant_data);
                            count += 1;
                        }
                    }
                    // malformed payloads are silently ignored for forward compat
                }
            }
        }
        Ok(count)
    }

    /// Return all currently discovered participants.
    pub fn discovered(&self) -> &[DiscoveredParticipant] {
        &self.discovered
    }

    /// Remove participants not heard from since `now - max_age`.
    pub fn remove_stale(&mut self, now: Instant, max_age: StdDuration) {
        self.discovered
            .retain(|p| now.duration_since(p.last_seen) < max_age);
    }

    /// Own participant data.
    pub fn own_data(&self) -> &ParticipantBuiltinTopicData {
        &self.own_data
    }

    /// Own GUID.
    pub fn own_guid(&self) -> &Guid {
        &self.own_data.participant_guid
    }

    /// Local socket address.
    pub fn local_addr(&self) -> Result<SocketAddr, DiscoveryError> {
        Ok(self.transport.local_addr()?)
    }

    /// Set the own participant's metatraffic unicast locator list.
    pub fn set_metatraffic_unicast_locators(&mut self, locators: HVec<Locator, 4>) {
        self.own_data.metatraffic_unicast_locators = locators;
    }

    /// Domain ID this participant operates in.
    pub fn domain_id(&self) -> u16 {
        self.domain_id
    }

    fn upsert_participant(&mut self, data: ParticipantBuiltinTopicData) {
        let guid = data.participant_guid;
        let now = Instant::now();
        // Skip ourselves
        if guid == self.own_data.participant_guid {
            return;
        }
        if let Some(existing) = self
            .discovered
            .iter_mut()
            .find(|p| p.data.participant_guid == guid)
        {
            existing.data = data;
            existing.last_seen = now;
        } else {
            self.discovered.push(DiscoveredParticipant {
                data,
                last_seen: now,
            });
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::time::{Duration as StdDuration, Instant};

    use super::*;
    use crate::protocol::dds::discovery::participant_data::{
        ParticipantBuiltinTopicData, BUILTIN_ENDPOINT_PARTICIPANT_ANNOUNCER,
        BUILTIN_ENDPOINT_PARTICIPANT_DETECTOR, BUILTIN_ENDPOINT_PUBLICATIONS_ANNOUNCER,
        BUILTIN_ENDPOINT_PUBLICATIONS_DETECTOR, BUILTIN_ENDPOINT_SUBSCRIPTIONS_ANNOUNCER,
        BUILTIN_ENDPOINT_SUBSCRIPTIONS_DETECTOR,
    };
    use crate::protocol::dds::transport::{TransportConfig, UdpTransport};
    use crate::protocol::dds::types::guid::{GuidPrefix, PROTOCOL_VERSION_2_3, VENDOR_ID_OXICTL};
    use crate::protocol::dds::types::locator::Locator;
    use crate::protocol::dds::types::time::Duration as RtpsDuration;

    // ── helper ───────────────────────────────────────────────────────────────

    fn make_transport_with_timeout() -> (UdpTransport, SocketAddr) {
        let bind = SocketAddr::from(([127, 0, 0, 1], 0));
        let cfg = TransportConfig {
            read_timeout: Some(StdDuration::from_millis(500)),
            ..TransportConfig::unicast(bind)
        };
        let t = UdpTransport::new(cfg).unwrap();
        let addr = t.local_addr().unwrap();
        (t, addr)
    }

    fn make_participant(guid: [u8; 12]) -> SpdpParticipant {
        let (transport, _) = make_transport_with_timeout();
        SpdpParticipant::with_transport(0, GuidPrefix(guid), transport).unwrap()
    }

    // ── participant_data tests ────────────────────────────────────────────────

    #[test]
    fn participant_data_default() {
        let d = ParticipantBuiltinTopicData::default();
        assert_eq!(d.protocol_version, PROTOCOL_VERSION_2_3);
        assert_eq!(d.vendor_id, VENDOR_ID_OXICTL);
        assert!(d.metatraffic_unicast_locators.is_empty());
        assert!(d.metatraffic_multicast_locators.is_empty());
        assert!(d.default_unicast_locators.is_empty());
        assert!(d.default_multicast_locators.is_empty());
        assert_eq!(
            d.lease_duration,
            RtpsDuration {
                seconds: 10,
                fraction: 0
            }
        );
        assert_eq!(
            d.builtin_endpoint_set,
            BUILTIN_ENDPOINT_PARTICIPANT_ANNOUNCER | BUILTIN_ENDPOINT_PARTICIPANT_DETECTOR
        );
        assert_eq!(d.manual_liveliness_count, 0);
    }

    #[test]
    fn participant_data_encode_decode_roundtrip() {
        let original = ParticipantBuiltinTopicData {
            participant_guid: crate::protocol::dds::types::guid::Guid::new(
                GuidPrefix([0xABu8; 12]),
                crate::protocol::dds::types::guid::ENTITYID_PARTICIPANT,
            ),
            lease_duration: RtpsDuration {
                seconds: 30,
                fraction: 0,
            },
            builtin_endpoint_set: BUILTIN_ENDPOINT_PARTICIPANT_ANNOUNCER
                | BUILTIN_ENDPOINT_PARTICIPANT_DETECTOR
                | BUILTIN_ENDPOINT_PUBLICATIONS_ANNOUNCER,
            manual_liveliness_count: 7,
            ..ParticipantBuiltinTopicData::default()
        };

        let mut buf = [0u8; 1024];
        let n = original.serialize_to_payload(&mut buf).unwrap();
        assert!(n >= 4, "payload must include CDR header");

        let decoded = ParticipantBuiltinTopicData::parse_from_payload(&buf[..n]).unwrap();
        assert_eq!(decoded.participant_guid, original.participant_guid);
        assert_eq!(decoded.protocol_version, original.protocol_version);
        assert_eq!(decoded.vendor_id, original.vendor_id);
        assert_eq!(decoded.lease_duration, original.lease_duration);
        assert_eq!(decoded.builtin_endpoint_set, original.builtin_endpoint_set);
        assert_eq!(
            decoded.manual_liveliness_count,
            original.manual_liveliness_count
        );
    }

    #[test]
    fn participant_data_with_locators() {
        let mut original = ParticipantBuiltinTopicData::default();
        original
            .metatraffic_unicast_locators
            .push(Locator::udp_v4(7410, [127, 0, 0, 1]))
            .unwrap();
        original
            .metatraffic_unicast_locators
            .push(Locator::udp_v4(7412, [192, 168, 1, 1]))
            .unwrap();

        let mut buf = [0u8; 1024];
        let n = original.serialize_to_payload(&mut buf).unwrap();
        let decoded = ParticipantBuiltinTopicData::parse_from_payload(&buf[..n]).unwrap();

        assert_eq!(
            decoded.metatraffic_unicast_locators.len(),
            2,
            "should recover both locators"
        );
        assert_eq!(
            decoded.metatraffic_unicast_locators[0],
            Locator::udp_v4(7410, [127, 0, 0, 1])
        );
        assert_eq!(
            decoded.metatraffic_unicast_locators[1],
            Locator::udp_v4(7412, [192, 168, 1, 1])
        );
    }

    #[test]
    fn participant_data_payload_too_small() {
        let result = ParticipantBuiltinTopicData::parse_from_payload(&[0x00, 0x03, 0x00]);
        assert!(
            matches!(result, Err(DiscoveryError::PayloadTooSmall)),
            "expected PayloadTooSmall, got {result:?}"
        );
    }

    #[test]
    fn participant_data_unknown_pids_ignored() {
        // Build a valid payload, then inject a synthetic unknown PID into the raw bytes
        // before the sentinel and verify it still parses.
        let original = ParticipantBuiltinTopicData::default();
        let mut buf = [0u8; 1024];
        let n = original.serialize_to_payload(&mut buf).unwrap();

        // The last 4 bytes of a valid payload are the sentinel [0x01, 0x00, 0x00, 0x00] (LE).
        // Insert an unknown PID entry just before the sentinel.
        // Unknown PID: 0x1234, length=4, value=[0x00; 4]
        let sentinel_pos = n - 4; // position of sentinel in buf
        let extra = [0x34u8, 0x12, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00]; // PID=0x1234, len=4, val=0
        let sentinel_bytes = buf[sentinel_pos..n].to_vec();

        // Shift sentinel forward by 8 bytes to make room
        let new_n = n + extra.len();
        assert!(new_n <= buf.len());
        buf.copy_within(sentinel_pos..n, sentinel_pos + extra.len());
        buf[sentinel_pos..sentinel_pos + extra.len()].copy_from_slice(&extra);
        // sentinel_bytes is now at sentinel_pos + extra.len()
        let _ = sentinel_bytes; // now copied in-place

        let decoded = ParticipantBuiltinTopicData::parse_from_payload(&buf[..new_n]).unwrap();
        assert_eq!(
            decoded.protocol_version, original.protocol_version,
            "unknown PID should not break parsing"
        );
    }

    // ── spdp loopback tests ───────────────────────────────────────────────────

    #[test]
    fn spdp_loopback_discovery() {
        let mut a = make_participant([0x01; 12]);
        let mut b = make_participant([0x02; 12]);

        let b_addr = b.local_addr().unwrap();
        let b_locator = Locator::udp_v4(b_addr.port() as u32, [127, 0, 0, 1]);

        a.send_beacon_to(&b_locator).unwrap();

        let n = b.process_incoming().unwrap();
        assert_eq!(n, 1, "b should discover exactly one participant");
        assert_eq!(b.discovered().len(), 1);
        assert_eq!(
            b.discovered()[0].data.participant_guid,
            *a.own_guid(),
            "discovered GUID must match a's GUID"
        );
    }

    #[test]
    fn spdp_loopback_no_self_discovery() {
        let mut a = make_participant([0xAA; 12]);
        let a_addr = a.local_addr().unwrap();
        let a_locator = Locator::udp_v4(a_addr.port() as u32, [127, 0, 0, 1]);

        a.send_beacon_to(&a_locator).unwrap();

        let n = a.process_incoming().unwrap();
        assert_eq!(n, 0, "self-beacons must not appear in the discovered list");
        assert_eq!(a.discovered().len(), 0);
    }

    #[test]
    fn builtin_endpoint_constants() {
        assert_eq!(BUILTIN_ENDPOINT_PARTICIPANT_ANNOUNCER, 0x00000001);
        assert_eq!(BUILTIN_ENDPOINT_PARTICIPANT_DETECTOR, 0x00000002);
        assert_eq!(BUILTIN_ENDPOINT_PUBLICATIONS_ANNOUNCER, 0x00000004);
        assert_eq!(BUILTIN_ENDPOINT_PUBLICATIONS_DETECTOR, 0x00000008);
        assert_eq!(BUILTIN_ENDPOINT_SUBSCRIPTIONS_ANNOUNCER, 0x00000010);
        assert_eq!(BUILTIN_ENDPOINT_SUBSCRIPTIONS_DETECTOR, 0x00000020);
    }

    #[test]
    fn with_multicast_transport_advertises_unicast_locators() {
        let guid_prefix = GuidPrefix([0xCC; 12]);
        let unicast_locator = Locator::udp_v4(7500, [127, 0, 0, 1]);
        let mut locs: HVec<Locator, 4> = HVec::new();
        locs.push(unicast_locator).unwrap();

        // Use an ephemeral unicast transport since multicast may not be available.
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
        let cfg = TransportConfig {
            read_timeout: Some(StdDuration::from_millis(50)),
            ..TransportConfig::unicast(addr)
        };
        let transport = UdpTransport::new(cfg).unwrap();
        let spdp =
            SpdpParticipant::with_multicast_transport(0, guid_prefix, transport, locs).unwrap();
        assert_eq!(spdp.own_data().metatraffic_unicast_locators.len(), 1);
    }

    #[test]
    fn remove_stale_works() {
        let mut a = make_participant([0x01; 12]);
        let mut b = make_participant([0x02; 12]);

        let b_addr = b.local_addr().unwrap();
        let b_locator = Locator::udp_v4(b_addr.port() as u32, [127, 0, 0, 1]);

        a.send_beacon_to(&b_locator).unwrap();
        b.process_incoming().unwrap();
        assert_eq!(b.discovered().len(), 1);

        // Remove with a tiny max_age to force expiry
        let now = Instant::now();
        let past = now + StdDuration::from_secs(100);
        b.remove_stale(past, StdDuration::from_nanos(1));
        assert_eq!(
            b.discovered().len(),
            0,
            "stale participant should have been removed"
        );
    }
}
