//! Type-safe DDS publisher.
//!
//! A `Publisher<T>` wraps a `StatefulWriter` and a `DdsType` codec.
//! It is created by `Participant::create_publisher` and drives
//! the reliable HEARTBEAT / ACKNACK cycle through `Participant::spin_once`.

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::Duration;

use heapless::String as HString;

use crate::protocol::dds::discovery::endpoint_data::PublicationBuiltinTopicData;
use crate::protocol::dds::discovery::qos_profile::QosProfile;
use crate::protocol::dds::stateful::{StatefulWriter, WriterConfig};
use crate::protocol::dds::transport::{TransportConfig, UdpTransport};
use crate::protocol::dds::types::guid::{EntityId, Guid, GuidPrefix};
use crate::protocol::dds::types::locator::Locator;

use super::dds_type::DdsType;
use super::error::DdsApiError;

/// Internal state for one writer endpoint.
pub(super) struct WriterEntry {
    /// The underlying reliable writer.
    pub(super) writer: StatefulWriter,
    /// Pre-built `PublicationBuiltinTopicData` for SEDP announcement.
    pub(super) pub_data: PublicationBuiltinTopicData,
    /// Whether we have announced this writer at least once.
    pub(super) announced: bool,
}

impl WriterEntry {
    /// Create a new writer entry on an ephemeral UDP port.
    pub(super) fn new(
        guid_prefix: GuidPrefix,
        entity_id: EntityId,
        topic_name: &str,
        type_name: &str,
        qos: &QosProfile,
    ) -> Result<Self, DdsApiError> {
        let bind = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0));
        let cfg = TransportConfig {
            read_timeout: Some(Duration::from_millis(1)),
            ..TransportConfig::unicast(bind)
        };
        let transport = UdpTransport::new(cfg)?;
        let local_addr = transport.local_addr()?;

        let guid = Guid::new(guid_prefix, entity_id);
        let writer_cfg = WriterConfig::new(guid);
        let writer = StatefulWriter::new(writer_cfg, transport);

        let unicast_locator = addr_to_locator(local_addr)?;
        let mut unicast_locators = heapless::Vec::new();
        unicast_locators
            .push(unicast_locator)
            .map_err(|_| DdsApiError::Serialization("locator list full"))?;

        let mut topic = HString::<256>::new();
        topic
            .push_str(topic_name)
            .map_err(|_| DdsApiError::TopicNameTooLong)?;

        let mut ttype = HString::<256>::new();
        ttype
            .push_str(type_name)
            .map_err(|_| DdsApiError::TypeNameTooLong)?;

        let pub_data = PublicationBuiltinTopicData {
            endpoint_guid: guid,
            topic_name: topic,
            type_name: ttype,
            unicast_locators,
            multicast_locators: heapless::Vec::new(),
            reliability: qos.reliability,
            history: qos.history,
            durability: qos.durability,
            liveliness: qos.liveliness,
            deadline: qos.deadline,
        };

        Ok(Self {
            writer,
            pub_data,
            announced: false,
        })
    }
}

/// Type-safe DDS publisher.
///
/// Created by [`Participant::create_publisher`].  Call [`Participant::publish`] to
/// send a sample; `Participant::spin_once` drives the HEARTBEAT / ACKNACK cycle
/// and discovery integration.
pub struct Publisher<T: DdsType> {
    pub(super) entry_idx: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: DdsType> Publisher<T> {
    pub(super) fn new(entry_idx: usize) -> Self {
        Self {
            entry_idx,
            _marker: core::marker::PhantomData,
        }
    }

    /// Serialize and send `value` to all matched readers.
    ///
    /// Returns the RTPS sequence number assigned to this sample.
    pub(super) fn publish(
        &self,
        entry: &mut WriterEntry,
        value: &T,
    ) -> Result<crate::protocol::dds::types::sequence::SequenceNumber, DdsApiError> {
        let mut buf = [0u8; 65536];
        let len = value.serialize(&mut buf)?;
        let sn = entry.writer.write(&buf[..len])?;
        Ok(sn)
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

pub(super) fn addr_to_locator(addr: SocketAddr) -> Result<Locator, DdsApiError> {
    match addr {
        SocketAddr::V4(v4) => Ok(Locator::udp_v4(v4.port() as u32, v4.ip().octets())),
        SocketAddr::V6(_) => Err(DdsApiError::Serialization("IPv6 locators not supported")),
    }
}
