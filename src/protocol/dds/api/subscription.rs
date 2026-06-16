//! Type-safe DDS subscription.
//!
//! A `Subscription<T>` wraps a `StatefulReader` and a `DdsType` codec.
//! Received samples are buffered in a fixed-capacity `heapless::Vec`
//! queue; call [`Participant::take`] to drain them.

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::Duration;

use heapless::String as HString;
use heapless::Vec as HVec;

use crate::protocol::dds::discovery::endpoint_data::SubscriptionBuiltinTopicData;
use crate::protocol::dds::discovery::qos_profile::QosProfile;
use crate::protocol::dds::stateful::{ReaderConfig, StatefulReader};
use crate::protocol::dds::transport::{TransportConfig, UdpTransport};
use crate::protocol::dds::types::guid::{EntityId, Guid, GuidPrefix};
use crate::protocol::dds::types::locator::Locator;

use super::dds_type::{DdsType, Sample};
use super::error::DdsApiError;
use super::publisher::addr_to_locator;

/// Maximum number of samples that can be buffered per `Subscription`.
pub const SUBSCRIPTION_QUEUE_DEPTH: usize = 16;

/// Internal state for one reader endpoint.
pub(super) struct ReaderEntry {
    /// The underlying reliable reader.
    pub(super) reader: StatefulReader,
    /// Pre-built `SubscriptionBuiltinTopicData` for SEDP announcement.
    pub(super) sub_data: SubscriptionBuiltinTopicData,
    /// Whether we have announced this subscription at least once.
    pub(super) announced: bool,
    /// Raw decoded payloads pending delivery.
    pub(super) raw_queue: HVec<(Vec<u8>, [u8; 16]), SUBSCRIPTION_QUEUE_DEPTH>,
}

impl ReaderEntry {
    /// Create a new reader entry on an ephemeral UDP port.
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
        let reader_cfg = ReaderConfig::new(guid);
        let reader = StatefulReader::new(reader_cfg, transport);

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

        let sub_data = SubscriptionBuiltinTopicData {
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
            reader,
            sub_data,
            announced: false,
            raw_queue: HVec::new(),
        })
    }
}

/// Type-safe DDS subscription.
///
/// Created by [`Participant::create_subscription`].  Call [`Participant::take`]
/// to drain buffered samples; `Participant::spin_once` populates the queue by
/// calling `StatefulReader::recv` and buffering decoded payloads.
pub struct Subscription<T: DdsType> {
    pub(super) entry_idx: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: DdsType> Subscription<T> {
    pub(super) fn new(entry_idx: usize) -> Self {
        Self {
            entry_idx,
            _marker: core::marker::PhantomData,
        }
    }

    /// Drain all buffered samples, deserializing each one.
    ///
    /// Returns a `Vec` of `Sample<T>`.  Samples that fail to deserialize are
    /// silently skipped.
    pub(super) fn take(&self, entry: &mut ReaderEntry) -> Vec<Sample<T>> {
        let raw: Vec<_> = entry.raw_queue.iter().cloned().collect();
        entry.raw_queue.clear();
        let mut out = Vec::with_capacity(raw.len());
        for (payload, guid_bytes) in raw {
            if let Ok(data) = T::deserialize(&payload) {
                out.push(Sample {
                    data,
                    writer_guid_bytes: guid_bytes,
                });
            }
        }
        out
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Build a 16-byte GUID byte array from prefix + entity_id.
pub(super) fn guid_to_bytes(g: &Guid) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[..12].copy_from_slice(&g.prefix.0);
    out[12] = g.entity_id.entity_key[0];
    out[13] = g.entity_id.entity_key[1];
    out[14] = g.entity_id.entity_key[2];
    out[15] = g.entity_id.entity_kind;
    out
}

/// Convert reader unicast locator into a `Locator` for SEDP advertisement.
pub(super) fn sub_data_locator(sub_data: &SubscriptionBuiltinTopicData) -> Option<Locator> {
    sub_data.unicast_locators.first().copied()
}

/// Convert pub data's first unicast locator back to a `SocketAddr` for matching.
pub(super) fn pub_data_locators(
    pub_data: &crate::protocol::dds::discovery::endpoint_data::PublicationBuiltinTopicData,
) -> Vec<Locator> {
    pub_data.unicast_locators.iter().copied().collect()
}
