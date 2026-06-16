//! Best-effort stateless RTPS writer (fire-and-forget DATA delivery).

use crate::protocol::dds::byte_cursor::Endianness;
use crate::protocol::dds::message::{Data, Message, MessageHeader, Submessage};
use crate::protocol::dds::transport::UdpTransport;
use crate::protocol::dds::types::guid::{
    Guid, ENTITYID_UNKNOWN, PROTOCOL_VERSION_2_3, VENDOR_ID_OXICTL,
};
use crate::protocol::dds::types::locator::Locator;
use crate::protocol::dds::types::sequence::SequenceNumber;

use super::cache::HistoryCache;
use super::error::StatelessError;

/// Configuration for a [`StatelessWriter`].
pub struct WriterConfig {
    /// The GUID of this writer.
    pub guid: Guid,
    /// Initial list of reader locators to send DATA to.
    pub reader_locators: Vec<Locator>,
    /// Maximum number of DATA entries to keep in the history cache.
    pub history_capacity: usize,
}

impl WriterConfig {
    /// Create a minimal writer config with the given GUID.
    ///
    /// Defaults: no reader locators, history capacity = 16.
    pub fn new(guid: Guid) -> Self {
        Self {
            guid,
            reader_locators: Vec::new(),
            history_capacity: 16,
        }
    }

    /// Add a reader locator to which DATA submessages will be unicast.
    pub fn with_reader_locator(mut self, loc: Locator) -> Self {
        self.reader_locators.push(loc);
        self
    }

    /// Override the history cache capacity (default 16).
    pub fn with_history_capacity(mut self, cap: usize) -> Self {
        self.history_capacity = cap;
        self
    }
}

/// Best-effort stateless RTPS writer.
///
/// Serializes each `write` call into a DATA submessage, sends it to all
/// configured reader locators, and caches the payload in a bounded history
/// cache. No ACK/NACK cycle is performed.
pub struct StatelessWriter {
    guid: Guid,
    reader_locators: Vec<Locator>,
    transport: UdpTransport,
    history: HistoryCache,
    next_sn: i64,
}

impl StatelessWriter {
    /// Create a new writer from `config`, using `transport` for network I/O.
    pub fn new(config: WriterConfig, transport: UdpTransport) -> Self {
        Self {
            guid: config.guid,
            reader_locators: config.reader_locators,
            transport,
            history: HistoryCache::new(config.history_capacity),
            next_sn: 1,
        }
    }

    /// Serialize `payload` as a DATA submessage and send it to all reader locators.
    ///
    /// The sequence number is 1-based and increments with each successful call.
    /// The payload is also stored in the history cache (oldest entry evicted if full).
    ///
    /// Returns the [`SequenceNumber`] assigned to this sample.
    pub fn write(&mut self, payload: &[u8]) -> Result<SequenceNumber, StatelessError> {
        let sn = SequenceNumber::new(self.next_sn);

        let data = Data {
            endianness: Endianness::Little,
            inline_qos_flag: false,
            data_flag: true,
            key_flag: false,
            non_standard_payload_flag: false,
            extra_flags: 0,
            reader_id: ENTITYID_UNKNOWN,
            writer_id: self.guid.entity_id,
            writer_sn: sn,
            inline_qos: None,
            serialized_payload: payload,
        };

        let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
        subs.push(Submessage::Data(data))
            .map_err(|_| StatelessError::BufferTooSmall)?;

        let msg = Message {
            header: MessageHeader {
                version: PROTOCOL_VERSION_2_3,
                vendor_id: VENDOR_ID_OXICTL,
                guid_prefix: self.guid.prefix,
            },
            submessages: subs,
        };

        // Send to all reader locators.
        for locator in &self.reader_locators {
            self.transport
                .send_to(&msg, locator)
                .map_err(StatelessError::Transport)?;
        }

        // Cache the payload (evicts oldest if full; we don't error on eviction).
        self.history.add(sn, payload.to_vec(), None);
        self.next_sn += 1;

        Ok(sn)
    }

    /// Add a reader locator at runtime.
    pub fn add_reader_locator(&mut self, locator: Locator) {
        self.reader_locators.push(locator);
    }

    /// Remove a reader locator by value.
    pub fn remove_reader_locator(&mut self, locator: &Locator) {
        self.reader_locators.retain(|l| l != locator);
    }

    /// Slice of currently configured reader locators.
    pub fn reader_locators(&self) -> &[Locator] {
        &self.reader_locators
    }

    /// Borrow the history cache.
    pub fn history(&self) -> &HistoryCache {
        &self.history
    }

    /// The GUID of this writer.
    pub fn own_guid(&self) -> &Guid {
        &self.guid
    }

    /// The sequence number that will be assigned to the *next* `write` call.
    pub fn next_sequence_number(&self) -> SequenceNumber {
        SequenceNumber::new(self.next_sn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::message::Submessage;
    use crate::protocol::dds::transport::TransportConfig;
    use crate::protocol::dds::types::guid::{EntityId, GuidPrefix};
    use std::net::{SocketAddr, UdpSocket};
    use std::time::Duration;

    fn test_guid(prefix_byte: u8) -> Guid {
        Guid {
            prefix: GuidPrefix([prefix_byte; 12]),
            entity_id: EntityId {
                entity_key: [0x00, 0x00, 0x01],
                entity_kind: 0x02,
            },
        }
    }

    fn make_writer(guid: Guid) -> StatelessWriter {
        let config = TransportConfig::unicast(SocketAddr::from(([127, 0, 0, 1], 0)));
        let transport = UdpTransport::new(config).unwrap();
        StatelessWriter::new(WriterConfig::new(guid).with_history_capacity(8), transport)
    }

    #[test]
    fn writer_sequence_numbers() {
        let mut writer = make_writer(test_guid(0xAA));
        assert_eq!(writer.next_sequence_number(), SequenceNumber::new(1));
        let sn1 = writer.write(b"one").unwrap();
        assert_eq!(sn1, SequenceNumber::new(1));
        let sn2 = writer.write(b"two").unwrap();
        assert_eq!(sn2, SequenceNumber::new(2));
        let sn3 = writer.write(b"three").unwrap();
        assert_eq!(sn3, SequenceNumber::new(3));
        assert_eq!(writer.next_sequence_number(), SequenceNumber::new(4));
    }

    #[test]
    fn writer_loopback_single() {
        // Bind a reader socket on an ephemeral port.
        let reader_sock = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        reader_sock
            .set_read_timeout(Some(Duration::from_millis(500)))
            .unwrap();
        let reader_port = reader_sock.local_addr().unwrap().port();

        let guid = test_guid(0xBB);
        let mut writer = make_writer(guid);
        writer.add_reader_locator(Locator::udp_v4(reader_port as u32, [127, 0, 0, 1]));

        let sn = writer.write(b"hello").unwrap();
        assert_eq!(sn, SequenceNumber::new(1));

        // Receive the raw datagram on the reader socket.
        let mut buf = [0u8; 65535];
        let (n, _) = reader_sock.recv_from(&mut buf).unwrap();

        // Parse the RTPS message.
        let msg = crate::protocol::dds::parse_message(&buf[..n]).unwrap();

        // Verify header prefix matches the writer GUID.
        assert_eq!(msg.header.guid_prefix, GuidPrefix([0xBBu8; 12]));

        // Find the DATA submessage and check its payload.
        let payload = msg
            .submessages
            .iter()
            .find_map(|s| {
                if let Submessage::Data(d) = s {
                    Some(d.serialized_payload)
                } else {
                    None
                }
            })
            .expect("no DATA submessage in received message");

        assert_eq!(payload, b"hello");
    }
}
