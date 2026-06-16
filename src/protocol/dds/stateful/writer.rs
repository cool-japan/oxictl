//! Reliable stateful RTPS writer with HEARTBEAT / ACKNACK cycle.

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use crate::protocol::dds::byte_cursor::Endianness;
use crate::protocol::dds::message::{Data, Heartbeat, Message, MessageHeader, Submessage};
use crate::protocol::dds::stateless::cache::HistoryCache;
use crate::protocol::dds::transport::error::TransportError;
use crate::protocol::dds::transport::UdpTransport;
use crate::protocol::dds::types::guid::{
    Guid, ENTITYID_UNKNOWN, PROTOCOL_VERSION_2_3, VENDOR_ID_OXICTL,
};
use crate::protocol::dds::types::locator::Locator;
use crate::protocol::dds::types::sequence::SequenceNumber;

use super::error::StatefulError;
use super::reader_proxy::ReaderProxy;

/// Configuration for a [`StatefulWriter`].
pub struct WriterConfig {
    /// The GUID of this writer.
    pub guid: Guid,
    /// Maximum number of DATA entries to keep in the history cache.
    pub history_capacity: usize,
    /// Duration between periodic HEARTBEATs sent to all matched readers.
    pub heartbeat_period: Duration,
}

impl WriterConfig {
    /// Create a minimal config with defaults: capacity=16, heartbeat_period=200ms.
    pub fn new(guid: Guid) -> Self {
        Self {
            guid,
            history_capacity: 16,
            heartbeat_period: Duration::from_millis(200),
        }
    }

    /// Override the history cache capacity.
    pub fn with_history_capacity(mut self, cap: usize) -> Self {
        self.history_capacity = cap;
        self
    }

    /// Override the heartbeat period.
    pub fn with_heartbeat_period(mut self, period: Duration) -> Self {
        self.heartbeat_period = period;
        self
    }
}

/// Reliable stateful RTPS writer.
///
/// Maintains a matched set of [`ReaderProxy`] entries, sends DATA to all of
/// them, and drives the HEARTBEAT / ACKNACK cycle to ensure reliable delivery.
pub struct StatefulWriter {
    guid: Guid,
    transport: UdpTransport,
    history: HistoryCache,
    readers: Vec<ReaderProxy>,
    next_sn: i64,
    heartbeat_count: i32,
    heartbeat_period: Duration,
    last_heartbeat_at: Instant,
}

impl StatefulWriter {
    /// Create a new writer from `config`, using `transport` for network I/O.
    pub fn new(config: WriterConfig, transport: UdpTransport) -> Self {
        Self {
            guid: config.guid,
            transport,
            history: HistoryCache::new(config.history_capacity),
            readers: Vec::new(),
            next_sn: 1,
            heartbeat_count: 0,
            heartbeat_period: config.heartbeat_period,
            last_heartbeat_at: Instant::now(),
        }
    }

    // ── Matched-reader management ─────────────────────────────────────────────

    /// Add a matched reader; DATA and HEARTBEATs are sent to its locators.
    pub fn add_matched_reader(&mut self, guid: Guid, unicast_locators: Vec<Locator>) {
        // Replace if already present (idempotent).
        if let Some(existing) = self
            .readers
            .iter_mut()
            .find(|r| r.remote_reader_guid == guid)
        {
            existing.unicast_locators = unicast_locators;
        } else {
            self.readers.push(ReaderProxy::new(guid, unicast_locators));
        }
    }

    /// Remove a matched reader by GUID.
    pub fn remove_matched_reader(&mut self, guid: &Guid) -> Result<(), StatefulError> {
        let pos = self
            .readers
            .iter()
            .position(|r| &r.remote_reader_guid == guid)
            .ok_or(StatefulError::NoSuchReader)?;
        self.readers.swap_remove(pos);
        Ok(())
    }

    // ── Writing ───────────────────────────────────────────────────────────────

    /// Serialize `payload` as a DATA submessage, send to all matched readers,
    /// store in the history cache, and return the assigned sequence number.
    pub fn write(&mut self, payload: &[u8]) -> Result<SequenceNumber, StatefulError> {
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
            .map_err(|_| StatefulError::BufferTooSmall)?;

        let msg = Message {
            header: MessageHeader {
                version: PROTOCOL_VERSION_2_3,
                vendor_id: VENDOR_ID_OXICTL,
                guid_prefix: self.guid.prefix,
            },
            submessages: subs,
        };

        // Send to every locator of every matched reader.
        for reader in &self.readers {
            for locator in &reader.unicast_locators {
                self.transport
                    .send_to(&msg, locator)
                    .map_err(StatefulError::Transport)?;
            }
        }

        // Store in history (evicts oldest if full — not an error).
        self.history.add(sn, payload.to_vec(), None);
        self.next_sn += 1;

        Ok(sn)
    }

    // ── Heartbeat ─────────────────────────────────────────────────────────────

    /// Send a HEARTBEAT to all matched readers, unconditionally.
    ///
    /// `final_flag=false` requires every reader to respond with an ACKNACK.
    pub fn send_heartbeat(&mut self) -> Result<(), StatefulError> {
        // Increment before use so count starts at 1.
        self.heartbeat_count = self.heartbeat_count.saturating_add(1);

        let first_sn = self
            .history
            .min_sn()
            .unwrap_or_else(|| SequenceNumber::new(1));
        let last_sn = SequenceNumber::new(self.next_sn - 1);

        let hb = Heartbeat {
            endianness: Endianness::Little,
            final_flag: false,
            liveliness_flag: false,
            group_info_flag: false,
            reader_id: ENTITYID_UNKNOWN,
            writer_id: self.guid.entity_id,
            first_sn,
            last_sn,
            count: self.heartbeat_count,
        };

        let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
        subs.push(Submessage::Heartbeat(hb))
            .map_err(|_| StatefulError::BufferTooSmall)?;

        let msg = Message {
            header: MessageHeader {
                version: PROTOCOL_VERSION_2_3,
                vendor_id: VENDOR_ID_OXICTL,
                guid_prefix: self.guid.prefix,
            },
            submessages: subs,
        };

        for reader in &self.readers {
            for locator in &reader.unicast_locators {
                self.transport
                    .send_to(&msg, locator)
                    .map_err(StatefulError::Transport)?;
            }
        }

        self.last_heartbeat_at = Instant::now();
        Ok(())
    }

    /// Send a HEARTBEAT if `heartbeat_period` has elapsed since the last one.
    ///
    /// Returns `true` if a HEARTBEAT was sent.
    pub fn send_heartbeat_if_due(&mut self) -> Result<bool, StatefulError> {
        if self.last_heartbeat_at.elapsed() >= self.heartbeat_period {
            self.send_heartbeat()?;
            return Ok(true);
        }
        Ok(false)
    }

    // ── Receiving ─────────────────────────────────────────────────────────────

    /// Process one incoming RTPS message (expected: ACKNACK from readers).
    ///
    /// Returns immediately (Ok(())) on timeout/WouldBlock.
    pub fn process_incoming(&mut self) -> Result<(), StatefulError> {
        // Collect (reader_guid, requested_sns) pairs extracted from the message,
        // so that we do not hold a borrow on `buf` while retransmitting.
        let pending = self.recv_acknack_pending()?;
        for sns in pending {
            self.retransmit_sns(&sns)?;
        }
        Ok(())
    }

    /// Receive one UDP datagram and collect any SNs that need to be retransmitted
    /// (as requested by ACKNACK submessages in that datagram).
    ///
    /// Returns `Ok(vec_of_sn_lists)` — one entry per matched proxy that had
    /// outstanding requests.  Returns `Ok(empty)` on timeout/WouldBlock.
    fn recv_acknack_pending(&mut self) -> Result<Vec<Vec<SequenceNumber>>, StatefulError> {
        // Owned record of one ACKNACK's relevant fields, extracted before `buf` drops.
        struct AckInfo {
            reader_guid: Guid,
            reader_sn_state: crate::protocol::dds::types::sequence::SequenceNumberSet,
        }

        let mut buf = vec![0u8; 65536];

        // Parse the packet and own all ACKNACK data before buf is released.
        let ack_infos: Vec<AckInfo> = {
            match self.transport.recv_into(&mut buf) {
                Ok((msg, _from)) => {
                    let sender_prefix = msg.header.guid_prefix;
                    let mut infos: Vec<AckInfo> = Vec::new();
                    for sub in msg.submessages.iter() {
                        if let Submessage::AckNack(ack) = sub {
                            infos.push(AckInfo {
                                reader_guid: Guid {
                                    prefix: sender_prefix,
                                    entity_id: ack.reader_id,
                                },
                                reader_sn_state: ack.reader_sn_state,
                            });
                        }
                    }
                    infos
                }
                Err(e) => {
                    if let TransportError::Io(ref io_err) = e {
                        if matches!(
                            io_err.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) {
                            return Ok(Vec::new());
                        }
                    }
                    return Err(StatefulError::Transport(e));
                }
            }
        };
        // buf lifetime ends here — all data is owned in ack_infos.

        // Now match proxies and collect retransmit lists.
        let mut result: Vec<Vec<SequenceNumber>> = Vec::new();
        for info in &ack_infos {
            // Build a minimal AckNack to reuse process_acknack logic.
            let ack = crate::protocol::dds::message::submessage::AckNack {
                endianness: Endianness::Little,
                final_flag: false,
                reader_id: info.reader_guid.entity_id,
                writer_id: self.guid.entity_id,
                reader_sn_state: info.reader_sn_state,
                count: 0,
            };
            if let Some(proxy) = self
                .readers
                .iter_mut()
                .find(|r| r.remote_reader_guid == info.reader_guid)
            {
                let needs_retransmit = proxy.process_acknack(&ack);
                if needs_retransmit {
                    result.push(proxy.drain_requested());
                }
            }
        }
        Ok(result)
    }

    /// Retransmit all requested changes for all readers (drain & send).
    pub fn retransmit_requested(&mut self) -> Result<(), StatefulError> {
        // Collect all (reader_locators, requested_sns) pairs first to avoid
        // holding a mutable borrow on `self.readers` while calling `send_to`.
        let mut work: Vec<(Vec<Locator>, Vec<SequenceNumber>)> = Vec::new();
        for reader in &mut self.readers {
            let sns = reader.drain_requested();
            if !sns.is_empty() {
                work.push((reader.unicast_locators.clone(), sns));
            }
        }

        for (locators, sns) in work {
            for sn in &sns {
                if let Some(entry) = self.history.get(*sn) {
                    let data = Data {
                        endianness: Endianness::Little,
                        inline_qos_flag: false,
                        data_flag: true,
                        key_flag: false,
                        non_standard_payload_flag: false,
                        extra_flags: 0,
                        reader_id: ENTITYID_UNKNOWN,
                        writer_id: self.guid.entity_id,
                        writer_sn: *sn,
                        inline_qos: None,
                        serialized_payload: &entry.data,
                    };
                    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
                    subs.push(Submessage::Data(data))
                        .map_err(|_| StatefulError::BufferTooSmall)?;
                    let msg = Message {
                        header: MessageHeader {
                            version: PROTOCOL_VERSION_2_3,
                            vendor_id: VENDOR_ID_OXICTL,
                            guid_prefix: self.guid.prefix,
                        },
                        submessages: subs,
                    };
                    for locator in &locators {
                        self.transport
                            .send_to(&msg, locator)
                            .map_err(StatefulError::Transport)?;
                    }
                }
            }
        }

        Ok(())
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// The sequence number that will be assigned to the next `write` call.
    pub fn next_sequence_number(&self) -> SequenceNumber {
        SequenceNumber::new(self.next_sn)
    }

    /// Local address this writer's socket is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, StatefulError> {
        self.transport
            .local_addr()
            .map_err(StatefulError::Transport)
    }

    /// Set or clear the read timeout on the transport socket.
    pub fn set_read_timeout(&mut self, dur: Option<Duration>) -> Result<(), StatefulError> {
        self.transport
            .set_read_timeout(dur)
            .map_err(StatefulError::Transport)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Retransmit a fixed list of SNs to all matched readers.
    fn retransmit_sns(&self, sns: &[SequenceNumber]) -> Result<(), StatefulError> {
        for sn in sns {
            if let Some(entry) = self.history.get(*sn) {
                let data = Data {
                    endianness: Endianness::Little,
                    inline_qos_flag: false,
                    data_flag: true,
                    key_flag: false,
                    non_standard_payload_flag: false,
                    extra_flags: 0,
                    reader_id: ENTITYID_UNKNOWN,
                    writer_id: self.guid.entity_id,
                    writer_sn: *sn,
                    inline_qos: None,
                    serialized_payload: &entry.data,
                };
                let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
                subs.push(Submessage::Data(data))
                    .map_err(|_| StatefulError::BufferTooSmall)?;
                let msg = Message {
                    header: MessageHeader {
                        version: PROTOCOL_VERSION_2_3,
                        vendor_id: VENDOR_ID_OXICTL,
                        guid_prefix: self.guid.prefix,
                    },
                    submessages: subs,
                };
                for reader in &self.readers {
                    for locator in &reader.unicast_locators {
                        self.transport
                            .send_to(&msg, locator)
                            .map_err(StatefulError::Transport)?;
                    }
                }
            }
        }
        Ok(())
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

    fn make_writer(guid: Guid) -> StatefulWriter {
        let cfg = TransportConfig::unicast(SocketAddr::from(([127, 0, 0, 1], 0)));
        let transport = UdpTransport::new(cfg).unwrap();
        StatefulWriter::new(WriterConfig::new(guid).with_history_capacity(8), transport)
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
    fn writer_loopback_sends_data() {
        // Bind a raw reader socket on an ephemeral port.
        let reader_sock = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        reader_sock
            .set_read_timeout(Some(Duration::from_millis(500)))
            .unwrap();
        let reader_port = reader_sock.local_addr().unwrap().port();

        let guid = test_guid(0xBB);
        let mut writer = make_writer(guid);
        writer.add_matched_reader(
            test_guid(0xCC),
            vec![Locator::udp_v4(reader_port as u32, [127, 0, 0, 1])],
        );

        let sn = writer.write(b"hello").unwrap();
        assert_eq!(sn, SequenceNumber::new(1));

        // Receive the raw datagram on the reader socket.
        let mut buf = [0u8; 65535];
        let (n, _) = reader_sock.recv_from(&mut buf).unwrap();

        // Parse the RTPS message.
        let msg = crate::protocol::dds::parse_message(&buf[..n]).unwrap();

        // Verify header prefix matches the writer GUID prefix.
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
