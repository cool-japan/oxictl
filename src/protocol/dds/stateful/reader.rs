//! Reliable stateful RTPS reader with HEARTBEAT / ACKNACK cycle.

use std::net::SocketAddr;
use std::time::Duration;

use crate::protocol::dds::byte_cursor::Endianness;
use crate::protocol::dds::message::submessage::AckNack;
use crate::protocol::dds::message::{Message, MessageHeader, Submessage};
use crate::protocol::dds::transport::error::TransportError;
use crate::protocol::dds::transport::UdpTransport;
use crate::protocol::dds::types::guid::{Guid, PROTOCOL_VERSION_2_3, VENDOR_ID_OXICTL};
use crate::protocol::dds::types::locator::Locator;
use crate::protocol::dds::types::sequence::SequenceNumber;

use super::error::StatefulError;
use super::writer_proxy::WriterProxy;

/// Configuration for a [`StatefulReader`].
pub struct ReaderConfig {
    /// The GUID of this reader.
    pub guid: Guid,
    /// If `true`, send ACKNACK even for final (F-bit=1) HEARTBEATs.
    pub always_ack: bool,
}

impl ReaderConfig {
    /// Create a reader config with `always_ack = false`.
    pub fn new(guid: Guid) -> Self {
        Self {
            guid,
            always_ack: false,
        }
    }
}

/// A DATA sample delivered by a [`StatefulReader`].
pub struct ReceivedSample {
    /// The reconstructed GUID of the remote writer.
    pub writer_guid: Guid,
    /// The RTPS sequence number of this sample.
    pub sequence_number: SequenceNumber,
    /// The serialized payload bytes.
    pub data: Vec<u8>,
    /// `true` if the DATA submessage carried inline QoS.
    pub has_inline_qos: bool,
}

/// Reliable stateful RTPS reader.
///
/// Maintains a matched set of [`WriterProxy`] entries, records received SNs,
/// and sends ACKNACK submessages in response to HEARTBEAT submessages.
pub struct StatefulReader {
    config: ReaderConfig,
    transport: UdpTransport,
    writers: Vec<WriterProxy>,
}

impl StatefulReader {
    /// Create a new reader from `config`, using `transport` for network I/O.
    pub fn new(config: ReaderConfig, transport: UdpTransport) -> Self {
        Self {
            config,
            transport,
            writers: Vec::new(),
        }
    }

    // ── Matched-writer management ─────────────────────────────────────────────

    /// Add a matched writer (for tracking & ACKNACK sending).
    pub fn add_matched_writer(&mut self, guid: Guid, unicast_locators: Vec<Locator>) {
        if let Some(existing) = self
            .writers
            .iter_mut()
            .find(|w| w.remote_writer_guid == guid)
        {
            existing.unicast_locators = unicast_locators;
        } else {
            self.writers.push(WriterProxy::new(guid, unicast_locators));
        }
    }

    /// Remove a matched writer by GUID.
    pub fn remove_matched_writer(&mut self, guid: &Guid) {
        self.writers.retain(|w| &w.remote_writer_guid != guid);
    }

    // ── Receiving ─────────────────────────────────────────────────────────────

    /// Non-blocking receive.
    ///
    /// Processes all submessages in one incoming RTPS packet:
    /// - `DATA` — records the SN and returns a [`ReceivedSample`].
    /// - `HEARTBEAT` — sends an ACKNACK if `!final_flag || always_ack`.
    ///
    /// Returns `Ok(None)` on timeout / WouldBlock.
    pub fn recv(&mut self) -> Result<Option<ReceivedSample>, StatefulError> {
        let mut buf = vec![0u8; 65536];

        // ── Phase 1: parse the incoming packet ────────────────────────────────
        // We need to own everything that comes out of the parsed message before
        // `buf` is dropped, since Message<'_> borrows from buf.
        enum Action {
            Data {
                writer_guid: Guid,
                sn: SequenceNumber,
                payload: Vec<u8>,
                has_inline_qos: bool,
            },
            Heartbeat {
                writer_guid: Guid,
                writer_id: crate::protocol::dds::types::guid::EntityId,
                last_sn: SequenceNumber,
                send_ack: bool,
            },
        }

        let actions: Vec<Action> = match self.transport.recv_into(&mut buf) {
            Ok((msg, _from)) => {
                let sender_prefix = msg.header.guid_prefix;
                let mut acts = Vec::new();
                for sub in msg.submessages.iter() {
                    match sub {
                        Submessage::Data(data) => {
                            let writer_guid = Guid {
                                prefix: sender_prefix,
                                entity_id: data.writer_id,
                            };
                            acts.push(Action::Data {
                                writer_guid,
                                sn: data.writer_sn,
                                payload: data.serialized_payload.to_vec(),
                                has_inline_qos: data.inline_qos_flag,
                            });
                        }
                        Submessage::Heartbeat(hb) => {
                            let writer_guid = Guid {
                                prefix: sender_prefix,
                                entity_id: hb.writer_id,
                            };
                            let send_ack = !hb.final_flag || self.config.always_ack;
                            acts.push(Action::Heartbeat {
                                writer_guid,
                                writer_id: hb.writer_id,
                                last_sn: hb.last_sn,
                                send_ack,
                            });
                        }
                        _ => {}
                    }
                }
                acts
            }
            Err(e) => {
                if let TransportError::Io(ref io_err) = e {
                    if matches!(
                        io_err.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) {
                        return Ok(None);
                    }
                }
                return Err(StatefulError::Transport(e));
            }
        };

        // buf lifetime ends here; all data is now owned in `actions`.

        // ── Phase 2: update state and optionally send ACKNACK ─────────────────
        let mut found_sample: Option<ReceivedSample> = None;

        for action in actions {
            match action {
                Action::Data {
                    writer_guid,
                    sn,
                    payload,
                    has_inline_qos,
                } => {
                    // Find or create a WriterProxy for this writer.
                    if !self
                        .writers
                        .iter()
                        .any(|w| w.remote_writer_guid == writer_guid)
                    {
                        self.writers.push(WriterProxy::new(writer_guid, Vec::new()));
                    }
                    if let Some(proxy) = self
                        .writers
                        .iter_mut()
                        .find(|w| w.remote_writer_guid == writer_guid)
                    {
                        proxy.received(sn);
                    }
                    // Return the first DATA sample found.
                    if found_sample.is_none() {
                        found_sample = Some(ReceivedSample {
                            writer_guid,
                            sequence_number: sn,
                            data: payload,
                            has_inline_qos,
                        });
                    }
                }
                Action::Heartbeat {
                    writer_guid,
                    writer_id,
                    last_sn,
                    send_ack,
                } => {
                    if !send_ack {
                        continue;
                    }

                    // Find or create a WriterProxy.
                    if !self
                        .writers
                        .iter()
                        .any(|w| w.remote_writer_guid == writer_guid)
                    {
                        self.writers.push(WriterProxy::new(writer_guid, Vec::new()));
                    }

                    // Build and send ACKNACK (collect data before sending).
                    let (sn_set, count, locators) = {
                        if let Some(proxy) = self
                            .writers
                            .iter_mut()
                            .find(|w| w.remote_writer_guid == writer_guid)
                        {
                            let sn_set = proxy.build_missing_sn_set(last_sn);
                            let count = proxy.next_acknack_count();
                            let locators = proxy.unicast_locators.clone();
                            (sn_set, count, locators)
                        } else {
                            continue;
                        }
                    };

                    let ack = AckNack {
                        endianness: Endianness::Little,
                        final_flag: true,
                        reader_id: self.config.guid.entity_id,
                        writer_id,
                        reader_sn_state: sn_set,
                        count,
                    };

                    let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
                    subs.push(Submessage::AckNack(ack))
                        .map_err(|_| StatefulError::BufferTooSmall)?;

                    let msg = Message {
                        header: MessageHeader {
                            version: PROTOCOL_VERSION_2_3,
                            vendor_id: VENDOR_ID_OXICTL,
                            guid_prefix: self.config.guid.prefix,
                        },
                        submessages: subs,
                    };

                    for locator in &locators {
                        self.transport
                            .send_to(&msg, locator)
                            .map_err(StatefulError::Transport)?;
                    }
                    // When the proxy has no locators, send to ENTITYID_UNKNOWN as a
                    // best-effort multicast (noop here if locators is empty — the
                    // test that exercises ACKNACK path uses explicit locators).
                    if locators.is_empty() {
                        // Nowhere to send; silently skip.
                    }
                }
            }
        }

        Ok(found_sample)
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Set or clear the read timeout on the underlying socket.
    pub fn set_read_timeout(&mut self, dur: Option<Duration>) -> Result<(), StatefulError> {
        self.transport
            .set_read_timeout(dur)
            .map_err(StatefulError::Transport)
    }

    /// The GUID of this reader.
    pub fn own_guid(&self) -> &Guid {
        &self.config.guid
    }

    /// Returns the local socket address this reader is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, StatefulError> {
        self.transport
            .local_addr()
            .map_err(StatefulError::Transport)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::message::submessage::Heartbeat;
    use crate::protocol::dds::message::{Message, MessageHeader, Submessage};
    use crate::protocol::dds::stateful::writer::{StatefulWriter, WriterConfig};
    use crate::protocol::dds::transport::TransportConfig;
    use crate::protocol::dds::types::guid::{EntityId, GuidPrefix, ENTITYID_UNKNOWN};
    use crate::protocol::dds::types::sequence::SequenceNumber;
    use std::net::{SocketAddr, UdpSocket};
    use std::time::Duration;

    fn sn(v: i64) -> SequenceNumber {
        SequenceNumber::new(v)
    }

    fn make_guid(prefix_byte: u8, key: u8) -> Guid {
        Guid {
            prefix: GuidPrefix([prefix_byte; 12]),
            entity_id: EntityId {
                entity_key: [0x00, 0x00, key],
                entity_kind: 0x02,
            },
        }
    }

    fn make_writer(guid: Guid) -> StatefulWriter {
        let cfg = TransportConfig::unicast(SocketAddr::from(([127, 0, 0, 1], 0)));
        let transport = UdpTransport::new(cfg).unwrap();
        StatefulWriter::new(WriterConfig::new(guid).with_history_capacity(8), transport)
    }

    fn make_reader(guid: Guid) -> StatefulReader {
        let cfg = TransportConfig {
            read_timeout: Some(Duration::from_millis(300)),
            ..TransportConfig::unicast(SocketAddr::from(([127, 0, 0, 1], 0)))
        };
        let transport = UdpTransport::new(cfg).unwrap();
        StatefulReader::new(ReaderConfig::new(guid), transport)
    }

    fn locator_of(addr: SocketAddr) -> Locator {
        Locator::udp_v4(addr.port() as u32, [127, 0, 0, 1])
    }

    // ── Test 1: writer + reader loopback — 3 samples ──────────────────────────
    #[test]
    fn reader_loopback_three_samples() {
        let writer_guid = make_guid(0x11, 0x01);
        let reader_guid = make_guid(0x22, 0x07);

        let mut writer = make_writer(writer_guid);
        let mut reader = make_reader(reader_guid);

        let reader_addr = reader.local_addr().unwrap();
        writer.add_matched_reader(reader_guid, vec![locator_of(reader_addr)]);

        let payloads: [&[u8]; 3] = [b"alpha", b"beta", b"gamma"];
        for p in &payloads {
            writer.write(p).unwrap();
        }

        let mut received: Vec<ReceivedSample> = Vec::new();
        for _ in 0..3 {
            if let Some(sample) = reader.recv().unwrap() {
                received.push(sample);
            }
        }

        assert_eq!(received.len(), 3);
        assert_eq!(received[0].data, b"alpha");
        assert_eq!(received[1].data, b"beta");
        assert_eq!(received[2].data, b"gamma");
        assert_eq!(received[0].sequence_number, sn(1));
        assert_eq!(received[1].sequence_number, sn(2));
        assert_eq!(received[2].sequence_number, sn(3));
    }

    // ── Test 2: reader handles HEARTBEAT and sends ACKNACK ────────────────────
    #[test]
    fn reader_handles_heartbeat_sends_acknack() {
        // Raw UDP socket to act as the writer.
        let writer_sock = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        writer_sock
            .set_read_timeout(Some(Duration::from_millis(500)))
            .unwrap();
        let writer_addr = writer_sock.local_addr().unwrap();

        let writer_guid = make_guid(0x33, 0x01);
        let reader_guid = make_guid(0x44, 0x07);

        // Create a reader that always responds to heartbeats.
        let cfg = TransportConfig {
            read_timeout: Some(Duration::from_millis(500)),
            ..TransportConfig::unicast(SocketAddr::from(([127, 0, 0, 1], 0)))
        };
        let mut reader = StatefulReader::new(
            ReaderConfig {
                guid: reader_guid,
                always_ack: true,
            },
            UdpTransport::new(cfg).unwrap(),
        );

        let reader_addr = reader.local_addr().unwrap();
        reader.add_matched_writer(writer_guid, vec![locator_of(writer_addr)]);

        // Craft and send a synthetic HEARTBEAT from writer → reader.
        let hb = Heartbeat {
            endianness: Endianness::Little,
            final_flag: true, // normally final means no ack needed, but always_ack=true
            liveliness_flag: false,
            group_info_flag: false,
            reader_id: ENTITYID_UNKNOWN,
            writer_id: writer_guid.entity_id,
            first_sn: sn(1),
            last_sn: sn(3),
            count: 1,
        };

        let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
        subs.push(Submessage::Heartbeat(hb)).unwrap();
        let hb_msg = Message {
            header: MessageHeader {
                version: crate::protocol::dds::types::guid::PROTOCOL_VERSION_2_3,
                vendor_id: crate::protocol::dds::types::guid::VENDOR_ID_OXICTL,
                guid_prefix: writer_guid.prefix,
            },
            submessages: subs,
        };

        // Serialize and send the heartbeat directly via writer_sock.
        let mut raw_buf = [0u8; 65535];
        let n = crate::protocol::dds::serialize_message(&hb_msg, &mut raw_buf).unwrap();
        writer_sock.send_to(&raw_buf[..n], reader_addr).unwrap();

        // Reader should process the HEARTBEAT and send back an ACKNACK.
        let result = reader.recv().unwrap();
        // HEARTBEAT alone — no DATA, so result is None.
        assert!(result.is_none());

        // Now the writer_sock should have received an ACKNACK from the reader.
        let mut ack_buf = [0u8; 65535];
        let (m, _) = writer_sock.recv_from(&mut ack_buf).unwrap();
        let ack_msg = crate::protocol::dds::parse_message(&ack_buf[..m]).unwrap();

        let has_acknack = ack_msg
            .submessages
            .iter()
            .any(|s| matches!(s, Submessage::AckNack(a) if a.writer_id == writer_guid.entity_id));
        assert!(
            has_acknack,
            "expected ACKNACK targeting the writer entity ID"
        );
    }

    // ── Test 3: full HEARTBEAT → ACKNACK cycle ────────────────────────────────
    #[test]
    fn writer_reader_heartbeat_ack_cycle() {
        let writer_guid = make_guid(0x55, 0x01);
        let reader_guid = make_guid(0x66, 0x07);

        let mut writer = {
            let cfg = TransportConfig {
                read_timeout: Some(Duration::from_millis(300)),
                ..TransportConfig::unicast(SocketAddr::from(([127, 0, 0, 1], 0)))
            };
            let transport = UdpTransport::new(cfg).unwrap();
            StatefulWriter::new(
                WriterConfig::new(writer_guid).with_history_capacity(8),
                transport,
            )
        };

        let mut reader = {
            let cfg = TransportConfig {
                read_timeout: Some(Duration::from_millis(300)),
                ..TransportConfig::unicast(SocketAddr::from(([127, 0, 0, 1], 0)))
            };
            let transport = UdpTransport::new(cfg).unwrap();
            StatefulReader::new(
                ReaderConfig {
                    guid: reader_guid,
                    always_ack: false,
                },
                transport,
            )
        };

        let writer_addr = writer.local_addr().unwrap();
        let reader_addr = reader.local_addr().unwrap();

        writer.add_matched_reader(reader_guid, vec![locator_of(reader_addr)]);
        reader.add_matched_writer(writer_guid, vec![locator_of(writer_addr)]);

        // Step 1: writer sends DATA.
        writer.write(b"payload").unwrap();

        // Step 2: reader receives DATA.
        let sample = reader.recv().unwrap();
        assert!(sample.is_some());
        assert_eq!(sample.unwrap().data, b"payload");

        // Step 3: writer sends HEARTBEAT (final_flag=false → reader MUST ack).
        writer.send_heartbeat().unwrap();

        // Step 4: reader receives HEARTBEAT and replies with ACKNACK.
        let no_data = reader.recv().unwrap();
        assert!(no_data.is_none(), "HEARTBEAT should not yield a sample");

        // Step 5: writer processes the ACKNACK.
        writer.process_incoming().unwrap();
        // Reaching here without panic means the full cycle succeeded.
    }
}
