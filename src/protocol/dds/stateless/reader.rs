//! Best-effort stateless RTPS reader (accepts fire-and-forget DATA).

use std::net::SocketAddr;

use crate::protocol::dds::message::Submessage;
use crate::protocol::dds::transport::error::TransportError;
use crate::protocol::dds::transport::UdpTransport;
use crate::protocol::dds::types::guid::Guid;
use crate::protocol::dds::types::sequence::SequenceNumber;

use super::error::StatelessError;

/// Configuration for a [`StatelessReader`].
pub struct ReaderConfig {
    /// The GUID of this reader.
    pub guid: Guid,
    /// If non-empty, only DATA from these writer GUIDs will be delivered.
    /// An empty list means accept DATA from any writer.
    pub accepted_writer_guids: Vec<Guid>,
}

impl ReaderConfig {
    /// Create a reader config that accepts DATA from *any* writer.
    pub fn new(guid: Guid) -> Self {
        Self {
            guid,
            accepted_writer_guids: Vec::new(),
        }
    }

    /// Restrict this reader to accept DATA only from `writer_guid`.
    pub fn accept_writer(mut self, writer_guid: Guid) -> Self {
        self.accepted_writer_guids.push(writer_guid);
        self
    }
}

/// A DATA sample delivered to a [`StatelessReader`].
pub struct ReceivedSample {
    /// The reconstructed GUID of the remote writer.
    pub writer_guid: Guid,
    /// The RTPS sequence number of this sample.
    pub sequence_number: SequenceNumber,
    /// The serialized payload bytes.
    pub data: Vec<u8>,
    /// True if the DATA submessage carried inline QoS.
    pub has_inline_qos: bool,
}

/// Best-effort stateless RTPS reader.
///
/// Calls to [`recv`](Self::recv) are non-blocking (returns `None` on
/// `WouldBlock`/timeout). The reader holds a UDP socket; use
/// [`set_read_timeout`](Self::set_read_timeout) to control blocking behavior.
pub struct StatelessReader {
    config: ReaderConfig,
    transport: UdpTransport,
    highest_received_sn: Option<SequenceNumber>,
}

impl StatelessReader {
    /// Create a new reader from `config`, using `transport` for network I/O.
    pub fn new(config: ReaderConfig, transport: UdpTransport) -> Self {
        Self {
            config,
            transport,
            highest_received_sn: None,
        }
    }

    /// Non-blocking receive. Returns `None` on timeout/WouldBlock, `Some` on a
    /// received DATA submessage that passes the writer-GUID filter.
    ///
    /// Allocates a 65536-byte heap buffer per call.
    pub fn recv(&mut self) -> Result<Option<ReceivedSample>, StatelessError> {
        let mut buf = vec![0u8; 65536];
        // Extract owned data from the parsed message inside this block so that
        // the lifetime of `buf` does not escape.
        let result: Result<Option<ReceivedSample>, StatelessError> = {
            match self.transport.recv_into(&mut buf) {
                Ok((msg, _from_addr)) => {
                    let mut found: Option<ReceivedSample> = None;
                    for sub in msg.submessages.iter() {
                        if let Submessage::Data(data) = sub {
                            let writer_guid = Guid {
                                prefix: msg.header.guid_prefix,
                                entity_id: data.writer_id,
                            };
                            if !self.config.accepted_writer_guids.is_empty()
                                && !self.config.accepted_writer_guids.contains(&writer_guid)
                            {
                                continue;
                            }
                            // Own all borrowed data before the buf lifetime ends.
                            found = Some(ReceivedSample {
                                writer_guid,
                                sequence_number: data.writer_sn,
                                data: data.serialized_payload.to_vec(),
                                has_inline_qos: data.inline_qos_flag,
                            });
                            break;
                        }
                    }
                    Ok(found)
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
                    Err(StatelessError::Transport(e))
                }
            }
        };
        // Update highest_received_sn after buf lifetime has ended.
        if let Ok(Some(ref sample)) = result {
            let sn = sample.sequence_number;
            self.highest_received_sn =
                Some(
                    self.highest_received_sn
                        .map_or(sn, |h| if sn > h { sn } else { h }),
                );
        }
        result
    }

    /// Set or clear the read timeout on the underlying socket.
    ///
    /// Pass `None` for blocking mode, `Some(Duration::ZERO)` for non-blocking.
    pub fn set_read_timeout(
        &mut self,
        dur: Option<std::time::Duration>,
    ) -> Result<(), StatelessError> {
        self.transport
            .set_read_timeout(dur)
            .map_err(StatelessError::Transport)
    }

    /// The GUID of this reader.
    pub fn own_guid(&self) -> &Guid {
        &self.config.guid
    }

    /// The highest sequence number observed so far, or `None` if nothing has been received.
    pub fn highest_received_sn(&self) -> Option<SequenceNumber> {
        self.highest_received_sn
    }

    /// Returns the local socket address this reader is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, StatelessError> {
        self.transport
            .local_addr()
            .map_err(StatelessError::Transport)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::stateless::writer::{StatelessWriter, WriterConfig};
    use crate::protocol::dds::transport::TransportConfig;
    use crate::protocol::dds::types::guid::{EntityId, GuidPrefix};
    use crate::protocol::dds::types::locator::Locator;
    use std::net::SocketAddr;
    use std::time::Duration;

    fn make_guid(prefix_byte: u8, key: u8) -> Guid {
        Guid {
            prefix: GuidPrefix([prefix_byte; 12]),
            entity_id: EntityId {
                entity_key: [0x00, 0x00, key],
                entity_kind: 0x02,
            },
        }
    }

    fn make_writer(guid: Guid) -> StatelessWriter {
        let cfg = TransportConfig::unicast(SocketAddr::from(([127, 0, 0, 1], 0)));
        let transport = UdpTransport::new(cfg).unwrap();
        StatelessWriter::new(WriterConfig::new(guid).with_history_capacity(8), transport)
    }

    fn make_reader(guid: Guid) -> StatelessReader {
        let cfg = TransportConfig {
            read_timeout: Some(Duration::from_millis(300)),
            ..TransportConfig::unicast(SocketAddr::from(([127, 0, 0, 1], 0)))
        };
        let transport = UdpTransport::new(cfg).unwrap();
        StatelessReader::new(ReaderConfig::new(guid), transport)
    }

    fn locator_from_reader(reader: &StatelessReader) -> Locator {
        let addr = reader.local_addr().unwrap();
        Locator::udp_v4(addr.port() as u32, [127, 0, 0, 1])
    }

    #[test]
    fn reader_loopback_three_samples() {
        let writer_guid = make_guid(0x11, 0x01);
        let reader_guid = make_guid(0x22, 0x07);

        let mut writer = make_writer(writer_guid);
        let mut reader = make_reader(reader_guid);
        writer.add_reader_locator(locator_from_reader(&reader));

        let payloads: [&[u8]; 3] = [b"alpha", b"beta", b"gamma"];
        for payload in &payloads {
            writer.write(payload).unwrap();
        }

        let mut received = Vec::new();
        for _ in 0..3 {
            if let Some(sample) = reader.recv().unwrap() {
                received.push(sample);
            }
        }

        assert_eq!(received.len(), 3);
        assert_eq!(received[0].data, b"alpha");
        assert_eq!(received[1].data, b"beta");
        assert_eq!(received[2].data, b"gamma");
        assert_eq!(received[0].sequence_number, SequenceNumber::new(1));
        assert_eq!(received[1].sequence_number, SequenceNumber::new(2));
        assert_eq!(received[2].sequence_number, SequenceNumber::new(3));
    }

    #[test]
    fn reader_filters_writer_guid() {
        let writer_a_guid = make_guid(0xAA, 0x01);
        let writer_b_guid = make_guid(0xBB, 0x01);
        let reader_guid = make_guid(0xCC, 0x07);

        let mut writer_a = make_writer(writer_a_guid);
        let mut writer_b = make_writer(writer_b_guid);

        // Reader accepts only writer A.
        let reader_cfg = {
            let cfg = TransportConfig {
                read_timeout: Some(Duration::from_millis(300)),
                ..TransportConfig::unicast(SocketAddr::from(([127, 0, 0, 1], 0)))
            };
            let transport = UdpTransport::new(cfg).unwrap();
            let config = ReaderConfig::new(reader_guid).accept_writer(writer_a_guid);
            StatelessReader::new(config, transport)
        };
        let mut reader = reader_cfg;

        let reader_locator = locator_from_reader(&reader);
        writer_a.add_reader_locator(reader_locator);
        writer_b.add_reader_locator(reader_locator);

        // Send from B first, then A.
        writer_b.write(b"from_b").unwrap();
        writer_a.write(b"from_a").unwrap();

        // First recv() should consume B's packet and return None (filtered out).
        let first = reader.recv().unwrap();
        assert!(
            first.is_none(),
            "expected B's message to be filtered (got Some)"
        );

        // Second recv() should consume A's packet and return Some.
        let second = reader.recv().unwrap();
        assert!(second.is_some(), "expected A's message to be accepted");
        let sample = second.unwrap();
        assert_eq!(sample.data, b"from_a");
        assert_eq!(sample.writer_guid, writer_a_guid);
    }

    #[test]
    fn reader_highest_sn_tracking() {
        let writer_guid = make_guid(0x33, 0x01);
        let reader_guid = make_guid(0x44, 0x07);

        let mut writer = make_writer(writer_guid);
        let mut reader = make_reader(reader_guid);
        writer.add_reader_locator(locator_from_reader(&reader));

        writer.write(b"one").unwrap();
        writer.write(b"two").unwrap();
        writer.write(b"three").unwrap();

        assert!(reader.highest_received_sn().is_none());

        reader.recv().unwrap();
        reader.recv().unwrap();
        reader.recv().unwrap();

        assert_eq!(reader.highest_received_sn(), Some(SequenceNumber::new(3)));
    }
}
