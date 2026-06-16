use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::time::Duration;

use crate::protocol::dds::message::Message;
use crate::protocol::dds::types::locator::Locator;
use crate::protocol::dds::{parse_message, serialize_message};

use super::error::TransportError;
use super::locator::locator_to_socket_addr;

/// RTPS SPDP multicast IPv4 address (239.255.0.1), per RTPS 2.3 spec 9.6.1.
pub const SPDP_MULTICAST_IPV4: [u8; 4] = [239, 255, 0, 1];

/// Compute the RTPS metatraffic multicast port for a domain.
///
/// Formula: PB + DG × domainId + d0, where PB=7400, DG=250, d0=0.
pub fn metatraffic_multicast_port(domain_id: u16) -> u16 {
    7400u16.saturating_add(250u16.saturating_mul(domain_id))
}

/// Compute the RTPS metatraffic unicast port for a participant in a domain.
///
/// Formula: PB + DG × domainId + d1 + PG × participantId, where d1=10, PG=2.
pub fn metatraffic_unicast_port(domain_id: u16, participant_id: u16) -> u16 {
    7400u16
        .saturating_add(250u16.saturating_mul(domain_id))
        .saturating_add(10)
        .saturating_add(2u16.saturating_mul(participant_id))
}

/// Compute the RTPS user traffic multicast port for a domain.
///
/// Formula: PB + DG × domainId + d2, where d2=1.
pub fn user_traffic_multicast_port(domain_id: u16) -> u16 {
    7400u16
        .saturating_add(250u16.saturating_mul(domain_id))
        .saturating_add(1)
}

/// Compute the RTPS user traffic unicast port for a participant in a domain.
///
/// Formula: PB + DG × domainId + d3 + PG × participantId, where d3=11, PG=2.
pub fn user_traffic_unicast_port(domain_id: u16, participant_id: u16) -> u16 {
    7400u16
        .saturating_add(250u16.saturating_mul(domain_id))
        .saturating_add(11)
        .saturating_add(2u16.saturating_mul(participant_id))
}

/// Configuration for a [`UdpTransport`].
pub struct TransportConfig {
    /// Local address to bind to (e.g. `0.0.0.0:7412`).
    pub bind_addr: SocketAddr,
    /// Optional multicast group to join on creation.
    pub multicast_group: Option<[u8; 4]>,
    /// Interface address for multicast (None = INADDR_ANY).
    pub multicast_interface: Option<[u8; 4]>,
    /// Multicast TTL (default 1 = same subnet).
    pub ttl: u32,
    /// Read timeout (`None` = blocking).
    pub read_timeout: Option<Duration>,
}

impl TransportConfig {
    /// Create a unicast-only configuration bound to the given address.
    pub fn unicast(bind_addr: SocketAddr) -> Self {
        Self {
            bind_addr,
            multicast_group: None,
            multicast_interface: None,
            ttl: 1,
            read_timeout: None,
        }
    }

    /// Create a multicast configuration bound to `INADDR_ANY:port`, joining `group`.
    ///
    /// The socket is bound to `0.0.0.0:port` (not to the multicast address itself)
    /// for cross-platform compatibility.
    pub fn multicast(port: u16, group: [u8; 4]) -> Self {
        Self {
            bind_addr: SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port)),
            multicast_group: Some(group),
            multicast_interface: None,
            ttl: 1,
            read_timeout: None,
        }
    }
}

/// UDPv4 transport for RTPS messages. Wraps `std::net::UdpSocket`.
///
/// Create via [`UdpTransport::new`] with a [`TransportConfig`], then call
/// [`send_to`](UdpTransport::send_to) / [`recv_into`](UdpTransport::recv_into)
/// to exchange RTPS [`Message`]s over the network.
pub struct UdpTransport {
    socket: UdpSocket,
}

impl UdpTransport {
    /// Create and bind a new transport according to `config`.
    ///
    /// If `config.multicast_group` is set the socket will join the multicast
    /// group immediately after binding.
    pub fn new(config: TransportConfig) -> Result<Self, TransportError> {
        let socket = UdpSocket::bind(config.bind_addr)?;
        socket.set_ttl(config.ttl)?;
        if let Some(timeout) = config.read_timeout {
            socket.set_read_timeout(Some(timeout))?;
        }
        if let Some(group) = config.multicast_group {
            let iface = config.multicast_interface.unwrap_or([0, 0, 0, 0]);
            socket.join_multicast_v4(&Ipv4Addr::from(group), &Ipv4Addr::from(iface))?;
        }
        Ok(Self { socket })
    }

    /// Serialize `msg` and send it to the destination described by `locator`.
    ///
    /// Returns the number of bytes sent on success.
    pub fn send_to(&self, msg: &Message<'_>, locator: &Locator) -> Result<usize, TransportError> {
        let addr = locator_to_socket_addr(locator).ok_or(TransportError::InvalidLocator)?;
        let mut buf = [0u8; 65535];
        let n = serialize_message(msg, &mut buf)?;
        Ok(self.socket.send_to(&buf[..n], addr)?)
    }

    /// Receive one RTPS datagram into `buf` and parse it.
    ///
    /// Returns the parsed [`Message`] (borrowing from `buf`) and the sender's
    /// [`SocketAddr`]. `buf` must be large enough to hold one UDP datagram;
    /// 65535 bytes is the recommended size.
    pub fn recv_into<'buf>(
        &self,
        buf: &'buf mut [u8],
    ) -> Result<(Message<'buf>, SocketAddr), TransportError> {
        let (n, addr) = self.socket.recv_from(buf)?;
        let msg = parse_message(&buf[..n])?;
        Ok((msg, addr))
    }

    /// Set or clear the read timeout on the underlying socket.
    pub fn set_read_timeout(&self, dur: Option<Duration>) -> Result<(), TransportError> {
        Ok(self.socket.set_read_timeout(dur)?)
    }

    /// Create a transport from a pre-built socket (e.g. one with SO_REUSEADDR/REUSEPORT set).
    pub fn from_socket(socket: UdpSocket) -> Self {
        Self { socket }
    }

    /// Returns the local address the socket is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, TransportError> {
        Ok(self.socket.local_addr()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::byte_cursor::Endianness;
    use crate::protocol::dds::message::submessage::Heartbeat;
    use crate::protocol::dds::message::{Message, MessageHeader, Submessage};
    use crate::protocol::dds::types::guid::{
        GuidPrefix, ENTITYID_UNKNOWN, PROTOCOL_VERSION_2_3, VENDOR_ID_OXICTL,
    };
    use crate::protocol::dds::types::sequence::SequenceNumber;

    fn make_test_header() -> MessageHeader {
        MessageHeader {
            version: PROTOCOL_VERSION_2_3,
            vendor_id: VENDOR_ID_OXICTL,
            guid_prefix: GuidPrefix([0xABu8; 12]),
        }
    }

    #[test]
    fn port_formulas_domain0() {
        assert_eq!(metatraffic_multicast_port(0), 7400);
        assert_eq!(metatraffic_unicast_port(0, 0), 7410);
        assert_eq!(metatraffic_unicast_port(0, 1), 7412);
        assert_eq!(user_traffic_multicast_port(0), 7401);
        assert_eq!(user_traffic_unicast_port(0, 0), 7411);
        assert_eq!(user_traffic_unicast_port(0, 1), 7413);
    }

    #[test]
    fn port_formulas_domain1() {
        assert_eq!(metatraffic_multicast_port(1), 7650);
        assert_eq!(metatraffic_unicast_port(1, 0), 7660);
        assert_eq!(user_traffic_multicast_port(1), 7651);
    }

    #[test]
    fn port_formulas_saturation() {
        // u16::MAX domain/participant should not panic.
        let _ = metatraffic_multicast_port(u16::MAX);
        let _ = metatraffic_unicast_port(u16::MAX, u16::MAX);
        let _ = user_traffic_multicast_port(u16::MAX);
        let _ = user_traffic_unicast_port(u16::MAX, u16::MAX);
    }

    #[test]
    fn send_recv_loopback() {
        let bind_a = SocketAddr::from(([127, 0, 0, 1], 0));
        let bind_b = SocketAddr::from(([127, 0, 0, 1], 0));

        let cfg_a = TransportConfig::unicast(bind_a);
        let cfg_b = TransportConfig {
            read_timeout: Some(Duration::from_millis(500)),
            ..TransportConfig::unicast(bind_b)
        };

        let a = UdpTransport::new(cfg_a).unwrap();
        let b = UdpTransport::new(cfg_b).unwrap();

        let b_addr = b.local_addr().unwrap();
        let b_locator = Locator::udp_v4(b_addr.port() as u32, [127, 0, 0, 1]);

        let hb = Heartbeat {
            endianness: Endianness::Little,
            final_flag: true,
            liveliness_flag: false,
            group_info_flag: false,
            reader_id: ENTITYID_UNKNOWN,
            writer_id: ENTITYID_UNKNOWN,
            first_sn: SequenceNumber::new(1),
            last_sn: SequenceNumber::new(10),
            count: 1,
        };

        let mut subs: heapless::Vec<Submessage<'_>, 64> = heapless::Vec::new();
        subs.push(Submessage::Heartbeat(hb)).unwrap();
        let msg = Message {
            header: make_test_header(),
            submessages: subs,
        };

        a.send_to(&msg, &b_locator).unwrap();

        let mut buf = [0u8; 65535];
        let (received, _sender) = b.recv_into(&mut buf).unwrap();

        assert_eq!(received.header.guid_prefix, GuidPrefix([0xABu8; 12]));
        assert_eq!(received.submessages.len(), 1);
        assert!(matches!(&received.submessages[0], Submessage::Heartbeat(h) if h.count == 1));
    }

    #[test]
    fn from_socket_local_addr() {
        let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
        let transport = UdpTransport::from_socket(sock);
        assert!(transport.local_addr().is_ok());
    }
}
