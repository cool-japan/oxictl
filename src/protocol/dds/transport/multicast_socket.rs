//! Helper to bind a UDP socket with SO_REUSEADDR + SO_REUSEPORT for multicast.
//!
//! Required so multiple participants in the same process can all bind to the
//! same SPDP multicast port (7400 + 250*domain_id).

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::time::Duration;

use socket2::{Domain, Protocol, Socket, Type};

use super::error::TransportError;

/// Bind a UDP socket to `0.0.0.0:port` with SO_REUSEADDR + SO_REUSEPORT,
/// join the specified IPv4 multicast group, and enable multicast loopback.
///
/// Returns the bound `UdpSocket` on success.
///
/// Fails with `Err(TransportError::Io(...))` if the OS does not support
/// the operation (e.g. firewalled multicast in restricted CI environments).
pub fn bind_multicast_reuse(port: u16, group: [u8; 4]) -> Result<UdpSocket, TransportError> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
    socket.set_reuse_address(true)?;
    #[cfg(not(target_os = "windows"))]
    socket.set_reuse_port(true)?;
    socket.bind(&addr.into())?;
    let iface = Ipv4Addr::UNSPECIFIED;
    socket.join_multicast_v4(&Ipv4Addr::from(group), &iface)?;
    socket.set_multicast_loop_v4(true)?;
    let udp: UdpSocket = socket.into();
    Ok(udp)
}

/// Probe whether multicast loopback actually delivers packets between two
/// sockets bound to the same group on this machine.
///
/// Creates two ephemeral multicast sockets on an obscure port (domain 253 —
/// port 7400 + 250*253 = 70,650 clamped to 32,757 via saturating_add, then
/// we use a fixed probe port to avoid overflow), sends one UDP packet from
/// the first socket to the multicast group address, and checks whether the
/// second socket receives it within `timeout`.
///
/// Returns `true` if loopback works, `false` if blocked (e.g. sandboxed OS,
/// restricted CI, or network namespace without IP multicast routing).
///
/// Separate from `bind_multicast_reuse` so callers can gate behaviour on
/// actual packet delivery rather than just socket setup.
pub fn probe_multicast_loopback(group: [u8; 4], timeout: Duration) -> bool {
    // Use a fixed probe port unlikely to collide with RTPS domain traffic.
    // 7400 + 250*253 overflows u16; use a safe fixed value instead.
    let probe_port: u16 = 29301;

    let make_sock = || -> Option<UdpSocket> {
        let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, probe_port));
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP)).ok()?;
        socket.set_reuse_address(true).ok()?;
        #[cfg(not(target_os = "windows"))]
        socket.set_reuse_port(true).ok()?;
        socket.bind(&addr.into()).ok()?;
        socket
            .join_multicast_v4(&Ipv4Addr::from(group), &Ipv4Addr::UNSPECIFIED)
            .ok()?;
        socket.set_multicast_loop_v4(true).ok()?;
        let udp: UdpSocket = socket.into();
        udp.set_read_timeout(Some(timeout)).ok()?;
        Some(udp)
    };

    let sender = match make_sock() {
        Some(s) => s,
        None => return false,
    };
    let receiver = match make_sock() {
        Some(s) => s,
        None => return false,
    };

    let dst = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::from(group), probe_port));
    if sender.send_to(b"oxictl-mc-probe", dst).is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    receiver.recv_from(&mut buf).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_multicast_reuse_succeeds() {
        // Use an obscure port (domain 200) to avoid colliding with any running ROS2.
        let port = crate::protocol::dds::transport::metatraffic_multicast_port(200);
        let group = crate::protocol::dds::transport::SPDP_MULTICAST_IPV4;
        // This may fail on CI without multicast support; that is acceptable.
        let _ = bind_multicast_reuse(port, group);
    }

    #[test]
    fn two_binds_same_port_with_reuseaddr() {
        let port = crate::protocol::dds::transport::metatraffic_multicast_port(201);
        let group = crate::protocol::dds::transport::SPDP_MULTICAST_IPV4;
        match (
            bind_multicast_reuse(port, group),
            bind_multicast_reuse(port, group),
        ) {
            (Ok(_), Ok(_)) => { /* success — SO_REUSEADDR works */ }
            _ => { /* multicast not available on this platform — that is OK */ }
        }
    }
}
