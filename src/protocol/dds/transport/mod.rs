//! UDPv4 transport for RTPS 2.3 messages.
//!
//! Requires `std` (uses `std::net::UdpSocket`). Feature-gated by `dds-transport`.
//! Wraps Phase 22.1 `parse_message` / `serialize_message` for real network I/O.

pub mod error;
pub mod locator;
pub mod multicast_socket;
pub mod udp;

pub use error::TransportError;
pub use locator::{locator_to_socket_addr, socket_addr_to_locator};
pub use multicast_socket::{bind_multicast_reuse, probe_multicast_loopback};
pub use udp::{
    metatraffic_multicast_port, metatraffic_unicast_port, user_traffic_multicast_port,
    user_traffic_unicast_port, TransportConfig, UdpTransport, SPDP_MULTICAST_IPV4,
};
