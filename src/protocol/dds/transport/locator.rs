use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};

use crate::protocol::dds::types::locator::{Locator, LOCATOR_KIND_UDP_V4, LOCATOR_KIND_UDP_V6};

/// Convert a UDPv4 or UDPv6 `Locator` to a `std::net::SocketAddr`.
///
/// Returns `None` for invalid or unsupported locator kinds (e.g. `LOCATOR_KIND_INVALID`).
pub fn locator_to_socket_addr(loc: &Locator) -> Option<SocketAddr> {
    match loc.kind {
        LOCATOR_KIND_UDP_V4 => {
            let ipv4 = loc.address_v4()?;
            Some(SocketAddr::V4(SocketAddrV4::new(
                Ipv4Addr::from(ipv4),
                loc.port as u16,
            )))
        }
        LOCATOR_KIND_UDP_V6 => {
            let ipv6: [u8; 16] = loc.address;
            Some(SocketAddr::V6(SocketAddrV6::new(
                Ipv6Addr::from(ipv6),
                loc.port as u16,
                0, // flowinfo
                0, // scope_id
            )))
        }
        _ => None,
    }
}

/// Convert a `std::net::SocketAddr` to a `Locator`.
pub fn socket_addr_to_locator(addr: SocketAddr) -> Locator {
    match addr {
        SocketAddr::V4(v4) => Locator::udp_v4(v4.port() as u32, v4.ip().octets()),
        SocketAddr::V6(v6) => Locator::udp_v6(v6.port() as u32, v6.ip().octets()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::types::locator::LOCATOR_INVALID;

    #[test]
    fn locator_v4_to_socket_addr() {
        let loc = Locator::udp_v4(7400, [192, 168, 1, 10]);
        let addr = locator_to_socket_addr(&loc).unwrap();
        assert!(
            matches!(addr, SocketAddr::V4(v4) if *v4.ip() == Ipv4Addr::new(192, 168, 1, 10) && v4.port() == 7400)
        );
    }

    #[test]
    fn socket_addr_v4_to_locator() {
        let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 1), 9999));
        let loc = socket_addr_to_locator(addr);
        assert_eq!(loc.kind, LOCATOR_KIND_UDP_V4);
        assert_eq!(loc.port, 9999);
        assert_eq!(loc.address_v4().unwrap(), [10, 0, 0, 1]);
    }

    #[test]
    fn locator_v4_round_trip() {
        let original = Locator::udp_v4(7410, [127, 0, 0, 1]);
        let addr = locator_to_socket_addr(&original).unwrap();
        let recovered = socket_addr_to_locator(addr);
        assert_eq!(original, recovered);
    }

    #[test]
    fn locator_invalid_kind_returns_none() {
        assert!(locator_to_socket_addr(&LOCATOR_INVALID).is_none());
    }
}
