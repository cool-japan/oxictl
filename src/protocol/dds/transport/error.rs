use crate::protocol::dds::error::RtpsError;

/// Errors produced by the UDPv4 transport layer.
#[derive(Debug)]
pub enum TransportError {
    /// An OS-level I/O error from `std::net::UdpSocket`.
    Io(std::io::Error),
    /// An error parsing or serializing an RTPS message.
    Parse(RtpsError),
    /// The caller-supplied buffer is too small to hold the serialized message.
    BufferTooSmall,
    /// The locator kind is not UDP v4 or UDP v6 and cannot be converted to a socket address.
    InvalidLocator,
}

impl core::fmt::Display for TransportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "transport I/O error: {e}"),
            Self::Parse(e) => write!(f, "RTPS parse error: {e}"),
            Self::BufferTooSmall => write!(f, "buffer too small for RTPS message"),
            Self::InvalidLocator => write!(f, "locator kind not supported for UDP transport"),
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TransportError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<RtpsError> for TransportError {
    fn from(e: RtpsError) -> Self {
        Self::Parse(e)
    }
}
