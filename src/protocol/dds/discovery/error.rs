use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::transport::TransportError;

/// Errors produced by the SPDP discovery layer.
#[derive(Debug)]
pub enum DiscoveryError {
    /// An error from the UDPv4 transport layer.
    Transport(TransportError),
    /// An error parsing or serializing an RTPS message.
    Parse(RtpsError),
    /// The CDR payload is shorter than the required 4-byte header.
    PayloadTooSmall,
    /// A required parameter field was absent from the ParameterList.
    MissingField(&'static str),
}

impl core::fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "discovery transport error: {e}"),
            Self::Parse(e) => write!(f, "discovery parse error: {e}"),
            Self::PayloadTooSmall => {
                write!(f, "discovery: CDR payload too small (< 4 bytes)")
            }
            Self::MissingField(name) => {
                write!(
                    f,
                    "discovery: required field '{name}' missing from ParameterList"
                )
            }
        }
    }
}

impl std::error::Error for DiscoveryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TransportError> for DiscoveryError {
    fn from(e: TransportError) -> Self {
        Self::Transport(e)
    }
}

impl From<RtpsError> for DiscoveryError {
    fn from(e: RtpsError) -> Self {
        Self::Parse(e)
    }
}
