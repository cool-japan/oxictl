//! Error type for the DDS user API.

use crate::protocol::dds::discovery::DiscoveryError;
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::stateful::StatefulError;
use crate::protocol::dds::transport::TransportError;

/// Errors that can occur in the high-level DDS API.
#[derive(Debug)]
pub enum DdsApiError {
    /// An error from the reliable stateful RTPS layer.
    Stateful(StatefulError),
    /// An error from the SPDP/SEDP discovery layer.
    Discovery(DiscoveryError),
    /// An error from the UDP transport layer.
    Transport(TransportError),
    /// An error from the RTPS wire protocol parser/serializer.
    Rtps(RtpsError),
    /// CDR encode/decode error (e.g., buffer too small, invalid encoding).
    Serialization(&'static str),
    /// The topic name is too long for the fixed-capacity buffer.
    TopicNameTooLong,
    /// The type name is too long for the fixed-capacity buffer.
    TypeNameTooLong,
    /// The payload buffer is too small to hold the serialized sample.
    PayloadBufferTooSmall,
}

impl core::fmt::Display for DdsApiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Stateful(e) => write!(f, "stateful RTPS error: {e:?}"),
            Self::Discovery(e) => write!(f, "DDS discovery error: {e:?}"),
            Self::Transport(e) => write!(f, "transport error: {e:?}"),
            Self::Rtps(e) => write!(f, "RTPS wire error: {e:?}"),
            Self::Serialization(msg) => write!(f, "CDR serialization error: {msg}"),
            Self::TopicNameTooLong => write!(f, "topic name exceeds 256-byte capacity"),
            Self::TypeNameTooLong => write!(f, "type name exceeds 256-byte capacity"),
            Self::PayloadBufferTooSmall => write!(f, "payload buffer too small for sample"),
        }
    }
}

impl From<StatefulError> for DdsApiError {
    fn from(e: StatefulError) -> Self {
        Self::Stateful(e)
    }
}

impl From<DiscoveryError> for DdsApiError {
    fn from(e: DiscoveryError) -> Self {
        Self::Discovery(e)
    }
}

impl From<TransportError> for DdsApiError {
    fn from(e: TransportError) -> Self {
        Self::Transport(e)
    }
}

impl From<RtpsError> for DdsApiError {
    fn from(e: RtpsError) -> Self {
        Self::Rtps(e)
    }
}
