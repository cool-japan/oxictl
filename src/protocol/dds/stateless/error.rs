//! Error type for the stateless DDS writer/reader.

use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::transport::error::TransportError;

/// Errors produced by [`StatelessWriter`] and [`StatelessReader`].
///
/// [`StatelessWriter`]: super::writer::StatelessWriter
/// [`StatelessReader`]: super::reader::StatelessReader
#[derive(Debug)]
pub enum StatelessError {
    /// A network I/O or RTPS parse error from the transport layer.
    Transport(TransportError),
    /// An RTPS parse or serialization error.
    Parse(RtpsError),
    /// The history cache has zero capacity and cannot store any changes.
    HistoryFull,
    /// The serialization buffer is too small for the message.
    BufferTooSmall,
}

impl core::fmt::Display for StatelessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "stateless transport error: {e}"),
            Self::Parse(e) => write!(f, "stateless RTPS parse error: {e}"),
            Self::HistoryFull => write!(f, "stateless: history cache is full (capacity == 0)"),
            Self::BufferTooSmall => write!(f, "stateless: serialization buffer too small"),
        }
    }
}

impl std::error::Error for StatelessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TransportError> for StatelessError {
    fn from(e: TransportError) -> Self {
        Self::Transport(e)
    }
}

impl From<RtpsError> for StatelessError {
    fn from(e: RtpsError) -> Self {
        Self::Parse(e)
    }
}
