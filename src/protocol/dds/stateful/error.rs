//! Error type for the stateful DDS writer/reader.

use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::transport::error::TransportError;

/// Errors produced by [`StatefulWriter`] and [`StatefulReader`].
///
/// [`StatefulWriter`]: super::writer::StatefulWriter
/// [`StatefulReader`]: super::reader::StatefulReader
#[derive(Debug)]
pub enum StatefulError {
    /// A network I/O or RTPS parse error from the transport layer.
    Transport(TransportError),
    /// An RTPS parse or serialization error.
    Parse(RtpsError),
    /// The serialization buffer is too small for the message.
    BufferTooSmall,
    /// ACKNACK received from a writer GUID not in the matched set.
    NoSuchWriter,
    /// Attempt to remove a reader that is not in the matched set.
    NoSuchReader,
}

impl core::fmt::Display for StatefulError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "stateful transport error: {e}"),
            Self::Parse(e) => write!(f, "stateful RTPS parse error: {e}"),
            Self::BufferTooSmall => write!(f, "stateful: serialization buffer too small"),
            Self::NoSuchWriter => write!(f, "stateful: ACKNACK from unknown writer GUID"),
            Self::NoSuchReader => write!(f, "stateful: attempt to remove unknown reader GUID"),
        }
    }
}

impl std::error::Error for StatefulError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TransportError> for StatefulError {
    fn from(e: TransportError) -> Self {
        Self::Transport(e)
    }
}

impl From<RtpsError> for StatefulError {
    fn from(e: RtpsError) -> Self {
        Self::Parse(e)
    }
}
