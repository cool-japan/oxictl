//! Reliable stateful RTPS writer/reader (Phase 22.6).
//!
//! Provides [`StatefulWriter`] and [`StatefulReader`] which implement the
//! RTPS heartbeat/ACKNACK reliability protocol on top of the UDPv4 transport.

pub mod error;
pub mod reader;
pub mod reader_proxy;
pub mod writer;
pub mod writer_proxy;

pub use error::StatefulError;
pub use reader::{ReaderConfig, ReceivedSample, StatefulReader};
pub use writer::{StatefulWriter, WriterConfig};
