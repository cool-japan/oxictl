//! Best-effort StatelessWriter and StatelessReader (RTPS 2.3).
//!
//! Requires `dds-stateless` feature (which pulls in `dds-transport`).
//! Stateless writer/reader pairs operate without discovery — the caller
//! configures locators manually. No ACK/NACK cycle; DATA is fire-and-forget.

pub mod cache;
pub mod error;
pub mod reader;
pub mod writer;

pub use cache::{CacheChange, HistoryCache};
pub use error::StatelessError;
pub use reader::{ReaderConfig, ReceivedSample, StatelessReader};
pub use writer::{StatelessWriter, WriterConfig};
