//! RTPS 2.3 wire-protocol parser and serializer.
//!
//! Zero-allocation, no_std-compatible implementation of the OMG Real-Time
//! Publish-Subscribe (RTPS) Protocol version 2.3.
//!
//! # Scope
//! This module covers the complete RTPS 2.3 wire-protocol: message framing,
//! all 13 submessage kinds, ParameterList (40+ PIDs), and all RTPS primitive types.
//! The `dds-transport` feature adds the UDPv4 transport layer (`transport` submodule).
//!
//! # Zero-alloc contract
//! The parser borrows from the input `&[u8]` slice and returns a `Message<'a>`
//! that holds lifetime-bound references into it. The serializer writes into a
//! caller-supplied `&mut [u8]` buffer and returns the byte count written.
//! Neither allocates from the heap.
//!
//! # no_std
//! This module builds with `#![no_std]`. Use `heapless::Vec` for bounded
//! collections. Do not import from `std::` in production paths.

#[cfg(feature = "dds-api")]
pub mod api;
pub mod byte_cursor;
#[cfg(feature = "dds-discovery")]
pub mod discovery;
pub mod error;
pub mod message;
pub mod parser;
#[cfg(feature = "dds-ros2")]
pub mod ros2;
pub mod serializer;
#[cfg(feature = "dds-stateful")]
pub mod stateful;
#[cfg(feature = "dds-stateless")]
pub mod stateless;
#[cfg(test)]
mod tests;
#[cfg(feature = "dds-transport")]
pub mod transport;
pub mod types;

#[cfg(feature = "dds-api")]
pub use api::{DdsApiError, DdsType, Participant, Publisher, Sample, Subscription};
pub use byte_cursor::Endianness;
pub use error::RtpsError;
pub use message::{Message, MessageHeader, Submessage, SubmessageKind};
pub use parser::parse_message;
#[cfg(feature = "dds-ros2")]
pub use ros2::{
    decode_topic_name, encode_action_subtopic, encode_topic_name, encode_type_name, ActionSubtopic,
    BuiltinTime, LogMsg, LogSeverity, ParameterEventMsg, ParameterValue, Ros2Error, Ros2Parameter,
    Ros2TopicKind, TypeNamespace, TypeSuffix, ROS2_BUILTIN_CLOCK, ROS2_BUILTIN_PARAMETER_EVENTS,
    ROS2_BUILTIN_ROSOUT, ROS2_DEFAULT_ENDPOINT_SET,
};
pub use serializer::serialize_message;
#[cfg(feature = "dds-stateful")]
pub use stateful::{
    ReaderConfig as StatefulReaderConfig, ReceivedSample as StatefulReceivedSample, StatefulError,
    StatefulReader, StatefulWriter, WriterConfig as StatefulWriterConfig,
};
#[cfg(feature = "dds-stateless")]
pub use stateless::{
    CacheChange, HistoryCache, ReaderConfig, ReceivedSample, StatelessError, StatelessReader,
    StatelessWriter, WriterConfig,
};

pub use types::{
    fragment::{FragmentNumber, FragmentNumberSet},
    guid::{
        EntityId, Guid, GuidPrefix, ProtocolVersion, VendorId, ENTITYID_PARTICIPANT,
        ENTITYID_SPDP_BUILTIN_PARTICIPANT_READER, ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER,
        ENTITYID_UNKNOWN, GUID_UNKNOWN, PROTOCOL_VERSION_2_3, VENDOR_ID_OXICTL,
    },
    locator::{Locator, LOCATOR_INVALID},
    parameter::{Parameter, ParameterList},
    sequence::{SequenceNumber, SequenceNumberSet, SEQUENCENUMBER_UNKNOWN, SEQUENCENUMBER_ZERO},
    time::{Duration, Time, TIME_INFINITE, TIME_INVALID, TIME_ZERO},
};
