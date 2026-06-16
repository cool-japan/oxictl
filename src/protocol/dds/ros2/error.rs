//! Error type for the ROS2-on-DDS bridge.

use crate::protocol::dds::RtpsError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ros2Error {
    /// Topic or service name contains invalid characters (whitespace, NUL, etc.)
    InvalidTopicName,
    /// Package or type name contains invalid characters.
    InvalidTypeName,
    /// Encoded name exceeds the heapless::String<256> capacity.
    NameBufferTooSmall,
    /// Log severity value is not one of the 5 known constants.
    InvalidLogLevel,
    /// ParameterValue type discriminant is not in 0..=9.
    UnknownParameterType,
    /// Array of parameters exceeds the heapless::Vec<_, 16> capacity.
    TooManyParameters,
    /// Parameter array value exceeds the heapless::Vec<_, 32> capacity.
    TooManyArrayElements,
    /// Underlying RTPS wire format error.
    Rtps(RtpsError),
}

impl From<RtpsError> for Ros2Error {
    fn from(e: RtpsError) -> Self {
        Ros2Error::Rtps(e)
    }
}

impl core::fmt::Display for Ros2Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Ros2Error::InvalidTopicName => f.write_str("ros2: invalid topic name"),
            Ros2Error::InvalidTypeName => f.write_str("ros2: invalid type name"),
            Ros2Error::NameBufferTooSmall => f.write_str("ros2: name buffer too small"),
            Ros2Error::InvalidLogLevel => f.write_str("ros2: invalid log level"),
            Ros2Error::UnknownParameterType => f.write_str("ros2: unknown parameter type"),
            Ros2Error::TooManyParameters => f.write_str("ros2: too many parameters"),
            Ros2Error::TooManyArrayElements => f.write_str("ros2: too many array elements"),
            Ros2Error::Rtps(e) => write!(f, "ros2: rtps: {}", e),
        }
    }
}
