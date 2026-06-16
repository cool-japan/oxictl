//! `DdsType` trait — implemented by any type that can be published over DDS.

use super::error::DdsApiError;

/// A received sample with its associated writer GUID.
pub struct Sample<T> {
    /// The deserialized data payload.
    pub data: T,
    /// Raw GUID bytes `[prefix(12) + entity_id(4)]` of the remote writer.
    pub writer_guid_bytes: [u8; 16],
}

/// Trait implemented by any type that can be sent and received over DDS.
///
/// The type must be able to serialize itself into a caller-provided buffer
/// (CDR little-endian, with a 4-byte `[0x00,0x01,0x00,0x00]` encapsulation
/// header prepended) and deserialize from the same format.
///
/// # Safety contract
/// - `serialize` writes exactly the bytes that `deserialize` expects.
/// - Both functions are deterministic and allocation-free.
pub trait DdsType: Sized {
    /// The DDS/ROS2 type name (e.g. `"std_msgs::msg::dds_::String_"`).
    const TYPE_NAME: &'static str;

    /// Serialize `self` into `buf`.
    ///
    /// Must write a 4-byte CDR encapsulation header `[0x00,0x01,0x00,0x00]`
    /// followed by the CDR little-endian payload.
    ///
    /// Returns the number of bytes written, or an error if `buf` is too small.
    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError>;

    /// Deserialize from a raw CDR payload (including the 4-byte header).
    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError>;
}
