//! `Service` / `ServiceField` traits and generic request/reply DDS wrappers.
//!
//! # Wire layout
//!
//! Both `RequestWrapper<S>` and `ReplyWrapper<S>` share the layout:
//!
//! ```text
//! [CDR header (4 B)] [SampleIdentity (24 B)] [body fields ...]
//! ```
//!
//! The `SampleIdentity` carries `writer_guid + sequence_number`.  The server
//! echoes it unchanged from request to reply so clients can correlate by GUID.

use crate::protocol::dds::api::dds_type::DdsType;
use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};

use super::sample_identity::SampleIdentity;

// ─── Traits ───────────────────────────────────────────────────────────────────

/// Trait for types that can be CDR-serialized as a service body field
/// (without the 4-byte CDR encapsulation header).
///
/// Implement `serialize_inner` / `deserialize_inner` following the same
/// conventions as the message types in `ros2::msgs`: body-relative alignment,
/// no CDR header, use `?` on `ByteWriter`/`ByteCursor` calls.
pub trait ServiceField: Sized {
    /// Serialize `self` into `w` (body-relative, no CDR header).
    fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError>;
    /// Deserialize `Self` from `r` (body-relative, no CDR header).
    fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError>;
}

/// Bundles a request type, a response type, and their DDS type names into a
/// single service descriptor.
///
/// Implement this trait for a unit struct named after the service, e.g.:
/// ```rust,ignore
/// struct AddTwoInts;
/// impl Service for AddTwoInts {
///     type Request = AddTwoInts_Request;
///     type Response = AddTwoInts_Response;
///     const REQUEST_TYPE_NAME: &'static str =
///         "example_interfaces::srv::dds_::AddTwoInts_Request_";
///     const RESPONSE_TYPE_NAME: &'static str =
///         "example_interfaces::srv::dds_::AddTwoInts_Response_";
/// }
/// ```
pub trait Service: Sized {
    /// CDR request body type.
    type Request: ServiceField + Clone;
    /// CDR response body type.
    type Response: ServiceField;
    /// Full DDS type name for the request topic.
    const REQUEST_TYPE_NAME: &'static str;
    /// Full DDS type name for the response topic.
    const RESPONSE_TYPE_NAME: &'static str;
}

// ─── RequestWrapper ───────────────────────────────────────────────────────────

/// DDS wrapper that prepends a `SampleIdentity` header before the request body.
pub struct RequestWrapper<S: Service> {
    pub header: SampleIdentity,
    pub body: S::Request,
}

impl<S: Service> DdsType for RequestWrapper<S> {
    const TYPE_NAME: &'static str = S::REQUEST_TYPE_NAME;

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        self.body.serialize_inner(&mut w)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = SampleIdentity::deserialize_inner(&mut r)?;
        let body = S::Request::deserialize_inner(&mut r)?;
        Ok(Self { header, body })
    }
}

// ─── ReplyWrapper ─────────────────────────────────────────────────────────────

/// DDS wrapper that prepends a `SampleIdentity` header before the response body.
///
/// The server copies the request's `SampleIdentity` into `header`; clients
/// compare `header.writer_guid` against their own request-writer GUID to discard
/// replies that belong to other clients.
pub struct ReplyWrapper<S: Service> {
    pub header: SampleIdentity,
    pub body: S::Response,
}

impl<S: Service> DdsType for ReplyWrapper<S> {
    const TYPE_NAME: &'static str = S::RESPONSE_TYPE_NAME;

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        self.body.serialize_inner(&mut w)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = SampleIdentity::deserialize_inner(&mut r)?;
        let body = S::Response::deserialize_inner(&mut r)?;
        Ok(Self { header, body })
    }
}

// ─── Local CDR helpers ────────────────────────────────────────────────────────

const CDR_LE_HEADER: [u8; 4] = [0x00, 0x01, 0x00, 0x00];

fn make_writer(buf: &mut [u8]) -> Result<ByteWriter<'_>, DdsApiError> {
    if buf.len() < 4 {
        return Err(DdsApiError::PayloadBufferTooSmall);
    }
    buf[..4].copy_from_slice(&CDR_LE_HEADER);
    Ok(ByteWriter::new(&mut buf[4..], Endianness::Little))
}

fn make_cursor(payload: &[u8]) -> Result<ByteCursor<'_>, DdsApiError> {
    if payload.len() < 4 {
        return Err(DdsApiError::Serialization(
            "payload shorter than CDR header",
        ));
    }
    let endianness = if payload[1] & 0x01 != 0 {
        Endianness::Little
    } else {
        Endianness::Big
    };
    Ok(ByteCursor::new(&payload[4..], endianness))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── minimal concrete service for testing ──────────────────────────────────

    struct TestService;

    #[derive(Debug, Clone, PartialEq)]
    struct TestRequest {
        value: i32,
    }

    #[derive(Debug, PartialEq)]
    struct TestResponse {
        doubled: i32,
    }

    impl ServiceField for TestRequest {
        fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
            w.write_i32(self.value)?;
            Ok(())
        }
        fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
            Ok(Self {
                value: r.read_i32()?,
            })
        }
    }

    impl ServiceField for TestResponse {
        fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
            w.write_i32(self.doubled)?;
            Ok(())
        }
        fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
            Ok(Self {
                doubled: r.read_i32()?,
            })
        }
    }

    impl Service for TestService {
        type Request = TestRequest;
        type Response = TestResponse;
        const REQUEST_TYPE_NAME: &'static str = "test::srv::dds_::TestService_Request_";
        const RESPONSE_TYPE_NAME: &'static str = "test::srv::dds_::TestService_Response_";
    }

    #[test]
    fn request_wrapper_round_trip() {
        let orig = RequestWrapper::<TestService> {
            header: SampleIdentity::new([0xAA; 16], 7),
            body: TestRequest { value: 42 },
        };
        let mut buf = [0u8; 64];
        let len = orig.serialize(&mut buf).unwrap();
        let decoded = RequestWrapper::<TestService>::deserialize(&buf[..len]).unwrap();
        assert_eq!(decoded.header, orig.header);
        assert_eq!(decoded.body, orig.body);
    }

    #[test]
    fn reply_wrapper_round_trip() {
        let orig = ReplyWrapper::<TestService> {
            header: SampleIdentity::new([0xBB; 16], 3),
            body: TestResponse { doubled: 84 },
        };
        let mut buf = [0u8; 64];
        let len = orig.serialize(&mut buf).unwrap();
        let decoded = ReplyWrapper::<TestService>::deserialize(&buf[..len]).unwrap();
        assert_eq!(decoded.header, orig.header);
        assert_eq!(decoded.body, orig.body);
    }

    #[test]
    fn request_wrapper_type_name() {
        assert_eq!(
            RequestWrapper::<TestService>::TYPE_NAME,
            "test::srv::dds_::TestService_Request_"
        );
    }

    #[test]
    fn reply_filter_rejects_foreign_guid() {
        // Simulates the client-side check: keep only replies where
        // header.writer_guid == my_request_writer_guid.
        let my_guid = [0xAA; 16];
        let foreign_guid = [0xBB; 16];

        let my_reply_header = SampleIdentity::new(my_guid, 1);
        let foreign_reply_header = SampleIdentity::new(foreign_guid, 1);

        // Accept matching guid
        assert_eq!(my_reply_header.writer_guid, my_guid);
        // Reject foreign guid
        assert_ne!(foreign_reply_header.writer_guid, my_guid);
    }
}
