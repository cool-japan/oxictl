//! `example_interfaces` ROS2 service types.
//!
//! Provides the `AddTwoInts` service (adds two integers) used in integration
//! tests and examples.

use crate::protocol::dds::api::dds_type::DdsType;
use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::api::service::wrappers::{Service, ServiceField};
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};

use super::{make_cursor, make_writer};

// ─── AddTwoInts_Request ───────────────────────────────────────────────────────

/// `example_interfaces/srv/AddTwoInts_Request` — request with two integers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AddTwoIntsRequest {
    pub a: i64,
    pub b: i64,
}

impl ServiceField for AddTwoIntsRequest {
    fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.write_i64(self.a)?;
        w.write_i64(self.b)?;
        Ok(())
    }

    fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let a = r.read_i64()?;
        let b = r.read_i64()?;
        Ok(Self { a, b })
    }
}

impl DdsType for AddTwoIntsRequest {
    const TYPE_NAME: &'static str = "example_interfaces::srv::dds_::AddTwoInts_Request_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.serialize_inner(&mut w)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        Self::deserialize_inner(&mut r)
    }
}

// ─── AddTwoInts_Response ──────────────────────────────────────────────────────

/// `example_interfaces/srv/AddTwoInts_Response` — response with the sum.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AddTwoIntsResponse {
    pub sum: i64,
}

impl ServiceField for AddTwoIntsResponse {
    fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.write_i64(self.sum)?;
        Ok(())
    }

    fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let sum = r.read_i64()?;
        Ok(Self { sum })
    }
}

impl DdsType for AddTwoIntsResponse {
    const TYPE_NAME: &'static str = "example_interfaces::srv::dds_::AddTwoInts_Response_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.serialize_inner(&mut w)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        Self::deserialize_inner(&mut r)
    }
}

// ─── AddTwoInts service descriptor ───────────────────────────────────────────

/// Service descriptor for `example_interfaces/srv/AddTwoInts`.
pub struct AddTwoInts;

impl Service for AddTwoInts {
    type Request = AddTwoIntsRequest;
    type Response = AddTwoIntsResponse;
    const REQUEST_TYPE_NAME: &'static str = "example_interfaces::srv::dds_::AddTwoInts_Request_";
    const RESPONSE_TYPE_NAME: &'static str = "example_interfaces::srv::dds_::AddTwoInts_Response_";
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_two_ints_request_type_name() {
        assert_eq!(
            AddTwoIntsRequest::TYPE_NAME,
            "example_interfaces::srv::dds_::AddTwoInts_Request_"
        );
    }

    #[test]
    fn add_two_ints_response_type_name() {
        assert_eq!(
            AddTwoIntsResponse::TYPE_NAME,
            "example_interfaces::srv::dds_::AddTwoInts_Response_"
        );
    }

    #[test]
    fn add_two_ints_request_round_trip() {
        let orig = AddTwoIntsRequest { a: 3, b: 4 };
        let mut buf = [0u8; 64];
        let n = orig.serialize(&mut buf).unwrap();
        let decoded = AddTwoIntsRequest::deserialize(&buf[..n]).unwrap();
        assert_eq!(orig, decoded);
    }

    #[test]
    fn add_two_ints_response_round_trip() {
        let orig = AddTwoIntsResponse { sum: 7 };
        let mut buf = [0u8; 64];
        let n = orig.serialize(&mut buf).unwrap();
        let decoded = AddTwoIntsResponse::deserialize(&buf[..n]).unwrap();
        assert_eq!(orig, decoded);
    }

    #[test]
    fn add_two_ints_service_type_names_match() {
        assert_eq!(AddTwoInts::REQUEST_TYPE_NAME, AddTwoIntsRequest::TYPE_NAME);
        assert_eq!(
            AddTwoInts::RESPONSE_TYPE_NAME,
            AddTwoIntsResponse::TYPE_NAME
        );
    }
}
