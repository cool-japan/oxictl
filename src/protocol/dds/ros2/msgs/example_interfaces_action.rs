//! `example_interfaces` action types — Fibonacci action.
//!
//! Provides `FibonacciGoal`, `FibonacciResult`, `FibonacciFeedback`, and the
//! `Fibonacci` action descriptor used in integration tests and examples.

use heapless::Vec as HVec;

use crate::protocol::dds::api::action::types::Action;
use crate::protocol::dds::api::dds_type::DdsType;
use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::api::service::wrappers::ServiceField;
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};
use crate::protocol::dds::ros2::msgs::unique_identifier_msgs::Uuid;

use super::{make_cursor, make_writer};

// ─── FibonacciGoal ────────────────────────────────────────────────────────────

/// `example_interfaces/action/Fibonacci_Goal` — how many Fibonacci numbers to compute.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FibonacciGoal {
    pub order: i32,
}

impl ServiceField for FibonacciGoal {
    fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.write_i32(self.order)?;
        Ok(())
    }

    fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let order = r.read_i32()?;
        Ok(Self { order })
    }
}

// ─── FibonacciResult ──────────────────────────────────────────────────────────

/// `example_interfaces/action/Fibonacci_Result` — the full Fibonacci sequence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FibonacciResult {
    pub sequence: HVec<i32, 32>,
}

impl ServiceField for FibonacciResult {
    fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.align_to(4)?;
        w.write_u32(self.sequence.len() as u32)?;
        for &v in &self.sequence {
            w.write_i32(v)?;
        }
        Ok(())
    }

    fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        r.align_to(4)?;
        let len = r.read_u32()? as usize;
        let mut sequence: HVec<i32, 32> = HVec::new();
        for _ in 0..len {
            let v = r.read_i32()?;
            let _ = sequence.push(v);
        }
        Ok(Self { sequence })
    }
}

// ─── FibonacciFeedback ────────────────────────────────────────────────────────

/// `example_interfaces/action/Fibonacci_FeedbackMessage` — goal ID + partial sequence.
///
/// NOTE: This implements `DdsType` directly (not `ServiceField`) because feedback
/// is published as a topic message, not as a service body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FibonacciFeedback {
    pub goal_id: Uuid,
    pub partial_sequence: HVec<i32, 32>,
}

impl DdsType for FibonacciFeedback {
    const TYPE_NAME: &'static str = "example_interfaces::action::dds_::Fibonacci_FeedbackMessage_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.goal_id.serialize_inner(&mut w)?;
        w.align_to(4)?;
        w.write_u32(self.partial_sequence.len() as u32)?;
        for &v in &self.partial_sequence {
            w.write_i32(v)?;
        }
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let goal_id = Uuid::deserialize_inner(&mut r)?;
        r.align_to(4)?;
        let len = r.read_u32()? as usize;
        let mut partial_sequence: HVec<i32, 32> = HVec::new();
        for _ in 0..len {
            let v = r.read_i32()?;
            let _ = partial_sequence.push(v);
        }
        Ok(Self {
            goal_id,
            partial_sequence,
        })
    }
}

// ─── Fibonacci action descriptor ─────────────────────────────────────────────

/// Action descriptor for `example_interfaces/action/Fibonacci`.
pub struct Fibonacci;

impl Action for Fibonacci {
    type Goal = FibonacciGoal;
    type Result = FibonacciResult;
    type Feedback = FibonacciFeedback;

    const SEND_GOAL_REQUEST_TYPE_NAME: &'static str =
        "example_interfaces::action::dds_::Fibonacci_SendGoal_Request_";
    const SEND_GOAL_RESPONSE_TYPE_NAME: &'static str =
        "example_interfaces::action::dds_::Fibonacci_SendGoal_Response_";
    const GET_RESULT_REQUEST_TYPE_NAME: &'static str =
        "example_interfaces::action::dds_::Fibonacci_GetResult_Request_";
    const GET_RESULT_RESPONSE_TYPE_NAME: &'static str =
        "example_interfaces::action::dds_::Fibonacci_GetResult_Response_";
    const FEEDBACK_TYPE_NAME: &'static str =
        "example_interfaces::action::dds_::Fibonacci_FeedbackMessage_";
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::byte_cursor::Endianness;

    #[test]
    fn fibonacci_goal_round_trip() {
        let goal = FibonacciGoal { order: 10 };
        let mut buf = [0u8; 64];
        let mut w = ByteWriter::new(&mut buf, Endianness::Little);
        goal.serialize_inner(&mut w).unwrap();
        let pos = w.position();

        let mut r = ByteCursor::new(&buf[..pos], Endianness::Little);
        let decoded = FibonacciGoal::deserialize_inner(&mut r).unwrap();
        assert_eq!(decoded, goal);
    }

    #[test]
    fn fibonacci_result_round_trip() {
        let mut result = FibonacciResult::default();
        let _ = result.sequence.push(0);
        let _ = result.sequence.push(1);
        let _ = result.sequence.push(1);
        let _ = result.sequence.push(2);

        let mut buf = [0u8; 128];
        let mut w = ByteWriter::new(&mut buf, Endianness::Little);
        result.serialize_inner(&mut w).unwrap();
        let pos = w.position();

        let mut r = ByteCursor::new(&buf[..pos], Endianness::Little);
        let decoded = FibonacciResult::deserialize_inner(&mut r).unwrap();
        assert_eq!(decoded, result);
    }

    #[test]
    fn fibonacci_feedback_round_trip() {
        let mut fb = FibonacciFeedback {
            goal_id: Uuid::from_bytes([7u8; 16]),
            partial_sequence: HVec::new(),
        };
        let _ = fb.partial_sequence.push(0);
        let _ = fb.partial_sequence.push(1);

        let mut buf = [0u8; 128];
        let len = fb.serialize(&mut buf).unwrap();
        let decoded = FibonacciFeedback::deserialize(&buf[..len]).unwrap();
        assert_eq!(decoded, fb);
    }

    #[test]
    fn fibonacci_type_names() {
        assert_eq!(
            Fibonacci::SEND_GOAL_REQUEST_TYPE_NAME,
            "example_interfaces::action::dds_::Fibonacci_SendGoal_Request_"
        );
        assert_eq!(
            FibonacciFeedback::TYPE_NAME,
            "example_interfaces::action::dds_::Fibonacci_FeedbackMessage_"
        );
    }

    #[test]
    fn fibonacci_feedback_type_name() {
        assert_eq!(
            <Fibonacci as Action>::FEEDBACK_TYPE_NAME,
            "example_interfaces::action::dds_::Fibonacci_FeedbackMessage_"
        );
    }
}
