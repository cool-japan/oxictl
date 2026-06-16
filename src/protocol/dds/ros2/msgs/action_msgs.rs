//! `action_msgs` ROS2 service and message types.
//!
//! Provides `GoalInfo`, `GoalStatus`, `GoalStatusArray`, and the
//! `CancelGoal` service types used by ROS2 action servers.

use heapless::Vec as HVec;

use crate::protocol::dds::api::dds_type::DdsType;
use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::api::service::wrappers::{Service, ServiceField};
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};

use super::builtin_interfaces::Time;
use super::unique_identifier_msgs::Uuid;
use super::{make_cursor, make_writer};

// ─── GoalStatus constants ─────────────────────────────────────────────────────

pub mod goal_status {
    pub const UNKNOWN: i8 = 0;
    pub const ACCEPTED: i8 = 1;
    pub const EXECUTING: i8 = 2;
    pub const CANCELING: i8 = 3;
    pub const SUCCEEDED: i8 = 4;
    pub const CANCELED: i8 = 5;
    pub const ABORTED: i8 = 6;
}

// ─── GoalInfo ─────────────────────────────────────────────────────────────────

/// `action_msgs/msg/GoalInfo` — identifies a goal by UUID + timestamp.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GoalInfo {
    pub goal_id: Uuid,
    pub stamp: Time,
}

impl GoalInfo {
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        self.goal_id.serialize_inner(w)?;
        self.stamp.serialize_inner(w)?;
        Ok(())
    }

    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let goal_id = Uuid::deserialize_inner(r)?;
        let stamp = Time::deserialize_inner(r)?;
        Ok(Self { goal_id, stamp })
    }
}

impl DdsType for GoalInfo {
    const TYPE_NAME: &'static str = "action_msgs::msg::dds_::GoalInfo_";

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

// ─── GoalStatus ───────────────────────────────────────────────────────────────

/// `action_msgs/msg/GoalStatus` — status of a single goal.
///
/// Use the [`goal_status`] module constants for the `status` field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GoalStatus {
    pub goal_info: GoalInfo,
    /// One of the `goal_status::*` constants.
    pub status: i8,
}

impl GoalStatus {
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        self.goal_info.serialize_inner(w)?;
        w.write_u8(self.status as u8)?;
        Ok(())
    }

    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let goal_info = GoalInfo::deserialize_inner(r)?;
        let status = r.read_u8()? as i8;
        Ok(Self { goal_info, status })
    }
}

impl DdsType for GoalStatus {
    const TYPE_NAME: &'static str = "action_msgs::msg::dds_::GoalStatus_";

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

// ─── GoalStatusArray ──────────────────────────────────────────────────────────

/// `action_msgs/msg/GoalStatusArray` — array of goal statuses.
///
/// Holds up to 16 goals (fixed capacity for no_std compatibility).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GoalStatusArray {
    pub status_list: HVec<GoalStatus, 16>,
}

impl DdsType for GoalStatusArray {
    const TYPE_NAME: &'static str = "action_msgs::msg::dds_::GoalStatusArray_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.align_to(4)?;
        w.write_u32(self.status_list.len() as u32)?;
        for status in &self.status_list {
            status.serialize_inner(&mut w)?;
        }
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        r.align_to(4)?;
        let len = r.read_u32()? as usize;
        let mut status_list: HVec<GoalStatus, 16> = HVec::new();
        for _ in 0..len {
            let s = GoalStatus::deserialize_inner(&mut r)?;
            let _ = status_list.push(s);
        }
        Ok(Self { status_list })
    }
}

// ─── CancelGoal_Request ───────────────────────────────────────────────────────

/// `action_msgs/srv/CancelGoal_Request`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CancelGoalRequest {
    pub goal_info: GoalInfo,
}

impl ServiceField for CancelGoalRequest {
    fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        self.goal_info.serialize_inner(w)?;
        Ok(())
    }

    fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let goal_info = GoalInfo::deserialize_inner(r)?;
        Ok(Self { goal_info })
    }
}

impl DdsType for CancelGoalRequest {
    const TYPE_NAME: &'static str = "action_msgs::srv::dds_::CancelGoal_Request_";

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

// ─── CancelGoal_Response ──────────────────────────────────────────────────────

/// `action_msgs/srv/CancelGoal_Response`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CancelGoalResponse {
    pub return_code: i8,
    pub goals_canceling: HVec<GoalInfo, 16>,
}

impl ServiceField for CancelGoalResponse {
    fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.write_u8(self.return_code as u8)?;
        w.align_to(4)?;
        w.write_u32(self.goals_canceling.len() as u32)?;
        for goal in &self.goals_canceling {
            goal.serialize_inner(w)?;
        }
        Ok(())
    }

    fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let return_code = r.read_u8()? as i8;
        r.align_to(4)?;
        let len = r.read_u32()? as usize;
        let mut goals_canceling: HVec<GoalInfo, 16> = HVec::new();
        for _ in 0..len {
            let g = GoalInfo::deserialize_inner(r)?;
            let _ = goals_canceling.push(g);
        }
        Ok(Self {
            return_code,
            goals_canceling,
        })
    }
}

impl DdsType for CancelGoalResponse {
    const TYPE_NAME: &'static str = "action_msgs::srv::dds_::CancelGoal_Response_";

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

// ─── CancelGoal service descriptor ───────────────────────────────────────────

/// Service descriptor for `action_msgs/srv/CancelGoal`.
pub struct CancelGoal;

impl Service for CancelGoal {
    type Request = CancelGoalRequest;
    type Response = CancelGoalResponse;
    const REQUEST_TYPE_NAME: &'static str = "action_msgs::srv::dds_::CancelGoal_Request_";
    const RESPONSE_TYPE_NAME: &'static str = "action_msgs::srv::dds_::CancelGoal_Response_";
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_status_constants() {
        assert_eq!(goal_status::UNKNOWN, 0);
        assert_eq!(goal_status::ACCEPTED, 1);
        assert_eq!(goal_status::EXECUTING, 2);
        assert_eq!(goal_status::CANCELING, 3);
        assert_eq!(goal_status::SUCCEEDED, 4);
        assert_eq!(goal_status::CANCELED, 5);
        assert_eq!(goal_status::ABORTED, 6);
    }

    #[test]
    fn goal_info_round_trip() {
        use super::super::builtin_interfaces::Time;
        let orig = GoalInfo {
            goal_id: Uuid::from_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            stamp: Time {
                sec: 10,
                nanosec: 500,
            },
        };
        let mut buf = [0u8; 128];
        let n = orig.serialize(&mut buf).unwrap();
        let decoded = GoalInfo::deserialize(&buf[..n]).unwrap();
        assert_eq!(orig, decoded);
    }

    #[test]
    fn goal_status_round_trip() {
        use super::super::builtin_interfaces::Time;
        let orig = GoalStatus {
            goal_info: GoalInfo {
                goal_id: Uuid::nil(),
                stamp: Time::default(),
            },
            status: goal_status::EXECUTING,
        };
        let mut buf = [0u8; 128];
        let n = orig.serialize(&mut buf).unwrap();
        let decoded = GoalStatus::deserialize(&buf[..n]).unwrap();
        assert_eq!(orig, decoded);
    }

    #[test]
    fn goal_status_array_byte_layout() {
        let arr = GoalStatusArray::default();
        let mut buf = [0u8; 64];
        let n = arr.serialize(&mut buf).unwrap();
        // Header(4) + align_to(4)(0 bytes extra since pos=0) + u32(0) = 8 bytes
        assert_eq!(n, 8);
        // CDR LE header
        assert_eq!(&buf[0..4], &[0x00, 0x01, 0x00, 0x00]);
        // length=0 as LE u32
        assert_eq!(&buf[4..8], &[0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn cancel_goal_response_round_trip() {
        let mut resp = CancelGoalResponse {
            return_code: 0,
            goals_canceling: HVec::new(),
        };
        use super::super::builtin_interfaces::Time;
        let _ = resp.goals_canceling.push(GoalInfo {
            goal_id: Uuid::nil(),
            stamp: Time::default(),
        });
        let mut buf = [0u8; 256];
        let n = resp.serialize(&mut buf).unwrap();
        let decoded = CancelGoalResponse::deserialize(&buf[..n]).unwrap();
        assert_eq!(resp, decoded);
    }

    #[test]
    fn cancel_goal_service_type_names() {
        assert_eq!(
            CancelGoal::REQUEST_TYPE_NAME,
            "action_msgs::srv::dds_::CancelGoal_Request_"
        );
        assert_eq!(
            CancelGoal::RESPONSE_TYPE_NAME,
            "action_msgs::srv::dds_::CancelGoal_Response_"
        );
    }
}
