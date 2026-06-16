//! ROS2-on-DDS bridge — topic/type naming, builtin endpoints, and CDR codecs.
//!
//! Requires `dds-ros2` feature (`dds-stateful` + `dds-discovery`).

pub mod builtin_endpoints;
pub mod error;
pub mod log;
#[cfg(feature = "dds-api")]
pub mod msgs;
pub mod parameter;
pub mod topic_naming;

pub use builtin_endpoints::{
    Ros2BuiltinEndpoint, DISC_BUILTIN_ENDPOINT_PARTICIPANT_ANNOUNCER,
    DISC_BUILTIN_ENDPOINT_PARTICIPANT_DETECTOR, DISC_BUILTIN_ENDPOINT_PUBLICATIONS_ANNOUNCER,
    DISC_BUILTIN_ENDPOINT_PUBLICATIONS_DETECTOR, DISC_BUILTIN_ENDPOINT_SUBSCRIPTIONS_ANNOUNCER,
    DISC_BUILTIN_ENDPOINT_SUBSCRIPTIONS_DETECTOR, DISC_BUILTIN_ENDPOINT_TOPICS_ANNOUNCER,
    DISC_BUILTIN_ENDPOINT_TOPICS_DETECTOR, ROS2_BUILTIN_CLOCK, ROS2_BUILTIN_PARAMETER_EVENTS,
    ROS2_BUILTIN_ROSOUT, ROS2_DEFAULT_ENDPOINT_SET,
};
pub use error::Ros2Error;
pub use log::{BuiltinTime, LogMsg, LogSeverity};
pub use parameter::{Parameter as Ros2Parameter, ParameterEventMsg, ParameterType, ParameterValue};
pub use topic_naming::{
    decode_topic_name, encode_action_subtopic, encode_topic_name, encode_type_name, ActionSubtopic,
    Ros2TopicKind, TypeNamespace, TypeSuffix,
};

// ─── Shared CDR string helpers ────────────────────────────────────────────────
//
// Used by both `log.rs` and `parameter.rs`.

use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::ros2::error::Ros2Error as Ros2ErrorAlias;

/// Read a CDR string from the cursor, returning a borrowed `&str` into the input buffer.
///
/// Format: `[u32 length (includes null terminator)][bytes][null][zero padding to 4-byte alignment]`.
pub(super) fn read_cdr_str<'a>(cur: &mut ByteCursor<'a>) -> Result<&'a str, Ros2ErrorAlias> {
    let len = cur.read_u32().map_err(Ros2ErrorAlias::from)? as usize;

    if len == 0 {
        // Zero length — non-standard; treat as empty
        return Ok("");
    }

    // `len` includes the trailing NUL
    let raw = cur.read_bytes(len).map_err(Ros2ErrorAlias::from)?;

    // Strip trailing NUL if present
    let s_bytes = if raw.last() == Some(&0) {
        &raw[..raw.len() - 1]
    } else {
        raw
    };

    let s = core::str::from_utf8(s_bytes)
        .map_err(|_| Ros2ErrorAlias::from(RtpsError::InvalidStringEncoding))?;

    // Advance past padding to 4-byte alignment (len includes NUL; total written = 4 + len)
    // The cursor is now at position `start + 4 + len`.
    // We need to align `4 + len` to the next multiple of 4.
    // Padding = ((len + 3) & !3) - len  = (4 - (len % 4)) % 4
    let pad = (4 - (len % 4)) % 4;
    if pad > 0 {
        cur.skip(pad).map_err(Ros2ErrorAlias::from)?;
    }

    Ok(s)
}

/// Write a CDR string into the writer.
///
/// Format: `[u32 length (includes null terminator)][bytes][null][zero padding to 4-byte alignment]`.
pub(super) fn write_cdr_str(w: &mut ByteWriter<'_>, s: &str) -> Result<(), Ros2ErrorAlias> {
    let with_null = s.len() + 1;
    w.write_u32(with_null as u32)
        .map_err(Ros2ErrorAlias::from)?;
    w.write_bytes(s.as_bytes()).map_err(Ros2ErrorAlias::from)?;
    w.write_u8(0).map_err(Ros2ErrorAlias::from)?; // NUL terminator
                                                  // Padding to align (len % 4) where len = s.len() + 1
    let pad = (4 - (with_null % 4)) % 4;
    if pad > 0 {
        let zeros = [0u8; 3];
        w.write_bytes(&zeros[..pad]).map_err(Ros2ErrorAlias::from)?;
    }
    Ok(())
}

/// Compute the serialized byte count for a CDR string (4-byte length prefix + NUL-terminated bytes padded to 4).
pub(super) fn cdr_str_len(s: &str) -> usize {
    let with_null = s.len() + 1;
    // Round up to 4-byte boundary
    let aligned = (with_null + 3) & !3;
    4 + aligned
}

// ─── msgs re-exports ──────────────────────────────────────────────────────────

#[cfg(feature = "dds-api")]
pub use msgs::{
    action_msgs::{
        goal_status, CancelGoal, CancelGoalRequest, CancelGoalResponse, GoalInfo, GoalStatus,
        GoalStatusArray,
    },
    builtin_interfaces::{Duration, Time},
    example_interfaces::{AddTwoInts, AddTwoIntsRequest, AddTwoIntsResponse},
    geometry_msgs::{Point, Pose, Quaternion, Transform, Twist, Vector3, Wrench},
    sensor_msgs::{Imu, JointState, Range, Temperature},
    std_msgs::{Bool, Float32, Float64, Header, Int32, Int64, StdString, UInt32, UInt64},
    unique_identifier_msgs::Uuid,
};
