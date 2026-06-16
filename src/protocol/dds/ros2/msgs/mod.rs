//! ROS2 standard message types with DDS CDR serialization.
//!
//! This module provides `DdsType` implementations for 35 ROS2 standard message
//! types covering `builtin_interfaces`, `std_msgs`, `geometry_msgs`, and
//! `sensor_msgs` packages.
//!
//! All types use CDR little-endian encoding with a 4-byte `[0x00,0x01,0x00,0x00]`
//! encapsulation header.

pub mod action_msgs;
pub mod builtin_interfaces;
pub mod example_interfaces;
pub mod example_interfaces_action;
pub mod geometry_msgs;
pub mod sensor_msgs;
pub mod std_msgs;
pub mod unique_identifier_msgs;

pub use action_msgs::{
    goal_status, CancelGoal, CancelGoalRequest, CancelGoalResponse, GoalInfo, GoalStatus,
    GoalStatusArray,
};
pub use builtin_interfaces::{Duration, Time};
pub use example_interfaces::{AddTwoInts, AddTwoIntsRequest, AddTwoIntsResponse};
pub use example_interfaces_action::{Fibonacci, FibonacciFeedback, FibonacciGoal, FibonacciResult};
pub use geometry_msgs::{Point, Pose, Quaternion, Twist, Vector3};
pub use sensor_msgs::{Imu, JointState};
pub use std_msgs::{Bool, Header, StdString};
pub use unique_identifier_msgs::Uuid;

use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter, Endianness};

/// CDR little-endian encapsulation header bytes.
const CDR_LE_HEADER: [u8; 4] = [0x00, 0x01, 0x00, 0x00];

/// Create a `ByteWriter` anchored at the body start (after the CDR header).
///
/// Writes the 4-byte `[0x00, 0x01, 0x00, 0x00]` header directly to `buf[0..4]`
/// and returns a writer over `buf[4..]` so that `writer.pos == 0` corresponds
/// to CDR body offset 0. This is essential for correct CDR alignment: all
/// `align_to` calls inside serialize methods are body-relative.
pub(crate) fn make_writer(buf: &mut [u8]) -> Result<ByteWriter<'_>, DdsApiError> {
    if buf.len() < 4 {
        return Err(DdsApiError::PayloadBufferTooSmall);
    }
    buf[..4].copy_from_slice(&CDR_LE_HEADER);
    Ok(ByteWriter::new(&mut buf[4..], Endianness::Little))
}

/// Create a `ByteCursor` anchored at the body start (after the CDR header).
///
/// Reads the endianness from byte 1 of the header and returns a cursor over
/// `payload[4..]` so that `cursor.pos == 0` corresponds to CDR body offset 0.
pub(crate) fn make_cursor(payload: &[u8]) -> Result<ByteCursor<'_>, DdsApiError> {
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
