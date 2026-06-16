//! `geometry_msgs` ROS2 message types.
//!
//! Provides geometric message types (`Vector3`, `Point`, `Quaternion`, `Pose`,
//! `PoseStamped`, `Twist`, `TwistStamped`, `Transform`, `TransformStamped`,
//! `Wrench`, `WrenchStamped`, `Accel`, `AccelStamped`, `Inertia`) with CDR
//! serialization.

use heapless::String as HString;

use crate::protocol::dds::api::dds_type::DdsType;
use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};

use super::std_msgs::Header;
use super::{make_cursor, make_writer};

// ─── Vector3 ─────────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/Vector3` — a 3D vector (x, y, z) in double precision.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Vector3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vector3 {
    /// Serialize fields (without CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.align_to(8)?;
        w.write_f64(self.x)?;
        w.write_f64(self.y)?;
        w.write_f64(self.z)?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        r.align_to(8)?;
        let x = r.read_f64()?;
        let y = r.read_f64()?;
        let z = r.read_f64()?;
        Ok(Self { x, y, z })
    }
}

impl DdsType for Vector3 {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::Vector3_";

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

// ─── Point ───────────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/Point` — a 3D point (x, y, z) in double precision.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Point {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
}

impl Point {
    /// Serialize fields (without CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.align_to(8)?;
        w.write_f64(self.x)?;
        w.write_f64(self.y)?;
        w.write_f64(self.z)?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        r.align_to(8)?;
        let x = r.read_f64()?;
        let y = r.read_f64()?;
        let z = r.read_f64()?;
        Ok(Self { x, y, z })
    }
}

impl DdsType for Point {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::Point_";

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

// ─── Quaternion ──────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/Quaternion` — unit quaternion for 3D orientation.
#[derive(Debug, Clone, PartialEq)]
pub struct Quaternion {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
    /// W (scalar) component.
    pub w: f64,
}

impl Default for Quaternion {
    fn default() -> Self {
        // Identity quaternion
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

impl Quaternion {
    /// Serialize fields (without CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.align_to(8)?;
        w.write_f64(self.x)?;
        w.write_f64(self.y)?;
        w.write_f64(self.z)?;
        w.write_f64(self.w)?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        r.align_to(8)?;
        let x = r.read_f64()?;
        let y = r.read_f64()?;
        let z = r.read_f64()?;
        let w = r.read_f64()?;
        Ok(Self { x, y, z, w })
    }
}

impl DdsType for Quaternion {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::Quaternion_";

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

// ─── Pose ────────────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/Pose` — 3D position and orientation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Pose {
    /// 3D position.
    pub position: Point,
    /// Orientation as a unit quaternion.
    pub orientation: Quaternion,
}

impl Pose {
    /// Serialize fields (without CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        self.position.serialize_inner(w)?;
        self.orientation.serialize_inner(w)?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let position = Point::deserialize_inner(r)?;
        let orientation = Quaternion::deserialize_inner(r)?;
        Ok(Self {
            position,
            orientation,
        })
    }
}

impl DdsType for Pose {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::Pose_";

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

// ─── PoseStamped ─────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/PoseStamped` — `Pose` with a `Header`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseStamped {
    /// Message header.
    pub header: Header,
    /// The pose data.
    pub pose: Pose,
}

impl DdsType for PoseStamped {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::PoseStamped_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        self.pose.serialize_inner(&mut w)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        let pose = Pose::deserialize_inner(&mut r)?;
        Ok(Self { header, pose })
    }
}

// ─── Twist ───────────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/Twist` — linear and angular velocity.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Twist {
    /// Linear velocity.
    pub linear: Vector3,
    /// Angular velocity.
    pub angular: Vector3,
}

impl Twist {
    /// Serialize fields (without CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        self.linear.serialize_inner(w)?;
        self.angular.serialize_inner(w)?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let linear = Vector3::deserialize_inner(r)?;
        let angular = Vector3::deserialize_inner(r)?;
        Ok(Self { linear, angular })
    }
}

impl DdsType for Twist {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::Twist_";

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

// ─── TwistStamped ────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/TwistStamped` — `Twist` with a `Header`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TwistStamped {
    /// Message header.
    pub header: Header,
    /// The twist data.
    pub twist: Twist,
}

impl DdsType for TwistStamped {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::TwistStamped_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        self.twist.serialize_inner(&mut w)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        let twist = Twist::deserialize_inner(&mut r)?;
        Ok(Self { header, twist })
    }
}

// ─── Transform ───────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/Transform` — translation and rotation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transform {
    /// Translation vector.
    pub translation: Vector3,
    /// Rotation as a unit quaternion.
    pub rotation: Quaternion,
}

impl Transform {
    /// Serialize fields (without CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        self.translation.serialize_inner(w)?;
        self.rotation.serialize_inner(w)?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let translation = Vector3::deserialize_inner(r)?;
        let rotation = Quaternion::deserialize_inner(r)?;
        Ok(Self {
            translation,
            rotation,
        })
    }
}

impl DdsType for Transform {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::Transform_";

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

// ─── TransformStamped ────────────────────────────────────────────────────────

/// `geometry_msgs/msg/TransformStamped` — `Transform` with `Header` and child frame ID.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TransformStamped {
    /// Message header.
    pub header: Header,
    /// The child frame ID.
    pub child_frame_id: HString<256>,
    /// The transform data.
    pub transform: Transform,
}

impl DdsType for TransformStamped {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::TransformStamped_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        w.write_cdr_string(self.child_frame_id.as_str())?;
        self.transform.serialize_inner(&mut w)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        let s = r.read_cdr_string()?;
        let mut child_frame_id = HString::<256>::new();
        child_frame_id
            .push_str(s)
            .map_err(|_| DdsApiError::Serialization("child_frame_id exceeds 256-byte capacity"))?;
        let transform = Transform::deserialize_inner(&mut r)?;
        Ok(Self {
            header,
            child_frame_id,
            transform,
        })
    }
}

// ─── Wrench ──────────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/Wrench` — force and torque.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Wrench {
    /// Force vector.
    pub force: Vector3,
    /// Torque vector.
    pub torque: Vector3,
}

impl Wrench {
    /// Serialize fields (without CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        self.force.serialize_inner(w)?;
        self.torque.serialize_inner(w)?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let force = Vector3::deserialize_inner(r)?;
        let torque = Vector3::deserialize_inner(r)?;
        Ok(Self { force, torque })
    }
}

impl DdsType for Wrench {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::Wrench_";

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

// ─── WrenchStamped ───────────────────────────────────────────────────────────

/// `geometry_msgs/msg/WrenchStamped` — `Wrench` with a `Header`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WrenchStamped {
    /// Message header.
    pub header: Header,
    /// The wrench data.
    pub wrench: Wrench,
}

impl DdsType for WrenchStamped {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::WrenchStamped_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        self.wrench.serialize_inner(&mut w)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        let wrench = Wrench::deserialize_inner(&mut r)?;
        Ok(Self { header, wrench })
    }
}

// ─── Accel ───────────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/Accel` — linear and angular acceleration.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Accel {
    /// Linear acceleration.
    pub linear: Vector3,
    /// Angular acceleration.
    pub angular: Vector3,
}

impl Accel {
    /// Serialize fields (without CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        self.linear.serialize_inner(w)?;
        self.angular.serialize_inner(w)?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let linear = Vector3::deserialize_inner(r)?;
        let angular = Vector3::deserialize_inner(r)?;
        Ok(Self { linear, angular })
    }
}

impl DdsType for Accel {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::Accel_";

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

// ─── AccelStamped ────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/AccelStamped` — `Accel` with a `Header`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AccelStamped {
    /// Message header.
    pub header: Header,
    /// The acceleration data.
    pub accel: Accel,
}

impl DdsType for AccelStamped {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::AccelStamped_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        self.accel.serialize_inner(&mut w)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        let accel = Accel::deserialize_inner(&mut r)?;
        Ok(Self { header, accel })
    }
}

// ─── Inertia ─────────────────────────────────────────────────────────────────

/// `geometry_msgs/msg/Inertia` — rigid body inertia (mass, CoM, inertia tensor).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Inertia {
    /// Mass in kg.
    pub m: f64,
    /// Centre of mass.
    pub com: Vector3,
    /// Inertia tensor element Ixx.
    pub ixx: f64,
    /// Inertia tensor element Ixy.
    pub ixy: f64,
    /// Inertia tensor element Ixz.
    pub ixz: f64,
    /// Inertia tensor element Iyy.
    pub iyy: f64,
    /// Inertia tensor element Iyz.
    pub iyz: f64,
    /// Inertia tensor element Izz.
    pub izz: f64,
}

impl DdsType for Inertia {
    const TYPE_NAME: &'static str = "geometry_msgs::msg::dds_::Inertia_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        w.align_to(8)?;
        w.write_f64(self.m)?;
        self.com.serialize_inner(&mut w)?;
        // After com (3 f64 = 24 bytes), still 8-byte aligned
        w.write_f64(self.ixx)?;
        w.write_f64(self.ixy)?;
        w.write_f64(self.ixz)?;
        w.write_f64(self.iyy)?;
        w.write_f64(self.iyz)?;
        w.write_f64(self.izz)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        r.align_to(8)?;
        let m = r.read_f64()?;
        let com = Vector3::deserialize_inner(&mut r)?;
        let ixx = r.read_f64()?;
        let ixy = r.read_f64()?;
        let ixz = r.read_f64()?;
        let iyy = r.read_f64()?;
        let iyz = r.read_f64()?;
        let izz = r.read_f64()?;
        Ok(Self {
            m,
            com,
            ixx,
            ixy,
            ixz,
            iyy,
            iyz,
            izz,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::ros2::msgs::builtin_interfaces::Time;

    #[test]
    fn vector3_round_trip() {
        let original = Vector3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let mut buf = [0u8; 64];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Vector3::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn vector3_type_name() {
        assert_eq!(Vector3::TYPE_NAME, "geometry_msgs::msg::dds_::Vector3_");
    }

    #[test]
    fn point_round_trip() {
        let original = Point {
            x: -1.5,
            y: 2.5,
            z: 0.0,
        };
        let mut buf = [0u8; 64];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Point::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn point_type_name() {
        assert_eq!(Point::TYPE_NAME, "geometry_msgs::msg::dds_::Point_");
    }

    #[test]
    fn quaternion_round_trip() {
        let original = Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.707,
            w: 0.707,
        };
        let mut buf = [0u8; 64];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Quaternion::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn quaternion_type_name() {
        assert_eq!(
            Quaternion::TYPE_NAME,
            "geometry_msgs::msg::dds_::Quaternion_"
        );
    }

    #[test]
    fn pose_round_trip() {
        let original = Pose {
            position: Point {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        };
        let mut buf = [0u8; 128];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Pose::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn pose_type_name() {
        assert_eq!(Pose::TYPE_NAME, "geometry_msgs::msg::dds_::Pose_");
    }

    #[test]
    fn pose_stamped_round_trip() {
        let mut frame_id = heapless::String::<256>::new();
        frame_id.push_str("base_link").unwrap();
        let original = PoseStamped {
            header: Header {
                stamp: Time { sec: 1, nanosec: 0 },
                frame_id,
            },
            pose: Pose::default(),
        };
        let mut buf = [0u8; 256];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = PoseStamped::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn pose_stamped_type_name() {
        assert_eq!(
            PoseStamped::TYPE_NAME,
            "geometry_msgs::msg::dds_::PoseStamped_"
        );
    }

    #[test]
    fn twist_round_trip() {
        let original = Twist {
            linear: Vector3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            angular: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
        };
        let mut buf = [0u8; 64];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Twist::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn twist_type_name() {
        assert_eq!(Twist::TYPE_NAME, "geometry_msgs::msg::dds_::Twist_");
    }

    #[test]
    fn twist_byte_layout() {
        // Twist{linear:{1,0,0}, angular:{0,0,1}}
        // Layout: 4 (header) + 4 pad (to align to 8) + 6×8 (f64 values) = 4 + 4 + 48 = 56 bytes
        // Wait — the body starts at offset 0 relative to writer, and writer.align_to(8) aligns to 8.
        // After header (4 bytes in buf but 0 in writer), first align_to(8) at writer pos 0 → no pad (already 0 mod 8 = 0).
        // So: 4 (header) + 0 (no pad since pos=0) + 48 (6 f64) = 52 bytes.
        let original = Twist {
            linear: Vector3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            angular: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
        };
        let mut buf = [0u8; 64];
        let written = original.serialize(&mut buf).unwrap();
        assert_eq!(written, 52); // 4 header + 6*8 f64
                                 // CDR LE header
        assert_eq!(&buf[0..4], &[0x00, 0x01, 0x00, 0x00]);
        // linear.x = 1.0_f64 as LE bytes
        let one_f64 = 1.0_f64.to_bits().to_le_bytes();
        assert_eq!(&buf[4..12], &one_f64);
    }

    #[test]
    fn twist_stamped_round_trip() {
        let mut frame_id = heapless::String::<256>::new();
        frame_id.push_str("odom").unwrap();
        let original = TwistStamped {
            header: Header {
                stamp: Time {
                    sec: 10,
                    nanosec: 0,
                },
                frame_id,
            },
            twist: Twist::default(),
        };
        let mut buf = [0u8; 256];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = TwistStamped::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn twist_stamped_type_name() {
        assert_eq!(
            TwistStamped::TYPE_NAME,
            "geometry_msgs::msg::dds_::TwistStamped_"
        );
    }

    #[test]
    fn transform_round_trip() {
        let original = Transform {
            translation: Vector3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            rotation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        };
        let mut buf = [0u8; 128];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Transform::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn transform_type_name() {
        assert_eq!(Transform::TYPE_NAME, "geometry_msgs::msg::dds_::Transform_");
    }

    #[test]
    fn transform_stamped_round_trip() {
        let mut frame_id = heapless::String::<256>::new();
        frame_id.push_str("world").unwrap();
        let mut child = heapless::String::<256>::new();
        child.push_str("robot_base").unwrap();
        let original = TransformStamped {
            header: Header {
                stamp: Time { sec: 5, nanosec: 0 },
                frame_id,
            },
            child_frame_id: child,
            transform: Transform::default(),
        };
        let mut buf = [0u8; 512];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = TransformStamped::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn transform_stamped_type_name() {
        assert_eq!(
            TransformStamped::TYPE_NAME,
            "geometry_msgs::msg::dds_::TransformStamped_"
        );
    }

    #[test]
    fn wrench_round_trip() {
        let original = Wrench {
            force: Vector3 {
                x: 10.0,
                y: 0.0,
                z: -5.0,
            },
            torque: Vector3 {
                x: 0.0,
                y: 2.0,
                z: 0.0,
            },
        };
        let mut buf = [0u8; 128];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Wrench::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn wrench_type_name() {
        assert_eq!(Wrench::TYPE_NAME, "geometry_msgs::msg::dds_::Wrench_");
    }

    #[test]
    fn wrench_stamped_round_trip() {
        let mut frame_id = heapless::String::<256>::new();
        frame_id.push_str("tool0").unwrap();
        let original = WrenchStamped {
            header: Header {
                stamp: Time { sec: 0, nanosec: 0 },
                frame_id,
            },
            wrench: Wrench::default(),
        };
        let mut buf = [0u8; 256];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = WrenchStamped::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn wrench_stamped_type_name() {
        assert_eq!(
            WrenchStamped::TYPE_NAME,
            "geometry_msgs::msg::dds_::WrenchStamped_"
        );
    }

    #[test]
    fn accel_round_trip() {
        let original = Accel {
            linear: Vector3 {
                x: 9.81,
                y: 0.0,
                z: 0.0,
            },
            angular: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.1,
            },
        };
        let mut buf = [0u8; 128];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Accel::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn accel_type_name() {
        assert_eq!(Accel::TYPE_NAME, "geometry_msgs::msg::dds_::Accel_");
    }

    #[test]
    fn accel_stamped_round_trip() {
        let mut frame_id = heapless::String::<256>::new();
        frame_id.push_str("imu_link").unwrap();
        let original = AccelStamped {
            header: Header {
                stamp: Time {
                    sec: 0,
                    nanosec: 500_000_000,
                },
                frame_id,
            },
            accel: Accel::default(),
        };
        let mut buf = [0u8; 256];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = AccelStamped::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn accel_stamped_type_name() {
        assert_eq!(
            AccelStamped::TYPE_NAME,
            "geometry_msgs::msg::dds_::AccelStamped_"
        );
    }

    #[test]
    fn inertia_round_trip() {
        let original = Inertia {
            m: 1.5,
            com: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.1,
            },
            ixx: 0.01,
            ixy: 0.0,
            ixz: 0.0,
            iyy: 0.01,
            iyz: 0.0,
            izz: 0.02,
        };
        let mut buf = [0u8; 256];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Inertia::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn inertia_type_name() {
        assert_eq!(Inertia::TYPE_NAME, "geometry_msgs::msg::dds_::Inertia_");
    }
}
