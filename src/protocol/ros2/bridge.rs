//! ROS2 message bridge — converts between oxictl types and ROS2 message formats.
//!
//! Provides serialization/deserialization stubs for common control messages:
//!   - sensor_msgs/JointState (position, velocity, effort)
//!   - geometry_msgs/Twist (linear/angular velocity commands)
//!   - std_msgs/Float64MultiArray (generic scalar arrays)

/// Joint state message (sensor_msgs/JointState).
#[derive(Debug, Clone, Copy)]
pub struct JointState<const N: usize> {
    pub stamp_ns: u64,
    pub position: [f64; N],
    pub velocity: [f64; N],
    pub effort: [f64; N],
}

impl<const N: usize> JointState<N> {
    pub fn zero(stamp_ns: u64) -> Self {
        Self {
            stamp_ns,
            position: [0.0; N],
            velocity: [0.0; N],
            effort: [0.0; N],
        }
    }
}

/// Twist message (geometry_msgs/Twist): linear + angular velocity.
#[derive(Debug, Clone, Copy, Default)]
pub struct Twist {
    pub linear_x: f64,
    pub linear_y: f64,
    pub linear_z: f64,
    pub angular_x: f64,
    pub angular_y: f64,
    pub angular_z: f64,
}

/// Float64MultiArray message for generic data.
#[derive(Debug, Clone, Copy)]
pub struct Float64Array<const N: usize> {
    pub stamp_ns: u64,
    pub data: [f64; N],
}

/// Serialization trait for ROS2 CDR encoding (simplified).
///
/// In production this would use the `rclrs` message generation pipeline.
pub trait RosCdrSerialize {
    fn cdr_size(&self) -> usize;
    fn serialize(&self, buf: &mut [u8]) -> usize;
}

impl<const N: usize> RosCdrSerialize for Float64Array<N> {
    fn cdr_size(&self) -> usize {
        8 + N * 8 // stamp (8) + N f64s
    }

    fn serialize(&self, buf: &mut [u8]) -> usize {
        if buf.len() < self.cdr_size() {
            return 0;
        }
        buf[..8].copy_from_slice(&self.stamp_ns.to_le_bytes());
        for (i, &v) in self.data.iter().enumerate() {
            let offset = 8 + i * 8;
            buf[offset..offset + 8].copy_from_slice(&v.to_le_bytes());
        }
        self.cdr_size()
    }
}

impl<const N: usize> Float64Array<N> {
    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < 8 + N * 8 {
            return None;
        }
        let stamp_ns = u64::from_le_bytes(buf[..8].try_into().ok()?);
        let data: [f64; N] = core::array::from_fn(|i| {
            let offset = 8 + i * 8;
            f64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap_or([0u8; 8]))
        });
        Some(Self { stamp_ns, data })
    }
}

/// Bridge: convert JointState to Float64Array for stamped publishing.
pub fn joint_positions_to_array<const N: usize>(js: &JointState<N>) -> Float64Array<N> {
    Float64Array {
        stamp_ns: js.stamp_ns,
        data: js.position,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float64_array_serialize_deserialize() {
        let arr = Float64Array {
            stamp_ns: 12345,
            data: [1.0, 2.5, -3.0],
        };
        let mut buf = [0u8; 32];
        let n = arr.serialize(&mut buf);
        assert_eq!(n, 8 + 3 * 8);

        let parsed = Float64Array::<3>::deserialize(&buf[..n]).unwrap();
        assert_eq!(parsed.stamp_ns, 12345);
        assert!((parsed.data[0] - 1.0).abs() < 1e-12);
        assert!((parsed.data[1] - 2.5).abs() < 1e-12);
        assert!((parsed.data[2] - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn joint_state_zero() {
        let js = JointState::<6>::zero(999);
        assert_eq!(js.stamp_ns, 999);
        assert_eq!(js.position, [0.0; 6]);
    }

    #[test]
    fn joint_positions_bridge() {
        let mut js = JointState::<3>::zero(100);
        js.position = [0.1, 0.2, 0.3];
        let arr = joint_positions_to_array(&js);
        assert_eq!(arr.data, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn deserialize_too_short_returns_none() {
        let buf = [0u8; 4];
        assert!(Float64Array::<3>::deserialize(&buf).is_none());
    }
}
