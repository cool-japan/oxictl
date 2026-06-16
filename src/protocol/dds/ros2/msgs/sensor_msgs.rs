//! `sensor_msgs` ROS2 message types.
//!
//! Provides `Imu`, `JointState`, `Range`, `Temperature`, `MagneticField`,
//! `FluidPressure`, `RelativeHumidity`, `NavSatStatus`, `NavSatFix`,
//! `BatteryState` with CDR serialization.

use heapless::{String as HString, Vec as HVec};

use crate::protocol::dds::api::dds_type::DdsType;
use crate::protocol::dds::api::error::DdsApiError;
use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};

use super::geometry_msgs::{Quaternion, Vector3};
use super::std_msgs::Header;
use super::{make_cursor, make_writer};

// ─── Imu ─────────────────────────────────────────────────────────────────────

/// `sensor_msgs/msg/Imu` — IMU measurement with orientation, angular velocity,
/// and linear acceleration, each with a 3×3 covariance matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Imu {
    /// Message header.
    pub header: Header,
    /// Orientation estimate as a unit quaternion.
    pub orientation: Quaternion,
    /// Row-major 3×3 orientation covariance (`-1` row-0 = unknown).
    pub orientation_covariance: [f64; 9],
    /// Angular velocity vector.
    pub angular_velocity: Vector3,
    /// Row-major 3×3 angular velocity covariance.
    pub angular_velocity_covariance: [f64; 9],
    /// Linear acceleration vector.
    pub linear_acceleration: Vector3,
    /// Row-major 3×3 linear acceleration covariance.
    pub linear_acceleration_covariance: [f64; 9],
}

impl Default for Imu {
    fn default() -> Self {
        Self {
            header: Header::default(),
            orientation: Quaternion::default(),
            orientation_covariance: [0.0; 9],
            angular_velocity: Vector3::default(),
            angular_velocity_covariance: [0.0; 9],
            linear_acceleration: Vector3::default(),
            linear_acceleration_covariance: [0.0; 9],
        }
    }
}

/// Serialize a fixed-length `[f64; 9]` array (no length prefix).
fn write_f64_array9(w: &mut ByteWriter<'_>, arr: &[f64; 9]) -> Result<(), DdsApiError> {
    w.align_to(8)?;
    for &v in arr.iter() {
        w.write_f64(v)?;
    }
    Ok(())
}

/// Deserialize a fixed-length `[f64; 9]` array (no length prefix).
fn read_f64_array9(r: &mut ByteCursor<'_>) -> Result<[f64; 9], DdsApiError> {
    r.align_to(8)?;
    let mut arr = [0.0f64; 9];
    for elem in arr.iter_mut() {
        *elem = r.read_f64()?;
    }
    Ok(arr)
}

impl DdsType for Imu {
    const TYPE_NAME: &'static str = "sensor_msgs::msg::dds_::Imu_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        self.orientation.serialize_inner(&mut w)?;
        write_f64_array9(&mut w, &self.orientation_covariance)?;
        self.angular_velocity.serialize_inner(&mut w)?;
        write_f64_array9(&mut w, &self.angular_velocity_covariance)?;
        self.linear_acceleration.serialize_inner(&mut w)?;
        write_f64_array9(&mut w, &self.linear_acceleration_covariance)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        let orientation = Quaternion::deserialize_inner(&mut r)?;
        let orientation_covariance = read_f64_array9(&mut r)?;
        let angular_velocity = Vector3::deserialize_inner(&mut r)?;
        let angular_velocity_covariance = read_f64_array9(&mut r)?;
        let linear_acceleration = Vector3::deserialize_inner(&mut r)?;
        let linear_acceleration_covariance = read_f64_array9(&mut r)?;
        Ok(Self {
            header,
            orientation,
            orientation_covariance,
            angular_velocity,
            angular_velocity_covariance,
            linear_acceleration,
            linear_acceleration_covariance,
        })
    }
}

// ─── JointState ──────────────────────────────────────────────────────────────

/// `sensor_msgs/msg/JointState` — robot joint positions, velocities, and efforts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JointState {
    /// Message header.
    pub header: Header,
    /// Joint names (up to 32 joints, each name up to 32 bytes).
    pub name: HVec<HString<32>, 32>,
    /// Joint positions in radians (or metres for prismatic).
    pub position: HVec<f64, 32>,
    /// Joint velocities.
    pub velocity: HVec<f64, 32>,
    /// Joint efforts (torques / forces).
    pub effort: HVec<f64, 32>,
}

/// Write a `heapless::Vec<f64, 32>` sequence: u32 length prefix + elements.
fn write_f64_seq(w: &mut ByteWriter<'_>, seq: &HVec<f64, 32>) -> Result<(), DdsApiError> {
    w.align_to(4)?;
    w.write_u32(seq.len() as u32)?;
    for &x in seq.iter() {
        w.align_to(8)?;
        w.write_f64(x)?;
    }
    Ok(())
}

/// Read a `heapless::Vec<f64, 32>` sequence: u32 length prefix + elements.
fn read_f64_seq(r: &mut ByteCursor<'_>) -> Result<HVec<f64, 32>, DdsApiError> {
    r.align_to(4)?;
    let n = r.read_u32()? as usize;
    let mut seq: HVec<f64, 32> = HVec::new();
    for _ in 0..n {
        r.align_to(8)?;
        let v = r.read_f64()?;
        seq.push(v)
            .map_err(|_| DdsApiError::Serialization("f64 sequence capacity exceeded (max 32)"))?;
    }
    Ok(seq)
}

/// Write a `heapless::Vec<HString<32>, 32>` sequence of CDR strings.
fn write_string_seq(
    w: &mut ByteWriter<'_>,
    seq: &HVec<HString<32>, 32>,
) -> Result<(), DdsApiError> {
    w.align_to(4)?;
    w.write_u32(seq.len() as u32)?;
    for s in seq.iter() {
        w.write_cdr_string(s.as_str())?;
    }
    Ok(())
}

/// Read a `heapless::Vec<HString<32>, 32>` sequence of CDR strings.
fn read_string_seq(r: &mut ByteCursor<'_>) -> Result<HVec<HString<32>, 32>, DdsApiError> {
    r.align_to(4)?;
    let n = r.read_u32()? as usize;
    let mut seq: HVec<HString<32>, 32> = HVec::new();
    for _ in 0..n {
        let raw = r.read_cdr_string()?;
        let mut s = HString::<32>::new();
        s.push_str(raw)
            .map_err(|_| DdsApiError::Serialization("joint name exceeds 32-byte capacity"))?;
        seq.push(s).map_err(|_| {
            DdsApiError::Serialization("joint name sequence capacity exceeded (max 32)")
        })?;
    }
    Ok(seq)
}

impl DdsType for JointState {
    const TYPE_NAME: &'static str = "sensor_msgs::msg::dds_::JointState_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        write_string_seq(&mut w, &self.name)?;
        write_f64_seq(&mut w, &self.position)?;
        write_f64_seq(&mut w, &self.velocity)?;
        write_f64_seq(&mut w, &self.effort)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        let name = read_string_seq(&mut r)?;
        let position = read_f64_seq(&mut r)?;
        let velocity = read_f64_seq(&mut r)?;
        let effort = read_f64_seq(&mut r)?;
        Ok(Self {
            header,
            name,
            position,
            velocity,
            effort,
        })
    }
}

// ─── Range ───────────────────────────────────────────────────────────────────

/// `sensor_msgs/msg/Range` — single range sensor measurement.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Range {
    /// Message header.
    pub header: Header,
    /// Radiation type: 0 = ultrasound, 1 = infrared.
    pub radiation_type: u8,
    /// Field of view in radians.
    pub field_of_view: f32,
    /// Minimum valid range in metres.
    pub min_range: f32,
    /// Maximum valid range in metres.
    pub max_range: f32,
    /// Measured range in metres.
    pub range: f32,
}

impl DdsType for Range {
    const TYPE_NAME: &'static str = "sensor_msgs::msg::dds_::Range_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        w.write_u8(self.radiation_type)?;
        // Align to 4 for f32 fields
        w.align_to(4)?;
        w.write_f32(self.field_of_view)?;
        w.write_f32(self.min_range)?;
        w.write_f32(self.max_range)?;
        w.write_f32(self.range)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        let radiation_type = r.read_u8()?;
        r.align_to(4)?;
        let field_of_view = r.read_f32()?;
        let min_range = r.read_f32()?;
        let max_range = r.read_f32()?;
        let range = r.read_f32()?;
        Ok(Self {
            header,
            radiation_type,
            field_of_view,
            min_range,
            max_range,
            range,
        })
    }
}

// ─── Temperature ─────────────────────────────────────────────────────────────

/// `sensor_msgs/msg/Temperature` — temperature measurement with variance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Temperature {
    /// Message header.
    pub header: Header,
    /// Temperature in degrees Celsius.
    pub temperature: f64,
    /// Variance of the measurement in °C² (0 = unknown).
    pub variance: f64,
}

impl DdsType for Temperature {
    const TYPE_NAME: &'static str = "sensor_msgs::msg::dds_::Temperature_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        w.align_to(8)?;
        w.write_f64(self.temperature)?;
        w.write_f64(self.variance)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        r.align_to(8)?;
        let temperature = r.read_f64()?;
        let variance = r.read_f64()?;
        Ok(Self {
            header,
            temperature,
            variance,
        })
    }
}

// ─── MagneticField ───────────────────────────────────────────────────────────

/// `sensor_msgs/msg/MagneticField` — magnetic field with 3×3 covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct MagneticField {
    /// Message header.
    pub header: Header,
    /// Magnetic field vector in Tesla.
    pub magnetic_field: Vector3,
    /// Row-major 3×3 covariance matrix.
    pub magnetic_field_covariance: [f64; 9],
}

impl Default for MagneticField {
    fn default() -> Self {
        Self {
            header: Header::default(),
            magnetic_field: Vector3::default(),
            magnetic_field_covariance: [0.0; 9],
        }
    }
}

impl DdsType for MagneticField {
    const TYPE_NAME: &'static str = "sensor_msgs::msg::dds_::MagneticField_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        self.magnetic_field.serialize_inner(&mut w)?;
        write_f64_array9(&mut w, &self.magnetic_field_covariance)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        let magnetic_field = Vector3::deserialize_inner(&mut r)?;
        let magnetic_field_covariance = read_f64_array9(&mut r)?;
        Ok(Self {
            header,
            magnetic_field,
            magnetic_field_covariance,
        })
    }
}

// ─── FluidPressure ───────────────────────────────────────────────────────────

/// `sensor_msgs/msg/FluidPressure` — fluid pressure measurement.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FluidPressure {
    /// Message header.
    pub header: Header,
    /// Absolute pressure in Pascals.
    pub fluid_pressure: f64,
    /// Pressure variance in Pa² (0 = unknown).
    pub variance: f64,
}

impl DdsType for FluidPressure {
    const TYPE_NAME: &'static str = "sensor_msgs::msg::dds_::FluidPressure_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        w.align_to(8)?;
        w.write_f64(self.fluid_pressure)?;
        w.write_f64(self.variance)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        r.align_to(8)?;
        let fluid_pressure = r.read_f64()?;
        let variance = r.read_f64()?;
        Ok(Self {
            header,
            fluid_pressure,
            variance,
        })
    }
}

// ─── RelativeHumidity ────────────────────────────────────────────────────────

/// `sensor_msgs/msg/RelativeHumidity` — relative humidity measurement.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RelativeHumidity {
    /// Message header.
    pub header: Header,
    /// Relative humidity [0, 1.0] (1.0 = 100%).
    pub relative_humidity: f64,
    /// Humidity variance (0 = unknown).
    pub variance: f64,
}

impl DdsType for RelativeHumidity {
    const TYPE_NAME: &'static str = "sensor_msgs::msg::dds_::RelativeHumidity_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        w.align_to(8)?;
        w.write_f64(self.relative_humidity)?;
        w.write_f64(self.variance)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        r.align_to(8)?;
        let relative_humidity = r.read_f64()?;
        let variance = r.read_f64()?;
        Ok(Self {
            header,
            relative_humidity,
            variance,
        })
    }
}

// ─── NavSatStatus ────────────────────────────────────────────────────────────

/// `sensor_msgs/msg/NavSatStatus` — GPS/GNSS fix status.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NavSatStatus {
    /// Fix status: −1 = no fix, 0 = fix, 1 = SBAS, 2 = GBAS.
    pub status: i8,
    /// Service mask: 1=GPS, 2=GLONASS, 4=COMPASS, 8=GALILEO.
    pub service: u16,
}

impl NavSatStatus {
    /// Serialize fields (without CDR header) into `w`.
    pub(crate) fn serialize_inner(&self, w: &mut ByteWriter<'_>) -> Result<(), DdsApiError> {
        w.write_u8(self.status as u8)?;
        w.align_to(2)?;
        w.write_u16(self.service)?;
        Ok(())
    }

    /// Deserialize fields (without CDR header) from `r`.
    pub(crate) fn deserialize_inner(r: &mut ByteCursor<'_>) -> Result<Self, DdsApiError> {
        let status = r.read_u8()? as i8;
        r.align_to(2)?;
        let service = r.read_u16()?;
        Ok(Self { status, service })
    }
}

impl DdsType for NavSatStatus {
    const TYPE_NAME: &'static str = "sensor_msgs::msg::dds_::NavSatStatus_";

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

// ─── NavSatFix ───────────────────────────────────────────────────────────────

/// `sensor_msgs/msg/NavSatFix` — GPS/GNSS fix with position covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct NavSatFix {
    /// Message header.
    pub header: Header,
    /// GPS status.
    pub status: NavSatStatus,
    /// Latitude in degrees (WGS84).
    pub latitude: f64,
    /// Longitude in degrees (WGS84).
    pub longitude: f64,
    /// Altitude in metres above WGS84 ellipsoid.
    pub altitude: f64,
    /// Row-major 3×3 ENU position covariance in m².
    pub position_covariance: [f64; 9],
    /// Covariance type: 0=unknown, 1=approximated, 2=diagonal_known, 3=known.
    pub position_covariance_type: u8,
}

impl Default for NavSatFix {
    fn default() -> Self {
        Self {
            header: Header::default(),
            status: NavSatStatus::default(),
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            position_covariance: [0.0; 9],
            position_covariance_type: 0,
        }
    }
}

impl DdsType for NavSatFix {
    const TYPE_NAME: &'static str = "sensor_msgs::msg::dds_::NavSatFix_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        self.status.serialize_inner(&mut w)?;
        w.align_to(8)?;
        w.write_f64(self.latitude)?;
        w.write_f64(self.longitude)?;
        w.write_f64(self.altitude)?;
        write_f64_array9(&mut w, &self.position_covariance)?;
        w.write_u8(self.position_covariance_type)?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        let status = NavSatStatus::deserialize_inner(&mut r)?;
        r.align_to(8)?;
        let latitude = r.read_f64()?;
        let longitude = r.read_f64()?;
        let altitude = r.read_f64()?;
        let position_covariance = read_f64_array9(&mut r)?;
        let position_covariance_type = r.read_u8()?;
        Ok(Self {
            header,
            status,
            latitude,
            longitude,
            altitude,
            position_covariance,
            position_covariance_type,
        })
    }
}

// ─── BatteryState ────────────────────────────────────────────────────────────

/// `sensor_msgs/msg/BatteryState` — battery status.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatteryState {
    /// Message header.
    pub header: Header,
    /// Voltage in Volts.
    pub voltage: f32,
    /// Temperature in °C (NaN if unknown).
    pub temperature: f32,
    /// Charge current in Amperes (negative when discharging).
    pub current: f32,
    /// Current charge in Ah (NaN if unknown).
    pub charge: f32,
    /// Design capacity in Ah.
    pub capacity: f32,
    /// Design (nominal) capacity in Ah.
    pub design_capacity: f32,
    /// State of charge [0.0, 1.0].
    pub percentage: f32,
    /// Power supply status (see ROS2 constants).
    pub power_supply_status: u8,
    /// Power supply health (see ROS2 constants).
    pub power_supply_health: u8,
    /// Power supply technology (see ROS2 constants).
    pub power_supply_technology: u8,
    /// Whether battery is present.
    pub present: bool,
    /// Per-cell voltage (up to 16 cells).
    pub cell_voltage: HVec<f32, 16>,
    /// Per-cell temperature (up to 16 cells).
    pub cell_temperature: HVec<f32, 16>,
    /// Physical location string.
    pub location: HString<32>,
    /// Battery serial number.
    pub serial_number: HString<32>,
}

/// Write a `heapless::Vec<f32, 16>` sequence: u32 length + f32 elements.
fn write_f32_seq16(w: &mut ByteWriter<'_>, seq: &HVec<f32, 16>) -> Result<(), DdsApiError> {
    w.align_to(4)?;
    w.write_u32(seq.len() as u32)?;
    for &x in seq.iter() {
        w.write_f32(x)?;
    }
    Ok(())
}

/// Read a `heapless::Vec<f32, 16>` sequence: u32 length + f32 elements.
fn read_f32_seq16(r: &mut ByteCursor<'_>) -> Result<HVec<f32, 16>, DdsApiError> {
    r.align_to(4)?;
    let n = r.read_u32()? as usize;
    let mut seq: HVec<f32, 16> = HVec::new();
    for _ in 0..n {
        let v = r.read_f32()?;
        seq.push(v)
            .map_err(|_| DdsApiError::Serialization("f32 sequence capacity exceeded (max 16)"))?;
    }
    Ok(seq)
}

impl DdsType for BatteryState {
    const TYPE_NAME: &'static str = "sensor_msgs::msg::dds_::BatteryState_";

    fn serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError> {
        let mut w = make_writer(buf)?;
        self.header.serialize_inner(&mut w)?;
        w.write_f32(self.voltage)?;
        w.write_f32(self.temperature)?;
        w.write_f32(self.current)?;
        w.write_f32(self.charge)?;
        w.write_f32(self.capacity)?;
        w.write_f32(self.design_capacity)?;
        w.write_f32(self.percentage)?;
        w.write_u8(self.power_supply_status)?;
        w.write_u8(self.power_supply_health)?;
        w.write_u8(self.power_supply_technology)?;
        w.write_u8(if self.present { 1 } else { 0 })?;
        write_f32_seq16(&mut w, &self.cell_voltage)?;
        write_f32_seq16(&mut w, &self.cell_temperature)?;
        w.write_cdr_string(self.location.as_str())?;
        w.write_cdr_string(self.serial_number.as_str())?;
        Ok(4 + w.position())
    }

    fn deserialize(payload: &[u8]) -> Result<Self, DdsApiError> {
        let mut r = make_cursor(payload)?;
        let header = Header::deserialize_inner(&mut r)?;
        let voltage = r.read_f32()?;
        let temperature = r.read_f32()?;
        let current = r.read_f32()?;
        let charge = r.read_f32()?;
        let capacity = r.read_f32()?;
        let design_capacity = r.read_f32()?;
        let percentage = r.read_f32()?;
        let power_supply_status = r.read_u8()?;
        let power_supply_health = r.read_u8()?;
        let power_supply_technology = r.read_u8()?;
        let present = r.read_u8()? != 0;
        let cell_voltage = read_f32_seq16(&mut r)?;
        let cell_temperature = read_f32_seq16(&mut r)?;
        let loc_s = r.read_cdr_string()?;
        let mut location = HString::<32>::new();
        location
            .push_str(loc_s)
            .map_err(|_| DdsApiError::Serialization("location exceeds 32-byte capacity"))?;
        let sn_s = r.read_cdr_string()?;
        let mut serial_number = HString::<32>::new();
        serial_number
            .push_str(sn_s)
            .map_err(|_| DdsApiError::Serialization("serial_number exceeds 32-byte capacity"))?;
        Ok(Self {
            header,
            voltage,
            temperature,
            current,
            charge,
            capacity,
            design_capacity,
            percentage,
            power_supply_status,
            power_supply_health,
            power_supply_technology,
            present,
            cell_voltage,
            cell_temperature,
            location,
            serial_number,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::ros2::msgs::builtin_interfaces::Time;

    fn make_header(frame: &str) -> Header {
        let mut frame_id = heapless::String::<256>::new();
        frame_id.push_str(frame).unwrap();
        Header {
            stamp: Time { sec: 1, nanosec: 0 },
            frame_id,
        }
    }

    #[test]
    fn imu_type_name() {
        assert_eq!(Imu::TYPE_NAME, "sensor_msgs::msg::dds_::Imu_");
    }

    #[test]
    fn imu_round_trip() {
        let original = Imu {
            header: make_header("imu_link"),
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            orientation_covariance: [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
            angular_velocity: Vector3 {
                x: 0.1,
                y: 0.0,
                z: 0.05,
            },
            angular_velocity_covariance: [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001],
            linear_acceleration: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 9.81,
            },
            linear_acceleration_covariance: [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
        };
        let mut buf = [0u8; 1024];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Imu::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn joint_state_type_name() {
        assert_eq!(JointState::TYPE_NAME, "sensor_msgs::msg::dds_::JointState_");
    }

    #[test]
    fn joint_state_round_trip() {
        let mut js = JointState {
            header: make_header("base_link"),
            ..Default::default()
        };
        let names = ["joint1", "joint2", "joint3"];
        for n in &names {
            let mut s = heapless::String::<32>::new();
            s.push_str(n).unwrap();
            js.name.push(s).unwrap();
        }
        js.position.push(0.1).unwrap();
        js.position.push(0.2).unwrap();
        js.position.push(0.3).unwrap();
        js.velocity.push(0.01).unwrap();
        js.velocity.push(0.02).unwrap();
        js.velocity.push(0.03).unwrap();
        js.effort.push(1.0).unwrap();
        js.effort.push(2.0).unwrap();
        js.effort.push(3.0).unwrap();

        let mut buf = [0u8; 1024];
        let written = js.serialize(&mut buf).unwrap();
        let decoded = JointState::deserialize(&buf[..written]).unwrap();
        assert_eq!(js, decoded);
    }

    #[test]
    fn joint_state_overflow() {
        // Push 33 joints to verify capacity exceeded error (max=32)
        let mut js = JointState::default();
        for i in 0..32u8 {
            let mut s = heapless::String::<32>::new();
            let _ = s.push(char::from(b'a' + i));
            js.name.push(s).unwrap();
        }
        // The 33rd push should fail at the heapless Vec level
        let mut s = heapless::String::<32>::new();
        s.push('z').unwrap();
        let result = js.name.push(s);
        assert!(
            result.is_err(),
            "expected overflow error at 33rd joint name"
        );
    }

    #[test]
    fn range_type_name() {
        assert_eq!(Range::TYPE_NAME, "sensor_msgs::msg::dds_::Range_");
    }

    #[test]
    fn range_round_trip() {
        let original = Range {
            header: make_header("ultrasound_link"),
            radiation_type: 0,
            field_of_view: 0.26_f32,
            min_range: 0.02_f32,
            max_range: 4.0_f32,
            range: 1.5_f32,
        };
        let mut buf = [0u8; 256];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Range::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn temperature_type_name() {
        assert_eq!(
            Temperature::TYPE_NAME,
            "sensor_msgs::msg::dds_::Temperature_"
        );
    }

    #[test]
    fn temperature_round_trip() {
        let original = Temperature {
            header: make_header("temp_sensor"),
            temperature: 25.5,
            variance: 0.01,
        };
        let mut buf = [0u8; 256];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = Temperature::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn magnetic_field_type_name() {
        assert_eq!(
            MagneticField::TYPE_NAME,
            "sensor_msgs::msg::dds_::MagneticField_"
        );
    }

    #[test]
    fn magnetic_field_round_trip() {
        let original = MagneticField {
            header: make_header("mag_link"),
            magnetic_field: Vector3 {
                x: 0.0001,
                y: 0.0002,
                z: 0.00005,
            },
            magnetic_field_covariance: [1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6],
        };
        let mut buf = [0u8; 512];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = MagneticField::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn fluid_pressure_type_name() {
        assert_eq!(
            FluidPressure::TYPE_NAME,
            "sensor_msgs::msg::dds_::FluidPressure_"
        );
    }

    #[test]
    fn fluid_pressure_round_trip() {
        let original = FluidPressure {
            header: make_header("baro_link"),
            fluid_pressure: 101_325.0,
            variance: 25.0,
        };
        let mut buf = [0u8; 256];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = FluidPressure::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn relative_humidity_type_name() {
        assert_eq!(
            RelativeHumidity::TYPE_NAME,
            "sensor_msgs::msg::dds_::RelativeHumidity_"
        );
    }

    #[test]
    fn relative_humidity_round_trip() {
        let original = RelativeHumidity {
            header: make_header("humidity_link"),
            relative_humidity: 0.65,
            variance: 0.0025,
        };
        let mut buf = [0u8; 256];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = RelativeHumidity::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn nav_sat_status_type_name() {
        assert_eq!(
            NavSatStatus::TYPE_NAME,
            "sensor_msgs::msg::dds_::NavSatStatus_"
        );
    }

    #[test]
    fn nav_sat_status_round_trip() {
        let original = NavSatStatus {
            status: 0,
            service: 1,
        };
        let mut buf = [0u8; 32];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = NavSatStatus::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn nav_sat_fix_type_name() {
        assert_eq!(NavSatFix::TYPE_NAME, "sensor_msgs::msg::dds_::NavSatFix_");
    }

    #[test]
    fn nav_sat_fix_round_trip() {
        let original = NavSatFix {
            header: make_header("gps_link"),
            status: NavSatStatus {
                status: 0,
                service: 1,
            },
            latitude: 35.6762,
            longitude: 139.6503,
            altitude: 40.0,
            position_covariance: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.0],
            position_covariance_type: 2,
        };
        let mut buf = [0u8; 512];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = NavSatFix::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn battery_state_type_name() {
        assert_eq!(
            BatteryState::TYPE_NAME,
            "sensor_msgs::msg::dds_::BatteryState_"
        );
    }

    #[test]
    fn battery_state_round_trip() {
        let mut cell_voltage: HVec<f32, 16> = HVec::new();
        cell_voltage.push(3.7).unwrap();
        cell_voltage.push(3.75).unwrap();
        let mut cell_temperature: HVec<f32, 16> = HVec::new();
        cell_temperature.push(25.0).unwrap();
        cell_temperature.push(25.5).unwrap();
        let mut location = HString::<32>::new();
        location.push_str("slot_A").unwrap();
        let mut serial_number = HString::<32>::new();
        serial_number.push_str("SN-1234567").unwrap();
        let original = BatteryState {
            header: make_header("battery_link"),
            voltage: 7.45,
            temperature: 25.2,
            current: -1.5,
            charge: 4.5,
            capacity: 5.0,
            design_capacity: 5.2,
            percentage: 0.87,
            power_supply_status: 2,
            power_supply_health: 1,
            power_supply_technology: 2,
            present: true,
            cell_voltage,
            cell_temperature,
            location,
            serial_number,
        };
        let mut buf = [0u8; 1024];
        let written = original.serialize(&mut buf).unwrap();
        let decoded = BatteryState::deserialize(&buf[..written]).unwrap();
        assert_eq!(original, decoded);
    }
}
