//! ROS2-compatible IMU publisher example using oxictl's DDS user API.
//!
//! Demonstrates `Publisher<Imu>` and `Subscription<Imu>` with
//! `sensor_msgs::msg::Imu`-compatible CDR encoding on topic `rt/imu/data`.
//!
//! A synthetic IMU stream is generated: the vehicle rotates slowly around the
//! Z-axis at 0.1 rad/s.  Ten messages are published at a simulated 100 Hz
//! (no real-time sleep — the loop runs as fast as the transport allows).
//! The received orientation quaternion's W component is printed for each sample.
//!
//! Usage: cargo run --example ros2_imu_publisher --features dds-api

use std::thread;
use std::time::Duration;

use oxictl::protocol::dds::api::participant::Participant;
use oxictl::protocol::dds::discovery::qos_profile::QosProfile;
use oxictl::protocol::dds::ros2::msgs::builtin_interfaces::Time;
use oxictl::protocol::dds::ros2::msgs::geometry_msgs::{Quaternion, Vector3};
use oxictl::protocol::dds::ros2::msgs::sensor_msgs::Imu;
use oxictl::protocol::dds::ros2::msgs::std_msgs::Header;
use oxictl::protocol::dds::types::guid::GuidPrefix;

// ─── Synthetic IMU generation ────────────────────────────────────────────────

/// Build a synthetic `Imu` message at simulated time `t` (seconds).
///
/// Simulates a slow rotation about the Z-axis: ω_z = 0.1 rad/s, which
/// produces the quaternion q = (0, 0, sin(θ/2), cos(θ/2)) for θ = ω_z * t.
fn make_imu(t: f64) -> Imu {
    let theta = 0.1 * t; // cumulative yaw angle (rad)
    let half = theta / 2.0;

    let mut frame_id = heapless::String::<256>::new();
    frame_id
        .push_str("imu_link")
        .expect("frame_id literal fits in 256 bytes");

    Imu {
        header: Header {
            stamp: Time {
                sec: t as i32,
                nanosec: ((t.fract() * 1_000_000_000.0) as u32),
            },
            frame_id,
        },
        orientation: Quaternion {
            x: 0.0,
            y: 0.0,
            z: half.sin(),
            w: half.cos(),
        },
        // Row-0 all -1 signals "orientation covariance unknown".
        orientation_covariance: [-1.0; 9],
        angular_velocity: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.1,
        },
        angular_velocity_covariance: [0.01; 9],
        linear_acceleration: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 9.81,
        },
        linear_acceleration_covariance: [0.001; 9],
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qos = QosProfile::ros2_default();
    let topic = "rt/imu/data";

    // Publisher participant.
    let prefix_pub = GuidPrefix([
        0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    ]);
    let mut p_pub =
        Participant::new(0, prefix_pub, qos).expect("failed to create publisher participant");

    // Subscriber participant.
    let prefix_sub = GuidPrefix([
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    ]);
    let mut p_sub =
        Participant::new(0, prefix_sub, qos).expect("failed to create subscriber participant");

    // Explicit peer registration for in-process loopback.
    let addr_pub = p_pub
        .local_metatraffic_addr()
        .expect("publisher metatraffic address unavailable");
    let addr_sub = p_sub
        .local_metatraffic_addr()
        .expect("subscriber metatraffic address unavailable");

    p_pub.add_peer(addr_sub).expect("p_pub add_peer failed");
    p_sub.add_peer(addr_pub).expect("p_sub add_peer failed");

    let pub_ = p_pub
        .create_publisher::<Imu>(topic, &qos)
        .expect("failed to create Imu publisher");
    let sub_ = p_sub
        .create_subscription::<Imu>(topic, &qos)
        .expect("failed to create Imu subscription");

    // Discovery rounds before publishing.
    for _ in 0..10 {
        let _ = p_pub.spin_once();
        let _ = p_sub.spin_once();
        thread::sleep(Duration::from_millis(5));
    }

    println!("Publishing 10 synthetic IMU messages at simulated 100 Hz:");
    println!("{:>5}  {:>12}  {:>10}", "seq", "t (s)", "quat.w");

    // Simulated 10 ms timestep (100 Hz).
    let dt = 0.01_f64;

    for seq in 0..10u32 {
        let t = f64::from(seq) * dt;
        let imu_msg = make_imu(t);

        p_pub.publish(&pub_, &imu_msg).expect("IMU publish failed");

        // Drive both ends to deliver the sample.
        for _ in 0..5 {
            let _ = p_pub.spin_once();
            let _ = p_sub.spin_once();
            thread::sleep(Duration::from_millis(5));
        }

        let samples = p_sub.take(&sub_);
        if samples.is_empty() {
            println!("{seq:>5}  {t:>12.4}  (pending)");
        } else {
            for sample in &samples {
                println!("{seq:>5}  {t:>12.4}  {:>10.6}", sample.data.orientation.w);
            }
        }
    }

    Ok(())
}
