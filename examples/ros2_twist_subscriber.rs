//! ROS2-compatible Twist subscriber driving a simulated unicycle.
//!
//! Demonstrates `Publisher<Twist>` and `Subscription<Twist>` using
//! `geometry_msgs::msg::Twist`-compatible CDR encoding on topic `rt/cmd_vel`.
//!
//! A publisher sends a constant-curvature command (v = 1.0 m/s, ω = 0.2 rad/s)
//! simulating a circular trajectory.  A subscriber receives the commands and
//! integrates a unicycle model (Euler method, dt = 0.1 s) to update the pose.
//!
//! The simulation runs for 10 iterations.  Expected trajectory: a circle of
//! radius v/ω = 5 m, completing roughly 2 rad of arc over 10 steps × 0.1 s.
//!
//! Usage: cargo run --example ros2_twist_subscriber --features dds-api

use std::thread;
use std::time::Duration;

use oxictl::protocol::dds::api::participant::Participant;
use oxictl::protocol::dds::discovery::qos_profile::QosProfile;
use oxictl::protocol::dds::ros2::msgs::geometry_msgs::{Twist, Vector3};
use oxictl::protocol::dds::types::guid::GuidPrefix;

// ─── Unicycle pose ────────────────────────────────────────────────────────────

/// Planar unicycle state (x, y, heading θ in radians).
#[derive(Debug, Clone)]
struct Pose {
    x: f64,
    y: f64,
    theta: f64,
}

impl Pose {
    fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
        }
    }

    /// Euler-integrate a `Twist` over time step `dt` (seconds).
    ///
    /// Unicycle kinematics:
    ///   ẋ = v·cos(θ),  ẏ = v·sin(θ),  θ̇ = ω
    fn integrate(&mut self, twist: &Twist, dt: f64) {
        self.x += twist.linear.x * self.theta.cos() * dt;
        self.y += twist.linear.x * self.theta.sin() * dt;
        self.theta += twist.angular.z * dt;
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qos = QosProfile::ros2_default();
    let topic = "rt/cmd_vel";

    // Publisher participant — velocity command source.
    let prefix_cmd = GuidPrefix([
        0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    ]);
    let mut p_cmd = Participant::new(0, prefix_cmd, qos)
        .expect("failed to create cmd_vel publisher participant");

    // Subscriber participant — unicycle controller.
    let prefix_ctl = GuidPrefix([
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
    ]);
    let mut p_ctl =
        Participant::new(0, prefix_ctl, qos).expect("failed to create controller participant");

    // Explicit peer registration.
    let addr_cmd = p_cmd
        .local_metatraffic_addr()
        .expect("cmd participant metatraffic address unavailable");
    let addr_ctl = p_ctl
        .local_metatraffic_addr()
        .expect("controller participant metatraffic address unavailable");

    p_cmd.add_peer(addr_ctl).expect("p_cmd add_peer failed");
    p_ctl.add_peer(addr_cmd).expect("p_ctl add_peer failed");

    let pub_ = p_cmd
        .create_publisher::<Twist>(topic, &qos)
        .expect("failed to create Twist publisher");
    let sub_ = p_ctl
        .create_subscription::<Twist>(topic, &qos)
        .expect("failed to create Twist subscription");

    // Discovery rounds before the control loop.
    for _ in 0..10 {
        let _ = p_cmd.spin_once();
        let _ = p_ctl.spin_once();
        thread::sleep(Duration::from_millis(5));
    }

    // Constant-curvature command: v = 1.0 m/s, ω = 0.2 rad/s → r = 5 m circle.
    let cmd_twist = Twist {
        linear: Vector3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        angular: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.2,
        },
    };

    let dt = 0.1_f64; // 100 ms per step
    let mut pose = Pose::new();

    println!("Unicycle integration (v=1.0 m/s, ω=0.2 rad/s, dt=0.1 s):");
    println!(
        "{:>5}  {:>10}  {:>10}  {:>10}",
        "step", "x (m)", "y (m)", "θ (rad)"
    );

    for step in 0..10u32 {
        p_cmd
            .publish(&pub_, &cmd_twist)
            .expect("Twist publish failed");

        // Drive both participants to deliver the sample.
        for _ in 0..5 {
            let _ = p_cmd.spin_once();
            let _ = p_ctl.spin_once();
            thread::sleep(Duration::from_millis(5));
        }

        let samples = p_ctl.take(&sub_);
        if samples.is_empty() {
            // No sample arrived yet; integrate the known command directly so
            // the example always produces meaningful output even in restricted
            // network environments.
            pose.integrate(&cmd_twist, dt);
            println!(
                "{step:>5}  {:>10.4}  {:>10.4}  {:>10.4}  (local fallback)",
                pose.x, pose.y, pose.theta
            );
        } else {
            for sample in &samples {
                pose.integrate(&sample.data, dt);
            }
            println!(
                "{step:>5}  {:>10.4}  {:>10.4}  {:>10.4}",
                pose.x, pose.y, pose.theta
            );
        }
    }

    let arc_len = 1.0_f64 * dt * 10.0;
    println!("\nExpected arc length ≈ {arc_len:.2} m over 10 steps × 0.1 s");

    Ok(())
}
