//! ROS2-compatible chatter example using oxictl's DDS user API.
//!
//! Demonstrates `Publisher<StdString>` and `Subscription<StdString>` with
//! `std_msgs::msg::String`-compatible CDR encoding on topic `rt/chatter`.
//!
//! Two participants are created in the same process.  Explicit `add_peer` is
//! used for reliable in-process loopback across CI platforms (no multicast
//! dependency).  After discovery, the publisher sends "hello world #N" for
//! N in 0..5 and the subscriber prints each received message.
//!
//! Usage: cargo run --example ros2_chatter --features dds-api

use std::thread;
use std::time::Duration;

use oxictl::protocol::dds::api::participant::Participant;
use oxictl::protocol::dds::discovery::qos_profile::QosProfile;
use oxictl::protocol::dds::ros2::msgs::std_msgs::StdString;
use oxictl::protocol::dds::types::guid::GuidPrefix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qos = QosProfile::ros2_default();

    // Participant 1 — publisher side.
    let prefix1 = GuidPrefix([
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    ]);
    let mut p1 = Participant::new(0, prefix1, qos).expect("failed to create publisher participant");

    // Participant 2 — subscriber side.
    let prefix2 = GuidPrefix([
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    ]);
    let mut p2 =
        Participant::new(0, prefix2, qos).expect("failed to create subscriber participant");

    // Retrieve metatraffic addresses for explicit peer registration.
    let addr1 = p1
        .local_metatraffic_addr()
        .expect("p1 metatraffic address unavailable");
    let addr2 = p2
        .local_metatraffic_addr()
        .expect("p2 metatraffic address unavailable");

    p1.add_peer(addr2).expect("p1 add_peer(addr2) failed");
    p2.add_peer(addr1).expect("p2 add_peer(addr1) failed");

    // Create endpoints on both participants.
    let topic = "rt/chatter";
    let pub_ = p1
        .create_publisher::<StdString>(topic, &qos)
        .expect("failed to create publisher");
    let sub_ = p2
        .create_subscription::<StdString>(topic, &qos)
        .expect("failed to create subscription");

    // Drive several discovery rounds so SEDP can announce and match endpoints.
    for _ in 0..10 {
        let _ = p1.spin_once();
        let _ = p2.spin_once();
        thread::sleep(Duration::from_millis(5));
    }

    // Publish + receive loop.
    for i in 0..5u32 {
        // Build the message text.
        let text = format!("hello world #{i}");
        let mut data = heapless::String::<256>::new();
        data.push_str(&text)
            .expect("message text fits in 256 bytes");
        let msg = StdString { data };

        p1.publish(&pub_, &msg).expect("publish failed");

        // Give UDP loopback time to deliver and drive both ends.
        for _ in 0..5 {
            let _ = p1.spin_once();
            let _ = p2.spin_once();
            thread::sleep(Duration::from_millis(5));
        }

        let samples = p2.take(&sub_);
        if samples.is_empty() {
            println!("[chatter] iteration {i}: no samples yet (in-flight)");
        } else {
            for sample in &samples {
                println!("[subscriber] received: {}", sample.data.data);
            }
        }
    }

    Ok(())
}
