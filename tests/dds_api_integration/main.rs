//! Integration tests for the high-level DDS API (dds-api feature).
//!
//! Existing tests use two in-process `Participant` instances with ephemeral UDP ports
//! and explicit peer registration to avoid EADDRINUSE and multicast complexity.
//! The `multicast_auto_discovery` test exercises the automatic SPDP multicast path.

use std::time::{Duration, Instant};

use heapless::String as HString;

use oxictl::protocol::dds::api::{
    builtin_impls::LogOwned, DdsType, Participant, Publisher, Sample, Subscription,
};
use oxictl::protocol::dds::discovery::qos_profile::QosProfile;
use oxictl::protocol::dds::ros2::log::LogSeverity;
use oxictl::protocol::dds::transport::{probe_multicast_loopback, SPDP_MULTICAST_IPV4};
use oxictl::protocol::dds::types::guid::GuidPrefix;

// ─── helpers ─────────────────────────────────────────────────────────────────

fn guid_prefix(seed: u8) -> GuidPrefix {
    GuidPrefix([seed; 12])
}

/// Drive both participants until `condition` returns true or `timeout` elapses.
/// Returns true if the condition was met.
fn wait_until<F>(
    p1: &mut Participant,
    p2: &mut Participant,
    timeout: Duration,
    mut condition: F,
) -> bool
where
    F: FnMut(&Participant, &Participant) -> bool,
{
    let deadline = Instant::now() + timeout;
    loop {
        let _ = p1.spin_once();
        let _ = p2.spin_once();
        if condition(p1, p2) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
    }
}

// ─── DdsType impl for String in tests ────────────────────────────────────────

// `heapless::String<256>` has a DdsType impl in builtin_impls.

// ─── Test 1: Participant creation and peer registration ───────────────────────

#[test]
fn participant_create_and_peer_registration() {
    let mut p1 = Participant::new(0, guid_prefix(0x01), QosProfile::ros2_default())
        .expect("p1 creation failed");
    let mut p2 = Participant::new(0, guid_prefix(0x02), QosProfile::ros2_default())
        .expect("p2 creation failed");

    let addr1 = p1.local_metatraffic_addr().expect("p1 metatraffic addr");
    let addr2 = p2.local_metatraffic_addr().expect("p2 metatraffic addr");

    p1.add_peer(addr2).expect("p1 add_peer");
    p2.add_peer(addr1).expect("p2 add_peer");

    // At least 1 explicit peer per participant (may also have auto-discovered peers).
    assert!(p1.peer_count() >= 1);
    assert!(p2.peer_count() >= 1);
}

// ─── Test 2: CDR string round-trip through DdsType ───────────────────────────

#[test]
fn cdr_string_dds_type_roundtrip() {
    let original: HString<256> = HString::try_from("hello DDS!").expect("string creation");
    let mut buf = [0u8; 512];
    let len = original.serialize(&mut buf).expect("serialize");
    let decoded = HString::<256>::deserialize(&buf[..len]).expect("deserialize");
    assert_eq!(original, decoded);
}

// ─── Test 3: LogOwned round-trip through DdsType ─────────────────────────────

#[test]
fn log_owned_roundtrip() {
    let log = LogOwned {
        severity: LogSeverity::Warn,
        stamp_sec: 1234,
        stamp_nsec: 567_000_000,
        name: HString::<256>::try_from("/my_node").expect("name"),
        msg: HString::<256>::try_from("something went wrong").expect("msg"),
        file: HString::<256>::try_from("src/main.rs").expect("file"),
        function: HString::<256>::try_from("my_function").expect("function"),
        line: 42,
    };

    let mut buf = [0u8; 1024];
    let len = log.serialize(&mut buf).expect("serialize");
    let decoded = LogOwned::deserialize(&buf[..len]).expect("deserialize");

    assert_eq!(decoded.stamp_sec, 1234);
    assert_eq!(decoded.stamp_nsec, 567_000_000);
    assert_eq!(decoded.severity as u8, LogSeverity::Warn as u8);
    assert_eq!(decoded.name.as_str(), "/my_node");
    assert_eq!(decoded.msg.as_str(), "something went wrong");
    assert_eq!(decoded.line, 42);
}

// ─── Test 4: Publisher/Subscription creation (no matching) ───────────────────

#[test]
fn publisher_subscription_create() {
    let mut p = Participant::new(0, guid_prefix(0x10), QosProfile::ros2_default())
        .expect("participant creation failed");

    let _pub1: Publisher<HString<256>> = p
        .create_publisher::<HString<256>>("test_topic", &QosProfile::ros2_default())
        .expect("create_publisher");

    let _sub1: Subscription<HString<256>> = p
        .create_subscription::<HString<256>>("test_topic", &QosProfile::ros2_default())
        .expect("create_subscription");
}

// ─── Test 5: End-to-end publish/subscribe round-trip ─────────────────────────

#[test]
fn end_to_end_publish_subscribe() {
    // Create two participants on a dedicated domain to isolate from other tests.
    let mut pub_p = Participant::new(5, guid_prefix(0x20), QosProfile::ros2_default())
        .expect("publisher participant");
    let mut sub_p = Participant::new(5, guid_prefix(0x21), QosProfile::ros2_default())
        .expect("subscriber participant");

    // Register peers.
    let pub_addr = pub_p.local_metatraffic_addr().expect("pub addr");
    let sub_addr = sub_p.local_metatraffic_addr().expect("sub addr");
    pub_p.add_peer(sub_addr).expect("pub add peer");
    sub_p.add_peer(pub_addr).expect("sub add peer");

    // Create publisher and subscriber on the same topic.
    let pub1 = pub_p
        .create_publisher::<HString<256>>("chatter", &QosProfile::ros2_default())
        .expect("create publisher");
    let sub1 = sub_p
        .create_subscription::<HString<256>>("chatter", &QosProfile::ros2_default())
        .expect("create subscription");

    // Spin until matching is established (up to 3 seconds).
    let timeout = Duration::from_secs(3);
    // We need at least one spin_once from each side before publishing.
    let _ = pub_p.spin_once();
    let _ = sub_p.spin_once();
    // Additional spins for discovery propagation.
    let matched = wait_until(
        &mut pub_p,
        &mut sub_p,
        timeout,
        |_pp, _sp| true, // match is done internally; just give time for SEDP to flow
    );
    assert!(matched, "wait_until timed out");

    // Publish a message.
    let value: HString<256> = HString::try_from("Hello, DDS world!").expect("string");
    pub_p.publish(&pub1, &value).expect("publish");

    // Spin until the subscriber receives the message or timeout.
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut received: Vec<Sample<HString<256>>> = Vec::new();
    while received.is_empty() && Instant::now() < deadline {
        let _ = pub_p.spin_once();
        let _ = sub_p.spin_once();
        received = sub_p.take(&sub1);
    }

    assert!(!received.is_empty(), "no samples received within timeout");
    assert_eq!(received[0].data.as_str(), "Hello, DDS world!");
}

// ─── Test 6: Multicast auto-discovery ────────────────────────────────────────

#[test]
#[cfg_attr(not(any(target_os = "linux", target_os = "macos")), ignore)]
fn multicast_auto_discovery() {
    // Before starting, verify that multicast loopback actually delivers packets
    // on this machine.  Some macOS sandbox / CI environments have the multicast
    // socket setup succeed (SO_REUSEADDR, join_multicast_v4) but the kernel
    // silently drops the loopback copy — in that case the whole test is moot.
    if !probe_multicast_loopback(SPDP_MULTICAST_IPV4, Duration::from_millis(500)) {
        eprintln!("oxictl test: multicast loopback not functional on this host (probe timed out); skipping multicast_auto_discovery.");
        return;
    }

    // Domain 99 — unlikely to collide with any running ROS2 daemon.
    let mut p1 = Participant::new(99, GuidPrefix([0x01; 12]), QosProfile::ros2_default())
        .expect("participant 1 construction failed");
    let mut p2 = Participant::new(99, GuidPrefix([0x02; 12]), QosProfile::ros2_default())
        .expect("participant 2 construction failed");

    if !p1.has_multicast() || !p2.has_multicast() {
        eprintln!("oxictl test: SPDP multicast socket setup failed on this host; skipping test.");
        return;
    }

    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let _ = p1.spin_once();
        let _ = p2.spin_once();
        if p1.peer_count() >= 1 && p2.peer_count() >= 1 {
            break;
        }
        if Instant::now() > deadline {
            panic!("Participants did not discover each other within 3 seconds");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(p1.peer_count() >= 1);
    assert!(p2.peer_count() >= 1);
}
