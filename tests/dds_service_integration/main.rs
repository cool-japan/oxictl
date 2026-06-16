//! Integration tests for the DDS service layer (`dds-api` feature).

use std::time::{Duration, Instant};

use oxictl::protocol::dds::api::{create_client, create_server, Participant};
use oxictl::protocol::dds::discovery::qos_profile::QosProfile;
use oxictl::protocol::dds::ros2::msgs::example_interfaces::{
    AddTwoInts, AddTwoIntsRequest, AddTwoIntsResponse,
};
use oxictl::protocol::dds::types::guid::GuidPrefix;

// ─── helpers ─────────────────────────────────────────────────────────────────

fn guid_prefix(seed: u8) -> GuidPrefix {
    GuidPrefix([seed; 12])
}

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

// ─── Test 1: Basic add_two_ints request/reply ─────────────────────────────────

#[test]
fn add_two_ints_request_reply() {
    let mut server_p = Participant::new(10, guid_prefix(0x10), QosProfile::ros2_default())
        .expect("server participant");
    let mut client_p = Participant::new(10, guid_prefix(0x11), QosProfile::ros2_default())
        .expect("client participant");

    let server_addr = server_p.local_metatraffic_addr().expect("server addr");
    let client_addr = client_p.local_metatraffic_addr().expect("client addr");
    server_p.add_peer(client_addr).expect("server add peer");
    client_p.add_peer(server_addr).expect("client add peer");

    let mut server =
        create_server::<AddTwoInts>(&mut server_p, "add_two_ints", &QosProfile::ros2_default())
            .expect("create server");
    let mut client =
        create_client::<AddTwoInts>(&mut client_p, "add_two_ints", &QosProfile::ros2_default())
            .expect("create client");

    // Allow discovery to propagate.
    let _ = wait_until(
        &mut server_p,
        &mut client_p,
        Duration::from_secs(3),
        |_, _| false,
    );

    // Send request.
    client
        .send_request(&mut client_p, &AddTwoIntsRequest { a: 3, b: 4 })
        .expect("send_request");

    // Spin until response is received or timeout.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut responses: Vec<(i64, AddTwoIntsResponse)> = Vec::new();
    while responses.is_empty() && Instant::now() < deadline {
        let _ = server_p.spin_once();
        let _ = client_p.spin_once();
        server
            .process(&mut server_p, |req| AddTwoIntsResponse {
                sum: req.a + req.b,
            })
            .expect("process");
        let got = client.take_responses(&mut client_p);
        responses.extend(got);
    }

    assert!(!responses.is_empty(), "no response received within timeout");
    assert_eq!(responses[0].1.sum, 7, "expected sum 7 for 3+4");
}

// ─── Test 2: Two clients, no cross-talk ──────────────────────────────────────

#[test]
fn two_clients_no_cross_talk() {
    let mut server_p = Participant::new(11, guid_prefix(0x20), QosProfile::ros2_default())
        .expect("server participant");
    let mut client1_p = Participant::new(11, guid_prefix(0x21), QosProfile::ros2_default())
        .expect("client1 participant");
    let mut client2_p = Participant::new(11, guid_prefix(0x22), QosProfile::ros2_default())
        .expect("client2 participant");

    let server_addr = server_p.local_metatraffic_addr().expect("server addr");
    let client1_addr = client1_p.local_metatraffic_addr().expect("client1 addr");
    let client2_addr = client2_p.local_metatraffic_addr().expect("client2 addr");

    server_p.add_peer(client1_addr).expect("server add client1");
    server_p.add_peer(client2_addr).expect("server add client2");
    client1_p.add_peer(server_addr).expect("client1 add server");
    client2_p.add_peer(server_addr).expect("client2 add server");

    let mut server =
        create_server::<AddTwoInts>(&mut server_p, "add_two_ints", &QosProfile::ros2_default())
            .expect("create server");
    let mut client1 =
        create_client::<AddTwoInts>(&mut client1_p, "add_two_ints", &QosProfile::ros2_default())
            .expect("create client1");
    let mut client2 =
        create_client::<AddTwoInts>(&mut client2_p, "add_two_ints", &QosProfile::ros2_default())
            .expect("create client2");

    // Discovery.
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        let _ = server_p.spin_once();
        let _ = client1_p.spin_once();
        let _ = client2_p.spin_once();
    }

    client1
        .send_request(&mut client1_p, &AddTwoIntsRequest { a: 1, b: 1 })
        .expect("client1 send");
    client2
        .send_request(&mut client2_p, &AddTwoIntsRequest { a: 100, b: 200 })
        .expect("client2 send");

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut resp1: Vec<(i64, AddTwoIntsResponse)> = Vec::new();
    let mut resp2: Vec<(i64, AddTwoIntsResponse)> = Vec::new();
    while (resp1.is_empty() || resp2.is_empty()) && Instant::now() < deadline {
        let _ = server_p.spin_once();
        let _ = client1_p.spin_once();
        let _ = client2_p.spin_once();
        server
            .process(&mut server_p, |req| AddTwoIntsResponse {
                sum: req.a + req.b,
            })
            .expect("process");
        let got1 = client1.take_responses(&mut client1_p);
        let got2 = client2.take_responses(&mut client2_p);
        resp1.extend(got1);
        resp2.extend(got2);
    }

    assert!(!resp1.is_empty(), "client1 got no response");
    assert!(!resp2.is_empty(), "client2 got no response");
    assert_eq!(resp1[0].1.sum, 2, "client1: expected 1+1=2");
    assert_eq!(resp2[0].1.sum, 300, "client2: expected 100+200=300");
}

// ─── Test 3: Server handles multiple sequential requests ──────────────────────

#[test]
fn server_handles_multiple_sequential() {
    let mut server_p = Participant::new(12, guid_prefix(0x30), QosProfile::ros2_default())
        .expect("server participant");
    let mut client_p = Participant::new(12, guid_prefix(0x31), QosProfile::ros2_default())
        .expect("client participant");

    let server_addr = server_p.local_metatraffic_addr().expect("server addr");
    let client_addr = client_p.local_metatraffic_addr().expect("client addr");
    server_p.add_peer(client_addr).expect("server add peer");
    client_p.add_peer(server_addr).expect("client add peer");

    let mut server =
        create_server::<AddTwoInts>(&mut server_p, "add_two_ints", &QosProfile::ros2_default())
            .expect("create server");
    let mut client =
        create_client::<AddTwoInts>(&mut client_p, "add_two_ints", &QosProfile::ros2_default())
            .expect("create client");

    // Discovery.
    let _ = wait_until(
        &mut server_p,
        &mut client_p,
        Duration::from_secs(3),
        |_, _| false,
    );

    // Send 3 requests.
    client
        .send_request(&mut client_p, &AddTwoIntsRequest { a: 1, b: 2 })
        .expect("send req 1");
    client
        .send_request(&mut client_p, &AddTwoIntsRequest { a: 3, b: 4 })
        .expect("send req 2");
    client
        .send_request(&mut client_p, &AddTwoIntsRequest { a: 5, b: 6 })
        .expect("send req 3");

    // Spin until 3 responses received or timeout.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut responses: Vec<(i64, AddTwoIntsResponse)> = Vec::new();
    while responses.len() < 3 && Instant::now() < deadline {
        let _ = server_p.spin_once();
        let _ = client_p.spin_once();
        server
            .process(&mut server_p, |req| AddTwoIntsResponse {
                sum: req.a + req.b,
            })
            .expect("process");
        let got = client.take_responses(&mut client_p);
        responses.extend(got);
    }

    assert_eq!(
        responses.len(),
        3,
        "expected 3 responses, got {}",
        responses.len()
    );
    let mut sums: Vec<i64> = responses.iter().map(|(_, r)| r.sum).collect();
    sums.sort_unstable();
    assert_eq!(sums, vec![3, 7, 11], "expected sums 3, 7, 11");
}

// ─── Test 4: Unmatched service — no reply without server ─────────────────────

#[test]
fn unmatched_service_no_reply() {
    let mut client_p = Participant::new(13, guid_prefix(0x40), QosProfile::ros2_default())
        .expect("client participant");

    let mut client =
        create_client::<AddTwoInts>(&mut client_p, "add_two_ints", &QosProfile::ros2_default())
            .expect("create client");

    client
        .send_request(&mut client_p, &AddTwoIntsRequest { a: 1, b: 1 })
        .expect("send request");

    // Spin 20 times — no server to reply.
    for _ in 0..20 {
        let _ = client_p.spin_once();
    }

    let responses = client.take_responses(&mut client_p);
    assert!(
        responses.is_empty(),
        "expected no responses without a server, got {}",
        responses.len()
    );
}
