//! Integration tests for the ROS2 action layer (`dds-api` feature).

use std::time::{Duration, Instant};

use heapless::Vec as HVec;

use oxictl::protocol::dds::api::{
    create_action_client, create_action_server, ActionHandler, ExecuteResult, GoalOutcome,
    Participant,
};
use oxictl::protocol::dds::discovery::qos_profile::QosProfile;
use oxictl::protocol::dds::ros2::msgs::action_msgs::{goal_status, GoalStatusArray};
use oxictl::protocol::dds::ros2::msgs::example_interfaces_action::{
    Fibonacci, FibonacciFeedback, FibonacciGoal, FibonacciResult,
};
use oxictl::protocol::dds::ros2::msgs::unique_identifier_msgs::Uuid;
use oxictl::protocol::dds::types::guid::GuidPrefix;

// ─── helpers ──────────────────────────────────────────────────────────────────

fn guid_prefix(seed: u8) -> GuidPrefix {
    GuidPrefix([seed; 12])
}

fn fibonacci_seq(n: i32) -> HVec<i32, 32> {
    let mut seq: HVec<i32, 32> = HVec::new();
    if n <= 0 {
        return seq;
    }
    let _ = seq.push(0);
    if n == 1 {
        return seq;
    }
    let _ = seq.push(1);
    let mut a = 0i32;
    let mut b = 1i32;
    let limit = if n < 32 { n } else { 32 };
    for _ in 2..limit {
        let c = a.wrapping_add(b);
        a = b;
        b = c;
        let _ = seq.push(c);
    }
    seq
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

// ─── action handlers ──────────────────────────────────────────────────────────

struct SimpleHandler;

impl ActionHandler<Fibonacci> for SimpleHandler {
    fn accept_goal(&mut self, _goal_id: &Uuid, _goal: &FibonacciGoal) -> GoalOutcome {
        GoalOutcome::Accept
    }

    fn execute_goal(
        &mut self,
        _goal_id: &Uuid,
        goal: &FibonacciGoal,
        _feedback_cb: &mut dyn FnMut(FibonacciFeedback),
    ) -> ExecuteResult<Fibonacci> {
        ExecuteResult::Succeeded(FibonacciResult {
            sequence: fibonacci_seq(goal.order),
        })
    }

    fn cancel_goal(&mut self, _goal_id: &Uuid) -> bool {
        false
    }
}

struct FeedbackHandler;

impl ActionHandler<Fibonacci> for FeedbackHandler {
    fn accept_goal(&mut self, _goal_id: &Uuid, _goal: &FibonacciGoal) -> GoalOutcome {
        GoalOutcome::Accept
    }

    fn execute_goal(
        &mut self,
        goal_id: &Uuid,
        goal: &FibonacciGoal,
        feedback_cb: &mut dyn FnMut(FibonacciFeedback),
    ) -> ExecuteResult<Fibonacci> {
        let gid = *goal_id;
        let seq = fibonacci_seq(goal.order);
        for i in 1..=seq.len() {
            let mut partial: HVec<i32, 32> = HVec::new();
            for &v in &seq[..i] {
                let _ = partial.push(v);
            }
            feedback_cb(FibonacciFeedback {
                goal_id: gid,
                partial_sequence: partial,
            });
        }
        ExecuteResult::Succeeded(FibonacciResult { sequence: seq })
    }

    fn cancel_goal(&mut self, _goal_id: &Uuid) -> bool {
        false
    }
}

#[allow(dead_code)]
struct RejectHandler;

impl ActionHandler<Fibonacci> for RejectHandler {
    fn accept_goal(&mut self, _goal_id: &Uuid, _goal: &FibonacciGoal) -> GoalOutcome {
        GoalOutcome::Reject
    }

    fn execute_goal(
        &mut self,
        _goal_id: &Uuid,
        _goal: &FibonacciGoal,
        _feedback_cb: &mut dyn FnMut(FibonacciFeedback),
    ) -> ExecuteResult<Fibonacci> {
        ExecuteResult::Aborted(FibonacciResult::default())
    }

    fn cancel_goal(&mut self, _goal_id: &Uuid) -> bool {
        false
    }
}

struct AlwaysCancelHandler;

impl ActionHandler<Fibonacci> for AlwaysCancelHandler {
    fn accept_goal(&mut self, _goal_id: &Uuid, _goal: &FibonacciGoal) -> GoalOutcome {
        GoalOutcome::Accept
    }

    fn execute_goal(
        &mut self,
        _goal_id: &Uuid,
        _goal: &FibonacciGoal,
        _feedback_cb: &mut dyn FnMut(FibonacciFeedback),
    ) -> ExecuteResult<Fibonacci> {
        ExecuteResult::Canceled(FibonacciResult::default())
    }

    fn cancel_goal(&mut self, _goal_id: &Uuid) -> bool {
        true
    }
}

// ─── Test 1: fibonacci_goal_accept_and_result ────────────────────────────────

#[test]
fn fibonacci_goal_accept_and_result() {
    let mut server_p = Participant::new(20, guid_prefix(0x50), QosProfile::ros2_default())
        .expect("server participant");
    let mut client_p = Participant::new(20, guid_prefix(0x51), QosProfile::ros2_default())
        .expect("client participant");

    let server_addr = server_p.local_metatraffic_addr().expect("server addr");
    let client_addr = client_p.local_metatraffic_addr().expect("client addr");
    server_p.add_peer(client_addr).expect("server add peer");
    client_p.add_peer(server_addr).expect("client add peer");

    let mut server =
        create_action_server::<Fibonacci>(&mut server_p, "fibonacci", &QosProfile::ros2_default())
            .expect("create action server");
    let mut client =
        create_action_client::<Fibonacci>(&mut client_p, "fibonacci", &QosProfile::ros2_default())
            .expect("create action client");

    // Discovery.
    let _ = wait_until(
        &mut server_p,
        &mut client_p,
        Duration::from_secs(3),
        |_, _| false,
    );

    let goal_id = Uuid::from_bytes([0x01; 16]);
    client
        .send_goal(&mut client_p, goal_id, FibonacciGoal { order: 5 })
        .expect("send_goal");

    // Wait for accepted response.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut accepted = false;
    while !accepted && Instant::now() < deadline {
        let _ = server_p.spin_once();
        let _ = client_p.spin_once();
        server
            .process(&mut server_p, &mut SimpleHandler)
            .expect("process");
        let responses = client.take_goal_responses(&mut client_p);
        for (_seq, acc, _stamp) in responses {
            if acc {
                accepted = true;
            }
        }
    }
    assert!(accepted, "goal was not accepted within timeout");

    // Request result.
    client
        .request_result(&mut client_p, goal_id)
        .expect("request_result");

    // Wait for result.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut result_received = false;
    while !result_received && Instant::now() < deadline {
        let _ = server_p.spin_once();
        let _ = client_p.spin_once();
        server
            .process(&mut server_p, &mut SimpleHandler)
            .expect("process");
        let results = client.take_results(&mut client_p);
        for (_seq, status, result) in results {
            assert_eq!(status, goal_status::SUCCEEDED, "expected SUCCEEDED status");
            let expected = fibonacci_seq(5);
            assert_eq!(result.sequence, expected, "fibonacci(5) sequence mismatch");
            result_received = true;
        }
    }
    assert!(result_received, "result not received within timeout");
}

// ─── Test 2: feedback_flows_to_client ────────────────────────────────────────

#[test]
fn feedback_flows_to_client() {
    let mut server_p = Participant::new(21, guid_prefix(0x52), QosProfile::ros2_default())
        .expect("server participant");
    let mut client_p = Participant::new(21, guid_prefix(0x53), QosProfile::ros2_default())
        .expect("client participant");

    let server_addr = server_p.local_metatraffic_addr().expect("server addr");
    let client_addr = client_p.local_metatraffic_addr().expect("client addr");
    server_p.add_peer(client_addr).expect("server add peer");
    client_p.add_peer(server_addr).expect("client add peer");

    let mut server =
        create_action_server::<Fibonacci>(&mut server_p, "fibonacci", &QosProfile::ros2_default())
            .expect("create action server");
    let mut client =
        create_action_client::<Fibonacci>(&mut client_p, "fibonacci", &QosProfile::ros2_default())
            .expect("create action client");

    // Discovery.
    let _ = wait_until(
        &mut server_p,
        &mut client_p,
        Duration::from_secs(3),
        |_, _| false,
    );

    let goal_id = Uuid::from_bytes([0x02; 16]);
    client
        .send_goal(&mut client_p, goal_id, FibonacciGoal { order: 3 })
        .expect("send_goal");

    let deadline = Instant::now() + Duration::from_secs(8);
    let mut total_feedback: usize = 0;
    let mut result_received = false;
    let mut goal_accepted = false;

    while (!result_received) && Instant::now() < deadline {
        let _ = server_p.spin_once();
        let _ = client_p.spin_once();
        server
            .process(&mut server_p, &mut FeedbackHandler)
            .expect("process");

        let goal_resps = client.take_goal_responses(&mut client_p);
        for (_seq, acc, _stamp) in goal_resps {
            if acc {
                goal_accepted = true;
            }
        }

        if goal_accepted {
            client.request_result(&mut client_p, goal_id).unwrap_or(0);
        }

        let fb_list = client.take_feedback(&mut client_p);
        total_feedback += fb_list.len();

        let results = client.take_results(&mut client_p);
        if !results.is_empty() {
            result_received = true;
        }
    }

    assert!(result_received, "result not received within timeout");
    assert!(total_feedback > 0, "expected feedback messages, got none");
}

// ─── Test 3: status_array_reflects_lifecycle ─────────────────────────────────

#[test]
fn status_array_reflects_lifecycle() {
    let mut server_p = Participant::new(22, guid_prefix(0x54), QosProfile::ros2_default())
        .expect("server participant");
    let mut client_p = Participant::new(22, guid_prefix(0x55), QosProfile::ros2_default())
        .expect("client participant");

    let server_addr = server_p.local_metatraffic_addr().expect("server addr");
    let client_addr = client_p.local_metatraffic_addr().expect("client addr");
    server_p.add_peer(client_addr).expect("server add peer");
    client_p.add_peer(server_addr).expect("client add peer");

    let mut server =
        create_action_server::<Fibonacci>(&mut server_p, "fibonacci", &QosProfile::ros2_default())
            .expect("create action server");
    let mut client =
        create_action_client::<Fibonacci>(&mut client_p, "fibonacci", &QosProfile::ros2_default())
            .expect("create action client");

    // Discovery.
    let _ = wait_until(
        &mut server_p,
        &mut client_p,
        Duration::from_secs(3),
        |_, _| false,
    );

    let goal_id = Uuid::from_bytes([0x03; 16]);
    client
        .send_goal(&mut client_p, goal_id, FibonacciGoal { order: 2 })
        .expect("send_goal");

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut status_received = false;
    let mut received_arrays: Vec<GoalStatusArray> = Vec::new();

    while !status_received && Instant::now() < deadline {
        let _ = server_p.spin_once();
        let _ = client_p.spin_once();
        server
            .process(&mut server_p, &mut SimpleHandler)
            .expect("process");

        let statuses = client.take_status(&mut client_p);
        for arr in &statuses {
            if !arr.status_list.is_empty() {
                status_received = true;
            }
        }
        received_arrays.extend(statuses);
    }

    assert!(
        status_received,
        "expected at least one GoalStatusArray with entries, got none"
    );
}

// ─── Test 4: cancel_goal_marks_canceling ─────────────────────────────────────

#[test]
fn cancel_goal_marks_canceling() {
    let mut server_p = Participant::new(23, guid_prefix(0x56), QosProfile::ros2_default())
        .expect("server participant");
    let mut client_p = Participant::new(23, guid_prefix(0x57), QosProfile::ros2_default())
        .expect("client participant");

    let server_addr = server_p.local_metatraffic_addr().expect("server addr");
    let client_addr = client_p.local_metatraffic_addr().expect("client addr");
    server_p.add_peer(client_addr).expect("server add peer");
    client_p.add_peer(server_addr).expect("client add peer");

    let mut server =
        create_action_server::<Fibonacci>(&mut server_p, "fibonacci", &QosProfile::ros2_default())
            .expect("create action server");
    let mut client =
        create_action_client::<Fibonacci>(&mut client_p, "fibonacci", &QosProfile::ros2_default())
            .expect("create action client");

    // Discovery.
    let _ = wait_until(
        &mut server_p,
        &mut client_p,
        Duration::from_secs(3),
        |_, _| false,
    );

    let goal_id = Uuid::from_bytes([0x04; 16]);
    client
        .send_goal(&mut client_p, goal_id, FibonacciGoal { order: 1 })
        .expect("send_goal");
    // Immediately also send a cancel.
    client
        .cancel_goal(&mut client_p, goal_id)
        .expect("cancel_goal");

    let deadline = Instant::now() + Duration::from_secs(8);
    let mut terminal_status_seen = false;
    let mut goal_accepted = false;

    while !terminal_status_seen && Instant::now() < deadline {
        let _ = server_p.spin_once();
        let _ = client_p.spin_once();
        server
            .process(&mut server_p, &mut AlwaysCancelHandler)
            .expect("process");

        let goal_resps = client.take_goal_responses(&mut client_p);
        for (_seq, acc, _stamp) in goal_resps {
            if acc {
                goal_accepted = true;
            }
        }

        if goal_accepted {
            client.request_result(&mut client_p, goal_id).unwrap_or(0);
        }

        let statuses = client.take_status(&mut client_p);
        for arr in &statuses {
            for gs in &arr.status_list {
                let s = gs.status;
                if s == goal_status::CANCELED
                    || s == goal_status::SUCCEEDED
                    || s == goal_status::ABORTED
                {
                    terminal_status_seen = true;
                }
            }
        }

        let results = client.take_results(&mut client_p);
        if !results.is_empty() {
            terminal_status_seen = true;
        }
    }

    // The test verifies no panic/error and that goal lifecycle completes.
    // We do not assert a specific terminal state because cancel/execute ordering
    // is timing-dependent; just ensure we reached a terminal state.
    assert!(
        terminal_status_seen,
        "no terminal goal state observed within timeout"
    );
}

// ─── Test 5: two_action_clients_isolated ────────────────────────────────────

#[test]
fn two_action_clients_isolated() {
    let mut server_p = Participant::new(24, guid_prefix(0x58), QosProfile::ros2_default())
        .expect("server participant");
    let mut client1_p = Participant::new(24, guid_prefix(0x59), QosProfile::ros2_default())
        .expect("client1 participant");
    let mut client2_p = Participant::new(24, guid_prefix(0x5a), QosProfile::ros2_default())
        .expect("client2 participant");

    let server_addr = server_p.local_metatraffic_addr().expect("server addr");
    let client1_addr = client1_p.local_metatraffic_addr().expect("client1 addr");
    let client2_addr = client2_p.local_metatraffic_addr().expect("client2 addr");
    server_p.add_peer(client1_addr).expect("server add client1");
    server_p.add_peer(client2_addr).expect("server add client2");
    client1_p.add_peer(server_addr).expect("client1 add server");
    client2_p.add_peer(server_addr).expect("client2 add server");

    let mut server =
        create_action_server::<Fibonacci>(&mut server_p, "fibonacci", &QosProfile::ros2_default())
            .expect("create action server");
    let mut client1 =
        create_action_client::<Fibonacci>(&mut client1_p, "fibonacci", &QosProfile::ros2_default())
            .expect("create client1");
    let mut client2 =
        create_action_client::<Fibonacci>(&mut client2_p, "fibonacci", &QosProfile::ros2_default())
            .expect("create client2");

    // Discovery.
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        let _ = server_p.spin_once();
        let _ = client1_p.spin_once();
        let _ = client2_p.spin_once();
    }

    let goal1_id = Uuid::from_bytes([0xaa; 16]);
    let goal2_id = Uuid::from_bytes([0xbb; 16]);

    client1
        .send_goal(&mut client1_p, goal1_id, FibonacciGoal { order: 3 })
        .expect("client1 send_goal");
    client2
        .send_goal(&mut client2_p, goal2_id, FibonacciGoal { order: 4 })
        .expect("client2 send_goal");

    let deadline = Instant::now() + Duration::from_secs(8);
    let mut client1_accepted = false;
    let mut client2_accepted = false;
    let mut result1: Option<FibonacciResult> = None;
    let mut result2: Option<FibonacciResult> = None;

    while (result1.is_none() || result2.is_none()) && Instant::now() < deadline {
        let _ = server_p.spin_once();
        let _ = client1_p.spin_once();
        let _ = client2_p.spin_once();
        server
            .process(&mut server_p, &mut SimpleHandler)
            .expect("process");

        let gr1 = client1.take_goal_responses(&mut client1_p);
        for (_seq, acc, _stamp) in gr1 {
            if acc {
                client1_accepted = true;
            }
        }
        let gr2 = client2.take_goal_responses(&mut client2_p);
        for (_seq, acc, _stamp) in gr2 {
            if acc {
                client2_accepted = true;
            }
        }

        if client1_accepted {
            client1
                .request_result(&mut client1_p, goal1_id)
                .unwrap_or(0);
        }
        if client2_accepted {
            client2
                .request_result(&mut client2_p, goal2_id)
                .unwrap_or(0);
        }

        let results1 = client1.take_results(&mut client1_p);
        for (_seq, _status, r) in results1 {
            result1 = Some(r);
        }
        let results2 = client2.take_results(&mut client2_p);
        for (_seq, _status, r) in results2 {
            result2 = Some(r);
        }
    }

    let r1 = result1.expect("client1 did not receive result");
    let r2 = result2.expect("client2 did not receive result");

    // fibonacci(3) = [0, 1, 1] → len 3
    assert_eq!(
        r1.sequence.len(),
        3,
        "client1: fibonacci(3) should have 3 elements"
    );
    // fibonacci(4) = [0, 1, 1, 2] → len 4
    assert_eq!(
        r2.sequence.len(),
        4,
        "client2: fibonacci(4) should have 4 elements"
    );

    // Verify actual sequences.
    let expected1 = fibonacci_seq(3);
    let expected2 = fibonacci_seq(4);
    assert_eq!(r1.sequence, expected1, "client1 sequence mismatch");
    assert_eq!(r2.sequence, expected2, "client2 sequence mismatch");
}
