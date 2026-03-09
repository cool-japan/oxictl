//! Safety Watchdog & Fault Handler Demo.
//!
//! Demonstrates:
//!   - Software watchdog timer
//!   - Range/Rate monitors
//!   - Fault handler with escalation
//!   - Safety monitor aggregation
//!   - Redundant voter logic

use oxictl::safety::fault::{FaultDef, FaultEvent, FaultResponse, FaultSeverity};
use oxictl::safety::handler::FaultHandler;
use oxictl::safety::monitor::range::RangeMonitor;
use oxictl::safety::monitor::rate::RateMonitor;
use oxictl::safety::redundancy::voter::{Voter, VoterStrategy};
use oxictl::safety::watchdog::Watchdog;

fn main() {
    println!("=== OxiCtl Safety Subsystem Demo ===\n");

    // ─────────────────────────────────────────────
    // 1. Watchdog Timer
    // ─────────────────────────────────────────────
    println!("--- Watchdog Demo ---");
    let mut wd = Watchdog::<f64>::new(0.1); // 100ms timeout

    // Normal operation: kick before timeout
    for i in 0..5 {
        let tripped = wd.check(0.02); // 20ms steps
        println!(
            "  step {}: elapsed={:.3}s, tripped={}",
            i,
            wd.elapsed(),
            tripped
        );
        if i == 2 {
            wd.kick();
            println!("  [WD] Kicked at step {}", i);
        }
    }

    // Simulate missed kick → watchdog trips
    let mut wd2 = Watchdog::<f64>::new(0.05);
    let mut t = 0.0f64;
    loop {
        let dt = 0.01;
        let tripped = wd2.check(dt);
        t += dt;
        if tripped {
            println!("  Watchdog tripped at t={:.3}s!", t);
            break;
        }
    }

    // ─────────────────────────────────────────────
    // 2. Range Monitor
    // ─────────────────────────────────────────────
    println!("\n--- Range Monitor Demo ---");
    let mut range_mon = RangeMonitor::<f64>::new(-10.0, 10.0);
    let test_values = [-15.0, -5.0, 0.0, 8.0, 12.0, 9.9];
    for &v in &test_values {
        let ok = range_mon.check(v);
        println!("  value={:5.1} → {}", v, if ok { "OK" } else { "FAULT" });
    }

    // ─────────────────────────────────────────────
    // 3. Rate Monitor
    // ─────────────────────────────────────────────
    println!("\n--- Rate Monitor Demo ---");
    let mut rate_mon = RateMonitor::<f64>::new(50.0); // max 50 units/s
    let values = [0.0f64, 0.3, 0.6, 1.2, 3.5, 4.0];
    for &v in &values {
        let ok = rate_mon.check(v, 0.01);
        println!(
            "  value={:.2} → {}",
            v,
            if ok { "OK" } else { "RATE EXCEEDED" }
        );
    }

    // ─────────────────────────────────────────────
    // 4. Fault Handler
    // ─────────────────────────────────────────────
    println!("\n--- Fault Handler Demo ---");
    let mut handler = FaultHandler::<16>::new();

    let fault_defs: &[(&str, u16, FaultSeverity)] = &[
        ("sensor_noise", 1, FaultSeverity::Info),
        ("over_voltage", 2, FaultSeverity::Warning),
        ("over_current", 3, FaultSeverity::Error),
    ];

    for &(name, id, severity) in fault_defs {
        let def = FaultDef {
            id,
            name,
            severity,
            response: severity.default_response(),
        };
        let event = FaultEvent::new(&def, 0.0);
        let response = handler.report(event);
        println!(
            "  [{}] severity={:?} → response={:?}",
            name, severity, response
        );
    }
    println!(
        "  Total faults: {}, has_critical: {}",
        handler.fault_count(),
        handler.has_critical()
    );

    // Critical fault → emergency stop
    let crit_def = FaultDef {
        id: 4,
        name: "motor_overheat",
        severity: FaultSeverity::Critical,
        response: FaultResponse::EmergencyStop,
    };
    let crit_event = FaultEvent::new(&crit_def, 1.0);
    let resp = handler.report(crit_event);
    println!(
        "  [motor_overheat] Critical → {:?} | has_critical: {}",
        resp,
        handler.has_critical()
    );

    // ─────────────────────────────────────────────
    // 5. Redundant Voter
    // ─────────────────────────────────────────────
    println!("\n--- Voter (2-of-3) Demo ---");
    let mut voter = Voter::<f64, 3>::new(VoterStrategy::TwoOfThree, 0.5);

    let test_channels = [
        [100.0, 100.1, 100.0], // All agree
        [100.0, 100.1, 95.0],  // One outlier (channel 2 faulty)
        [100.0, 50.0, 100.0],  // One outlier (channel 1 faulty)
        [0.0, 100.0, 200.0],   // All disagree
    ];
    for channels in &test_channels {
        match voter.vote(channels) {
            Some(v) => println!(
                "  channels={:?} → voted={:.2}, health={:?}",
                channels,
                v,
                voter.healthy_channels()
            ),
            None => println!("  channels={:?} → NO CONSENSUS", channels),
        }
    }

    // ─────────────────────────────────────────────
    // Summary
    // ─────────────────────────────────────────────
    println!("\n=== Safety demo complete ===");
    println!("All safety mechanisms functional:");
    println!("  - Watchdog timer");
    println!("  - Range monitor");
    println!("  - Rate monitor");
    println!("  - Fault handler with escalation");
    println!("  - 2-of-3 voter with fault isolation");
}
