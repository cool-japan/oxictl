//! Watchdog trip validation tests.

use oxictl::safety::monitor::timeout::TimeoutMonitor;
use oxictl::safety::watchdog::Watchdog;

/// Watchdog trips after timeout without kick.
#[test]
fn watchdog_trips_on_timeout() {
    let mut wd = Watchdog::<f64>::new(0.1);
    assert!(!wd.is_tripped());

    // Advance past timeout
    let tripped = wd.check(0.15);
    assert!(tripped, "Should trip after exceeding timeout");
    assert!(wd.is_tripped());
}

/// Watchdog does not trip if kicked in time.
#[test]
fn watchdog_does_not_trip_if_kicked() {
    let mut wd = Watchdog::<f64>::new(0.1);

    for _ in 0..10 {
        wd.check(0.008); // 8ms each, kick every loop iteration
        wd.kick();
    }

    assert!(!wd.is_tripped(), "Should not trip if kicked regularly");
}

/// Watchdog elapsed time tracking.
#[test]
fn watchdog_elapsed_tracks_time() {
    let mut wd = Watchdog::<f64>::new(1.0);
    wd.check(0.3);
    wd.check(0.2);
    assert!(
        (wd.elapsed() - 0.5).abs() < 1e-10,
        "Elapsed={:.3}",
        wd.elapsed()
    );
}

/// Watchdog kick resets elapsed time.
#[test]
fn watchdog_kick_resets_elapsed() {
    let mut wd = Watchdog::<f64>::new(1.0);
    wd.check(0.4);
    wd.kick();
    assert!(wd.elapsed() < 1e-10, "Elapsed should be zero after kick");
}

/// Watchdog reset clears tripped state.
#[test]
fn watchdog_reset_clears_trip() {
    let mut wd = Watchdog::<f64>::new(0.05);
    wd.check(0.1);
    assert!(wd.is_tripped());

    wd.reset();
    assert!(!wd.is_tripped(), "Should not be tripped after reset");
    assert!(wd.elapsed() < 1e-10);
}

/// Watchdog: multiple kicks in rapid succession.
#[test]
fn watchdog_multiple_kicks() {
    let mut wd = Watchdog::<f64>::new(0.1);

    for i in 0..20 {
        let tripped = wd.check(0.004); // 4ms steps
        assert!(!tripped, "Should not trip at step {}", i);
        wd.kick(); // Kick every step
    }
}

/// Watchdog: timeout property preserved.
#[test]
fn watchdog_timeout_property() {
    let wd = Watchdog::<f64>::new(0.25);
    assert!((wd.timeout() - 0.25).abs() < 1e-10);
}

/// TimeoutMonitor: detects missing updates.
#[test]
fn timeout_monitor_detects_missing_update() {
    let mut mon = TimeoutMonitor::<f64>::new(0.1);

    // Update periodically
    for _ in 0..5 {
        assert!(mon.check(0.02), "Should be OK within timeout");
        mon.feed();
    }

    // Miss several updates
    for _ in 0..6 {
        mon.check(0.02); // 6 * 0.02 = 0.12 > 0.1
    }
    let result = mon.check(0.0);
    assert!(!result, "Should detect timeout after missing updates");
}
