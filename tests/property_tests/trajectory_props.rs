//! Property-based tests for trajectory generator invariants.

use oxictl::trajectory::quintic::QuinticPolynomial;
use oxictl::trajectory::trapezoidal::TrapezoidalProfile;
use proptest::prelude::*;

proptest! {
    /// Trapezoidal: position at t=0 is 0.
    #[test]
    fn trapezoidal_starts_at_zero(
        distance in 0.1f64..100.0,
        v_max in 0.1f64..20.0,
        a_max in 0.1f64..100.0,
    ) {
        let mut profile = TrapezoidalProfile::new(v_max, a_max);
        profile.plan(distance);
        let (pos, _, _) = profile.query(0.0);
        prop_assert!(pos.abs() < 1e-12, "Position at t=0 should be 0, got {pos}");
    }

    /// Trapezoidal: position at t=total_time reaches target.
    #[test]
    fn trapezoidal_reaches_target(
        distance in 0.1f64..100.0,
        v_max in 0.1f64..20.0,
        a_max in 0.1f64..100.0,
    ) {
        let mut profile = TrapezoidalProfile::new(v_max, a_max);
        profile.plan(distance);
        let t_end = profile.total_time();
        let (pos, _, _) = profile.query(t_end);
        prop_assert!(
            (pos - distance).abs() < 1e-6 * distance.max(1.0),
            "Should reach {distance} at end, got {pos}"
        );
    }

    /// Trapezoidal: velocity is zero at t=0.
    #[test]
    fn trapezoidal_zero_velocity_at_start(
        distance in 0.1f64..100.0,
        v_max in 0.1f64..20.0,
        a_max in 0.1f64..100.0,
    ) {
        let mut profile = TrapezoidalProfile::new(v_max, a_max);
        profile.plan(distance);
        let (_, v_start, _) = profile.query(0.0);
        prop_assert!(v_start.abs() < 1e-12, "Velocity at start should be 0, got {v_start}");
    }

    /// Trapezoidal: position is monotonically non-decreasing for positive distance.
    #[test]
    fn trapezoidal_monotone_position(
        distance in 0.1f64..50.0,
        v_max in 0.1f64..10.0,
        a_max in 0.1f64..50.0,
    ) {
        let mut profile = TrapezoidalProfile::new(v_max, a_max);
        profile.plan(distance);
        let t_end = profile.total_time();
        let n = 20usize;
        let mut prev_pos = -1.0f64;
        for i in 0..=n {
            let t = t_end * i as f64 / n as f64;
            let (pos, _, _) = profile.query(t);
            prop_assert!(pos >= prev_pos - 1e-9, "Position should increase: {prev_pos} → {pos} at t={t}");
            prev_pos = pos;
        }
    }

    /// Trapezoidal: velocity never exceeds v_max.
    #[test]
    fn trapezoidal_velocity_bounded(
        distance in 0.1f64..50.0,
        v_max in 0.1f64..10.0,
        a_max in 0.1f64..50.0,
    ) {
        let mut profile = TrapezoidalProfile::new(v_max, a_max);
        profile.plan(distance);
        let t_end = profile.total_time();
        let n = 50usize;
        for i in 0..=n {
            let t = t_end * i as f64 / n as f64;
            let (_, v, _) = profile.query(t);
            prop_assert!(
                v <= v_max + 1e-9,
                "Velocity {v} exceeds v_max={v_max} at t={t}"
            );
            prop_assert!(v >= -1e-9, "Negative velocity {v} at t={t}");
        }
    }

    /// Quintic: rest-to-rest boundary conditions satisfied.
    #[test]
    fn quintic_rest_to_rest_boundary_conditions(
        distance in -10.0f64..10.0,
        duration in 0.1f64..5.0,
    ) {
        prop_assume!(distance.abs() > 0.01 && duration > 0.05);
        if let Some(poly) = QuinticPolynomial::rest_to_rest(0.0, distance, duration) {
            let p0 = poly.position(0.0);
            let v0 = poly.velocity(0.0);
            let a0 = poly.acceleration(0.0);
            let pf = poly.position(duration);
            let vf = poly.velocity(duration);
            let af = poly.acceleration(duration);
            prop_assert!(p0.abs() < 1e-10, "Start pos should be 0: {p0}");
            prop_assert!(v0.abs() < 1e-10, "Start vel should be 0: {v0}");
            prop_assert!(a0.abs() < 1e-10, "Start acc should be 0: {a0}");
            prop_assert!((pf - distance).abs() < 1e-9 * distance.abs().max(1.0), "End pos={pf} should be {distance}");
            prop_assert!(vf.abs() < 1e-9, "End vel should be 0: {vf}");
            prop_assert!(af.abs() < 1e-9, "End acc should be 0: {af}");
        }
    }

    /// Quintic: position is monotone for positive distance with sufficient duration.
    #[test]
    fn quintic_positive_distance_positive_position(
        distance in 0.1f64..10.0,
        duration in 0.5f64..5.0,
    ) {
        if let Some(poly) = QuinticPolynomial::rest_to_rest(0.0, distance, duration) {
            // At the midpoint, position should be between 0 and distance
            let pmid = poly.position(duration * 0.5);
            prop_assert!(pmid >= -1e-9 && pmid <= distance + 1e-9,
                "Mid-position {pmid} outside [0, {distance}]");
        }
    }
}
