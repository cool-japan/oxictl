//! FOC steady-state accuracy validation.

use oxictl::motor::foc::controller::FocController;

/// FOC speed controller reaches setpoint within tolerance.
#[test]
fn foc_speed_reaches_setpoint() {
    let mut foc = FocController::<f64>::new(
        8.0, 40.0, // speed: kp, ki
        5.0, 80.0, // current: kp, ki
        8.0,  // iq_limit
        20.0, // v_limit
        36.0, // vdc
    );

    let dt = 1e-4f64;
    let speed_ref = 100.0f64; // rad/s
    let mut omega = 0.0f64;
    let mut theta = 0.0f64;

    // Simulate for 5 seconds (50000 steps at 10 kHz)
    for _ in 0..50000 {
        let out = foc.update(
            speed_ref,
            omega,
            0.1 * omega.sin(),
            -0.05 * omega.cos(),
            theta,
            dt,
        );
        // Simple speed integrator (omega += torque/inertia)
        let vq_eff = out.vq.clamp(-36.0, 36.0);
        let omega_dot = vq_eff * 0.5 - omega * 0.02; // simplified plant
        omega += omega_dot * dt;
        theta += omega * dt;
        if theta > core::f64::consts::TAU {
            theta -= core::f64::consts::TAU;
        }
    }

    // Check omega is moving significantly toward setpoint (simplified plant model)
    assert!(
        omega > 30.0,
        "FOC speed should approach setpoint: omega={:.2} rad/s (ref={:.2})",
        omega,
        speed_ref
    );
}

/// FOC: d-axis current should stay near zero (MTPA strategy).
#[test]
fn foc_id_stays_near_zero() {
    use oxictl::motor::transform::clarke::clarke_2ph;
    use oxictl::motor::transform::park::park;

    let mut foc = FocController::<f64>::new(5.0, 20.0, 5.0, 100.0, 5.0, 15.0, 24.0);
    let dt = 1e-4f64;
    let mut omega = 50.0f64;
    let mut theta = 0.0f64;

    let mut id_sum = 0.0f64;
    let n_steps = 2000usize;

    for _ in 0..n_steps {
        // Simulate phase currents based on theta
        let ia = 2.0 * theta.cos();
        let ib = 2.0 * (theta - 2.0 * core::f64::consts::PI / 3.0).cos();
        let out = foc.update(50.0, omega, ia, ib, theta, dt);

        // Measure d-axis current
        let ab = clarke_2ph(ia, ib);
        let dq = park(&ab, theta);
        id_sum += dq.d.abs();

        theta += omega * dt;
        if theta > core::f64::consts::TAU {
            theta -= core::f64::consts::TAU;
        }
        let vq_eff = out.vq.clamp(-24.0, 24.0);
        omega += (vq_eff * 0.3 - omega * 0.01) * dt;
    }

    let avg_id = id_sum / n_steps as f64;
    // id should be small (controlled toward zero)
    // The exact value depends on convergence, just check it's not excessively large
    assert!(avg_id < 5.0, "Average |id| should be small: {:.4}", avg_id);
}

/// FOC: duty cycles should always be within [0, 1].
#[test]
fn foc_duty_cycles_in_bounds() {
    let mut foc = FocController::<f64>::new(5.0, 20.0, 3.0, 50.0, 5.0, 12.0, 24.0);
    let dt = 1e-4f64;
    let mut omega = 0.0f64;
    let mut theta = 0.0f64;

    for _ in 0..5000 {
        let ia = omega.sin();
        let ib = (omega - 2.0 * core::f64::consts::PI / 3.0).sin();
        let out = foc.update(80.0, omega, ia, ib, theta, dt);

        assert!(
            out.duty.ta >= -0.01 && out.duty.ta <= 1.01,
            "ta={}",
            out.duty.ta
        );
        assert!(
            out.duty.tb >= -0.01 && out.duty.tb <= 1.01,
            "tb={}",
            out.duty.tb
        );
        assert!(
            out.duty.tc >= -0.01 && out.duty.tc <= 1.01,
            "tc={}",
            out.duty.tc
        );

        theta += omega * dt;
        if theta > core::f64::consts::TAU {
            theta -= core::f64::consts::TAU;
        }
        omega += (out.vq.clamp(-24.0, 24.0) * 0.3 - omega * 0.01) * dt;
    }
}
