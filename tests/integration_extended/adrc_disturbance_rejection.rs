//! ADRC disturbance rejection integration test.
//!
//! Tests ADRC controlling a double-integrator plant with combined sinusoidal
//! and step disturbances over 1000 steps. The output should track the reference
//! with RMS error below a meaningful threshold.

#[cfg(feature = "state_feedback")]
mod inner {
    use oxictl::state_feedback::adrc::SecondOrderAdrc;

    /// Double integrator plant: ẍ = u + d(t)
    /// Discrete Euler: x[k+1] = [x[0] + dt*x[1], x[1] + dt*(u + d)]
    fn double_integrator_step(state: [f64; 2], u: f64, d: f64, dt: f64) -> [f64; 2] {
        [state[0] + dt * state[1], state[1] + dt * (u + d)]
    }

    /// ADRC tracks a step reference with sinusoidal + step disturbance.
    #[test]
    fn adrc_rejects_combined_disturbance() {
        let dt = 0.005_f64;
        let omega_o = 80.0_f64;
        let omega_c = 12.0_f64;
        let b = 1.0_f64;
        let r_ref = 1.0_f64;

        let mut adrc = SecondOrderAdrc::new(omega_o, omega_c, b, dt)
            .expect("SecondOrderAdrc should construct with valid parameters");

        let mut state = [0.0_f64; 2];
        let mut u_prev = 0.0_f64;

        // Phase 1: settle on reference without disturbance (200 steps)
        for _ in 0..200 {
            let u = adrc.update(state[0], r_ref, 0.0, u_prev);
            state = double_integrator_step(state, u, 0.0, dt);
            u_prev = u;
        }

        // Phase 2: apply combined disturbance for 1000 steps
        let mut rms_error_sq = 0.0_f64;
        let n_steps = 1000_usize;
        let d_step = 1.5_f64; // constant step disturbance
        let d_sin_amp = 0.5_f64; // sinusoidal disturbance amplitude
        let d_freq = 3.0_f64; // rad/s

        for k in 0..n_steps {
            let t = k as f64 * dt;
            let d_sin = d_sin_amp * (d_freq * t).sin();
            let d = d_step + d_sin;

            let u = adrc.update(state[0], r_ref, 0.0, u_prev);
            state = double_integrator_step(state, u, d, dt);
            u_prev = u;

            let error = state[0] - r_ref;
            rms_error_sq += error * error;
        }

        let rms_error = (rms_error_sq / n_steps as f64).sqrt();

        assert!(
            rms_error < 0.5,
            "ADRC RMS error under combined disturbance: {rms_error:.4} (expected < 0.5)"
        );
    }

    /// ADRC disturbance estimate is non-trivial under active disturbance.
    #[test]
    fn adrc_eso_estimates_disturbance() {
        let dt = 0.005_f64;
        let mut adrc = SecondOrderAdrc::new(80.0, 12.0, 1.0, dt).expect("ADRC construction");

        let mut state = [0.0_f64; 2];
        let mut u_prev = 0.0_f64;
        let r_ref = 1.0_f64;
        let d = 2.0_f64; // constant disturbance

        // Run for 500 steps with disturbance
        for _ in 0..500 {
            let u = adrc.update(state[0], r_ref, 0.0, u_prev);
            state = double_integrator_step(state, u, d, dt);
            u_prev = u;
        }

        let d_est = adrc.disturbance_estimate();
        // ESO z3 should have built up a significant disturbance estimate
        assert!(
            d_est.abs() > 0.5,
            "ADRC disturbance estimate z3={d_est:.4} should be significantly non-zero"
        );
    }

    /// ADRC output is bounded (does not diverge) under periodic disturbance.
    #[test]
    fn adrc_output_bounded_under_periodic_disturbance() {
        let dt = 0.002_f64;
        let mut adrc = SecondOrderAdrc::new(100.0, 15.0, 1.0, dt).expect("ADRC construction");

        let mut state = [0.0_f64; 2];
        let mut u_prev = 0.0_f64;
        let r_ref = 0.5_f64;

        for k in 0..1000 {
            let t = k as f64 * dt;
            let d = 1.0 * (5.0 * t).sin() + 0.5 * (2.0 * t).cos();

            let u = adrc.update(state[0], r_ref, 0.0, u_prev);
            state = double_integrator_step(state, u, d, dt);
            u_prev = u;

            // Check for divergence
            assert!(
                state[0].abs() < 20.0,
                "ADRC output diverged at step {k}: x={:.4}",
                state[0]
            );
            assert!(
                state[1].abs() < 50.0,
                "ADRC velocity diverged at step {k}: v={:.4}",
                state[1]
            );
        }
    }

    /// ADRC achieves lower RMS error than open-loop (no control) under disturbance.
    #[test]
    fn adrc_outperforms_open_loop_under_disturbance() {
        let dt = 0.005_f64;
        let r_ref = 1.0_f64;
        let d_amp = 1.0_f64;
        let n_steps = 1000_usize;

        // ADRC controlled
        let mut adrc = SecondOrderAdrc::new(80.0, 12.0, 1.0, dt).expect("ADRC construction");
        let mut state_adrc = [0.0_f64; 2];
        let mut u_prev = 0.0_f64;
        let mut rms_adrc = 0.0_f64;

        for k in 0..n_steps {
            let t = k as f64 * dt;
            let d = d_amp * (3.0 * t).sin();
            let u = adrc.update(state_adrc[0], r_ref, 0.0, u_prev);
            state_adrc = double_integrator_step(state_adrc, u, d, dt);
            u_prev = u;
            let e = state_adrc[0] - r_ref;
            rms_adrc += e * e;
        }
        let rms_adrc = (rms_adrc / n_steps as f64).sqrt();

        // Open loop (u = 0, disturbance pushes state)
        let mut state_ol = [0.0_f64; 2];
        let mut rms_ol = 0.0_f64;
        for k in 0..n_steps {
            let t = k as f64 * dt;
            let d = d_amp * (3.0 * t).sin();
            state_ol = double_integrator_step(state_ol, 0.0, d, dt);
            let e = state_ol[0] - r_ref;
            rms_ol += e * e;
        }
        let rms_ol = (rms_ol / n_steps as f64).sqrt();

        assert!(
            rms_adrc < rms_ol,
            "ADRC RMS ({rms_adrc:.4}) should be < open-loop RMS ({rms_ol:.4})"
        );
    }
}

#[cfg(not(feature = "state_feedback"))]
#[test]
fn adrc_disturbance_rejection_skipped_without_feature() {
    // Skipped: requires state_feedback feature.
}
