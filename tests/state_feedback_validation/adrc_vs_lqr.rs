//! ADRC vs LQR comparison on a second-order plant with step disturbance.
//!
//! ADRC uses its ESO to estimate and cancel the disturbance, enabling offset-free
//! tracking. Vanilla LQR (no integral action) will exhibit a persistent steady-state
//! error under a constant disturbance matched through the input channel.

#[cfg(feature = "state_feedback")]
mod inner {
    use oxictl::core::matrix::Matrix;
    use oxictl::state_feedback::adrc::SecondOrderAdrc;
    use oxictl::state_feedback::lqr::Lqr;

    /// Second-order plant (discrete, Euler): ẍ = u + d
    /// State: x = [position, velocity]
    /// x0[k+1] = x0[k] + dt * x1[k]
    /// x1[k+1] = x1[k] + dt * (u[k] + d[k])
    fn plant_step(state: [f64; 2], u: f64, d: f64, dt: f64) -> [f64; 2] {
        [state[0] + dt * state[1], state[1] + dt * (u + d)]
    }

    /// ADRC recovers from step disturbance and tracks reference within N steps.
    #[test]
    fn adrc_recovers_from_step_disturbance() {
        let dt = 0.005_f64;
        let omega_o = 60.0_f64;
        let omega_c = 10.0_f64;
        let b = 1.0_f64; // high-frequency gain (known)
        let r = 1.0_f64; // reference position

        let mut adrc = SecondOrderAdrc::new(omega_o, omega_c, b, dt)
            .expect("SecondOrderAdrc construction should succeed with positive params");

        let mut state = [0.0_f64; 2];
        let mut u_prev = 0.0_f64;

        // Phase 1: settle without disturbance (500 steps)
        for _ in 0..500 {
            let u = adrc.update(state[0], r, 0.0, u_prev);
            state = plant_step(state, u, 0.0, dt);
            u_prev = u;
        }

        // Phase 2: step disturbance applied (d = 2.0)
        let d = 2.0_f64;
        for _ in 0..500 {
            let u = adrc.update(state[0], r, 0.0, u_prev);
            state = plant_step(state, u, d, dt);
            u_prev = u;
        }

        // ADRC should recover — position tracks reference
        let position_error = (state[0] - r).abs();
        assert!(
            position_error < 0.15,
            "ADRC position error after disturbance recovery: {:.4} (expected < 0.15)",
            position_error
        );

        // ESO should have a non-trivial disturbance estimate
        let d_est = adrc.disturbance_estimate();
        assert!(
            d_est.abs() > 0.5,
            "ADRC disturbance estimate z3={:.4} should reflect active disturbance",
            d_est
        );
    }

    /// LQR without integral action has a non-zero steady-state error under disturbance.
    #[test]
    fn lqr_has_steady_state_error_without_integral() {
        let dt = 0.005_f64;
        // Discrete-time double-integrator matrices
        let a = Matrix::<f64, 2, 2> {
            data: [[1.0, dt], [0.0, 1.0]],
        };
        let b_mat = Matrix::<f64, 2, 1> {
            data: [[0.5 * dt * dt], [dt]],
        };
        let q = Matrix::<f64, 2, 2> {
            data: [[10.0, 0.0], [0.0, 1.0]],
        };
        let r_mat = Matrix::<f64, 1, 1> { data: [[0.01]] };

        let lqr = Lqr::design(&a, &b_mat, &q, &r_mat)
            .expect("LQR design should succeed for controllable double integrator");

        let r_ref = 1.0_f64;
        let mut x = [0.0_f64; 2];
        let d = 2.0_f64; // constant force disturbance applied at velocity state

        for _ in 0..3000 {
            let u_arr = lqr.control(&x, &[r_ref, 0.0]);
            let u = u_arr[0];
            // Apply plant dynamics with disturbance added at the input channel
            let x0_new = a.data[0][0] * x[0]
                + a.data[0][1] * x[1]
                + b_mat.data[0][0] * u
                + b_mat.data[0][0] * d;
            let x1_new = a.data[1][0] * x[0]
                + a.data[1][1] * x[1]
                + b_mat.data[1][0] * u
                + b_mat.data[1][0] * d;
            x = [x0_new, x1_new];
        }

        // Pure LQR (without integral) cannot eliminate the steady-state error
        // caused by the unknown persistent disturbance
        let steady_state_error = (x[0] - r_ref).abs();
        assert!(
            steady_state_error > 0.01,
            "LQR without integral should have non-zero steady-state error: {:.4}",
            steady_state_error
        );
    }

    /// Quantitative comparison: ADRC error is significantly smaller than LQR error
    /// after disturbance settles.
    #[test]
    fn adrc_outperforms_lqr_under_step_disturbance() {
        let dt = 0.005_f64;
        let r_ref = 1.0_f64;
        let d = 1.5_f64;
        let n_settle = 600_usize;
        let n_disturb = 600_usize;

        // --- ADRC simulation ---
        let mut adrc = SecondOrderAdrc::new(60.0, 10.0, 1.0, dt).expect("ADRC params valid");
        let mut x_adrc = [0.0_f64; 2];
        let mut u_prev = 0.0_f64;
        for _ in 0..n_settle {
            let u = adrc.update(x_adrc[0], r_ref, 0.0, u_prev);
            x_adrc = plant_step(x_adrc, u, 0.0, dt);
            u_prev = u;
        }
        for _ in 0..n_disturb {
            let u = adrc.update(x_adrc[0], r_ref, 0.0, u_prev);
            x_adrc = plant_step(x_adrc, u, d, dt);
            u_prev = u;
        }
        let adrc_error = (x_adrc[0] - r_ref).abs();

        // --- LQR simulation ---
        let a = Matrix::<f64, 2, 2> {
            data: [[1.0, dt], [0.0, 1.0]],
        };
        let b_mat = Matrix::<f64, 2, 1> {
            data: [[0.5 * dt * dt], [dt]],
        };
        let q = Matrix::<f64, 2, 2> {
            data: [[10.0, 0.0], [0.0, 1.0]],
        };
        let r_mat = Matrix::<f64, 1, 1> { data: [[0.01]] };
        let lqr = Lqr::design(&a, &b_mat, &q, &r_mat).expect("LQR design should succeed");

        let mut x_lqr = [0.0_f64; 2];
        for _ in 0..(n_settle + n_disturb) {
            let u_arr = lqr.control(&x_lqr, &[r_ref, 0.0]);
            let u = u_arr[0];
            let disturbance = d;
            let x0_new = a.data[0][0] * x_lqr[0]
                + a.data[0][1] * x_lqr[1]
                + b_mat.data[0][0] * (u + disturbance);
            let x1_new = a.data[1][0] * x_lqr[0]
                + a.data[1][1] * x_lqr[1]
                + b_mat.data[1][0] * (u + disturbance);
            x_lqr = [x0_new, x1_new];
        }
        let lqr_error = (x_lqr[0] - r_ref).abs();

        assert!(
            adrc_error < lqr_error,
            "ADRC error ({:.4}) should be less than LQR error ({:.4}) under disturbance",
            adrc_error,
            lqr_error
        );
    }
}

#[cfg(not(feature = "state_feedback"))]
#[test]
fn adrc_vs_lqr_skipped_without_feature() {
    // Test is skipped without the state_feedback feature.
}
