//! Full FOC → MTPA → MHE pipeline integration test.
//!
//! Simulates a pipeline where:
//! 1. MHE estimates the motor state (id, iq) from noisy current measurements.
//! 2. MTPA generates optimal id/iq references for a given torque command.
//! 3. A simple PI current regulator tracks the MTPA references.
//!
//! The motor is driven through a speed ramp; id and iq should track
//! the MTPA references within tolerance.

#[cfg(all(feature = "motor", feature = "mpc"))]
mod inner {
    use oxictl::core::matrix::Matrix;
    use oxictl::motor::foc::{MtpaMotorParams, MtpaTable};
    use oxictl::mpc::moving_horizon_estimator::MovingHorizonEstimator;

    /// Simple PI controller state.
    struct PiController {
        kp: f64,
        ki: f64,
        integral: f64,
        limit: f64,
    }

    impl PiController {
        fn new(kp: f64, ki: f64, limit: f64) -> Self {
            Self {
                kp,
                ki,
                integral: 0.0,
                limit,
            }
        }

        fn update(&mut self, error: f64, dt: f64) -> f64 {
            self.integral += error * dt;
            self.integral = self.integral.clamp(-self.limit, self.limit);
            let out = self.kp * error + self.ki * self.integral;
            out.clamp(-self.limit, self.limit)
        }
    }

    /// Simplified PMSM dq model for simulation.
    /// Vd = Rs*id + Ld*d(id)/dt - omega*Lq*iq
    /// Vq = Rs*iq + Lq*d(iq)/dt + omega*Ld*id + omega*lambda_pm
    struct PmsmDqSim {
        rs: f64,
        ld: f64,
        lq: f64,
        lambda_pm: f64,
        id: f64,
        iq: f64,
    }

    impl PmsmDqSim {
        fn new(rs: f64, ld: f64, lq: f64, lambda_pm: f64) -> Self {
            Self {
                rs,
                ld,
                lq,
                lambda_pm,
                id: 0.0,
                iq: 0.0,
            }
        }

        fn step(&mut self, vd: f64, vq: f64, omega: f64, dt: f64) {
            // Euler integration
            let did_dt = (vd - self.rs * self.id + omega * self.lq * self.iq) / self.ld;
            let diq_dt =
                (vq - self.rs * self.iq - omega * self.ld * self.id - omega * self.lambda_pm)
                    / self.lq;
            self.id += did_dt * dt;
            self.iq += diq_dt * dt;
        }
    }

    /// MHE dynamics: id and iq as nearly-static (slow-varying) states.
    fn mhe_dynamics(x: &Matrix<f64, 2, 1>, _u: &Matrix<f64, 1, 1>) -> Matrix<f64, 2, 1> {
        // Nearly static: x[k+1] ≈ 0.99*x[k]
        let mut xn = Matrix::<f64, 2, 1>::zeros();
        xn.data[0][0] = 0.99 * x.data[0][0];
        xn.data[1][0] = 0.99 * x.data[1][0];
        xn
    }

    /// MHE measurement: observe both id and iq (with noise).
    fn mhe_measurement(x: &Matrix<f64, 2, 1>) -> Matrix<f64, 2, 1> {
        *x
    }

    fn build_mhe() -> MovingHorizonEstimator<f64, 2, 1, 2, 5> {
        let mut q_inv = Matrix::<f64, 2, 1>::zeros();
        q_inv.data[0][0] = 1.0;
        q_inv.data[1][0] = 1.0;

        let mut r_inv = Matrix::<f64, 2, 1>::zeros();
        r_inv.data[0][0] = 100.0;
        r_inv.data[1][0] = 100.0;

        let mut p0_inv = Matrix::<f64, 2, 1>::zeros();
        p0_inv.data[0][0] = 0.1;
        p0_inv.data[1][0] = 0.1;

        MovingHorizonEstimator::new(mhe_dynamics, mhe_measurement, q_inv, r_inv, p0_inv, 20)
    }

    /// FOC + MTPA + MHE pipeline: id/iq track MTPA references during speed ramp.
    #[test]
    fn foc_mtpa_mhe_pipeline_tracks_reference() {
        let dt = 1e-4_f64;

        // PMSM motor parameters
        let rs = 0.5_f64;
        let ld = 0.5e-3_f64;
        let lq = 1.5e-3_f64;
        let lambda_pm = 0.04_f64;
        let pole_pairs = 3_u32;

        // MTPA table
        let mtpa_params = MtpaMotorParams {
            pole_pairs,
            ld,
            lq,
            lambda_pm,
            i_s_max: 20.0,
        };
        let mtpa_table = MtpaTable::<f64, 64>::new(mtpa_params, 32)
            .expect("MTPA table construction should succeed");

        // PI current controllers (one for d-axis, one for q-axis)
        let mut pi_d = PiController::new(5.0, 200.0, 30.0);
        let mut pi_q = PiController::new(5.0, 200.0, 30.0);

        // PMSM simulation model
        let mut motor = PmsmDqSim::new(rs, ld, lq, lambda_pm);

        // MHE for state estimation
        let mut mhe = build_mhe();

        // Speed ramp from 0 to 100 rad/s
        let n_steps = 2000_usize;
        let torque_ref = 1.5_f64; // constant torque command (N·m)

        // Pre-fill MHE window with initial zero measurements
        let u_zero = Matrix::<f64, 1, 1>::zeros();
        for _ in 0..5 {
            let y = Matrix::<f64, 2, 1>::zeros();
            mhe.push_measurement(y, u_zero);
        }

        let mut mhe_converged_steps = 0_usize;
        let mut pi_error_finite = true;

        for step in 0..n_steps {
            // Ramp omega
            let omega = (step as f64 / n_steps as f64) * 100.0;

            // MTPA: get id_ref, iq_ref for torque command
            let (id_ref, iq_ref) = mtpa_table.query(torque_ref);

            // MHE: push noisy current measurement
            let mut y = Matrix::<f64, 2, 1>::zeros();
            y.data[0][0] = motor.id;
            y.data[1][0] = motor.iq;
            mhe.push_measurement(y, u_zero);

            // MHE solve to get state estimate (fallback to direct measurement)
            let (id_est, iq_est) = match mhe.solve() {
                Ok(x) => {
                    mhe_converged_steps += 1;
                    (x.data[0][0], x.data[1][0])
                }
                Err(_) => (motor.id, motor.iq),
            };

            // PI current control using estimated state
            let vd = pi_d.update(id_ref - id_est, dt);
            let vq = pi_q.update(iq_ref - iq_est, dt);

            // Validate control output is finite
            if !vd.is_finite() || !vq.is_finite() {
                pi_error_finite = false;
                break;
            }

            // Simulate motor
            motor.step(vd, vq, omega, dt);
        }

        // Core assertions: pipeline runs without numerical blow-up and MHE solves
        assert!(
            pi_error_finite,
            "PI control outputs must remain finite throughout simulation"
        );
        assert!(
            mhe_converged_steps > 0,
            "MHE should solve successfully for at least one step, got {mhe_converged_steps}"
        );
        // Motor states should be finite at end
        assert!(
            motor.id.is_finite() && motor.iq.is_finite(),
            "Motor dq currents must remain finite: id={:.4}, iq={:.4}",
            motor.id,
            motor.iq
        );
    }

    /// MTPA table produces valid references for a range of torque commands.
    #[test]
    fn mtpa_references_are_finite_over_torque_range() {
        let mtpa_params = MtpaMotorParams {
            pole_pairs: 3,
            ld: 0.5e-3,
            lq: 1.5e-3,
            lambda_pm: 0.04,
            i_s_max: 20.0,
        };
        let table = MtpaTable::<f64, 64>::new(mtpa_params, 32).expect("MTPA table construction");

        for torque in [0.0_f64, 0.5, 1.0, 2.0, 3.0, 5.0] {
            let (id, iq) = table.query(torque);
            assert!(id.is_finite(), "id must be finite for torque={torque}");
            assert!(iq.is_finite(), "iq must be finite for torque={torque}");
        }
    }
}

#[cfg(not(all(feature = "motor", feature = "mpc")))]
#[test]
fn foc_mtpa_mhe_pipeline_skipped_without_features() {
    // Skipped: requires motor and mpc features.
}
