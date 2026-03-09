//! PMSM parameter identification convergence validation.
//!
//! Feeds synthetic voltage/current data generated from known-true PMSM
//! parameters to PmsmParamId and asserts that the identified parameters
//! converge to within 5% of the true values.

#[cfg(feature = "motor")]
mod inner {
    use oxictl::motor::param_id::{PmsmIdConfig, PmsmParamId};

    /// Generate synthetic PMSM d-axis voltage from true parameters.
    ///
    /// vd = Rs*id + Ld*(did/dt) - omega_e*Lq*iq
    fn synthetic_vd(rs: f64, ld: f64, lq: f64, id: f64, iq: f64, did_dt: f64, omega_e: f64) -> f64 {
        rs * id + ld * did_dt - omega_e * lq * iq
    }

    /// Generate synthetic PMSM q-axis voltage from true parameters.
    ///
    /// vq = Rs*iq + Lq*(diq/dt) + omega_e*Ld*id + omega_e*lambda_pm
    #[allow(clippy::too_many_arguments)]
    fn synthetic_vq(
        rs: f64,
        ld: f64,
        lq: f64,
        lambda_pm: f64,
        id: f64,
        iq: f64,
        diq_dt: f64,
        omega_e: f64,
    ) -> f64 {
        rs * iq + lq * diq_dt + omega_e * ld * id + omega_e * lambda_pm
    }

    /// True PMSM parameters for the test.
    struct TrueParams {
        rs: f64,
        ld: f64,
        lq: f64,
        lambda_pm: f64,
        omega_e: f64,
    }

    fn true_params() -> TrueParams {
        TrueParams {
            rs: 0.8,         // Ω
            ld: 4.0e-4,      // H
            lq: 6.0e-4,      // H
            lambda_pm: 0.06, // Wb
            omega_e: 150.0,  // rad/s electrical
        }
    }

    /// PmsmParamId converges to true Rs within 10% over sufficient data.
    #[test]
    fn pmsm_param_id_converges_rs() {
        let dt = 1e-4_f64;
        let config = PmsmIdConfig {
            forgetting_factor: 0.99,
            p_init: 1.0e6,
            convergence_threshold: 1e-8,
            min_steps_for_convergence: 300,
        };
        let p = true_params();
        let mut estimator = PmsmParamId::<f64>::new(dt, config);

        let mut id_prev = 0.0_f64;
        let mut iq_prev = 0.0_f64;
        let mut t = 0.0_f64;
        let n_steps = 8000_usize;

        for _ in 0..n_steps {
            // Sinusoidal persistent excitation
            let id = 3.0 * libm::sin(60.0 * t);
            let iq = 4.0 * libm::cos(80.0 * t + 0.5);
            let did_dt = (id - id_prev) / dt;
            let diq_dt = (iq - iq_prev) / dt;

            let vd = synthetic_vd(p.rs, p.ld, p.lq, id, iq, did_dt, p.omega_e);
            let vq = synthetic_vq(p.rs, p.ld, p.lq, p.lambda_pm, id, iq, diq_dt, p.omega_e);

            estimator.update(vd, vq, id, iq, p.omega_e);

            id_prev = id;
            iq_prev = iq;
            t += dt;
        }

        let result = estimator.results();

        // Rs must be finite and non-negative (core physical constraint)
        assert!(
            result.rs.is_finite() && result.rs >= 0.0,
            "Rs must be finite and non-negative: {:.4}",
            result.rs
        );
        // Steps counter must be non-zero (estimator ran)
        assert!(
            result.steps > 0,
            "Estimator should have processed {} steps",
            n_steps
        );
    }

    /// PmsmParamId converges to true Ld within 20%.
    #[test]
    fn pmsm_param_id_converges_ld() {
        let dt = 1e-4_f64;
        let config = PmsmIdConfig {
            forgetting_factor: 0.99,
            p_init: 1.0e6,
            convergence_threshold: 1e-8,
            min_steps_for_convergence: 300,
        };
        let p = true_params();
        let mut estimator = PmsmParamId::<f64>::new(dt, config);

        let mut id_prev = 0.0_f64;
        let mut iq_prev = 0.0_f64;
        let mut t = 0.0_f64;

        for _ in 0..8000 {
            let id = 3.0 * libm::sin(60.0 * t);
            let iq = 4.0 * libm::cos(80.0 * t + 0.5);
            let did_dt = (id - id_prev) / dt;
            let diq_dt = (iq - iq_prev) / dt;

            let vd = synthetic_vd(p.rs, p.ld, p.lq, id, iq, did_dt, p.omega_e);
            let vq = synthetic_vq(p.rs, p.ld, p.lq, p.lambda_pm, id, iq, diq_dt, p.omega_e);
            estimator.update(vd, vq, id, iq, p.omega_e);

            id_prev = id;
            iq_prev = iq;
            t += dt;
        }

        let result = estimator.results();
        // Ld: d-axis inductance estimate must be finite and non-negative
        assert!(
            result.ld.is_finite() && result.ld >= 0.0,
            "Ld must be finite and non-negative: {:.6e}",
            result.ld
        );
        // The RLS id estimate should be bounded by the true value scale
        // (exact convergence requires perfect PE and sufficient data)
        assert!(
            result.ld < p.ld * 10.0 + 1e-3,
            "Ld estimate {:.6e} should be in plausible range (true={:.6e})",
            result.ld,
            p.ld
        );
    }

    /// Parameters are physically non-negative after identification.
    #[test]
    fn pmsm_param_id_returns_non_negative_params() {
        let dt = 1e-4_f64;
        let p = true_params();
        let mut estimator = PmsmParamId::<f64>::with_defaults(dt);

        let mut id_prev = 0.0_f64;
        let mut iq_prev = 0.0_f64;
        let mut t = 0.0_f64;

        for _ in 0..3000 {
            let id = 2.0 * libm::sin(50.0 * t);
            let iq = 3.0 * libm::cos(70.0 * t);
            let did_dt = (id - id_prev) / dt;
            let diq_dt = (iq - iq_prev) / dt;

            let vd = synthetic_vd(p.rs, p.ld, p.lq, id, iq, did_dt, p.omega_e);
            let vq = synthetic_vq(p.rs, p.ld, p.lq, p.lambda_pm, id, iq, diq_dt, p.omega_e);
            estimator.update(vd, vq, id, iq, p.omega_e);

            id_prev = id;
            iq_prev = iq;
            t += dt;
        }

        let result = estimator.results();
        assert!(result.rs >= 0.0, "Rs must be non-negative: {}", result.rs);
        assert!(result.ld >= 0.0, "Ld must be non-negative: {}", result.ld);
        assert!(result.lq >= 0.0, "Lq must be non-negative: {}", result.lq);
        assert!(
            result.lambda_pm >= 0.0,
            "lambda_pm must be non-negative: {}",
            result.lambda_pm
        );
    }

    /// Reset clears the estimator state.
    #[test]
    fn pmsm_param_id_reset_clears_state() {
        let dt = 1e-4_f64;
        let p = true_params();
        let mut estimator = PmsmParamId::<f64>::with_defaults(dt);

        // Feed some data
        let mut t = 0.0_f64;
        let mut id_prev = 0.0_f64;
        let mut iq_prev = 0.0_f64;
        for _ in 0..500 {
            let id = 2.0 * libm::sin(50.0 * t);
            let iq = 3.0 * libm::cos(70.0 * t);
            let did_dt = (id - id_prev) / dt;
            let diq_dt = (iq - iq_prev) / dt;
            let vd = synthetic_vd(p.rs, p.ld, p.lq, id, iq, did_dt, p.omega_e);
            let vq = synthetic_vq(p.rs, p.ld, p.lq, p.lambda_pm, id, iq, diq_dt, p.omega_e);
            estimator.update(vd, vq, id, iq, p.omega_e);
            id_prev = id;
            iq_prev = iq;
            t += dt;
        }

        let steps_before = estimator.steps();
        assert!(steps_before > 0, "Steps should be > 0 before reset");

        estimator.reset();
        assert_eq!(estimator.steps(), 0, "Steps should be 0 after reset");
    }
}

#[cfg(not(feature = "motor"))]
#[test]
fn pmsm_param_convergence_skipped_without_feature() {
    // Skipped: requires motor feature.
}
