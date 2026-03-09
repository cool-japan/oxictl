//! Information Filter multi-sensor fusion validation.
//!
//! Fuses 3 sensors with different noise levels using the Information Filter.
//! The fused estimate covariance should be lower than any individual sensor's
//! covariance, demonstrating the benefit of sensor fusion.

#[cfg(feature = "estimator")]
mod inner {
    use oxictl::core::matrix::Matrix;
    use oxictl::estimator::information_filter::InformationFilter;

    /// Build an Information Filter for a 2D position tracking system.
    ///
    /// System: static position (A=I, B=0) with measurement noise.
    fn build_info_filter(r_diag: f64) -> InformationFilter<f64, 2, 2, 1> {
        // Static model: x[k+1] = x[k]
        let a = Matrix::<f64, 2, 2>::identity();
        let b = Matrix::<f64, 2, 1>::zeros();
        // Full state measurement: H = I
        let h = Matrix::<f64, 2, 2>::identity();
        // Small process noise (system is nearly static)
        let q = Matrix::<f64, 2, 2>::identity().scale(1e-6);
        // Measurement noise covariance
        let r = Matrix::<f64, 2, 2>::identity().scale(r_diag);

        // Initial state = zero, high uncertainty
        let x0 = [0.0_f64; 2];
        let p0 = Matrix::<f64, 2, 2>::identity().scale(100.0);

        InformationFilter::new(a, b, h, q, r, x0, p0)
            .expect("InformationFilter construction should succeed for PD p0")
    }

    /// Fused covariance is lower than any single sensor's covariance.
    #[test]
    fn fused_covariance_lower_than_individual_sensors() {
        // Three sensors with different noise levels (variances: 1.0, 4.0, 9.0)
        let r_sensor1 = 1.0_f64;
        let r_sensor2 = 4.0_f64;
        let r_sensor3 = 9.0_f64;

        let true_position = [3.0_f64, 1.5_f64];
        let n_steps = 100_usize;

        // Run each sensor individually
        let mut f1 = build_info_filter(r_sensor1);
        let mut f2 = build_info_filter(r_sensor2);
        let mut f3 = build_info_filter(r_sensor3);

        for _ in 0..n_steps {
            let z = [true_position[0], true_position[1]];
            f1.predict(&[0.0]).expect("f1 predict");
            f1.update(&z).expect("f1 update");

            f2.predict(&[0.0]).expect("f2 predict");
            f2.update(&z).expect("f2 update");

            f3.predict(&[0.0]).expect("f3 predict");
            f3.update(&z).expect("f3 update");
        }

        let cov1 = f1.covariance().expect("f1 covariance should be invertible");
        let cov2 = f2.covariance().expect("f2 covariance should be invertible");
        let cov3 = f3.covariance().expect("f3 covariance should be invertible");

        // Trace of covariance = total variance
        let trace1 = cov1.data[0][0] + cov1.data[1][1];
        let trace2 = cov2.data[0][0] + cov2.data[1][1];
        let trace3 = cov3.data[0][0] + cov3.data[1][1];

        // Build a fused filter using all 3 sensors simultaneously
        // We use the information form's additive property:
        // Omega_fused = Omega1 + Omega2 + Omega3 - 2*Omega_prior (to avoid double counting prior)
        // Simpler: run one filter that receives all 3 measurements per step
        let _f_fused_initial = build_info_filter(r_sensor1); // built but replaced below

        // Override: use sensor 1 filter but add sensors 2 and 3 via extra update calls
        // Reset and use manually constructed fusion
        let a = Matrix::<f64, 2, 2>::identity();
        let b = Matrix::<f64, 2, 1>::zeros();
        let h = Matrix::<f64, 2, 2>::identity();
        let q = Matrix::<f64, 2, 2>::identity().scale(1e-6);
        let p0 = Matrix::<f64, 2, 2>::identity().scale(100.0);
        let x0 = [0.0_f64; 2];

        // Use sensor 1 as primary, then additionally fuse sensor 2 and 3 measurements
        let r1 = Matrix::<f64, 2, 2>::identity().scale(r_sensor1);
        let mut f_fused =
            InformationFilter::new(a, b, h, q, r1, x0, p0).expect("Fused filter construction");

        for _ in 0..n_steps {
            let z = [true_position[0], true_position[1]];
            f_fused.predict(&[0.0]).expect("fused predict");
            // Update with sensor 1
            f_fused.update(&z).expect("fused update s1");
            // Manually add information from sensors 2 and 3 via fuse_sensors
            // fuse_sensors takes (H, R_inv, z) tuples
            let r2_inv = Matrix::<f64, 2, 2>::identity().scale(1.0 / r_sensor2);
            let r3_inv = Matrix::<f64, 2, 2>::identity().scale(1.0 / r_sensor3);
            f_fused
                .fuse_sensors(&[(h, r2_inv, z), (h, r3_inv, z)])
                .expect("fuse_sensors should succeed");
        }

        let cov_fused = f_fused.covariance().expect("fused covariance");
        let trace_fused = cov_fused.data[0][0] + cov_fused.data[1][1];

        // Fused covariance trace should be smaller than all individual sensor traces
        assert!(
            trace_fused < trace1,
            "Fused trace ({trace_fused:.6}) should be < sensor1 trace ({trace1:.6})"
        );
        assert!(
            trace_fused < trace2,
            "Fused trace ({trace_fused:.6}) should be < sensor2 trace ({trace2:.6})"
        );
        assert!(
            trace_fused < trace3,
            "Fused trace ({trace_fused:.6}) should be < sensor3 trace ({trace3:.6})"
        );
    }

    /// Better sensors (lower noise) yield lower covariance.
    #[test]
    fn lower_noise_sensor_yields_lower_covariance() {
        let true_pos = [1.0_f64, 2.0_f64];
        let n_steps = 50_usize;

        let mut f_good = build_info_filter(0.1); // low noise sensor
        let mut f_bad = build_info_filter(10.0); // high noise sensor

        for _ in 0..n_steps {
            let z = [true_pos[0], true_pos[1]];
            f_good.predict(&[0.0]).expect("good predict");
            f_good.update(&z).expect("good update");
            f_bad.predict(&[0.0]).expect("bad predict");
            f_bad.update(&z).expect("bad update");
        }

        let cov_good = f_good.covariance().expect("good cov");
        let cov_bad = f_bad.covariance().expect("bad cov");

        let trace_good = cov_good.data[0][0] + cov_good.data[1][1];
        let trace_bad = cov_bad.data[0][0] + cov_bad.data[1][1];

        assert!(
            trace_good < trace_bad,
            "Good sensor trace ({trace_good:.6}) should be < bad sensor trace ({trace_bad:.6})"
        );
    }

    /// Information matrix stays symmetric after multiple predict/update cycles.
    #[test]
    fn information_matrix_remains_symmetric() {
        let mut f = build_info_filter(1.0);
        let z = [2.0_f64, 0.5_f64];

        for _ in 0..50 {
            f.predict(&[0.0]).expect("predict");
            f.update(&z).expect("update");
        }

        let omega = f.information_matrix();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (omega.data[i][j] - omega.data[j][i]).abs() < 1e-9,
                    "Information matrix not symmetric at ({i},{j}): {} vs {}",
                    omega.data[i][j],
                    omega.data[j][i]
                );
            }
        }
    }

    /// State estimate converges to true position.
    #[test]
    fn state_converges_to_true_position() {
        let mut f = build_info_filter(0.5);
        let true_pos = [4.0_f64, -1.0_f64];

        for _ in 0..200 {
            f.predict(&[0.0]).expect("predict");
            f.update(&[true_pos[0], true_pos[1]]).expect("update");
        }

        let x_est = f.state().expect("state estimate should be available");
        assert!(
            (x_est[0] - true_pos[0]).abs() < 0.1,
            "Position x0 estimate {:.4} should be near {}",
            x_est[0],
            true_pos[0]
        );
        assert!(
            (x_est[1] - true_pos[1]).abs() < 0.1,
            "Position x1 estimate {:.4} should be near {}",
            x_est[1],
            true_pos[1]
        );
    }
}

#[cfg(not(feature = "estimator"))]
#[test]
fn information_filter_fusion_skipped_without_feature() {
    // Skipped: requires estimator feature.
}
