//! Property-based tests for estimator invariants.

use oxictl::core::matrix::Matrix;
use oxictl::estimator::kalman::KalmanFilter;
use proptest::prelude::*;

/// Build a stable 2-state, 1-input, 1-output Kalman filter.
fn make_kalman(q_scale: f64, r_scale: f64) -> KalmanFilter<f64, 2, 1, 1> {
    let a = Matrix::<f64, 2, 2> {
        data: [[0.9, 0.1], [0.0, 0.8]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.1], [0.0]],
    };
    let c = Matrix::<f64, 1, 2> { data: [[1.0, 0.0]] };
    let q = Matrix::<f64, 2, 2> {
        data: [[q_scale, 0.0], [0.0, q_scale]],
    };
    let r = Matrix::<f64, 1, 1> { data: [[r_scale]] };
    let p0 = Matrix::<f64, 2, 2>::identity();
    KalmanFilter::new(a, b, c, q, r, [0.0, 0.0], p0)
}

proptest! {
    /// Kalman: state estimate is finite after predict+update.
    #[test]
    fn kalman_state_finite_after_update(
        q_scale in 1e-4f64..1.0,
        r_scale in 1e-3f64..10.0,
        z_val in -10.0f64..10.0,
        u_val in -5.0f64..5.0,
        n_steps in 1usize..20,
    ) {
        let mut kf = make_kalman(q_scale, r_scale);
        for _ in 0..n_steps {
            kf.predict(&[u_val]);
            kf.update(&[z_val]);
        }
        let x = kf.state();
        prop_assert!(x[0].is_finite() && x[1].is_finite(),
            "state became non-finite: {:?}", x);
    }

    /// Kalman: covariance diagonal stays non-negative after predict.
    #[test]
    fn kalman_covariance_diagonal_nonnegative(
        q_scale in 1e-4f64..1.0,
        r_scale in 1e-3f64..10.0,
        n_steps in 1usize..30,
    ) {
        let mut kf = make_kalman(q_scale, r_scale);
        for _ in 0..n_steps {
            kf.predict(&[0.0]);
            kf.update(&[0.0]);
            let p = kf.covariance();
            prop_assert!(p.data[0][0] >= -1e-12, "P[0][0]={}", p.data[0][0]);
            prop_assert!(p.data[1][1] >= -1e-12, "P[1][1]={}", p.data[1][1]);
        }
    }

    /// Kalman: reset restores initial conditions.
    #[test]
    fn kalman_reset_restores_initial(
        q_scale in 1e-4f64..1.0,
        r_scale in 1e-3f64..10.0,
        z_val in -5.0f64..5.0,
    ) {
        let mut kf = make_kalman(q_scale, r_scale);
        // Run for several steps
        for _ in 0..10 {
            kf.predict(&[1.0]);
            kf.update(&[z_val]);
        }
        // Reset to initial
        let p0 = Matrix::<f64, 2, 2>::identity();
        kf.reset([0.0, 0.0], p0);
        let x = kf.state();
        prop_assert!(x[0].abs() < 1e-12 && x[1].abs() < 1e-12);
        let p = kf.covariance();
        prop_assert!((p.data[0][0] - 1.0).abs() < 1e-12);
        prop_assert!((p.data[1][1] - 1.0).abs() < 1e-12);
    }

    /// Kalman: zero input and zero measurement → state converges toward zero.
    #[test]
    fn kalman_zero_input_converges(
        q_scale in 1e-6f64..0.01,
        r_scale in 0.01f64..1.0,
        x0 in -5.0f64..5.0,
        x1 in -5.0f64..5.0,
    ) {
        let mut kf = make_kalman(q_scale, r_scale);
        let p0 = Matrix::<f64, 2, 2>::identity();
        kf.reset([x0, x1], p0);
        let initial_norm = x0 * x0 + x1 * x1;
        prop_assume!(initial_norm > 0.1);
        for _ in 0..50 {
            kf.predict(&[0.0]);
            kf.update(&[0.0]);
        }
        let x = kf.state();
        let final_norm = x[0] * x[0] + x[1] * x[1];
        // Should converge significantly
        prop_assert!(final_norm < initial_norm,
            "final_norm={final_norm} should be less than initial_norm={initial_norm}");
    }
}
