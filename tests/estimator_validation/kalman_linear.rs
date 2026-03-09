//! Linear Kalman filter accuracy validation.

use oxictl::core::matrix::Matrix;
use oxictl::estimator::kalman::KalmanFilter;

/// Position-velocity tracker: KF tracks linear motion to within noise level.
#[test]
fn kalman_tracks_constant_velocity() {
    let dt = 0.01f64;
    let a = Matrix {
        data: [[1.0, dt], [0.0, 1.0]],
    };
    let b = Matrix::<f64, 2, 1>::zeros();
    let h = Matrix {
        data: [[1.0, 0.0f64]],
    };
    let q = Matrix {
        data: [[1e-6, 0.0], [0.0, 1e-5f64]],
    };
    let r = Matrix { data: [[0.01f64]] };
    let p0 = Matrix {
        data: [[1.0, 0.0], [0.0, 1.0f64]],
    };

    let mut kf = KalmanFilter::new(a, b, h, q, r, [0.0f64, 1.0], p0);

    let v_true = 1.0f64; // 1 m/s
    let mut x_true = 0.0f64;

    let mut rmse_sum = 0.0f64;
    for step in 0..200usize {
        x_true += v_true * dt;
        let z = [x_true];
        kf.predict(&[0.0f64]);
        kf.update(&z);

        if step > 50 {
            let err = kf.state()[0] - x_true;
            rmse_sum += err * err;
        }
    }
    let rmse = (rmse_sum / 150.0).sqrt();
    assert!(
        rmse < 0.01,
        "KF tracking RMSE should be < 0.01m: {:.6}",
        rmse
    );
}

/// Kalman filter covariance decreases (filter gains confidence over time).
#[test]
fn kalman_covariance_decreases() {
    let dt = 0.05f64;
    let a = Matrix {
        data: [[1.0, dt], [0.0, 1.0]],
    };
    let b = Matrix::<f64, 2, 1>::zeros();
    let h = Matrix {
        data: [[1.0, 0.0f64]],
    };
    let q = Matrix {
        data: [[1e-4, 0.0], [0.0, 1e-3f64]],
    };
    let r = Matrix { data: [[0.1f64]] };
    let p0 = Matrix {
        data: [[10.0, 0.0], [0.0, 10.0f64]],
    };

    let mut kf = KalmanFilter::new(a, b, h, q, r, [0.0f64, 0.0], p0);
    let p_init = kf.covariance().data[0][0];

    for i in 0..50 {
        let z = [(i as f64) * 0.05f64];
        kf.predict(&[0.0f64]);
        kf.update(&z);
    }

    let p_final = kf.covariance().data[0][0];
    assert!(
        p_final < p_init,
        "Covariance should decrease: init={:.4}, final={:.4}",
        p_init,
        p_final
    );
}

/// 4-state 2D position tracker.
#[test]
fn kalman_2d_position_tracker() {
    let dt = 0.1f64;
    let a = Matrix {
        data: [
            [1.0f64, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };
    let b = Matrix::<f64, 4, 1>::zeros();
    let h = Matrix {
        data: [[1.0f64, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    };
    let q = Matrix {
        data: [
            [1e-3f64, 0.0, 0.0, 0.0],
            [0.0, 1e-3, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1],
        ],
    };
    let r = Matrix {
        data: [[0.04f64, 0.0], [0.0, 0.04]],
    };
    let p0 = Matrix::<f64, 4, 4>::identity();

    let mut kf = KalmanFilter::new(a, b, h, q, r, [0.0f64; 4], p0);

    let omega = 0.5f64;
    let mut t = 0.0f64;
    let mut rmse_sum = 0.0f64;
    let n_steps = 200usize;

    for step in 0..n_steps {
        t += dt;
        let x_true = 2.0 * (omega * t).cos();
        let y_true = 2.0 * (omega * t).sin();

        kf.predict(&[0.0f64]);
        kf.update(&[x_true, y_true]);

        if step > 50 {
            let ex = kf.state()[0] - x_true;
            let ey = kf.state()[1] - y_true;
            rmse_sum += ex * ex + ey * ey;
        }
    }

    let rmse = (rmse_sum / (n_steps - 50) as f64).sqrt();
    assert!(rmse < 0.3, "2D tracking RMSE={:.6} should be < 0.3", rmse);
}

/// Kalman with wrong initial estimate converges to true state.
#[test]
fn kalman_converges_from_wrong_init() {
    let a = Matrix { data: [[1.0f64]] };
    let b = Matrix::<f64, 1, 1>::zeros();
    let h = Matrix { data: [[1.0f64]] };
    let q = Matrix { data: [[1e-5f64]] };
    let r = Matrix { data: [[0.01f64]] };
    let p0 = Matrix { data: [[100.0f64]] }; // High initial uncertainty

    let x_true = 5.0f64;
    let mut kf = KalmanFilter::new(a, b, h, q, r, [0.0f64], p0); // Wrong init: 0 vs 5

    for _ in 0..100 {
        kf.predict(&[0.0f64]);
        kf.update(&[x_true]);
    }

    assert!(
        (kf.state()[0] - x_true).abs() < 0.01,
        "Filter should converge: est={:.4}, true={}",
        kf.state()[0],
        x_true
    );
}
