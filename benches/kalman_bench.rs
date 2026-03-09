//! Benchmark: Kalman filter and EKF update throughput.

use criterion::{criterion_group, criterion_main, Criterion};
use oxictl::core::matrix::Matrix;
use oxictl::estimator::kalman::KalmanFilter;

fn kalman_predict_update_2x1(c: &mut Criterion) {
    // 2-state, 1-measurement, 1-input: position-velocity tracker
    let dt = 0.01f64;
    let a = Matrix::<f64, 2, 2> {
        data: [[1.0, dt], [0.0, 1.0]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.5 * dt * dt], [dt]],
    };
    let h = Matrix::<f64, 1, 2> { data: [[1.0, 0.0]] };
    let q = Matrix::<f64, 2, 2> {
        data: [[1e-4, 0.0], [0.0, 1e-3]],
    };
    let r = Matrix::<f64, 1, 1> { data: [[0.01]] };
    let p0 = Matrix::<f64, 2, 2>::identity();

    let mut kf = KalmanFilter::new(a, b, h, q, r, [0.0f64, 0.0], p0);
    let u = [0.0f64];
    let z = [1.0f64];

    c.bench_function("kalman_predict_update_2x1", |b| {
        b.iter(|| {
            kf.predict(&u);
            kf.update(&z)
        });
    });
}

fn kalman_4x2_tracker(c: &mut Criterion) {
    // 4-state (x,y,vx,vy) position tracker, 2 measurements (x,y)
    let dt = 0.05f64;
    let a = Matrix::<f64, 4, 4> {
        data: [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };
    let b = Matrix::<f64, 4, 1>::zeros();
    let h = Matrix::<f64, 2, 4> {
        data: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    };
    let sigma_a = 0.1f64;
    let q = Matrix::<f64, 4, 4> {
        data: [
            [
                dt.powi(4) / 4.0 * sigma_a,
                0.0,
                dt.powi(3) / 2.0 * sigma_a,
                0.0,
            ],
            [
                0.0,
                dt.powi(4) / 4.0 * sigma_a,
                0.0,
                dt.powi(3) / 2.0 * sigma_a,
            ],
            [dt.powi(3) / 2.0 * sigma_a, 0.0, dt * dt * sigma_a, 0.0],
            [0.0, dt.powi(3) / 2.0 * sigma_a, 0.0, dt * dt * sigma_a],
        ],
    };
    let r = Matrix::<f64, 2, 2> {
        data: [[0.25, 0.0], [0.0, 0.25]],
    };
    let p0 = Matrix::<f64, 4, 4>::identity();

    let mut kf = KalmanFilter::new(a, b, h, q, r, [0.0f64; 4], p0);
    let u = [0.0f64];
    let z = [1.0f64, 0.5];

    c.bench_function("kalman_4x2_predict_update", |b| {
        b.iter(|| {
            kf.predict(&u);
            kf.update(&z)
        });
    });
}

fn kalman_tracking_loop(c: &mut Criterion) {
    // Simulate 200 steps of Kalman tracking
    let dt = 0.01f64;
    let a = Matrix::<f64, 2, 2> {
        data: [[1.0, dt], [0.0, 1.0]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.5 * dt * dt], [dt]],
    };
    let h = Matrix::<f64, 1, 2> { data: [[1.0, 0.0]] };
    let q = Matrix::<f64, 2, 2> {
        data: [[1e-4, 0.0], [0.0, 1e-3]],
    };
    let r = Matrix::<f64, 1, 1> { data: [[0.01]] };
    let p0 = Matrix::<f64, 2, 2>::identity();

    let mut kf = KalmanFilter::new(a, b, h, q, r, [0.0f64, 0.0], p0);

    c.bench_function("kalman_tracking_200_steps", |b| {
        b.iter(|| {
            let mut x_true = 0.0f64;
            let v_true = 1.0f64;
            for step in 0..200 {
                x_true += v_true * dt;
                let z = [x_true + 0.01 * ((step as f64) * 0.1).sin()];
                kf.predict(&[0.0f64]);
                kf.update(&z);
            }
            kf.state()[0]
        });
    });
}

criterion_group!(
    benches,
    kalman_predict_update_2x1,
    kalman_4x2_tracker,
    kalman_tracking_loop
);
criterion_main!(benches);
