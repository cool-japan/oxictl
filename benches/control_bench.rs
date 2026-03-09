//! Comprehensive criterion benchmark for oxictl.
//!
//! Covers: PID controller, Kalman filter, signal processing filters,
//! and Bezier trajectory evaluation.  Each module is grouped under its
//! own `BenchmarkGroup` so reports are neatly organised.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

// ─────────────────────────────────────────────────────────────
//  PID benchmarks
// ─────────────────────────────────────────────────────────────

fn bench_pid(c: &mut Criterion) {
    use oxictl::core::signal::{Feedback, Setpoint};
    use oxictl::core::traits::Controller;
    use oxictl::pid::standard::PidConfig;

    let mut group = c.benchmark_group("pid");

    // Single PID update: measures raw throughput of one controller step.
    {
        let mut pid = PidConfig::pid(2.0_f64, 0.5, 0.05).build();
        let sp = Setpoint::new(1.0_f64);
        let fb = Feedback::new(0.5_f64);
        let dt = 0.001_f64;

        group.bench_function("standard_update", |b| {
            b.iter(|| pid.update(black_box(&sp), black_box(&fb), black_box(dt)));
        });
    }

    // 1 kHz closed-loop simulation: 1000 steps of PID + first-order plant.
    {
        let mut pid = PidConfig::pid(2.0_f64, 0.5, 0.02)
            .with_limits(-10.0, 10.0)
            .build();
        let dt = 0.001_f64;

        group.bench_function("1khz_loop", |b| {
            b.iter(|| {
                let mut y = 0.0_f64;
                for _ in 0..1000 {
                    let sp = Setpoint::new(black_box(1.0_f64));
                    let fb = Feedback::new(y);
                    let u = pid.update(&sp, &fb, dt).value();
                    // First-order plant: dy/dt = (u - y) / tau, tau = 0.5 s
                    y += (u - y) * dt * 2.0;
                }
                y
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────
//  Kalman filter benchmarks
// ─────────────────────────────────────────────────────────────

fn bench_kf(c: &mut Criterion) {
    use oxictl::core::matrix::Matrix;
    use oxictl::estimator::kalman::KalmanFilter;

    let mut group = c.benchmark_group("kalman");

    // Single predict + update cycle on a 2-state position-velocity tracker.
    {
        let dt = 0.01_f64;
        let mut kf = KalmanFilter::<f64, 2, 1, 1>::position_velocity(dt, 1.0, 0.5);
        let u = [0.0_f64];
        let z = [1.0_f64];

        group.bench_function("predict_update_2x1", |b| {
            b.iter(|| {
                kf.predict(black_box(&u));
                kf.update(black_box(&z))
            });
        });
    }

    // 100-step tracking loop: position-velocity estimation of constant-velocity target.
    {
        let dt = 0.01_f64;
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
        let mut kf = KalmanFilter::new(a, b, h, q, r, [0.0_f64, 0.0], p0);

        group.bench_function("position_velocity_100_steps", |b| {
            b.iter(|| {
                let v_true = black_box(2.0_f64);
                let mut x_true = 0.0_f64;
                for step in 0_u32..100 {
                    x_true += v_true * dt;
                    // Deterministic pseudo-noise using a sine wave
                    let noise = 0.01 * libm::sin(step as f64 * 0.3);
                    let z = [x_true + noise];
                    kf.predict(&[0.0_f64]);
                    kf.update(&z);
                }
                kf.state()[0]
            });
        });
    }

    // Parameterised benchmark: vary measurement noise R ∈ {0.01, 0.1, 1.0}
    {
        let dt = 0.01_f64;
        for &r_val in &[0.01_f64, 0.1, 1.0] {
            let mut kf = KalmanFilter::<f64, 2, 1, 1>::position_velocity(dt, 1.0, r_val.sqrt());
            let u = [0.0_f64];
            let z = [1.0_f64];

            group.bench_with_input(
                BenchmarkId::new("predict_update_r", r_val),
                &r_val,
                |b, _| {
                    b.iter(|| {
                        kf.predict(black_box(&u));
                        kf.update(black_box(&z))
                    });
                },
            );
        }
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────
//  Signal processing filter benchmarks
// ─────────────────────────────────────────────────────────────

fn bench_filter(c: &mut Criterion) {
    use oxictl::core::filters::butterworth::design_butterworth_lp;
    use oxictl::core::filters::moving_average::MovingAverage;

    let mut group = c.benchmark_group("filter");

    // 4th-order Butterworth lowpass — single sample throughput.
    {
        let mut filt = design_butterworth_lp::<f64, 4>(100.0_f64, 1000.0_f64)
            .expect("butterworth_lp design must succeed");

        group.bench_function("butterworth_lp4_sample", |b| {
            b.iter(|| filt.update(black_box(0.5_f64)));
        });
    }

    // 4th-order Butterworth — 1000-sample block throughput.
    {
        let mut filt = design_butterworth_lp::<f64, 4>(100.0_f64, 1000.0_f64)
            .expect("butterworth_lp design must succeed");

        group.bench_function("butterworth_lp4_block_1000", |b| {
            b.iter(|| {
                let mut acc = 0.0_f64;
                for k in 0_u32..1000 {
                    // Deterministic test signal: sum of two sinusoids
                    let x = libm::sin(k as f64 * 0.1) + 0.3 * libm::sin(k as f64 * 2.5);
                    acc += filt.update(black_box(x));
                }
                acc
            });
        });
    }

    // Moving average, window N=16 — single sample.
    {
        let mut ma = MovingAverage::<f64, 16>::new();

        group.bench_function("moving_average_16_sample", |b| {
            b.iter(|| ma.update(black_box(0.5_f64)));
        });
    }

    // Moving average, window N=16 — 1000-sample block.
    {
        let mut ma = MovingAverage::<f64, 16>::new();

        group.bench_function("moving_average_16_block_1000", |b| {
            b.iter(|| {
                let mut acc = 0.0_f64;
                for k in 0_u32..1000 {
                    let x = libm::sin(k as f64 * 0.05);
                    acc += ma.update(black_box(x));
                }
                acc
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────
//  Bezier trajectory benchmarks
// ─────────────────────────────────────────────────────────────

fn bench_trajectory(c: &mut Criterion) {
    use oxictl::trajectory::bezier::BezierCurve;

    let mut group = c.benchmark_group("trajectory");

    let curve = BezierCurve::<f64, 3>::new(
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.5],
        [3.0, 2.0, 1.0],
        [4.0, 0.0, 0.0],
    );

    // Single point evaluation at a fixed t.
    group.bench_function("bezier3d_eval", |b| {
        b.iter(|| curve.evaluate(black_box(0.5_f64)));
    });

    // First-derivative evaluation at a fixed t.
    group.bench_function("bezier3d_derivative", |b| {
        b.iter(|| curve.derivative(black_box(0.5_f64)));
    });

    // Path sampling: evaluate at 100 uniformly spaced t values.
    // This models a motion planner sampling the trajectory for interpolation.
    group.bench_function("bezier3d_eval_sweep_100", |b| {
        b.iter(|| {
            let n = 100_u32;
            let mut last = [0.0_f64; 3];
            for i in 0..=n {
                let t = i as f64 / n as f64;
                last = curve.evaluate(black_box(t));
            }
            last
        });
    });

    // Parameterised: compare 3D vs 2D curve eval cost.
    let curve2d = BezierCurve::<f64, 2>::new([0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]);
    for (label, n_pts) in [("sweep_50", 50_u32), ("sweep_200", 200_u32)] {
        group.bench_with_input(
            BenchmarkId::new("bezier2d_eval_sweep", label),
            &n_pts,
            |b, &n| {
                b.iter(|| {
                    let mut last = [0.0_f64; 2];
                    for i in 0..=n {
                        let t = i as f64 / n as f64;
                        last = curve2d.evaluate(black_box(t));
                    }
                    last
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────
//  Wiring
// ─────────────────────────────────────────────────────────────

criterion_group!(benches, bench_pid, bench_kf, bench_filter, bench_trajectory);
criterion_main!(benches);
