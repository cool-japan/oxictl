//! Benchmark: state feedback (LQR/LQG/pole placement) throughput.

use criterion::{criterion_group, criterion_main, Criterion};
use oxictl::core::matrix::Matrix;
use oxictl::state_feedback::lqr::{solve_dare, Lqr};
use oxictl::state_feedback::pole_placement::ackermann;

fn dare_solve_2x2(c: &mut Criterion) {
    let a = Matrix::<f64, 2, 2> {
        data: [[0.9, 0.1], [0.0, 0.8]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.1], [0.2]],
    };
    let q = Matrix::<f64, 2, 2> {
        data: [[1.0, 0.0], [0.0, 1.0]],
    };
    let r = Matrix::<f64, 1, 1> { data: [[0.1]] };

    c.bench_function("dare_solve_2x2", |b_| {
        b_.iter(|| solve_dare(&a, &b, &q, &r, 1000, 1e-10f64));
    });
}

fn dare_solve_4x1(c: &mut Criterion) {
    let a = Matrix::<f64, 4, 4> {
        data: [
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };
    let b = Matrix::<f64, 4, 1> {
        data: [[0.0], [0.1], [0.0], [0.1]],
    };
    let q = Matrix::<f64, 4, 4>::identity();
    let r = Matrix::<f64, 1, 1> { data: [[0.01]] };

    c.bench_function("dare_solve_4x1", |b_| {
        b_.iter(|| solve_dare(&a, &b, &q, &r, 1000, 1e-10f64));
    });
}

fn lqr_compute_2x1(c: &mut Criterion) {
    let a = Matrix::<f64, 2, 2> {
        data: [[0.9, 0.1], [0.0, 0.8]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.1], [0.2]],
    };
    let q = Matrix::<f64, 2, 2>::identity();
    let r = Matrix::<f64, 1, 1> { data: [[0.1]] };
    let lqr = Lqr::design(&a, &b, &q, &r).unwrap();

    c.bench_function("lqr_control_2x1", |b_| {
        b_.iter(|| {
            let x = [0.5f64, -0.3];
            lqr.control(&x, &[0.0, 0.0])
        });
    });
}

fn pole_placement_2x1(c: &mut Criterion) {
    let a = Matrix::<f64, 2, 2> {
        data: [[0.0, 1.0], [-2.0, -3.0]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.0], [1.0]],
    };
    let poles = [0.5f64, 0.6];

    c.bench_function("pole_placement_2x1", |b_| {
        b_.iter(|| ackermann(&a, &b, &poles));
    });
}

fn lqr_closed_loop_100_steps(c: &mut Criterion) {
    let a = Matrix::<f64, 2, 2> {
        data: [[1.0, 0.01], [0.0, 1.0]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.00005], [0.01]],
    };
    let q = Matrix::<f64, 2, 2> {
        data: [[10.0, 0.0], [0.0, 1.0]],
    };
    let r = Matrix::<f64, 1, 1> { data: [[0.001]] };
    let lqr = Lqr::design(&a, &b, &q, &r).unwrap();

    c.bench_function("lqr_closed_loop_100_steps", |b_| {
        b_.iter(|| {
            let mut x = [1.0f64, 0.0];
            for _ in 0..100 {
                let u = lqr.control(&x, &[0.0, 0.0]);
                // Euler integration
                let ax0 = 1.0 * x[0] + 0.01 * x[1] + 0.00005 * u[0];
                let ax1 = 0.01 * u[0] + x[1];
                x = [ax0, ax1];
            }
            x[0]
        });
    });
}

criterion_group!(
    benches,
    dare_solve_2x2,
    dare_solve_4x1,
    lqr_compute_2x1,
    pole_placement_2x1,
    lqr_closed_loop_100_steps,
);
criterion_main!(benches);
