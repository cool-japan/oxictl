//! Benchmark: MPC solver throughput.

use criterion::{criterion_group, criterion_main, Criterion};
use oxictl::core::matrix::Matrix;
use oxictl::mpc::linear_mpc::{LinearMpc, MpcConstraints};

fn mpc_2x1_horizon10(c: &mut Criterion) {
    // Double integrator: N=2, I=1, H=10
    // x[k+1] = [[1,dt],[0,1]]*x + [[dt²/2],[dt]]*u
    let dt = 0.1f64;
    let a = Matrix::<f64, 2, 2> {
        data: [[1.0, dt], [0.0, 1.0]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[dt * dt / 2.0], [dt]],
    };
    let q = Matrix::<f64, 2, 2> {
        data: [[1.0, 0.0], [0.0, 0.1]],
    };
    let q_f = Matrix::<f64, 2, 2> {
        data: [[10.0, 0.0], [0.0, 1.0]],
    };
    let r = Matrix::<f64, 1, 1> { data: [[0.01]] };
    let constraints = MpcConstraints::<f64, 1>::box_input([-5.0], [5.0]);

    let mut mpc = LinearMpc::<f64, 2, 1, 10>::new(a, b, q, q_f, r, constraints)
        .with_optimizer(0.01, 50, 1e-4);
    let x0 = [1.0f64, 0.0];
    let xref = [0.0f64, 0.0];

    c.bench_function("mpc_2x1_horizon10_solve", |b| {
        b.iter(|| mpc.solve(&x0, &xref));
    });
}

fn mpc_4x2_horizon5(c: &mut Criterion) {
    // 4-state, 2-input system: N=4, I=2, H=5
    let a = Matrix::<f64, 4, 4> {
        data: [
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.0, 0.9],
        ],
    };
    let b = Matrix::<f64, 4, 2> {
        data: [[0.1, 0.0], [0.0, 0.1], [0.05, 0.05], [0.0, 0.1]],
    };
    let q = Matrix::<f64, 4, 4>::identity();
    let q_f = Matrix::<f64, 4, 4>::identity();
    let r = Matrix::<f64, 2, 2>::identity();
    let constraints = MpcConstraints::<f64, 2>::unconstrained();

    let mut mpc = LinearMpc::<f64, 4, 2, 5>::new(a, b, q, q_f, r, constraints)
        .with_optimizer(0.005, 30, 1e-4);
    let x0 = [1.0f64, 0.5, 0.0, 0.0];
    let xref = [0.0f64; 4];

    c.bench_function("mpc_4x2_horizon5_solve", |b| {
        b.iter(|| mpc.solve(&x0, &xref));
    });
}

fn mpc_closed_loop(c: &mut Criterion) {
    // MPC closed-loop simulation: 100 steps
    let dt = 0.05f64;
    let a = Matrix::<f64, 2, 2> {
        data: [[1.0, dt], [0.0, 1.0]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[dt * dt / 2.0], [dt]],
    };
    let q = Matrix::<f64, 2, 2> {
        data: [[1.0, 0.0], [0.0, 0.1]],
    };
    let q_f = Matrix::<f64, 2, 2> {
        data: [[10.0, 0.0], [0.0, 1.0]],
    };
    let r = Matrix::<f64, 1, 1> { data: [[0.01]] };
    let constraints = MpcConstraints::<f64, 1>::box_input([-2.0], [2.0]);

    let mut mpc =
        LinearMpc::<f64, 2, 1, 8>::new(a, b, q, q_f, r, constraints).with_optimizer(0.01, 30, 1e-4);

    c.bench_function("mpc_closed_loop_100_steps", |b| {
        b.iter(|| {
            let mut x = [1.0f64, 0.0];
            for _ in 0..100 {
                let (u, _) = mpc.solve(&x, &[0.0, 0.0]);
                // Apply u: Euler integration of double integrator
                let xn0 = x[0] + dt * x[1] + 0.5 * dt * dt * u[0];
                let xn1 = x[1] + dt * u[0];
                x = [xn0, xn1];
            }
            x
        });
    });
}

criterion_group!(
    benches,
    mpc_2x1_horizon10,
    mpc_4x2_horizon5,
    mpc_closed_loop
);
criterion_main!(benches);
