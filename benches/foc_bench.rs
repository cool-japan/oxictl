//! Benchmark: FOC (Field-Oriented Control) pipeline throughput.

use criterion::{criterion_group, criterion_main, Criterion};
use oxictl::motor::foc::controller::FocController;
use oxictl::motor::transform::clarke::{clarke_2ph, clarke_inverse};
use oxictl::motor::transform::park::{park, park_inverse, Dq};
use oxictl::motor::transform::svpwm::svpwm;

fn foc_full_cycle(c: &mut Criterion) {
    let mut foc = FocController::<f64>::new(
        10.0, 50.0, // speed: kp, ki
        5.0, 100.0, // current: kp, ki
        10.0,  // iq_limit
        24.0,  // v_limit
        48.0,  // vdc
    );
    let dt = 1e-4f64; // 10 kHz

    c.bench_function("foc_update_10khz", |b| {
        b.iter(|| {
            foc.update(
                100.0, // speed_ref rad/s
                95.0,  // speed_meas
                1.5,   // ia
                -0.8,  // ib
                0.3,   // theta
                dt,
            )
        });
    });
}

fn clarke_park_cycle(c: &mut Criterion) {
    c.bench_function("clarke_park_inverse_cycle", |b| {
        b.iter(|| {
            let ab = clarke_2ph(1.5f64, -0.8f64);
            let theta = 0.3f64;
            let dq = park(&ab, theta);
            let ab2 = park_inverse(&Dq { d: dq.d, q: dq.q }, theta);
            let _ = clarke_inverse(&ab2);
            dq
        });
    });
}

fn svpwm_bench(c: &mut Criterion) {
    use oxictl::motor::transform::clarke::AlphaBeta;
    let ab = AlphaBeta::<f64> {
        alpha: 0.3,
        beta: 0.5,
        zero: 0.0,
    };
    let vdc = 48.0f64;

    c.bench_function("svpwm", |b| {
        b.iter(|| svpwm(&ab, vdc));
    });
}

fn foc_10khz_loop(c: &mut Criterion) {
    // Simulate 10 kHz FOC loop for 1ms (10 steps)
    let mut foc = FocController::<f64>::new(10.0, 50.0, 5.0, 100.0, 10.0, 24.0, 48.0);
    let dt = 1e-4f64;

    c.bench_function("foc_10khz_loop_100_steps", |b| {
        b.iter(|| {
            let mut omega = 0.0f64;
            let mut theta = 0.0f64;
            for _ in 0..100 {
                let out = foc.update(100.0, omega, 0.5, -0.3, theta, dt);
                // Tiny integrator to simulate plant
                omega += out.vq * dt * 0.1;
                theta += omega * dt;
                if theta > core::f64::consts::TAU {
                    theta -= core::f64::consts::TAU;
                }
            }
            omega
        });
    });
}

criterion_group!(
    benches,
    foc_full_cycle,
    clarke_park_cycle,
    svpwm_bench,
    foc_10khz_loop
);
criterion_main!(benches);
