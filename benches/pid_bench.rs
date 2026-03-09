//! Benchmark: PID controller update throughput.

use criterion::{criterion_group, criterion_main, Criterion};
use oxictl::core::signal::{Feedback, Setpoint};
use oxictl::core::traits::Controller;
use oxictl::pid::standard::PidConfig;

fn pid_update_f64(c: &mut Criterion) {
    let mut pid = PidConfig::pid(1.0f64, 0.1, 0.05).build();
    let sp = Setpoint::new(1.0f64);
    let fb = Feedback::new(0.5f64);
    let dt = 0.001f64;

    c.bench_function("pid_update_f64", |b| {
        b.iter(|| {
            let _ = pid.update(&sp, &fb, dt);
        });
    });
}

fn pid_update_f32(c: &mut Criterion) {
    let mut pid = PidConfig::pid(1.0f32, 0.1, 0.05).build();
    let sp = Setpoint::new(1.0f32);
    let fb = Feedback::new(0.5f32);
    let dt = 0.001f32;

    c.bench_function("pid_update_f32", |b| {
        b.iter(|| {
            let _ = pid.update(&sp, &fb, dt);
        });
    });
}

fn pid_1khz_loop(c: &mut Criterion) {
    // Simulate a 1 kHz control loop: 1000 PID updates
    let mut pid = PidConfig::pid(2.0f64, 0.5, 0.02).build();
    let dt = 0.001f64;

    c.bench_function("pid_1khz_loop_1000_steps", |b| {
        b.iter(|| {
            let mut y = 0.0f64;
            for _ in 0..1000 {
                let sp = Setpoint::new(1.0f64);
                let fb = Feedback::new(y);
                let u = pid.update(&sp, &fb, dt).value();
                // Simulate simple first-order plant: y += (u - y) * dt
                y += (u - y) * dt;
            }
            y
        });
    });
}

criterion_group!(benches, pid_update_f64, pid_update_f32, pid_1khz_loop);
criterion_main!(benches);
