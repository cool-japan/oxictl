//! Benchmark: scheduler tick throughput.

use criterion::{criterion_group, criterion_main, Criterion};
use oxictl::scheduler::fixed_rate::FixedRateTask;
use oxictl::scheduler::multi_rate::{
    MultiRateScheduler, PriorityScheduler, TaskDescriptor, TaskPriority,
};

fn fixed_rate_tick(c: &mut Criterion) {
    let mut task = FixedRateTask::<f64>::new(0.001); // 1 kHz task

    c.bench_function("fixed_rate_tick", |b| {
        b.iter(|| task.tick(0.001f64));
    });
}

fn multi_rate_4tasks(c: &mut Criterion) {
    // 4 tasks: 1 kHz, 500 Hz, 200 Hz, 100 Hz
    let mut sched = MultiRateScheduler::<f64, 4>::new([0.001, 0.002, 0.005, 0.01]);

    c.bench_function("multi_rate_4tasks_tick", |b| {
        b.iter(|| sched.tick(0.001f64));
    });
}

fn multi_rate_1000_steps(c: &mut Criterion) {
    // Simulate 1000 base ticks with 4 tasks
    let mut sched = MultiRateScheduler::<f64, 4>::new([0.001, 0.002, 0.005, 0.01]);
    let dt = 0.001f64;

    c.bench_function("multi_rate_4tasks_1000_ticks", |b| {
        b.iter(|| {
            let mut fire_count = 0usize;
            for _ in 0..1000 {
                let fired = sched.tick(dt);
                for &f in fired.iter() {
                    if f {
                        fire_count += 1;
                    }
                }
            }
            fire_count
        });
    });
}

fn priority_scheduler_4tasks(c: &mut Criterion) {
    let descs = [
        TaskDescriptor {
            name: "fast",
            priority: TaskPriority::Critical,
        },
        TaskDescriptor {
            name: "medium",
            priority: TaskPriority::High,
        },
        TaskDescriptor {
            name: "slow",
            priority: TaskPriority::Normal,
        },
        TaskDescriptor {
            name: "bg",
            priority: TaskPriority::Low,
        },
    ];
    let mut sched = PriorityScheduler::<f64, 4>::new(
        [0.001, 0.005, 0.01, 0.1],
        descs,
        [0.0005, 0.003, 0.008, 0.05],
    );

    c.bench_function("priority_scheduler_4tasks_tick", |b| {
        b.iter(|| sched.scheduler.tick(0.001f64));
    });
}

criterion_group!(
    benches,
    fixed_rate_tick,
    multi_rate_4tasks,
    multi_rate_1000_steps,
    priority_scheduler_4tasks,
);
criterion_main!(benches);
