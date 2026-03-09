//! Benchmark: kinematics FK/IK/Jacobian throughput.

use criterion::{criterion_group, criterion_main, Criterion};
use oxictl::kinematics::forward::Transform2D;
use oxictl::kinematics::jacobian::Jacobian2R;
use oxictl::kinematics::serial::scara::{ScaraConfig, ScaraRobot};
use oxictl::kinematics::serial::six_dof::robot6_ur5_like;

fn transform2d_compose(c: &mut Criterion) {
    let t1 = Transform2D::<f64>::new(1.0, 0.0, 0.5);
    let t2 = Transform2D::<f64>::new(0.5, 1.0, -0.3);

    c.bench_function("transform2d_compose", |b| {
        b.iter(|| t1.compose(&t2));
    });
}

fn jacobian2r_compute(c: &mut Criterion) {
    let jac = Jacobian2R::<f64>::new(0.5, 0.4);

    c.bench_function("jacobian2r_compute", |b| {
        b.iter(|| jac.compute(0.3f64, -0.5));
    });
}

fn jacobian2r_ik_step(c: &mut Criterion) {
    let jac = Jacobian2R::<f64>::new(0.5, 0.4);

    c.bench_function("jacobian2r_ik_step", |b| {
        b.iter(|| jac.ik_step(0.3f64, -0.5, 0.01, 0.01));
    });
}

fn scara_fk(c: &mut Criterion) {
    let cfg = ScaraConfig::<f64>::desktop();
    let mut robot = ScaraRobot::new(cfg);
    robot.set_joints([0.3, -0.4, 0.05, 0.1]);

    c.bench_function("scara_fk", |b| {
        b.iter(|| robot.forward());
    });
}

fn scara_ik(c: &mut Criterion) {
    let cfg = ScaraConfig::<f64>::desktop();
    let robot = ScaraRobot::new(cfg);

    c.bench_function("scara_ik", |b| {
        b.iter(|| robot.inverse(0.3f64, 0.1, 0.05, 0.2));
    });
}

fn robot6dof_fk(c: &mut Criterion) {
    let mut robot = robot6_ur5_like();
    robot.set_joints([0.1f64, -0.2, 0.3, -0.1, 0.4, -0.3]);

    c.bench_function("robot6dof_fk", |b| {
        b.iter(|| robot.forward());
    });
}

fn robot6dof_jacobian(c: &mut Criterion) {
    let mut robot = robot6_ur5_like();
    robot.set_joints([0.1f64, -0.2, 0.3, -0.1, 0.4, -0.3]);

    c.bench_function("robot6dof_jacobian", |b| {
        b.iter(|| robot.jacobian());
    });
}

fn robot6dof_ik_step(c: &mut Criterion) {
    let mut robot = robot6_ur5_like();
    robot.set_joints([0.1f64, -0.2, 0.3, -0.1, 0.4, -0.3]);
    let target_pos = [0.4f64, 0.1, 0.5];

    c.bench_function("robot6dof_ik_step", |b| {
        b.iter(|| robot.inverse_ik(target_pos, 10, 1e-4f64, 0.01f64));
    });
}

criterion_group!(
    benches,
    transform2d_compose,
    jacobian2r_compute,
    jacobian2r_ik_step,
    scara_fk,
    scara_ik,
    robot6dof_fk,
    robot6dof_jacobian,
    robot6dof_ik_step,
);
criterion_main!(benches);
