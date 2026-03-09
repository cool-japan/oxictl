//! Integration tests for kinematics: FK/IK round-trips.

use oxictl::kinematics::forward::Transform2D;
use oxictl::kinematics::jacobian::Jacobian2R;
use oxictl::kinematics::serial::scara::{ScaraConfig, ScaraRobot};

/// SCARA FK → IK → FK round-trip: recover original end-effector position.
#[test]
fn scara_fk_ik_roundtrip() {
    let cfg = ScaraConfig::<f64>::desktop();
    let mut robot = ScaraRobot::new(cfg);

    // Set joint config in the reachable workspace
    robot.set_joints([0.3, -0.4, 0.05, 0.2]);
    let (x_fk, y_fk, z_fk, psi_fk) = robot.forward();

    // IK from the FK result should recover same joints (within tolerance)
    let ik_result = robot.inverse(x_fk, y_fk, z_fk, psi_fk);
    assert!(
        ik_result.is_some(),
        "IK should find a solution in the reachable workspace"
    );

    let q_ik = ik_result.unwrap();
    robot.set_joints(q_ik);
    let (x2, y2, z2, psi2) = robot.forward();

    assert!((x2 - x_fk).abs() < 1e-6, "x: {x2} vs {x_fk}");
    assert!((y2 - y_fk).abs() < 1e-6, "y: {y2} vs {y_fk}");
    assert!((z2 - z_fk).abs() < 1e-6, "z: {z2} vs {z_fk}");
    assert!((psi2 - psi_fk).abs() < 1e-6, "psi: {psi2} vs {psi_fk}");
}

/// SCARA FK gives different results for different joint configs.
#[test]
fn scara_fk_is_injective_for_distinct_joints() {
    let cfg = ScaraConfig::<f64>::desktop();
    let mut robot = ScaraRobot::new(cfg);

    robot.set_joints([0.3, -0.4, 0.05, 0.2]);
    let (x1, y1, _, _) = robot.forward();

    robot.set_joints([0.8, 0.2, 0.02, 0.5]);
    let (x2, y2, _, _) = robot.forward();

    let dist = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
    assert!(
        dist > 0.01,
        "Different joints should give different EE positions"
    );
}

/// Jacobian2R IK step: linearized prediction matches Cartesian delta.
#[test]
fn jacobian2r_ik_step_linearized_prediction() {
    let jac = Jacobian2R::<f64>::new(0.5, 0.4);

    let (q1, q2) = (0.5f64, -0.4f64);
    let dx = 0.005f64;
    let dy = 0.003f64;

    let step = jac.ik_step(q1, q2, dx, dy);
    assert!(
        step.is_some(),
        "IK step should succeed away from singularity"
    );

    let (dq1, dq2) = step.unwrap();
    // Verify J * dq ≈ [dx, dy] (linearized: dq = J^+ * [dx,dy] => J*dq = [dx,dy])
    let (vx, vy) = jac.apply(q1, q2, dq1, dq2);
    assert!((vx - dx).abs() < 1e-10, "Linearized vx={vx} ≠ dx={dx}");
    assert!((vy - dy).abs() < 1e-10, "Linearized vy={vy} ≠ dy={dy}");
}

/// Jacobian2R IK step: small error step reduces actual end-effector error.
#[test]
fn jacobian2r_ik_step_reduces_small_error() {
    let jac = Jacobian2R::<f64>::new(0.5, 0.4);

    fn fk(l1: f64, l2: f64, q1: f64, q2: f64) -> (f64, f64) {
        (
            l1 * q1.cos() + l2 * (q1 + q2).cos(),
            l1 * q1.sin() + l2 * (q1 + q2).sin(),
        )
    }

    let (q1_target, q2_target) = (0.6f64, -0.5f64);
    let (x_target, y_target) = fk(0.5, 0.4, q1_target, q2_target);
    // Start close: only 0.02 rad away
    let (q1_start, q2_start) = (0.62f64, -0.48f64);

    let (x0, y0) = fk(0.5, 0.4, q1_start, q2_start);
    let err0 = ((x0 - x_target).powi(2) + (y0 - y_target).powi(2)).sqrt();

    let dx = x_target - x0;
    let dy = y_target - y0;
    let (dq1, dq2) = jac.ik_step(q1_start, q2_start, dx, dy).unwrap();

    // Take a partial step (0.5x) to avoid overshoot
    let (x1, y1) = fk(0.5, 0.4, q1_start + 0.5 * dq1, q2_start + 0.5 * dq2);
    let err1 = ((x1 - x_target).powi(2) + (y1 - y_target).powi(2)).sqrt();
    assert!(err1 < err0, "IK step should reduce error: {err1} < {err0}");
}

/// Jacobian2R: pseudo-inverse exists away from singularity.
#[test]
fn jacobian2r_pseudo_inverse_exists_away_from_singularity() {
    let jac = Jacobian2R::<f64>::new(0.5, 0.4);

    // q2 = 0 is singularity; use q2 = 0.5
    let pinv = jac.pseudo_inverse(0.3, 0.5);
    assert!(
        pinv.is_some(),
        "Pseudo-inverse should exist when not at singularity"
    );

    let j = jac.compute(0.3, 0.5);
    let p = pinv.unwrap();
    // J * J^+ should be close to identity (2x2)
    let j00 = j[0][0] * p[0][0] + j[0][1] * p[1][0];
    let j11 = j[1][0] * p[0][1] + j[1][1] * p[1][1];
    assert!((j00 - 1.0).abs() < 1e-9, "J*J^+ [0][0] = {j00}");
    assert!((j11 - 1.0).abs() < 1e-9, "J*J^+ [1][1] = {j11}");
}

/// Transform2D compose is right-to-left: A.compose(B) applies B then A.
#[test]
fn transform2d_chain_consistency() {
    use core::f64::consts::FRAC_PI_4;
    let t1 = Transform2D::<f64>::new(1.0, 0.0, FRAC_PI_4);
    let t2 = Transform2D::<f64>::new(0.0, 1.0, FRAC_PI_4);
    let t3 = Transform2D::<f64>::new(0.5, -0.5, 0.0);

    // t1.compose(t2).compose(t3) applies: t3 first, then t2, then t1
    let chain = t1.compose(&t2).compose(&t3);
    let point = [1.0, 0.0];
    let (px, py) = chain.transform_point(point[0], point[1]);

    // Apply in reverse: t3 → t2 → t1
    let (p1x, p1y) = t3.transform_point(point[0], point[1]);
    let (p2x, p2y) = t2.transform_point(p1x, p1y);
    let (p3x, p3y) = t1.transform_point(p2x, p2y);

    assert!((px - p3x).abs() < 1e-10, "x: {px} vs {p3x}");
    assert!((py - p3y).abs() < 1e-10, "y: {py} vs {p3y}");
}
