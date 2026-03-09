//! Integration tests for state feedback: LQR closed-loop stability.

use oxictl::core::matrix::Matrix;
use oxictl::state_feedback::lqr::Lqr;
use oxictl::state_feedback::pole_placement::{ackermann, StateFeedback};

/// LQR stabilizes a double integrator: state converges to zero from x0=[1,0].
#[test]
fn lqr_stabilizes_double_integrator() {
    let dt = 0.01f64;
    let a = Matrix::<f64, 2, 2> {
        data: [[1.0, dt], [0.0, 1.0]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.5 * dt * dt], [dt]],
    };
    let q = Matrix::<f64, 2, 2> {
        data: [[10.0, 0.0], [0.0, 1.0]],
    };
    let r = Matrix::<f64, 1, 1> { data: [[0.01]] };

    let lqr =
        Lqr::design(&a, &b, &q, &r).expect("LQR design should succeed for controllable system");

    let mut x = [1.0f64, 0.0];
    let ref_zero = [0.0f64, 0.0];

    for _ in 0..500 {
        let u = lqr.control(&x, &ref_zero);
        let x0 = a.data[0][0] * x[0] + a.data[0][1] * x[1] + b.data[0][0] * u[0];
        let x1 = a.data[1][0] * x[0] + a.data[1][1] * x[1] + b.data[1][0] * u[0];
        x = [x0, x1];
    }

    assert!(x[0].abs() < 1e-3, "Position should converge to 0: {}", x[0]);
    assert!(x[1].abs() < 1e-3, "Velocity should converge to 0: {}", x[1]);
}

/// Ackermann pole placement: closed-loop eigenvalue matches desired pole for 1st-order.
#[test]
fn ackermann_places_poles_correctly() {
    let a = Matrix::<f64, 1, 1> { data: [[0.9]] };
    let b = Matrix::<f64, 1, 1> { data: [[1.0]] };
    let desired_poles = [0.5f64];

    let k = ackermann(&a, &b, &desired_poles)
        .expect("Ackermann should succeed for controllable system");

    // For 1x1: A_cl = A - B*K = 0.9 - 1.0*k[0]
    let a_cl = 0.9f64 - k[0];
    assert!(
        (a_cl - 0.5).abs() < 1e-9,
        "Closed-loop pole: {a_cl:.6} ≠ 0.5"
    );
}

/// Ackermann pole placement: closed-loop system converges for poles inside unit circle.
#[test]
fn ackermann_achieves_stability() {
    let a = Matrix::<f64, 2, 2> {
        data: [[1.0, 0.1], [0.0, 1.0]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.0], [0.1]],
    };
    let poles = [0.7f64, 0.8];

    let k = ackermann(&a, &b, &poles).expect("Ackermann should succeed");
    let sf = StateFeedback::new(k);

    let mut x = [1.0f64, 0.5];
    for _ in 0..200 {
        let u = sf.control(&x, &[0.0, 0.0]);
        let x0 = a.data[0][0] * x[0] + a.data[0][1] * x[1] + b.data[0][0] * u;
        let x1 = a.data[1][0] * x[0] + a.data[1][1] * x[1] + b.data[1][0] * u;
        x = [x0, x1];
    }

    let norm = (x[0] * x[0] + x[1] * x[1]).sqrt();
    assert!(norm < 1e-4, "State should converge to 0: norm={norm}");
}

/// LQR regulate: same as control with zero reference.
#[test]
fn lqr_regulate_equals_control_with_zero_ref() {
    let a = Matrix::<f64, 2, 2> {
        data: [[0.9, 0.05], [0.0, 0.85]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.05], [0.1]],
    };
    let q = Matrix::<f64, 2, 2>::identity();
    let r = Matrix::<f64, 1, 1> { data: [[0.1]] };

    let lqr = Lqr::design(&a, &b, &q, &r).unwrap();
    let x = [0.4f64, -0.2];

    let u_regulate = lqr.regulate(&x);
    let u_control = lqr.control(&x, &[0.0, 0.0]);

    assert!((u_regulate[0] - u_control[0]).abs() < 1e-12);
}

/// LQR closed-loop: position tracking from nonzero reference.
#[test]
fn lqr_tracks_nonzero_reference() {
    let dt = 0.01f64;
    let a = Matrix::<f64, 2, 2> {
        data: [[1.0, dt], [0.0, 1.0]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.5 * dt * dt], [dt]],
    };
    let q = Matrix::<f64, 2, 2> {
        data: [[100.0, 0.0], [0.0, 1.0]],
    };
    let r = Matrix::<f64, 1, 1> { data: [[0.01]] };

    let lqr = Lqr::design(&a, &b, &q, &r).unwrap();

    let mut x = [0.0f64, 0.0];
    let x_ref = [1.0f64, 0.0];

    for _ in 0..1000 {
        let u = lqr.control(&x, &x_ref);
        let x0 = a.data[0][0] * x[0] + a.data[0][1] * x[1] + b.data[0][0] * u[0];
        let x1 = a.data[1][0] * x[0] + a.data[1][1] * x[1] + b.data[1][0] * u[0];
        x = [x0, x1];
    }

    assert!(
        (x[0] - 1.0).abs() < 0.05,
        "Should track reference: x[0]={}",
        x[0]
    );
}
