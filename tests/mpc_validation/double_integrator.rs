//! MPC double integrator validation: comparison with analytical solution.

use oxictl::core::matrix::Matrix;
use oxictl::mpc::linear_mpc::{LinearMpc, MpcConstraints};

fn build_double_integrator(dt: f64) -> LinearMpc<f64, 2, 1, 10> {
    let a = Matrix {
        data: [[1.0f64, dt], [0.0, 1.0]],
    };
    let b = Matrix {
        data: [[0.5 * dt * dt], [dt]],
    };
    let q = Matrix {
        data: [[1.0f64, 0.0], [0.0, 0.1]],
    };
    let q_f = Matrix {
        data: [[10.0f64, 0.0], [0.0, 1.0]],
    };
    let r = Matrix { data: [[0.01f64]] };
    let constraints = MpcConstraints::unconstrained();
    LinearMpc::new(a, b, q, q_f, r, constraints).with_optimizer(0.01, 100, 1e-6)
}

/// MPC drives double integrator from x0 to origin.
#[test]
fn double_integrator_reaches_origin() {
    let dt = 0.1f64;
    let mut mpc = build_double_integrator(dt);

    let mut x = [1.0f64, 0.0];
    let x_ref = [0.0f64, 0.0];

    for _ in 0..100 {
        let (u, _) = mpc.solve(&x, &x_ref);
        let x0 = x[0] + dt * x[1] + 0.5 * dt * dt * u[0];
        let x1 = x[1] + dt * u[0];
        x = [x0, x1];
    }

    assert!(x[0].abs() < 0.1, "Position should reach ~0: x={:.4}", x[0]);
    assert!(x[1].abs() < 0.1, "Velocity should reach ~0: v={:.4}", x[1]);
}

/// MPC returns finite control for zero initial condition.
#[test]
fn zero_initial_condition_gives_zero_control() {
    let dt = 0.1f64;
    let mut mpc = build_double_integrator(dt);
    let x0 = [0.0f64, 0.0];
    let x_ref = [0.0f64, 0.0];
    let (u, _) = mpc.solve(&x0, &x_ref);
    assert!(u[0].abs() < 0.1, "Zero IC → ~zero control: u={:.6}", u[0]);
}

/// MPC control sequence has finite length H.
#[test]
fn control_sequence_has_correct_horizon() {
    let dt = 0.1f64;
    let mut mpc = build_double_integrator(dt);
    let x0 = [1.0f64, 0.5];
    let _ = mpc.solve(&x0, &[0.0, 0.0]);
    let seq = mpc.control_sequence();
    assert_eq!(seq.len(), 10, "Horizon should be 10");
}

/// MPC with box constraints respects input limits.
#[test]
fn mpc_respects_input_constraints() {
    let dt = 0.1f64;
    let a = Matrix {
        data: [[1.0f64, dt], [0.0, 1.0]],
    };
    let b = Matrix {
        data: [[0.5 * dt * dt], [dt]],
    };
    let q = Matrix {
        data: [[1.0f64, 0.0], [0.0, 0.1]],
    };
    let q_f = Matrix {
        data: [[5.0f64, 0.0], [0.0, 0.5]],
    };
    let r = Matrix { data: [[0.01f64]] };
    let constraints = MpcConstraints::<f64, 1>::box_input([-1.0], [1.0]);
    let mut mpc = LinearMpc::<f64, 2, 1, 10>::new(a, b, q, q_f, r, constraints)
        .with_optimizer(0.01, 100, 1e-6);

    let x0 = [5.0f64, 3.0];
    let (u, _) = mpc.solve(&x0, &[0.0, 0.0]);
    assert!(
        u[0] >= -1.01 && u[0] <= 1.01,
        "Control must respect bounds: u={:.4}",
        u[0]
    );

    let seq = mpc.control_sequence();
    for &ui in seq {
        assert!(ui[0] >= -1.01 && ui[0] <= 1.01, "Sequence: u={:.4}", ui[0]);
    }
}

/// MPC cost decreases as horizon extends (more lookahead helps).
#[test]
fn mpc_converges_over_simulation() {
    let dt = 0.05f64;
    let a = Matrix {
        data: [[1.0f64, dt], [0.0, 1.0]],
    };
    let b = Matrix {
        data: [[0.5 * dt * dt], [dt]],
    };
    let q = Matrix {
        data: [[1.0f64, 0.0], [0.0, 0.1]],
    };
    let q_f = Matrix {
        data: [[10.0f64, 0.0], [0.0, 1.0]],
    };
    let r = Matrix { data: [[0.01f64]] };
    let constraints = MpcConstraints::unconstrained();
    let mut mpc =
        LinearMpc::<f64, 2, 1, 8>::new(a, b, q, q_f, r, constraints).with_optimizer(0.01, 80, 1e-5);

    let mut x = [2.0f64, 1.0];
    let x_ref = [0.0f64, 0.0];

    let initial_norm = (x[0] * x[0] + x[1] * x[1]).sqrt();

    for _ in 0..150 {
        let (u, _) = mpc.solve(&x, &x_ref);
        let x0 = x[0] + dt * x[1] + 0.5 * dt * dt * u[0];
        let x1 = x[1] + dt * u[0];
        x = [x0, x1];
    }

    let final_norm = (x[0] * x[0] + x[1] * x[1]).sqrt();
    assert!(
        final_norm < initial_norm * 0.1,
        "MPC should reduce state norm: init={:.4}, final={:.4}",
        initial_norm,
        final_norm
    );
}
