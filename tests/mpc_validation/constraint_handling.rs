//! MPC constraint handling validation.

use oxictl::core::matrix::Matrix;
use oxictl::mpc::linear_mpc::{LinearMpc, MpcConstraints};

/// Builds simple integrator: x[k+1] = x[k] + dt*u
fn build_integrator(dt: f64) -> LinearMpc<f64, 1, 1, 8> {
    let a = Matrix { data: [[1.0f64]] };
    let b = Matrix { data: [[dt]] };
    let q = Matrix { data: [[1.0f64]] };
    let q_f = Matrix { data: [[5.0f64]] };
    let r = Matrix { data: [[0.1f64]] };
    let constraints = MpcConstraints::<f64, 1>::box_input([-1.0], [1.0]);
    LinearMpc::new(a, b, q, q_f, r, constraints).with_optimizer(0.01, 100, 1e-6)
}

/// Control is always within box bounds under saturation conditions.
#[test]
fn control_within_box_bounds() {
    let dt = 0.1f64;
    let mut mpc = build_integrator(dt);
    let x_ref = [0.0f64];

    for &x0 in &[-10.0f64, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0] {
        let (u, _) = mpc.solve(&[x0], &x_ref);
        assert!(
            u[0] >= -1.01 && u[0] <= 1.01,
            "x0={}: u={:.4} out of [-1,1]",
            x0,
            u[0]
        );
    }
}

/// Unconstrained MPC gives larger control for same problem.
#[test]
fn unconstrained_gives_larger_control_than_constrained() {
    let dt = 0.1f64;
    let a = Matrix { data: [[1.0f64]] };
    let b = Matrix { data: [[dt]] };
    let q = Matrix { data: [[1.0f64]] };
    let q_f = Matrix { data: [[5.0f64]] };
    let r = Matrix { data: [[0.1f64]] };

    let c_unconstrained = MpcConstraints::<f64, 1>::unconstrained();
    let c_constrained = MpcConstraints::<f64, 1>::box_input([-0.5], [0.5]);

    let mut mpc_unc = LinearMpc::<f64, 1, 1, 8>::new(a, b, q, q_f, r, c_unconstrained)
        .with_optimizer(0.01, 100, 1e-6);
    let mut mpc_con = LinearMpc::<f64, 1, 1, 8>::new(a, b, q, q_f, r, c_constrained)
        .with_optimizer(0.01, 100, 1e-6);

    let x0 = [3.0f64];
    let x_ref = [0.0f64];

    let (u_unc, _) = mpc_unc.solve(&x0, &x_ref);
    let (u_con, _) = mpc_con.solve(&x0, &x_ref);

    assert!(
        u_unc[0].abs() >= u_con[0].abs() - 0.01,
        "Unconstrained u={:.4} should be >= constrained u={:.4}",
        u_unc[0].abs(),
        u_con[0].abs()
    );
    assert!(u_con[0] >= -0.51 && u_con[0] <= 0.51);
}

/// Rate constraint (du_max) limits control increment.
#[test]
fn rate_constraint_limits_increment() {
    let dt = 0.1f64;
    let a = Matrix { data: [[1.0f64]] };
    let b = Matrix { data: [[dt]] };
    let q = Matrix { data: [[1.0f64]] };
    let q_f = Matrix { data: [[5.0f64]] };
    let r = Matrix { data: [[0.01f64]] };

    let c = MpcConstraints::<f64, 1> {
        u_min: [-5.0],
        u_max: [5.0],
        du_max: [0.2], // Max rate of change: 0.2 per step
    };
    let mut mpc =
        LinearMpc::<f64, 1, 1, 6>::new(a, b, q, q_f, r, c).with_optimizer(0.005, 100, 1e-6);

    let x0 = [5.0f64];
    let x_ref = [0.0f64];

    let (u1, _) = mpc.solve(&x0, &x_ref);
    assert!(
        u1[0].abs() <= 0.21,
        "First control should be ≤ du_max=0.2: u={:.4}",
        u1[0]
    );
}

/// Multi-input MPC respects independent bounds per input.
#[test]
fn multi_input_independent_bounds() {
    let dt = 0.1f64;
    let a = Matrix {
        data: [[0.9f64, 0.1], [0.0, 0.8]],
    };
    let b = Matrix {
        data: [[dt, 0.0f64], [0.0, dt]],
    };
    let q = Matrix::<f64, 2, 2>::identity();
    let q_f = Matrix::<f64, 2, 2>::identity();
    let r = Matrix {
        data: [[0.1f64, 0.0], [0.0, 0.5]],
    };

    let constraints = MpcConstraints::<f64, 2> {
        u_min: [-2.0, -0.5],
        u_max: [2.0, 0.5],
        du_max: [1e9, 1e9],
    };

    let mut mpc = LinearMpc::<f64, 2, 2, 5>::new(a, b, q, q_f, r, constraints)
        .with_optimizer(0.005, 100, 1e-5);

    let x0 = [3.0f64, 2.0];
    let x_ref = [0.0f64, 0.0];

    let (u, _) = mpc.solve(&x0, &x_ref);
    assert!(
        u[0] >= -2.01 && u[0] <= 2.01,
        "u[0]={:.4} out of [-2,2]",
        u[0]
    );
    assert!(
        u[1] >= -0.51 && u[1] <= 0.51,
        "u[1]={:.4} out of [-0.5,0.5]",
        u[1]
    );
}
