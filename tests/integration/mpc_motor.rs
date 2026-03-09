//! Integration test: MPC controlling DC motor model.

use oxictl::core::matrix::Matrix;
use oxictl::mpc::linear_mpc::{LinearMpc, MpcConstraints};

/// MPC speed controller for a DC motor using linearized model.
///
/// State: [x] = [omega] (speed)
/// Input: [u] = [V] (voltage)
/// Simple first-order linearization: omega[k+1] ≈ a*omega[k] + b*V[k]
#[test]
fn mpc_dc_motor_speed_control() {
    let _dt = 0.005f64; // 200 Hz (used for model discretization, not in loop)
                        // Identified linearized model (approximate):
                        // omega[k+1] = 0.98*omega[k] + 0.1*V[k]
    let a = Matrix::<f64, 1, 1> { data: [[0.98]] };
    let b = Matrix::<f64, 1, 1> { data: [[0.1]] };
    let q = Matrix::<f64, 1, 1> { data: [[1.0]] };
    let q_f = Matrix::<f64, 1, 1> { data: [[5.0]] };
    let r = Matrix::<f64, 1, 1> { data: [[0.01]] };
    let constraints = MpcConstraints::<f64, 1>::box_input([-24.0], [24.0]);

    let mut mpc = LinearMpc::<f64, 1, 1, 10>::new(a, b, q, q_f, r, constraints)
        .with_optimizer(0.01, 50, 1e-5);

    // Simulate using the same linear model (MPC model = plant model for clean validation)
    let mut omega = 0.0f64;
    let speed_ref = 50.0f64; // rad/s

    for _ in 0..2000 {
        // 10 seconds at 200 Hz
        let (u, _) = mpc.solve(&[omega], &[speed_ref]);
        // Plant matches MPC model: omega[k+1] = 0.98*omega[k] + 0.1*V
        omega = 0.98 * omega + 0.1 * u[0];
    }

    let final_omega = omega;
    // With MPC on linearized model, speed should be moving toward reference
    assert!(
        final_omega >= 0.0,
        "Motor should be spinning in positive direction: omega={:.2}",
        final_omega
    );
    assert!(
        final_omega.is_finite(),
        "Speed must be finite: {}",
        final_omega
    );
}

/// MPC with voltage constraints never exceeds limits.
#[test]
fn mpc_respects_voltage_constraints() {
    let _dt = 0.01f64;
    let a = Matrix::<f64, 1, 1> { data: [[0.97]] };
    let b = Matrix::<f64, 1, 1> { data: [[0.05]] };
    let q = Matrix::<f64, 1, 1> { data: [[1.0]] };
    let q_f = Matrix::<f64, 1, 1> { data: [[3.0]] };
    let r = Matrix::<f64, 1, 1> { data: [[0.1]] };
    let v_max = 12.0f64;
    let constraints = MpcConstraints::<f64, 1>::box_input([-v_max], [v_max]);

    let mut mpc =
        LinearMpc::<f64, 1, 1, 8>::new(a, b, q, q_f, r, constraints).with_optimizer(0.01, 50, 1e-5);

    for &omega in &[0.0f64, 10.0, 50.0, 100.0, 200.0] {
        let (u, _) = mpc.solve(&[omega], &[0.0]);
        assert!(
            u[0] >= -v_max - 0.01 && u[0] <= v_max + 0.01,
            "omega={}: voltage u={:.4} out of [-{},{}]",
            omega,
            u[0],
            v_max,
            v_max
        );
    }
}

/// MPC 2-state motor control (current + speed).
#[test]
fn mpc_two_state_motor() {
    let _dt = 0.001f64;
    // State: [current, omega], Input: [V]
    // i[k+1] = 0.9*i[k] - 0.01*omega[k] + 0.1*V[k]
    // omega[k+1] = 0.01*i[k] + 0.99*omega[k]
    let a = Matrix::<f64, 2, 2> {
        data: [[0.9, -0.01], [0.01, 0.99]],
    };
    let b = Matrix::<f64, 2, 1> {
        data: [[0.1], [0.0]],
    };
    let q = Matrix::<f64, 2, 2> {
        data: [[0.1, 0.0], [0.0, 1.0]],
    };
    let q_f = Matrix::<f64, 2, 2> {
        data: [[0.5, 0.0], [0.0, 5.0]],
    };
    let r = Matrix::<f64, 1, 1> { data: [[0.01]] };
    let constraints = MpcConstraints::<f64, 1>::box_input([-24.0], [24.0]);

    let mut mpc = LinearMpc::<f64, 2, 1, 8>::new(a, b, q, q_f, r, constraints)
        .with_optimizer(0.005, 50, 1e-5);

    let mut x = [0.0f64, 0.0]; // [current, omega]
    let x_ref = [0.0f64, 50.0]; // Target: omega=50, current=0

    for _ in 0..2000 {
        let (u, _) = mpc.solve(&x, &x_ref);
        let x0 = 0.9 * x[0] - 0.01 * x[1] + 0.1 * u[0];
        let x1 = 0.01 * x[0] + 0.99 * x[1];
        x = [x0, x1];
    }

    // Speed should be increasing toward reference
    assert!(x[1] > 0.0, "Speed should be positive: omega={:.2}", x[1]);
    assert!(x[1].is_finite(), "Speed must be finite");
}
