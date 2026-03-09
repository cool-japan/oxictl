//! MPC pendulum stabilization validation (linearized model).

use oxictl::core::matrix::Matrix;
use oxictl::mpc::linear_mpc::{LinearMpc, MpcConstraints};

/// Linearized inverted pendulum model around upright equilibrium.
/// State: [x, x_dot, theta, theta_dot], Input: F (force on cart)
fn build_pendulum_mpc() -> LinearMpc<f64, 4, 1, 12> {
    let m_cart = 1.0f64;
    let m_pole = 0.1f64;
    let l_pole = 0.5f64;
    let g = 9.81f64;
    let dt = 0.02f64;

    let alpha = (m_cart + m_pole) * g / (l_pole * m_cart);
    let beta = m_pole * g / m_cart;

    let a = Matrix::<f64, 4, 4> {
        data: [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, -beta * dt, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, alpha * dt, 1.0],
        ],
    };
    let b = Matrix::<f64, 4, 1> {
        data: [[0.0], [dt / m_cart], [0.0], [-dt / (l_pole * m_cart)]],
    };

    // Heavy angle weight: keep pole upright
    let q = Matrix::<f64, 4, 4> {
        data: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 10.0],
        ],
    };
    let q_f = Matrix::<f64, 4, 4> {
        data: [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 500.0, 0.0],
            [0.0, 0.0, 0.0, 50.0],
        ],
    };
    let r = Matrix::<f64, 1, 1> { data: [[0.01]] };
    let constraints = MpcConstraints::<f64, 1>::box_input([-20.0], [20.0]);

    LinearMpc::new(a, b, q, q_f, r, constraints).with_optimizer(0.005, 60, 1e-5)
}

/// Simulate nonlinear pendulum (Euler integration).
fn pendulum_step(state: &[f64; 4], force: f64, dt: f64) -> [f64; 4] {
    let (x, xd, th, thd) = (state[0], state[1], state[2], state[3]);
    let m_cart = 1.0f64;
    let m_pole = 0.1f64;
    let l_pole = 0.5f64;
    let g = 9.81f64;
    let m_total = m_cart + m_pole;
    let cos_th = th.cos();
    let sin_th = th.sin();
    let denom = l_pole * (m_total / m_pole - cos_th * cos_th * m_pole / m_total);

    let thdd = (g * sin_th - cos_th * (force + m_pole * l_pole * thd * thd * sin_th) / m_total)
        / denom.max(1e-6);
    let xdd = (force + m_pole * l_pole * (thd * thd * sin_th - thdd * cos_th)) / m_total;

    [x + xd * dt, xd + xdd * dt, th + thd * dt, thd + thdd * dt]
}

/// MPC keeps pendulum upright from small initial perturbation.
#[test]
fn mpc_stabilizes_small_perturbation() {
    let mut mpc = build_pendulum_mpc();
    let mut state = [0.0f64, 0.0, 0.05, 0.0]; // 0.05 rad = ~3 deg
    let x_ref = [0.0f64; 4];
    let dt = 0.02f64;

    for _ in 0..200 {
        let (u, _) = mpc.solve(&state, &x_ref);
        state = pendulum_step(&state, u[0], dt);

        // Fail if pendulum falls over
        if state[2].abs() > 0.8 {
            panic!("Pendulum fell over: theta={:.3} rad", state[2]);
        }
    }

    // After 4 seconds, pendulum should be near upright
    assert!(
        state[2].abs() < 0.15,
        "MPC should stabilize: theta={:.4} rad",
        state[2]
    );
}

/// MPC control respects force bounds throughout stabilization.
#[test]
fn mpc_force_within_bounds_during_stabilization() {
    let mut mpc = build_pendulum_mpc();
    let mut state = [0.0f64, 0.0, 0.1, 0.0]; // 0.1 rad perturbation
    let x_ref = [0.0f64; 4];
    let dt = 0.02f64;

    for _ in 0..150 {
        let (u, _) = mpc.solve(&state, &x_ref);
        assert!(
            u[0] >= -20.1 && u[0] <= 20.1,
            "Force out of bounds: {:.4}",
            u[0]
        );
        state = pendulum_step(&state, u[0], dt);
        if state[2].abs() > 1.5 {
            break;
        } // Stop if fallen
    }
}

/// Larger initial perturbation may not be stabilizable (just check no panic).
#[test]
fn mpc_handles_large_perturbation_gracefully() {
    let mut mpc = build_pendulum_mpc();
    let mut state = [0.0f64, 0.0, 0.4, 0.0]; // Large perturbation
    let x_ref = [0.0f64; 4];
    let dt = 0.02f64;

    let mut final_theta = state[2];
    for _ in 0..100 {
        let (u, _) = mpc.solve(&state, &x_ref);
        state = pendulum_step(&state, u[0], dt);
        final_theta = state[2];
        if final_theta.abs() > 1.5 {
            break;
        }
    }
    // Just check no panic occurred; large perturbation may fall over
    assert!(final_theta.is_finite(), "State must remain finite");
}
