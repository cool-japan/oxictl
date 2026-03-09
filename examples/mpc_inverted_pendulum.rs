//! MPC Inverted Pendulum: linearized LQR+MPC control of cart-pole system.
//!
//! System: inverted pendulum on a cart
//!   State: [x, x_dot, theta, theta_dot]
//!   Input: F (force on cart)
//!
//! We use a linearized model around the upright equilibrium (theta=0)
//! and control it with LinearMpc + LQR terminal cost.

use oxictl::core::matrix::Matrix;
use oxictl::mpc::linear_mpc::{LinearMpc, MpcConstraints, MpcStatus};
use oxictl::sim::pendulum::InvertedPendulum;

fn main() {
    // Linearized model parameters (SI units)
    let m_cart = 1.0f64;
    let m_pole = 0.1f64;
    let l_pole = 0.5f64;
    let g = 9.81f64;
    let dt = 0.02f64; // 50 Hz

    // Linearized dynamics at theta=0:
    //   A_c = [[0,1,0,0],[0,0,-m_pole*g/m_cart,0],[0,0,0,1],[0,0,(m_cart+m_pole)*g/(l_pole*m_cart),0]]
    //   B_c = [[0],[1/m_cart],[0],[-1/(l_pole*m_cart)]]
    // Euler discretization: A = I + A_c*dt, B = B_c*dt

    let alpha = (m_cart + m_pole) * g / (l_pole * m_cart);
    let beta = m_pole * g / m_cart;

    // Discrete A matrix
    let a = Matrix::<f64, 4, 4> {
        data: [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, -beta * dt, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, alpha * dt, 1.0],
        ],
    };

    // Discrete B matrix
    let b = Matrix::<f64, 4, 1> {
        data: [[0.0], [dt / m_cart], [0.0], [-dt / (l_pole * m_cart)]],
    };

    // Cost matrices: penalize angle most
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
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1000.0, 0.0],
            [0.0, 0.0, 0.0, 100.0],
        ],
    };
    let r = Matrix::<f64, 1, 1> { data: [[0.01]] };

    // Force limits
    let constraints = MpcConstraints::<f64, 1>::box_input([-20.0], [20.0]);

    let mut mpc = LinearMpc::<f64, 4, 1, 15>::new(a, b, q, q_f, r, constraints)
        .with_optimizer(0.005, 50, 1e-5);

    // Nonlinear pendulum simulation
    let mut plant = InvertedPendulum::<f64>::new(m_cart, m_pole, l_pole, g);
    plant.set_state([0.0, 0.0, 0.1, 0.0]); // Small initial angle (rad)

    println!("t(s),x(m),x_dot,theta(rad),theta_dot,F(N),status");

    let mut t = 0.0f64;
    let n_steps = 400; // 8 seconds
    let mut stabilized = false;

    for step in 0..n_steps {
        let state = *plant.state(); // copy to avoid borrow conflict
        let x_ref = [0.0f64; 4]; // Regulate to origin

        let (u, status) = mpc.solve(&state, &x_ref);
        let force = u[0];

        let status_str = match status {
            MpcStatus::Optimal => "OK",
            MpcStatus::MaxIter => "MAX_ITER",
            MpcStatus::Infeasible => "INFEASIBLE",
        };

        if step % 10 == 0 {
            println!(
                "{:.3},{:.4},{:.4},{:.4},{:.4},{:.3},{}",
                t, state[0], state[1], state[2], state[3], force, status_str
            );
        }

        // Apply control to nonlinear plant
        plant.step(force, dt);
        t += dt;

        // Check if stabilized
        if state[2].abs() < 0.01 && state[3].abs() < 0.05 && !stabilized {
            stabilized = true;
            eprintln!("Stabilized at t={:.3}s, theta={:.4} rad", t, state[2]);
        }
    }

    let final_state = plant.state();
    eprintln!(
        "Final: x={:.4}m, theta={:.4}rad ({:.2}deg)",
        final_state[0],
        final_state[2],
        final_state[2].to_degrees()
    );

    if final_state[2].abs() > 0.5 {
        eprintln!("WARNING: Pendulum fell over!");
    } else {
        eprintln!("SUCCESS: Pendulum balanced.");
    }
}
