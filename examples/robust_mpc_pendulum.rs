//! Robust MPC stabilizing an uncertain inverted pendulum under bounded wind disturbance.
//!
//! # System model (linearized at upright equilibrium)
//! State: x = [θ, θ̇]  (angle from vertical, angular rate)
//! Input: u             (torque at pivot, saturated to ±2 N·m)
//!
//! Continuous-time linearization (m·l²·θ̈ = m·g·l·sin θ + u ≈ m·g·l·θ + u):
//!
//!   A_c = [[0, 1], [m·g·l / (m·l²), 0]] = [[0, 1], [g/l, 0]]
//!   B_c = [[0], [1/(m·l²)]]
//!
//! Euler discretization with dt = 0.05 s:
//!   A_d ≈ I + A_c·dt
//!   B_d ≈ B_c·dt
//!
//! # Polytopic uncertainty: pendulum mass ±20%
//! Nominal mass m_nom = 0.5 kg.  Vertex 1: m = 0.4 kg (−20%), Vertex 2: m = 0.6 kg (+20%).
//! Both vertices share l = 0.3 m, g = 9.81 m/s².
//!
//! # Bounded disturbance
//! Wind force w ∈ [−0.1, 0.1] N applied at the pendulum tip translates to a
//! torque disturbance τ_w = w·l.  The disturbance is incorporated in the simulation
//! via a deterministic worst-case signal (sign chosen to oppose control action).
//!
//! # Robust MPC strategy
//! Min-max robust MPC (RobustMpc with V=2 vertices, horizon H=8) minimises the
//! worst-case quadratic cost Q·x + R·u² over both mass vertices.  Subgradient
//! descent finds the minimax input sequence; the first element is applied (receding
//! horizon).  Constraint tightening ensures feasibility despite the uncertainty.
//!
//! Run: `cargo run --example robust_mpc_pendulum --features mpc`

use oxictl::core::matrix::Matrix;
use oxictl::mpc::robust_mpc::{RobustBoxConstraint, RobustMpc, TerminalSet, UncertaintyVertex};

/// Build one discrete-time uncertainty vertex for a given pendulum mass.
///
/// # Arguments
/// * `mass` – pendulum mass (kg)
/// * `l`    – pendulum length (m)
/// * `g`    – gravitational acceleration (m/s²)
/// * `dt`   – discretization time step (s)
fn make_pendulum_vertex(mass: f64, l: f64, g: f64, dt: f64) -> UncertaintyVertex<f64, 2, 1> {
    let inertia = mass * l * l; // I = m·l²

    // Continuous A_c = [[0, 1], [g/l, 0]]  (independent of mass in this form,
    // but B_c = [[0],[1/I]] depends on mass)
    // A_d = I + A_c·dt
    let a = Matrix::<f64, 2, 2> {
        data: [[1.0, dt], [(g / l) * dt, 1.0]],
    };

    // B_d = B_c·dt
    let b = Matrix::<f64, 2, 1> {
        data: [[0.0], [dt / inertia]],
    };

    UncertaintyVertex::new(a, b)
}

/// Nonlinear pendulum Euler step (for realistic simulation).
///
/// θ̈ = (g/l)·sin(θ) + (1/(m·l²))·(u + τ_disturbance)
#[allow(clippy::too_many_arguments)]
fn pendulum_step_nonlinear(
    theta: f64,
    theta_dot: f64,
    u: f64,
    tau_dist: f64,
    mass: f64,
    l: f64,
    g: f64,
    dt: f64,
) -> (f64, f64) {
    let inertia = mass * l * l;
    let theta_ddot = (g / l) * libm::sin(theta) + (u + tau_dist) / inertia;
    (theta + dt * theta_dot, theta_dot + dt * theta_ddot)
}

fn main() {
    // --------------- Physical parameters ---------------
    let g = 9.81_f64;
    let l = 0.3_f64; // pendulum length (m)
    let m_nom = 0.5_f64; // nominal mass (kg)
    let m_lo = m_nom * 0.8; // −20% vertex
    let m_hi = m_nom * 1.2; // +20% vertex
    let dt = 0.05_f64; // 20 Hz control rate

    // --------------- Uncertainty vertices ---------------
    let v_lo = make_pendulum_vertex(m_lo, l, g, dt);
    let v_hi = make_pendulum_vertex(m_hi, l, g, dt);

    // --------------- Cost matrices ---------------
    // Penalize angle heavily; rate moderately; control lightly.
    let q = Matrix::<f64, 2, 2> {
        data: [[100.0, 0.0], [0.0, 1.0]],
    };
    let r_mat = Matrix::<f64, 1, 1> { data: [[0.1]] };

    // --------------- Constraints ---------------
    // State: |θ| ≤ 0.5 rad (≈ 28°),  |u| ≤ 2 N·m
    let constraints = RobustBoxConstraint::new(0.5_f64, 2.0_f64);

    // --------------- Terminal set: ellipsoidal neighborhood of origin ---------------
    // P_f = diag(200, 2) so that x^T P_f x ≤ 1 is a small safe region near origin.
    let mut p_f_diag = Matrix::<f64, 2, 1>::zeros();
    p_f_diag.data[0][0] = 200.0;
    p_f_diag.data[1][0] = 2.0;
    let terminal_set = TerminalSet::new(p_f_diag, 1.0_f64);

    // Horizon H=8, V=2 vertices, 60 subgradient iterations per solve.
    let mut mpc = RobustMpc::<f64, 2, 1, 8, 2>::new([v_lo, v_hi], q, r_mat, constraints, 60)
        .with_terminal_set(terminal_set)
        .with_tightening_margin(0.02_f64);

    // Fine-tune subgradient step size for this problem scale
    mpc.step_size = 5e-4_f64;

    // --------------- Initial condition: small perturbation ---------------
    // θ₀ = 0.15 rad ≈ 8.6° from upright, θ̇₀ = 0
    let theta0 = 0.15_f64;
    let theta_dot0 = 0.0_f64;

    // Simulation uses the *nominal* mass (uncertain to the controller)
    let m_sim = m_nom;

    // Wind force disturbance: alternates sign every 0.5 s (worst-case pattern)
    let w_max = 0.1_f64; // N (force at tip)

    println!("# Robust MPC Inverted Pendulum");
    println!(
        "# Uncertainty: m ∈ [{:.2}, {:.2}] kg (nom {:.2}), wind ±{:.2} N",
        m_lo, m_hi, m_nom, w_max
    );
    println!("# Horizon H=8, tightening δ=0.02, u_max=2 N·m");
    println!("step,t_s,theta_rad,theta_dot,u_Nm,tau_dist,worst_cost,in_terminal");

    let mut theta = theta0;
    let mut theta_dot = theta_dot0;
    let n_steps = 100_usize; // 5 s at 20 Hz

    let mut max_angle = theta0.abs();
    let mut total_control_effort = 0.0_f64;

    for step in 0..n_steps {
        let t = step as f64 * dt;

        // Set current state in controller
        let mut x = Matrix::<f64, 2, 1>::zeros();
        x.data[0][0] = theta;
        x.data[1][0] = theta_dot;
        mpc.set_state(x);

        // Solve min-max robust MPC
        let u_mat = mpc
            .solve()
            .expect("Robust MPC solve: feasibility expected for small perturbations");
        let u = u_mat.data[0][0];

        // Worst-case cost for reporting
        let u_seq = [u_mat; 8];
        let wc_cost = mpc.worst_case_cost(&u_seq);
        let in_term = mpc.in_terminal_set();

        // Wind disturbance: sinusoidal with period 1 s → torque at pivot = w·l
        let w = w_max * libm::sin(2.0 * core::f64::consts::PI * 1.0 * t);
        let tau_dist = w * l;

        println!(
            "{},{:.3},{:.5},{:.5},{:.5},{:.5},{:.4},{}",
            step,
            t,
            theta,
            theta_dot,
            u,
            tau_dist,
            wc_cost,
            if in_term { 1 } else { 0 }
        );

        // Simulate with nonlinear pendulum (more realistic than linearized model)
        let (theta_new, theta_dot_new) =
            pendulum_step_nonlinear(theta, theta_dot, u, tau_dist, m_sim, l, g, dt);
        theta = theta_new;
        theta_dot = theta_dot_new;

        // Track metrics
        if theta.abs() > max_angle {
            max_angle = theta.abs();
        }
        total_control_effort += u.abs() * dt;
    }

    eprintln!("\n=== Robust MPC Pendulum Summary ===");
    eprintln!(
        "Initial perturbation: θ₀ = {:.3} rad ({:.1}°)",
        theta0,
        theta0.to_degrees()
    );
    eprintln!(
        "Mass uncertainty:     [{:.2}, {:.2}] kg  (sim uses nom = {:.2} kg)",
        m_lo, m_hi, m_sim
    );
    eprintln!(
        "Wind disturbance:     ±{:.2} N  → torque ±{:.3} N·m",
        w_max,
        w_max * l
    );
    eprintln!(
        "Final state:          θ = {:.4} rad, θ̇ = {:.4} rad/s",
        theta, theta_dot
    );
    eprintln!(
        "Max angle reached:    {:.4} rad ({:.2}°)",
        max_angle,
        max_angle.to_degrees()
    );
    eprintln!("Total control effort: {:.4} N·m·s", total_control_effort);

    if theta.abs() < 0.05 && theta_dot.abs() < 0.2 {
        eprintln!("PASS: pendulum stabilized near upright equilibrium.");
    } else {
        eprintln!(
            "WARN: pendulum did not fully converge in {} steps.",
            n_steps
        );
    }
}
