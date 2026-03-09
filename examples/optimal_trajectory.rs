//! Optimal control: minimum-energy trajectory for a double integrator.
//!
//! Uses single shooting with gradient descent (Armijo line search) to find
//! the minimum-energy control sequence driving the double integrator:
//!
//!   ẋ₁ = x₂         (position rate = velocity)
//!   ẋ₂ = u           (velocity rate = force/mass)
//!
//! from x₀ = [2.0, 0.0] to near the origin over a 2-second horizon.
//!
//! Run with:
//!   cargo run --example optimal_trajectory --features "optimal"

use oxictl::optimal::{ControlConstraints, SingleShootingProblem};

// ── System dynamics: double integrator ────────────────────────────────────────
fn dynamics(x: &[f64; 2], u: &[f64; 1]) -> [f64; 2] {
    [x[1], u[0]]
}

// ── Stage cost: minimum energy (quadratic control effort) ─────────────────────
fn stage_cost(_x: &[f64; 2], u: &[f64; 1]) -> f64 {
    u[0] * u[0]
}

// ── Terminal cost: penalise deviation from the origin ─────────────────────────
fn terminal_cost(x: &[f64; 2]) -> f64 {
    // High penalty on terminal state to enforce approximate state constraint
    50.0 * (x[0] * x[0] + x[1] * x[1])
}

fn main() {
    println!("=== Optimal Trajectory: Minimum-Energy Double Integrator ===\n");

    // ── Problem setup ──────────────────────────────────────────────────────────
    // Horizon: M=20 intervals × dt=0.1 s = 2 seconds total
    // Control bounds: u ∈ [-3, 3] m/s²
    const M: usize = 20;
    let x0 = [2.0_f64, 0.0]; // Initial state: position=2, velocity=0

    let constraints = ControlConstraints::box_input([-3.0_f64], [3.0_f64]);
    let mut prob: SingleShootingProblem<f64, 2, 1, M> = SingleShootingProblem::new(
        dynamics,
        stage_cost,
        terminal_cost,
        0.1_f64, // dt
        constraints,
    );

    // Solver tuning
    prob.max_iter = 600;
    prob.step_size = 0.02;
    prob.tol = 1e-6;
    prob.ode_steps = 4;
    prob.armijo_beta = 0.5;
    prob.armijo_c1 = 1e-4;

    // ── Initial guess: zero control (bang does nothing) ──────────────────────
    let u_init = [[0.0_f64]; M];
    let j_init = prob
        .cost(&x0, &u_init)
        .expect("Initial cost evaluation should succeed");

    println!("Initial cost (u=0): {:.4}", j_init);
    println!("Initial state:      x = [{:.3}, {:.3}]", x0[0], x0[1]);
    println!("Solving...");

    // ── Solve optimal control problem ─────────────────────────────────────────
    let (u_opt, j_opt) = prob
        .solve(&x0, &u_init)
        .expect("Single shooting solver should converge");

    println!("Optimal cost:       {:.4}", j_opt);
    println!(
        "Cost reduction:     {:.2}%\n",
        100.0 * (j_init - j_opt) / j_init
    );

    // ── Compute and print state trajectory ────────────────────────────────────
    let traj = prob
        .trajectory(&x0, &u_opt)
        .expect("Trajectory computation should succeed");

    println!(
        "{:>6}  {:>8}  {:>12}  {:>12}  {:>12}",
        "Step", "Time(s)", "Position", "Velocity", "Control"
    );
    println!("{}", "-".repeat(58));

    // Print initial state
    println!(
        "{:>6}  {:>8.2}  {:>12.5}  {:>12.5}  {:>12.5}",
        0, 0.0, x0[0], x0[1], 0.0
    );

    for k in 0..M {
        let t = (k + 1) as f64 * 0.1;
        let xk = traj[k];
        let uk = u_opt[k][0];
        println!(
            "{:>6}  {:>8.2}  {:>12.5}  {:>12.5}  {:>12.5}",
            k + 1,
            t,
            xk[0],
            xk[1],
            uk
        );
    }

    // ── Analysis: control sequence statistics ─────────────────────────────────
    let u_max = u_opt.iter().map(|u| u[0].abs()).fold(0.0_f64, f64::max);
    let u_rms = (u_opt.iter().map(|u| u[0] * u[0]).sum::<f64>() / M as f64).sqrt();
    let energy = u_opt.iter().map(|u| u[0] * u[0] * 0.1).sum::<f64>();

    let final_state = traj[M - 1];
    let final_norm = (final_state[0].powi(2) + final_state[1].powi(2)).sqrt();

    println!("\n=== Trajectory Analysis ===");
    println!(
        "Final state:       x = [{:.5}, {:.5}]",
        final_state[0], final_state[1]
    );
    println!("Final state norm:  {:.5}", final_norm);
    println!("Max control:       |u_max| = {:.4}", u_max);
    println!("RMS control:       {:.4}", u_rms);
    println!("Total energy:      ∫u² dt  = {:.4}", energy);

    // ── Verify results ────────────────────────────────────────────────────────
    println!("\n=== Verification ===");

    if j_opt < j_init {
        println!("[PASS] Optimised cost < initial cost");
    } else {
        println!("[FAIL] Optimised cost >= initial cost");
    }

    if u_max <= 3.0 + 1e-9 {
        println!("[PASS] Control within bounds [-3, 3]");
    } else {
        println!("[FAIL] Control violates bounds: |u_max|={:.4}", u_max);
    }

    if final_norm < 1.0 {
        println!(
            "[PASS] Final state norm {:.4} < 1.0 (near origin)",
            final_norm
        );
    } else {
        println!(
            "[WARN] Final state norm {:.4} >= 1.0 (consider more iterations)",
            final_norm
        );
    }

    println!("\nDone. Minimum-energy double integrator trajectory computed.");
}
