//! MPPI obstacle-avoidance demo on a 2D planar system.
//!
//! Models a point mass moving in the plane with state [x, y, vx, vy] and
//! control inputs [ax, ay] (accelerations).  The cost function combines:
//!   - Quadratic tracking cost toward the goal at (5.0, 3.0)
//!   - Repulsive potential from an obstacle centred at (2.0, 1.0)
//!
//! Ten MPPI update iterations are run from the starting position (0, 0)
//! and the best trajectory cost and current state are printed each iteration.
//!
//! Run with:
//!   cargo run --example mppi_obstacle --all-features

use oxictl::mpc::mppi::{Mppi, MppiConfigBuilder};

// ── System parameters ──────────────────────────────────────────────────────────

const DT: f64 = 0.05; // time step [s]

/// Goal position in the plane.
const GOAL: [f64; 2] = [5.0, 3.0];

/// Obstacle centre in the plane.
const OBSTACLE: [f64; 2] = [2.0, 1.0];

/// Obstacle influence radius [m].
const OBS_RADIUS: f64 = 0.8;

/// Obstacle repulsion gain.
const OBS_GAIN: f64 = 5.0;

// ── Dynamics: double-integrator (2D point mass, dt = DT) ─────────────────────

/// x = [px, py, vx, vy],  u = [ax, ay]
fn dynamics(x: &[f64; 4], u: &[f64; 2]) -> [f64; 4] {
    [
        x[0] + DT * x[2] + 0.5 * DT * DT * u[0],
        x[1] + DT * x[3] + 0.5 * DT * DT * u[1],
        x[2] + DT * u[0],
        x[3] + DT * u[1],
    ]
}

// ── Cost function ─────────────────────────────────────────────────────────────

/// Stage and terminal cost combining goal tracking and obstacle repulsion.
///
/// cost = Q_pos * dist_to_goal² + Q_vel * speed² + repulsion + R * u²
fn cost(x: &[f64; 4], u: &[f64; 2], is_terminal: bool) -> f64 {
    let px = x[0];
    let py = x[1];
    let vx = x[2];
    let vy = x[3];

    // Goal tracking (stronger weight at terminal step)
    let q_pos = if is_terminal { 20.0 } else { 1.0 };
    let dx_goal = px - GOAL[0];
    let dy_goal = py - GOAL[1];
    let goal_cost = q_pos * (dx_goal * dx_goal + dy_goal * dy_goal);

    // Velocity damping
    let q_vel = if is_terminal { 2.0 } else { 0.1 };
    let vel_cost = q_vel * (vx * vx + vy * vy);

    // Obstacle repulsion: OBS_GAIN / dist² when inside influence radius
    let dx_obs = px - OBSTACLE[0];
    let dy_obs = py - OBSTACLE[1];
    let dist_sq = dx_obs * dx_obs + dy_obs * dy_obs;
    let dist = libm::sqrt(dist_sq);
    let obs_cost = if dist < OBS_RADIUS {
        // Infinite barrier approximation: penalise heavily near obstacle centre
        OBS_GAIN * (OBS_RADIUS / dist.max(1e-3)).powi(2)
    } else {
        0.0
    };

    // Control effort (only at non-terminal steps)
    let r_ctrl = if is_terminal { 0.0 } else { 0.05 };
    let ctrl_cost = r_ctrl * (u[0] * u[0] + u[1] * u[1]);

    goal_cost + vel_cost + obs_cost + ctrl_cost
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MPPI Obstacle Avoidance: 2D Point Mass ===\n");
    println!("  Start:    (0.0, 0.0)");
    println!("  Goal:     ({:.1}, {:.1})", GOAL[0], GOAL[1]);
    println!(
        "  Obstacle: ({:.1}, {:.1})  radius {:.1} m",
        OBSTACLE[0], OBSTACLE[1], OBS_RADIUS
    );
    println!();

    // ── MPPI configuration ────────────────────────────────────────────────────
    // N=4 (state), I=2 (control: [ax, ay]), H=20 (horizon steps), K=80 (samples)
    let config = MppiConfigBuilder::<f64, 4, 2>::new()
        .temperature(2.0) // moderate temperature for blending
        .sigma_uniform(1.5) // noise std-dev per control channel [m/s²]
        .bounds_symmetric(4.0) // acceleration limits ±4 m/s²
        .gamma(0.98) // slight discounting for receding horizon
        .lcg_seed(20260308)
        .build()
        .expect("MPPI config should be valid");

    let mut mppi: Mppi<f64, 4, 2, 20, 80> =
        Mppi::new(config).expect("MPPI controller should construct");

    // ── Simulation state ──────────────────────────────────────────────────────
    let mut x: [f64; 4] = [0.0, 0.0, 0.0, 0.0];

    println!(
        "{:>6}  {:>12}  {:>8}  {:>8}  {:>8}  {:>8}",
        "iter", "best_cost", "px", "py", "vx", "vy"
    );
    println!("{}", "-".repeat(66));

    for iter in 1..=10 {
        // Run one MPPI update step — returns the first control u*[0]
        let u0 = mppi
            .update(x, dynamics, cost)
            .map_err(|e| format!("MPPI update failed: {}", e))?;

        // Retrieve post-update statistics for reporting
        let stats = mppi.compute_stats();

        println!(
            "{:>6}  {:>12.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
            iter, stats.min_cost, x[0], x[1], x[2], x[3]
        );

        // Apply the computed control to advance the state one step
        x = dynamics(&x, &u0);
    }

    // ── Final report ──────────────────────────────────────────────────────────
    let dx = x[0] - GOAL[0];
    let dy = x[1] - GOAL[1];
    let dist_to_goal = libm::sqrt(dx * dx + dy * dy);

    let dxo = x[0] - OBSTACLE[0];
    let dyo = x[1] - OBSTACLE[1];
    let dist_to_obs = libm::sqrt(dxo * dxo + dyo * dyo);

    println!("\n=== Summary after 10 iterations ===");
    println!(
        "Final state:         px={:.4}  py={:.4}  vx={:.4}  vy={:.4}",
        x[0], x[1], x[2], x[3]
    );
    println!("Distance to goal:    {:.4} m", dist_to_goal);
    println!(
        "Distance to obstacle:{:.4} m (safe radius = {:.1} m)",
        dist_to_obs, OBS_RADIUS
    );

    if dist_to_obs >= OBS_RADIUS {
        println!("Obstacle status: [SAFE] path stays outside obstacle radius");
    } else {
        println!("Obstacle status: [NOTE] short horizon — extend iterations for full avoidance");
    }

    Ok(())
}
