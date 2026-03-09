//! IDA-PBC on a mass-spring-damper system.
//!
//! Demonstrates Interconnection and Damping Assignment Passivity-Based Control
//! on a simple mechanical port-Hamiltonian system:
//!
//!   ẋ = [J - R] ∂H/∂x + g·u     (port-Hamiltonian form)
//!
//! Plant: mass m=1, spring k=1, damper b=0.1
//!   H(x) = ½(q² + p²)  [q = displacement, p = momentum]
//!   J = [[0, 1],[-1, 0]],  R = [[0, 0],[0, 0.1]],  g = [[0],[1]]
//!
//! IDA-PBC design: shift equilibrium to q* = 1.0 with added damping r_a = 1.0
//!   H_d(x) = ½((q-1)² + p²)
//!   J_d = J,  R_d = [[0,0],[0, 1.1]] = R + R_a
//!   ∂H_d/∂x = [q-1, p]
//!
//! Energy (Hamiltonian H) is printed alongside position and velocity each step.
//!
//! Run with:
//!   cargo run --example passive_control --all-features

use oxictl::passivity::{IdaPbcConfig, IdaPbcController};

// ── Plant matrices ────────────────────────────────────────────────────────────

/// Plant interconnection matrix J (skew-symmetric).
const J_PLANT: [[f64; 2]; 2] = [[0.0, 1.0], [-1.0, 0.0]];

/// Plant damping matrix R (PSD; b=0.1 velocity damping).
const R_PLANT: [[f64; 2]; 2] = [[0.0, 0.0], [0.0, 0.1]];

/// Input matrix g (force applied to momentum channel).
const G_PLANT: [[f64; 1]; 2] = [[0.0], [1.0]];

// ── Gradient functions ────────────────────────────────────────────────────────

/// Gradient of plant Hamiltonian H(x) = ½(q² + p²):
///   ∂H/∂x = [q, p]
fn grad_h_plant(x: &[f64; 2]) -> [f64; 2] {
    [x[0], x[1]]
}

/// Gradient of desired Hamiltonian H_d(x) = ½((q-1)² + p²):
///   ∂H_d/∂x = [q-1, p]
fn grad_hd(x: &[f64; 2]) -> [f64; 2] {
    [x[0] - 1.0, x[1]]
}

/// Evaluate plant Hamiltonian H(x) = ½(q² + p²).
fn hamiltonian(x: [f64; 2]) -> f64 {
    0.5 * (x[0] * x[0] + x[1] * x[1])
}

/// Evaluate desired Hamiltonian H_d(x) = ½((q-1)² + p²).
fn hamiltonian_d(x: [f64; 2]) -> f64 {
    let dq = x[0] - 1.0;
    0.5 * (dq * dq + x[1] * x[1])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IDA-PBC: Mass-Spring-Damper Closed-Loop Simulation ===\n");
    println!("  Plant:  m=1, k=1, b=0.1");
    println!("  Design: desired equilibrium q* = 1.0, added damping r_a = 1.0");
    println!("  Start:  q=0.0, p=0.0");
    println!();

    // ── IDA-PBC controller design ─────────────────────────────────────────────
    // Desired interconnection: same as plant J (canonical symplectic).
    let j_desired: [[f64; 2]; 2] = [[0.0, 1.0], [-1.0, 0.0]];

    // Desired damping: plant damping R + additional damping R_a = diag(0, 1.0).
    // R_d must be PSD; here b_total = 0.1 + 1.0 = 1.1.
    let r_desired: [[f64; 2]; 2] = [[0.0, 0.0], [0.0, 1.1]];

    let ida_config = IdaPbcConfig::<f64, 2, 1>::new(j_desired, r_desired, grad_hd)
        .expect("IDA-PBC config should be valid: J_d skew-symmetric, R_d PSD");

    let controller = IdaPbcController::new(ida_config);

    // ── Simulation ────────────────────────────────────────────────────────────
    let dt = 0.05_f64;
    let mut x: [f64; 2] = [0.0, 0.0]; // [q, p]

    println!(
        "{:>6}  {:>12}  {:>12}  {:>14}  {:>14}",
        "step", "position_q", "velocity_p", "H (plant)", "H_d (desired)"
    );
    println!("{}", "-".repeat(66));

    for step in 0..=20_usize {
        let h_val = hamiltonian(x);
        let hd_val = hamiltonian_d(x);

        println!(
            "{:>6}  {:>12.6}  {:>12.6}  {:>14.6}  {:>14.6}",
            step, x[0], x[1], h_val, hd_val
        );

        if step == 20 {
            break;
        }

        // Compute IDA-PBC control law: u = (gᵀg)⁻¹ gᵀ { [J_d-R_d]∇H_d − [J-R]∇H }
        let u = controller
            .compute_with_closures(&J_PLANT, &R_PLANT, &G_PLANT, grad_h_plant, &x)
            .map_err(|e| format!("IDA-PBC compute failed: {}", e))?;

        // Plant dynamics: ẋ = (J-R)∇H + g·u
        let grad_h = grad_h_plant(&x);

        // (J-R)·∇H
        let jr00 = J_PLANT[0][0] - R_PLANT[0][0];
        let jr01 = J_PLANT[0][1] - R_PLANT[0][1];
        let jr10 = J_PLANT[1][0] - R_PLANT[1][0];
        let jr11 = J_PLANT[1][1] - R_PLANT[1][1];

        let jrg_q = jr00 * grad_h[0] + jr01 * grad_h[1];
        let jrg_p = jr10 * grad_h[0] + jr11 * grad_h[1];

        // g·u (u is scalar; g = [[0],[1]])
        let gu_q = G_PLANT[0][0] * u[0];
        let gu_p = G_PLANT[1][0] * u[0];

        let xdot = [jrg_q + gu_q, jrg_p + gu_p];

        // Euler integration
        x = [x[0] + dt * xdot[0], x[1] + dt * xdot[1]];
    }

    // ── Final assessment ──────────────────────────────────────────────────────
    let h_final = hamiltonian(x);
    let hd_final = hamiltonian_d(x);
    let err_q = (x[0] - 1.0).abs();
    let err_p = x[1].abs();

    println!("\n=== Summary ===");
    println!(
        "Final position q:       {:.6}  (target q* = 1.0, error = {:.6})",
        x[0], err_q
    );
    println!(
        "Final momentum p:       {:.6}  (target p* = 0.0, error = {:.6})",
        x[1], err_p
    );
    println!("Final H  (plant):       {:.6}", h_final);
    println!(
        "Final H_d (desired):    {:.6}  (minimum = 0.0 at equilibrium)",
        hd_final
    );

    if err_q < 0.1 && err_p < 0.1 {
        println!("Result: [CONVERGED] state within 0.1 of desired equilibrium");
    } else {
        println!("Result: [CONVERGING] extend simulation for full convergence");
    }

    Ok(())
}
