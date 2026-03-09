//! Phase 19 integration example: PSO-based PID auto-tuning
//!
//! Uses Particle Swarm Optimization to search for PID gains that minimise
//! the Integral Absolute Error (IAE) of a first-order discrete plant's
//! step response.
//!
//! Plant: y[k+1] = 0.7 * y[k] + 0.3 * u[k]
//!
//! Decision variables: [Kp, Ki, Kd] ∈ [0, 20]³
//!
//! PSO settings:
//! - 10 particles, 3 dimensions
//! - Inertia w=0.7, cognitive c1=2.0, social c2=2.0
//! - 30 iterations

use oxictl::optim::particle_swarm::ParticleSwarm;

/// Simulate a step response and return IAE (integral absolute error).
///
/// Plant: y[k+1] = 0.7 * y[k] + 0.3 * u[k]
/// PID:   u[k]  = Kp*e[k] + Ki*∑e + Kd*(e[k]-e[k-1])
/// Reference: r = 1.0 (unit step)
fn iae_cost(gains: &[f64; 3]) -> f64 {
    const HORIZON: usize = 60;
    const SETPOINT: f64 = 1.0;
    const U_CLAMP: f64 = 20.0;

    let kp = gains[0];
    let ki = gains[1];
    let kd = gains[2];

    let mut y = 0.0_f64;
    let mut integral_e = 0.0_f64;
    let mut e_prev = 0.0_f64;
    let mut iae = 0.0_f64;

    for _ in 0..HORIZON {
        let e = SETPOINT - y;
        integral_e += e;
        let derivative = e - e_prev;
        e_prev = e;

        let u = (kp * e + ki * integral_e + kd * derivative).clamp(-U_CLAMP, U_CLAMP);

        // Advance plant: y[k+1] = 0.7*y[k] + 0.3*u[k]
        y = 0.7 * y + 0.3 * u;

        iae += e.abs();
    }

    iae
}

fn main() -> Result<(), String> {
    // --- PSO setup -------------------------------------------------------
    // 3 decision variables: [Kp, Ki, Kd], each bounded in [0.0, 20.0].
    let bounds_min = [0.0_f64; 3];
    let bounds_max = [20.0_f64; 3];

    // ParticleSwarm<S=f64, D=3 dims, N=10 particles>
    let mut pso = ParticleSwarm::<f64, 3, 10>::new(
        bounds_min, bounds_max, 0.7, // inertia weight w ∈ (0, 1.5)
        2.0, // cognitive coefficient c1
        2.0, // social coefficient c2
        42,  // LCG seed for reproducibility
    )
    .map_err(|e| format!("ParticleSwarm::new failed: {e}"))?;

    println!("PSO PID Tuning — plant: y[k+1] = 0.7·y[k] + 0.3·u[k]");
    println!("Decision space: Kp, Ki, Kd ∈ [0, 20]");
    println!("Particles: 10,  Iterations: 30,  Horizon: 60 steps");
    println!("{}", "─".repeat(55));

    // --- optimisation loop -----------------------------------------------
    const MAX_ITER: usize = 30;

    for iter in 0..MAX_ITER {
        pso.step(&iae_cost)
            .map_err(|e| format!("PSO step {iter} failed: {e}"))?;

        if (iter + 1) % 10 == 0 {
            let pos = pso.best_position();
            println!(
                "Iter {:>3}: IAE = {:>8.4}  Kp={:.4}  Ki={:.4}  Kd={:.4}",
                iter + 1,
                pso.best_value(),
                pos[0],
                pos[1],
                pos[2],
            );
        }
    }

    // --- results ---------------------------------------------------------
    let best_iae = pso.best_value();
    let best_pos = pso.best_position();
    let kp_best = best_pos[0];
    let ki_best = best_pos[1];
    let kd_best = best_pos[2];

    println!("{}", "─".repeat(55));
    println!("Best PID gains found by PSO:");
    println!("  Kp = {kp_best:.6}");
    println!("  Ki = {ki_best:.6}");
    println!("  Kd = {kd_best:.6}");
    println!("  IAE over 60-step horizon = {best_iae:.6}");

    // Sanity: IAE must be finite and non-negative.
    if !best_iae.is_finite() || best_iae < 0.0 {
        return Err(format!("Unexpected IAE value: {best_iae}"));
    }

    println!("PSO tuning complete after {MAX_ITER} iterations.");

    Ok(())
}
