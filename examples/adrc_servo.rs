//! ADRC-based DC servo position control with torque disturbance rejection.
//!
//! # System model
//! A DC servo motor modelled as a second-order mechanical system:
//!
//!   J·θ̈ = K_t·i ≈ b·u   (simplified: K_t/J lumped into gain b)
//!
//! with states [θ, θ̇] (position, velocity) and control input u (current/torque).
//! An unknown sinusoidal torque disturbance d(t) acts on the output.
//!
//! # ADRC design (bandwidth parameterization, Gao 2006)
//! - Controller bandwidth  ω_c = 10 rad/s → settling ≈ 0.4 s
//! - Observer bandwidth    ω_o = 5 × ω_c = 50 rad/s (must be faster than ω_c)
//! - High-frequency gain   b   = 1.0  (approximate; ADRC is robust to errors here)
//!
//! The Extended State Observer (ESO) augments [θ, θ̇] with a third state z3 that
//! estimates the *total disturbance* f = d(t) + (unmodeled dynamics).  The control
//! law then cancels f online, reducing the closed-loop to a double integrator
//! tracked by a PD law tuned purely via ω_c.
//!
//! Run: `cargo run --example adrc_servo --features state_feedback`

use oxictl::state_feedback::adrc::SecondOrderAdrc;

/// Euler step of the second-order servo plant.
///
///   θ̈ = b·u + d
///
/// Returns updated [position, velocity].
fn plant_step(state: [f64; 2], u: f64, disturbance: f64, b_plant: f64, dt: f64) -> [f64; 2] {
    let theta = state[0];
    let theta_dot = state[1];
    // Euler integration
    let theta_ddot = b_plant * u + disturbance;
    [theta + dt * theta_dot, theta_dot + dt * theta_ddot]
}

fn main() {
    // --------------- Plant parameters -----------------
    let b_plant = 1.0_f64; // true high-frequency gain J^{-1}·K_t
    let dt = 0.001_f64; // 1 kHz control loop (typical for servo drives)

    // --------------- ADRC tuning (bandwidth-parameterized) ---------------
    // Gao's scaling rule: ω_o ≈ 3–10 × ω_c for good disturbance rejection.
    let omega_c = 10.0_f64; // closed-loop bandwidth (rad/s)
    let omega_o = 50.0_f64; // ESO bandwidth (rad/s) — 5× ω_c

    // Approximate gain fed to ESO; 20% model error deliberately introduced
    // to demonstrate ADRC robustness.
    let b_nominal = b_plant * 0.8;

    let mut ctrl = SecondOrderAdrc::<f64>::new(omega_o, omega_c, b_nominal, dt)
        .expect("ADRC construction: parameters must be positive");

    // --------------- Simulation state ---------------
    let mut plant_state = [0.0_f64; 2]; // [θ, θ̇]
    let mut u_prev = 0.0_f64;

    // --------------- Reference: step at t=0 ---------------
    let r_final = 1.0_f64; // target position (rad)

    // --------------- Metrics ---------------
    let mut rise_time: Option<f64> = None;
    let mut overshoot_peak = 0.0_f64;
    let mut settling_start: Option<f64> = None;
    let mut settling_time: Option<f64> = None;
    let settling_band = 0.02 * r_final; // ±2% settling criterion

    println!(
        "# ADRC Servo: ω_c={:.1} rad/s, ω_o={:.1} rad/s, b_nom={:.2} (true b={:.2})",
        omega_c, omega_o, b_nominal, b_plant
    );
    println!("t_ms,theta,theta_dot,u,z3_dist,d_true");

    let n_steps = 5_000_usize; // 5 s simulation
    for step in 0..n_steps {
        let t = step as f64 * dt;

        // Sinusoidal torque disturbance (1 Hz, amplitude 0.5 N·m)
        // Applied after 0.5 s so we can see the step response first.
        let d_true = if t > 0.5 {
            0.5 * libm::sin(2.0 * core::f64::consts::PI * 1.0 * t)
        } else {
            0.0
        };

        let theta = plant_state[0];
        let r = r_final; // step reference, constant after t=0
        let dr = 0.0_f64; // reference derivative (step → zero)

        // ADRC update: compute control action
        let u = ctrl.update(theta, r, dr, u_prev);

        // Plant simulation step (apply previous u to maintain causality)
        plant_state = plant_step(plant_state, u_prev, d_true, b_plant, dt);
        u_prev = u;

        // ---- Compute metrics ----
        let err = (theta - r_final).abs();
        if rise_time.is_none() && theta >= 0.9 * r_final {
            rise_time = Some(t);
        }
        if theta > overshoot_peak {
            overshoot_peak = theta;
        }
        if err <= settling_band {
            if settling_start.is_none() {
                settling_start = Some(t);
            }
        } else {
            settling_start = None;
        }
        // Settled if within band for at least 50 ms
        if let Some(ts) = settling_start {
            if settling_time.is_none() && (t - ts) >= 0.05 {
                settling_time = Some(ts);
            }
        }

        // Print every 10 ms (every 10th step)
        if step % 10 == 0 {
            println!(
                "{:.1},{:.5},{:.5},{:.5},{:.5},{:.5}",
                t * 1000.0,
                theta,
                plant_state[1],
                u,
                ctrl.disturbance_estimate(),
                d_true,
            );
        }
    }

    // --------------- Summary ---------------
    let final_err = (plant_state[0] - r_final).abs();
    let overshoot_pct = if r_final > 0.0 {
        100.0 * (overshoot_peak - r_final) / r_final
    } else {
        0.0
    };

    eprintln!("\n=== ADRC Servo Summary ===");
    eprintln!(
        "Tuning:      ω_c = {:.1} rad/s,  ω_o = {:.1} rad/s",
        omega_c, omega_o
    );
    eprintln!(
        "Plant gain:  b_true = {:.2},  b_nominal = {:.2}  (20% model error)",
        b_plant, b_nominal
    );
    eprintln!(
        "Rise time:   {}",
        rise_time.map_or("not reached".into(), |v| format!("{:.1} ms", v * 1000.0))
    );
    eprintln!(
        "Settling:    {}",
        settling_time.map_or("not reached".into(), |v| format!(
            "{:.1} ms (±2% band)",
            v * 1000.0
        ))
    );
    eprintln!("Overshoot:   {:.2}%", overshoot_pct.max(0.0));
    eprintln!(
        "Final error: {:.5} rad  (reference = {:.1} rad)",
        final_err, r_final
    );
    eprintln!(
        "ESO disturbance estimate (final): {:.4}",
        ctrl.disturbance_estimate()
    );

    if final_err < 0.05 {
        eprintln!("PASS: position tracking within 5% of reference.");
    } else {
        eprintln!("WARN: position error exceeds 5%.");
    }
}
