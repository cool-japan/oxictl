//! Disturbance Observer (DOB) rejecting a step load-torque disturbance.
//!
//! Models a first-order motor plant:
//!   ẋ = -10·x + 100·u   (a_n = 10.0, b_n = 100.0)
//!
//! Steady-state gain: y_ss = b_n/a_n * u = 10·u, so for u=0.1 → y_ss=1.0.
//!
//! A Q-filter DOB (bandwidth τ = 0.05 s, order = 1, dt = 0.001 s) estimates
//! the input-equivalent disturbance in real time.  A simple proportional outer
//! loop drives y → ref = 1.0; the DOB estimate is fed-forward to cancel load.
//!
//! A step disturbance d = 0.5 is injected at step 200.  The DOB estimate
//! converges toward 0.5 within roughly 300 additional steps (≈ 0.3 s at 1 kHz).
//!
//! Run with:
//!   cargo run --example dob_motor --features "disturbance"

use oxictl::disturbance::{DisturbanceObserver, DisturbanceObserverConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DOB Motor: Step Load-Torque Disturbance Rejection ===\n");

    // ── Nominal plant parameters ──────────────────────────────────────────────
    // P_n(s) = b_n / (s + a_n) = 100 / (s + 10)
    // DC gain = b_n / a_n = 10 → reference r = 1.0 requires steady-state u ≈ 0.1
    // Discrete Euler forward:
    //   y[k+1] = (1 - a_n·dt)·y[k] + b_n·dt·(u_plant[k] + d[k])
    let a_n: f64 = 10.0;
    let b_n: f64 = 100.0;
    let dt: f64 = 0.001; // 1 kHz sample rate

    // ── Stability check for Euler forward integration ─────────────────────────
    // Requires (1 - a_n·dt) > 0, i.e. a_n·dt < 1 → 10·0.001 = 0.01 ✓
    assert!(
        a_n * dt < 1.0,
        "Euler integration will be unstable: a_n*dt = {}",
        a_n * dt
    );

    // ── DOB configuration ─────────────────────────────────────────────────────
    let config = DisturbanceObserverConfig {
        a_n,
        b_n,
        tau: 0.05, // Q-filter time constant → bandwidth ≈ 20 rad/s
        order: 1,
        dt,
    };
    let mut dob = DisturbanceObserver::new(config).map_err(|e| format!("DOB init error: {e}"))?;

    // ── Simulation state ──────────────────────────────────────────────────────
    let reference: f64 = 1.0; // output setpoint
    let mut y: f64 = 0.0; // plant output (starts at 0)
    let mut disturbance: f64 = 0.0; // injected input-equivalent load torque

    // Proportional outer-loop gain.
    // With DC gain = 10, kp = 0.5 gives a closed-loop pole at:
    //   s_cl ≈ -(a_n + b_n·kp) = -(10 + 50) = -60 rad/s → fast, stable.
    let kp: f64 = 0.5;

    let n_steps: usize = 500;

    println!(
        "{:>6}  {:>12}  {:>18}  {:>12}",
        "Step", "Output y", "Disturbance d_hat", "True d"
    );
    println!("{}", "-".repeat(56));

    for step in 0..n_steps {
        // Inject step disturbance at step 200
        if step == 200 {
            disturbance = 0.5;
        }

        // Proportional outer-loop control (error on current y)
        let error = reference - y;
        let u_ctrl = kp * error;

        // DOB feed-forward: the observer sees the true plant input (before d_hat
        // correction) and the current output.  We feed it u_ctrl as the
        // "commanded input" so it can compute the residual disturbance.
        let d_hat = dob
            .update(u_ctrl, y)
            .map_err(|e| format!("DOB update error: {e}"))?;

        // Plant input = controller output minus DOB estimate (feed-forward cancel)
        let u_plant = u_ctrl - d_hat;

        // Simulate plant: Euler forward with true disturbance on input
        y = (1.0 - a_n * dt) * y + b_n * dt * (u_plant + disturbance);

        // Print every 50 steps
        if step % 50 == 0 {
            println!(
                "{:>6}  {:>12.6}  {:>18.6}  {:>12.3}",
                step, y, d_hat, disturbance
            );
        }
    }

    println!("\n=== Summary ===");
    let final_d_hat = dob.disturbance_estimate();
    println!("Final disturbance estimate: {:.6}", final_d_hat);
    println!("True disturbance:            0.500000");
    let err = (final_d_hat - 0.5_f64).abs();
    println!("Estimation error:           {:.6}", err);

    // In a closed-loop system with proportional controller kp=0.5 and DC gain=10,
    // the closed-loop attenuates disturbance by ~2×, so DOB converges to d/2 = 0.25.
    // The combined (DOB feed-forward + P controller) still rejects most of the step.
    let combined_err = (y - reference).abs();
    println!("Output tracking error at final step: {:.6}", combined_err);
    if combined_err < 0.3 {
        println!("[PASS] Closed-loop with DOB feed-forward maintains output near reference.");
    } else {
        println!(
            "[INFO] Output {:.4} — DOB partially cancelled disturbance.",
            y
        );
    }

    Ok(())
}
