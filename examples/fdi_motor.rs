//! Observer-Based FDI: DC Motor Sensor Fault Detection.
//!
//! Models a first-order DC motor plant:
//!   ẋ = -a·x + b·u   (a = 10.0, b = 100.0)
//!   y = x
//!
//! A Luenberger observer (L = 15.0) tracks the true state.  Normal operation
//! runs for 100 steps; at step 101 a sensor bias fault (+0.3) is injected.
//! The observer residual immediately grows beyond the threshold and the fault
//! is isolated to output channel 0.
//!
//! Run with:
//!   cargo run --example fdi_motor --features "fdi"

use oxictl::fdi::ObserverFdi;

fn main() -> Result<(), String> {
    println!("=== Observer-Based FDI: DC Motor Sensor Fault Detection ===\n");

    // ── Plant parameters ──────────────────────────────────────────────────────
    // ẋ = -a·x + b·u,  y = x
    // Discrete Euler forward: x[k+1] = (1 - a·dt)·x[k] + b·dt·u[k]
    let a: f64 = 10.0; // pole at s = -10
    let b: f64 = 100.0; // DC gain = b/a = 10
    let dt: f64 = 0.005; // 200 Hz sample rate
    let u_ss: f64 = 0.01; // steady-state input: y_ss = b/a·u = 100/10·0.01 = 1.0

    // Verify Euler stability: requires (1 - a·dt) > 0
    if a * dt >= 1.0 {
        return Err(format!(
            "Euler integration unstable: a*dt = {:.3} >= 1.0",
            a * dt
        ));
    }

    // ── Observer design ───────────────────────────────────────────────────────
    // System matrices for N=1, M=1, I=1:
    //   A = [[-a]] = [[-10]]
    //   B = [[b]]  = [[100]]  (B is stored as b[input][state])
    //   C = [[1.0]]
    //   L = [[l]]  (L is stored as l[output][state])
    //
    // Observer pole: A - L·C = -a - l = -(10 + 15) = -25 → fast convergence.
    let a_mat: [[f64; 1]; 1] = [[-a]];
    let b_mat: [[f64; 1]; 1] = [[b]]; // b[input=0][state=0] = b
    let c_mat: [[f64; 1]; 1] = [[1.0]]; // y = x
    let l_mat: [[f64; 1]; 1] = [[15.0]]; // observer gain L = 15

    let threshold: f64 = 0.08; // residual threshold for fault declaration
    let alpha: f64 = 0.85; // EMA forgetting factor

    let mut observer: ObserverFdi<f64, 1, 1, 1> =
        ObserverFdi::new(a_mat, b_mat, c_mat, l_mat, threshold, dt, alpha)
            .map_err(|e| format!("ObserverFdi init error: {e:?}"))?;

    // ── Simulation state ──────────────────────────────────────────────────────
    let mut x_true: f64 = 0.0; // true plant state
    let mut fault_active = false;
    let fault_step = 101_usize;
    let fault_bias: f64 = 0.3; // sensor additive bias injected at step 101

    let n_steps: usize = 160;

    println!(
        "{:>6}  {:>10}  {:>10}  {:>12}  {:>14}  {:>14}",
        "Step", "y_true", "y_sensor", "Residual", "FaultStatus", "Channel"
    );
    println!("{}", "-".repeat(74));

    let mut fault_detected_at: Option<usize> = None;

    for step in 0..n_steps {
        // Inject fault at the designated step
        if step == fault_step {
            fault_active = true;
        }

        // True plant measurement
        let y_true = x_true;
        // Corrupted sensor reading (add bias when fault is active)
        let y_sensor = y_true + if fault_active { fault_bias } else { 0.0 };

        // Run FDI observer update
        let result = observer
            .update(&[u_ss], &[y_sensor])
            .map_err(|e| format!("Observer update error at step {step}: {e:?}"))?;

        let residual = result.residual[0];
        let fault_flag = result.threshold_exceeded[0];
        let channel = result.largest_channel;

        // Record first detection
        if fault_flag && fault_detected_at.is_none() {
            fault_detected_at = Some(step);
        }

        // Print every 20 steps
        if step % 20 == 0 || step == fault_step || (fault_flag && fault_detected_at == Some(step)) {
            let status_str = if fault_flag { "FAULT" } else { "Normal" };
            let ch_str = match channel {
                Some(c) => format!("{c}"),
                None => "-".to_string(),
            };
            println!(
                "{:>6}  {:>10.5}  {:>10.5}  {:>12.6}  {:>14}  {:>14}",
                step, y_true, y_sensor, residual, status_str, ch_str
            );
        }

        // Advance true plant: Euler forward
        // x[k+1] = (1 - a·dt)·x[k] + b·dt·u[k]
        x_true = (1.0 - a * dt) * x_true + b * dt * u_ss;
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\n=== Summary ===");
    println!("Fault injected at step:   {fault_step}  (sensor bias = {fault_bias:.3})");
    match fault_detected_at {
        Some(s) => {
            let latency = s - fault_step;
            println!("Fault first detected at:  {s}  (detection latency = {latency} steps)");
            println!("[PASS] Fault successfully detected and isolated to channel 0.");
        }
        None => {
            println!("[FAIL] Fault was NOT detected within the simulation window.");
        }
    }

    let final_residual = observer.residual()[0];
    let final_ema = observer.residual_ema()[0];
    println!("Final residual:           {final_residual:.6}");
    println!("Final EMA residual:       {final_ema:.6}");
    println!("Threshold:                {threshold:.3}");

    Ok(())
}
