//! Phase 19 integration example: Plug-in Repetitive Control
//!
//! Demonstrates periodic disturbance rejection using [`RepetitiveController`].
//!
//! Plant (first-order discrete):
//!   y[k] = 0.8 * y[k-1] + 0.2 * u[k]
//!
//! Periodic disturbance (period = 20 samples):
//!   d[k] = 0.3 * sin(2π * k / 20)
//!
//! Architecture:
//! - Base PID feedback: Kp=2.0, Ki=0.1, Kd=0.05
//! - Plug-in RC (q=0.95, kr=0.8, N=20) adds u_rep to cancel the disturbance
//!
//! After the first period (20 steps) the repetitive signal begins to learn
//! the disturbance shape and the RMS convergence metric decreases.

use oxictl::repetitive::RepetitiveController;

/// Discrete-time first-order plant with additive periodic disturbance.
///
/// State equation: y[k] = 0.8 * y[k-1] + 0.2 * (u[k] + d[k])
struct Plant {
    y_prev: f64,
}

impl Plant {
    fn new() -> Self {
        Self { y_prev: 0.0 }
    }

    /// Advance one step. Returns y[k].
    fn step(&mut self, u: f64, disturbance: f64) -> f64 {
        let y = 0.8 * self.y_prev + 0.2 * (u + disturbance);
        self.y_prev = y;
        y
    }
}

/// Discrete-time incremental PID controller (no anti-windup for brevity).
struct Pid {
    kp: f64,
    ki: f64,
    kd: f64,
    integral: f64,
    e_prev: f64,
}

impl Pid {
    fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            e_prev: 0.0,
        }
    }

    /// Compute the PID output for the current error.
    fn compute(&mut self, error: f64) -> f64 {
        self.integral += error;
        let derivative = error - self.e_prev;
        self.e_prev = error;
        self.kp * error + self.ki * self.integral + self.kd * derivative
    }
}

fn main() -> Result<(), String> {
    // --- setup -----------------------------------------------------------
    const PERIOD: usize = 20;
    const STEPS: usize = 100;
    const SETPOINT: f64 = 1.0;
    const TWO_PI: f64 = 2.0 * core::f64::consts::PI;

    // Plug-in repetitive controller: q=0.95, kr=0.8, period=20 samples.
    let mut rc = RepetitiveController::<f64, PERIOD>::new(0.95, 0.8)
        .map_err(|e| format!("RepetitiveController::new failed: {e}"))?;

    let mut pid = Pid::new(2.0, 0.1, 0.05);
    let mut plant = Plant::new();

    println!(
        "{:>5}  {:>10}  {:>10}  {:>10}",
        "step", "y", "error", "u_rep"
    );
    println!("{}", "-".repeat(45));

    let mut error_norm_period0 = 0.0_f64; // |error| sum over first period
    let mut error_norm_later = 0.0_f64; // |error| sum over last 20 steps

    // --- simulation loop -------------------------------------------------
    for k in 0..STEPS {
        // Periodic disturbance with known period N=20.
        let d = 0.3 * (TWO_PI * (k as f64) / PERIOD as f64).sin();

        // Measure current output (from previous step's u; plant is updated below).
        // Compute error against setpoint.
        let y_meas = plant.y_prev; // current output before applying new u
        let error = SETPOINT - y_meas;

        // Base PID feedback signal.
        let u_pid = pid.compute(error);

        // Repetitive control correction (learns periodic disturbance shape).
        let u_rep = rc
            .update(error)
            .map_err(|e| format!("rc.update failed at step {k}: {e}"))?;

        // Combined control input.
        let u_total = u_pid + u_rep;

        // Advance plant with disturbance injected at input.
        let y_new = plant.step(u_total, d);

        // Accumulate error norm for convergence analysis.
        if k < PERIOD {
            error_norm_period0 += error.abs();
        } else if k >= STEPS - PERIOD {
            error_norm_later += error.abs();
        }

        // Print every 20 steps.
        if k % 20 == 0 {
            println!(
                "{:>5}  {:>10.5}  {:>10.5}  {:>10.5}",
                k, y_new, error, u_rep
            );
        }
    }

    // --- convergence summary ---------------------------------------------
    println!("{}", "-".repeat(45));
    println!("Error norm (first period, steps 0-19):    {error_norm_period0:.5}");
    println!("Error norm (last period, steps 80-99):    {error_norm_later:.5}");
    println!(
        "RMS convergence metric:                    {:.6}",
        rc.rms_convergence()
    );

    // Verify the repetitive controller is driving the error down.
    if error_norm_later < error_norm_period0 {
        println!("Convergence confirmed: error norm decreased after first period.");
    } else {
        println!("Note: error norms: first={error_norm_period0:.5}, later={error_norm_later:.5}");
    }

    Ok(())
}
