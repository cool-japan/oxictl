//! Phase 19 integration example: VRFT (Virtual Reference Feedback Tuning)
//!
//! Tunes a PID controller from a single open-loop step experiment without
//! requiring an explicit plant model, using the Campi–Savaresi 2002 algorithm.
//!
//! Plant (first-order discrete, open-loop step):
//!   y[k] = 0.8 * y[k-1] + 0.2 * u[k-1]
//!
//! Experiment:
//!   u[k] = 1.0  for all k  (constant step input)
//!   DATA_LEN = 50 samples
//!
//! Reference model:
//!   M(z) = (1 - m) / (z - m),  m = 0.5  (pole at 0.5)
//!
//! Output: tuned Kp, Ki, Kd

use oxictl::data_driven::VrftPid;

fn main() -> Result<(), String> {
    // --- Experiment parameters -------------------------------------------
    const DATA_LEN: usize = 50;
    // Reference model pole m ∈ (0, 1). Choosing m=0.5 gives a moderately
    // fast closed-loop response (time constant ≈ 1/ln(2) ≈ 1.44 samples).
    const M: f64 = 0.5;
    // Sampling period: 1 second (normalised discrete-time experiment).
    const DT: f64 = 1.0;

    // --- Generate open-loop step-response data ---------------------------
    // Input: constant unit step.
    let u_data = [1.0_f64; DATA_LEN];

    // Plant: y[k] = 0.8 * y[k-1] + 0.2 * u[k-1]
    let mut y_data = [0.0_f64; DATA_LEN];
    for k in 1..DATA_LEN {
        y_data[k] = 0.8 * y_data[k - 1] + 0.2 * u_data[k - 1];
    }

    println!("VRFT PID Tuning — open-loop step experiment");
    println!(
        "Plant:      y[k] = 0.8·y[k-1] + 0.2·u[k-1],  {} samples",
        DATA_LEN
    );
    println!("Ref model:  M(z) = (1-m)/(z-m),  m = {M},  dt = {DT}");
    println!("{}", "─".repeat(50));

    // Print first few and last few samples as a sanity check.
    println!("Sample data (first 5 steps):");
    println!("  {:>5}  {:>10}  {:>10}", "k", "u[k]", "y[k]");
    for k in 0..5_usize.min(DATA_LEN) {
        println!("  {:>5}  {:>10.5}  {:>10.5}", k, u_data[k], y_data[k]);
    }
    if DATA_LEN > 10 {
        println!("  ...");
        for k in (DATA_LEN - 3)..DATA_LEN {
            println!("  {:>5}  {:>10.5}  {:>10.5}", k, u_data[k], y_data[k]);
        }
    }
    println!("{}", "─".repeat(50));

    // --- VRFT tuning -----------------------------------------------------
    let mut tuner =
        VrftPid::<f64, DATA_LEN>::new(M, DT).map_err(|e| format!("VrftPid::new failed: {e}"))?;

    tuner
        .tune(&u_data, &y_data)
        .map_err(|e| format!("VrftPid::tune failed: {e}"))?;

    let kp = tuner.kp().map_err(|e| format!("kp() failed: {e}"))?;
    let ki = tuner.ki().map_err(|e| format!("ki() failed: {e}"))?;
    let kd = tuner.kd().map_err(|e| format!("kd() failed: {e}"))?;

    // --- Results ---------------------------------------------------------
    println!("VRFT tuned PID gains:");
    println!("  Kp = {kp:.8}");
    println!("  Ki = {ki:.8}");
    println!("  Kd = {kd:.8}");
    println!();
    println!("Reference model pole: m = {}", tuner.reference_model_pole());
    println!(
        "Tuning status: {}",
        if tuner.is_tuned() {
            "complete"
        } else {
            "not tuned"
        }
    );

    // Sanity: gains must be finite.
    if !kp.is_finite() || !ki.is_finite() || !kd.is_finite() {
        return Err(format!(
            "Non-finite gains returned: Kp={kp}, Ki={ki}, Kd={kd}"
        ));
    }

    // At least one gain should be non-trivially non-zero to indicate that
    // the algorithm produced a meaningful result from the step-response data.
    let gain_magnitude = kp.abs() + ki.abs() + kd.abs();
    if gain_magnitude < 1e-10 {
        return Err(format!(
            "All gains are effectively zero (magnitude={gain_magnitude:.2e}); \
             check that the experiment data is sufficiently informative."
        ));
    }

    println!("\nVRFT tuning completed successfully.");

    Ok(())
}
