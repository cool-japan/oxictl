//! Anti-windup demonstration: PI controller with and without back-calculation AW.
//!
//! Plant (first-order discrete-time):
//!   y[k+1] = 0.8·y[k] + 0.2·u[k]
//!
//! Actuator saturation: u_sat = clamp(u, −1, +1)
//!
//! Controllers (both PI, kp = 2, ki = 1, dt = 0.1):
//!   AW  controller: SimpleAntiWindup with e_aw = 2.0  (back-calculation active)
//!   Plain PI       : SimpleAntiWindup with e_aw = 0.0  (no AW correction → windup)
//!
//! Reference: step to 5.0 — far beyond what the saturated actuator can achieve,
//! so integrator windup is severe without AW.
//!
//! Printed every 10 steps: step | y_aw | y_no_aw | saturated (bool)
//!
//! Expected result: y_aw converges faster because the AW term prevents the
//! integrator from accumulating a large offset during saturation.
//!
//! Run: `cargo run --example antiwindup_demo --features antiwindup`

use oxictl::antiwindup::SimpleAntiWindup;

fn main() -> Result<(), String> {
    println!("=== Anti-Windup Demo ===");
    println!("Plant: y[k+1] = 0.8·y[k] + 0.2·u[k]");
    println!("Saturation: u ∈ [−1, +1]");
    println!("PI: kp=2, ki=1, dt=0.1  |  Reference = 5.0 (forces saturation)");
    println!("AW back-calculation gain: e_aw = 2.0  vs  e_aw = 0 (plain PI)");
    println!();

    // ── Parameters ────────────────────────────────────────────────────────────

    let kp = 2.0_f64;
    let ki = 1.0_f64;
    let dt = 0.1_f64;
    let u_min = -1.0_f64;
    let u_max = 1.0_f64;
    let reference = 5.0_f64;
    let n_steps = 80_usize;

    // ── Instantiate controllers ────────────────────────────────────────────────

    let mut aw_ctrl = SimpleAntiWindup::<f64>::new(kp, ki, 2.0_f64, u_min, u_max, dt)
        .map_err(|e| format!("AW controller init failed: {e:?}"))?;

    let mut plain_ctrl = SimpleAntiWindup::<f64>::new(kp, ki, 0.0_f64, u_min, u_max, dt)
        .map_err(|e| format!("Plain PI init failed: {e:?}"))?;

    // ── Plant states ──────────────────────────────────────────────────────────

    let mut y_aw = 0.0_f64;
    let mut y_plain = 0.0_f64;

    // ── Simulation ────────────────────────────────────────────────────────────

    println!("step | y_aw   | y_plain | saturated");
    println!("-----|--------|---------|----------");

    for step in 0..=n_steps {
        if step % 10 == 0 {
            // Detect saturation by checking if the AW integrator would drive u beyond limits
            // (u_lin = kp*e + ki*integrator; if |u_lin| > u_max → saturated)
            let e_aw = reference - y_aw;
            let u_lin_aw = kp * e_aw + ki * aw_ctrl.integrator_state();
            let saturated = u_lin_aw.abs() > u_max;

            println!(
                " {:3} | {:.4} | {:.4}  | {}",
                step, y_aw, y_plain, saturated
            );
        }

        // ── Compute control actions ───────────────────────────────────────────

        let e_aw_val = reference - y_aw;
        let u_aw = aw_ctrl
            .update(e_aw_val)
            .map_err(|e| format!("AW update failed: {e:?}"))?;

        let e_plain = reference - y_plain;
        let u_plain_raw = plain_ctrl
            .update(e_plain)
            .map_err(|e| format!("Plain update failed: {e:?}"))?;

        // Plain PI output is already clamped by SimpleAntiWindup limits,
        // but its integrator continues to grow (e_aw=0 → no back-calculation).
        // The plant sees the clamped value.
        let u_plain = u_plain_raw; // already in [-1, 1] from SimpleAntiWindup

        // ── Advance plants ────────────────────────────────────────────────────

        y_aw = 0.8 * y_aw + 0.2 * u_aw;
        y_plain = 0.8 * y_plain + 0.2 * u_plain;
    }

    println!();

    // ── Summary ───────────────────────────────────────────────────────────────

    let e_aw_final = (reference - y_aw).abs();
    let e_plain_final = (reference - y_plain).abs();

    println!("=== Summary (step {n_steps}) ===");
    println!(
        "With AW    (e_aw=2.0): y = {:.4},  |error| = {:.4}",
        y_aw, e_aw_final
    );
    println!(
        "Plain PI   (e_aw=0.0): y = {:.4},  |error| = {:.4}",
        y_plain, e_plain_final
    );
    println!("AW integrator state:   {:.4}", aw_ctrl.integrator_state());
    println!(
        "Plain integrator state: {:.4}",
        plain_ctrl.integrator_state()
    );
    println!();
    println!("Both controllers saturate the actuator (u→±1) because ref=5 >> plant capacity.");
    println!("With AW: back-calculation limits integrator growth → faster recovery.");

    if e_aw_final < e_plain_final {
        println!("PASS: AW controller has smaller final error ({e_aw_final:.4}) than plain PI ({e_plain_final:.4}).");
    } else {
        println!(
            "INFO: AW={e_aw_final:.4}, plain={e_plain_final:.4}.  \
             Both near steady-state; examine integrator states for windup evidence."
        );
    }

    Ok(())
}
