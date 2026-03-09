//! Switched LTI control: two-mode plant with mode-dependent dynamics.
//!
//! Plant modes (SISO, 1-dimensional state):
//!   Mode 0 (fast): x[k+1] = 0.7·x[k] + 1.0·u[k],   y = x[k+1]
//!   Mode 1 (slow): x[k+1] = 0.9·x[k] + 0.5·u[k],   y = x[k+1]
//!
//! Controller: proportional  u = 0.5·(r − y)
//! Minimum dwell time: 5 steps (prevents chattering).
//!
//! Simulation:
//!   Steps  0–19:  mode 0 (fast settling)
//!   Step   20:    attempt switch to mode 1
//!   Steps 20–39:  mode 1 (slower, smaller input gain)
//!
//! Output printed every 5 steps: step | mode | output y
//!
//! Run: `cargo run --example switched_controller --features hybrid`

use oxictl::hybrid::{SwitchedError, SwitchedLti};

fn main() -> Result<(), String> {
    println!("=== Switched LTI Controller Demo ===");
    println!("Mode 0 (fast): A=0.7, B=1.0, C=1.0");
    println!("Mode 1 (slow): A=0.9, B=0.5, C=1.0");
    println!("Min dwell time: 5 steps");
    println!("P-controller gain: 0.5");
    println!();

    // ── Build the SwitchedLti system ──────────────────────────────────────────
    //
    // Template params: SwitchedLti<S, N, I, M>
    //   N = 1 (state dim), I = 1 (input dim), M = 2 (number of modes)
    //
    // a_modes : [M][N][N]  →  [[[f64; 1]; 1]; 2]
    // b_modes : [M][N][I]  →  [[[f64; 1]; 1]; 2]
    // c_modes : [M][N]     →  [[f64; 1]; 2]

    let a_modes: [[[f64; 1]; 1]; 2] = [[[0.7_f64]], [[0.9_f64]]];
    let b_modes: [[[f64; 1]; 1]; 2] = [[[1.0_f64]], [[0.5_f64]]];
    let c_modes: [[f64; 1]; 2] = [[1.0_f64], [1.0_f64]];
    let min_dwell: usize = 5;

    let mut plant = SwitchedLti::<f64, 1, 1, 2>::new(a_modes, b_modes, c_modes, min_dwell)
        .map_err(|e| format!("SwitchedLti::new failed: {e:?}"))?;

    // ── Simulation parameters ─────────────────────────────────────────────────

    let reference = 1.0_f64;
    let k_p = 0.5_f64;
    let n_steps = 40_usize;
    let switch_at_step = 20_usize;

    println!("step | mode | output y");
    println!("-----|------|----------");

    let mut y = 0.0_f64; // initial output (plant starts at x=0)

    for step in 0..n_steps {
        // Attempt mode switch at step 20 (dwell will have been satisfied by then)
        if step == switch_at_step {
            match plant.switch_to(1) {
                Ok(()) => {}
                Err(SwitchedError::DwellViolation) => {
                    // Minimum dwell not yet met — silently continue in current mode
                }
                Err(e) => return Err(format!("switch_to(1) failed: {e:?}")),
            }
        }

        // Proportional controller
        let error = reference - y;
        let u = k_p * error;

        // Advance plant
        y = plant
            .step(&[u])
            .map_err(|e| format!("plant.step failed: {e:?}"))?;

        // Print every 5 steps
        if step % 5 == 0 {
            println!("  {:3} |  {:1}   | {:.5}", step, plant.mode(), y);
        }
    }

    println!();
    println!("Total mode switches: {}", plant.total_switches());
    println!("Final mode: {}", plant.mode());
    println!("Final output y = {:.5}", y);
    println!();

    // ── Summary ───────────────────────────────────────────────────────────────

    println!("=== Summary ===");
    println!("Mode 0 (steps 0–19): fast settling toward reference = {reference:.1}");
    println!("Mode 1 (steps 20–39): slower dynamics, smaller input gain (B=0.5)");

    if plant.total_switches() >= 1 && (y - reference).abs() < 0.5 {
        println!("PASS: Mode switch occurred and output is tracking reference.");
    } else {
        println!(
            "INFO: switches={}, final |e| = {:.4}",
            plant.total_switches(),
            (y - reference).abs()
        );
    }

    Ok(())
}
