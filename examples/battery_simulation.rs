//! Thevenin battery model simulation: discharge → rest → charge.
//!
//! Demonstrates the 18650 Li-ion cell model (Thevenin equivalent circuit) across
//! three operating phases:
//!
//!  1. **Discharge** (steps 0–99):   I = +2 A  (positive = discharge convention)
//!  2. **Rest**      (steps 100–149): I =  0 A
//!  3. **Charge**    (steps 150–199): I = −1 A  (negative = charge)
//!
//! The default 18650 cell has dt = 1 ms internally, so each `step()` call
//! advances the simulation by 1 ms.
//!
//! Run: `cargo run --example battery_simulation --features sim`

use oxictl::sim::TheveninBattery;

fn main() -> Result<(), String> {
    // Construct a default 18650 Li-ion cell (SOC=1.0, dt=1ms).
    let mut bat = TheveninBattery::<f64>::default_18650().map_err(|e| e.to_string())?;

    println!("# Thevenin Battery Simulation (18650 Li-ion cell)");
    println!("# Phase 1: discharge at I=+2 A (steps 0–99)");
    println!("# Phase 2: rest      at I= 0 A (steps 100–149)");
    println!("# Phase 3: charge    at I=−1 A (steps 150–199)");
    println!(
        "{:>6}  {:>10}  {:>10}  {:>10}",
        "step", "soc", "v_term", "phase"
    );

    let n_steps = 200_usize;
    for step in 0..n_steps {
        // Determine current and phase label for this step.
        let (current, phase) = if step < 100 {
            (2.0_f64, "discharge")
        } else if step < 150 {
            (0.0_f64, "rest")
        } else {
            (-1.0_f64, "charge")
        };

        // Advance battery model one step; handle terminal conditions gracefully.
        let v_term = match bat.step(current) {
            Ok(v) => v,
            Err(oxictl::sim::BatteryError::Overdischarged) => {
                eprintln!("step {step}: battery overdischarged — stopping simulation.");
                break;
            }
            Err(oxictl::sim::BatteryError::Overcharged) => {
                eprintln!("step {step}: battery overcharged — stopping simulation.");
                break;
            }
            Err(e) => return Err(e.to_string()),
        };

        let soc = bat.soc();

        // Print every 25 steps.
        if step % 25 == 0 {
            println!("{step:6}  {soc:10.5}  {v_term:10.5}  {phase:>10}");
        }
    }

    // Final summary
    eprintln!("\n=== Battery Simulation Summary ===");
    eprintln!("Final SOC:          {:.5}", bat.soc());
    eprintln!("Final OCV:          {:.5} V", bat.open_circuit_voltage());
    eprintln!("Final V_term (no-load): {:.5} V", bat.terminal_voltage());
    eprintln!("State of health:    {:.3}", bat.state_of_health());
    eprintln!("Remaining energy:   {:.3} J", bat.remaining_energy_joules());

    Ok(())
}
