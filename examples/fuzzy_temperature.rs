//! Fuzzy logic temperature controller example.
//!
//! Demonstrates a Mamdani fuzzy inference engine controlling a simulated
//! thermostat. Linguistic variables for temperature error and error rate
//! drive a heating output through a set of IF-THEN rules.
//!
//! Run with:
//!   cargo run --example fuzzy_temperature --features "fuzzy"

use oxictl::fuzzy::{
    centroid_of_gravity, Antecedent, Consequent, FuzzyRule, MamdaniEngine, MembershipFn, TNorm,
    Trapezoidal, Triangular,
};

fn main() {
    println!("=== Fuzzy Logic Thermostat Controller ===\n");

    // ── Universe of discourse ──────────────────────────────────────────────────
    // Temperature error: [-10, 10] °C (NB, NS, ZE, PS, PB)
    // Error rate:        [-5, 5]  °C/s (NB, NS, ZE, PS, PB)
    // Heating output:    [0, 100] % power (VL, L, ME, H, VH)

    // ── Error membership functions (universe: [-10, 10]) ─────────────────────
    // NB = Negative Big,  NS = Negative Small,  ZE = Zero,
    // PS = Positive Small, PB = Positive Big
    let err_nb = Trapezoidal::new(-10.0_f64, -10.0, -6.0, -3.0).expect("NB error MF construction");
    let err_ns = Triangular::new(-6.0_f64, -3.0, 0.0).expect("NS error MF construction");
    let err_ze = Triangular::new(-3.0_f64, 0.0, 3.0).expect("ZE error MF construction");
    let err_ps = Triangular::new(0.0_f64, 3.0, 6.0).expect("PS error MF construction");
    let err_pb = Trapezoidal::new(3.0_f64, 6.0, 10.0, 10.0).expect("PB error MF construction");

    // ── Error rate membership functions (universe: [-5, 5]) ──────────────────
    let rate_nb = Trapezoidal::new(-5.0_f64, -5.0, -3.0, -1.0).expect("Rate NB MF construction");
    let rate_ns = Triangular::new(-3.0_f64, -1.5, 0.0).expect("Rate NS MF construction");
    let rate_ze = Triangular::new(-1.5_f64, 0.0, 1.5).expect("Rate ZE MF construction");
    let rate_ps = Triangular::new(0.0_f64, 1.5, 3.0).expect("Rate PS MF construction");
    let rate_pb = Trapezoidal::new(1.0_f64, 3.0, 5.0, 5.0).expect("Rate PB MF construction");

    // ── Output (heating) membership functions (universe: [0, 100]) ───────────
    // VL = Very Low, L = Low, ME = Medium, H = High, VH = Very High
    let heat_vl = Trapezoidal::new(0.0_f64, 0.0, 10.0, 25.0).expect("VL heating MF construction");
    let heat_lo = Triangular::new(10.0_f64, 25.0, 45.0).expect("L heating MF construction");
    let heat_me = Triangular::new(30.0_f64, 50.0, 70.0).expect("ME heating MF construction");
    let heat_hi = Triangular::new(55.0_f64, 75.0, 90.0).expect("H heating MF construction");
    let heat_vh =
        Trapezoidal::new(75.0_f64, 90.0, 100.0, 100.0).expect("VH heating MF construction");

    // ── MF slices for inference ───────────────────────────────────────────────
    // var 0 = error  (indices: 0=NB, 1=NS, 2=ZE, 3=PS, 4=PB)
    // var 1 = rate   (indices: 0=NB, 1=NS, 2=ZE, 3=PS, 4=PB)
    let err_mfs: &[&dyn MembershipFn<f64>] = &[&err_nb, &err_ns, &err_ze, &err_ps, &err_pb];
    let rate_mfs: &[&dyn MembershipFn<f64>] = &[&rate_nb, &rate_ns, &rate_ze, &rate_ps, &rate_pb];
    let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[err_mfs, rate_mfs];

    // output MF indices: 0=VL, 1=L, 2=ME, 3=H, 4=VH
    let output_mfs: &[&dyn MembershipFn<f64>] = &[&heat_vl, &heat_lo, &heat_me, &heat_hi, &heat_vh];

    // ── Mamdani rule base (9 representative rules) ────────────────────────────
    // Rule format: IF error=X AND rate=Y THEN heating=Z
    // Indices: error terms [NB=0,NS=1,ZE=2,PS=3,PB=4]
    //          rate  terms [NB=0,NS=1,ZE=2,PS=3,PB=4]
    //          heat  terms [VL=0,L=1,ME=2,H=3,VH=4]
    let rules: &[(usize, usize, usize)] = &[
        // (error_set_idx, rate_set_idx, heat_set_idx)
        (4, 3, 4), // IF error=PB AND rate=PS THEN heating=VH
        (4, 2, 4), // IF error=PB AND rate=ZE THEN heating=VH
        (4, 1, 3), // IF error=PB AND rate=NS THEN heating=H
        (3, 2, 3), // IF error=PS AND rate=ZE THEN heating=H
        (3, 1, 2), // IF error=PS AND rate=NS THEN heating=ME
        (2, 2, 2), // IF error=ZE AND rate=ZE THEN heating=ME
        (2, 3, 3), // IF error=ZE AND rate=PS THEN heating=H
        (1, 2, 1), // IF error=NS AND rate=ZE THEN heating=L
        (0, 2, 0), // IF error=NB AND rate=ZE THEN heating=VL
    ];

    // Build the Mamdani engine with capacity for 16 rules and 8 MFs per var
    let mut engine: MamdaniEngine<f64, 16, 8> = MamdaniEngine::new(TNorm::Min);
    for &(e_idx, r_idx, h_idx) in rules {
        let mut ant = Antecedent::new();
        ant.add(0, e_idx).expect("Add error condition");
        ant.add(1, r_idx).expect("Add rate condition");
        let con = Consequent::unit(0, h_idx);
        engine
            .add_rule(FuzzyRule::new(ant, con))
            .expect("Add Mamdani rule");
    }

    // ── Thermostat simulation ──────────────────────────────────────────────────
    // Simple first-order thermal plant: T[k+1] = T[k] + dt*(P*heat/100 - k_loss*(T-T_env))
    // Parameters: P=2.5 °C/s per 100% power, k_loss=0.05/s, T_env=20 °C
    let setpoint = 60.0_f64; // Target temperature
    let t_env = 20.0_f64; // Ambient temperature
    let p_heat = 2.5_f64; // Heating power coefficient (°C/s at 100%)
    let k_loss = 0.05_f64; // Heat loss rate (1/s)
    let dt = 0.5_f64; // Time step (s)
    let n_steps = 100_usize;

    let mut temp = t_env; // Start at ambient
    let mut prev_error = setpoint - temp;

    println!(
        "{:>6}  {:>8}  {:>10}  {:>10}  {:>10}",
        "Step", "Time(s)", "Temp(°C)", "Error(°C)", "Heat(%)"
    );
    println!("{}", "-".repeat(52));

    for k in 0..n_steps {
        let t_sim = k as f64 * dt;
        let error = setpoint - temp;
        let error_rate = (error - prev_error) / dt;

        // Clamp inputs to universe of discourse
        let error_clamped = error.clamp(-10.0, 10.0);
        let rate_clamped = error_rate.clamp(-5.0, 5.0);
        let crisp_inputs = [error_clamped, rate_clamped];

        // Fuzzy inference → crisp heating percentage
        let heat_pct = match engine.infer(&crisp_inputs, input_mfs, output_mfs, 0.0_f64, 100.0_f64)
        {
            Ok(samples) => {
                // Defuzzify with centroid of gravity
                centroid_of_gravity(&samples).unwrap_or(50.0)
            }
            Err(_) => 50.0, // fallback to 50% if inference fails
        };

        // Print every 10 steps
        if k % 10 == 0 {
            println!(
                "{:>6}  {:>8.1}  {:>10.3}  {:>10.3}  {:>10.2}",
                k, t_sim, temp, error, heat_pct
            );
        }

        // Update thermal plant
        let t_dot = p_heat * (heat_pct / 100.0) - k_loss * (temp - t_env);
        temp += dt * t_dot;
        prev_error = error;
    }

    // ── Final report ──────────────────────────────────────────────────────────
    let final_error = (temp - setpoint).abs();
    println!("\n=== Final State ===");
    println!("Temperature: {:.3} °C", temp);
    println!("Setpoint:    {:.3} °C", setpoint);
    println!("Final error: {:.3} °C", final_error);

    if final_error < 5.0 {
        println!("Status: CONVERGED (error < 5°C)");
    } else {
        println!("Status: NOT YET CONVERGED (more steps needed)");
    }

    println!("\nFuzzy rule base: {} active rules", rules.len());
    println!("Inference: Mamdani with Min T-norm, CoG defuzzification");
}
