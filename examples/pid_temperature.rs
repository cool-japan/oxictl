use oxictl::core::signal::{Feedback, Setpoint};
use oxictl::core::traits::{Controller, Plant};
use oxictl::pid::{AntiWindupMethod, PidConfig};
use oxictl::safety::{FaultResponse, FaultSeverity, SafetyMonitor, Watchdog};
use oxictl::sim::{Scope, ThermalPlant};

fn main() {
    // === Plant parameters ===
    let tau = 10.0; // thermal time constant (seconds)
    let gain = 100.0; // heater gain (degrees per unit power)
    let ambient = 25.0; // ambient temperature
    let mut plant = ThermalPlant::new(ambient, tau, gain, ambient);

    // === PID controller ===
    let mut pid = PidConfig::pid(0.5, 0.1, 2.0)
        .with_limits(0.0, 1.0) // heater power 0..1
        .with_anti_windup(AntiWindupMethod::Clamping)
        .build();

    // === Safety monitors ===
    let mut watchdog = Watchdog::<f64>::new(0.5);
    let mut safety = SafetyMonitor::<f64, 1, 1, 0>::new();
    safety.set_range(0, -10.0, 150.0, FaultSeverity::Critical);
    safety.set_rate(0, 50.0, FaultSeverity::Warning); // max 50 deg/s rate

    // === Simulation ===
    let dt = 0.01; // 100 Hz control loop
    let total_time = 60.0; // 60 seconds
    let steps = (total_time / dt) as usize;

    let mut scope_temp = Scope::with_capacity("temperature", steps);
    let mut scope_setpoint = Scope::with_capacity("setpoint", steps);
    let mut scope_output = Scope::with_capacity("heater_power", steps);

    // Setpoint profile: ramp up, hold, step change, disturbance
    let setpoint_fn = |t: f64| -> f64 {
        if t < 30.0 {
            60.0
        } else {
            80.0
        }
    };

    // Print CSV header
    println!("time,setpoint,temperature,heater_power,safety_ok");

    let mut time = 0.0;
    for _ in 0..steps {
        let sp_value = setpoint_fn(time);
        let temp = plant.output();

        // Safety check
        let safety_status = safety.evaluate(&[temp], &[temp], dt);
        watchdog.kick();
        let _ = watchdog.check(dt);

        let heater_power = if safety_status.response == FaultResponse::EmergencyStop {
            0.0 // emergency: shut off heater
        } else {
            let sp = Setpoint::new(sp_value);
            let fb = Feedback::new(temp);
            let out = pid.update(&sp, &fb, dt);
            out.value().clamp(0.0, 1.0)
        };

        // Record
        scope_temp.record(time, temp);
        scope_setpoint.record(time, sp_value);
        scope_output.record(time, heater_power);

        // Print every 100 steps (1 second)
        if (time * 100.0) as u64 % 100 == 0 {
            println!(
                "{:.2},{:.2},{:.4},{:.4},{}",
                time, sp_value, temp, heater_power, !safety_status.any_violation
            );
        }

        // Inject disturbance at t=45s
        if (time - 45.0).abs() < dt / 2.0 {
            plant.add_disturbance(-10.0);
            eprintln!(">>> Disturbance injected at t={:.2}s: -10 deg", time);
        }

        // Step plant
        plant.step(heater_power, dt);
        time += dt;
    }

    // Summary
    eprintln!("\n=== Simulation Summary ===");
    eprintln!(
        "Temperature range: {:.2} .. {:.2}",
        scope_temp.min_value().unwrap(),
        scope_temp.max_value().unwrap()
    );
    eprintln!("Final temperature: {:.2}", plant.output());
    eprintln!("Final setpoint: {:.2}", setpoint_fn(total_time));
    eprintln!(
        "Steady-state error: {:.4}",
        (plant.output() - setpoint_fn(total_time)).abs()
    );
    eprintln!("Total data points: {}", scope_temp.len());
}
