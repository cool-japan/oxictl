use oxictl::core::signal::{Feedback, Setpoint};
use oxictl::core::traits::{Controller, Plant};
use oxictl::pid::PidConfig;
use oxictl::sim::ThermalPlant;

#[test]
fn pid_thermal_step_response_converges() {
    let mut plant = ThermalPlant::new(25.0, 5.0, 50.0, 25.0);
    let mut pid = PidConfig::pi(0.5, 0.2).with_limits(0.0, 1.0).build();

    let setpoint = 50.0;
    let dt = 0.01;

    // Run for 30 seconds
    for _ in 0..3000 {
        let sp = Setpoint::new(setpoint);
        let fb = Feedback::new(plant.output());
        let out = pid.update(&sp, &fb, dt);
        plant.step(out.value().clamp(0.0, 1.0), dt);
    }

    let error = (plant.output() - setpoint).abs();
    assert!(
        error < 1.0,
        "Should converge within 1 degree: error={:.4}, temp={:.4}",
        error,
        plant.output()
    );
}

#[test]
fn pid_thermal_disturbance_rejection() {
    let mut plant = ThermalPlant::new(25.0, 5.0, 50.0, 25.0);
    let mut pid = PidConfig::pi(0.5, 0.2).with_limits(0.0, 1.0).build();

    let setpoint = 50.0;
    let dt = 0.01;

    // Reach steady state
    for _ in 0..5000 {
        let sp = Setpoint::new(setpoint);
        let fb = Feedback::new(plant.output());
        let out = pid.update(&sp, &fb, dt);
        plant.step(out.value().clamp(0.0, 1.0), dt);
    }

    // Inject disturbance
    plant.add_disturbance(-10.0);

    // Run for another 30 seconds
    for _ in 0..3000 {
        let sp = Setpoint::new(setpoint);
        let fb = Feedback::new(plant.output());
        let out = pid.update(&sp, &fb, dt);
        plant.step(out.value().clamp(0.0, 1.0), dt);
    }

    let error = (plant.output() - setpoint).abs();
    assert!(
        error < 1.0,
        "Should recover from disturbance: error={:.4}",
        error
    );
}

#[test]
fn pid_thermal_setpoint_change() {
    let mut plant = ThermalPlant::new(25.0, 5.0, 50.0, 25.0);
    let mut pid = PidConfig::pi(0.5, 0.2).with_limits(0.0, 1.0).build();

    let dt = 0.01;

    // First setpoint
    for _ in 0..5000 {
        let sp = Setpoint::new(40.0);
        let fb = Feedback::new(plant.output());
        let out = pid.update(&sp, &fb, dt);
        plant.step(out.value().clamp(0.0, 1.0), dt);
    }

    // Change setpoint
    for _ in 0..5000 {
        let sp = Setpoint::new(60.0);
        let fb = Feedback::new(plant.output());
        let out = pid.update(&sp, &fb, dt);
        plant.step(out.value().clamp(0.0, 1.0), dt);
    }

    let error = (plant.output() - 60.0).abs();
    assert!(
        error < 1.0,
        "Should track setpoint change: error={:.4}",
        error
    );
}
