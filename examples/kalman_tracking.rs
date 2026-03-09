use oxictl::estimator::KalmanFilter;
use oxictl::sim::Scope;

fn main() {
    let dt = 0.01_f64;

    // Position-velocity Kalman filter
    let mut kf = KalmanFilter::<f64, 2, 1, 1>::position_velocity(dt, 1.0, 2.0);

    let mut scope_true = Scope::with_capacity("true_pos", 1000);
    let mut scope_est = Scope::with_capacity("est_pos", 1000);
    let mut scope_meas = Scope::with_capacity("measured_pos", 1000);

    // True motion: constant velocity + sinusoidal acceleration
    let v0 = 2.0;
    let a0 = 0.5;

    println!("time,true_pos,measured_pos,est_pos,est_vel");

    let mut t = 0.0_f64;
    for _ in 0..1000 {
        let true_pos = v0 * t + 0.5 * a0 * t * t;
        let accel = a0;

        // Noisy measurement
        let noise = (t * 17.3).sin() * 0.8; // pseudo-random noise
        let meas = true_pos + noise;

        // KF step
        kf.predict(&[accel]);
        kf.update(&[meas]);

        let state = kf.state();
        let est_pos = state[0];
        let est_vel = state[1];

        scope_true.record(t, true_pos);
        scope_est.record(t, est_pos);
        scope_meas.record(t, meas);

        println!(
            "{:.3},{:.4},{:.4},{:.4},{:.4}",
            t, true_pos, meas, est_pos, est_vel
        );

        t += dt;
    }

    let final_true = v0 * t + 0.5 * a0 * t * t;
    let final_est = kf.state()[0];

    eprintln!("\n=== Kalman Filter Summary ===");
    eprintln!("Final true position: {:.3}", final_true);
    eprintln!("Final estimated:     {:.3}", final_est);
    eprintln!("Final error:         {:.4}", (final_est - final_true).abs());
    eprintln!("Max measurement noise amplitude: 0.8");
}
