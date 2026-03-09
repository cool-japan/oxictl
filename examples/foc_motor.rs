use oxictl::motor::foc::FocController;
use oxictl::sim::Scope;

fn main() {
    // FOC motor simulation parameters
    let vdc = 24.0_f64;
    let pole_pairs = 4.0_f64;
    let r_stator = 0.5; // ohm
    let l_stator = 0.001; // H
    let ke = 0.05; // V/(rad/s) back-EMF constant
    let j_motor = 0.0001; // kg*m^2
    let b_friction = 0.001; // N*m*s/rad

    // FOC controller
    let mut foc = FocController::<f64>::new(
        0.5,   // speed Kp
        10.0,  // speed Ki
        5.0,   // current Kp
        100.0, // current Ki
        10.0,  // Iq limit (A)
        12.0,  // voltage limit (V)
        vdc,
    );

    let dt = 0.0001; // 10kHz control loop
    let t_end = 2.0;
    let steps = (t_end / dt) as usize;

    let mut scope_speed = Scope::with_capacity("speed_rads", steps / 10);
    let mut scope_ia = Scope::with_capacity("phase_a_current", steps / 10);

    // Motor state
    let mut omega = 0.0_f64; // mechanical speed (rad/s)
    let mut theta_e = 0.0_f64; // electrical angle (rad)
    let mut ia = 0.0_f64;
    let mut ib = 0.0_f64;

    let speed_ref_fn = |t: f64| -> f64 {
        if t < 0.5 {
            50.0 * t / 0.5
        } else if t < 1.0 {
            50.0
        } else if t < 1.5 {
            100.0
        } else {
            80.0
        }
    };

    println!("time,speed_ref,speed,ia,vq");
    let mut t = 0.0_f64;

    for step in 0..steps {
        let speed_ref = speed_ref_fn(t) * pole_pairs; // convert to electrical rad/s

        // Run FOC
        let foc_out = foc.update(speed_ref, omega * pole_pairs, ia, ib, theta_e, dt);

        // Reconstruct phase voltages from duty cycles
        let va = (foc_out.duty.ta - 0.5) * vdc;
        let vb_phase = (foc_out.duty.tb - 0.5) * vdc;

        // Simple motor model (first-order current dynamics + Newton's law)
        let back_emf_a = ke * omega * (theta_e).cos();
        let back_emf_b = ke * omega * (theta_e - 2.0 * core::f64::consts::PI / 3.0).cos();

        let dia = (va - r_stator * ia - back_emf_a) / l_stator;
        let dib = (vb_phase - r_stator * ib - back_emf_b) / l_stator;
        ia += dia * dt;
        ib += dib * dt;

        // Torque and speed
        let ic = -ia - ib;
        let te = ke
            * pole_pairs
            * (ia * (theta_e).sin()
                + ib * (theta_e - 2.0 * core::f64::consts::PI / 3.0).sin()
                + ic * (theta_e - 4.0 * core::f64::consts::PI / 3.0).sin());
        let domega = (te - b_friction * omega) / j_motor;
        omega += domega * dt;
        theta_e += omega * pole_pairs * dt;
        theta_e = theta_e.rem_euclid(2.0 * core::f64::consts::PI);

        // Record and print at 100 Hz
        if step % 100 == 0 {
            scope_speed.record(t, omega);
            scope_ia.record(t, ia);
            println!(
                "{:.4},{:.2},{:.4},{:.4},{:.4}",
                t,
                speed_ref_fn(t),
                omega,
                ia,
                foc_out.vq
            );
        }

        t += dt;
    }

    eprintln!("\n=== FOC Motor Summary ===");
    eprintln!("Final speed: {:.2} rad/s", omega);
    eprintln!("Speed ref:   {:.2} rad/s", speed_ref_fn(t_end));
    eprintln!(
        "Speed range: {:.2}..{:.2}",
        scope_speed.min_value().unwrap_or(0.0),
        scope_speed.max_value().unwrap_or(0.0)
    );
}
