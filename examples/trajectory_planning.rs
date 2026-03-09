//! Example: trajectory planning with trapezoidal, quintic polynomial, and jerk-limited profiles.
//!
//! Demonstrates planning a 1D motion profile and printing the position/velocity/acceleration
//! at regular intervals.

use oxictl::trajectory::jerk_limited::JerkLimitedProfile;
use oxictl::trajectory::quintic::QuinticPolynomial;
use oxictl::trajectory::trapezoidal::TrapezoidalProfile;

fn main() {
    println!("=== Trajectory Planning Demo ===\n");

    // ──────────────────────────────────────────────────────────────────────
    // 1. Trapezoidal profile: move 2.0 m at v_max=1.0 m/s, a_max=2.0 m/s²
    // ──────────────────────────────────────────────────────────────────────
    println!("--- Trapezoidal Profile (distance=2.0, v_max=1.0, a_max=2.0) ---");
    let mut trap = TrapezoidalProfile::<f64>::new(1.0, 2.0);
    trap.plan(2.0);
    let t_total = trap.total_time();
    println!("Total time: {t_total:.4} s");

    let n = 10usize;
    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}",
        "t (s)", "pos (m)", "vel (m/s)", "acc (m/s²)"
    );
    for i in 0..=n {
        let t = t_total * i as f64 / n as f64;
        let (pos, vel, acc) = trap.query(t);
        println!("{t:8.4}  {pos:10.4}  {vel:10.4}  {acc:10.4}");
    }
    println!();

    // ──────────────────────────────────────────────────────────────────────
    // 2. Quintic polynomial: rest-to-rest from 0 to 1.5 m in 2.0 s
    // ──────────────────────────────────────────────────────────────────────
    println!("--- Quintic Polynomial (0 → 1.5 m in 2.0 s) ---");
    let duration = 2.0f64;
    let poly = QuinticPolynomial::<f64>::rest_to_rest(0.0, 1.5, duration)
        .expect("rest_to_rest should succeed");

    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}",
        "t (s)", "pos (m)", "vel (m/s)", "acc (m/s²)"
    );
    for i in 0..=n {
        let t = duration * i as f64 / n as f64;
        let pos = poly.position(t);
        let vel = poly.velocity(t);
        let acc = poly.acceleration(t);
        println!("{t:8.4}  {pos:10.4}  {vel:10.4}  {acc:10.4}");
    }
    // Verify boundary conditions
    let p_end = poly.position(duration);
    let v_end = poly.velocity(duration);
    println!("End: pos={p_end:.6} (target 1.5), vel={v_end:.2e} (target 0)");
    println!();

    // ──────────────────────────────────────────────────────────────────────
    // 3. Jerk-limited profile: 0 → 3.0 m, j_max=10, a_max=2, v_max=1.5
    // ──────────────────────────────────────────────────────────────────────
    println!("--- Jerk-Limited Profile (0 → 3.0 m, j_max=10, a_max=2, v_max=1.5) ---");
    let mut jlp = JerkLimitedProfile::<f64>::new(10.0, 2.0, 1.5);
    jlp.plan(3.0);

    let dt = 0.02f64;
    let n_steps = 300usize;
    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}",
        "t (s)", "pos (m)", "vel (m/s)", "acc (m/s²)"
    );
    let print_at: &[usize] = &[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 299];
    for step in 0..n_steps {
        let (pos, vel, acc) = jlp.update(dt);
        if print_at.contains(&step) {
            let t = step as f64 * dt;
            println!("{t:8.4}  {pos:10.4}  {vel:10.4}  {acc:10.4}");
        }
        if jlp.is_done() {
            let t = step as f64 * dt;
            println!("Profile done at t={t:.4} s, final pos={pos:.4}");
            break;
        }
    }
    println!();

    println!("=== Done ===");
}
