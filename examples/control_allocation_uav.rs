//! Over-actuated UAV control allocation demo.
//!
//! A quadrotor-style UAV has 4 motors (M=4) and 3 control objectives (N=3):
//!   v[0] = roll moment   (motors 1 and 2 contribute)
//!   v[1] = pitch moment  (motors 3 and 4 contribute)
//!   v[2] = total thrust  (all motors contribute equally)
//!
//! Effectiveness matrix B (3×4):
//!   B = [ 1  -1   0   0  ]   (roll:   M1 positive, M2 negative)
//!       [ 0   0   1  -1  ]   (pitch:  M3 positive, M4 negative)
//!       [ 1   1   1   1  ]   (thrust: all motors)
//!
//! The WeightedPseudoInverse allocator finds the minimum-norm actuator command
//! u ∈ [0, 1]^4 such that B u ≈ v_des.  Equal weights w = [1,1,1,1] are used,
//! so the solution is the ordinary (Moore-Penrose) pseudo-inverse clamped to bounds.
//!
//! Run with:
//!   cargo run --example control_allocation_uav --features "allocation"

use oxictl::allocation::WeightedPseudoInverse;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Over-actuated UAV Control Allocation ===\n");

    // ── Effectiveness matrix B (3 objectives × 4 motors) ─────────────────────
    //
    // Layout:
    //   Motor 1 (front-left):  +roll, thrust
    //   Motor 2 (front-right): −roll, thrust
    //   Motor 3 (rear-left):   +pitch, thrust
    //   Motor 4 (rear-right):  −pitch, thrust
    //
    // B is stored row-major: b[row][col]
    let b: [[f64; 4]; 3] = [
        [1.0, -1.0, 0.0, 0.0], // row 0: roll moment
        [0.0, 0.0, 1.0, -1.0], // row 1: pitch moment
        [1.0, 1.0, 1.0, 1.0],  // row 2: total thrust
    ];

    // ── Weights: equal priority for all four motors ───────────────────────────
    let w: [f64; 4] = [1.0; 4];

    // ── Actuator bounds: throttle in [0, 1] (normalised motor command) ────────
    let u_min: [f64; 4] = [0.0; 4];
    let u_max: [f64; 4] = [1.0; 4];

    // ── Construct allocator ───────────────────────────────────────────────────
    let alloc = WeightedPseudoInverse::<f64, 3, 4>::new(b, w, u_min, u_max)
        .map_err(|e| format!("Allocator construction error: {e}"))?;

    // ── Desired virtual control: v_des = [roll, pitch, thrust] ───────────────
    //
    // Interpretation:
    //   roll  = +0.5  → tilt right (M1 spins faster than M2)
    //   pitch = +0.3  → nose-down  (M3 spins faster than M4)
    //   thrust= +0.8  → 80 % of maximum collective thrust
    let v_des: [f64; 3] = [0.5, 0.3, 0.8];

    println!("Desired virtual control v_des:");
    println!("  roll   = {:.3}", v_des[0]);
    println!("  pitch  = {:.3}", v_des[1]);
    println!("  thrust = {:.3}", v_des[2]);
    println!();

    // ── Solve allocation ──────────────────────────────────────────────────────
    let u = alloc
        .allocate(&v_des)
        .map_err(|e| format!("Allocation error: {e}"))?;

    println!("Allocated motor commands u:");
    for (idx, &cmd) in u.iter().enumerate() {
        println!("  Motor {} = {:.6}", idx + 1, cmd);
    }
    println!();

    // ── Verify: compute B u and compare to v_des ──────────────────────────────
    let v_actual = alloc.virtual_control(&u);

    println!("Verification — achieved virtual control B·u:");
    let labels = ["roll  ", "pitch ", "thrust"];
    for (i, (&achieved, &desired)) in v_actual.iter().zip(v_des.iter()).enumerate() {
        let err = (achieved - desired).abs();
        println!(
            "  {} : achieved = {:.6}  desired = {:.3}  |err| = {:.2e}",
            labels[i], achieved, desired, err
        );
    }
    println!();

    // ── Tracking error (Euclidean norm ‖B u − v_des‖) ────────────────────────
    let track_err = alloc.tracking_error(&u, &v_des);
    println!("Tracking error ‖B·u − v_des‖ = {:.2e}", track_err);

    // ── Weighted cost (minimum-norm objective) ────────────────────────────────
    let cost = alloc.weighted_cost(&u);
    println!("Weighted cost  uᵀ W u         = {:.6}", cost);
    println!();

    // ── Bounds check ─────────────────────────────────────────────────────────
    let all_feasible = u
        .iter()
        .all(|&ui| (0.0 - 1e-12..=1.0 + 1e-12).contains(&ui));
    if all_feasible {
        println!("[PASS] All motor commands are within [0, 1].");
    } else {
        println!("[WARN] One or more commands out of bounds.");
    }

    if track_err < 1e-9 {
        println!("[PASS] Perfect allocation — B·u == v_des (within numerical tolerance).");
    } else if track_err < 0.05 {
        println!(
            "[PASS] Good allocation — tracking error is small ({:.2e}).",
            track_err
        );
    } else {
        println!(
            "[INFO] Tracking error {:.2e} — v_des may be partially outside the attainable set.",
            track_err
        );
    }

    // ── Physical interpretation ───────────────────────────────────────────────
    println!();
    println!("=== Physical Interpretation ===");
    println!(
        "  Motor 1 (front-left):  {:.3} — generates +roll and +thrust",
        u[0]
    );
    println!(
        "  Motor 2 (front-right): {:.3} — generates −roll and +thrust",
        u[1]
    );
    println!(
        "  Motor 3 (rear-left):   {:.3} — generates +pitch and +thrust",
        u[2]
    );
    println!(
        "  Motor 4 (rear-right):  {:.3} — generates −pitch and +thrust",
        u[3]
    );
    println!();
    println!("Roll differential  (u1 − u2) = {:.3}", u[0] - u[1]);
    println!("Pitch differential (u3 − u4) = {:.3}", u[2] - u[3]);
    println!(
        "Mean thrust        Σu_i / 4  = {:.3}",
        u.iter().sum::<f64>() / 4.0
    );

    Ok(())
}
