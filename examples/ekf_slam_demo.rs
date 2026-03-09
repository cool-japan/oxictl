//! 2D EKF-SLAM demonstration: vehicle moves in a circle while estimating
//! the positions of three landmarks from range–bearing measurements.
//!
//! # Setup
//! - STATE_DIM = 9: vehicle pose [x, y, theta] + 3 landmarks x [lx, ly]
//! - Vehicle kinematics: unicycle model (v = 0.5 m/s, omega = 0.1 rad/s, dt = 0.1 s)
//! - True landmark positions (unknown to the filter):
//!   lm0 = (5.0, 0.0),  lm1 = (0.0, 5.0),  lm2 = (-3.0, 3.0)
//! - Every 5 steps, all three landmarks are observed (range + bearing)
//!
//! # Observation model
//! The `EkfSlam2D::update` API expects the bearing as the **relative** bearing:
//!   relative_bearing = atan2(dy, dx) - theta
//!
//! Run: `cargo run --example ekf_slam_demo --features navigation`

use oxictl::navigation::{EkfSlam2D, NavigationError};

/// True landmark positions (treated as unknown by the SLAM filter).
const LANDMARKS: [[f64; 2]; 3] = [[5.0, 0.0], [0.0, 5.0], [-3.0, 3.0]];

/// Compute range and relative bearing from vehicle to landmark.
///
/// Returns (range, relative_bearing) where relative_bearing = atan2(dy, dx) - theta.
fn observe(vx: f64, vy: f64, vth: f64, lx: f64, ly: f64) -> (f64, f64) {
    let dx = lx - vx;
    let dy = ly - vy;
    let range = libm::sqrt(dx * dx + dy * dy);
    let world_bearing = libm::atan2(dy, dx);
    let rel_bearing = world_bearing - vth;
    (range, rel_bearing)
}

/// Convert NavigationError to String for error propagation.
fn nav_err(e: NavigationError) -> String {
    format!("{e:?}")
}

fn main() -> Result<(), String> {
    // ------------ EKF-SLAM parameters ------------
    // STATE_DIM = 3 + 2*3 = 9 (vehicle pose + 3 landmarks)
    let mut slam = EkfSlam2D::<f64, 9>::new(
        0.01, // q_v:       process noise variance for linear velocity
        0.01, // q_omega:   process noise variance for angular velocity
        0.1,  // r_range:   measurement noise variance for range
        0.05, // r_bearing: measurement noise variance for bearing
        0.1,  // dt:        100 ms integration step
    )
    .map_err(nav_err)?;

    // ------------ Vehicle motion (unicycle) ------------
    let v = 0.5_f64; // linear velocity  [m/s]
    let omega = 0.1_f64; // angular velocity [rad/s]

    println!("# EKF-SLAM Demo: 3-landmark, 30-step unicycle trajectory");
    println!("# True landmarks: lm0=(5,0)  lm1=(0,5)  lm2=(-3,3)");
    println!(
        "{:>5}  {:>22}  {:>22}",
        "step", "vehicle_pose (x,y,th)", "lm0_estimate (lx,ly)"
    );

    let n_steps = 30_usize;
    for step in 0..=n_steps {
        // Prediction step: propagate vehicle pose with unicycle model.
        slam.predict(v, omega).map_err(nav_err)?;

        // Measurement update every 5 steps.
        if step % 5 == 0 {
            let pose = slam.vehicle_pose();
            let (vx, vy, vth) = (pose[0], pose[1], pose[2]);

            for (id, lm) in LANDMARKS.iter().enumerate() {
                let (range, rel_bearing) = observe(vx, vy, vth, lm[0], lm[1]);
                // Skip degenerate range (vehicle on top of landmark).
                if range > 1e-6 {
                    slam.update(id, range, rel_bearing).map_err(nav_err)?;
                }
            }
        }

        // Print every 5 steps.
        if step % 5 == 0 {
            let pose = slam.vehicle_pose();
            let lm0 = slam.landmark(0).map_err(nav_err)?;
            println!(
                "{step:5}  ({:7.3},{:7.3},{:7.3})  ({:7.3},{:7.3})",
                pose[0], pose[1], pose[2], lm0[0], lm0[1]
            );
        }
    }

    // Final summary: show all landmark estimates vs. ground truth.
    eprintln!("\n=== EKF-SLAM Final Landmark Estimates ===");
    for (id, true_lm) in LANDMARKS.iter().enumerate() {
        let est = slam.landmark(id).map_err(nav_err)?;
        let err = libm::sqrt(
            (est[0] - true_lm[0]) * (est[0] - true_lm[0])
                + (est[1] - true_lm[1]) * (est[1] - true_lm[1]),
        );
        eprintln!(
            "lm{id}: true=({:.3},{:.3})  est=({:.3},{:.3})  err={:.4} m",
            true_lm[0], true_lm[1], est[0], est[1], err
        );
    }

    let final_pose = slam.vehicle_pose();
    eprintln!(
        "Final vehicle pose: ({:.3}, {:.3}, {:.3} rad)",
        final_pose[0], final_pose[1], final_pose[2]
    );

    Ok(())
}
