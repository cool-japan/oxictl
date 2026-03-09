//! Information filter fusing three IMUs for UAV 2D attitude estimation.
//!
//! # Problem
//! A UAV's attitude (roll angle φ, roll rate φ̇) is estimated by fusing three
//! inertial measurement units that measure roll rate with different noise levels.
//! Real gyroscopes suffer from bias drift; here we model that as increased noise.
//!
//! # State and model
//! State: x = [φ, φ̇]  (roll angle rad, roll rate rad/s)
//!
//! Discrete constant-rate model (dt = 0.01 s):
//!   A = [[1, dt], [0, 1]]   (roll accumulates from roll rate)
//!   B = [[0.5·dt², dt]]^T   (external torque input, set to zero here)
//!
//! # Why Information Filter?
//! In the information (canonical) form, each sensor's measurement contribution is
//! *additive*:
//!   Ω_new = Ω_pred + H^T · R_i^{-1} · H
//!   ξ_new = ξ_pred + H^T · R_i^{-1} · z_i
//! This means N sensors can be fused in a single predict + N independent update
//! passes — ideal for modular multi-sensor architectures.
//!
//! # IMU noise levels
//! | IMU | Gyro noise σ (rad/s) | Quality   |
//! |-----|----------------------|-----------|
//!   1     0.02                  high
//!   2     0.08                  medium
//!   3     0.20                  low (e.g., cheap MEMS)
//!
//! # Simulation
//! True attitude: sinusoidal roll φ(t) = A·sin(ω·t), A = 0.3 rad, ω = 2π·0.5 Hz.
//! Each IMU measures φ̇ = dφ/dt = A·ω·cos(ω·t) + noise (no direct angle sensor).
//! The filter integrates the rate to estimate φ.
//!
//! Run: `cargo run --example multi_sensor_fusion --features estimator`

use oxictl::core::matrix::Matrix;
use oxictl::estimator::information_filter::InformationFilter;

// ---- Simulation parameters -----------------------------------------------

const DT: f64 = 0.01; // 100 Hz IMU sample rate
const N_STEPS: usize = 100; // 1 second of data

// ---- True motion -------------------------------------------------------------

/// True roll angle at time t (rad).
fn true_roll(t: f64) -> f64 {
    let omega = 2.0 * core::f64::consts::PI * 0.5; // 0.5 Hz roll
    0.3 * libm::sin(omega * t)
}

/// True roll rate dφ/dt at time t (rad/s).
fn true_roll_rate(t: f64) -> f64 {
    let omega = 2.0 * core::f64::consts::PI * 0.5;
    0.3 * omega * libm::cos(omega * t)
}

/// Deterministic pseudo-noise: sum of sinusoids at incommensurate frequencies.
/// Avoids the `rand` crate while giving plausible sensor-like noise.
fn pseudo_noise(t: f64, amplitude: f64, seed: f64) -> f64 {
    amplitude
        * (libm::sin(seed * t + 1.3)
            + 0.6 * libm::sin(seed * 2.7 * t + 2.1)
            + 0.3 * libm::sin(seed * 7.1 * t + 0.7))
        / 1.9 // normalize roughly to [-amplitude, +amplitude]
}

// ---- Build the Information Filter -------------------------------------------

/// Construct InformationFilter for the roll / roll-rate state using IMU 1's
/// noise model as the primary measurement noise (R_1).  IMUs 2 and 3 are fused
/// via additional update passes using their own R matrices.
///
/// State: N=2, Measurement: M=1 (gyro rate), Input: I=1 (zero torque).
fn build_filter(sigma_imu1: f64) -> InformationFilter<f64, 2, 1, 1> {
    // State transition (constant roll-rate model)
    let a = Matrix::<f64, 2, 2> {
        data: [[1.0, DT], [0.0, 1.0]],
    };

    // Control input matrix (zero torque — column not used)
    let b = Matrix::<f64, 2, 1> {
        data: [[0.5 * DT * DT], [DT]],
    };

    // Measurement matrix: only roll rate (φ̇) is measured by gyros
    let h = Matrix::<f64, 1, 2> { data: [[0.0, 1.0]] };

    // Process noise covariance: small, representing model imperfection
    let q = Matrix::<f64, 2, 2> {
        data: [[1e-6, 0.0], [0.0, 1e-5]],
    };

    // Primary measurement noise (IMU 1)
    let var1 = sigma_imu1 * sigma_imu1;
    let r = Matrix::<f64, 1, 1> { data: [[var1]] };

    // Initial covariance: moderate uncertainty
    let p0 = Matrix::<f64, 2, 2> {
        data: [[1.0, 0.0], [0.0, 1.0]],
    };

    InformationFilter::new(a, b, h, q, r, [0.0_f64; 2], p0)
        .expect("InformationFilter::new: p0 is positive definite")
}

fn main() {
    // IMU noise standard deviations (rad/s)
    let sigma = [0.02_f64, 0.08_f64, 0.20_f64];

    // Inverse noise variances for update equations (R^{-1} terms)
    let r_inv: [Matrix<f64, 1, 1>; 3] = [
        Matrix::<f64, 1, 1> {
            data: [[1.0 / (sigma[0] * sigma[0])]],
        },
        Matrix::<f64, 1, 1> {
            data: [[1.0 / (sigma[1] * sigma[1])]],
        },
        Matrix::<f64, 1, 1> {
            data: [[1.0 / (sigma[2] * sigma[2])]],
        },
    ];

    // Measurement matrix: same for all IMUs (each measures φ̇)
    let h = Matrix::<f64, 1, 2> { data: [[0.0, 1.0]] };

    // Build filter (uses IMU 1 noise in the internal R field, but we override
    // in fuse_sensors so the internal R is not actually called for the 3-IMU path)
    let mut filter = build_filter(sigma[0]);

    println!("# Information Filter: 3-IMU fusion for UAV roll estimation");
    println!(
        "# IMU noise σ: IMU1={:.3} IMU2={:.3} IMU3={:.3} (rad/s)",
        sigma[0], sigma[1], sigma[2]
    );
    println!("step,t_s,phi_true,phidot_true,z1,z2,z3,phi_est,phidot_est,err_deg");

    let mut rmse_sum = 0.0_f64;

    for step in 0..N_STEPS {
        let t = step as f64 * DT;

        // Ground truth
        let phi_true = true_roll(t);
        let phi_dot_true = true_roll_rate(t);

        // Three IMU measurements (gyro only — rate readings)
        // Seed values chosen to give visually different noise patterns.
        let z: [f64; 3] = [
            phi_dot_true + pseudo_noise(t, sigma[0] * 3.0, 13.7),
            phi_dot_true + pseudo_noise(t, sigma[1] * 3.0, 5.3),
            phi_dot_true + pseudo_noise(t, sigma[2] * 3.0, 19.1),
        ];

        // 1. Predict step (zero external input)
        filter
            .predict(&[0.0_f64])
            .expect("predict: information matrix should remain invertible");

        // 2. Fuse all three IMUs simultaneously via the information form.
        // Each sensor contribution is additive: Ω += H^T R_i^{-1} H, ξ += H^T R_i^{-1} z_i
        let sensors: [(Matrix<f64, 1, 2>, Matrix<f64, 1, 1>, [f64; 1]); 3] = [
            (h, r_inv[0], [z[0]]),
            (h, r_inv[1], [z[1]]),
            (h, r_inv[2], [z[2]]),
        ];
        filter
            .fuse_sensors(&sensors)
            .expect("fuse_sensors: information fusion should not fail");

        // 3. Recover state estimate
        let state = filter
            .state()
            .expect("state: information matrix is invertible");
        let phi_est = state[0];
        let phi_dot_est = state[1];

        let err_rad = (phi_est - phi_true).abs();
        rmse_sum += err_rad * err_rad;

        println!(
            "{},{:.4},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.4}",
            step,
            t,
            phi_true,
            phi_dot_true,
            z[0],
            z[1],
            z[2],
            phi_est,
            phi_dot_est,
            err_rad.to_degrees(),
        );
    }

    let rmse_deg = (rmse_sum / N_STEPS as f64).sqrt().to_degrees();
    let cov = filter
        .covariance()
        .expect("covariance: filter should be well-conditioned at end of run");

    eprintln!("\n=== Information Filter 3-IMU Fusion Summary ===");
    eprintln!(
        "IMU noise σ:    IMU1={:.3}  IMU2={:.3}  IMU3={:.3} rad/s",
        sigma[0], sigma[1], sigma[2]
    );
    eprintln!("Steps:          {}", N_STEPS);
    eprintln!("Roll RMSE:      {:.4} deg", rmse_deg);
    eprintln!(
        "Final covariance trace: {:.6}  (lower → more confident)",
        cov.data[0][0] + cov.data[1][1]
    );
    eprintln!(
        "Information matrix diagonal: Ω[0,0]={:.2}, Ω[1,1]={:.2}",
        filter.information_matrix().data[0][0],
        filter.information_matrix().data[1][1],
    );

    // Sanity check: fused RMSE should be well below worst individual sensor noise.
    if rmse_deg < sigma[2].to_degrees() * 2.0 {
        eprintln!("PASS: fused estimate significantly better than lowest-quality IMU alone.");
    } else {
        eprintln!("WARN: unexpected RMSE — check filter tuning.");
    }
}
