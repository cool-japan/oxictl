//! Lee geometric attitude controller demo on a quadrotor.
//!
//! Demonstrates the SO(3) geometric PD controller (Lee 2010) tracking attitude
//! convergence over 50 time steps.  The quadrotor starts with an initial roll
//! perturbation and the controller drives it back to the hover reference.
//!
//! The example uses conservative gains (k_R=0.5, k_Ω=0.4) with large inertia
//! so that the Euler integrator (dt=0.01 s) remains stable for demonstration.
//!
//! Run with:
//!   cargo run --example geometric_attitude --all-features

use oxictl::geometric::{
    rotation_error, GeometricConfig, GeometricController, GeometricRef, QuadRotorGeomState, SO3,
};

/// Norm of a 3-vector.
fn norm3(v: [f64; 3]) -> f64 {
    libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

/// Euler integration of angular velocity using the rotational equation of motion.
///
///   J · Ω̇ = τ − Ω × (J·Ω)
///   → Ω̇_i = (τ_i − (J_{jj} - J_{kk}) · Ω_j · Ω_k) / J_{ii}
///
/// For a diagonal inertia tensor J = diag(Ixx, Iyy, Izz).
fn propagate_omega(omega: [f64; 3], torque: [f64; 3], inertia: [f64; 3], dt: f64) -> [f64; 3] {
    let [ixx, iyy, izz] = inertia;
    let [ox, oy, oz] = omega;
    let [tx, ty, tz] = torque;
    // Euler equations: Ω̇ = J^{-1}(τ - Ω × JΩ)
    let omega_dot = [
        (tx - (iyy - izz) * oy * oz) / ixx,
        (ty - (izz - ixx) * oz * ox) / iyy,
        (tz - (ixx - iyy) * ox * oy) / izz,
    ];
    [
        omega[0] + dt * omega_dot[0],
        omega[1] + dt * omega_dot[1],
        omega[2] + dt * omega_dot[2],
    ]
}

/// Geodesic attitude integration: R_new = R · exp(Ω̂ · dt).
///
/// Uses the public `SO3::from_axis_angle` + `SO3::multiply` methods;
/// does not access the private `mat` field.
fn propagate_attitude(r: SO3<f64>, omega: [f64; 3], dt: f64) -> SO3<f64> {
    let angle = norm3(omega) * dt;
    if angle < 1e-14 {
        return r;
    }
    let inv_norm = 1.0 / norm3(omega);
    let axis = [
        omega[0] * inv_norm,
        omega[1] * inv_norm,
        omega[2] * inv_norm,
    ];
    let delta_r =
        SO3::<f64>::from_axis_angle(axis, angle).expect("axis is unit-norm by construction");
    r.multiply(&delta_r)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Lee Geometric Attitude Controller: SO(3) Convergence Demo ===\n");

    // ── Controller configuration ──────────────────────────────────────────────
    // Conservative gains matched to a slow Euler integrator (dt=0.01 s).
    // A real quadrotor would run at 500–1000 Hz; we use 100 Hz here.
    // Inertia is set 10× larger than a typical 500 g quad to damp oscillations.
    let config = GeometricConfig::<f64> {
        k_r: 0.5,
        k_omega: 0.4,
        k_x: 0.0, // position control disabled (attitude only)
        k_v: 0.0,
        mass: 0.5,
        inertia: [0.04, 0.04, 0.08], // 10× typical: [40e-3, 40e-3, 80e-3] kg·m²
        gravity: 9.81,
    };
    let inertia = config.inertia;
    let ctrl = GeometricController::new(config);

    // ── Initial state: 20° roll tilt about body-x, zero angular velocity ─────
    let initial_roll = 20.0_f64 * core::f64::consts::PI / 180.0; // 20° in rad
    let r_init = SO3::<f64>::from_axis_angle([1.0, 0.0, 0.0], initial_roll)
        .expect("non-zero axis for initial roll perturbation");

    let mut state = QuadRotorGeomState::<f64> {
        position: [0.0; 3],
        velocity: [0.0; 3],
        attitude: r_init,
        omega: [0.0; 3],
    };

    // ── Reference: hover at origin, zero yaw ──────────────────────────────────
    let r_ref = SO3::<f64>::identity();
    let ref_state = GeometricRef::<f64>::hover_at_origin();

    // ── Simulation parameters ─────────────────────────────────────────────────
    let dt = 0.01_f64; // 100 Hz control loop
    let steps = 50_usize;

    // Record initial attitude error norm for convergence comparison.
    let initial_e_r = rotation_error(&r_ref, &state.attitude);
    let initial_e_r_norm = norm3(initial_e_r);

    println!(
        "{:>6}  {:>18}  {:>22}",
        "step", "attitude_err_norm", "omega_err_norm"
    );
    println!("{}", "-".repeat(52));

    for step in 0..=steps {
        // Attitude error:  e_R = 0.5·vee(R_d^T·R − R^T·R_d) ∈ so(3) ≅ ℝ³
        let e_r = rotation_error(&r_ref, &state.attitude);
        let e_r_norm = norm3(e_r);

        // Angular velocity error: e_Ω = Ω − 0 = Ω  (since Ω_d = 0 at hover)
        let e_omega_norm = norm3(state.omega);

        if step % 5 == 0 {
            println!("{:>6}  {:>18.6}  {:>22.6}", step, e_r_norm, e_omega_norm);
        }

        if step == steps {
            break;
        }

        // Compute thrust and torque from the Lee geometric controller
        let (_thrust, torque) = ctrl.update(&state, &ref_state);

        // Integrate angular velocity (Euler equations of motion)
        state.omega = propagate_omega(state.omega, torque, inertia, dt);

        // Integrate rotation matrix on SO(3)
        state.attitude = propagate_attitude(state.attitude, state.omega, dt);
    }

    // ── Final assessment ──────────────────────────────────────────────────────
    let e_r_final = rotation_error(&r_ref, &state.attitude);
    let e_r_norm_final = norm3(e_r_final);
    let e_omega_norm_final = norm3(state.omega);

    println!("\n=== Summary ===");
    println!(
        "Initial roll perturbation: {:.1}°",
        initial_roll.to_degrees()
    );
    println!(
        "Attitude error:  initial={:.6}  final={:.6} rad",
        initial_e_r_norm, e_r_norm_final
    );
    println!("Omega error:     final={:.6} rad/s", e_omega_norm_final);

    if e_r_norm_final < initial_e_r_norm {
        println!(
            "Result: [CONVERGING] error reduced {:.1}% in {} steps",
            100.0 * (1.0 - e_r_norm_final / initial_e_r_norm),
            steps
        );
    } else {
        println!("Result: [CHECK] consider smaller dt or higher damping gain");
    }

    Ok(())
}
