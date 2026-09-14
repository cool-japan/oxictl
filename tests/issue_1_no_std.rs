//! Regression tests for cool-japan/oxictl#1 — `no_std` support of the
//! `pid`, `estimator`, `motor` and `trajectory` modules.
//!
//! These tests exercise exactly the code paths that previously relied on
//! `std`-only inherent float methods (`f64::ceil`, `atan2`, `sqrt`, `ln`,
//! `floor`, `round`) or on `std::vec::Vec`.  Run them with the library built
//! as `no_std` so any `std` usage creeping back into those modules breaks the
//! build:
//!
//! ```text
//! cargo test --no-default-features --features "pid estimator motor trajectory" --test issue_1_no_std
//! cargo test --no-default-features --features "pid estimator motor trajectory alloc" --test issue_1_no_std
//! ```
//!
//! Note: inside a test build, dev-dependencies link `std` into the crate
//! graph, which makes `std`'s inherent float methods resolvable even from the
//! `no_std` library.  [`test_issue_1_library_builds_without_std`] therefore
//! re-checks the library on its own (no dev-dependencies).  The real
//! bare-metal check (no `std` available at all) is:
//!
//! ```text
//! cargo check --no-default-features --features "pid estimator motor trajectory alloc" \
//!     --target thumbv7em-none-eabihf
//! ```

use oxictl::core::matrix::Matrix;
use oxictl::estimator::MlEstimator;
use oxictl::motor::foc::dtc::FluxEstimator;
use oxictl::motor::foc::overmodulation::overmodulate;
use oxictl::motor::transform::SynRmController;
use oxictl::pid::fractional::FopidAutoTune;
use oxictl::trajectory::{BSpline, DubinsPath};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// `FopidAutoTune::tune` sized its grid with `f64::ceil` (the originally
/// reported failure).
#[test]
fn test_issue_1_pid_fopid_auto_tune() -> TestResult {
    let tuner = FopidAutoTune::new(1.0_f64, 0.5, 0.1, 45.0, 1.0);
    let result = tuner.tune().map_err(|e| format!("tune failed: {e:?}"))?;
    assert!((0.5..=1.5).contains(&result.lambda));
    assert!((0.5..=1.5).contains(&result.mu));
    assert!(result.objective.is_finite());
    Ok(())
}

/// `MlEstimator::fit` used `f64::ln` for the `ln(2π)` constant.
#[test]
fn test_issue_1_estimator_batch_ml_fit() -> TestResult {
    let mut y = [[0.0_f64; 1]; 16];
    for (k, row) in y.iter_mut().enumerate() {
        // Deterministic, slightly noisy ramp around a constant state.
        row[0] = if k % 2 == 0 { 0.3 } else { -0.3 };
    }
    let u = [[0.0_f64; 1]; 16];
    let est = MlEstimator::<f64, 1, 1, 16>::new(
        Matrix::identity(),
        Matrix::zeros(),
        Matrix::identity(),
        [0.0],
        Matrix { data: [[10.0]] },
        1e-4,
        1e-3,
    );
    let (q_est, r_est) = est
        .fit(&y, &u, 16, 20)
        .map_err(|e| format!("fit failed: {e:?}"))?;
    assert!(q_est.data[0][0] > 0.0 && q_est.data[0][0].is_finite());
    assert!(r_est.data[0][0] > 0.0 && r_est.data[0][0].is_finite());
    Ok(())
}

/// `ParticleFilter` needs a heap: it must be available on `no_std` + `alloc`.
#[cfg(feature = "alloc")]
#[test]
fn test_issue_1_estimator_particle_filter_alloc() -> TestResult {
    use oxictl::estimator::{gaussian_log_likelihood, ParticleFilter};

    fn transition(x: &[f64; 1], u: &[f64; 1]) -> [f64; 1] {
        [x[0] + 0.1 * u[0]]
    }
    fn likelihood(x: &[f64; 1], y: &[f64; 1]) -> f64 {
        gaussian_log_likelihood(&[y[0] - x[0]], &[1.0])
    }

    let mut pf = ParticleFilter::new(64, [0.0_f64], 0.5, [0.05], [0.1], transition, likelihood);
    let mut est = [0.0_f64];
    for _ in 0..20 {
        est = pf.step(&[0.0], &[1.0]);
    }
    assert_eq!(pf.n_particles(), 64);
    assert!(est[0].is_finite() && est[0] > 0.0, "estimate={}", est[0]);
    Ok(())
}

/// `FluxEstimator::flux_sector` used `f64::atan2` / `f64::floor`.
#[test]
fn test_issue_1_motor_dtc_flux_sector() -> TestResult {
    let mut est = FluxEstimator::<f64>::new(0.0);
    let cases = [
        ((1.0, 0.0), 1_usize),
        ((0.0, 1.0), 2),
        ((-1.0, 0.0), 4),
        ((0.0, -1.0), 5),
    ];
    for ((alpha, beta), expected) in cases {
        est.psi_alpha = alpha;
        est.psi_beta = beta;
        assert_eq!(est.flux_sector(), expected, "psi=({alpha}, {beta})");
    }
    Ok(())
}

/// Overmodulation mode 2 and the SynRM MTPA path used `f64::sqrt`.
#[test]
fn test_issue_1_motor_overmodulation_and_synrm() -> TestResult {
    let (a, b) = overmodulate(300.0_f64, 100.0, 400.0, 0.99);
    assert!(a.is_finite() && b.is_finite());
    assert!((a * a + b * b).sqrt() <= 400.0 * 2.0 / 3.0 + 1e-9);

    let ctrl = SynRmController::<f32>::new(0.03, 0.01, 2, 20.0);
    let (id, iq) = ctrl.mtpa_current_refs(1.0);
    let tau = ctrl.reluctance_torque(id, iq);
    assert!((tau - 1.0).abs() < 0.01, "tau={tau}");
    Ok(())
}

/// `BSpline::clamped_uniform` built its knots through a `Vec`.
#[test]
fn test_issue_1_trajectory_bspline_clamped_uniform() -> TestResult {
    let cp = [
        [0.0_f64, 0.0],
        [1.0, 2.0],
        [2.0, 0.0],
        [3.0, 2.0],
        [4.0, 0.0],
    ];
    let bs = BSpline::<f64, 2, 5>::clamped_uniform(cp, 3).ok_or("clamped_uniform failed")?;
    let (t0, t1) = bs.param_range();
    let p0 = bs.evaluate(t0);
    let p1 = bs.evaluate(t1);
    assert!(p0[0].abs() < 1e-9 && p0[1].abs() < 1e-9, "start={p0:?}");
    assert!(
        (p1[0] - 4.0).abs() < 1e-9 && p1[1].abs() < 1e-9,
        "end={p1:?}"
    );
    Ok(())
}

/// Dubins angle wrapping used `f64::floor` / `f64::round`.
#[test]
fn test_issue_1_trajectory_dubins_shortest_path() -> TestResult {
    let q0 = [1.0_f64, 2.0, core::f64::consts::FRAC_PI_4];
    let q1 = [5.0_f64, 5.0, core::f64::consts::FRAC_PI_2];
    let path = DubinsPath::shortest_path(q0, q1, 0.8).map_err(|e| format!("{e:?}"))?;
    let end = path.sample_at(path.length());
    assert!((end[0] - q1[0]).abs() < 1e-6, "end={end:?}");
    assert!((end[1] - q1[1]).abs() < 1e-6, "end={end:?}");
    Ok(())
}

/// Checks the library alone (without dev-dependencies, so `std` is not in the
/// crate graph) with only the `no_std` features enabled.  Uses the
/// `thumbv7em-none-eabihf` target when it is installed, the host otherwise.
#[test]
fn test_issue_1_library_builds_without_std() -> TestResult {
    use std::path::PathBuf;
    use std::process::Command;

    const EMBEDDED_TARGET: &str = "thumbv7em-none-eabihf";

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_root = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("target"));
    // Separate target dir: avoids contending for the lock of the outer build.
    let target_dir = target_root.join("issue_1_no_std_check");

    let embedded_available = Command::new("rustc")
        .args(["--print", "target-libdir", "--target", EMBEDDED_TARGET])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()))
        .is_some_and(|dir| dir.is_dir());

    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(&manifest_dir)
        .args(["check", "--lib", "--no-default-features", "--features"])
        .arg("pid estimator motor trajectory alloc")
        .arg("--target-dir")
        .arg(&target_dir);
    if embedded_available {
        cmd.args(["--target", EMBEDDED_TARGET]);
    }
    let output = cmd.output()?;
    assert!(
        output.status.success(),
        "no_std check failed (embedded target: {embedded_available}):\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}
