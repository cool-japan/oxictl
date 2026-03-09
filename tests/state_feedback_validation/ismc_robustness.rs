//! ISMC robustness validation under ±30% parametric uncertainty.
//!
//! Integral Sliding Mode Control should track the reference despite
//! significant plant parameter variations. The boundary-layer saturation
//! law limits chattering while guaranteeing that the tracking error
//! converges to within the boundary layer thickness.

#[cfg(feature = "state_feedback")]
mod inner {
    use oxictl::state_feedback::integral_sliding_mode::{FirstOrderIsmc, SwitchingLaw};

    /// First-order plant with parametric uncertainty: ẋ = a_true·x + b_true·u + d
    /// Nominal model uses a_nom and b_nom; true plant differs by ±30%.
    fn plant_step(x: f64, u: f64, a_true: f64, b_true: f64, d: f64, dt: f64) -> f64 {
        x + dt * (a_true * x + b_true * u + d)
    }

    /// ISMC with saturation boundary layer tracks reference under +30% parameter error.
    #[test]
    fn ismc_tracks_with_positive_parameter_uncertainty() {
        let dt = 0.001_f64;
        // Nominal model parameters
        let a_nom = -2.0_f64;
        let b_nom = 1.0_f64;
        // True plant: +30% deviation in both a and b
        let a_true = a_nom * 1.30;
        let b_true = b_nom * 1.30;
        let d = 0.0_f64; // no disturbance; only parametric uncertainty

        let k_nom = 5.0_f64; // nominal gain: a_nom - b_nom*k_nom = -2 - 5 = -7
        let eta = 2.0_f64; // switching gain covers mismatch
        let phi = 0.15_f64; // boundary layer thickness
        let r_ref = 1.0_f64;

        let mut ctrl = FirstOrderIsmc::new(
            a_nom,
            b_nom,
            k_nom,
            eta,
            phi,
            SwitchingLaw::SaturationLayer,
            dt,
        )
        .expect("FirstOrderIsmc construction should succeed");

        let mut x = 0.0_f64;
        ctrl.initialise(x - r_ref);

        for _ in 0..8000 {
            let u = ctrl.update(x, r_ref);
            x = plant_step(x, u, a_true, b_true, d, dt);
        }

        // Error should converge to within boundary layer + tolerance for parametric error
        let error = (x - r_ref).abs();
        assert!(
            error < 0.30,
            "ISMC tracking error with +30% param uncertainty: {:.4} (expected < 0.30)",
            error
        );
    }

    /// ISMC with saturation boundary layer tracks reference under -30% parameter error.
    #[test]
    fn ismc_tracks_with_negative_parameter_uncertainty() {
        let dt = 0.001_f64;
        let a_nom = -2.0_f64;
        let b_nom = 1.0_f64;
        let a_true = a_nom * 0.70;
        let b_true = b_nom * 0.70;
        let d = 0.0_f64;

        let k_nom = 5.0_f64;
        let eta = 2.0_f64;
        let phi = 0.15_f64;
        let r_ref = 1.0_f64;

        let mut ctrl = FirstOrderIsmc::new(
            a_nom,
            b_nom,
            k_nom,
            eta,
            phi,
            SwitchingLaw::SaturationLayer,
            dt,
        )
        .expect("ISMC construction should succeed");

        let mut x = 0.0_f64;
        ctrl.initialise(x - r_ref);

        for _ in 0..8000 {
            let u = ctrl.update(x, r_ref);
            x = plant_step(x, u, a_true, b_true, d, dt);
        }

        let error = (x - r_ref).abs();
        assert!(
            error < 0.30,
            "ISMC tracking error with -30% param uncertainty: {:.4} (expected < 0.30)",
            error
        );
    }

    /// Sliding surface converges to small values after transient.
    #[test]
    fn ismc_sliding_surface_bounded_after_transient() {
        let dt = 0.001_f64;
        let a_nom = -1.5_f64;
        let b_nom = 1.0_f64;
        let a_true = a_nom * 1.20;
        let b_true = b_nom * 0.85;
        let k_nom = 4.0_f64;
        let eta = 1.5_f64;
        let phi = 0.10_f64;
        let r_ref = 2.0_f64;

        let mut ctrl = FirstOrderIsmc::new(
            a_nom,
            b_nom,
            k_nom,
            eta,
            phi,
            SwitchingLaw::SaturationLayer,
            dt,
        )
        .expect("ISMC construction should succeed");

        let mut x = 0.0_f64;
        ctrl.initialise(x - r_ref);

        // Transient phase
        for _ in 0..4000 {
            let u = ctrl.update(x, r_ref);
            x = plant_step(x, u, a_true, b_true, 0.0, dt);
        }

        // Evaluate sliding surface over the last 1000 steps
        let mut max_surface = 0.0_f64;
        for _ in 0..1000 {
            let u = ctrl.update(x, r_ref);
            let s = ctrl.surface(x, r_ref);
            let s_abs = s.abs();
            if s_abs > max_surface {
                max_surface = s_abs;
            }
            x = plant_step(x, u, a_true, b_true, 0.0, dt);
        }

        // Sliding surface should be bounded by the boundary layer thickness
        assert!(
            max_surface < phi * 3.0,
            "Sliding surface max={:.4} should be within ~3x boundary layer phi={:.2}",
            max_surface,
            phi
        );
    }

    /// ISMC with hard switching achieves tighter tracking but accepts chattering.
    #[test]
    fn ismc_hard_switching_achieves_tight_tracking() {
        let dt = 0.0005_f64;
        let a_nom = -1.0_f64;
        let b_nom = 1.0_f64;
        let a_true = a_nom * 1.25;
        let b_true = b_nom * 0.80;
        let k_nom = 3.0_f64;
        let eta = 1.0_f64;
        let phi = 0.05_f64; // phi ignored for HardSign
        let r_ref = 1.0_f64;

        let mut ctrl =
            FirstOrderIsmc::new(a_nom, b_nom, k_nom, eta, phi, SwitchingLaw::HardSign, dt)
                .expect("ISMC HardSign construction should succeed");

        let mut x = 0.0_f64;
        ctrl.initialise(x - r_ref);

        for _ in 0..8000 {
            let u = ctrl.update(x, r_ref);
            x = plant_step(x, u, a_true, b_true, 0.0, dt);
        }

        let error = (x - r_ref).abs();
        assert!(
            error < 0.15,
            "ISMC HardSign tracking error under ±25% uncertainty: {:.4} (expected < 0.15)",
            error
        );
    }
}

#[cfg(not(feature = "state_feedback"))]
#[test]
fn ismc_robustness_skipped_without_feature() {
    // Skipped: requires state_feedback feature.
}
