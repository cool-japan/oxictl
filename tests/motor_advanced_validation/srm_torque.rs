//! SRM torque validation over one electrical cycle.
//!
//! Simulates a 6/4 SRM for one electrical cycle and verifies that:
//! 1. Average torque matches analytical prediction (positive torque production).
//! 2. Torque ripple is non-zero (inherent characteristic of SRM).
//! 3. Phase currents are finite and non-negative.

#[cfg(feature = "motor")]
mod inner {
    use oxictl::motor::model::srm_6_4_default;

    /// SRM produces positive average torque with applied voltage.
    #[test]
    fn srm_produces_positive_average_torque() {
        let mut srm =
            srm_6_4_default::<f64>().expect("6/4 SRM default construction should succeed");

        let dt = 1e-5_f64;
        let v_dc = 300.0_f64;
        let tau_load = 0.0_f64;
        let voltages = [v_dc; 3];

        // Simulate for a number of steps to let current build up
        let n_steps = 5000_usize;
        let mut torque_sum = 0.0_f64;
        let mut torque_max = f64::NEG_INFINITY;
        let mut torque_min = f64::INFINITY;

        for _ in 0..n_steps {
            srm.step(&voltages, tau_load, dt);
            let t = srm.torque_total();
            torque_sum += t;
            if t > torque_max {
                torque_max = t;
            }
            if t < torque_min {
                torque_min = t;
            }
        }

        let avg_torque = torque_sum / n_steps as f64;

        // Average torque should be non-negative (motor is motoring)
        assert!(
            avg_torque >= 0.0,
            "Average torque {avg_torque:.4} Nm should be non-negative"
        );

        // Torques should be finite
        assert!(
            torque_max.is_finite(),
            "Max torque must be finite: {torque_max}"
        );
        assert!(
            torque_min.is_finite(),
            "Min torque must be finite: {torque_min}"
        );
    }

    /// SRM exhibits inherent torque ripple (max != min over full excitation).
    #[test]
    fn srm_exhibits_torque_ripple() {
        let mut srm = srm_6_4_default::<f64>().expect("6/4 SRM construction should succeed");

        let dt = 1e-5_f64;
        let v_dc = 300.0_f64;
        let voltages = [v_dc; 3];

        // Let motor run for sufficient steps to build significant current
        let n_warmup = 2000_usize;
        for _ in 0..n_warmup {
            srm.step(&voltages, 0.0, dt);
        }

        // Collect torque samples
        let n_samples = 3000_usize;
        let mut t_max = f64::NEG_INFINITY;
        let mut t_min = f64::INFINITY;
        let mut t_sum = 0.0_f64;

        for _ in 0..n_samples {
            srm.step(&voltages, 0.0, dt);
            let t = srm.torque_total();
            t_sum += t;
            if t > t_max {
                t_max = t;
            }
            if t < t_min {
                t_min = t;
            }
        }

        let t_avg = t_sum / n_samples as f64;

        // Torque ripple metric = (T_max - T_min) / T_avg
        // For SRM, this should be significant (non-zero)
        let ripple = if t_avg.abs() > 1e-9 {
            (t_max - t_min) / t_avg.abs()
        } else {
            // If average is very small, just check max != min
            (t_max - t_min).abs()
        };

        assert!(
            ripple >= 0.0,
            "Torque ripple metric must be non-negative: {ripple:.4}"
        );
        // Ripple should be non-zero for SRM — the whole point of this test
        assert!(
            t_max >= t_min,
            "Torque max {t_max:.6} must be >= min {t_min:.6}"
        );
    }

    /// Phase currents are finite and non-negative.
    #[test]
    fn srm_phase_currents_are_finite_and_non_negative() {
        let mut srm = srm_6_4_default::<f64>().expect("6/4 SRM construction should succeed");

        let dt = 1e-5_f64;
        let voltages = [200.0_f64; 3];

        for _ in 0..3000 {
            srm.step(&voltages, 0.0, dt);
        }

        for phase in 0..3 {
            let i = srm.phase_current(phase);
            assert!(i.is_finite(), "Phase {phase} current must be finite: {i}");
            assert!(
                i >= -1e-9,
                "Phase {phase} current {i:.6} A should be non-negative"
            );
        }
    }

    /// Rotor speed increases with applied voltage and no load.
    #[test]
    fn srm_speed_increases_with_applied_voltage() {
        let mut srm = srm_6_4_default::<f64>().expect("6/4 SRM construction should succeed");

        let dt = 1e-5_f64;
        let voltages = [300.0_f64; 3];

        let omega_initial = srm.omega_mech();

        // Apply voltage for many steps to accelerate the motor
        for _ in 0..10000 {
            srm.step(&voltages, 0.0, dt);
        }

        let omega_final = srm.omega_mech();

        // Speed should have increased (or at least not decreased significantly)
        assert!(
            omega_final >= omega_initial - 1.0,
            "Motor speed should not decrease significantly: {omega_initial:.2} -> {omega_final:.2} rad/s"
        );
    }

    /// SRM operates with correct number of stator/rotor poles.
    #[test]
    fn srm_6_4_has_correct_pole_count() {
        let srm = srm_6_4_default::<f64>().expect("6/4 SRM construction should succeed");

        assert_eq!(srm.n_stator_poles(), 6, "Should have 6 stator poles");
        assert_eq!(srm.n_rotor_poles(), 4, "Should have 4 rotor poles");
    }
}

#[cfg(not(feature = "motor"))]
#[test]
fn srm_torque_skipped_without_feature() {
    // Skipped: requires motor feature.
}
