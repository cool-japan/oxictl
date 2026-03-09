//! MTPA curve optimality validation.
//!
//! For each torque command, the MTPA algorithm should produce id/iq references
//! that satisfy the MTPA optimality condition: the stator current magnitude
//! is minimised for the given torque output.

#[cfg(feature = "motor")]
mod inner {
    use oxictl::motor::foc::{MtpaMotorParams, MtpaTable};

    /// Build salient PMSM parameters for MTPA.
    fn salient_params() -> MtpaMotorParams<f64> {
        MtpaMotorParams {
            pole_pairs: 3,
            ld: 0.001,       // H — d-axis inductance
            lq: 0.003,       // H — q-axis inductance (salient: Lq > Ld)
            lambda_pm: 0.05, // Wb — PM flux linkage
            i_s_max: 20.0,   // A
        }
    }

    /// Build non-salient PMSM parameters (surface-mounted PMSM).
    fn non_salient_params() -> MtpaMotorParams<f64> {
        MtpaMotorParams {
            pole_pairs: 4,
            ld: 0.002,
            lq: 0.002, // Ld ≈ Lq → non-salient
            lambda_pm: 0.08,
            i_s_max: 15.0,
        }
    }

    /// MTPA table construction succeeds for valid salient motor.
    #[test]
    fn mtpa_table_builds_for_salient_pmsm() {
        let params = salient_params();
        let table = MtpaTable::<f64, 64>::new(params, 32)
            .expect("MTPA table should build for salient PMSM");

        // Table should cover non-trivial torque range — query mid-range torque
        let (id, iq) = table.query(0.5);
        assert!(id.is_finite(), "id_ref must be finite: {id}");
        assert!(iq.is_finite(), "iq_ref must be finite: {iq}");
    }

    /// For a salient motor, optimal id should be negative (demagnetising) at positive torque.
    #[test]
    fn salient_mtpa_id_is_non_positive_at_positive_torque() {
        let params = salient_params();
        let table = MtpaTable::<f64, 64>::new(params, 32).expect("MTPA table construction");

        for torque in [0.5_f64, 1.0, 2.0, 3.0] {
            let (id, iq) = table.query(torque);
            assert!(
                id <= 0.0 + 1e-9,
                "id_ref={id:.4} should be <= 0 for positive torque={torque}"
            );
            assert!(
                iq >= 0.0 - 1e-9,
                "iq_ref={iq:.4} should be >= 0 for positive torque={torque}"
            );
        }
    }

    /// For a non-salient motor (Ld ≈ Lq), id_ref should be near zero.
    #[test]
    fn non_salient_mtpa_id_near_zero() {
        let params = non_salient_params();
        let table = MtpaTable::<f64, 64>::new(params, 32)
            .expect("MTPA table construction for non-salient PMSM");

        for torque in [0.5_f64, 1.0, 2.0] {
            let (id, _iq) = table.query(torque);
            // For Ld = Lq, the MTPA condition gives id = 0
            assert!(
                id.abs() < 1.0,
                "Non-salient PMSM: id_ref={id:.4} should be small for torque={torque}"
            );
        }
    }

    /// MTPA current magnitude constraint: Is² = id² + iq² should not exceed Is_max².
    #[test]
    fn mtpa_current_magnitude_within_limit() {
        let params = salient_params();
        let i_s_max = params.i_s_max;
        let table = MtpaTable::<f64, 64>::new(params, 32).expect("MTPA table construction");

        // Query at several torque levels
        let torques = [0.0_f64, 0.5, 1.0, 2.0, 5.0, 10.0];
        for &torque in &torques {
            let (id, iq) = table.query(torque);
            let is_sq = id * id + iq * iq;
            let is_mag = is_sq.sqrt();
            assert!(
                is_mag <= i_s_max + 0.5,
                "Is={is_mag:.3} A exceeds Is_max={i_s_max:.1} A for torque={torque}"
            );
        }
    }

    /// Sign symmetry: negative torque gives negated iq (and same id).
    #[test]
    fn mtpa_negative_torque_sign_symmetry() {
        let params = salient_params();
        let table = MtpaTable::<f64, 64>::new(params, 32).expect("MTPA table construction");

        let (id_pos, iq_pos) = table.query(2.0);
        let (id_neg, iq_neg) = table.query(-2.0);

        // id should be same magnitude (MTPA: id <= 0, so id_pos = id_neg ≤ 0)
        assert!(
            (id_pos - id_neg).abs() < 0.1,
            "id_ref should be symmetric: pos={id_pos:.4}, neg={id_neg:.4}"
        );
        // iq should flip sign
        assert!(
            (iq_pos + iq_neg).abs() < 0.1,
            "iq_ref should flip sign: pos={iq_pos:.4}, neg={iq_neg:.4}"
        );
    }

    /// Larger torque commands require larger current magnitude (monotonicity).
    #[test]
    fn mtpa_current_monotone_with_torque() {
        let params = salient_params();
        let table = MtpaTable::<f64, 64>::new(params, 32).expect("MTPA table construction");

        let (id1, iq1) = table.query(1.0);
        let (id2, iq2) = table.query(3.0);

        let is1 = (id1 * id1 + iq1 * iq1).sqrt();
        let is2 = (id2 * id2 + iq2 * iq2).sqrt();

        assert!(
            is2 >= is1,
            "Higher torque should require >= current: Is(1Nm)={is1:.4}, Is(3Nm)={is2:.4}"
        );
    }
}

#[cfg(not(feature = "motor"))]
#[test]
fn mtpa_curve_skipped_without_feature() {
    // Skipped: requires motor feature.
}
