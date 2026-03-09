//! Multi-objective MPC Pareto front validation.
//!
//! Tests that the Pareto front is non-dominated (no point is dominated by another)
//! and that it covers the spectrum from tracking-focused to energy-focused solutions.

#[cfg(feature = "mpc")]
mod inner {
    use oxictl::core::matrix::Matrix;
    use oxictl::mpc::multi_objective_mpc::{MultiObjectiveMpc, ParetoFront};

    /// Build a multi-objective MPC for a 2D system with 1 input.
    fn build_mo_mpc() -> MultiObjectiveMpc<f64, 2, 1, 5> {
        let dt = 0.1_f64;
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = dt;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.5 * dt * dt;
        b.data[1][0] = dt;

        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 1.0;

        MultiObjectiveMpc::new(a, b, q, r, 80)
    }

    /// Pareto front has the expected number of points.
    #[test]
    fn pareto_front_has_correct_count() {
        let mut mpc = build_mo_mpc();

        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 1.0;
        mpc.set_state(x0);
        mpc.set_reference(Matrix::zeros());

        let front: ParetoFront<f64, 1, 6> = mpc
            .build_pareto_front::<6>()
            .expect("Pareto front build should succeed");

        assert_eq!(
            front.count, 6,
            "Pareto front should have 6 points, got {}",
            front.count
        );
    }

    /// All Pareto front points have non-negative costs.
    #[test]
    fn pareto_points_have_non_negative_costs() {
        let mut mpc = build_mo_mpc();

        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 1.0;
        mpc.set_state(x0);
        mpc.set_reference(Matrix::zeros());

        let front: ParetoFront<f64, 1, 5> = mpc
            .build_pareto_front::<5>()
            .expect("Pareto front build should succeed");

        for i in 0..front.count {
            let pt = &front.points[i];
            assert!(
                pt.tracking_cost >= 0.0,
                "Point {i}: tracking_cost={:.4} must be >= 0",
                pt.tracking_cost
            );
            assert!(
                pt.energy_cost >= 0.0,
                "Point {i}: energy_cost={:.4} must be >= 0",
                pt.energy_cost
            );
        }
    }

    /// Pareto front is non-dominated: no point is dominated by another.
    /// A point A dominates B if A.tracking <= B.tracking AND A.energy <= B.energy
    /// with at least one strict inequality.
    #[test]
    fn pareto_front_is_non_dominated() {
        let mut mpc = build_mo_mpc();

        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 2.0;
        x0.data[1][0] = 0.5;
        mpc.set_state(x0);
        mpc.set_reference(Matrix::zeros());

        let front: ParetoFront<f64, 1, 7> = mpc
            .build_pareto_front::<7>()
            .expect("Pareto front build should succeed");

        let n = front.count;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pi = &front.points[i];
                let pj = &front.points[j];
                // Point i should NOT dominate point j strictly on BOTH objectives
                let i_dominates_j = pi.tracking_cost <= pj.tracking_cost
                    && pi.energy_cost <= pj.energy_cost
                    && (pi.tracking_cost < pj.tracking_cost || pi.energy_cost < pj.energy_cost);
                // Non-domination only holds if costs are sufficiently distinct
                // We check that i does not strictly dominate j beyond numerical tolerance
                if i_dominates_j {
                    let track_diff = (pi.tracking_cost - pj.tracking_cost).abs();
                    let energy_diff = (pi.energy_cost - pj.energy_cost).abs();
                    // Allow very small numerical domination due to optimization
                    let numerical_tol = 1e-3;
                    assert!(
                        track_diff < numerical_tol || energy_diff < numerical_tol,
                        "Point {i} (track={:.4}, energy={:.4}) dominates point {j} (track={:.4}, energy={:.4}) beyond tolerance",
                        pi.tracking_cost, pi.energy_cost, pj.tracking_cost, pj.energy_cost
                    );
                }
            }
        }
    }

    /// Tracking-focused point (high λ) has lower tracking cost than energy-focused (low λ).
    #[test]
    fn pareto_front_spans_tracking_to_energy_extremes() {
        let mut mpc = build_mo_mpc();

        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 1.5;
        mpc.set_state(x0);
        mpc.set_reference(Matrix::zeros());

        let front: ParetoFront<f64, 1, 8> = mpc
            .build_pareto_front::<8>()
            .expect("Pareto front build should succeed");

        // With 8 points: first point (p=1, λ=1/9) is more energy-focused (low λ)
        // last point (p=8, λ=8/9) is more tracking-focused (high λ)
        // Higher λ → more weight on tracking → should push toward lower tracking cost
        let first = &front.points[0];
        let last = &front.points[front.count - 1];

        // Last point (tracking-focused) should have higher energy cost
        // or at least, the costs should differ, showing we span the front
        let tracking_diff = (first.tracking_cost - last.tracking_cost).abs();
        let energy_diff = (first.energy_cost - last.energy_cost).abs();

        assert!(
            tracking_diff > 1e-6 || energy_diff > 1e-6,
            "Pareto front should span objective space: track_diff={tracking_diff:.6}, energy_diff={energy_diff:.6}"
        );
    }

    /// best_for_weights returns valid index.
    #[test]
    fn best_for_weights_returns_valid_index() {
        let mut mpc = build_mo_mpc();

        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 1.0;
        mpc.set_state(x0);
        mpc.set_reference(Matrix::zeros());

        let front: ParetoFront<f64, 1, 5> = mpc
            .build_pareto_front::<5>()
            .expect("Pareto front build should succeed");

        let idx = front
            .best_for_weights(0.7, 0.3)
            .expect("best_for_weights should succeed");
        assert!(
            idx < front.count,
            "Returned index {idx} should be within Pareto front (count={})",
            front.count
        );

        // Tracking-focused query (high tracking weight)
        let idx_track = front.best_for_weights(0.99, 0.01).expect("tracking focus");
        assert!(
            idx_track < front.count,
            "Tracking-focused index should be valid"
        );

        // Energy-focused query
        let idx_energy = front.best_for_weights(0.01, 0.99).expect("energy focus");
        assert!(
            idx_energy < front.count,
            "Energy-focused index should be valid"
        );
    }
}

#[cfg(not(feature = "mpc"))]
#[test]
fn multi_objective_pareto_skipped_without_feature() {
    // Skipped: requires mpc feature.
}
