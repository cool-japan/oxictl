//! Robust MPC stability and feasibility validation.
//!
//! Tests that the min-max robust MPC maintains bounded worst-case cost and
//! feasibility across uncertainty vertices over multiple horizon steps.

#[cfg(feature = "mpc")]
mod inner {
    use oxictl::core::matrix::Matrix;
    use oxictl::mpc::robust_mpc::{RobustBoxConstraint, RobustMpc, UncertaintyVertex};

    /// Build a 2-state, 1-input robust MPC with two uncertainty vertices.
    /// The vertices represent ±10% variation in the coupling coefficient.
    fn build_robust_mpc() -> RobustMpc<f64, 2, 1, 5, 2> {
        let dt = 0.1_f64;
        // Vertex 1: nominal coupling
        let a1 = Matrix::<f64, 2, 2> {
            data: [[1.0, dt * 0.9], [0.0, 1.0]],
        };
        let b1 = Matrix::<f64, 2, 1> {
            data: [[0.5 * dt * dt], [dt]],
        };
        // Vertex 2: +10% coupling variation
        let a2 = Matrix::<f64, 2, 2> {
            data: [[1.0, dt * 1.1], [0.0, 1.0]],
        };
        let b2 = Matrix::<f64, 2, 1> {
            data: [[0.5 * dt * dt], [dt]],
        };

        let v1 = UncertaintyVertex::new(a1, b1);
        let v2 = UncertaintyVertex::new(a2, b2);

        let q = Matrix::<f64, 2, 2> {
            data: [[1.0, 0.0], [0.0, 0.1]],
        };
        let r = Matrix::<f64, 1, 1> { data: [[0.05]] };
        let constraints = RobustBoxConstraint::new(10.0_f64, 2.0_f64);

        RobustMpc::new([v1, v2], q, r, constraints, 50)
    }

    /// Worst-case cost is non-negative for any state.
    #[test]
    fn worst_case_cost_is_non_negative() {
        let mut mpc = build_robust_mpc();

        let mut x = Matrix::<f64, 2, 1>::zeros();
        x.data[0][0] = 1.5;
        x.data[1][0] = 0.5;
        mpc.set_state(x);

        let u_seq = [Matrix::<f64, 1, 1>::zeros(); 5];
        let cost = mpc.worst_case_cost(&u_seq);
        assert!(cost >= 0.0, "Worst-case cost must be non-negative: {cost}");
    }

    /// solve() returns Ok and the control input is within tightened bounds.
    #[test]
    fn solve_returns_feasible_control() {
        let mut mpc = build_robust_mpc();

        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 1.0;
        mpc.set_state(x0);

        let result = mpc.solve();
        assert!(
            result.is_ok(),
            "Robust MPC solve should succeed: {:?}",
            result
        );

        let u = result.expect("Control input from robust MPC");
        let u_bound = mpc
            .constraints
            .tighten_input(mpc.tightening_margin)
            .expect("Tightened bound should be positive");

        assert!(
            u.data[0][0].abs() <= u_bound + 1e-9,
            "Control input {:.4} should respect tightened bound {:.4}",
            u.data[0][0],
            u_bound
        );
    }

    /// Worst-case cost at origin with zero input is zero.
    #[test]
    fn worst_case_cost_at_origin_is_zero() {
        let mpc = build_robust_mpc();
        let u_seq = [Matrix::<f64, 1, 1>::zeros(); 5];
        let cost = mpc.worst_case_cost(&u_seq);
        assert!(
            cost < 1e-12,
            "Cost at origin with zero input should be ~0: {cost}"
        );
    }

    /// Cost decreases from non-zero state after Robust MPC optimization.
    #[test]
    fn robust_mpc_reduces_worst_case_cost() {
        let mut mpc = build_robust_mpc();
        mpc.step_size = 0.005;
        mpc.iterations = 100;

        let mut x = Matrix::<f64, 2, 1>::zeros();
        x.data[0][0] = 2.0;
        x.data[1][0] = 1.0;
        mpc.set_state(x);

        // Cost with zero input
        let u_zero = [Matrix::<f64, 1, 1>::zeros(); 5];
        let cost_before = mpc.worst_case_cost(&u_zero);

        // After solve, build optimal input and compute cost
        let u_opt = mpc.solve().expect("Solve should succeed");
        let mut u_seq_opt = [Matrix::<f64, 1, 1>::zeros(); 5];
        u_seq_opt[0] = u_opt;
        let cost_after = mpc.worst_case_cost(&u_seq_opt);

        assert!(
            cost_after <= cost_before + 1e-3,
            "Robust MPC optimized cost ({cost_after:.4}) should not exceed zero-input cost ({cost_before:.4})"
        );
    }

    /// Constraint tightening produces a smaller bound than the original.
    #[test]
    fn constraint_tightening_reduces_bound() {
        let mpc = build_robust_mpc();
        let original_u_max = mpc.constraints.u_max;
        let tightened = mpc
            .constraints
            .tighten_input(mpc.tightening_margin)
            .expect("Tightening with small margin should succeed");

        assert!(
            tightened < original_u_max,
            "Tightened bound {tightened:.4} must be smaller than original {original_u_max:.4}"
        );
    }

    /// Feasibility maintained for N successive horizon steps.
    #[test]
    fn feasibility_maintained_across_horizon_steps() {
        let mut mpc = build_robust_mpc();
        let dt = 0.1_f64;
        let a_nominal = Matrix::<f64, 2, 2> {
            data: [[1.0, dt], [0.0, 1.0]],
        };
        let b_nominal = Matrix::<f64, 2, 1> {
            data: [[0.5 * dt * dt], [dt]],
        };

        let mut x = Matrix::<f64, 2, 1>::zeros();
        x.data[0][0] = 1.0;
        mpc.set_state(x);

        for step in 0..5 {
            let result = mpc.solve();
            assert!(
                result.is_ok(),
                "Robust MPC solve failed at step {step}: {:?}",
                result
            );
            let u = result.expect("Control input");

            // Propagate nominal system
            let ax = {
                let x0 = a_nominal.data[0][0] * x.data[0][0] + a_nominal.data[0][1] * x.data[1][0];
                let x1 = a_nominal.data[1][0] * x.data[0][0] + a_nominal.data[1][1] * x.data[1][0];
                (x0, x1)
            };
            let bu = (
                b_nominal.data[0][0] * u.data[0][0],
                b_nominal.data[1][0] * u.data[0][0],
            );
            x.data[0][0] = ax.0 + bu.0;
            x.data[1][0] = ax.1 + bu.1;
            mpc.set_state(x);
        }
    }
}

#[cfg(not(feature = "mpc"))]
#[test]
fn robust_mpc_skipped_without_feature() {
    // Skipped: requires mpc feature.
}
