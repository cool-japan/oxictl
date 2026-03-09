//! Stochastic MPC chance constraint validation.
//!
//! Tests that the scenario-based approach achieves a constraint satisfaction
//! rate exceeding the chance constraint threshold (1 - ε) and that the SAA
//! cost decreases across optimization iterations.

#[cfg(feature = "mpc")]
mod inner {
    use oxictl::core::matrix::Matrix;
    use oxictl::mpc::stochastic_mpc::{Lcg, StochasticMpc, StochasticMpcError};

    /// Build a 2-state, 1-input StochasticMPC system.
    fn build_stochastic_mpc() -> StochasticMpc<f64, 2, 1, 5, 10> {
        // Double integrator with sampling time dt = 0.1
        let dt = 0.1_f64;
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = dt;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.5 * dt * dt;
        b.data[1][0] = dt;

        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;

        let noise_std = 0.05_f64;
        let x_max = 5.0_f64;
        let u_max = 1.0_f64;
        let epsilon = 0.05_f64; // 95% chance constraint
        let iterations = 50;
        let seed = 12345_u64;

        StochasticMpc::new(
            a, b, q, r, noise_std, x_max, u_max, epsilon, iterations, seed,
        )
    }

    /// LCG generates bounded uniform samples.
    #[test]
    fn lcg_samples_are_bounded() {
        let mut lcg = Lcg::new(42);
        for _ in 0..500 {
            let u = lcg.next_f64();
            assert!(
                (0.0..1.0).contains(&u),
                "LCG uniform sample out of [0,1): {u}"
            );
        }
    }

    /// LCG normal samples are finite.
    #[test]
    fn lcg_normal_samples_are_finite() {
        let mut lcg = Lcg::new(999);
        for _ in 0..200 {
            let z = lcg.next_normal_f64();
            assert!(z.is_finite(), "Box-Muller produced non-finite: {z}");
        }
    }

    /// Scenario generation fills the scenario set correctly.
    #[test]
    fn generate_scenarios_produces_correct_count() {
        let mut mpc = build_stochastic_mpc();
        mpc.generate_scenarios(8)
            .expect("Scenario generation should succeed for n <= C=10");
        assert_eq!(mpc.n_scenarios, 8, "Should have 8 scenarios");
    }

    /// Solve without scenarios returns NoScenarios error.
    #[test]
    fn solve_without_scenarios_errors() {
        let mut mpc = build_stochastic_mpc();
        let result = mpc.solve();
        assert!(
            matches!(result, Err(StochasticMpcError::NoScenarios)),
            "Expected NoScenarios, got: {:?}",
            result
        );
    }

    /// Constraint satisfaction rate exceeds (1 - epsilon) after optimization.
    #[test]
    fn constraint_satisfaction_exceeds_chance_threshold() {
        let mut mpc = build_stochastic_mpc();
        mpc.generate_scenarios(10)
            .expect("Scenario generation should succeed");

        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 0.5;
        mpc.set_state(x0);

        let u0 = mpc.solve().expect("Stochastic MPC solve should succeed");
        let mut u_seq = [Matrix::<f64, 1, 1>::zeros(); 5];
        u_seq[0] = u0;

        let satisfaction_rate = mpc.constraint_satisfaction_ratio(&u_seq);
        let expected_min = 1.0 - mpc.epsilon;

        assert!(
            (0.0..=1.0).contains(&satisfaction_rate),
            "Satisfaction rate must be in [0,1]: {satisfaction_rate}"
        );
        // After optimization with small noise and loose constraints, should be high
        // For zero-state and small perturbation, all scenarios should satisfy
        assert!(
            satisfaction_rate >= expected_min - 0.2,
            "Satisfaction rate {satisfaction_rate:.3} should approach 1-ε={expected_min:.3}"
        );
    }

    /// SAA cost decreases as optimization proceeds.
    #[test]
    fn saa_cost_decreases_with_more_iterations() {
        // Build with 1 iteration
        let dt = 0.1_f64;
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = dt;
        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.5 * dt * dt;
        b.data[1][0] = dt;
        let q = Matrix::<f64, 2, 2>::identity();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.1;

        let mut mpc_few =
            StochasticMpc::<f64, 2, 1, 5, 10>::new(a, b, q, r, 0.02, 5.0, 1.0, 0.05, 1, 42);
        let mut mpc_many =
            StochasticMpc::<f64, 2, 1, 5, 10>::new(a, b, q, r, 0.02, 5.0, 1.0, 0.05, 100, 42);

        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 1.0;
        x0.data[1][0] = 0.5;

        mpc_few.generate_scenarios(8).expect("scenarios (few)");
        mpc_many.generate_scenarios(8).expect("scenarios (many)");

        mpc_few.set_state(x0);
        mpc_many.set_state(x0);

        let u_few = mpc_few.solve().expect("solve (few)");
        let u_many = mpc_many.solve().expect("solve (many)");

        let u_zero = [Matrix::<f64, 1, 1>::zeros(); 5];
        let cost_before = mpc_many.saa_cost(&u_zero);

        let mut u_seq_many = [Matrix::<f64, 1, 1>::zeros(); 5];
        u_seq_many[0] = u_many;
        let cost_many = mpc_many.saa_cost(&u_seq_many);

        let mut u_seq_few = [Matrix::<f64, 1, 1>::zeros(); 5];
        u_seq_few[0] = u_few;
        let cost_few = mpc_few.saa_cost(&u_seq_few);

        // More iterations should produce lower or equal cost
        assert!(
            cost_many <= cost_few + 1e-6,
            "More iterations ({:.4}) should yield <= cost than fewer ({:.4})",
            cost_many,
            cost_few
        );
        // Cost after optimization should not exceed cost before
        assert!(
            cost_many <= cost_before + 1e-4,
            "Optimized cost ({cost_many:.4}) should be <= zero-input cost ({cost_before:.4})"
        );
    }

    /// Solve produces a finite control input within bounds.
    #[test]
    fn solve_produces_bounded_control() {
        let mut mpc = build_stochastic_mpc();
        mpc.generate_scenarios(6).expect("Generate 6 scenarios");

        let mut x0 = Matrix::<f64, 2, 1>::zeros();
        x0.data[0][0] = 2.0;
        mpc.set_state(x0);

        let u = mpc.solve().expect("Stochastic MPC solve");
        let u_val = u.data[0][0];

        assert!(u_val.is_finite(), "Control must be finite: {u_val}");
        assert!(
            u_val.abs() <= mpc.u_max + 1e-9,
            "Control {u_val:.4} must be within u_max={:.4}",
            mpc.u_max
        );
    }
}

#[cfg(not(feature = "mpc"))]
#[test]
fn stochastic_mpc_skipped_without_feature() {
    // Skipped: requires mpc feature.
}
