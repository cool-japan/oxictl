//! Integration tests for fuzzy inference engines.
//!
//! Validates:
//! - Mamdani: output is within universe of discourse
//! - Sugeno: weighted average lies between min/max consequents
//! - Rule base: zero firing for inputs outside membership support
//! - Membership functions: partition of unity for covering partition
//! - FuzzyPID: output is finite and bounded

use oxictl::fuzzy::{
    centroid_of_gravity, Antecedent, Consequent, FuzzyError, FuzzyPid, FuzzyPidConfig, FuzzyRule,
    MamdaniEngine, MembershipFn, SugenoEngine, SugenoRule, TNorm, Trapezoidal, Triangular,
};

// ── Test 1: Mamdani output within universe of discourse ───────────────────────

#[test]
fn mamdani_output_within_universe_of_discourse() {
    // Two-input Mamdani system.
    // Input 1 (x): [0, 10] with 3 terms: Low, Med, High
    // Input 2 (y): [0, 10] with 3 terms: Low, Med, High
    // Output (z):  [0, 10] with 3 terms: Low, Med, High
    let out_min = 0.0_f64;
    let out_max = 10.0_f64;

    // Input MFs for x (var 0)
    let x_lo = Trapezoidal::new(0.0_f64, 0.0, 2.0, 5.0).expect("x_lo MF");
    let x_me = Triangular::new(2.0_f64, 5.0, 8.0).expect("x_me MF");
    let x_hi = Trapezoidal::new(5.0_f64, 8.0, 10.0, 10.0).expect("x_hi MF");

    // Input MFs for y (var 1)
    let y_lo = Trapezoidal::new(0.0_f64, 0.0, 2.0, 5.0).expect("y_lo MF");
    let y_me = Triangular::new(2.0_f64, 5.0, 8.0).expect("y_me MF");
    let y_hi = Trapezoidal::new(5.0_f64, 8.0, 10.0, 10.0).expect("y_hi MF");

    // Output MFs for z
    let z_lo = Trapezoidal::new(0.0_f64, 0.0, 2.0, 5.0).expect("z_lo MF");
    let z_me = Triangular::new(2.0_f64, 5.0, 8.0).expect("z_me MF");
    let z_hi = Trapezoidal::new(5.0_f64, 8.0, 10.0, 10.0).expect("z_hi MF");

    let x_mfs: &[&dyn MembershipFn<f64>] = &[&x_lo, &x_me, &x_hi];
    let y_mfs: &[&dyn MembershipFn<f64>] = &[&y_lo, &y_me, &y_hi];
    let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[x_mfs, y_mfs];
    let output_mfs: &[&dyn MembershipFn<f64>] = &[&z_lo, &z_me, &z_hi];

    let mut engine: MamdaniEngine<f64, 9, 4> = MamdaniEngine::new(TNorm::Min);

    // 9 rules: each combination of x and y terms maps to a z term
    // IF x=Lo AND y=Lo THEN z=Lo
    // IF x=Lo AND y=Me THEN z=Lo
    // IF x=Lo AND y=Hi THEN z=Me
    // IF x=Me AND y=Lo THEN z=Lo
    // IF x=Me AND y=Me THEN z=Me
    // IF x=Me AND y=Hi THEN z=Hi
    // IF x=Hi AND y=Lo THEN z=Me
    // IF x=Hi AND y=Me THEN z=Hi
    // IF x=Hi AND y=Hi THEN z=Hi
    let rule_table: &[(usize, usize, usize)] = &[
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 1),
        (1, 0, 0),
        (1, 1, 1),
        (1, 2, 2),
        (2, 0, 1),
        (2, 1, 2),
        (2, 2, 2),
    ];
    for &(xi, yi, zi) in rule_table {
        let mut ant = Antecedent::new();
        ant.add(0, xi).expect("Add x condition");
        ant.add(1, yi).expect("Add y condition");
        engine
            .add_rule(FuzzyRule::new(ant, Consequent::unit(0, zi)))
            .expect("Add rule");
    }

    // Test multiple input combinations
    // Some boundary inputs may produce all-zero aggregate (DivisionByZero from CoG)
    // when the input lies exactly at MF breakpoints with zero membership overlap.
    // We use interior points where rules always fire.
    let test_inputs = [
        [1.0_f64, 1.0], // Low-low region: x_lo and y_lo both fire
        [5.0, 5.0],     // Medium-medium: x_me and y_me fire
        [8.5, 8.5],     // High-high: x_hi and y_hi fire
        [2.5, 7.0],     // Low-high transition
        [7.0, 2.5],     // High-low transition
        [3.5, 6.5],     // Low-med-high overlap region
    ];

    for inputs in &test_inputs {
        let crisp_out = engine
            .infer_crisp(inputs, input_mfs, output_mfs, out_min, out_max)
            .expect("Mamdani inference should succeed for interior inputs");

        assert!(
            crisp_out >= out_min && crisp_out <= out_max,
            "Mamdani output {crisp_out:.4} out of bounds [{out_min}, {out_max}] \
             for inputs [{:.2}, {:.2}]",
            inputs[0],
            inputs[1]
        );
        assert!(
            crisp_out.is_finite(),
            "Mamdani output should be finite for inputs [{:.2}, {:.2}]",
            inputs[0],
            inputs[1]
        );
    }

    // For inputs at exact breakpoints, aggregate may be all-zero; handle gracefully.
    let boundary_inputs = [[0.0_f64, 0.0], [10.0, 10.0]];
    for inputs in &boundary_inputs {
        let samples = engine
            .infer(inputs, input_mfs, output_mfs, out_min, out_max)
            .expect("Mamdani infer should succeed (may produce zero aggregate)");
        // Every sample mu must be in [0, 1]
        for (_, mu) in samples.iter() {
            assert!(
                *mu >= 0.0 && *mu <= 1.0,
                "Membership mu={mu:.4} must be in [0, 1]"
            );
        }
    }
}

// ── Test 2: Sugeno weighted average is between min and max consequents ─────────

#[test]
fn sugeno_output_between_min_and_max_consequents() {
    // Zeroth-order Sugeno: 3 rules with consequents 1, 5, 9
    // Output must lie in [1, 9] for any input combination.
    let x_lo = Trapezoidal::new(0.0_f64, 0.0, 3.0, 6.0).expect("x_lo");
    let x_me = Triangular::new(3.0_f64, 5.0, 7.0).expect("x_me");
    let x_hi = Trapezoidal::new(4.0_f64, 7.0, 10.0, 10.0).expect("x_hi");

    let x_mfs: &[&dyn MembershipFn<f64>] = &[&x_lo, &x_me, &x_hi];
    let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[x_mfs];

    let mut engine: SugenoEngine<f64, 4> = SugenoEngine::new(TNorm::Product);

    // Rule 1: x=Lo → z=1
    let mut ant1 = Antecedent::new();
    ant1.add(0, 0).expect("Ant1");
    engine
        .add_rule(SugenoRule::new(ant1, 0.0, 0.0, 1.0))
        .expect("Rule 1");

    // Rule 2: x=Me → z=5
    let mut ant2 = Antecedent::new();
    ant2.add(0, 1).expect("Ant2");
    engine
        .add_rule(SugenoRule::new(ant2, 0.0, 0.0, 5.0))
        .expect("Rule 2");

    // Rule 3: x=Hi → z=9
    let mut ant3 = Antecedent::new();
    ant3.add(0, 2).expect("Ant3");
    engine
        .add_rule(SugenoRule::new(ant3, 0.0, 0.0, 9.0))
        .expect("Rule 3");

    // Test at multiple inputs
    let test_xs: &[f64] = &[0.0, 1.5, 3.5, 5.0, 6.5, 8.0, 10.0];
    for &x in test_xs {
        let inputs = [x];
        let result = engine.infer(&inputs, input_mfs);
        match result {
            Ok(z) => {
                assert!(
                    (1.0 - 1e-9..=9.0 + 1e-9).contains(&z),
                    "Sugeno output {z:.4} not in [1, 9] for x={x}"
                );
                assert!(z.is_finite(), "Sugeno output should be finite for x={x}");
            }
            Err(FuzzyError::DivisionByZero) => {
                // All firing strengths are zero — input outside all supports.
                // This is acceptable (e.g. x outside all MF supports).
            }
            Err(e) => {
                panic!("Unexpected Sugeno error for x={x}: {e:?}");
            }
        }
    }
}

// ── Test 3: Zero firing for inputs outside support ────────────────────────────

#[test]
fn mamdani_zero_aggregate_for_out_of_support_input() {
    // A single-rule Mamdani system with a narrow triangular input MF.
    // When input is outside the MF support, the output should be all-zeros
    // (every point in the output has aggregated membership = 0).
    let x_narrow = Triangular::new(4.0_f64, 5.0, 6.0).expect("Narrow MF");
    let y_narrow = Triangular::new(4.0_f64, 5.0, 6.0).expect("Output narrow MF");

    let x_mfs: &[&dyn MembershipFn<f64>] = &[&x_narrow];
    let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[x_mfs];
    let output_mfs: &[&dyn MembershipFn<f64>] = &[&y_narrow];

    let mut engine: MamdaniEngine<f64, 4, 4> = MamdaniEngine::new(TNorm::Min);
    let mut ant = Antecedent::new();
    ant.add(0, 0).expect("Add condition");
    engine
        .add_rule(FuzzyRule::new(ant, Consequent::unit(0, 0)))
        .expect("Add rule");

    // Input x = 0.0 is far outside support [4, 6], so firing strength = 0.
    let samples = engine
        .infer(&[0.0_f64], input_mfs, output_mfs, 0.0, 10.0)
        .expect("Inference should succeed (returns zero-aggregate samples)");

    // Every sample should have mu = 0 (all firing strengths are 0)
    for (x, mu) in samples.iter() {
        assert!(
            *mu <= 1e-12,
            "Expected zero membership at x={x:.4} for out-of-support input, got {mu:.6}"
        );
    }

    // Defuzzification of an all-zero distribution should return DivisionByZero
    let result = centroid_of_gravity(&samples);
    assert!(
        matches!(result, Err(FuzzyError::DivisionByZero)),
        "CoG of all-zero distribution should return DivisionByZero"
    );
}

#[test]
fn sugeno_zero_firing_returns_division_by_zero() {
    // Sugeno engine with a high-end MF. Input at zero → all firing strengths zero.
    let x_hi = Trapezoidal::new(8.0_f64, 9.0, 10.0, 10.0).expect("x_hi MF");
    let x_mfs: &[&dyn MembershipFn<f64>] = &[&x_hi];
    let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[x_mfs];

    let mut engine: SugenoEngine<f64, 2> = SugenoEngine::new(TNorm::Min);
    let mut ant = Antecedent::new();
    ant.add(0, 0).expect("Antecedent");
    engine
        .add_rule(SugenoRule::new(ant, 0.0, 0.0, 5.0))
        .expect("Rule");

    // x=0 gives x_hi.membership(0) = 0 → sum_w = 0 → DivisionByZero
    let result = engine.infer(&[0.0_f64], input_mfs);
    assert!(
        matches!(result, Err(FuzzyError::DivisionByZero)),
        "Expected DivisionByZero for out-of-support input, got {result:?}"
    );
}

// ── Test 4: Partition of unity for covering triangular partition ───────────────

#[test]
fn triangular_partition_of_unity() {
    // A classic Ruspini triangular partition of the open interval (0, 10).
    // Triangular MFs with overlap: each adjacent pair sums to 1 in their overlap.
    //
    // The standard uniform triangular partition uses 5 MFs with centers at
    // 0, 2.5, 5, 7.5, 10 and width 5.0 (each overlaps by 50%):
    // MF0 = tri(-2.5, 0.0, 2.5),  MF1 = tri(0.0, 2.5, 5.0),
    // MF2 = tri(2.5, 5.0, 7.5),   MF3 = tri(5.0, 7.5, 10.0),
    // MF4 = tri(7.5, 10.0, 12.5)
    // This is a proper Ruspini partition: Σ μ_k(x) = 1 for x ∈ (0, 10).
    // Note: the triangular MF returns 0 at the exact left/right boundary
    // (since `x <= left` → 0), so we test at interior points only.
    let mf0 = Triangular::new(-2.5_f64, 0.0, 2.5).expect("MF0");
    let mf1 = Triangular::new(0.0_f64, 2.5, 5.0).expect("MF1");
    let mf2 = Triangular::new(2.5_f64, 5.0, 7.5).expect("MF2");
    let mf3 = Triangular::new(5.0_f64, 7.5, 10.0).expect("MF3");
    let mf4 = Triangular::new(7.5_f64, 10.0, 12.5).expect("MF4");

    let mfs: &[&dyn MembershipFn<f64>] = &[&mf0, &mf1, &mf2, &mf3, &mf4];

    // Sample at interior points in (0, 10) — exclusive of endpoints.
    // At interior points, exactly two adjacent MFs are active and sum to 1.
    for i in 1..40_usize {
        let x = i as f64 * 0.25; // 0.25, 0.5, ..., 9.75
        let sum: f64 = mfs.iter().map(|mf| mf.membership(x)).sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Partition of unity violated at x={x:.2}: Σμ_k = {sum:.12}"
        );
    }
}

#[test]
fn trapezoidal_membership_in_unit_interval() {
    // All membership function values must lie in [0, 1]
    let mf = Trapezoidal::new(2.0_f64, 4.0, 6.0, 8.0).expect("Trapezoidal MF");

    let test_xs = [
        0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, -5.0, 15.0,
    ];
    for &x in &test_xs {
        let mu = mf.membership(x);
        assert!(
            (0.0..=1.0).contains(&mu),
            "Membership {mu:.6} at x={x} must be in [0, 1]"
        );
    }
}

// ── Test 5: FuzzyPID output is finite and bounded ─────────────────────────────

#[test]
fn fuzzy_pid_output_finite_and_bounded() {
    // Build a simple Sugeno-based FuzzyPID.
    // Two-input engine (error, error-rate) with 3 rules each.
    // The output adjusts Kp by ±0.5 based on error magnitude.

    // Error MFs: NB, ZE, PB over [-10, 10]
    let err_nb = Trapezoidal::new(-10.0_f64, -10.0, -5.0, 0.0).expect("err_nb");
    let err_ze = Triangular::new(-5.0_f64, 0.0, 5.0).expect("err_ze");
    let err_pb = Trapezoidal::new(0.0_f64, 5.0, 10.0, 10.0).expect("err_pb");

    // Rate MFs: same shape
    let rate_nb = Trapezoidal::new(-5.0_f64, -5.0, -2.0, 0.0).expect("rate_nb");
    let rate_ze = Triangular::new(-2.0_f64, 0.0, 2.0).expect("rate_ze");
    let rate_pb = Trapezoidal::new(0.0_f64, 2.0, 5.0, 5.0).expect("rate_pb");

    let err_mfs: &[&dyn MembershipFn<f64>] = &[&err_nb, &err_ze, &err_pb];
    let rate_mfs: &[&dyn MembershipFn<f64>] = &[&rate_nb, &rate_ze, &rate_pb];
    let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[err_mfs, rate_mfs];

    // Build three Sugeno engines (kp, ki, kd adjustments)
    // Each outputs a value in [-0.5, 0.5] using zeroth-order rules.
    fn build_engine() -> SugenoEngine<f64, 4> {
        let mut eng: SugenoEngine<f64, 4> = SugenoEngine::new(TNorm::Min);

        // Rule: err=NB → output=-0.5 (reduce gain for big negative error)
        let mut ant0 = Antecedent::new();
        ant0.add(0, 0).expect("ant0 err");
        eng.add_rule(SugenoRule::new(ant0, 0.0, 0.0, -0.5))
            .expect("Rule 0");

        // Rule: err=ZE → output=0.0
        let mut ant1 = Antecedent::new();
        ant1.add(0, 1).expect("ant1 err");
        eng.add_rule(SugenoRule::new(ant1, 0.0, 0.0, 0.0))
            .expect("Rule 1");

        // Rule: err=PB → output=0.5
        let mut ant2 = Antecedent::new();
        ant2.add(0, 2).expect("ant2 err");
        eng.add_rule(SugenoRule::new(ant2, 0.0, 0.0, 0.5))
            .expect("Rule 2");

        eng
    }

    let kp_engine = build_engine();
    let ki_engine = build_engine();
    let kd_engine = build_engine();

    let config = FuzzyPidConfig::new(2.0_f64, 0.5_f64, 0.1_f64)
        .with_deltas(0.5, 0.2, 0.05)
        .with_out_max(10.0);

    let mut fpid: FuzzyPid<f64, 4> = FuzzyPid::new(config, kp_engine, ki_engine, kd_engine);

    // Simulate 20 steps with varying setpoint and measurement
    let setpoints = [
        5.0_f64, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 8.0, 8.0, 8.0, 8.0, 5.0, 5.0, 5.0, 0.0, 0.0,
        0.0, 10.0, 10.0,
    ];
    let measurements = [
        0.0_f64, 1.0, 2.5, 3.5, 4.5, 4.5, 4.0, 3.5, 3.5, 5.0, 6.5, 7.5, 7.5, 6.5, 5.5, 5.5, 3.0,
        1.0, 1.0, 4.0,
    ];
    let dt = 0.1_f64;

    for (sp, meas) in setpoints.iter().zip(measurements.iter()) {
        let output = fpid
            .update(*sp, *meas, dt, input_mfs)
            .expect("FuzzyPID update should succeed");

        assert!(
            output.is_finite(),
            "FuzzyPID output should be finite: got {output}"
        );
        // Output should be clamped to [-10, 10]
        assert!(
            (-10.0 - 1e-9..=10.0 + 1e-9).contains(&output),
            "FuzzyPID output {output:.4} should be within [-10, 10]"
        );
    }
}

// ── Test 6: Mamdani consistency — symmetric input gives central output ─────────

#[test]
fn mamdani_symmetric_rule_gives_central_output() {
    // A single "medium" rule: IF x=Me THEN z=Me
    // At x=5 (center of medium), the output should be approximately 5.
    let x_me = Triangular::new(2.0_f64, 5.0, 8.0).expect("x_me");
    let z_me = Triangular::new(2.0_f64, 5.0, 8.0).expect("z_me");

    let x_mfs: &[&dyn MembershipFn<f64>] = &[&x_me];
    let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[x_mfs];
    let output_mfs: &[&dyn MembershipFn<f64>] = &[&z_me];

    let mut engine: MamdaniEngine<f64, 4, 4> = MamdaniEngine::new(TNorm::Min);
    let mut ant = Antecedent::new();
    ant.add(0, 0).expect("Condition");
    engine
        .add_rule(FuzzyRule::new(ant, Consequent::unit(0, 0)))
        .expect("Add rule");

    // At x=5.0, the triangular MF has membership 1.0 → firing = 1.0
    // Output = the full z_me triangle → CoG ≈ 5.0
    let out = engine
        .infer_crisp(&[5.0_f64], input_mfs, output_mfs, 0.0, 10.0)
        .expect("Inference at center should succeed");

    assert!(
        (out - 5.0).abs() < 0.5,
        "Symmetric rule at center should give ~5.0 output, got {out:.4}"
    );
}

// ── Test 7: Sugeno first-order rule uses input value correctly ─────────────────

#[test]
fn sugeno_first_order_rule_uses_inputs() {
    // First-order rule: z = 2*x + 1 (coeffs: p=2, q=0, r=1)
    // At x=3.0 and full membership (μ=1), output should be 2*3 + 1 = 7.
    let x_full = Trapezoidal::new(0.0_f64, 0.0, 10.0, 10.0).expect("Full coverage MF");
    let x_mfs: &[&dyn MembershipFn<f64>] = &[&x_full];
    let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[x_mfs];

    let mut engine: SugenoEngine<f64, 2> = SugenoEngine::new(TNorm::Min);
    let mut ant = Antecedent::new();
    ant.add(0, 0).expect("Antecedent");
    // z = 2*x + 0*y + 1 (only one input, coeff[2] = bias = 1, coeff[0] = 2)
    engine
        .add_rule(SugenoRule::new(ant, 2.0, 0.0, 1.0))
        .expect("Rule");

    let inputs = [3.0_f64];
    let z = engine
        .infer(&inputs, input_mfs)
        .expect("Inference should succeed");

    // z = 2*3.0 + 1 = 7.0 (at full membership, weight=1 so weighted avg = z)
    assert!(
        (z - 7.0).abs() < 1e-9,
        "First-order Sugeno rule output should be 7.0, got {z:.6}"
    );
}
