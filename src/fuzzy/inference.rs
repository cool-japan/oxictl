//! Mamdani and Sugeno (TSK) fuzzy inference engines.
//!
//! Both engines are generic over the floating-point scalar type `S` and use
//! const-generic capacities to remain `no_std + no_alloc`.

use crate::core::scalar::ControlScalar;
use crate::fuzzy::defuzzify::centroid_of_gravity;
use crate::fuzzy::membership::MembershipFn;
use crate::fuzzy::rule_base::{Antecedent, FuzzyRule, RuleBase, TNorm};
use crate::fuzzy::FuzzyError;
use core::marker::PhantomData;
use heapless::Vec;

// ────────────────────────────────────────────────────────────────────────────
// MamdaniEngine
// ────────────────────────────────────────────────────────────────────────────

/// Maximum number of output discretization points for Mamdani defuzzification.
pub const MAMDANI_SAMPLE_COUNT: usize = 128;

/// Mamdani fuzzy inference engine.
///
/// Type parameters:
/// - `S`: floating-point scalar (`f32` or `f64`).
/// - `R`: maximum number of rules.
/// - `N`: maximum number of input/output membership functions per variable.
///
/// The engine stores its rule base. Membership functions are supplied at
/// inference time to avoid self-referential structs.
pub struct MamdaniEngine<S: ControlScalar, const R: usize, const N: usize> {
    rule_base: RuleBase<R>,
    _phantom: PhantomData<S>,
}

impl<S: ControlScalar, const R: usize, const N: usize> MamdaniEngine<S, R, N> {
    /// Construct a new Mamdani engine with the given T-norm.
    pub fn new(t_norm: TNorm) -> Self {
        Self {
            rule_base: RuleBase::new(t_norm),
            _phantom: PhantomData,
        }
    }

    /// Add a fuzzy rule.
    pub fn add_rule(&mut self, rule: FuzzyRule) -> Result<(), FuzzyError> {
        self.rule_base.add_rule(rule)
    }

    /// Perform Mamdani inference and return the discretized output MF as
    /// `(x, mu)` sample pairs over `[out_min, out_max]`.
    ///
    /// Steps:
    /// 1. **Fuzzify** each crisp input using the provided input MFs.
    /// 2. **Fire** each rule → firing strength via T-norm.
    /// 3. **Imply**: clip each output MF at the firing strength (min implication).
    /// 4. **Aggregate**: union (max S-norm) across all active rule outputs.
    ///
    /// # Arguments
    /// - `crisp_inputs`: slice of crisp scalar inputs, one per input variable.
    /// - `input_mfs`: `input_mfs[var_idx]` is a slice of membership functions
    ///   for variable `var_idx`, ordered by term index.
    /// - `output_mfs`: membership functions for the single output variable,
    ///   one per term index (consequent `set_idx` indexes into this slice).
    /// - `out_min`, `out_max`: universe-of-discourse bounds for the output.
    ///
    /// Returns an error if `out_min >= out_max`.
    pub fn infer(
        &self,
        crisp_inputs: &[S],
        input_mfs: &[&[&dyn MembershipFn<S>]],
        output_mfs: &[&dyn MembershipFn<S>],
        out_min: S,
        out_max: S,
    ) -> Result<Vec<(S, S), MAMDANI_SAMPLE_COUNT>, FuzzyError> {
        if out_min >= out_max {
            return Err(FuzzyError::InvalidParameter(
                "MamdaniEngine: out_min must be less than out_max",
            ));
        }

        // ── Step 1: Fuzzify inputs ──────────────────────────────────────────
        // Build a 2-level membership table: memberships[var][set]
        // We store them in a fixed-size heapless Vec of heapless Vec<S, N>.
        let mut memberships: Vec<Vec<S, N>, N> = Vec::new();
        for (var_idx, &var_mfs) in input_mfs.iter().enumerate() {
            let mut row: Vec<S, N> = Vec::new();
            let x = crisp_inputs
                .get(var_idx)
                .copied()
                .ok_or(FuzzyError::InvalidParameter(
                    "MamdaniEngine: not enough crisp inputs",
                ))?;
            for mf in var_mfs.iter() {
                let mu = mf.membership(x).clamp_val(S::ZERO, S::ONE);
                row.push(mu).map_err(|_| FuzzyError::CapacityExceeded)?;
            }
            memberships
                .push(row)
                .map_err(|_| FuzzyError::CapacityExceeded)?;
        }

        // Build slice-of-slice view for fire_all
        let row_slices: Vec<&[S], N> = memberships
            .iter()
            .map(|r: &Vec<S, N>| r.as_slice())
            .collect();
        let mem_ref: &[&[S]] = row_slices.as_slice();

        // ── Step 2 & 3: Fire rules and collect (strength, set_idx) ─────────
        let fired = self.rule_base.fire_all::<S>(mem_ref);

        // ── Step 4: Aggregate over the output universe ─────────────────────
        let dx = (out_max - out_min) / S::from_f64((MAMDANI_SAMPLE_COUNT - 1) as f64);

        let mut samples: Vec<(S, S), MAMDANI_SAMPLE_COUNT> = Vec::new();

        for i in 0..MAMDANI_SAMPLE_COUNT {
            let x = out_min + dx * S::from_f64(i as f64);
            let mut agg_mu = S::ZERO;

            for &(strength, consequent) in fired.iter() {
                if let Some(&out_mf) = output_mfs.get(consequent.set_idx) {
                    let raw_mu = out_mf.membership(x).clamp_val(S::ZERO, S::ONE);
                    // Min implication: clip at firing strength
                    let implied = if raw_mu < strength { raw_mu } else { strength };
                    // Union (max S-norm)
                    if implied > agg_mu {
                        agg_mu = implied;
                    }
                }
            }

            samples
                .push((x, agg_mu))
                .map_err(|_| FuzzyError::CapacityExceeded)?;
        }

        Ok(samples)
    }

    /// Convenience method: perform full Mamdani inference **and** defuzzify
    /// using centroid-of-gravity.
    pub fn infer_crisp(
        &self,
        crisp_inputs: &[S],
        input_mfs: &[&[&dyn MembershipFn<S>]],
        output_mfs: &[&dyn MembershipFn<S>],
        out_min: S,
        out_max: S,
    ) -> Result<S, FuzzyError> {
        let samples = self.infer(crisp_inputs, input_mfs, output_mfs, out_min, out_max)?;
        centroid_of_gravity(&samples)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SugenoRule
// ────────────────────────────────────────────────────────────────────────────

/// A Sugeno (TSK) fuzzy rule with a linear output function.
///
/// For a two-input system: `z = p·x + q·y + r`.
/// `output_coeffs = [p, q, r]` (up to 3 coefficients).
///
/// For zeroth-order rules set `p = q = 0`, `r = constant`.
#[derive(Debug, Clone)]
pub struct SugenoRule<S: ControlScalar> {
    pub antecedent: Antecedent,
    /// Output coefficients `[c0, c1, bias]`. The bias is the last element.
    /// For an n-input system use n coefficients + 1 bias = n+1 total.
    pub output_coeffs: [S; 3],
}

impl<S: ControlScalar> SugenoRule<S> {
    /// Construct a first-order Sugeno rule: `z = p*x + q*y + r`.
    pub fn new(antecedent: Antecedent, p: S, q: S, r: S) -> Self {
        Self {
            antecedent,
            output_coeffs: [p, q, r],
        }
    }

    /// Compute the crisp output of this rule given crisp inputs.
    ///
    /// `z = Σ coeffs[i]*inputs[i] + bias (last coeff)`
    pub fn crisp_output(&self, inputs: &[S]) -> S {
        let n_coeffs = self.output_coeffs.len();
        // Last coefficient is the bias
        let bias = self.output_coeffs[n_coeffs - 1];
        let mut z = bias;
        for (i, &coeff) in self.output_coeffs[..n_coeffs - 1].iter().enumerate() {
            let x = inputs.get(i).copied().unwrap_or(S::ZERO);
            z += coeff * x;
        }
        z
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SugenoEngine
// ────────────────────────────────────────────────────────────────────────────

/// Sugeno (TSK) fuzzy inference engine.
///
/// `S` = floating-point scalar type.
/// `R` = maximum number of Sugeno rules.
pub struct SugenoEngine<S: ControlScalar, const R: usize> {
    rules: Vec<SugenoRule<S>, R>,
    t_norm: TNorm,
}

impl<S: ControlScalar, const R: usize> SugenoEngine<S, R> {
    /// Construct an empty Sugeno engine.
    pub fn new(t_norm: TNorm) -> Self {
        Self {
            rules: Vec::new(),
            t_norm,
        }
    }

    /// Add a Sugeno rule.
    pub fn add_rule(&mut self, rule: SugenoRule<S>) -> Result<(), FuzzyError> {
        self.rules
            .push(rule)
            .map_err(|_| FuzzyError::CapacityExceeded)
    }

    /// Perform Sugeno inference: weighted-average defuzzification.
    ///
    /// `z = Σ(wi · zi) / Σwi`
    ///
    /// # Arguments
    /// - `crisp_inputs`: crisp input values, one per input variable.
    /// - `input_mfs`: `input_mfs[var][set]` — membership function for
    ///   variable `var`, term `set`.
    ///
    /// Returns `FuzzyError::DivisionByZero` if all rule firing strengths are 0.
    pub fn infer(
        &self,
        crisp_inputs: &[S],
        input_mfs: &[&[&dyn MembershipFn<S>]],
    ) -> Result<S, FuzzyError> {
        // Fuzzify all inputs — 8-capacity inner Vecs suffice for typical problems
        let mut memberships: Vec<Vec<S, 8>, 8> = Vec::new();
        for (var_idx, &var_mfs) in input_mfs.iter().enumerate() {
            let mut row: Vec<S, 8> = Vec::new();
            let x = crisp_inputs
                .get(var_idx)
                .copied()
                .ok_or(FuzzyError::InvalidParameter(
                    "SugenoEngine: insufficient crisp inputs",
                ))?;
            for mf in var_mfs.iter() {
                let mu = mf.membership(x).clamp_val(S::ZERO, S::ONE);
                row.push(mu).map_err(|_| FuzzyError::CapacityExceeded)?;
            }
            memberships
                .push(row)
                .map_err(|_| FuzzyError::CapacityExceeded)?;
        }

        let row_slices: Vec<&[S], 8> = memberships
            .iter()
            .map(|r: &Vec<S, 8>| r.as_slice())
            .collect();

        // Weighted average
        let mut sum_w = S::ZERO;
        let mut sum_wz = S::ZERO;

        for rule in self.rules.iter() {
            // Compute firing strength via T-norm over antecedent conditions
            let strength = if rule.antecedent.conditions.is_empty() {
                S::ZERO
            } else {
                let mut s = S::ONE;
                for cond in rule.antecedent.conditions.iter() {
                    let mu = row_slices
                        .get(cond.var_idx)
                        .and_then(|sets| sets.get(cond.set_idx))
                        .copied()
                        .unwrap_or(S::ZERO);
                    s = self.t_norm.apply(s, mu);
                }
                s
            };

            let z = rule.crisp_output(crisp_inputs);
            sum_w += strength;
            sum_wz += strength * z;
        }

        if sum_w <= S::ZERO {
            return Err(FuzzyError::DivisionByZero);
        }
        Ok(sum_wz / sum_w)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Helper
// ────────────────────────────────────────────────────────────────────────────

/// Build a single-condition `Antecedent` for `(var_idx, set_idx)`.
pub fn single_condition_antecedent(
    var_idx: usize,
    set_idx: usize,
) -> Result<Antecedent, FuzzyError> {
    let mut ant = Antecedent::new();
    ant.add(var_idx, set_idx)?;
    Ok(ant)
}

pub use crate::fuzzy::rule_base::Condition;

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzy::membership::{Trapezoidal, Triangular};
    use crate::fuzzy::rule_base::Consequent;

    // ── Sugeno simple 2-input test ────────────────────────────────────────

    /// Two input variables, each with 2 terms.
    ///
    /// - x in [0, 10]: "low" (trapezoid left), "high" (trapezoid right)
    /// - y in [0, 10]: "slow", "fast"
    ///
    /// Two rules (zeroth-order):
    ///   IF x=low AND y=slow THEN z = 1
    ///   IF x=high AND y=fast THEN z = 9
    #[test]
    fn sugeno_weighted_average_two_inputs() {
        let x_low_t = Trapezoidal::new(0.0_f64, 0.0, 5.0, 10.0).unwrap();
        let x_high_t = Trapezoidal::new(0.0_f64, 5.0, 10.0, 10.0).unwrap();
        let y_slow_t = Trapezoidal::new(0.0_f64, 0.0, 4.0, 8.0).unwrap();
        let y_fast_t = Trapezoidal::new(2.0_f64, 6.0, 10.0, 10.0).unwrap();

        let x_mfs: &[&dyn MembershipFn<f64>] = &[&x_low_t, &x_high_t];
        let y_mfs: &[&dyn MembershipFn<f64>] = &[&y_slow_t, &y_fast_t];
        let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[x_mfs, y_mfs];

        let mut engine: SugenoEngine<f64, 4> = SugenoEngine::new(TNorm::Product);

        // Rule 1: x=low(0) AND y=slow(0) → z = 1
        let mut ant1 = Antecedent::new();
        ant1.add(0, 0).unwrap();
        ant1.add(1, 0).unwrap();
        engine
            .add_rule(SugenoRule::new(ant1, 0.0, 0.0, 1.0))
            .unwrap();

        // Rule 2: x=high(1) AND y=fast(1) → z = 9
        let mut ant2 = Antecedent::new();
        ant2.add(0, 1).unwrap();
        ant2.add(1, 1).unwrap();
        engine
            .add_rule(SugenoRule::new(ant2, 0.0, 0.0, 9.0))
            .unwrap();

        // At x=2, y=2: x_low fires more, y_slow fires more → output < 5
        let inputs = [2.0_f64, 2.0_f64];
        let result = engine.infer(&inputs, input_mfs).unwrap();
        assert!(
            result < 5.0,
            "Expected output < 5 for low x, slow y: got {result}"
        );

        // At x=8, y=8: x_high fires more, y_fast fires more → output > 5
        let inputs2 = [8.0_f64, 8.0_f64];
        let result2 = engine.infer(&inputs2, input_mfs).unwrap();
        assert!(
            result2 > 5.0,
            "Expected output > 5 for high x, fast y: got {result2}"
        );
    }

    #[test]
    fn sugeno_zeroth_order_single_rule() {
        let x_med = Trapezoidal::new(3.0_f64, 4.0, 6.0, 7.0).unwrap();
        let x_mfs: &[&dyn MembershipFn<f64>] = &[&x_med];
        let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[x_mfs];

        let mut engine: SugenoEngine<f64, 2> = SugenoEngine::new(TNorm::Min);
        let mut ant = Antecedent::new();
        ant.add(0, 0).unwrap();
        engine
            .add_rule(SugenoRule::new(ant, 0.0, 0.0, 5.0))
            .unwrap();

        // x = 5.0 is fully in "medium" → firing = 1.0 → z = 5.0
        let inputs = [5.0_f64];
        let result = engine.infer(&inputs, input_mfs).unwrap();
        assert!((result - 5.0).abs() < 1e-9, "Expected 5.0, got {result}");
    }

    #[test]
    fn sugeno_all_zero_firing_returns_error() {
        let x_high = Trapezoidal::new(8.0_f64, 9.0, 10.0, 10.0).unwrap();
        let x_mfs: &[&dyn MembershipFn<f64>] = &[&x_high];
        let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[x_mfs];

        let mut engine: SugenoEngine<f64, 2> = SugenoEngine::new(TNorm::Min);
        let mut ant = Antecedent::new();
        ant.add(0, 0).unwrap();
        engine
            .add_rule(SugenoRule::new(ant, 0.0, 0.0, 5.0))
            .unwrap();

        // x = 0.0 → membership in x_high = 0 → all weights = 0
        let inputs = [0.0_f64];
        let result = engine.infer(&inputs, input_mfs);
        assert!(
            matches!(result, Err(FuzzyError::DivisionByZero)),
            "Expected DivisionByZero, got {result:?}"
        );
    }

    #[test]
    fn mamdani_single_rule_output_shape() {
        let x_med = Trapezoidal::new(3.0_f64, 4.0, 6.0, 7.0).unwrap();
        let y_med = Trapezoidal::new(3.0_f64, 4.0, 6.0, 7.0).unwrap();

        let x_mfs: &[&dyn MembershipFn<f64>] = &[&x_med];
        let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[x_mfs];
        let output_mfs: &[&dyn MembershipFn<f64>] = &[&y_med];

        let mut engine: MamdaniEngine<f64, 4, 4> = MamdaniEngine::new(TNorm::Min);
        let mut ant = Antecedent::new();
        ant.add(0, 0).unwrap();
        let con = Consequent::unit(0, 0);
        engine.add_rule(FuzzyRule::new(ant, con)).unwrap();

        // Crisp input x = 5.0 (fully inside medium) → firing = 1.0
        let crisp_out = engine
            .infer_crisp(&[5.0_f64], input_mfs, output_mfs, 0.0, 10.0)
            .unwrap();

        // CoG of a symmetric trapezoid [3,4,6,7] should be 5.0
        assert!(
            (crisp_out - 5.0).abs() < 0.2,
            "Mamdani CoG output should be ~5.0, got {crisp_out}"
        );
    }

    #[test]
    fn mamdani_invalid_bounds_returns_error() {
        let y_med = Trapezoidal::new(3.0_f64, 4.0, 6.0, 7.0).unwrap();
        let x_med = Trapezoidal::new(3.0_f64, 4.0, 6.0, 7.0).unwrap();

        let x_mfs: &[&dyn MembershipFn<f64>] = &[&x_med];
        let input_mfs: &[&[&dyn MembershipFn<f64>]] = &[x_mfs];
        let output_mfs: &[&dyn MembershipFn<f64>] = &[&y_med];

        let engine: MamdaniEngine<f64, 4, 4> = MamdaniEngine::new(TNorm::Min);

        let result = engine.infer(&[5.0_f64], input_mfs, output_mfs, 10.0, 0.0);
        assert!(
            matches!(result, Err(FuzzyError::InvalidParameter(_))),
            "Expected InvalidParameter for reversed bounds"
        );
    }

    // Suppress unused import in tests
    fn _use_triangular() {
        let _ = Triangular::new(0.0_f64, 5.0, 10.0);
    }
}
