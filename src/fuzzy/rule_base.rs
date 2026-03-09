//! Fuzzy rule base: linguistic variables, fuzzy sets, rules, and T-norm firing.
//!
//! All fixed-size collections use `heapless` to remain `no_std` compatible.

use crate::core::scalar::ControlScalar;
use crate::fuzzy::FuzzyError;
use heapless::Vec;

// ────────────────────────────────────────────────────────────────────────────
// T-norm
// ────────────────────────────────────────────────────────────────────────────

/// T-norm operator used to combine antecedent memberships.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TNorm {
    /// Algebraic product: `a * b`.
    Product,
    /// Minimum (Zadeh): `min(a, b)`.
    Min,
}

impl TNorm {
    /// Apply the T-norm to two membership values.
    #[inline]
    pub fn apply<S: ControlScalar>(&self, a: S, b: S) -> S {
        match self {
            TNorm::Product => a * b,
            TNorm::Min => {
                if a < b {
                    a
                } else {
                    b
                }
            }
        }
    }

    /// Fold a slice of membership values with this T-norm. Returns `ONE` for empty slices.
    pub fn fold<S: ControlScalar>(&self, values: &[S]) -> S {
        values
            .iter()
            .copied()
            .fold(S::ONE, |acc, v| self.apply(acc, v))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FuzzySet
// ────────────────────────────────────────────────────────────────────────────

/// A computed fuzzy set: the degree of membership of an input in a named set.
#[derive(Debug, Clone, Copy)]
pub struct FuzzySet<S: ControlScalar> {
    /// FNV-1a hash of the linguistic set name (lightweight identity without `alloc`).
    pub name_hash: u32,
    /// Computed membership value in `[0, 1]`.
    pub membership: S,
}

impl<S: ControlScalar> FuzzySet<S> {
    /// Construct from a raw hash and membership.
    pub fn new(name_hash: u32, membership: S) -> Self {
        Self {
            name_hash,
            membership,
        }
    }
}

/// Compute FNV-1a 32-bit hash of a byte string at compile time or runtime.
pub fn fnv1a_hash(name: &str) -> u32 {
    let mut hash: u32 = 2_166_136_261;
    for byte in name.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

// ────────────────────────────────────────────────────────────────────────────
// LinguisticVar
// ────────────────────────────────────────────────────────────────────────────

/// A named linguistic variable with up to `N` membership functions.
///
/// Each slot holds a boxed trait object — we use a `heapless::Vec` of function
/// pointers (as `fn(S) -> S`) to stay `no_std` without `Box`. To retain full
/// generality while remaining `no_std + no_alloc` we store membership functions
/// as an array of static references.
///
/// For practical use the `N` const-generic should match the number of
/// linguistic terms (e.g. 3 for {low, medium, high}).
pub struct LinguisticVar<S: ControlScalar, const N: usize> {
    /// Name hash of the variable itself.
    pub var_hash: u32,
    /// One entry per fuzzy term; each entry is (term_name_hash, membership_fn_ptr).
    #[allow(clippy::type_complexity)]
    fns: Vec<(u32, fn(S) -> S), N>,
}

impl<S: ControlScalar, const N: usize> LinguisticVar<S, N> {
    /// Create a new empty linguistic variable.
    pub fn new(name: &str) -> Self {
        Self {
            var_hash: fnv1a_hash(name),
            fns: Vec::new(),
        }
    }

    /// Add a membership function term.
    ///
    /// Returns `FuzzyError::CapacityExceeded` when the `N` limit is reached.
    pub fn add_term(&mut self, term_name: &str, f: fn(S) -> S) -> Result<(), FuzzyError> {
        self.fns
            .push((fnv1a_hash(term_name), f))
            .map_err(|_| FuzzyError::CapacityExceeded)
    }

    /// Evaluate all membership functions for a crisp input `x`.
    ///
    /// Returns a `heapless::Vec` of `FuzzySet` values, one per term.
    pub fn fuzzify(&self, x: S) -> Vec<FuzzySet<S>, N> {
        let mut out: Vec<FuzzySet<S>, N> = Vec::new();
        for (hash, f) in self.fns.iter() {
            let mu = f(x);
            // Clamp to [0, 1] for safety
            let mu_clamped = mu.clamp_val(S::ZERO, S::ONE);
            // Infallible: `out` has same capacity as `self.fns`
            let _ = out.push(FuzzySet::new(*hash, mu_clamped));
        }
        out
    }

    /// Return the membership at slot index `idx`, or `None` if out of bounds.
    pub fn membership_at(&self, x: S, idx: usize) -> Option<S> {
        self.fns.get(idx).map(|(_, f)| {
            let mu = f(x);
            mu.clamp_val(S::ZERO, S::ONE)
        })
    }

    /// Number of terms registered.
    pub fn len(&self) -> usize {
        self.fns.len()
    }

    /// Whether no terms are registered.
    pub fn is_empty(&self) -> bool {
        self.fns.is_empty()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Antecedent / Consequent
// ────────────────────────────────────────────────────────────────────────────

/// Maximum number of antecedent conditions in a single rule.
pub const MAX_ANTECEDENTS: usize = 4;

/// A single antecedent condition: `(var_idx, set_idx)`.
///
/// `var_idx` indexes into the array of linguistic variables provided at
/// inference time; `set_idx` indexes into that variable's terms.
#[derive(Debug, Clone, Copy)]
pub struct Condition {
    pub var_idx: usize,
    pub set_idx: usize,
}

/// Conjunction (AND) of up to `MAX_ANTECEDENTS` conditions.
#[derive(Debug, Clone)]
pub struct Antecedent {
    pub conditions: Vec<Condition, MAX_ANTECEDENTS>,
}

impl Antecedent {
    /// Construct an empty antecedent.
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Add a condition. Returns error if capacity is exceeded.
    pub fn add(&mut self, var_idx: usize, set_idx: usize) -> Result<(), FuzzyError> {
        self.conditions
            .push(Condition { var_idx, set_idx })
            .map_err(|_| FuzzyError::CapacityExceeded)
    }
}

impl Default for Antecedent {
    fn default() -> Self {
        Self::new()
    }
}

/// The consequent of a fuzzy rule: which output variable / set is activated,
/// and an optional weight `w ∈ (0, 1]`.
#[derive(Debug, Clone, Copy)]
pub struct Consequent {
    pub var_idx: usize,
    pub set_idx: usize,
    /// Rule weight in `(0, 1]`.
    pub weight: f64,
}

impl Consequent {
    /// Construct with explicit weight.
    pub fn new(var_idx: usize, set_idx: usize, weight: f64) -> Self {
        Self {
            var_idx,
            set_idx,
            weight,
        }
    }

    /// Construct with default weight = 1.
    pub fn unit(var_idx: usize, set_idx: usize) -> Self {
        Self::new(var_idx, set_idx, 1.0)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FuzzyRule
// ────────────────────────────────────────────────────────────────────────────

/// A single fuzzy rule: IF `antecedent` THEN `consequent`.
#[derive(Debug, Clone)]
pub struct FuzzyRule {
    pub antecedent: Antecedent,
    pub consequent: Consequent,
}

impl FuzzyRule {
    pub fn new(antecedent: Antecedent, consequent: Consequent) -> Self {
        Self {
            antecedent,
            consequent,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// RuleBase
// ────────────────────────────────────────────────────────────────────────────

/// A fixed-capacity heapless array of fuzzy rules.
pub struct RuleBase<const R: usize> {
    rules: Vec<FuzzyRule, R>,
    pub t_norm: TNorm,
}

impl<const R: usize> RuleBase<R> {
    /// Construct an empty rule base with a given T-norm.
    pub fn new(t_norm: TNorm) -> Self {
        Self {
            rules: Vec::new(),
            t_norm,
        }
    }

    /// Add a rule. Returns error when capacity `R` is exceeded.
    pub fn add_rule(&mut self, rule: FuzzyRule) -> Result<(), FuzzyError> {
        self.rules
            .push(rule)
            .map_err(|_| FuzzyError::CapacityExceeded)
    }

    /// Number of rules currently stored.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Whether the rule base is empty.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Compute the firing strength of a single rule given a 2-D slice of
    /// pre-computed membership values.
    ///
    /// `input_memberships[var_idx][set_idx]` gives the membership degree of
    /// input variable `var_idx` in term `set_idx`.
    ///
    /// Returns `0` if any index is out of bounds (degenerate rule).
    pub fn fire_rule<S: ControlScalar>(&self, rule: &FuzzyRule, input_memberships: &[&[S]]) -> S {
        if rule.antecedent.conditions.is_empty() {
            return S::ZERO;
        }
        let mut strength = S::ONE;
        for cond in rule.antecedent.conditions.iter() {
            let mu = match input_memberships.get(cond.var_idx) {
                Some(sets) => match sets.get(cond.set_idx) {
                    Some(&m) => m,
                    None => return S::ZERO,
                },
                None => return S::ZERO,
            };
            strength = self.t_norm.apply(strength, mu);
        }
        // Apply rule weight
        let w = S::from_f64(rule.consequent.weight);
        strength * w
    }

    /// Iterate over rules to fire all of them.
    ///
    /// Returns a `heapless::Vec` of `(firing_strength, consequent)` pairs.
    pub fn fire_all<S: ControlScalar>(
        &self,
        input_memberships: &[&[S]],
    ) -> Vec<(S, Consequent), R> {
        let mut out: Vec<(S, Consequent), R> = Vec::new();
        for rule in self.rules.iter() {
            let strength = self.fire_rule(rule, input_memberships);
            // Infallible: same capacity bound as `self.rules`
            let _ = out.push((strength, rule.consequent));
        }
        out
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple two-condition antecedent
    fn make_rule(v0: usize, s0: usize, v1: usize, s1: usize) -> FuzzyRule {
        let mut ant = Antecedent::new();
        ant.add(v0, s0).unwrap();
        ant.add(v1, s1).unwrap();
        let con = Consequent::unit(0, 0);
        FuzzyRule::new(ant, con)
    }

    #[test]
    fn tnorm_product_both_half() {
        let t = TNorm::Product;
        let v: f64 = t.apply(0.5, 0.5);
        assert!((v - 0.25).abs() < 1e-12);
    }

    #[test]
    fn tnorm_min_selects_smaller() {
        let t = TNorm::Min;
        assert_eq!(t.apply(0.3_f64, 0.7_f64), 0.3);
    }

    #[test]
    fn tnorm_fold_empty_is_one() {
        let t = TNorm::Product;
        let v: f64 = t.fold(&[]);
        assert_eq!(v, 1.0);
    }

    #[test]
    fn single_rule_fires_correctly_product() {
        let mut rb: RuleBase<4> = RuleBase::new(TNorm::Product);
        rb.add_rule(make_rule(0, 0, 1, 0)).unwrap();

        // var0_set0 = 0.8, var1_set0 = 0.6 → product = 0.48
        let v0 = [0.8_f64];
        let v1 = [0.6_f64];
        let memberships: &[&[f64]] = &[&v0, &v1];

        let fired = rb.fire_all(memberships);
        assert_eq!(fired.len(), 1);
        assert!((fired[0].0 - 0.48).abs() < 1e-12);
    }

    #[test]
    fn single_rule_fires_correctly_min() {
        let mut rb: RuleBase<4> = RuleBase::new(TNorm::Min);
        rb.add_rule(make_rule(0, 0, 1, 0)).unwrap();

        let v0 = [0.8_f64];
        let v1 = [0.6_f64];
        let memberships: &[&[f64]] = &[&v0, &v1];

        let fired = rb.fire_all(memberships);
        assert!((fired[0].0 - 0.6).abs() < 1e-12);
    }

    #[test]
    fn contradictory_input_zero_firing() {
        let mut rb: RuleBase<4> = RuleBase::new(TNorm::Min);
        rb.add_rule(make_rule(0, 0, 1, 0)).unwrap();

        // var0_set0 = 0.0 → any T-norm yields 0
        let v0 = [0.0_f64];
        let v1 = [1.0_f64];
        let memberships: &[&[f64]] = &[&v0, &v1];

        let fired = rb.fire_all(memberships);
        assert_eq!(fired[0].0, 0.0);
    }

    #[test]
    fn out_of_bounds_var_returns_zero() {
        let rb: RuleBase<4> = RuleBase::new(TNorm::Product);
        let mut ant = Antecedent::new();
        ant.add(99, 0).unwrap(); // var_idx=99 does not exist
        let con = Consequent::unit(0, 0);
        let rule = FuzzyRule::new(ant, con);

        let v0 = [1.0_f64];
        let memberships: &[&[f64]] = &[&v0];
        let strength = rb.fire_rule::<f64>(&rule, memberships);
        assert_eq!(strength, 0.0);
    }

    #[test]
    fn rule_weight_scales_output() {
        let mut rb: RuleBase<4> = RuleBase::new(TNorm::Min);
        let mut ant = Antecedent::new();
        ant.add(0, 0).unwrap();
        let con = Consequent::new(0, 0, 0.5); // weight = 0.5
        rb.add_rule(FuzzyRule::new(ant, con)).unwrap();

        let v0 = [1.0_f64];
        let memberships: &[&[f64]] = &[&v0];
        let fired = rb.fire_all(memberships);
        assert!((fired[0].0 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn linguistic_var_fuzzify() {
        let mut var: LinguisticVar<f64, 3> = LinguisticVar::new("temperature");
        var.add_term("cold", |x| if x < 20.0 { 1.0 } else { 0.0 })
            .unwrap();
        var.add_term("warm", |x| {
            if (15.0..=25.0).contains(&x) {
                (x - 15.0) / 10.0
            } else {
                0.0
            }
        })
        .unwrap();
        var.add_term("hot", |x| if x > 30.0 { 1.0 } else { 0.0 })
            .unwrap();

        let sets = var.fuzzify(18.0);
        assert_eq!(sets.len(), 3);
        // cold at 18 → 1.0
        assert!((sets[0].membership - 1.0).abs() < 1e-9);
        // hot at 18 → 0.0
        assert_eq!(sets[2].membership, 0.0);
    }

    #[test]
    fn fnv1a_hash_deterministic() {
        let h1 = fnv1a_hash("error");
        let h2 = fnv1a_hash("error");
        assert_eq!(h1, h2);
        assert_ne!(fnv1a_hash("error"), fnv1a_hash("error_rate"));
    }
}
