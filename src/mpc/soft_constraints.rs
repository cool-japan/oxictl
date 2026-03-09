//! Soft MPC constraints via slack variables with quadratic penalty.
//!
//! Hard constraints may make an MPC problem infeasible.  Soft constraints
//! introduce slack variables ε ≥ 0 so that the constraint C*x ≤ b becomes
//! C*x ≤ b + ε, and ρ*ε² is added to the cost.  This always keeps the
//! problem feasible while penalising constraint violation.
#![allow(unused)]

use crate::core::scalar::ControlScalar;

/// A single soft constraint: C*x ≤ b + ε  with penalty ρ*ε².
///
/// The slack ε is computed as max(0, c*x - b).
#[derive(Debug, Clone, Copy)]
pub struct SoftConstraint<S: ControlScalar> {
    /// Constraint coefficient c (scalar SISO form).
    pub c: S,
    /// Constraint bound b.
    pub b: S,
    /// Penalty weight ρ.
    pub rho: S,
    /// Current slack variable ε (updated by `compute_slack`).
    pub epsilon: S,
}

impl<S: ControlScalar> SoftConstraint<S> {
    /// Create a new soft constraint with zero initial slack.
    pub fn new(c: S, b: S, rho: S) -> Self {
        Self {
            c,
            b,
            rho,
            epsilon: S::ZERO,
        }
    }

    /// Evaluate and store the slack variable: ε = max(0, c*x - b).
    ///
    /// Returns the slack value.
    pub fn compute_slack(&mut self, x: S) -> S {
        let violation = self.c * x - self.b;
        self.epsilon = if violation > S::ZERO {
            violation
        } else {
            S::ZERO
        };
        self.epsilon
    }

    /// Penalty cost: ρ * ε².
    ///
    /// Requires `compute_slack` to have been called first.
    pub fn penalty_cost(&self) -> S {
        self.rho * self.epsilon * self.epsilon
    }

    /// Constraint violation: c*x - b.
    ///
    /// Negative means the constraint is satisfied; positive means violated.
    pub fn violation(&self, x: S) -> S {
        self.c * x - self.b
    }

    /// Check if constraint is currently satisfied (ε = 0).
    pub fn is_satisfied(&self) -> bool {
        self.epsilon <= S::ZERO
    }

    /// Exact penalty cost for a given x (does NOT update stored epsilon).
    pub fn penalty_cost_at(&self, x: S) -> S {
        let v = self.c * x - self.b;
        if v > S::ZERO {
            self.rho * v * v
        } else {
            S::ZERO
        }
    }
}

/// An ordered set of N soft constraints with priority labels.
///
/// Constraints are evaluated in priority order (lower number = higher priority).
pub struct SoftConstraintSet<S: ControlScalar, const N: usize> {
    /// Array of soft constraints.
    pub constraints: [SoftConstraint<S>; N],
    /// Priority for each constraint (lower = higher priority).
    pub priority: [u8; N],
}

impl<S: ControlScalar, const N: usize> SoftConstraintSet<S, N> {
    /// Create from an array of constraints (default priority = index order).
    pub fn new(constraints: [SoftConstraint<S>; N]) -> Self {
        let priority: [u8; N] = core::array::from_fn(|i| i as u8);
        Self {
            constraints,
            priority,
        }
    }

    /// Create with explicit priority ordering.
    pub fn with_priority(constraints: [SoftConstraint<S>; N], priority: [u8; N]) -> Self {
        Self {
            constraints,
            priority,
        }
    }

    /// Total penalty cost (sum of all ρ_i * ε_i²).
    ///
    /// Uses the stored slack values from the most recent `evaluate_all` call.
    pub fn total_penalty(&self) -> S {
        let mut total = S::ZERO;
        for c in &self.constraints {
            total += c.penalty_cost();
        }
        total
    }

    /// Evaluate all constraints at x, updating stored slack variables.
    ///
    /// Returns the total penalty cost.
    pub fn evaluate_all(&mut self, x: S) -> S {
        let mut total = S::ZERO;
        for c in self.constraints.iter_mut() {
            c.compute_slack(x);
            total += c.penalty_cost();
        }
        total
    }

    /// Maximum constraint violation value: max_i(c_i * x - b_i).
    pub fn max_violation(&self, x: S) -> S {
        let mut max = S::ZERO;
        for c in &self.constraints {
            let v = c.violation(x);
            if v > max {
                max = v;
            }
        }
        max
    }

    /// Check if any constraint is violated at x.
    pub fn any_violated(&self, x: S) -> bool {
        for c in &self.constraints {
            if c.violation(x) > S::ZERO {
                return true;
            }
        }
        false
    }

    /// Number of constraints violated at x.
    pub fn count_violated(&self, x: S) -> usize {
        let mut count = 0;
        for c in &self.constraints {
            if c.violation(x) > S::ZERO {
                count += 1;
            }
        }
        count
    }

    /// Return the highest-priority violated constraint index, if any.
    pub fn highest_priority_violation(&self, x: S) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_prio = u8::MAX;
        for (i, c) in self.constraints.iter().enumerate() {
            if c.violation(x) > S::ZERO && self.priority[i] < best_prio {
                best_prio = self.priority[i];
                best_idx = Some(i);
            }
        }
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slack_zero_when_satisfied() {
        let mut sc = SoftConstraint::new(1.0_f64, 5.0, 10.0);
        let eps = sc.compute_slack(3.0); // 1*3 - 5 = -2 < 0 → ε = 0
        assert!(eps.abs() < 1e-12, "Slack should be zero: {}", eps);
        assert!(sc.penalty_cost() < 1e-12);
    }

    #[test]
    fn slack_positive_when_violated() {
        let mut sc = SoftConstraint::new(1.0_f64, 2.0, 10.0);
        let eps = sc.compute_slack(5.0); // 1*5 - 2 = 3 > 0 → ε = 3
        assert!((eps - 3.0).abs() < 1e-12, "Slack should be 3: {}", eps);
        // Penalty = 10 * 9 = 90
        assert!(
            (sc.penalty_cost() - 90.0).abs() < 1e-12,
            "Penalty: {}",
            sc.penalty_cost()
        );
    }

    #[test]
    fn constraint_set_total_penalty() {
        let c0 = SoftConstraint::new(1.0_f64, 2.0, 1.0); // violated at x=5: ε=3, cost=9
        let c1 = SoftConstraint::new(1.0_f64, 10.0, 2.0); // satisfied at x=5: ε=0
        let mut set = SoftConstraintSet::new([c0, c1]);

        let penalty = set.evaluate_all(5.0);
        assert!((penalty - 9.0).abs() < 1e-12, "Total penalty: {}", penalty);
        assert_eq!(set.count_violated(5.0), 1);
    }

    #[test]
    fn max_violation_correct() {
        let c0 = SoftConstraint::new(1.0_f64, 1.0, 1.0); // viol = 4
        let c1 = SoftConstraint::new(2.0_f64, 0.0, 1.0); // viol = 10
        let set = SoftConstraintSet::new([c0, c1]);

        let max_v = set.max_violation(5.0);
        assert!((max_v - 10.0).abs() < 1e-12, "Max violation: {}", max_v);
    }

    #[test]
    fn any_violated_false_when_all_satisfied() {
        let c0 = SoftConstraint::new(1.0_f64, 10.0, 1.0);
        let c1 = SoftConstraint::new(1.0_f64, 20.0, 1.0);
        let set = SoftConstraintSet::new([c0, c1]);
        assert!(!set.any_violated(5.0));
    }

    #[test]
    fn highest_priority_violation() {
        let c0 = SoftConstraint::new(1.0_f64, 2.0, 1.0); // violated
        let c1 = SoftConstraint::new(1.0_f64, 3.0, 1.0); // violated
        let c2 = SoftConstraint::new(1.0_f64, 10.0, 1.0); // satisfied
        let set = SoftConstraintSet::with_priority([c0, c1, c2], [2, 0, 1]);
        // c1 has priority 0 (highest), and is violated at x=5
        let idx = set.highest_priority_violation(5.0);
        assert_eq!(idx, Some(1), "Highest priority violated: {:?}", idx);
    }
}
