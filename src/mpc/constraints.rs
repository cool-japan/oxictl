use crate::core::scalar::ControlScalar;

/// Box constraint on a scalar signal.
#[derive(Debug, Clone, Copy)]
pub struct BoxConstraint<S: ControlScalar> {
    pub min: S,
    pub max: S,
}

impl<S: ControlScalar> BoxConstraint<S> {
    pub fn new(min: S, max: S) -> Self {
        Self { min, max }
    }

    pub fn unconstrained() -> Self {
        Self {
            min: -S::from_f64(f64::MAX / 2.0),
            max: S::from_f64(f64::MAX / 2.0),
        }
    }

    /// Clamp value to constraint range.
    pub fn clamp(&self, v: S) -> S {
        v.clamp_val(self.min, self.max)
    }

    /// Returns true if value satisfies the constraint.
    pub fn is_satisfied(&self, v: S) -> bool {
        v >= self.min && v <= self.max
    }
}

/// Input constraints for M-dimensional control input.
#[derive(Debug, Clone, Copy)]
pub struct InputConstraints<S: ControlScalar, const M: usize> {
    /// Absolute bounds on each input.
    pub bounds: [BoxConstraint<S>; M],
    /// Rate-of-change bounds (Δu per step).
    pub delta_bounds: [BoxConstraint<S>; M],
}

impl<S: ControlScalar, const M: usize> InputConstraints<S, M> {
    pub fn new(min: S, max: S, delta_min: S, delta_max: S) -> Self {
        Self {
            bounds: core::array::from_fn(|_| BoxConstraint::new(min, max)),
            delta_bounds: core::array::from_fn(|_| BoxConstraint::new(delta_min, delta_max)),
        }
    }

    /// Apply input + rate constraints given previous input u_prev.
    pub fn apply(&self, u: &mut [S; M], u_prev: &[S; M]) {
        for i in 0..M {
            // Rate constraint first
            let du = u[i] - u_prev[i];
            let du_clamped = self.delta_bounds[i].clamp(du);
            u[i] = u_prev[i] + du_clamped;
            // Then absolute constraint
            u[i] = self.bounds[i].clamp(u[i]);
        }
    }

    /// Check if all constraints are satisfied.
    pub fn all_satisfied(&self, u: &[S; M], u_prev: &[S; M]) -> bool {
        for i in 0..M {
            if !self.bounds[i].is_satisfied(u[i]) {
                return false;
            }
            let du = u[i] - u_prev[i];
            if !self.delta_bounds[i].is_satisfied(du) {
                return false;
            }
        }
        true
    }
}

/// State constraints for N-dimensional state vector.
#[derive(Debug, Clone, Copy)]
pub struct StateConstraints<S: ControlScalar, const N: usize> {
    pub bounds: [BoxConstraint<S>; N],
}

impl<S: ControlScalar, const N: usize> StateConstraints<S, N> {
    pub fn new(min: S, max: S) -> Self {
        Self {
            bounds: core::array::from_fn(|_| BoxConstraint::new(min, max)),
        }
    }

    pub fn is_feasible(&self, x: &[S; N]) -> bool {
        x.iter()
            .enumerate()
            .all(|(i, &xi)| self.bounds[i].is_satisfied(xi))
    }

    /// Soft projection: clamp state to feasible region.
    pub fn project(&self, x: &mut [S; N]) {
        for (xi, bound) in x.iter_mut().zip(self.bounds.iter()) {
            *xi = bound.clamp(*xi);
        }
    }
}

/// Output (y = Cx) constraints for P-dimensional output.
#[derive(Debug, Clone, Copy)]
pub struct OutputConstraints<S: ControlScalar, const P: usize> {
    pub bounds: [BoxConstraint<S>; P],
}

impl<S: ControlScalar, const P: usize> OutputConstraints<S, P> {
    pub fn new(min: S, max: S) -> Self {
        Self {
            bounds: core::array::from_fn(|_| BoxConstraint::new(min, max)),
        }
    }

    pub fn is_feasible(&self, y: &[S; P]) -> bool {
        y.iter()
            .enumerate()
            .all(|(i, &yi)| self.bounds[i].is_satisfied(yi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_constraint_clamp() {
        let c = BoxConstraint::new(-1.0_f64, 1.0);
        assert_eq!(c.clamp(2.0), 1.0);
        assert_eq!(c.clamp(-3.0), -1.0);
        assert_eq!(c.clamp(0.5), 0.5);
    }

    #[test]
    fn input_constraints_rate_and_abs() {
        let ic = InputConstraints::<f64, 2>::new(-10.0, 10.0, -2.0, 2.0);
        let u_prev = [0.0_f64; 2];
        let mut u = [5.0_f64, -15.0]; // second violates absolute
        ic.apply(&mut u, &u_prev);
        // Rate limit: du[0]=5 > 2 → clamped to 2; absolute ok
        assert!((u[0] - 2.0).abs() < 1e-10, "u[0]={}", u[0]);
        // du[1]=-15 < -2 → clamped to -2; absolute: -2 >= -10 ok
        assert!((u[1] - (-2.0)).abs() < 1e-10, "u[1]={}", u[1]);
    }

    #[test]
    fn state_constraints_feasibility() {
        let sc = StateConstraints::<f64, 3>::new(-5.0, 5.0);
        assert!(sc.is_feasible(&[1.0, -2.0, 3.0]));
        assert!(!sc.is_feasible(&[6.0, 0.0, 0.0]));
    }

    #[test]
    fn state_projection() {
        let sc = StateConstraints::<f64, 2>::new(-1.0, 1.0);
        let mut x = [3.0_f64, -4.0];
        sc.project(&mut x);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], -1.0);
    }
}
