//! Constrained Kalman Filter: post-update projection onto state constraints.
//!
//! After each standard KF update, the state estimate is projected iteratively
//! onto the intersection of linear inequality constraints:
//!   lo ≤ d_row · x ≤ hi
//!
//! Reference: Simon & Simon (2006) "Constrained Kalman filtering via density
//! truncation." IEE Proceedings — Control Theory and Applications.
#![allow(clippy::needless_range_loop)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Linear state constraint: lo ≤ D[row] · x ≤ hi.
#[derive(Debug, Clone, Copy)]
pub struct StateConstraint<S: ControlScalar, const N: usize> {
    /// Row vector d such that lo ≤ d·x ≤ hi (1×N).
    pub d_row: Matrix<S, 1, N>,
    /// Lower bound.
    pub lo: S,
    /// Upper bound.
    pub hi: S,
}

impl<S: ControlScalar, const N: usize> StateConstraint<S, N> {
    /// Create a new state constraint.
    pub fn new(d_row: Matrix<S, 1, N>, lo: S, hi: S) -> Self {
        Self { d_row, lo, hi }
    }

    /// Check whether x satisfies this constraint.
    pub fn is_satisfied(&self, x: &Matrix<S, N, 1>) -> bool {
        let val = matmul(&self.d_row, x).data[0][0];
        val >= self.lo && val <= self.hi
    }

    /// Project x onto the constraint by clamping d·x into [lo, hi].
    ///
    /// Uses the projection: x ← x + d^T * (clamp(d·x, lo, hi) - d·x) / (d·d^T)
    pub fn project(&self, x: &mut Matrix<S, N, 1>) {
        let val = matmul(&self.d_row, x).data[0][0];
        let clamped = val.clamp_val(self.lo, self.hi);
        let diff = clamped - val;
        if diff.abs() < S::EPSILON {
            return;
        }
        // d·d^T (scalar)
        let dt = self.d_row.transpose(); // N×1
        let ddt = matmul(&self.d_row, &dt).data[0][0]; // scalar
        if ddt < S::EPSILON {
            return;
        }
        // x += d^T * diff / (d·d^T)
        let scale = diff / ddt;
        for i in 0..N {
            x.data[i][0] += dt.data[i][0] * scale;
        }
    }
}

/// Constrained KF: standard KF + iterative projection onto constraint set.
///
/// N: state dim, I: input dim, M: measurement dim, C: max constraints.
pub struct ConstrainedKf<
    S: ControlScalar,
    const N: usize,
    const I: usize,
    const M: usize,
    const C: usize,
> {
    /// State transition matrix (N×N).
    pub a: Matrix<S, N, N>,
    /// Input matrix (N×I).
    pub b: Matrix<S, N, I>,
    /// Measurement matrix (M×N).
    pub c_mat: Matrix<S, M, N>,
    /// Process noise covariance (N×N).
    pub q: Matrix<S, N, N>,
    /// Measurement noise covariance (M×M).
    pub r: Matrix<S, M, M>,
    /// Error covariance (N×N).
    pub p: Matrix<S, N, N>,
    /// State estimate (N×1).
    pub x_hat: Matrix<S, N, 1>,
    /// Constraint slots (up to C).
    pub constraints: [Option<StateConstraint<S, N>>; C],
    /// Number of constraint projection iterations per update.
    pub proj_iterations: usize,
}

impl<S: ControlScalar, const N: usize, const I: usize, const M: usize, const C: usize>
    ConstrainedKf<S, N, I, M, C>
{
    /// Create a new constrained KF with no constraints active.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        c_mat: Matrix<S, M, N>,
        q: Matrix<S, N, N>,
        r: Matrix<S, M, M>,
    ) -> Self {
        Self {
            a,
            b,
            c_mat,
            q,
            r,
            p: Matrix::<S, N, N>::identity(),
            x_hat: Matrix::<S, N, 1>::zeros(),
            constraints: core::array::from_fn(|_| None),
            proj_iterations: 3,
        }
    }

    /// Add a constraint at slot `slot`.
    pub fn with_constraint(mut self, slot: usize, d_row: Matrix<S, 1, N>, lo: S, hi: S) -> Self {
        if slot < C {
            self.constraints[slot] = Some(StateConstraint::new(d_row, lo, hi));
        }
        self
    }

    /// Predict step.
    pub fn predict(&mut self, u: &Matrix<S, I, 1>) {
        // x_hat = A * x_hat + B * u
        let ax = matmul(&self.a, &self.x_hat);
        let bu = matmul(&self.b, u);
        self.x_hat = ax.add_mat(&bu);

        // P = A * P * A^T + Q
        let ap = matmul(&self.a, &self.p);
        let at = self.a.transpose();
        let apat = matmul(&ap, &at);
        self.p = apat.add_mat(&self.q);
    }

    /// Update step followed by iterative constraint projection.
    pub fn update(&mut self, y: &Matrix<S, M, 1>) {
        let ct = self.c_mat.transpose();

        // Innovation covariance: S = C*P*C^T + R
        let cp = matmul(&self.c_mat, &self.p);
        let cpct = matmul(&cp, &ct);
        let s_mat = cpct.add_mat(&self.r);

        let s_inv = match s_mat.inv() {
            Some(inv) => inv,
            None => return,
        };

        // Kalman gain: K = P*C^T*S^{-1}
        let pct = matmul(&self.p, &ct);
        let k = matmul(&pct, &s_inv);

        // Innovation
        let cx = matmul(&self.c_mat, &self.x_hat);
        let innov = y.sub_mat(&cx);

        // State update
        let k_innov = matmul(&k, &innov);
        self.x_hat = self.x_hat.add_mat(&k_innov);

        // Covariance update: P = (I - K*C)*P
        let kc = matmul(&k, &self.c_mat);
        let eye = Matrix::<S, N, N>::identity();
        let i_minus_kc = eye.sub_mat(&kc);
        self.p = matmul(&i_minus_kc, &self.p);

        // Project onto constraints
        self.project_constraints();
    }

    /// Iteratively project the state onto all active constraints.
    fn project_constraints(&mut self) {
        for _ in 0..self.proj_iterations {
            for slot in self.constraints.iter().flatten() {
                slot.project(&mut self.x_hat);
            }
        }
    }

    /// State estimate accessor.
    pub fn state(&self) -> &Matrix<S, N, 1> {
        &self.x_hat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_kf() -> ConstrainedKf<f64, 2, 1, 1, 2> {
        let a = Matrix::<f64, 2, 2>::identity();
        let b = Matrix::<f64, 2, 1>::zeros();
        let c = Matrix::<f64, 1, 2> { data: [[1.0, 0.0]] };
        let q = Matrix::<f64, 2, 2>::identity().scale(0.01);
        let r = Matrix::<f64, 1, 1> { data: [[0.1]] };
        ConstrainedKf::new(a, b, c, q, r)
    }

    #[test]
    fn unconstrained_tracks_measurement() {
        let mut kf = identity_kf();
        let u = Matrix::<f64, 1, 1>::zeros();
        let y = Matrix::<f64, 1, 1> { data: [[3.0]] };
        for _ in 0..50 {
            kf.predict(&u);
            kf.update(&y);
        }
        let x = kf.state().data[0][0];
        assert!((x - 3.0).abs() < 0.1, "x={x}");
    }

    #[test]
    fn constraint_clamps_state() {
        // Constrain x[0] ≤ 2.0 (d = [1,0], lo=-inf, hi=2)
        let mut d_row = Matrix::<f64, 1, 2>::zeros();
        d_row.data[0][0] = 1.0;
        let kf = identity_kf().with_constraint(0, d_row, f64::NEG_INFINITY, 2.0);
        let mut kf = kf;
        let u = Matrix::<f64, 1, 1>::zeros();
        let y = Matrix::<f64, 1, 1> { data: [[5.0]] }; // measurement > constraint
        for _ in 0..30 {
            kf.predict(&u);
            kf.update(&y);
        }
        let x = kf.state().data[0][0];
        assert!(x <= 2.0 + 1e-9, "x={x} should be ≤ 2.0");
    }

    #[test]
    fn constraint_is_satisfied_check() {
        let mut d_row = Matrix::<f64, 1, 2>::zeros();
        d_row.data[0][0] = 1.0;
        let c = StateConstraint::new(d_row, -1.0, 1.0);
        let mut x_in = Matrix::<f64, 2, 1>::zeros();
        x_in.data[0][0] = 0.5;
        assert!(c.is_satisfied(&x_in));
        let mut x_out = Matrix::<f64, 2, 1>::zeros();
        x_out.data[0][0] = 2.0;
        assert!(!c.is_satisfied(&x_out));
    }

    #[test]
    fn projection_brings_inside_bounds() {
        let mut d_row = Matrix::<f64, 1, 2>::zeros();
        d_row.data[0][0] = 1.0;
        let c = StateConstraint::new(d_row, 0.0, 1.0);
        let mut x = Matrix::<f64, 2, 1>::zeros();
        x.data[0][0] = 3.0; // violates hi=1
        c.project(&mut x);
        assert!(x.data[0][0] <= 1.0 + 1e-9, "x={}", x.data[0][0]);
    }
}
