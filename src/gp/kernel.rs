//! GP covariance kernel functions.
//!
//! Each kernel implements the [`Kernel`] trait which computes k(x1, x2) for
//! input vectors of compile-time dimension `D`.

#![allow(clippy::needless_range_loop)]

use crate::core::scalar::ControlScalar;

// ──────────────────────────────────────────────────────────────────────────────
// Core trait
// ──────────────────────────────────────────────────────────────────────────────

/// Trait for GP covariance kernel functions k(x1, x2) → S.
///
/// Kernels must be symmetric and positive semi-definite.
pub trait Kernel<S: ControlScalar, const D: usize>: Clone {
    /// Evaluate the kernel at the pair of D-dimensional input vectors.
    fn eval(&self, x1: &[S; D], x2: &[S; D]) -> S;
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper: squared Euclidean distance
// ──────────────────────────────────────────────────────────────────────────────

#[inline]
fn squared_dist<S: ControlScalar, const D: usize>(x1: &[S; D], x2: &[S; D]) -> S {
    let mut acc = S::ZERO;
    for d in 0..D {
        let diff = x1[d] - x2[d];
        acc += diff * diff;
    }
    acc
}

// ──────────────────────────────────────────────────────────────────────────────
// Squared Exponential (RBF) kernel
// ──────────────────────────────────────────────────────────────────────────────

/// Squared Exponential (RBF) kernel:
///
/// k(x, x') = σ² · exp(−‖x − x'‖² / (2 l²))
///
/// where σ² = `variance` and l = `length_scale`.
#[derive(Debug, Clone, Copy)]
pub struct RbfKernel<S: ControlScalar> {
    /// Output variance σ².
    pub variance: S,
    /// Length-scale l.
    pub length_scale: S,
}

impl<S: ControlScalar, const D: usize> Kernel<S, D> for RbfKernel<S> {
    fn eval(&self, x1: &[S; D], x2: &[S; D]) -> S {
        let r2 = squared_dist(x1, x2);
        let two_l2 = self.length_scale * self.length_scale * S::TWO;
        let exponent = -(r2.to_f64() / two_l2.to_f64());
        self.variance * S::from_f64(libm::exp(exponent))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Matérn 5/2 kernel
// ──────────────────────────────────────────────────────────────────────────────

/// Matérn 5/2 kernel:
///
/// k(x, x') = σ² (1 + √5 r/l + 5r²/(3l²)) · exp(−√5 r/l)
///
/// where r = ‖x − x'‖.
#[derive(Debug, Clone, Copy)]
pub struct Matern52Kernel<S: ControlScalar> {
    /// Output variance σ².
    pub variance: S,
    /// Length-scale l.
    pub length_scale: S,
}

impl<S: ControlScalar, const D: usize> Kernel<S, D> for Matern52Kernel<S> {
    fn eval(&self, x1: &[S; D], x2: &[S; D]) -> S {
        let r2 = squared_dist(x1, x2).to_f64();
        let r = libm::sqrt(r2);
        let l = self.length_scale.to_f64();
        let sqrt5 = libm::sqrt(5.0_f64);
        let sr_over_l = sqrt5 * r / l;
        let poly = 1.0 + sr_over_l + 5.0 * r2 / (3.0 * l * l);
        let k = poly * libm::exp(-sr_over_l);
        self.variance * S::from_f64(k)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Linear (polynomial) kernel
// ──────────────────────────────────────────────────────────────────────────────

/// Linear / polynomial kernel:
///
/// k(x, x') = σ² (x · x' + c)^p
///
/// where σ² = `variance`, c = `bias`, p = `degree`.
#[derive(Debug, Clone, Copy)]
pub struct LinearKernel<S: ControlScalar> {
    /// Output variance σ².
    pub variance: S,
    /// Bias term c.
    pub bias: S,
    /// Polynomial degree p.
    pub degree: u32,
}

impl<S: ControlScalar, const D: usize> Kernel<S, D> for LinearKernel<S> {
    fn eval(&self, x1: &[S; D], x2: &[S; D]) -> S {
        let mut dot = S::ZERO;
        for d in 0..D {
            dot += x1[d] * x2[d];
        }
        let inner = (dot + self.bias).to_f64();
        let k = libm::pow(inner, self.degree as f64);
        self.variance * S::from_f64(k)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Additive kernel
// ──────────────────────────────────────────────────────────────────────────────

/// Additive kernel: k(x, x') = k1(x, x') + k2(x, x').
#[derive(Debug, Clone, Copy)]
pub struct AdditiveKernel<K1, K2> {
    /// First sub-kernel.
    pub k1: K1,
    /// Second sub-kernel.
    pub k2: K2,
}

impl<S, K1, K2, const D: usize> Kernel<S, D> for AdditiveKernel<K1, K2>
where
    S: ControlScalar,
    K1: Kernel<S, D>,
    K2: Kernel<S, D>,
{
    fn eval(&self, x1: &[S; D], x2: &[S; D]) -> S {
        self.k1.eval(x1, x2) + self.k2.eval(x1, x2)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── RBF kernel ────────────────────────────────────────────────────────────

    #[test]
    fn rbf_symmetry() {
        let k = RbfKernel::<f64> {
            variance: 1.0,
            length_scale: 1.0,
        };
        let x1 = [1.0_f64, 2.0];
        let x2 = [3.0_f64, 4.0];
        assert!(approx_eq(k.eval(&x1, &x2), k.eval(&x2, &x1), EPS));
    }

    #[test]
    fn rbf_diagonal_nonneg() {
        let k = RbfKernel::<f64> {
            variance: 2.5,
            length_scale: 0.5,
        };
        let x = [1.0_f64, -1.0, 3.0];
        assert!(k.eval(&x, &x) >= 0.0);
    }

    #[test]
    fn rbf_at_zero_distance() {
        let k = RbfKernel::<f64> {
            variance: 3.0,
            length_scale: 1.0,
        };
        let x = [0.0_f64];
        // k(x,x) = variance * exp(0) = variance
        assert!(approx_eq(k.eval(&x, &x), 3.0, 1e-12));
    }

    #[test]
    fn rbf_known_value() {
        // l=1, σ²=1 → k([0],[1]) = exp(-0.5)
        let k = RbfKernel::<f64> {
            variance: 1.0,
            length_scale: 1.0,
        };
        let x1 = [0.0_f64];
        let x2 = [1.0_f64];
        let expected = libm::exp(-0.5_f64);
        assert!(approx_eq(k.eval(&x1, &x2), expected, 1e-12));
    }

    // ── Matérn 5/2 kernel ─────────────────────────────────────────────────────

    #[test]
    fn matern52_symmetry() {
        let k = Matern52Kernel::<f64> {
            variance: 1.0,
            length_scale: 1.0,
        };
        let x1 = [1.0_f64, 2.0];
        let x2 = [3.0_f64, 4.0];
        assert!(approx_eq(k.eval(&x1, &x2), k.eval(&x2, &x1), EPS));
    }

    #[test]
    fn matern52_at_zero_distance() {
        let k = Matern52Kernel::<f64> {
            variance: 2.0,
            length_scale: 1.0,
        };
        let x = [0.0_f64];
        // k(x,x) = variance * (1 + 0 + 0) * exp(0) = variance
        assert!(approx_eq(k.eval(&x, &x), 2.0, 1e-12));
    }

    #[test]
    fn matern52_diagonal_nonneg() {
        let k = Matern52Kernel::<f64> {
            variance: 1.5,
            length_scale: 0.7,
        };
        let x = [2.0_f64, -3.0];
        assert!(k.eval(&x, &x) >= 0.0);
    }

    // ── Linear kernel ─────────────────────────────────────────────────────────

    #[test]
    fn linear_symmetry() {
        let k = LinearKernel::<f64> {
            variance: 1.0,
            bias: 1.0,
            degree: 2,
        };
        let x1 = [1.0_f64, 2.0];
        let x2 = [3.0_f64, 4.0];
        assert!(approx_eq(k.eval(&x1, &x2), k.eval(&x2, &x1), EPS));
    }

    #[test]
    fn linear_known_value() {
        // σ²=1, c=0, p=1: k(x,y) = x·y
        let k = LinearKernel::<f64> {
            variance: 1.0,
            bias: 0.0,
            degree: 1,
        };
        let x1 = [2.0_f64, 3.0];
        let x2 = [4.0_f64, 5.0];
        // dot = 2*4 + 3*5 = 8+15 = 23; (23+0)^1 * 1 = 23
        assert!(approx_eq(k.eval(&x1, &x2), 23.0, 1e-12));
    }

    // ── Additive kernel ───────────────────────────────────────────────────────

    #[test]
    fn additive_is_sum() {
        let k1 = RbfKernel::<f64> {
            variance: 1.0,
            length_scale: 1.0,
        };
        let k2 = Matern52Kernel::<f64> {
            variance: 0.5,
            length_scale: 2.0,
        };
        let k_add = AdditiveKernel { k1, k2 };
        let x1 = [1.0_f64];
        let x2 = [2.0_f64];
        let expected = k_add.k1.eval(&x1, &x2) + k_add.k2.eval(&x1, &x2);
        assert!(approx_eq(k_add.eval(&x1, &x2), expected, EPS));
    }

    #[test]
    fn additive_symmetry() {
        let k_add = AdditiveKernel {
            k1: RbfKernel::<f64> {
                variance: 1.0,
                length_scale: 1.0,
            },
            k2: LinearKernel::<f64> {
                variance: 0.5,
                bias: 1.0,
                degree: 2,
            },
        };
        let x1 = [1.0_f64, 0.5];
        let x2 = [0.3_f64, 2.1];
        assert!(approx_eq(k_add.eval(&x1, &x2), k_add.eval(&x2, &x1), EPS));
    }
}
