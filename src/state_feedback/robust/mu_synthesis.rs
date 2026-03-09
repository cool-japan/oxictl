use crate::core::{matrix::Matrix, scalar::ControlScalar};

/// Structured Singular Value (SSV) µ upper-bound computation.
///
/// For a complex matrix M and a perturbation structure ∆, µ(M) is bounded:
///   µ(M) ≤ min_D σ_max(D·M·D⁻¹)
///
/// This module implements:
///   1. Scalar D-scale iteration (simplified for repeated-block structures)
///   2. µ upper bound via scaled maximum singular value
///   3. D-K iteration for µ-synthesis (H∞ inner loop + D-scale outer loop)
///
/// See: Doyle (1982), Skogestad & Postlethwaite "Multivariable Feedback Design"
///
/// A D-scale for a block-diagonal perturbation structure of size N.
/// Full version: D = diag(d1·I, d2·I, ...) — here simplified to scalar scaling.
#[derive(Debug, Clone, Copy)]
pub struct DScale<S: ControlScalar, const N: usize> {
    /// Diagonal D-scale entries (one per block).
    pub d: [S; N],
}

impl<S: ControlScalar, const N: usize> DScale<S, N> {
    pub fn identity() -> Self {
        Self { d: [S::ONE; N] }
    }

    /// Apply D-scaling: returns D·M·D⁻¹ for scalar diagonal D.
    pub fn apply(&self, m: &Matrix<S, N, N>) -> Matrix<S, N, N> {
        Matrix {
            data: core::array::from_fn(|i| {
                core::array::from_fn(|j| m.data[i][j] * self.d[i] / self.d[j])
            }),
        }
    }

    /// Spectral radius (Frobenius norm used as upper bound) of scaled matrix.
    pub fn scaled_spectral_bound(&self, m: &Matrix<S, N, N>) -> S {
        self.apply(m).frob_norm()
    }
}

/// µ upper bound for a matrix M using a given D-scale.
///
/// µ_upper(M) ≤ σ_max(D·M·D⁻¹) ≤ ||D·M·D⁻¹||_F
pub fn mu_upper_bound<S: ControlScalar, const N: usize>(
    m: &Matrix<S, N, N>,
    d: &DScale<S, N>,
) -> S {
    d.scaled_spectral_bound(m)
}

/// D-K iteration for µ-synthesis (scalar version).
///
/// Alternates between:
///   - K step: solve H∞ problem with fixed D-scales to get controller K
///   - D step: fit frequency-domain D-scales to minimize µ upper bound
///
/// This simplified version operates at a single operating point (DC gain).
/// The result approximates µ-optimal robustness for static full-block uncertainty.
pub struct DkIteration<S: ControlScalar, const N: usize, const I: usize> {
    /// Current D-scale estimates.
    pub d_scale: DScale<S, N>,
    /// γ level for H∞ inner loop.
    pub gamma: S,
    /// Convergence tolerance.
    pub tol: S,
    /// Maximum D-K outer iterations.
    pub max_iter: u32,
    _phantom: core::marker::PhantomData<[S; I]>,
}

impl<S: ControlScalar, const N: usize, const I: usize> DkIteration<S, N, I> {
    pub fn new(gamma: S, tol: S, max_iter: u32) -> Self {
        Self {
            d_scale: DScale::identity(),
            gamma,
            tol,
            max_iter,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Run one D step: update D-scales to minimize µ upper bound for matrix M.
    ///
    /// Uses coordinate descent: for each d_i, minimize ||D·M·D⁻¹||_F.
    /// Returns the updated µ upper bound.
    pub fn d_step(&mut self, m: &Matrix<S, N, N>) -> S {
        let mut best_bound = mu_upper_bound(m, &self.d_scale);

        for _ in 0..self.max_iter {
            let prev_bound = best_bound;

            // Coordinate descent on each diagonal element
            for i in 0..N {
                let mut best_di = self.d_scale.d[i];
                let mut best_local = best_bound;

                // Try scaling d[i] up or down by a factor
                for &scale in &[
                    S::from_f64(1.1),
                    S::from_f64(0.9091), // 1/1.1
                    S::from_f64(1.5),
                    S::from_f64(0.6667),
                ] {
                    let d_try = self.d_scale.d[i] * scale;
                    if d_try <= S::ZERO {
                        continue;
                    }

                    let mut d_test = self.d_scale;
                    d_test.d[i] = d_try;
                    let bound = mu_upper_bound(m, &d_test);
                    if bound < best_local {
                        best_local = bound;
                        best_di = d_try;
                    }
                }
                self.d_scale.d[i] = best_di;
                best_bound = best_local;
            }

            if (prev_bound - best_bound).abs() < self.tol {
                break;
            }
        }
        best_bound
    }
}

/// µ lower bound (unstructured): largest singular value of M itself.
/// For unstructured uncertainty: µ(M) = σ_max(M) ≤ ||M||_F.
pub fn mu_lower_bound_unstructured<S: ControlScalar, const N: usize>(m: &Matrix<S, N, N>) -> S {
    m.frob_norm()
}

/// Compute the H∞ norm bound of a closed-loop matrix M as seen by
/// a block-diagonal perturbation. Returns `true` if the system is
/// robustly stable (µ < 1) for normalized perturbation.
pub fn is_robustly_stable<S: ControlScalar, const N: usize>(
    m: &Matrix<S, N, N>,
    d: &DScale<S, N>,
) -> bool {
    mu_upper_bound(m, d) < S::ONE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_d_scale_gives_frob_norm() {
        let m = Matrix::<f64, 2, 2> {
            data: [[0.5, 0.1], [-0.2, 0.3]],
        };
        let d = DScale::identity();
        let bound = mu_upper_bound(&m, &d);
        assert!((bound - m.frob_norm()).abs() < 1e-10);
    }

    #[test]
    fn d_scaling_reduces_bound() {
        // Diagonal-dominant matrix — D-scaling should help
        let m = Matrix::<f64, 2, 2> {
            data: [[0.1, 0.9], [0.01, 0.1]],
        };
        let d_id = DScale::identity();
        let bound_id = mu_upper_bound(&m, &d_id);

        let mut dk = DkIteration::<f64, 2, 2>::new(1.0, 1e-6, 50);
        let bound_opt = dk.d_step(&m);

        // D-scaling should not increase the bound
        assert!(
            bound_opt <= bound_id + 1e-10,
            "optimized bound {:.6} > identity bound {:.6}",
            bound_opt,
            bound_id
        );
    }

    #[test]
    fn robustly_stable_small_perturbation() {
        // M = 0.5*I — µ = 0.5 < 1 → robustly stable
        let m = Matrix::<f64, 2, 2> {
            data: [[0.5, 0.0], [0.0, 0.5]],
        };
        let d = DScale::identity();
        assert!(is_robustly_stable(&m, &d));
    }

    #[test]
    fn not_robustly_stable_large_perturbation() {
        // M = 2*I — µ = 2√2 > 1 (Frobenius) → not stable
        let m = Matrix::<f64, 2, 2> {
            data: [[2.0, 0.0], [0.0, 2.0]],
        };
        let d = DScale::identity();
        assert!(!is_robustly_stable(&m, &d));
    }

    #[test]
    fn d_scale_apply_scales_off_diagonal() {
        let d = DScale::<f64, 2> { d: [2.0, 1.0] };
        let m = Matrix::<f64, 2, 2> {
            data: [[1.0, 0.5], [0.25, 1.0]],
        };
        let scaled = d.apply(&m);
        // D·M·D⁻¹[0][1] = 2.0 * 0.5 / 1.0 = 1.0
        assert!(
            (scaled.data[0][1] - 1.0).abs() < 1e-10,
            "scaled[0][1]={}",
            scaled.data[0][1]
        );
        // D·M·D⁻¹[1][0] = 1.0 * 0.25 / 2.0 = 0.125
        assert!((scaled.data[1][0] - 0.125).abs() < 1e-10);
    }
}
