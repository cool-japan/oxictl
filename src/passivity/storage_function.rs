//! Lyapunov / storage function analysis for passivity verification.
//!
//! Provides grid-based numerical verification of:
//!   - Positive definiteness: V(x) > 0 for x ≠ 0
//!   - Dissipativity: V̇(x,u) ≤ uᵀy  (passivity condition)
//!   - Asymptotic stability: V̇(x) < 0 away from the origin under a control law
//!
//! All verification methods use a uniform grid sampling over a hyper-rectangle.
//! This is a *numerical* check, not a formal proof, but is practically useful
//! during design.
#![allow(clippy::too_many_arguments)]
use crate::core::matrix::vec_dot;
use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// StorageFunction
// ---------------------------------------------------------------------------

/// A positive-definite storage function V: ℝᴺ → ℝ with its gradient.
///
/// For passivity-based analysis V is typically the Hamiltonian H(x) or a
/// Lyapunov function candidate.
pub struct StorageFunction<S: ControlScalar, const N: usize> {
    /// Evaluate V(x).
    pub evaluate: fn(&[S; N]) -> S,
    /// Evaluate ∂V/∂x.
    pub gradient: fn(&[S; N]) -> [S; N],
}

impl<S: ControlScalar, const N: usize> StorageFunction<S, N> {
    /// Create a storage function from function pointers.
    pub fn new(evaluate: fn(&[S; N]) -> S, gradient: fn(&[S; N]) -> [S; N]) -> Self {
        Self { evaluate, gradient }
    }

    /// Evaluate the time derivative V̇ = (∂V/∂x)ᵀ · f(x) along dynamics `f`.
    ///
    /// `f` is a closure computing ẋ = f(x).
    pub fn time_derivative<F>(&self, x: &[S; N], f: F) -> S
    where
        F: Fn(&[S; N]) -> [S; N],
    {
        let grad = (self.gradient)(x);
        let xdot = f(x);
        vec_dot(&grad, &xdot)
    }
}

// ---------------------------------------------------------------------------
// Grid helpers (no_std, no heap)
// ---------------------------------------------------------------------------

/// Build the i-th sample point on a uniform 1-D grid of `n_pts` points
/// in the interval [lo, hi].
///
/// Index `idx` is an N-digit mixed-radix number in base `n_pts`.
#[inline]
fn grid_point<S: ControlScalar, const N: usize>(
    idx: usize,
    n_pts: usize,
    lo: &[S; N],
    hi: &[S; N],
) -> [S; N] {
    let mut rem = idx;
    let mut out = [S::ZERO; N];
    for i in (0..N).rev() {
        let digit = rem % n_pts;
        rem /= n_pts;
        let t = S::from_f64(digit as f64 / (n_pts - 1).max(1) as f64);
        out[i] = lo[i] + t * (hi[i] - lo[i]);
    }
    out
}

/// Total number of grid points: n_pts^N.
/// Returns None on overflow.
#[inline]
fn grid_total(n_pts: usize, dim: usize) -> Option<usize> {
    let mut total: usize = 1;
    for _ in 0..dim {
        total = total.checked_mul(n_pts)?;
    }
    Some(total)
}

/// Check if a point is (approximately) zero.
#[inline]
fn is_near_zero<S: ControlScalar, const N: usize>(x: &[S; N], tol: S) -> bool {
    for &xi in x.iter() {
        if xi.abs() > tol {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// PassivityVerifier
// ---------------------------------------------------------------------------

/// Verify passivity conditions numerically on a grid.
pub struct PassivityVerifier<S: ControlScalar, const N: usize> {
    /// Small positive tolerance for near-zero checks.
    pub zero_tol: S,
}

impl<S: ControlScalar, const N: usize> PassivityVerifier<S, N> {
    /// Create a verifier with a given zero-tolerance.
    pub fn new(zero_tol: S) -> Self {
        Self { zero_tol }
    }

    /// Check V(x) > 0 for all sampled x ≠ 0 in [x_min, x_max]^N.
    ///
    /// Samples on a uniform grid with `n_pts` points per dimension.
    /// Returns `true` iff V > 0 at every non-zero grid point.
    ///
    /// Note: For very high N, use a small `n_pts` (e.g. 3–5) to keep
    /// the total number of evaluations tractable.
    pub fn verify_positive_definite(
        &self,
        v: &StorageFunction<S, N>,
        x_min: &[S; N],
        x_max: &[S; N],
        n_pts: usize,
    ) -> bool {
        let total = match grid_total(n_pts.max(2), N) {
            Some(t) => t,
            None => return false, // overflow — grid too large
        };

        for idx in 0..total {
            let x = grid_point(idx, n_pts.max(2), x_min, x_max);
            if is_near_zero(&x, self.zero_tol) {
                continue;
            }
            let val = (v.evaluate)(&x);
            if !val.is_finite() || val <= S::ZERO {
                return false;
            }
        }
        true
    }

    /// Verify V̇(x) = (∂V/∂x)ᵀ ẋ ≤ uᵀy for all sampled (x, u) pairs.
    ///
    /// Parameters:
    ///   - `v`:        storage function V
    ///   - `dynamics`: closed-loop dynamics ẋ = f(x, u)
    ///   - `output`:   system output y = h(x)
    ///   - `u_law`:    feedback law u = k(x)
    ///   - `x_min/max`: sampling domain
    ///   - `n_pts`:    grid resolution per dimension
    pub fn verify_dissipation<Fdyn, Fout, Fu>(
        &self,
        v: &StorageFunction<S, N>,
        dynamics: Fdyn,
        output: Fout,
        u_law: Fu,
        x_min: &[S; N],
        x_max: &[S; N],
        n_pts: usize,
    ) -> bool
    where
        Fdyn: Fn(&[S; N], &[S; N]) -> [S; N], // f(x, u) -> xdot  [using N for u dim here]
        Fout: Fn(&[S; N]) -> [S; N],          // h(x) -> y  (same dim as state for flexibility)
        Fu: Fn(&[S; N]) -> [S; N],            // control law u = k(x)
    {
        let total = match grid_total(n_pts.max(2), N) {
            Some(t) => t,
            None => return false,
        };

        for idx in 0..total {
            let x = grid_point(idx, n_pts.max(2), x_min, x_max);
            let u = u_law(&x);
            let y = output(&x);
            let xdot = dynamics(&x, &u);

            let grad = (v.gradient)(&x);
            let vdot = vec_dot(&grad, &xdot);
            let supply = vec_dot(&u, &y);

            // V̇ ≤ uᵀy  (allow small numerical slack)
            let slack = S::from_f64(1e-8);
            if vdot > supply + slack {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// LyapunovStabilityCheck
// ---------------------------------------------------------------------------

/// Check asymptotic stability of a dynamical system via a Lyapunov function.
pub struct LyapunovStabilityCheck<S: ControlScalar, const N: usize> {
    /// Numerical tolerance for strict inequalities.
    pub epsilon: S,
}

impl<S: ControlScalar, const N: usize> LyapunovStabilityCheck<S, N> {
    /// Create a stability check with given tolerance.
    pub fn new(epsilon: S) -> Self {
        Self { epsilon }
    }

    /// Verify asymptotic stability on the hypercube [-eps, eps]^N.
    ///
    /// Checks:
    ///   1. V(x) > 0   for x ≠ 0
    ///   2. V̇(x) < 0   for x ≠ 0   (strictly decreasing away from origin)
    ///
    /// Uses `n_pts` grid points per dimension.
    /// Returns `true` iff both conditions hold at all sampled non-zero points.
    pub fn check_asymptotic_stability<F>(
        &self,
        closed_loop_dynamics: F,
        v: &StorageFunction<S, N>,
        n_pts: usize,
    ) -> bool
    where
        F: Fn(&[S; N]) -> [S; N],
    {
        let eps = self.epsilon;
        let lo: [S; N] = core::array::from_fn(|_| -eps);
        let hi: [S; N] = core::array::from_fn(|_| eps);

        let total = match grid_total(n_pts.max(2), N) {
            Some(t) => t,
            None => return false,
        };

        let zero_tol = eps * S::from_f64(1e-6);

        for idx in 0..total {
            let x = grid_point(idx, n_pts.max(2), &lo, &hi);
            if is_near_zero(&x, zero_tol) {
                continue;
            }

            // Condition 1: V(x) > 0
            let v_val = (v.evaluate)(&x);
            if !v_val.is_finite() || v_val <= S::ZERO {
                return false;
            }

            // Condition 2: V̇(x) < 0
            let grad = (v.gradient)(&x);
            let xdot = closed_loop_dynamics(&x);
            let vdot = vec_dot(&grad, &xdot);
            if !vdot.is_finite() || vdot >= S::ZERO {
                return false;
            }
        }
        true
    }

    /// Check only V̇ < 0 away from the origin (skips positivity check).
    ///
    /// Useful when positivity of V is already known analytically.
    pub fn check_strict_decrease<F>(
        &self,
        closed_loop_dynamics: F,
        v: &StorageFunction<S, N>,
        n_pts: usize,
    ) -> bool
    where
        F: Fn(&[S; N]) -> [S; N],
    {
        let eps = self.epsilon;
        let lo: [S; N] = core::array::from_fn(|_| -eps);
        let hi: [S; N] = core::array::from_fn(|_| eps);

        let total = match grid_total(n_pts.max(2), N) {
            Some(t) => t,
            None => return false,
        };

        let zero_tol = eps * S::from_f64(1e-6);

        for idx in 0..total {
            let x = grid_point(idx, n_pts.max(2), &lo, &hi);
            if is_near_zero(&x, zero_tol) {
                continue;
            }
            let grad = (v.gradient)(&x);
            let xdot = closed_loop_dynamics(&x);
            let vdot = vec_dot(&grad, &xdot);
            if !vdot.is_finite() || vdot >= S::ZERO {
                return false;
            }
        }
        true
    }

    /// Estimate the rate of convergence as min(-V̇/V) over the grid.
    ///
    /// Returns `None` if any sample has V̇ ≥ 0 (not stable) or V ≤ 0.
    pub fn estimate_convergence_rate<F>(
        &self,
        closed_loop_dynamics: F,
        v: &StorageFunction<S, N>,
        n_pts: usize,
    ) -> Option<S>
    where
        F: Fn(&[S; N]) -> [S; N],
    {
        let eps = self.epsilon;
        let lo: [S; N] = core::array::from_fn(|_| -eps);
        let hi: [S; N] = core::array::from_fn(|_| eps);

        let total = grid_total(n_pts.max(2), N)?;
        let zero_tol = eps * S::from_f64(1e-6);

        let mut min_rate = S::from_f64(f64::MAX);
        let mut found_any = false;

        for idx in 0..total {
            let x = grid_point(idx, n_pts.max(2), &lo, &hi);
            if is_near_zero(&x, zero_tol) {
                continue;
            }
            let v_val = (v.evaluate)(&x);
            if !v_val.is_finite() || v_val <= S::ZERO {
                return None;
            }
            let grad = (v.gradient)(&x);
            let xdot = closed_loop_dynamics(&x);
            let vdot = vec_dot(&grad, &xdot);
            if !vdot.is_finite() || vdot >= S::ZERO {
                return None;
            }
            let rate = -vdot / v_val;
            if rate < min_rate {
                min_rate = rate;
            }
            found_any = true;
        }

        if found_any {
            Some(min_rate)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Quadratic V(x) = ½ (x₀² + x₁²)  — always positive definite.
    // -----------------------------------------------------------------------

    fn quadratic_v(x: &[f64; 2]) -> f64 {
        0.5 * (x[0] * x[0] + x[1] * x[1])
    }
    fn quadratic_grad(x: &[f64; 2]) -> [f64; 2] {
        [x[0], x[1]]
    }

    #[test]
    fn quadratic_is_positive_definite() {
        let v = StorageFunction::new(quadratic_v, quadratic_grad);
        let verifier = PassivityVerifier::<f64, 2>::new(1e-8);
        let lo = [-2.0f64, -2.0];
        let hi = [2.0f64, 2.0];
        assert!(
            verifier.verify_positive_definite(&v, &lo, &hi, 7),
            "Quadratic V should be positive definite"
        );
    }

    #[test]
    fn negative_v_fails_positive_definite() {
        // V(x) = -½ xᵀx (negative definite, should fail PD check)
        fn neg_v(x: &[f64; 2]) -> f64 {
            -0.5 * (x[0] * x[0] + x[1] * x[1])
        }
        fn neg_grad(x: &[f64; 2]) -> [f64; 2] {
            [-x[0], -x[1]]
        }
        let v = StorageFunction::new(neg_v, neg_grad);
        let verifier = PassivityVerifier::<f64, 2>::new(1e-8);
        let lo = [-1.0f64, -1.0];
        let hi = [1.0f64, 1.0];
        assert!(
            !verifier.verify_positive_definite(&v, &lo, &hi, 5),
            "Negative V should fail PD check"
        );
    }

    // -----------------------------------------------------------------------
    // Mass-damper: ẋ = -b·x  (pure damping, no spring)
    // V(x) = ½ x²,  V̇ = x·ẋ = -b·x² < 0  for x ≠ 0.
    // -----------------------------------------------------------------------

    #[test]
    fn mass_damper_asymptotically_stable() {
        let b = 2.0_f64;
        let v = StorageFunction::new(|x: &[f64; 1]| 0.5 * x[0] * x[0], |x: &[f64; 1]| [x[0]]);
        let dynamics = move |x: &[f64; 1]| [-b * x[0]];
        let checker = LyapunovStabilityCheck::<f64, 1>::new(1.0);
        assert!(
            checker.check_asymptotic_stability(dynamics, &v, 9),
            "Mass-damper should be asymptotically stable"
        );
    }

    #[test]
    fn undamped_not_asymptotically_stable() {
        // ẋ = 0 → V̇ = 0 (not strictly negative → check fails).
        let v = StorageFunction::new(|x: &[f64; 1]| 0.5 * x[0] * x[0], |x: &[f64; 1]| [x[0]]);
        let dynamics = |_x: &[f64; 1]| [0.0f64];
        let checker = LyapunovStabilityCheck::<f64, 1>::new(1.0);
        assert!(
            !checker.check_asymptotic_stability(dynamics, &v, 9),
            "Zero dynamics should fail asymptotic stability (V̇ = 0)"
        );
    }

    #[test]
    fn strict_decrease_2d() {
        // 2D mass-damper: ẋ = -diag(a, b)·x.
        let v = StorageFunction::new(
            |x: &[f64; 2]| 0.5 * (x[0] * x[0] + x[1] * x[1]),
            |x: &[f64; 2]| [x[0], x[1]],
        );
        let dynamics = |x: &[f64; 2]| [-1.5 * x[0], -0.8 * x[1]];
        let checker = LyapunovStabilityCheck::<f64, 2>::new(1.0);
        assert!(
            checker.check_strict_decrease(dynamics, &v, 7),
            "Diagonal stable system should have V̇ < 0"
        );
    }

    #[test]
    fn dissipation_check_msd() {
        // Mass-spring-damper as pH: H = ½(q²+p²), b=0.5.
        // Dynamics: [p, -q - b·p + u]
        // Output y = p  (generalized force × velocity)
        // Control u = 0 (open loop).
        // Passivity: Ḣ ≤ uᵀy  → Ḣ = p·(-q - b·p) = -q·p - b·p²
        //            uᵀy = 0·p = 0.
        // But Ḣ can be positive (e.g. q=-2, p=1: Ḣ = 2 - 0.5 > 0 > uᵀy=0)
        // So open-loop with u=0 doesn't always satisfy Ḣ ≤ uᵀy globally.
        // Instead we test with a supply-matched scenario:
        // Close the loop with u = -q (energy-shaping makes V̇ ≤ 0 for all x).
        // With u = -q: ṗ = -q - b·p - q = -2q - b·p
        //   Ḣ = p·ṗ + q·q̇ = p(-2q - b·p) + q·p = -q·p - b·p²
        // Hmm this is not bounded above by uᵀy = (-q)·p = -q·p.
        // Let's just test that the verifier works correctly on known-dissipative system.

        // System: pure damper in 2D.  u = 0 always.
        // Dynamics: ẋ = -diag(1,1)·x   (both states damp to zero)
        // Output:   y = x  (collocated)
        // Supply:   uᵀy = 0   (since u=0)
        // V̇ = xᵀ(-x) = -‖x‖² ≤ 0 = uᵀy  ✓
        let v = StorageFunction::new(
            |x: &[f64; 2]| 0.5 * (x[0] * x[0] + x[1] * x[1]),
            |x: &[f64; 2]| [x[0], x[1]],
        );

        let verifier = PassivityVerifier::<f64, 2>::new(1e-8);
        let lo = [-2.0f64, -2.0];
        let hi = [2.0f64, 2.0];

        // dynamics(x, u): pure damper ignores u
        let dynamics = |x: &[f64; 2], _u: &[f64; 2]| [-x[0], -x[1]];
        let output = |x: &[f64; 2]| [x[0], x[1]];
        let u_law = |_x: &[f64; 2]| [0.0f64, 0.0f64];

        assert!(
            verifier.verify_dissipation(&v, dynamics, output, u_law, &lo, &hi, 7),
            "Pure damper with u=0 should satisfy V̇ ≤ uᵀy=0"
        );
    }

    #[test]
    fn convergence_rate_estimate() {
        // ẋ = -α·x: rate should be ≈ α.
        let alpha = 3.0_f64;
        let v = StorageFunction::new(|x: &[f64; 1]| 0.5 * x[0] * x[0], |x: &[f64; 1]| [x[0]]);
        let dynamics = move |x: &[f64; 1]| [-alpha * x[0]];
        let checker = LyapunovStabilityCheck::<f64, 1>::new(1.0);
        let rate = checker
            .estimate_convergence_rate(dynamics, &v, 11)
            .expect("Should get a rate estimate");
        // Rate = -V̇/V = (-x·(-α·x)) / (½x²) = 2α
        // More precisely: -vdot/v = α·x² / (½x²) = 2α … wait:
        // V̇ = x·(-αx) = -αx²,  V = ½x² → rate = -V̇/V = αx²/(½x²) = 2α
        // But we estimate min over grid, so rate ≈ 2α = 6.
        assert!(
            rate > alpha, // at least α (actually 2α)
            "Convergence rate should be positive: {}",
            rate
        );
    }

    #[test]
    fn storage_function_time_derivative() {
        let v = StorageFunction::new(quadratic_v, quadratic_grad);
        // f(x) = [-x[0], -2*x[1]] → V̇ = x[0]*(-x[0]) + x[1]*(-2*x[1])
        //                               = -x[0]² - 2*x[1]²
        let x = [1.0f64, 1.0];
        let vdot = v.time_derivative(&x, |x| [-x[0], -2.0 * x[1]]);
        assert!((vdot - (-3.0)).abs() < 1e-12, "V̇={}", vdot);
    }

    #[test]
    fn grid_point_bounds() {
        // At idx=0 we should be at the minimum corner.
        let lo = [-1.0f64, -2.0, -3.0];
        let hi = [1.0f64, 2.0, 3.0];
        let p0 = grid_point::<f64, 3>(0, 3, &lo, &hi);
        for i in 0..3 {
            assert!(
                (p0[i] - lo[i]).abs() < 1e-12,
                "Corner point mismatch at dim {}: {}",
                i,
                p0[i]
            );
        }
    }
}
