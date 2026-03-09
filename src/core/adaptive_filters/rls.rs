//! Recursive Least Squares (RLS) adaptive filter.
//!
//! RLS minimizes a weighted least-squares criterion with exponential
//! forgetting, enabling fast convergence and tracking of time-varying systems.
//!
//! The forgetting factor `λ ∈ (0, 1]` controls the memory of the algorithm:
//! - `λ = 1.0` — infinite memory (stationary signals)
//! - `λ < 1.0` — exponential weighting; effectively a window of ~1/(1-λ) samples
//!
//! # Complexity
//! Each `update` call performs O(N²) operations (matrix-vector products and
//! rank-one matrix updates).

use crate::core::adaptive_filters::lms::AdaptiveFilterError;
use crate::core::scalar::ControlScalar;

// ─────────────────────────────────────────────────────────────
//  RlsFilter
// ─────────────────────────────────────────────────────────────

/// Recursive Least Squares (RLS) adaptive filter.
///
/// Maintains an N×N inverse correlation matrix `P` and a weight vector `w`.
///
/// # Type Parameters
/// - `S` — scalar type implementing [`ControlScalar`]
/// - `N` — filter order (number of taps)
///
/// # Update equations
/// ```text
/// Px    = P · x
/// denom = λ + x^T · Px
/// k     = Px / denom                   (Kalman gain, length N)
/// y     = w^T · x
/// e     = d - y
/// w     = w + k · e
/// P     = (P − k · (x^T · P)) · λ⁻¹
/// ```
#[derive(Debug, Clone)]
pub struct RlsFilter<S: ControlScalar, const N: usize> {
    weights: [S; N],
    /// Inverse correlation matrix (N×N), stored row-major.
    p: [[S; N]; N],
    lambda: S,
    lambda_inv: S,
}

impl<S: ControlScalar, const N: usize> RlsFilter<S, N> {
    /// Create a new RLS filter.
    ///
    /// # Arguments
    /// * `lambda` — forgetting factor; must be in `(0, 1]`
    /// * `delta`  — initial diagonal value of P (large δ ≡ diffuse prior); must be > 0
    ///
    /// # Errors
    /// Returns [`AdaptiveFilterError::InvalidStepSize`] if `lambda` or `delta` are invalid.
    pub fn new(lambda: S, delta: S) -> Result<Self, AdaptiveFilterError> {
        if lambda <= S::ZERO || lambda > S::ONE {
            return Err(AdaptiveFilterError::InvalidStepSize);
        }
        if delta <= S::ZERO {
            return Err(AdaptiveFilterError::InvalidStepSize);
        }
        let mut p = [[S::ZERO; N]; N];
        for (i, row) in p.iter_mut().enumerate() {
            row[i] = delta;
        }
        Ok(Self {
            weights: [S::ZERO; N],
            p,
            lambda,
            lambda_inv: S::ONE / lambda,
        })
    }

    /// Process one sample and return the filter output before the weight update.
    ///
    /// # Errors
    /// Returns [`AdaptiveFilterError::Divergence`] if weights or P become non-finite.
    pub fn update(&mut self, x: &[S; N], d: S) -> Result<S, AdaptiveFilterError> {
        // Step 1: Px = P * x  (N-vector)
        let mut px = [S::ZERO; N];
        for (pxi, p_row) in px.iter_mut().zip(self.p.iter()) {
            for (&pij, &xj) in p_row.iter().zip(x.iter()) {
                *pxi += pij * xj;
            }
        }

        // Step 2: denom = lambda + x^T * Px
        let xtpx: S = x
            .iter()
            .zip(px.iter())
            .map(|(&xi, &pxi)| xi * pxi)
            .fold(S::ZERO, |a, b| a + b);
        let denom = self.lambda + xtpx;
        if denom.abs() < S::from_f64(1e-30) {
            return Err(AdaptiveFilterError::Divergence);
        }

        // Step 3: k = Px / denom  (Kalman gain)
        let mut k = [S::ZERO; N];
        for (ki, &pxi) in k.iter_mut().zip(px.iter()) {
            *ki = pxi / denom;
        }

        // Step 4: y = w^T * x,  e = d - y
        let y: S = self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .fold(S::ZERO, |a, b| a + b);
        let e = d - y;

        // Step 5: w = w + k * e
        for (w, &ki) in self.weights.iter_mut().zip(k.iter()) {
            *w += ki * e;
            if !w.is_finite() {
                return Err(AdaptiveFilterError::Divergence);
            }
        }

        // Step 6: P = (P - k * (x^T * P)) * lambda_inv
        // x^T * P:  xtp[j] = sum_l x[l] * P[l][j]
        let mut xtp = [S::ZERO; N];
        for (&xl, p_row) in x.iter().zip(self.p.iter()) {
            for (xtpj, &plj) in xtp.iter_mut().zip(p_row.iter()) {
                *xtpj += xl * plj;
            }
        }
        let lambda_inv = self.lambda_inv;
        for (p_row, &ki) in self.p.iter_mut().zip(k.iter()) {
            for (pij, &xtpj) in p_row.iter_mut().zip(xtp.iter()) {
                *pij = (*pij - ki * xtpj) * lambda_inv;
                if !pij.is_finite() {
                    return Err(AdaptiveFilterError::Divergence);
                }
            }
        }

        Ok(y)
    }

    /// Return a reference to the current weight vector.
    pub fn weights(&self) -> &[S; N] {
        &self.weights
    }

    /// Reset the filter: zero weights, reinitialize P to `delta * I`.
    pub fn reset(&mut self, delta: S) {
        self.weights = [S::ZERO; N];
        for (i, p_row) in self.p.iter_mut().enumerate() {
            for (j, pij) in p_row.iter_mut().enumerate() {
                *pij = if i == j { delta } else { S::ZERO };
            }
        }
    }

    /// Return the forgetting factor.
    pub fn lambda(&self) -> S {
        self.lambda
    }
}

// ─────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic LCG for test signal generation.
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_f64(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (self.state >> 11) as f64 / (1u64 << 53) as f64;
            u * 2.0 - 1.0
        }
    }

    fn apply_fir(x_history: &[f64; 4], coeffs: &[f64; 4]) -> f64 {
        coeffs
            .iter()
            .zip(x_history.iter())
            .map(|(c, x)| c * x)
            .sum()
    }

    #[test]
    fn rls_convergence_faster_than_lms_baseline() {
        // RLS should converge to the unknown FIR system in fewer iterations.
        use crate::core::adaptive_filters::lms::LmsFilter;

        let true_coeffs: [f64; 4] = [0.5, 0.3, 0.1, 0.05];
        let mut rls = RlsFilter::<f64, 4>::new(1.0, 1000.0).expect("valid");
        let mut lms = LmsFilter::<f64, 4>::new(0.01).expect("valid");

        let mut lcg = Lcg::new(42);
        let mut x_hist = [0.0_f64; 4];

        // Evaluate MSE over first 200 iterations (early-stage learning)
        let n_early = 200usize;
        let mut mse_rls = 0.0_f64;
        let mut mse_lms = 0.0_f64;

        for _ in 0..n_early {
            x_hist[3] = x_hist[2];
            x_hist[2] = x_hist[1];
            x_hist[1] = x_hist[0];
            x_hist[0] = lcg.next_f64();
            let d = apply_fir(&x_hist, &true_coeffs);

            let y_rls = rls.update(&x_hist, d).expect("rls ok");
            let y_lms = lms.update(&x_hist, d).expect("lms ok");
            mse_rls += (d - y_rls).powi(2);
            mse_lms += (d - y_lms).powi(2);
        }
        mse_rls /= n_early as f64;
        mse_lms /= n_early as f64;

        assert!(
            mse_rls < mse_lms,
            "RLS MSE ({mse_rls:.6}) should be lower than LMS MSE ({mse_lms:.6})"
        );
    }

    #[test]
    fn rls_convergence_to_steady_state() {
        let true_coeffs: [f64; 4] = [0.5, 0.3, 0.1, 0.05];
        let mut rls = RlsFilter::<f64, 4>::new(1.0, 1000.0).expect("valid");

        let mut lcg = Lcg::new(77);
        let mut x_hist = [0.0_f64; 4];
        let n_iter = 1000usize;
        let mut mse = 0.0_f64;

        for iter in 0..n_iter {
            x_hist[3] = x_hist[2];
            x_hist[2] = x_hist[1];
            x_hist[1] = x_hist[0];
            x_hist[0] = lcg.next_f64();
            let d = apply_fir(&x_hist, &true_coeffs);
            let y = rls.update(&x_hist, d).expect("ok");
            if iter >= n_iter - 200 {
                mse += (d - y).powi(2);
            }
        }
        mse /= 200.0;
        assert!(mse < 1e-10, "RLS did not converge to near-zero MSE: {mse}");
    }

    #[test]
    fn rls_tracking_time_varying_system() {
        // System switches coefficients at midpoint; lower lambda should re-converge faster.
        let coeffs_phase1: [f64; 4] = [0.5, 0.3, 0.1, 0.05];
        let coeffs_phase2: [f64; 4] = [-0.4, 0.2, 0.6, -0.1];
        let mut rls_fast = RlsFilter::<f64, 4>::new(0.95, 100.0).expect("valid");
        let mut rls_slow = RlsFilter::<f64, 4>::new(1.00, 100.0).expect("valid");

        let mut lcg = Lcg::new(13);
        let mut x_hist = [0.0_f64; 4];
        let n_phase = 500usize;
        let n_track = 200usize; // measurement window after switch

        // Phase 1: both adapt to coeffs_phase1
        for _ in 0..n_phase {
            x_hist[3] = x_hist[2];
            x_hist[2] = x_hist[1];
            x_hist[1] = x_hist[0];
            x_hist[0] = lcg.next_f64();
            let d = apply_fir(&x_hist, &coeffs_phase1);
            let _ = rls_fast.update(&x_hist, d).expect("ok");
            let _ = rls_slow.update(&x_hist, d).expect("ok");
        }

        // Phase 2: system switches to coeffs_phase2
        let mut mse_fast = 0.0_f64;
        let mut mse_slow = 0.0_f64;
        for _ in 0..n_track {
            x_hist[3] = x_hist[2];
            x_hist[2] = x_hist[1];
            x_hist[1] = x_hist[0];
            x_hist[0] = lcg.next_f64();
            let d = apply_fir(&x_hist, &coeffs_phase2);
            let y_fast = rls_fast.update(&x_hist, d).expect("ok");
            let y_slow = rls_slow.update(&x_hist, d).expect("ok");
            mse_fast += (d - y_fast).powi(2);
            mse_slow += (d - y_slow).powi(2);
        }
        mse_fast /= n_track as f64;
        mse_slow /= n_track as f64;

        assert!(
            mse_fast < mse_slow,
            "λ=0.95 should track better than λ=1.0 after switch: fast={mse_fast:.6} slow={mse_slow:.6}"
        );
    }

    #[test]
    fn rls_reset_clears_state() {
        let true_coeffs: [f64; 4] = [0.5, 0.3, 0.1, 0.05];
        let mut rls = RlsFilter::<f64, 4>::new(1.0, 1000.0).expect("valid");

        let mut lcg = Lcg::new(99);
        let mut x_hist = [0.0_f64; 4];
        // Train for 500 iterations
        for _ in 0..500 {
            x_hist[3] = x_hist[2];
            x_hist[2] = x_hist[1];
            x_hist[1] = x_hist[0];
            x_hist[0] = lcg.next_f64();
            let d = apply_fir(&x_hist, &true_coeffs);
            let _ = rls.update(&x_hist, d).expect("ok");
        }

        rls.reset(1000.0);

        // Weights should be zero after reset
        for &w in rls.weights().iter() {
            assert_eq!(w, 0.0, "weight not zero after reset: {w}");
        }
    }

    #[test]
    fn rls_invalid_lambda() {
        assert!(
            RlsFilter::<f64, 4>::new(0.0, 100.0).is_err(),
            "lambda=0 should fail"
        );
        assert!(
            RlsFilter::<f64, 4>::new(1.1, 100.0).is_err(),
            "lambda>1 should fail"
        );
        assert!(
            RlsFilter::<f64, 4>::new(-0.5, 100.0).is_err(),
            "negative lambda should fail"
        );
    }

    #[test]
    fn rls_invalid_delta() {
        assert!(
            RlsFilter::<f64, 4>::new(0.99, 0.0).is_err(),
            "delta=0 should fail"
        );
        assert!(
            RlsFilter::<f64, 4>::new(0.99, -1.0).is_err(),
            "negative delta should fail"
        );
    }

    #[test]
    fn rls_weight_accuracy_after_convergence() {
        let true_coeffs: [f64; 4] = [0.5, 0.3, 0.1, 0.05];
        let mut rls = RlsFilter::<f64, 4>::new(1.0, 1000.0).expect("valid");

        let mut lcg = Lcg::new(314);
        let mut x_hist = [0.0_f64; 4];

        for _ in 0..2000 {
            x_hist[3] = x_hist[2];
            x_hist[2] = x_hist[1];
            x_hist[1] = x_hist[0];
            x_hist[0] = lcg.next_f64();
            let d = apply_fir(&x_hist, &true_coeffs);
            let _ = rls.update(&x_hist, d).expect("ok");
        }

        let w = rls.weights();
        for (i, (&wi, &ti)) in w.iter().zip(true_coeffs.iter()).enumerate() {
            assert!(
                (wi - ti).abs() < 1e-6,
                "weight[{i}] = {wi:.8}, expected {ti:.8}"
            );
        }
    }
}
