//! Statistical Hypothesis Testing for Fault Detection & Isolation
//!
//! Provides two classical statistical tests applied to FDI residuals:
//!
//! ## χ² (Chi-Squared) Test
//! Given a residual vector `r` and inverse covariance `Σ⁻¹`, computes the
//! quadratic form `T = rᵀ·Σ⁻¹·r`.  Under a Gaussian zero-mean normal
//! distribution, `T ~ χ²(M)`.  A threshold exceedance signals a fault.
//!
//! ## SPRT (Sequential Probability Ratio Test)
//! Maintains a running log-likelihood ratio `Λ` comparing the fault
//! hypothesis `H₁: r ~ N(μ_fault, Σ)` against the null `H₀: r ~ N(0, Σ)`.
//! Decisions are made when `Λ` crosses upper (`B`) or lower (`A`) boundaries.

#![cfg_attr(not(feature = "std"), no_std)]

use crate::core::scalar::ControlScalar;
use crate::fdi::parity_space::{FaultStatus, FdiError};

// ---------------------------------------------------------------------------
// χ² Test
// ---------------------------------------------------------------------------

/// Chi-squared hypothesis test for Gaussian residuals.
///
/// # Type Parameters
/// * `S` — scalar type implementing [`ControlScalar`].
/// * `M` — residual (output) dimension.
///
/// The test statistic is `T = rᵀ·Σ⁻¹·r`.  The user supplies `Σ⁻¹` directly,
/// which avoids matrix inversion inside the control loop.
#[derive(Debug, Clone)]
pub struct ChiSquareTest<S: ControlScalar, const M: usize> {
    /// Inverse covariance matrix Σ⁻¹ (M×M), supplied by the caller.
    sigma_inv: [[S; M]; M],
    /// Detection threshold on the test statistic T.
    threshold: S,
    /// Total number of calls to [`test`].
    n_samples: usize,
    /// Number of calls that resulted in [`FaultStatus::FaultDetected`].
    n_alarms: usize,
}

impl<S: ControlScalar, const M: usize> ChiSquareTest<S, M> {
    /// Create a new [`ChiSquareTest`].
    ///
    /// # Arguments
    /// * `sigma_inv` — inverse covariance matrix Σ⁻¹ (must be symmetric PD).
    /// * `threshold` — test threshold; values above this trigger an alarm.
    pub fn new(sigma_inv: [[S; M]; M], threshold: S) -> Self {
        Self {
            sigma_inv,
            threshold,
            n_samples: 0,
            n_alarms: 0,
        }
    }

    /// Compute the chi-squared test statistic `T = rᵀ·Σ⁻¹·r`.
    ///
    /// This is a pure function — it does **not** update internal counters.
    pub fn statistic(residual: &[S; M], sigma_inv: &[[S; M]; M]) -> S {
        let mut t = S::ZERO;
        for i in 0..M {
            for j in 0..M {
                t += residual[i] * sigma_inv[i][j] * residual[j];
            }
        }
        t
    }

    /// Evaluate the chi-squared test and update alarm statistics.
    ///
    /// Returns [`FaultStatus::FaultDetected`] when `T > threshold`.
    pub fn test(&mut self, residual: &[S; M]) -> FaultStatus {
        let t = Self::statistic(residual, &self.sigma_inv);
        self.n_samples += 1;
        if t > self.threshold {
            self.n_alarms += 1;
            FaultStatus::FaultDetected
        } else {
            FaultStatus::Normal
        }
    }

    /// Fraction of calls to [`test`] that resulted in an alarm.
    ///
    /// Returns `S::ZERO` when no samples have been processed.
    pub fn alarm_rate(&self) -> S {
        if self.n_samples == 0 {
            S::ZERO
        } else {
            S::from_f64(self.n_alarms as f64 / self.n_samples as f64)
        }
    }

    /// Reset alarm statistics (does **not** change `sigma_inv` or `threshold`).
    pub fn reset(&mut self) {
        self.n_samples = 0;
        self.n_alarms = 0;
    }

    /// Total number of test evaluations since construction or last reset.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Total number of fault alarms since construction or last reset.
    pub fn n_alarms(&self) -> usize {
        self.n_alarms
    }
}

// ---------------------------------------------------------------------------
// SPRT Decision
// ---------------------------------------------------------------------------

/// Decision returned by a single SPRT update step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprtDecision {
    /// Log-likelihood ratio exceeded the upper threshold B → fault declared.
    Fault,
    /// Log-likelihood ratio fell below the lower threshold A → normal confirmed.
    Normal,
    /// Log-likelihood ratio is between A and B — continue collecting data.
    Indeterminate,
}

// ---------------------------------------------------------------------------
// SPRT
// ---------------------------------------------------------------------------

/// Sequential Probability Ratio Test for fault detection.
///
/// Maintains a running log-likelihood ratio `Λ` comparing:
/// - `H₁`: `r ~ N(μ_fault, Σ)` (fault present)
/// - `H₀`: `r ~ N(0, Σ)`       (no fault)
///
/// The diagonal of `Σ` is used for computational efficiency, making this an
/// independent-channel approximation suitable for embedded real-time code.
///
/// # Type Parameters
/// * `S` — scalar type implementing [`ControlScalar`].
/// * `M` — residual dimension.
#[derive(Debug, Clone)]
pub struct Sprt<S: ControlScalar, const M: usize> {
    /// Residual covariance matrix Σ (only diagonal elements are used).
    sigma: [[S; M]; M],
    /// Expected residual under the fault hypothesis H₁.
    mu_fault: [S; M],
    /// Running log-likelihood ratio Λ.
    log_lambda: S,
    /// Upper decision boundary B: Λ > B → declare fault.
    threshold_upper: S,
    /// Lower decision boundary A: Λ < A → declare normal.
    threshold_lower: S,
}

impl<S: ControlScalar, const M: usize> Sprt<S, M> {
    /// Construct a new [`Sprt`].
    ///
    /// # Arguments
    /// * `sigma`           — covariance matrix Σ (diagonal used for Gaussian log-likelihood).
    /// * `mu_fault`        — expected residual mean under the fault hypothesis.
    /// * `threshold_upper` — upper boundary B (should be positive, e.g. `ln(β/(1−α))`).
    /// * `threshold_lower` — lower boundary A (should be negative, e.g. `ln(α/(1−β))`).
    ///
    /// # Errors
    /// Returns [`FdiError::InvalidParameter`] if `threshold_upper ≤ threshold_lower`.
    pub fn new(
        sigma: [[S; M]; M],
        mu_fault: [S; M],
        threshold_upper: S,
        threshold_lower: S,
    ) -> Result<Self, FdiError> {
        if threshold_upper <= threshold_lower {
            return Err(FdiError::InvalidParameter);
        }
        Ok(Self {
            sigma,
            mu_fault,
            log_lambda: S::ZERO,
            threshold_upper,
            threshold_lower,
        })
    }

    /// Update the SPRT with a new residual observation.
    ///
    /// Uses diagonal-only Gaussian log-likelihood:
    ///
    /// ```text
    /// log p₀ = −½ Σᵢ r[i]² / σ[i][i]
    /// log p₁ = −½ Σᵢ (r[i] − μ[i])² / σ[i][i]
    /// Λ ← Λ + log(p₁/p₀) = Λ + (log p₁ − log p₀)
    /// ```
    ///
    /// Returns a [`SprtDecision`] based on the updated `Λ`.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, residual: &[S; M]) -> SprtDecision {
        // Compute incremental log-likelihood ratio using diagonal covariance
        let mut delta = S::ZERO;
        for i in 0..M {
            let sigma_ii = self.sigma[i][i];
            // Guard against zero or negative diagonal entries
            if sigma_ii <= S::ZERO {
                continue;
            }
            let ri = residual[i];
            let mu_i = self.mu_fault[i];
            // log p0 contribution: -0.5 * r² / σ
            let log_p0_i = S::from_f64(-0.5) * ri * ri / sigma_ii;
            // log p1 contribution: -0.5 * (r - μ)² / σ
            let diff = ri - mu_i;
            let log_p1_i = S::from_f64(-0.5) * diff * diff / sigma_ii;
            delta += log_p1_i - log_p0_i;
        }
        self.log_lambda += delta;

        if self.log_lambda > self.threshold_upper {
            SprtDecision::Fault
        } else if self.log_lambda < self.threshold_lower {
            SprtDecision::Normal
        } else {
            SprtDecision::Indeterminate
        }
    }

    /// Reset the log-likelihood ratio to zero (restart hypothesis test).
    pub fn reset(&mut self) {
        self.log_lambda = S::ZERO;
    }

    /// Current value of the log-likelihood ratio Λ.
    pub fn log_ratio(&self) -> S {
        self.log_lambda
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------ χ² tests ------

    #[test]
    fn chi2_zero_residual_normal() {
        // With sigma_inv = I, T = ||r||² = 0 → Normal
        let sigma_inv = [[1.0_f64, 0.0], [0.0, 1.0]];
        let mut test = ChiSquareTest::new(sigma_inv, 1.0);
        let status = test.test(&[0.0, 0.0]);
        assert_eq!(status, FaultStatus::Normal);
        assert_eq!(test.n_alarms(), 0);
    }

    #[test]
    fn chi2_large_residual_detected() {
        // r=[10,10], sigma_inv=I → T=200 >> threshold=1
        let sigma_inv = [[1.0_f64, 0.0], [0.0, 1.0]];
        let mut test = ChiSquareTest::new(sigma_inv, 1.0);
        let status = test.test(&[10.0, 10.0]);
        assert_eq!(status, FaultStatus::FaultDetected);
    }

    #[test]
    fn chi2_alarm_rate_calculation() {
        // 10 calls: 5 with large residual (fault), 5 with zero (normal) → rate = 0.5
        let sigma_inv = [[1.0_f64]];
        let mut test = ChiSquareTest::new(sigma_inv, 1.0);
        for _ in 0..5 {
            test.test(&[5.0]); // T=25 > 1 → alarm
        }
        for _ in 0..5 {
            test.test(&[0.0]); // T=0 → normal
        }
        let rate = test.alarm_rate();
        assert!((rate - 0.5).abs() < 1e-9, "expected 0.5, got {rate}");
    }

    #[test]
    fn chi2_reset_clears_counts() {
        let sigma_inv = [[1.0_f64]];
        let mut test = ChiSquareTest::new(sigma_inv, 1.0);
        test.test(&[5.0]);
        test.test(&[5.0]);
        assert_eq!(test.n_samples(), 2);
        assert_eq!(test.n_alarms(), 2);
        test.reset();
        assert_eq!(test.n_samples(), 0);
        assert_eq!(test.n_alarms(), 0);
        assert_eq!(test.alarm_rate(), 0.0);
    }

    #[test]
    fn chi2_statistic_helper() {
        // T = r^T * I * r = ||r||²
        let sigma_inv = [[1.0_f64, 0.0], [0.0, 1.0]];
        let r = [3.0_f64, 4.0];
        let t = ChiSquareTest::statistic(&r, &sigma_inv);
        assert!((t - 25.0).abs() < 1e-9, "expected T=25, got {t}");
    }

    #[test]
    fn chi2_alarm_rate_zero_samples() {
        let sigma_inv = [[1.0_f64]];
        let test: ChiSquareTest<f64, 1> = ChiSquareTest::new(sigma_inv, 1.0);
        assert_eq!(test.alarm_rate(), 0.0);
    }

    // ------ SPRT tests ------

    #[test]
    fn sprt_zero_residual_stays_indeterminate() {
        // r=0, mu_fault=[1]: log p1 < log p0 each step → Λ decreases or stagnates
        // With r=0 and mu=[1], delta = -0.5*(0-1)²/1 - (-0.5*0²/1) = -0.5
        // So log_lambda decreases → eventually Normal
        let sigma = [[1.0_f64]];
        let mu_fault = [1.0_f64];
        let mut sprt = Sprt::new(sigma, mu_fault, 5.0_f64, -5.0_f64).expect("ok");
        let mut saw_normal = false;
        for _ in 0..20 {
            let d = sprt.update(&[0.0]);
            if d == SprtDecision::Normal {
                saw_normal = true;
                break;
            }
        }
        assert!(
            saw_normal,
            "SPRT should declare Normal for zero residual against fault hypothesis"
        );
    }

    #[test]
    fn sprt_fault_residual_accumulates_to_fault() {
        // r = mu_fault → Λ increases monotonically → eventually Fault
        let sigma = [[1.0_f64]];
        let mu_fault = [3.0_f64];
        // With r=mu: delta = -0.5*(mu-mu)²/σ - (-0.5*mu²/σ) = 0.5*mu²/σ = 4.5 per step
        let mut sprt = Sprt::new(sigma, mu_fault, 10.0_f64, -10.0_f64).expect("ok");
        let mut saw_fault = false;
        for _ in 0..10 {
            let d = sprt.update(&[3.0]);
            if d == SprtDecision::Fault {
                saw_fault = true;
                break;
            }
        }
        assert!(
            saw_fault,
            "SPRT should declare Fault when residual matches fault mean"
        );
    }

    #[test]
    fn sprt_reset_clears_log_lambda() {
        let sigma = [[1.0_f64]];
        let mu_fault = [3.0_f64];
        let mut sprt = Sprt::new(sigma, mu_fault, 100.0_f64, -100.0_f64).expect("ok");
        sprt.update(&[3.0]);
        sprt.update(&[3.0]);
        assert!(sprt.log_ratio() > 0.0, "log_lambda should have grown");
        sprt.reset();
        assert_eq!(sprt.log_ratio(), 0.0);
    }

    #[test]
    fn sprt_threshold_order_validation() {
        let sigma = [[1.0_f64]];
        let mu_fault = [1.0_f64];
        // upper == lower → invalid
        assert!(
            Sprt::new(sigma, mu_fault, 5.0_f64, 5.0_f64).is_err(),
            "equal thresholds should be invalid"
        );
        // upper < lower → invalid
        assert!(
            Sprt::new(sigma, mu_fault, -1.0_f64, 5.0_f64).is_err(),
            "upper < lower should be invalid"
        );
        // valid
        assert!(Sprt::new(sigma, mu_fault, 5.0_f64, -5.0_f64).is_ok());
    }

    #[test]
    fn sprt_log_ratio_monotone_under_fault() {
        // When r consistently equals mu_fault, log_lambda should increase each step.
        let sigma = [[2.0_f64]];
        let mu_fault = [2.0_f64];
        let mut sprt = Sprt::new(sigma, mu_fault, 1000.0_f64, -1000.0_f64).expect("ok");
        let mut prev = sprt.log_ratio();
        for _ in 0..20 {
            sprt.update(&[2.0]);
            let current = sprt.log_ratio();
            assert!(
                current > prev,
                "log_ratio should increase under fault: prev={prev}, current={current}"
            );
            prev = current;
        }
    }

    #[test]
    fn sprt_indeterminate_region() {
        // With very wide boundaries, all updates should be Indeterminate initially.
        let sigma = [[1.0_f64]];
        let mu_fault = [0.1_f64]; // tiny fault → very slow accumulation
        let mut sprt = Sprt::new(sigma, mu_fault, 1e9_f64, -1e9_f64).expect("ok");
        // After a few steps, still Indeterminate
        for _ in 0..10 {
            let d = sprt.update(&[0.05]);
            assert_eq!(d, SprtDecision::Indeterminate);
        }
    }
}
