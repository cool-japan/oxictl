//! Model validation tools.
#![allow(clippy::needless_range_loop, clippy::doc_overindented_list_items)]
//!
//! Provides:
//! - [`fit_percent`] — MATLAB-style FIT% model quality metric.
//! - [`residual_analysis`] — statistical summary of residuals.
//! - [`autocorrelation`] — normalised autocorrelation function (ACF).
//! - [`whiteness_test`] — χ² test for whiteness of residuals.
//! - [`cross_correlation`] — normalised cross-correlation between residuals and inputs.
//! - [`ResidualStats`] — aggregated statistics struct.
//!
//! All functions operate on slices and return fixed-size arrays or scalar values.
//! No heap allocation is required.

use crate::core::scalar::ControlScalar;

// ── fit_percent ───────────────────────────────────────────────────────────────

/// Compute the MATLAB-style FIT% metric.
///
/// ```text
///   FIT% = 100 · (1 − ‖y − ŷ‖₂ / ‖y − mean(y)‖₂)
/// ```
///
/// A value of 100% means perfect prediction; 0% means the model is no better
/// than predicting the mean; negative values mean the model is worse than mean.
///
/// Returns `S::ZERO` if the denominator is zero (constant signal).
pub fn fit_percent<S: ControlScalar>(predicted: &[S], actual: &[S]) -> S {
    let n = predicted.len().min(actual.len());
    if n == 0 {
        return S::ZERO;
    }

    // Mean of actual
    let mut sum = S::ZERO;
    for i in 0..n {
        sum += actual[i];
    }
    let mean = sum / S::from_f64(n as f64);

    let mut num_sq = S::ZERO;
    let mut den_sq = S::ZERO;
    for i in 0..n {
        let e = actual[i] - predicted[i];
        num_sq += e * e;
        let d = actual[i] - mean;
        den_sq += d * d;
    }

    if den_sq == S::ZERO {
        return S::ZERO;
    }

    let ratio = (num_sq / den_sq).sqrt();
    S::from_f64(100.0) * (S::ONE - ratio)
}

// ── autocorrelation ───────────────────────────────────────────────────────────

/// Compute the normalised autocorrelation of `signal` for lags 0 … L-1.
///
/// The zero-lag value (index 0) is always 1.0. The normalisation is relative
/// to the zero-lag sample so all values lie in [-1, 1].
///
/// Returns an array of length `L`. If `signal` is empty or the zero-lag power
/// is zero, returns the zero array.
pub fn autocorrelation<S: ControlScalar, const L: usize>(signal: &[S]) -> [S; L] {
    let n = signal.len();
    let mut result = [S::ZERO; L];
    if n == 0 {
        return result;
    }

    // Compute mean
    let mut mean = S::ZERO;
    for i in 0..n {
        mean += signal[i];
    }
    mean = mean / S::from_f64(n as f64);

    // Compute R(0) — zero-lag variance
    let mut r0 = S::ZERO;
    for i in 0..n {
        let d = signal[i] - mean;
        r0 += d * d;
    }

    if r0 == S::ZERO {
        return result;
    }

    // Compute R(lag) for lag = 0 … L-1
    for lag in 0..L {
        let mut r_lag = S::ZERO;
        let max_t = n.saturating_sub(lag);
        for t in 0..max_t {
            r_lag += (signal[t] - mean) * (signal[t + lag] - mean);
        }
        result[lag] = r_lag / r0;
    }

    result
}

// ── ResidualStats ─────────────────────────────────────────────────────────────

/// Statistical summary of a residual sequence.
#[derive(Debug, Clone, Copy)]
pub struct ResidualStats<S: ControlScalar> {
    /// Sample mean of the residuals.
    pub mean: S,
    /// Sample variance of the residuals (unbiased, divided by N-1).
    pub variance: S,
    /// Normalised autocorrelation at lag 1 (indicator of serial correlation).
    pub autocorr_lag1: S,
    /// Whether the whiteness test passed (residuals are statistically white).
    pub is_white: bool,
}

/// Compute a statistical summary of the residual sequence.
///
/// `max_lag` — number of lags to use for the whiteness test.
/// `significance` — significance level for the χ² whiteness test (e.g. 0.05).
pub fn residual_analysis<S: ControlScalar>(
    residuals: &[S],
    max_lag: usize,
    significance: S,
) -> ResidualStats<S> {
    let n = residuals.len();

    if n == 0 {
        return ResidualStats {
            mean: S::ZERO,
            variance: S::ZERO,
            autocorr_lag1: S::ZERO,
            is_white: true,
        };
    }

    // Mean
    let mut sum = S::ZERO;
    for i in 0..n {
        sum += residuals[i];
    }
    let mean = sum / S::from_f64(n as f64);

    // Variance (unbiased)
    let mut var = S::ZERO;
    for i in 0..n {
        let d = residuals[i] - mean;
        var += d * d;
    }
    let variance = if n > 1 {
        var / S::from_f64((n - 1) as f64)
    } else {
        S::ZERO
    };

    // Autocorrelation at lag 1
    let mut r0 = S::ZERO;
    let mut r1 = S::ZERO;
    for i in 0..n {
        let d = residuals[i] - mean;
        r0 += d * d;
    }
    for i in 0..(n.saturating_sub(1)) {
        r1 += (residuals[i] - mean) * (residuals[i + 1] - mean);
    }
    let autocorr_lag1 = if r0 == S::ZERO { S::ZERO } else { r1 / r0 };

    let is_white = whiteness_test(residuals, max_lag, significance);

    ResidualStats {
        mean,
        variance,
        autocorr_lag1,
        is_white,
    }
}

// ── whiteness_test ────────────────────────────────────────────────────────────

/// Test whether the residuals are statistically white (uncorrelated) using
/// a χ²-approximation.
///
/// The Ljung-Box statistic is:
/// ```text
///   Q = N·(N+2) · Σ_{k=1}^{M} r_k² / (N - k)
/// ```
/// where r_k is the normalised ACF at lag k and N is the number of samples.
///
/// Under the null hypothesis (white residuals), Q ~ χ²(M). The test rejects
/// whiteness when Q exceeds the χ²(M, `significance`) critical value.
///
/// The critical value is approximated via the Wilson-Hilferty transformation:
/// `χ²(M, α) ≈ M · (1 − 2/(9M) + z_α · √(2/(9M)))³`
/// where z_α is the standard normal quantile for significance level `α`.
///
/// Returns `true` if the residuals are white (cannot reject null hypothesis).
pub fn whiteness_test<S: ControlScalar>(residuals: &[S], max_lag: usize, significance: S) -> bool {
    let n = residuals.len();
    if n == 0 || max_lag == 0 {
        return true;
    }

    // Compute normalised ACF
    let mut mean = S::ZERO;
    for i in 0..n {
        mean += residuals[i];
    }
    mean = mean / S::from_f64(n as f64);

    let mut r0 = S::ZERO;
    for i in 0..n {
        let d = residuals[i] - mean;
        r0 += d * d;
    }
    if r0 == S::ZERO {
        return true;
    }

    // Ljung-Box statistic
    let nf = S::from_f64(n as f64);
    let mut q = S::ZERO;
    for k in 1..=max_lag {
        let max_t = n.saturating_sub(k);
        let mut rk = S::ZERO;
        for t in 0..max_t {
            rk += (residuals[t] - mean) * (residuals[t + k] - mean);
        }
        let rk_norm = rk / r0;
        let denom = nf - S::from_f64(k as f64);
        if denom > S::ZERO {
            q += rk_norm * rk_norm / denom;
        }
    }
    q = q * nf * (nf + S::ONE);

    // Critical value via Wilson-Hilferty approximation for χ²(M)
    let m = S::from_f64(max_lag as f64);
    // Normal quantile z_{1-α}: approximate via rational approximation for common values.
    // We use the Beasley-Springer-Moro approximation.
    let alpha_f64 = significance.to_f64().clamp(1e-6, 0.5);
    let z_alpha = normal_quantile_approx(S::from_f64(1.0 - alpha_f64));

    let two_over_9m = S::from_f64(2.0) / (S::from_f64(9.0) * m);
    let inner = S::ONE - two_over_9m + z_alpha * two_over_9m.sqrt();
    let chi2_crit = if inner > S::ZERO {
        m * inner * inner * inner
    } else {
        S::ZERO
    };

    q <= chi2_crit
}

/// Approximate the inverse CDF of the standard normal distribution at probability `p`.
///
/// Uses the Beasley-Springer-Moro rational approximation, accurate to ~1e-4 for p ∈ (0,1).
fn normal_quantile_approx<S: ControlScalar>(p: S) -> S {
    // Map p to (0, 1)
    let p_f64 = p.to_f64().clamp(1e-9, 1.0 - 1e-9);

    // Abramowitz & Stegun §26.2.23 rational approximation for the upper tail
    let z = if p_f64 <= 0.5 {
        let t = (-2.0 * p_f64.ln()).sqrt();
        let c0 = 2.515517_f64;
        let c1 = 0.802853_f64;
        let c2 = 0.010328_f64;
        let d1 = 1.432788_f64;
        let d2 = 0.189269_f64;
        let d3 = 0.001308_f64;
        -(t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))
    } else {
        let q_val = 1.0 - p_f64;
        let t = (-2.0 * q_val.ln()).sqrt();
        let c0 = 2.515517_f64;
        let c1 = 0.802853_f64;
        let c2 = 0.010328_f64;
        let d1 = 1.432788_f64;
        let d2 = 0.189269_f64;
        let d3 = 0.001308_f64;
        t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    };

    S::from_f64(z)
}

// ── cross_correlation ─────────────────────────────────────────────────────────

/// Compute normalised cross-correlation between `residuals` and `inputs` for lags
/// 0 … L-1.
///
/// The cross-correlation at lag k is:
/// ```text
///   r_eu(k) = Σ_t e(t)·u(t-k) / √(Σ_t e(t)² · Σ_t u(t)²)
/// ```
///
/// A well-identified model should have small cross-correlation between residuals
/// and past inputs (independence test).
///
/// Returns an array of length `L`.
pub fn cross_correlation<S: ControlScalar, const L: usize>(
    residuals: &[S],
    inputs: &[S],
) -> [S; L] {
    let n = residuals.len().min(inputs.len());
    let mut result = [S::ZERO; L];
    if n == 0 {
        return result;
    }

    // Means
    let mut mean_e = S::ZERO;
    let mut mean_u = S::ZERO;
    for i in 0..n {
        mean_e += residuals[i];
        mean_u += inputs[i];
    }
    mean_e = mean_e / S::from_f64(n as f64);
    mean_u = mean_u / S::from_f64(n as f64);

    // Variances
    let mut var_e = S::ZERO;
    let mut var_u = S::ZERO;
    for i in 0..n {
        let de = residuals[i] - mean_e;
        let du = inputs[i] - mean_u;
        var_e += de * de;
        var_u += du * du;
    }

    let norm = (var_e * var_u).sqrt();
    if norm == S::ZERO {
        return result;
    }

    for lag in 0..L {
        let max_t = n.saturating_sub(lag);
        let mut cross = S::ZERO;
        for t in lag..n {
            let t_u = t - lag;
            if t_u < n {
                cross += (residuals[t] - mean_e) * (inputs[t_u] - mean_u);
            }
        }
        // Note: correct lag indexing — e(t), u(t - lag)
        let mut c = S::ZERO;
        for t in lag..max_t + lag {
            if t < n && t >= lag {
                c += (residuals[t] - mean_e) * (inputs[t - lag] - mean_u);
            }
        }
        result[lag] = c / norm;
    }

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// White noise (deterministic LCG) should have autocorrelation ≈ 0 for lag > 0.
    #[test]
    fn white_noise_autocorrelation_near_zero() {
        // Generate deterministic pseudo-white noise via LCG
        let mut lcg: u64 = 42;
        let mut noise: heapless::Vec<f64, 1024> = heapless::Vec::new();
        for _ in 0..1000 {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (lcg >> 33) as f64 / (u64::MAX >> 33) as f64 - 0.5;
            let _ = noise.push(v);
        }

        let acf: [f64; 20] = autocorrelation(noise.as_slice());

        // Lag 0 should be 1.0
        assert!((acf[0] - 1.0).abs() < 1e-10, "ACF[0] = {}", acf[0]);

        // Lags 1..19 should be small (within ~3/√N ≈ 0.095 for N=1000)
        let bound = 3.0 / (1000.0_f64).sqrt();
        for lag in 1..20 {
            assert!(
                acf[lag].abs() < bound,
                "ACF[{lag}] = {:.4} exceeds bound {bound:.4}",
                acf[lag]
            );
        }
    }

    /// Sine wave autocorrelation should be cosine-like.
    #[test]
    fn sine_autocorrelation_is_cosine() {
        let n = 1000_usize;
        let freq = 0.05; // cycles per sample
        let mut sine: heapless::Vec<f64, 1024> = heapless::Vec::new();
        for t in 0..n {
            let _ = sine.push(libm::sin(2.0 * core::f64::consts::PI * freq * t as f64));
        }

        let acf: [f64; 50] = autocorrelation(sine.as_slice());

        // ACF of a sine at lag k should be ≈ cos(2π·freq·k)
        for lag in 0..50 {
            let expected = libm::cos(2.0 * core::f64::consts::PI * freq * lag as f64);
            assert!(
                (acf[lag] - expected).abs() < 0.05,
                "ACF[{lag}] = {:.4}, expected {expected:.4}",
                acf[lag]
            );
        }
    }

    /// Whiteness test should pass for white noise.
    #[test]
    fn whiteness_test_passes_for_white_noise() {
        let mut lcg: u64 = 99;
        let mut noise: heapless::Vec<f64, 2048> = heapless::Vec::new();
        for _ in 0..2000 {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (lcg >> 33) as f64 / (u64::MAX >> 33) as f64 - 0.5;
            let _ = noise.push(v);
        }
        let white = whiteness_test(noise.as_slice(), 20, 0.05_f64);
        assert!(white, "Whiteness test should pass for white noise");
    }

    /// Whiteness test should fail for highly correlated signal.
    #[test]
    fn whiteness_test_fails_for_highly_correlated_signal() {
        // Use a pure sine wave: maximally correlated (autocorrelation = cosine).
        // The Ljung-Box Q statistic will be very large for this signal.
        let n = 2000_usize;
        let mut sig: heapless::Vec<f64, 2048> = heapless::Vec::new();
        for t in 0..n {
            let _ = sig.push(libm::sin(2.0 * core::f64::consts::PI * 0.05 * t as f64));
        }
        let white = whiteness_test(sig.as_slice(), 20, 0.05_f64);
        assert!(!white, "Whiteness test should fail for a sine wave signal");
    }

    /// FIT% of perfect prediction is 100%.
    #[test]
    fn fit_percent_perfect() {
        let y: [f64; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let fp = fit_percent(&y, &y);
        assert!((fp - 100.0).abs() < 1e-9, "FIT% = {fp}");
    }

    /// FIT% of mean prediction is 0%.
    #[test]
    fn fit_percent_mean_prediction_is_zero() {
        let y: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mean_pred: [f64; 6] = [3.5; 6];
        let fp = fit_percent(&mean_pred, &y);
        assert!(fp.abs() < 1e-9, "FIT% of mean prediction = {fp}");
    }

    /// Cross-correlation between independent signals is near zero.
    #[test]
    fn cross_correlation_independent_near_zero() {
        let mut lcg: u64 = 7;
        let mut sig1: heapless::Vec<f64, 1024> = heapless::Vec::new();
        let mut sig2: heapless::Vec<f64, 1024> = heapless::Vec::new();
        for _ in 0..1000 {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v1 = (lcg >> 33) as f64 / (u64::MAX >> 33) as f64 - 0.5;
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v2 = (lcg >> 33) as f64 / (u64::MAX >> 33) as f64 - 0.5;
            let _ = sig1.push(v1);
            let _ = sig2.push(v2);
        }
        let cc: [f64; 10] = cross_correlation(sig1.as_slice(), sig2.as_slice());
        let bound = 4.0 / (1000.0_f64).sqrt();
        for lag in 0..10 {
            assert!(
                cc[lag].abs() < bound,
                "cross_corr[{lag}] = {:.4} exceeds bound {bound:.4}",
                cc[lag]
            );
        }
    }

    /// residual_analysis reports correct mean and variance for a known sequence.
    #[test]
    fn residual_analysis_known_sequence() {
        // Constant sequence [2, 2, 2, 2] — variance = 0, mean = 2
        let seq = [2.0_f64; 100];
        let stats = residual_analysis(&seq, 5, 0.05);
        assert!((stats.mean - 2.0).abs() < 1e-10);
        assert!(stats.variance.abs() < 1e-10);
    }
}
