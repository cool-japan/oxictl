//! Chebyshev Type I and Type II IIR filter design via bilinear transform.
//!
//! # Chebyshev Type I
//! Equiripple in the passband, monotone in the stopband.
//! Prototype poles in the s-plane are on an ellipse derived from ε and N:
//!   ε = sqrt(10^(Rp/10) - 1)       (ripple parameter from passband ripple in dB)
//!   a = (1/N)·arcsinh(1/ε)
//!   Analog prototype poles: sk = -sinh(a)·sin(θk) + j·cosh(a)·cos(θk)
//!   where θk = π(2k-1)/(2N),  k = 1..N
//!
//! # Chebyshev Type II
//! Monotone in the passband, equiripple in the stopband.
//! Obtained by inverting the Type I prototype and applying a frequency transformation.

use crate::core::filters::FilterError;
use crate::core::scalar::ControlScalar;

// ─────────────────────────────────────────────────────────────
//  Internal biquad (Direct Form II transposed)
//  (re-defined locally to avoid cross-module coupling issues)
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub(super) struct ChebBiquad<S: ControlScalar> {
    b0: S,
    b1: S,
    b2: S,
    a1: S,
    a2: S,
    w1: S,
    w2: S,
}

impl<S: ControlScalar> ChebBiquad<S> {
    pub fn new(b0: S, b1: S, b2: S, a1: S, a2: S) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            w1: S::ZERO,
            w2: S::ZERO,
        }
    }

    pub fn identity() -> Self {
        Self::new(S::ONE, S::ZERO, S::ZERO, S::ZERO, S::ZERO)
    }

    #[inline]
    pub fn process(&mut self, x: S) -> S {
        let y = self.b0 * x + self.w1;
        self.w1 = self.b1 * x - self.a1 * y + self.w2;
        self.w2 = self.b2 * x - self.a2 * y;
        y
    }

    pub fn reset(&mut self) {
        self.w1 = S::ZERO;
        self.w2 = S::ZERO;
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ChebFirstOrder<S: ControlScalar> {
    b0: S,
    b1: S,
    a1: S,
    w: S,
}

impl<S: ControlScalar> ChebFirstOrder<S> {
    pub fn new(b0: S, b1: S, a1: S) -> Self {
        Self {
            b0,
            b1,
            a1,
            w: S::ZERO,
        }
    }

    #[inline]
    pub fn process(&mut self, x: S) -> S {
        let y = self.b0 * x + self.w;
        self.w = self.b1 * x - self.a1 * y;
        y
    }

    pub fn reset(&mut self) {
        self.w = S::ZERO;
    }
}

// ─────────────────────────────────────────────────────────────
//  ChebyshevI<S, N>
// ─────────────────────────────────────────────────────────────

/// Nth-order Chebyshev Type I lowpass filter.
/// Equiripple in the passband, monotone in the stopband.
#[derive(Debug, Clone, Copy)]
pub struct ChebyshevI<S: ControlScalar, const N: usize> {
    biquads: [ChebBiquad<S>; 4],
    first_order: Option<ChebFirstOrder<S>>,
    n_biquads: usize,
    has_first_order: bool,
}

impl<S: ControlScalar, const N: usize> ChebyshevI<S, N> {
    /// Process one input sample.
    #[inline]
    pub fn update(&mut self, x: S) -> S {
        let mut y = x;
        if self.has_first_order {
            y = self.first_order.as_mut().map_or(y, |f| f.process(y));
        }
        for i in 0..self.n_biquads {
            y = self.biquads[i].process(y);
        }
        y
    }

    /// Reset all filter state.
    pub fn reset(&mut self) {
        for bq in self.biquads.iter_mut() {
            bq.reset();
        }
        if let Some(f) = self.first_order.as_mut() {
            f.reset();
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  ChebyshevII<S, N>
// ─────────────────────────────────────────────────────────────

/// Nth-order Chebyshev Type II lowpass filter.
/// Monotone in the passband, equiripple in the stopband.
#[derive(Debug, Clone, Copy)]
pub struct ChebyshevII<S: ControlScalar, const N: usize> {
    biquads: [ChebBiquad<S>; 4],
    first_order: Option<ChebFirstOrder<S>>,
    n_biquads: usize,
    has_first_order: bool,
}

impl<S: ControlScalar, const N: usize> ChebyshevII<S, N> {
    /// Process one input sample.
    #[inline]
    pub fn update(&mut self, x: S) -> S {
        let mut y = x;
        if self.has_first_order {
            y = self.first_order.as_mut().map_or(y, |f| f.process(y));
        }
        for i in 0..self.n_biquads {
            y = self.biquads[i].process(y);
        }
        y
    }

    /// Reset all filter state.
    pub fn reset(&mut self) {
        for bq in self.biquads.iter_mut() {
            bq.reset();
        }
        if let Some(f) = self.first_order.as_mut() {
            f.reset();
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Math helpers: arcsinh, acosh in no_std via libm
// ─────────────────────────────────────────────────────────────

/// arcsinh via identity: arcsinh(x) = ln(x + sqrt(x²+1))
fn arcsinh<S: ControlScalar>(x: S) -> S {
    (x + (x * x + S::ONE).sqrt()).ln()
}

// ─────────────────────────────────────────────────────────────
//  Design function: Chebyshev Type I lowpass
// ─────────────────────────────────────────────────────────────

/// Design an Nth-order Chebyshev Type I lowpass filter.
///
/// # Arguments
/// * `cutoff_hz`   — passband-edge (−Rp dB) frequency in Hz
/// * `ripple_db`   — passband ripple in dB (e.g. 0.5 or 1.0)
/// * `sample_rate_hz` — sample rate in Hz
pub fn design_chebyshev1_lp<S: ControlScalar, const N: usize>(
    cutoff_hz: S,
    ripple_db: S,
    sample_rate_hz: S,
) -> Result<ChebyshevI<S, N>, FilterError> {
    validate_params(cutoff_hz, sample_rate_hz, N)?;
    if ripple_db <= S::ZERO {
        return Err(FilterError::InvalidRipple);
    }

    let pi = S::PI;
    let two = S::TWO;

    // ε = sqrt(10^(Rp/10) - 1)
    let rp_linear = S::from_f64(10.0).powf(ripple_db / S::from_f64(10.0)) - S::ONE;
    let epsilon = rp_linear.sqrt();

    // a = (1/N)·arcsinh(1/ε)
    let n_f = S::from_f64(N as f64);
    let a = arcsinh(S::ONE / epsilon) / n_f;
    let sinh_a = a.sinh();
    let cosh_a = a.cosh();

    // Pre-warp digital cutoff → analog prototype frequency k
    let omega_d = pi * cutoff_hz / sample_rate_hz;
    let k = omega_d.tan();

    let n_biquads = N / 2;
    let has_first_order = N % 2 == 1;

    let mut biquads = [ChebBiquad::identity(); 4];
    let mut first_order: Option<ChebFirstOrder<S>> = None;

    #[allow(clippy::needless_range_loop)]
    for m in 0..n_biquads {
        // Pole index: k_idx = m+1 selects the upper-left quadrant pole pairs
        let k_idx = m + 1;
        let theta = pi * S::from_f64((2 * k_idx - 1) as f64) / (two * n_f);
        let sigma = -sinh_a * theta.sin(); // negative real part of prototype pole
        let omega_im = cosh_a * theta.cos(); // imaginary part

        let (b0, b1, b2, a1, a2) = cheb1_lp_biquad(sigma, omega_im, k);
        biquads[m] = ChebBiquad::new(b0, b1, b2, a1, a2);
    }

    if has_first_order {
        // The middle (real) pole for odd N: θ = π/2, cos(θ)=0
        // Prototype real pole: σ = -sinh(a)
        let sigma = -sinh_a; // = -sinh_a * sin(π/2)
        let (b0, b1, a1) = cheb1_lp_first_order(sigma, k);
        first_order = Some(ChebFirstOrder::new(b0, b1, a1));
    }

    Ok(ChebyshevI {
        biquads,
        first_order,
        n_biquads,
        has_first_order,
    })
}

/// Returns (b0,b1,b2,a1,a2) for one Chebyshev Type I LP biquad.
///
/// Each biquad comes from a conjugate pole pair (σ ± jω) of the prototype.
/// The prototype is frequency-scaled so its passband edge is at ωa = k:
///   H_a(s/k) = p_sq·k² / (s² - 2σk·s + p_sq·k²)
///
/// After BLT s → (z-1)/(z+1):
///   Numerator:   p_sq·k²·(z+1)²  →  coeffs [p_sq·k², 2p_sq·k², p_sq·k²]
///   Denominator: same formula as Butterworth LP biquad with scaled poles
///
/// DC gain = 1 per section ✓.
fn cheb1_lp_biquad<S: ControlScalar>(sigma: S, omega_im: S, k: S) -> (S, S, S, S, S) {
    let two = S::TWO;
    let p_sq = sigma * sigma + omega_im * omega_im;
    let k2 = k * k;
    let pk2 = p_sq * k2;

    // Denominator (frequency-scaled prototype denominator after BLT):
    let a0 = S::ONE - two * sigma * k + pk2;
    let a1_z = two * (pk2 - S::ONE);
    let a2 = S::ONE + two * sigma * k + pk2;

    // Numerator: p_sq·k²·(1+z⁻¹)²
    let b_all = pk2 / a0;
    let b0 = b_all;
    let b1 = two * b_all;
    let b2 = b_all;
    let a1 = a1_z / a0;
    let a2_n = a2 / a0;
    (b0, b1, b2, a1, a2_n)
}

/// Returns (b0,b1,a1) for the real-pole first-order Cheb1 LP section.
///
/// Prototype: H_a(s) = |σ| / (s + |σ|) (unit DC gain).
/// After frequency scaling s → s/k: pole moves to k·|σ|.
/// BLT s → (z-1)/(z+1):
///   H(z) = k|σ|/(1+k|σ|) · (1+z⁻¹)/(1 + (k|σ|-1)/(1+k|σ|)·z⁻¹)
fn cheb1_lp_first_order<S: ControlScalar>(sigma: S, k: S) -> (S, S, S) {
    let abs_sigma = (-sigma).max(S::ZERO); // σ < 0 for stable prototype pole
    let k_abs = k * abs_sigma; // k·|σ|
    let inv = S::ONE / (S::ONE + k_abs);
    let b0 = k_abs * inv;
    let b1 = k_abs * inv;
    let a1 = (k_abs - S::ONE) * inv;
    (b0, b1, a1)
}

// ─────────────────────────────────────────────────────────────
//  Design function: Chebyshev Type II lowpass
// ─────────────────────────────────────────────────────────────

/// Design an Nth-order Chebyshev Type II lowpass filter.
///
/// # Arguments
/// * `cutoff_hz`    — stopband-edge frequency in Hz (stopband starts here)
/// * `stopband_db`  — minimum stopband attenuation in dB (e.g. 40.0 or 60.0)
/// * `sample_rate_hz` — sample rate in Hz
///
/// The filter has monotone passband and equiripple stopband.
pub fn design_chebyshev2_lp<S: ControlScalar, const N: usize>(
    cutoff_hz: S,
    stopband_db: S,
    sample_rate_hz: S,
) -> Result<ChebyshevII<S, N>, FilterError> {
    validate_params(cutoff_hz, sample_rate_hz, N)?;
    if stopband_db <= S::ZERO {
        return Err(FilterError::InvalidRipple);
    }

    let pi = S::PI;
    let two = S::TWO;

    // ε = sqrt(10^(As/10) - 1)  for stopband attenuation
    let as_linear = S::from_f64(10.0).powf(stopband_db / S::from_f64(10.0)) - S::ONE;
    let epsilon = as_linear.sqrt();

    let n_f = S::from_f64(N as f64);
    let a = arcsinh(epsilon) / n_f;
    let sinh_a = a.sinh();
    let cosh_a = a.cosh();

    // Pre-warp
    let omega_d = pi * cutoff_hz / sample_rate_hz;
    let k = omega_d.tan();

    let n_biquads = N / 2;
    let has_first_order = N % 2 == 1;

    let mut biquads = [ChebBiquad::identity(); 4];
    let mut first_order: Option<ChebFirstOrder<S>> = None;

    #[allow(clippy::needless_range_loop)]
    for m in 0..n_biquads {
        let k_idx = m + 1;
        let theta = pi * S::from_f64((2 * k_idx - 1) as f64) / (two * n_f);

        // Type II: invert Type I poles and apply LP→LP frequency transformation
        // Type I pole: s_I = σ + jω where σ = -sinh(a)·sin(θ), ω = cosh(a)·cos(θ)
        // Type II pole: s_II = 1 / conj(s_I)  (inversion in s-plane)
        // Zero at: s_z = j / cos(θ)  (on imaginary axis)

        let sigma_i = -sinh_a * theta.sin();
        let omega_i = cosh_a * theta.cos();

        // Type II LP pole: p_II = conj(s_I) / |s_I|²  -- inversion
        let mag_sq = sigma_i * sigma_i + omega_i * omega_i;
        let sigma_ii = sigma_i / mag_sq; // still negative
        let omega_ii = -omega_i / mag_sq; // may be positive or negative

        // Type II zero (on imaginary axis): ±j/cos(θ)
        let zero_im = S::ONE / theta.cos().abs();

        let (b0, b1, b2, a1, a2) = cheb2_lp_biquad(sigma_ii, omega_ii, zero_im, k);
        biquads[m] = ChebBiquad::new(b0, b1, b2, a1, a2);
    }

    if has_first_order {
        // Middle Type I pole: θ = π/2, ω_I = 0, σ_I = -sinh(a)
        // Type II real pole: s_II = 1/σ_I = -1/sinh(a)
        let sigma_ii = -S::ONE / sinh_a;
        let (b0, b1, a1) = cheb2_lp_first_order(sigma_ii, k);
        first_order = Some(ChebFirstOrder::new(b0, b1, a1));
    }

    Ok(ChebyshevII {
        biquads,
        first_order,
        n_biquads,
        has_first_order,
    })
}

/// Returns (b0,b1,b2,a1,a2) for a Chebyshev Type II LP biquad.
///
/// The Type II prototype section has:
///   - Pole pair: σ_II ± j·ω_II  (complex, in LHP)
///   - Zero pair: ±j·z_im        (imaginary axis zeros)
///
/// Prototype transfer function:
///   H_a(s) = (p_sq/z_sq)·(s² + z_sq) / (s² - 2σ·s + p_sq)
///   where p_sq = σ²+ω², z_sq = z_im²
///
/// After frequency scaling s → s/k and BLT s → (z-1)/(z+1):
///
/// Denominator (same as Type I LP biquad with these poles):
///   a0 = 1 - 2σk + p_sq·k²
///   a1 = 2(p_sq·k² - 1)
///   a2 = 1 + 2σk + p_sq·k²
///
/// Numerator: (p_sq/z_sq)·(s² + z_sq·k²) in BLT maps to:
///   BLT of (s² + z_sq·k²) with s→(z-1)/(z+1):
///   (z-1)² + z_sq·k²·(z+1)² = (1+z_sq·k²)z² + 2(z_sq·k²-1)z + (1+z_sq·k²)
///
///   After multiplying by p_sq/z_sq:
///   b0 = (p_sq/z_sq)·(1 + z_sq·k²)/a0
///   b1 = 2(p_sq/z_sq)·(z_sq·k²-1)/a0
///   b2 = b0
///
/// DC gain = 1 ✓ (verified: num_sum = 4p_sq·k², den_sum = 4p_sq·k²).
fn cheb2_lp_biquad<S: ControlScalar>(sigma: S, omega_im: S, zero_im: S, k: S) -> (S, S, S, S, S) {
    let two = S::TWO;
    let p_sq = sigma * sigma + omega_im * omega_im;
    let z_sq = zero_im * zero_im;
    let k2 = k * k;
    let pk2 = p_sq * k2;
    let zk2 = z_sq * k2; // z_sq·k²

    // Denominator (frequency-scaled prototype after BLT)
    let a0 = S::ONE - two * sigma * k + pk2;
    let a1_z = two * (pk2 - S::ONE);
    let a2 = S::ONE + two * sigma * k + pk2;

    // Numerator: (p_sq/z_sq) · (1 + z_sq·k²·...) after BLT
    let pz_ratio = p_sq / z_sq;
    let b0 = pz_ratio * (S::ONE + zk2) / a0;
    let b1 = pz_ratio * two * (zk2 - S::ONE) / a0;
    let b2 = b0;
    let a1 = a1_z / a0;
    let a2_n = a2 / a0;
    (b0, b1, b2, a1, a2_n)
}

/// Returns (b0,b1,a1) for a Chebyshev Type II first-order LP section.
///
/// Prototype: H_a(s) = |σ_II|/(s + |σ_II|), cutoff at ω = |σ_II|.
/// After frequency scaling s → s/k and BLT s → (z-1)/(z+1):
///   H(z) = k|σ|/(1+k|σ|) · (1+z⁻¹)/(1 + (k|σ|-1)/(1+k|σ|)·z⁻¹)
fn cheb2_lp_first_order<S: ControlScalar>(sigma: S, k: S) -> (S, S, S) {
    let abs_sigma = (-sigma).max(S::ZERO);
    let k_abs = k * abs_sigma;
    let inv = S::ONE / (S::ONE + k_abs);
    let b0 = k_abs * inv;
    let b1 = k_abs * inv;
    let a1 = (k_abs - S::ONE) * inv;
    (b0, b1, a1)
}

// ─────────────────────────────────────────────────────────────
//  Parameter validation
// ─────────────────────────────────────────────────────────────

fn validate_params<S: ControlScalar>(
    cutoff_hz: S,
    sample_rate_hz: S,
    order: usize,
) -> Result<(), FilterError> {
    if order == 0 || order > 8 {
        return Err(FilterError::InvalidOrder);
    }
    if sample_rate_hz <= S::ZERO {
        return Err(FilterError::InvalidSampleRate);
    }
    if cutoff_hz <= S::ZERO || cutoff_hz >= sample_rate_hz * S::HALF {
        return Err(FilterError::InvalidFrequency);
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn measure_gain<F: FnMut(f64) -> f64>(mut filter: F, freq_hz: f64, sample_rate: f64) -> f64 {
        let n_settle = 10000usize;
        let n_measure = 2000usize;
        let mut max_out = 0.0_f64;
        for i in 0..(n_settle + n_measure) {
            let x = (2.0 * core::f64::consts::PI * freq_hz * i as f64 / sample_rate).sin();
            let y = filter(x);
            if i >= n_settle {
                let ya = y.abs();
                if ya > max_out {
                    max_out = ya;
                }
            }
        }
        max_out
    }

    #[test]
    fn chebyshev1_lp2_dc_gain() {
        let mut filt = design_chebyshev1_lp::<f64, 2>(100.0, 1.0, 1000.0).unwrap();
        for _ in 0..5000 {
            filt.update(1.0);
        }
        let y = filt.update(1.0);
        assert!(
            (y - 1.0).abs() < 0.05,
            "ChebyI DC gain should be ~1, got {y}"
        );
    }

    #[test]
    fn chebyshev1_lp2_stopband_attenuation() {
        let fs = 10000.0_f64;
        let fc = 500.0_f64;
        let mut filt = design_chebyshev1_lp::<f64, 2>(fc, 1.0, fs).unwrap();
        // At 3× cutoff, should be attenuated
        let gain = measure_gain(|x| filt.update(x), fc * 3.0, fs);
        assert!(
            gain < 0.3,
            "ChebyI stopband gain should be < 0.3, got {gain}"
        );
    }

    #[test]
    fn chebyshev1_lp4_stopband() {
        let fs = 10000.0_f64;
        let fc = 500.0_f64;
        let mut filt = design_chebyshev1_lp::<f64, 4>(fc, 0.5, fs).unwrap();
        // 4th order should give > 40 dB at 5× cutoff
        let gain = measure_gain(|x| filt.update(x), fc * 5.0, fs);
        assert!(
            gain < 0.01,
            "ChebyI N=4 stopband should be < 0.01, got {gain}"
        );
    }

    #[test]
    fn chebyshev1_lp_odd_order() {
        let mut filt = design_chebyshev1_lp::<f64, 3>(200.0, 1.0, 1000.0).unwrap();
        for _ in 0..5000 {
            filt.update(1.0);
        }
        let y = filt.update(1.0);
        assert!((y - 1.0).abs() < 0.05, "ChebyI N=3 DC gain, got {y}");
    }

    #[test]
    fn chebyshev2_lp2_dc_gain() {
        let mut filt = design_chebyshev2_lp::<f64, 2>(200.0, 40.0, 1000.0).unwrap();
        for _ in 0..5000 {
            filt.update(1.0);
        }
        let y = filt.update(1.0);
        assert!(
            (y - 1.0).abs() < 0.1,
            "ChebyII DC gain should be ~1, got {y}"
        );
    }

    #[test]
    fn chebyshev2_lp2_stopband_attenuation() {
        let fs = 10000.0_f64;
        let fc = 1000.0_f64; // stopband edge
        let mut filt = design_chebyshev2_lp::<f64, 4>(fc, 40.0, fs).unwrap();
        // At exactly the stopband edge, attenuation should be ≥ 40 dB
        let gain = measure_gain(|x| filt.update(x), fc, fs);
        assert!(
            gain < 0.01,
            "ChebyII stopband gain should be < 0.01 at {fc} Hz, got {gain}"
        );
    }

    #[test]
    fn chebyshev2_lp_odd_order() {
        let mut filt = design_chebyshev2_lp::<f64, 3>(300.0, 40.0, 1000.0).unwrap();
        for _ in 0..5000 {
            filt.update(1.0);
        }
        let y = filt.update(1.0);
        assert!((y - 1.0).abs() < 0.1, "ChebyII N=3 DC gain, got {y}");
    }

    #[test]
    fn chebyshev_invalid_order() {
        assert!(design_chebyshev1_lp::<f64, 0>(100.0, 1.0, 1000.0).is_err());
        assert!(design_chebyshev1_lp::<f64, 9>(100.0, 1.0, 1000.0).is_err());
        assert!(design_chebyshev2_lp::<f64, 0>(100.0, 40.0, 1000.0).is_err());
    }

    #[test]
    fn chebyshev_invalid_ripple() {
        assert!(design_chebyshev1_lp::<f64, 2>(100.0, -1.0, 1000.0).is_err());
        assert!(design_chebyshev1_lp::<f64, 2>(100.0, 0.0, 1000.0).is_err());
    }

    #[test]
    fn chebyshev1_reset() {
        let mut filt = design_chebyshev1_lp::<f64, 2>(100.0, 1.0, 1000.0).unwrap();
        for _ in 0..100 {
            filt.update(1.0);
        }
        filt.reset();
        let y = filt.update(0.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn chebyshev2_reset() {
        let mut filt = design_chebyshev2_lp::<f64, 2>(200.0, 40.0, 1000.0).unwrap();
        for _ in 0..100 {
            filt.update(1.0);
        }
        filt.reset();
        let y = filt.update(0.0);
        assert_eq!(y, 0.0);
    }
}
