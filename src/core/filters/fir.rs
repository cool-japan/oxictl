//! FIR (Finite Impulse Response) filter with windowed-sinc design.
//!
//! `FirFilter<S, N>` is an N-tap FIR filter computed by direct convolution
//! of the input circular buffer with the coefficient array.
//!
//! Window functions available: Rectangular, Hamming, Hanning, Blackman.
//! The lowpass design is via the windowed-sinc method.

use crate::core::filters::FilterError;
use crate::core::scalar::ControlScalar;
use heapless::Deque;

// ─────────────────────────────────────────────────────────────
//  WindowType enum
// ─────────────────────────────────────────────────────────────

/// Window function applied to the ideal sinc coefficients during FIR design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// No windowing (rectangular window).  Fastest roll-off but highest sidelobes.
    Rectangular,
    /// Hamming window.  Good sidelobe rejection, moderate transition band.
    Hamming,
    /// Hanning (Hann) window.  Smooth, good sidelobe rejection.
    Hanning,
    /// Blackman window.  Excellent sidelobe rejection, wider transition band.
    Blackman,
}

// ─────────────────────────────────────────────────────────────
//  FirFilter<S, N>
// ─────────────────────────────────────────────────────────────

/// N-tap FIR filter.
///
/// Coefficients are computed at construction time by `design_fir_lp`.
/// Processing uses a circular buffer of N samples and computes the
/// dot product with the coefficient array each call (direct-form convolution).
///
/// `N` must be at least 1.
#[derive(Debug)]
pub struct FirFilter<S: ControlScalar, const N: usize> {
    /// FIR coefficients h[0..N].
    coeffs: [S; N],
    /// Circular buffer of the N most recent input samples.
    buf: Deque<S, N>,
}

impl<S: ControlScalar, const N: usize> FirFilter<S, N> {
    /// Create an FIR filter from pre-computed coefficients.
    pub fn from_coeffs(coeffs: [S; N]) -> Self {
        Self {
            coeffs,
            buf: Deque::new(),
        }
    }

    /// Process one input sample and return the filtered output.
    ///
    /// The filter maintains an internal circular buffer.  The output is the
    /// inner product of the buffer with the coefficient array.
    #[inline]
    pub fn update(&mut self, x: S) -> S {
        // Push new sample, dropping the oldest when full.
        if self.buf.len() == N {
            let _ = self.buf.pop_front();
        }
        let _ = self.buf.push_back(x);

        // Direct-form FIR convolution.
        // buf[back] = newest, buf[front] = oldest.
        // h[0] multiplies the newest sample.
        let mut acc = S::ZERO;
        for (i, &sample) in self.buf.iter().rev().enumerate() {
            if i < N {
                acc += self.coeffs[i] * sample;
            }
        }
        acc
    }

    /// Reset the internal state (flush the circular buffer).
    pub fn reset(&mut self) {
        self.buf.clear();
    }

    /// Access the filter coefficients.
    pub fn coeffs(&self) -> &[S; N] {
        &self.coeffs
    }
}

// ─────────────────────────────────────────────────────────────
//  Windowed-sinc lowpass design
// ─────────────────────────────────────────────────────────────

/// Design an N-tap windowed-sinc lowpass FIR filter.
///
/// # Arguments
/// * `cutoff_normalized` — normalised cutoff frequency in (0, 0.5)
///   (i.e. `cutoff_hz / sample_rate_hz`).  0.5 = Nyquist.
/// * `window` — window function to apply to the ideal sinc coefficients.
///
/// The filter has linear phase (symmetric coefficients) and zero group-delay
/// distortion.  The delay is (N-1)/2 samples.
///
/// # Errors
/// Returns `FilterError` if:
/// * N == 0
/// * `cutoff_normalized` is not in (0, 0.5)
pub fn design_fir_lp<S: ControlScalar, const N: usize>(
    cutoff_normalized: S,
    window: WindowType,
) -> Result<FirFilter<S, N>, FilterError> {
    if N == 0 {
        return Err(FilterError::InvalidOrder);
    }
    if cutoff_normalized <= S::ZERO || cutoff_normalized >= S::HALF {
        return Err(FilterError::InvalidFrequency);
    }

    let pi = S::PI;
    let two_pi = S::TWO * pi;
    let n_f = S::from_f64(N as f64);
    // Half-length M = (N-1)/2
    let m = S::from_f64((N - 1) as f64) * S::HALF;

    let mut coeffs = [S::ZERO; N];
    let mut sum = S::ZERO;

    for (i, coeff) in coeffs.iter_mut().enumerate().take(N) {
        let i_f = S::from_f64(i as f64);
        // Normalised position relative to centre: n = i - M
        let n = i_f - m;

        // Ideal sinc coefficient
        let h_ideal = if n == S::ZERO {
            S::TWO * cutoff_normalized
        } else {
            let arg = S::TWO * pi * cutoff_normalized * n;
            arg.sin() / (pi * n)
        };

        // Window weight
        let w = window_weight::<S>(window, i_f, n_f, two_pi);

        *coeff = h_ideal * w;
        sum += *coeff;
    }

    // Normalise so that DC gain = 1 (sum of coefficients = 1).
    if sum.abs() > S::EPSILON {
        for c in coeffs.iter_mut() {
            *c = *c / sum;
        }
    }

    Ok(FirFilter::from_coeffs(coeffs))
}

/// Compute the window weight for sample index `i` of an N-point window.
#[inline]
fn window_weight<S: ControlScalar>(window: WindowType, i: S, n: S, two_pi: S) -> S {
    match window {
        WindowType::Rectangular => S::ONE,
        WindowType::Hamming => {
            // w(n) = 0.54 - 0.46·cos(2π·n/(N-1))
            let a0 = S::from_f64(0.54);
            let a1 = S::from_f64(0.46);
            a0 - a1 * (two_pi * i / (n - S::ONE)).cos()
        }
        WindowType::Hanning => {
            // w(n) = 0.5·(1 - cos(2π·n/(N-1)))
            S::HALF * (S::ONE - (two_pi * i / (n - S::ONE)).cos())
        }
        WindowType::Blackman => {
            // w(n) = 0.42 - 0.5·cos(2π·n/(N-1)) + 0.08·cos(4π·n/(N-1))
            let a0 = S::from_f64(0.42);
            let a1 = S::HALF;
            let a2 = S::from_f64(0.08);
            let arg = two_pi * i / (n - S::ONE);
            a0 - a1 * arg.cos() + a2 * (S::TWO * arg).cos()
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn measure_gain<F: FnMut(f64) -> f64>(mut filter: F, freq_norm: f64, n_samples: usize) -> f64 {
        let n_settle = n_samples;
        let n_measure = 2000usize;
        let mut max_out = 0.0_f64;
        let two_pi = 2.0 * core::f64::consts::PI;
        for i in 0..(n_settle + n_measure) {
            let x = (two_pi * freq_norm * i as f64).sin();
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
    fn fir_hamming_dc_gain() {
        let mut filt = design_fir_lp::<f64, 31>(0.1, WindowType::Hamming).unwrap();
        // DC gain should be 1.0 (sum of coefficients = 1)
        let dc = measure_gain(|x| filt.update(x), 0.0, 200);
        // Drive with constant 1.0 to measure DC gain
        let mut filt2 = design_fir_lp::<f64, 31>(0.1, WindowType::Hamming).unwrap();
        for _ in 0..500 {
            filt2.update(1.0);
        }
        let y = filt2.update(1.0);
        assert!(
            (y - 1.0).abs() < 0.001,
            "FIR DC gain should be 1.0, got {y}"
        );
        let _ = dc;
    }

    #[test]
    fn fir_hamming_cutoff_attenuation() {
        // At cutoff (normalised), gain should be approximately -6 dB (0.5)
        // For windowed sinc, the -6dB point is at the cutoff frequency
        let cutoff = 0.1_f64;
        let mut filt = design_fir_lp::<f64, 63>(cutoff, WindowType::Hamming).unwrap();
        let gain = measure_gain(|x| filt.update(x), cutoff, 500);
        // Expect ~0.5 ± 0.15 (windowing affects exact -6dB point)
        assert!(gain < 0.7, "At cutoff, gain should be < 0.7, got {gain}");
        assert!(gain > 0.1, "At cutoff, gain should be > 0.1, got {gain}");
    }

    #[test]
    fn fir_hamming_stopband() {
        let cutoff = 0.1_f64;
        let mut filt = design_fir_lp::<f64, 63>(cutoff, WindowType::Hamming).unwrap();
        // At 3× cutoff (normalised), should be well attenuated
        let gain = measure_gain(|x| filt.update(x), cutoff * 3.0, 500);
        assert!(gain < 0.1, "Stopband at 3× cutoff: {gain}");
    }

    #[test]
    fn fir_blackman_stopband() {
        let cutoff = 0.1_f64;
        let mut filt = design_fir_lp::<f64, 63>(cutoff, WindowType::Blackman).unwrap();
        // Blackman has better sidelobe suppression
        let gain = measure_gain(|x| filt.update(x), cutoff * 4.0, 500);
        assert!(gain < 0.05, "Blackman stopband at 4× cutoff: {gain}");
    }

    #[test]
    fn fir_rectangular_dc_gain() {
        let mut filt = design_fir_lp::<f64, 31>(0.2, WindowType::Rectangular).unwrap();
        for _ in 0..200 {
            filt.update(1.0);
        }
        let y = filt.update(1.0);
        assert!((y - 1.0).abs() < 0.001, "Rectangular FIR DC gain: {y}");
    }

    #[test]
    fn fir_hanning_dc_gain() {
        let mut filt = design_fir_lp::<f64, 31>(0.2, WindowType::Hanning).unwrap();
        for _ in 0..200 {
            filt.update(1.0);
        }
        let y = filt.update(1.0);
        assert!((y - 1.0).abs() < 0.001, "Hanning FIR DC gain: {y}");
    }

    #[test]
    fn fir_coefficient_symmetry() {
        // Windowed-sinc LP filter should have symmetric coefficients (linear phase)
        let filt = design_fir_lp::<f64, 31>(0.15, WindowType::Hamming).unwrap();
        let c = filt.coeffs();
        let n = c.len();
        for i in 0..n / 2 {
            assert!(
                (c[i] - c[n - 1 - i]).abs() < 1e-12,
                "Coefficients not symmetric at i={i}: {} vs {}",
                c[i],
                c[n - 1 - i]
            );
        }
    }

    #[test]
    fn fir_invalid_cutoff() {
        assert!(design_fir_lp::<f64, 31>(0.0, WindowType::Hamming).is_err());
        assert!(design_fir_lp::<f64, 31>(0.5, WindowType::Hamming).is_err());
        assert!(design_fir_lp::<f64, 31>(-0.1, WindowType::Hamming).is_err());
    }

    #[test]
    fn fir_reset() {
        let mut filt = design_fir_lp::<f64, 15>(0.1, WindowType::Hamming).unwrap();
        for _ in 0..50 {
            filt.update(1.0);
        }
        filt.reset();
        let y = filt.update(0.0);
        assert_eq!(y, 0.0, "After reset, zero input should give zero output");
    }

    #[test]
    fn fir_nyquist_attenuation() {
        let mut filt = design_fir_lp::<f64, 63>(0.1, WindowType::Hamming).unwrap();
        // Nyquist (normalised 0.5): alternating ±1
        let gain = measure_gain(|x| filt.update(x), 0.499, 500);
        assert!(gain < 0.01, "Nyquist should be strongly attenuated: {gain}");
    }

    #[test]
    fn window_type_eq() {
        assert_eq!(WindowType::Hamming, WindowType::Hamming);
        assert_ne!(WindowType::Hamming, WindowType::Blackman);
    }
}
