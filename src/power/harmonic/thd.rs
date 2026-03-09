//! Total Harmonic Distortion (THD) analyzer using the Goertzel algorithm.
//!
//! The Goertzel algorithm efficiently computes the DFT magnitude at a single
//! frequency bin in O(N) time and O(1) extra space per bin — ideal for
//! embedded applications where a full FFT is unavailable.
//!
//! THD is defined as:
//! ```text
//!   THD (%) = 100 · √(Σₙ₌₂^{N_harm} Vₙ²) / V₁
//! ```
//! where V₁ is the fundamental RMS and Vₙ are harmonic RMS magnitudes.
use crate::core::scalar::ControlScalar;

/// N-sample circular-window THD analyzer.
pub struct ThdAnalyzer<S: ControlScalar, const N: usize> {
    /// Circular sample buffer.
    buffer: [S; N],
    /// Write head.
    head: usize,
    /// Number of valid samples (saturates at N).
    count: usize,
    /// Fundamental frequency (Hz).
    pub fundamental_hz: S,
    /// Sample rate (Hz).
    pub sample_rate: S,
}

impl<S: ControlScalar, const N: usize> ThdAnalyzer<S, N> {
    /// Create a new analyzer.
    ///
    /// * `fundamental_hz` – fundamental frequency (Hz), e.g. 50.0 or 60.0.
    /// * `sample_rate`    – sample rate (Hz).
    pub fn new(fundamental_hz: S, sample_rate: S) -> Self {
        Self {
            buffer: core::array::from_fn(|_| S::ZERO),
            head: 0,
            count: 0,
            fundamental_hz,
            sample_rate,
        }
    }

    /// Push a new sample into the circular buffer.
    pub fn push(&mut self, sample: S) {
        self.buffer[self.head] = sample;
        self.head = (self.head + 1) % N;
        if self.count < N {
            self.count += 1;
        }
    }

    /// Compute the Goertzel magnitude at bin index `k` (1-indexed harmonic).
    ///
    /// The Goertzel algorithm evaluates X[k] = Σₙ xₙ · e^{-j2πkn/N}.
    /// The magnitude is √(s1² + s2² − s1·s2·2cos(ω)).
    pub fn goertzel_magnitude(&self, k: usize) -> S {
        let n = self.count;
        if n == 0 {
            return S::ZERO;
        }

        let pi = S::PI;
        let omega = S::TWO * pi * S::from_f64(k as f64) / S::from_f64(n as f64);
        let coeff = S::TWO * omega.cos();

        let mut s_prev2 = S::ZERO;
        let mut s_prev1 = S::ZERO;

        // Iterate through samples in chronological order.
        for i in 0..n {
            let idx = if self.count < N {
                i
            } else {
                (self.head + i) % N
            };
            let s = coeff * s_prev1 - s_prev2 + self.buffer[idx];
            s_prev2 = s_prev1;
            s_prev1 = s;
        }

        // Power = s_prev1² + s_prev2² - coeff·s_prev1·s_prev2
        let power = s_prev1 * s_prev1 + s_prev2 * s_prev2 - coeff * s_prev1 * s_prev2;
        if power <= S::ZERO {
            return S::ZERO;
        }
        power.sqrt()
    }

    /// DFT bin index (1-indexed) that best matches `fundamental_hz`.
    ///
    /// For a window of `count` samples at `sample_rate` Hz the bin spacing is
    /// `sample_rate / count` Hz, so the closest bin to `fundamental_hz` is
    /// `round(fundamental_hz * count / sample_rate)`.  Clamped to `[1, count/2]`.
    fn fundamental_bin(&self) -> usize {
        if self.count == 0 {
            return 1;
        }
        let bin_f = self.fundamental_hz.to_f64() * self.count as f64 / self.sample_rate.to_f64();
        // Round to nearest integer, then clamp.
        let bin = (bin_f + 0.5) as usize;
        bin.max(1).min(self.count / 2)
    }

    /// Fundamental harmonic magnitude at the bin closest to `fundamental_hz`.
    pub fn fundamental_magnitude(&self) -> S {
        self.goertzel_magnitude(self.fundamental_bin())
    }

    /// Compute THD (%) including harmonics 2 through N/2 of the fundamental bin.
    ///
    /// The k-th harmonic is at bin `k * fundamental_bin`.
    /// Returns 0 when the fundamental is zero to avoid division by zero.
    pub fn compute_thd(&self) -> S {
        let k1 = self.fundamental_bin();
        let v1 = self.goertzel_magnitude(k1);
        if v1 < S::EPSILON {
            return S::ZERO;
        }

        let max_bin = self.count / 2;
        let mut sum_sq = S::ZERO;
        let mut h = 2_usize;
        loop {
            let bin = h * k1;
            if bin > max_bin {
                break;
            }
            let vn = self.goertzel_magnitude(bin);
            sum_sq += vn * vn;
            h += 1;
        }

        let hundred = S::from_f64(100.0);
        sum_sq.sqrt() / v1 * hundred
    }

    /// Individual harmonic ratio Vn/V1 × 100 (%).
    pub fn harmonic_ratio(&self, n: usize) -> S {
        let v1 = self.fundamental_magnitude();
        if v1 < S::EPSILON {
            return S::ZERO;
        }
        let vn = self.goertzel_magnitude(n);
        vn / v1 * S::from_f64(100.0)
    }

    /// Number of samples currently in the window.
    pub fn sample_count(&self) -> usize {
        self.count
    }

    /// Clear the sample buffer.
    pub fn clear(&mut self) {
        for s in &mut self.buffer {
            *s = S::ZERO;
        }
        self.head = 0;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fill the analyzer with a pure sine at frequency `freq_hz`.
    fn fill_sine<const N: usize>(
        analyzer: &mut ThdAnalyzer<f64, N>,
        freq_hz: f64,
        sample_rate: f64,
        n: usize,
    ) {
        use core::f64::consts::PI;
        for i in 0..n {
            let t = i as f64 / sample_rate;
            let s = (2.0 * PI * freq_hz * t).sin();
            analyzer.push(s);
        }
    }

    #[test]
    fn pure_sine_has_low_thd() {
        let sample_rate = 1000.0_f64;
        let f0 = 50.0_f64;
        let mut analyzer = ThdAnalyzer::<f64, 64>::new(f0, sample_rate);
        // Fill with exactly N samples of fundamental.
        fill_sine(&mut analyzer, f0, sample_rate, 64);
        let thd = analyzer.compute_thd();
        // For a pure sine the THD should be very low (dominated by series truncation).
        assert!(thd < 20.0, "THD for pure sine too high: {}%", thd);
    }

    #[test]
    fn fundamental_magnitude_nonzero_for_sine() {
        let sample_rate = 1000.0_f64;
        let f0 = 50.0_f64;
        let mut analyzer = ThdAnalyzer::<f64, 64>::new(f0, sample_rate);
        fill_sine(&mut analyzer, f0, sample_rate, 64);
        let v1 = analyzer.fundamental_magnitude();
        assert!(v1 > 0.0, "fundamental magnitude should be > 0, got {}", v1);
    }

    #[test]
    fn zero_signal_gives_zero_thd() {
        let mut analyzer = ThdAnalyzer::<f64, 32>::new(50.0, 1000.0);
        for _ in 0..32 {
            analyzer.push(0.0);
        }
        assert_eq!(analyzer.compute_thd(), 0.0);
    }

    #[test]
    fn clear_resets_state() {
        let mut analyzer = ThdAnalyzer::<f64, 32>::new(50.0, 1000.0);
        fill_sine(&mut analyzer, 50.0, 1000.0, 32);
        analyzer.clear();
        assert_eq!(analyzer.sample_count(), 0);
        assert_eq!(analyzer.fundamental_magnitude(), 0.0);
    }
}
