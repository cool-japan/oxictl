use crate::core::scalar::ControlScalar;

/// Discrete-time IIR filter in transposed direct-form II.
///
/// Represents H(z) = (b[0] + b[1]*z^-1 + ... + b[N-1]*z^{-(N-1)}) /
///                   (1   + a[0]*z^-1 + ... + a[N-1]*z^{-(N-1)})
///
/// N is the filter order. Both b and a arrays have N elements.
/// The denominator has an implicit leading coefficient of 1.
#[derive(Debug, Clone, Copy)]
pub struct TransferFn<S: ControlScalar, const N: usize> {
    b: [S; N],
    a: [S; N],
    w: [S; N],
}

impl<S: ControlScalar, const N: usize> TransferFn<S, N> {
    /// Create from numerator b and denominator a coefficients (both length N).
    pub fn new(b: [S; N], a: [S; N]) -> Self {
        Self {
            b,
            a,
            w: core::array::from_fn(|_| S::ZERO),
        }
    }

    /// Process one sample. Returns filtered output.
    ///
    /// State equations (transposed form II):
    ///   y[n]          = b[0]*x[n] + w[0][n]
    ///   w[i][n+1]     = b[i+1]*x[n] - a[i]*y[n] + w[i+1][n]  for i=0..N-2
    ///   w[N-1][n+1]   = -a[N-1]*y[n]   (no b[N] term — strictly proper)
    pub fn process(&mut self, x: S) -> S {
        if N == 0 {
            return x;
        }
        let y = self.b[0] * x + self.w[0];
        for i in 0..(N.saturating_sub(1)) {
            self.w[i] = self.b[i + 1] * x - self.a[i] * y + self.w[i + 1];
        }
        // Last state has no b[N] term (strictly proper: numerator degree < denominator degree)
        self.w[N - 1] = -self.a[N - 1] * y;
        y
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.w = core::array::from_fn(|_| S::ZERO);
    }

    pub fn b(&self) -> &[S; N] {
        &self.b
    }

    pub fn a(&self) -> &[S; N] {
        &self.a
    }
}

impl<S: ControlScalar> TransferFn<S, 1> {
    /// First-order lowpass: H(z) = (1-α) / (1 - α*z^-1)
    /// where α = exp(-dt/tau).
    pub fn first_order_lowpass(tau: S, dt: S) -> Self {
        let alpha = if tau > S::ZERO {
            (-(dt / tau)).exp()
        } else {
            S::ZERO
        };
        Self {
            b: [S::ONE - alpha],
            a: [-alpha],
            w: [S::ZERO],
        }
    }
}

/// Second-order IIR section (biquad) in transposed direct-form II.
/// H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
#[derive(Debug, Clone, Copy)]
pub struct Biquad<S: ControlScalar> {
    b0: S,
    b1: S,
    b2: S,
    a1: S,
    a2: S,
    w1: S,
    w2: S,
}

impl<S: ControlScalar> Biquad<S> {
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

    /// Butterworth lowpass biquad via bilinear transform.
    /// fc = cutoff frequency (Hz), fs = sample rate (Hz).
    pub fn lowpass(fc: S, fs: S) -> Self {
        let omega = S::PI * fc / fs;
        let k = omega.tan();
        let sqrt2 = S::from_f64(core::f64::consts::SQRT_2);
        let norm = S::ONE / (S::ONE + sqrt2 * k + k * k);
        let b0 = k * k * norm;
        let b1 = S::TWO * b0;
        let b2 = b0;
        let a1 = S::TWO * (k * k - S::ONE) * norm;
        let a2 = (S::ONE - sqrt2 * k + k * k) * norm;
        Self::new(b0, b1, b2, a1, a2)
    }

    /// Butterworth highpass biquad via bilinear transform.
    pub fn highpass(fc: S, fs: S) -> Self {
        let omega = S::PI * fc / fs;
        let k = omega.tan();
        let sqrt2 = S::from_f64(core::f64::consts::SQRT_2);
        let norm = S::ONE / (S::ONE + sqrt2 * k + k * k);
        let b0 = norm;
        let b1 = -(S::TWO * norm);
        let b2 = norm;
        let a1 = S::TWO * (k * k - S::ONE) * norm;
        let a2 = (S::ONE - sqrt2 * k + k * k) * norm;
        Self::new(b0, b1, b2, a1, a2)
    }

    /// Notch filter at frequency fc.
    pub fn notch(fc: S, fs: S, q: S) -> Self {
        let omega = S::TWO * S::PI * fc / fs;
        let (sin_w, cos_w) = omega.sin_cos();
        let alpha = sin_w / (S::TWO * q);
        let norm = S::ONE / (S::ONE + alpha);
        let b0 = norm;
        let b1 = -S::TWO * cos_w * norm;
        let b2 = norm;
        let a1 = b1;
        let a2 = (S::ONE - alpha) * norm;
        Self::new(b0, b1, b2, a1, a2)
    }

    /// Process one sample (transposed direct form II).
    pub fn process(&mut self, x: S) -> S {
        let y = self.b0 * x + self.w1;
        self.w1 = self.b1 * x - self.a1 * y + self.w2;
        self.w2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.w1 = S::ZERO;
        self.w2 = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_order_lowpass_correct_gain() {
        // DC gain should be 1.0
        let mut filt = TransferFn::<f64, 1>::first_order_lowpass(1.0, 0.01);
        for _ in 0..2000 {
            filt.process(1.0);
        }
        let y = filt.process(1.0);
        assert!((y - 1.0).abs() < 0.01, "DC gain should be 1.0, got {}", y);
    }

    #[test]
    fn first_order_lowpass_step_response() {
        // After 5*tau = 5s at 100Hz, should be ≥ 99% of steady state
        let mut filt = TransferFn::<f64, 1>::first_order_lowpass(1.0, 0.01);
        for _ in 0..500 {
            filt.process(1.0);
        }
        let y = filt.process(1.0);
        assert!(y > 0.99, "Should be ≥ 99% at 5τ, got {}", y);
    }

    #[test]
    fn first_order_lowpass_tau_zero() {
        let mut filt = TransferFn::<f64, 1>::first_order_lowpass(0.0, 0.01);
        // With tau=0, alpha=0, b=[1], a=[0] → pure passthrough
        let y = filt.process(5.0);
        assert_eq!(y, 5.0);
    }

    #[test]
    fn transferfn_reset() {
        let mut filt = TransferFn::<f64, 1>::first_order_lowpass(1.0, 0.01);
        for _ in 0..100 {
            filt.process(1.0);
        }
        filt.reset();
        let y = filt.process(0.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn biquad_lowpass_dc_gain() {
        let mut filt = Biquad::<f64>::lowpass(100.0, 1000.0);
        for _ in 0..2000 {
            filt.process(1.0);
        }
        let y = filt.process(1.0);
        assert!((y - 1.0).abs() < 0.01, "DC gain should be ~1, got {}", y);
    }

    #[test]
    fn biquad_highpass_dc_rejection() {
        let mut filt = Biquad::<f64>::highpass(100.0, 1000.0);
        for _ in 0..2000 {
            filt.process(1.0);
        }
        let y = filt.process(1.0);
        assert!(y.abs() < 0.01, "HP should reject DC, got {}", y);
    }

    #[test]
    fn biquad_noise_attenuation() {
        let mut filt = Biquad::<f64>::lowpass(10.0, 1000.0);
        let mut max_out = 0.0_f64;
        for i in 0..2000_usize {
            let x = if i % 2 == 0 { 1.0 } else { -1.0 };
            let y = filt.process(x).abs();
            if i > 500 {
                max_out = max_out.max(y);
            }
        }
        assert!(max_out < 0.05, "Nyquist should be attenuated: {}", max_out);
    }

    #[test]
    fn biquad_notch_attenuation() {
        // Notch at 100Hz, fs=1000Hz
        let mut filt = Biquad::<f64>::notch(100.0, 1000.0, 10.0);
        // Drive at exactly 100Hz: 10 samples per cycle
        let mut max_out = 0.0_f64;
        for i in 0..2000_usize {
            let x = (2.0 * core::f64::consts::PI * 100.0 * i as f64 / 1000.0).sin();
            let y = filt.process(x).abs();
            if i > 1000 {
                max_out = max_out.max(y);
            }
        }
        assert!(max_out < 0.1, "Notch should attenuate 100Hz: {}", max_out);
    }

    #[test]
    fn biquad_reset() {
        let mut filt = Biquad::<f64>::lowpass(100.0, 1000.0);
        for _ in 0..100 {
            filt.process(1.0);
        }
        filt.reset();
        let y = filt.process(0.0);
        assert_eq!(y, 0.0);
    }
}
