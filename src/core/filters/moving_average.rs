//! Moving average and related online statistical filters.
//!
//! All filters are `no_std` compatible and use `heapless` for circular buffers.
//! They implement `update(&mut self, x: S) -> S` for sample-by-sample processing.

use crate::core::scalar::ControlScalar;
use heapless::Deque;

// ─────────────────────────────────────────────────────────────
//  MovingAverage<S, N>
// ─────────────────────────────────────────────────────────────

/// Sliding-window mean filter with window length N.
///
/// Maintains a circular buffer of the N most recent samples and returns their
/// arithmetic mean on each call to `update`.  The running sum is maintained
/// incrementally so that each update costs O(1).
///
/// # Initialisation
/// The buffer is initialised with zeros; the first N samples build up the window.
#[derive(Debug)]
pub struct MovingAverage<S: ControlScalar, const N: usize> {
    buf: Deque<S, N>,
    sum: S,
}

impl<S: ControlScalar, const N: usize> MovingAverage<S, N> {
    /// Create a new `MovingAverage` initialised to zero.
    pub fn new() -> Self {
        Self {
            buf: Deque::new(),
            sum: S::ZERO,
        }
    }

    /// Process one sample and return the windowed mean.
    ///
    /// Before the buffer is full the mean is computed over the samples seen so far.
    pub fn update(&mut self, x: S) -> S {
        if self.buf.len() == N {
            // Remove the oldest sample from the running sum.
            if let Some(old) = self.buf.pop_front() {
                self.sum -= old;
            }
        }
        self.sum += x;
        // push_back: since we just made room (or buffer not full yet), this won't fail.
        let _ = self.buf.push_back(x);

        let count = S::from_f64(self.buf.len() as f64);
        if count > S::ZERO {
            self.sum / count
        } else {
            S::ZERO
        }
    }

    /// Reset the filter state to zero.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.sum = S::ZERO;
    }

    /// Returns the current number of samples in the window.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns true if the window is not yet full.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

impl<S: ControlScalar, const N: usize> Default for MovingAverage<S, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
//  ExponentialMovingAverage<S>
// ─────────────────────────────────────────────────────────────

/// First-order IIR exponential moving average.
///
/// `y[n] = α·x[n] + (1-α)·y[n-1]`
///
/// where `α ∈ (0, 1]` is the smoothing coefficient.  A larger α gives faster
/// response (less smoothing); α = 1 is a pass-through.
///
/// The time constant in samples is `τ ≈ 1/α` for small α.
#[derive(Debug, Clone, Copy)]
pub struct ExponentialMovingAverage<S: ControlScalar> {
    alpha: S,
    one_minus_alpha: S,
    state: S,
}

impl<S: ControlScalar> ExponentialMovingAverage<S> {
    /// Create a new EMA with smoothing coefficient `alpha`.
    ///
    /// # Errors
    /// Returns `None` if `alpha` is not in the range (0, 1].
    pub fn new(alpha: S) -> Option<Self> {
        if alpha <= S::ZERO || alpha > S::ONE {
            return None;
        }
        Some(Self {
            alpha,
            one_minus_alpha: S::ONE - alpha,
            state: S::ZERO,
        })
    }

    /// Create an EMA from a time constant `tau` (seconds) and sample period `dt` (seconds).
    ///
    /// Uses `alpha = 1 - exp(-dt/tau)`.  Returns `None` if parameters are invalid.
    pub fn from_time_constant(tau: S, dt: S) -> Option<Self> {
        if tau <= S::ZERO || dt <= S::ZERO {
            return None;
        }
        let alpha = S::ONE - (-(dt / tau)).exp();
        Self::new(alpha)
    }

    /// Process one sample and return the filtered output.
    #[inline]
    pub fn update(&mut self, x: S) -> S {
        self.state = self.alpha * x + self.one_minus_alpha * self.state;
        self.state
    }

    /// Reset internal state to zero.
    pub fn reset(&mut self) {
        self.state = S::ZERO;
    }

    /// Initialise the internal state to a specific value (warm-start).
    pub fn set_state(&mut self, v: S) {
        self.state = v;
    }

    /// Returns the current alpha coefficient.
    pub fn alpha(&self) -> S {
        self.alpha
    }
}

// ─────────────────────────────────────────────────────────────
//  MovingRms<S, N>
// ─────────────────────────────────────────────────────────────

/// Sliding-window root-mean-square filter with window length N.
///
/// The running sum of squares is maintained incrementally for O(1) updates.
#[derive(Debug)]
pub struct MovingRms<S: ControlScalar, const N: usize> {
    buf: Deque<S, N>,
    sum_sq: S,
}

impl<S: ControlScalar, const N: usize> MovingRms<S, N> {
    /// Create a new `MovingRms` initialised to zero.
    pub fn new() -> Self {
        Self {
            buf: Deque::new(),
            sum_sq: S::ZERO,
        }
    }

    /// Process one sample and return the windowed RMS.
    pub fn update(&mut self, x: S) -> S {
        if self.buf.len() == N {
            if let Some(old) = self.buf.pop_front() {
                self.sum_sq = (self.sum_sq - old * old).max(S::ZERO);
            }
        }
        self.sum_sq += x * x;
        let _ = self.buf.push_back(x);

        let count = S::from_f64(self.buf.len() as f64);
        if count > S::ZERO {
            (self.sum_sq / count).sqrt()
        } else {
            S::ZERO
        }
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.sum_sq = S::ZERO;
    }

    /// Current window length (number of samples buffered).
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns true if no samples have been buffered yet.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

impl<S: ControlScalar, const N: usize> Default for MovingRms<S, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
//  MovingVariance<S, N>
// ─────────────────────────────────────────────────────────────

/// Sliding-window variance filter using Welford's online algorithm adapted for
/// a fixed-size window.
///
/// Returns the *population* variance of the N most recent samples.
/// The mean is also available via `mean()`.
#[derive(Debug)]
pub struct MovingVariance<S: ControlScalar, const N: usize> {
    buf: Deque<S, N>,
    sum: S,
    sum_sq: S,
}

impl<S: ControlScalar, const N: usize> MovingVariance<S, N> {
    /// Create a new `MovingVariance` initialised to zero.
    pub fn new() -> Self {
        Self {
            buf: Deque::new(),
            sum: S::ZERO,
            sum_sq: S::ZERO,
        }
    }

    /// Process one sample and return the windowed population variance.
    pub fn update(&mut self, x: S) -> S {
        if self.buf.len() == N {
            if let Some(old) = self.buf.pop_front() {
                self.sum -= old;
                self.sum_sq = (self.sum_sq - old * old).max(S::ZERO);
            }
        }
        self.sum += x;
        self.sum_sq += x * x;
        let _ = self.buf.push_back(x);

        let count = S::from_f64(self.buf.len() as f64);
        if count > S::ZERO {
            // Var = E[X²] - (E[X])²
            let mean = self.sum / count;
            let mean_sq = self.sum_sq / count;
            (mean_sq - mean * mean).max(S::ZERO)
        } else {
            S::ZERO
        }
    }

    /// Returns the current windowed mean (without advancing the filter).
    pub fn mean(&self) -> S {
        let count = S::from_f64(self.buf.len() as f64);
        if count > S::ZERO {
            self.sum / count
        } else {
            S::ZERO
        }
    }

    /// Returns the current windowed standard deviation.
    pub fn std_dev(&self) -> S {
        let count = S::from_f64(self.buf.len() as f64);
        if count > S::ZERO {
            let mean = self.sum / count;
            let mean_sq = self.sum_sq / count;
            (mean_sq - mean * mean).max(S::ZERO).sqrt()
        } else {
            S::ZERO
        }
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.sum = S::ZERO;
        self.sum_sq = S::ZERO;
    }

    /// Current window length.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns true if no samples have been buffered yet.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

impl<S: ControlScalar, const N: usize> Default for MovingVariance<S, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moving_average_dc_gain() {
        let mut ma = MovingAverage::<f64, 8>::new();
        // After filling the window with 1.0, output should be 1.0
        for _ in 0..20 {
            ma.update(1.0);
        }
        let y = ma.update(1.0);
        assert!((y - 1.0).abs() < 1e-12, "DC gain should be 1.0, got {y}");
    }

    #[test]
    fn moving_average_step_response() {
        let mut ma = MovingAverage::<f64, 4>::new();
        // Feed a step: 0...0 then 1...1
        for _ in 0..4 {
            ma.update(0.0);
        }
        // First 1.0 sample: mean = 1/4
        let y1 = ma.update(1.0);
        assert!((y1 - 0.25).abs() < 1e-12, "Step after 4 zeros: got {y1}");
        // After 4 more 1.0 samples
        for _ in 0..3 {
            ma.update(1.0);
        }
        let y2 = ma.update(1.0);
        assert!((y2 - 1.0).abs() < 1e-12, "Fully settled at 1.0: got {y2}");
    }

    #[test]
    fn moving_average_nyquist_attenuation() {
        let mut ma = MovingAverage::<f64, 8>::new();
        // Nyquist: alternating ±1 — should average to near zero once window fills
        let mut out = 0.0;
        for i in 0..200 {
            let x = if i % 2 == 0 { 1.0 } else { -1.0 };
            out = ma.update(x);
        }
        assert!(out.abs() < 0.01, "Nyquist should be near zero, got {out}");
    }

    #[test]
    fn moving_average_reset() {
        let mut ma = MovingAverage::<f64, 4>::new();
        for _ in 0..10 {
            ma.update(1.0);
        }
        ma.reset();
        let y = ma.update(0.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn ema_dc_gain() {
        let mut ema = ExponentialMovingAverage::<f64>::new(0.1).unwrap();
        for _ in 0..1000 {
            ema.update(1.0);
        }
        let y = ema.update(1.0);
        assert!((y - 1.0).abs() < 0.01, "EMA DC gain: {y}");
    }

    #[test]
    fn ema_alpha_one_passthrough() {
        let mut ema = ExponentialMovingAverage::<f64>::new(1.0).unwrap();
        let y = ema.update(core::f64::consts::PI);
        assert!(
            (y - core::f64::consts::PI).abs() < 1e-12,
            "alpha=1 should be passthrough: {y}"
        );
    }

    #[test]
    fn ema_from_time_constant() {
        let ema = ExponentialMovingAverage::<f64>::from_time_constant(1.0, 0.01);
        assert!(ema.is_some());
        let ema_invalid = ExponentialMovingAverage::<f64>::from_time_constant(-1.0, 0.01);
        assert!(ema_invalid.is_none());
    }

    #[test]
    fn ema_invalid_alpha() {
        assert!(ExponentialMovingAverage::<f64>::new(0.0).is_none());
        assert!(ExponentialMovingAverage::<f64>::new(1.1).is_none());
        assert!(ExponentialMovingAverage::<f64>::new(-0.1).is_none());
    }

    #[test]
    fn ema_reset() {
        let mut ema = ExponentialMovingAverage::<f64>::new(0.1).unwrap();
        for _ in 0..100 {
            ema.update(1.0);
        }
        ema.reset();
        let y = ema.update(0.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn moving_rms_dc() {
        let mut rms = MovingRms::<f64, 8>::new();
        for _ in 0..20 {
            rms.update(3.0);
        }
        let y = rms.update(3.0);
        assert!(
            (y - 3.0).abs() < 1e-10,
            "RMS of DC 3.0 should be 3.0, got {y}"
        );
    }

    #[test]
    fn moving_rms_alternating() {
        let mut rms = MovingRms::<f64, 8>::new();
        for i in 0..100 {
            let x = if i % 2 == 0 { 1.0 } else { -1.0 };
            rms.update(x);
        }
        let y = rms.update(1.0);
        // RMS of ±1 signal = 1.0
        assert!((y - 1.0).abs() < 0.01, "RMS of ±1 = 1.0, got {y}");
    }

    #[test]
    fn moving_rms_reset() {
        let mut rms = MovingRms::<f64, 4>::new();
        for _ in 0..10 {
            rms.update(5.0);
        }
        rms.reset();
        let y = rms.update(0.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn moving_variance_constant_signal() {
        let mut mv = MovingVariance::<f64, 8>::new();
        for _ in 0..20 {
            mv.update(5.0);
        }
        let v = mv.update(5.0);
        assert!(v.abs() < 1e-10, "Variance of constant should be 0, got {v}");
    }

    #[test]
    fn moving_variance_known_values() {
        let mut mv = MovingVariance::<f64, 4>::new();
        // Feed [1,2,3,4,5] → window becomes [2,3,4,5]
        // Mean=3.5, E[X²]=(4+9+16+25)/4=13.5, Var=13.5-12.25=1.25
        for v in [1.0, 2.0, 3.0, 4.0] {
            mv.update(v);
        }
        let var = mv.update(5.0);
        assert!(
            (var - 1.25).abs() < 1e-10,
            "Variance of [2,3,4,5] should be 1.25, got {var}"
        );
    }

    #[test]
    fn moving_variance_mean() {
        let mut mv = MovingVariance::<f64, 4>::new();
        for v in [2.0, 4.0, 6.0, 8.0] {
            mv.update(v);
        }
        let mean = mv.mean();
        assert!(
            (mean - 5.0).abs() < 1e-10,
            "Mean of [2,4,6,8] should be 5.0, got {mean}"
        );
    }

    #[test]
    fn moving_variance_reset() {
        let mut mv = MovingVariance::<f64, 4>::new();
        for _ in 0..10 {
            mv.update(7.0);
        }
        mv.reset();
        let v = mv.update(0.0);
        assert_eq!(v, 0.0);
    }
}
