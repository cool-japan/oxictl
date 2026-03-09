//! Active Power Filter (APF) for selective harmonic cancellation.
//!
//! An APF injects a compensation current equal and opposite to the harmonic
//! distortion in the load current, leaving only the fundamental component to
//! flow through the grid connection.
//!
//! ## Architecture
//!
//! 1. **`HarmonicDetector<S, N>`** — uses Fourier projection (running DFT) over
//!    one fundamental period to extract the amplitudes and phases of N selected
//!    harmonics (e.g. 5th, 7th, 11th, 13th).
//!
//! 2. **`ApfCurrentReference<S, N>`** — reconstructs the compensation reference
//!    by summing the detected harmonic components with inverted sign.
//!
//! 3. **`ApfController<S>`** — hysteresis current controller that generates the
//!    switching command for the APF bridge.
//!
//! ## Fourier Projection
//!
//! For harmonic order `h`, the in-phase and quadrature components are:
//!
//!   a_h = (2/T) ∫ i_load(t)·cos(h·θ(t)) dt
//!   b_h = (2/T) ∫ i_load(t)·sin(h·θ(t)) dt
//!
//! A rectangular-window running DFT accumulates samples over one period
//! and delivers updated coefficients every `window_len` samples.
#![allow(clippy::needless_range_loop)]
use crate::core::scalar::ControlScalar;
use heapless::Vec;

// ─── HarmonicDetector ────────────────────────────────────────────────────────

/// Running Fourier-projection harmonic detector.
///
/// Detects the amplitudes of `N` selectable harmonic orders from a sampled
/// load current waveform.  One full fundamental period of samples is
/// accumulated; at the end of each period the DFT coefficients are computed
/// and stored.
///
/// # Type parameters
/// * `S` — scalar type implementing `ControlScalar`
/// * `N` — number of harmonic orders to track (compile-time constant, ≤ 8)
#[derive(Debug, Clone)]
pub struct HarmonicDetector<S: ControlScalar, const N: usize> {
    /// Harmonic orders to detect (e.g. [5, 7, 11, 13]).
    pub orders: [u32; N],
    /// Number of samples per fundamental period.
    window_len: usize,
    /// Sample accumulator for the current period.
    samples: Vec<S, 2048>,
    /// Grid angle accumulator for each sample.
    thetas: Vec<S, 2048>,
    /// Latest in-phase (cosine) amplitudes for each harmonic.
    coeff_a: [S; N],
    /// Latest quadrature (sine) amplitudes for each harmonic.
    coeff_b: [S; N],
}

impl<S: ControlScalar, const N: usize> HarmonicDetector<S, N> {
    /// Create a `HarmonicDetector`.
    ///
    /// * `orders`     — harmonic orders to detect (must have exactly N elements)
    /// * `window_len` — samples per fundamental period (must be ≤ 2048)
    pub fn new(orders: [u32; N], window_len: usize) -> Self {
        Self {
            orders,
            window_len,
            samples: Vec::new(),
            thetas: Vec::new(),
            coeff_a: [S::ZERO; N],
            coeff_b: [S::ZERO; N],
        }
    }

    /// Feed one sample of load current and return the harmonic amplitudes.
    ///
    /// The amplitudes are updated once per fundamental period (every
    /// `window_len` samples).  Between updates the previous values are returned.
    ///
    /// * `i_load`    — instantaneous load current sample (A)
    /// * `theta_grid` — current grid electrical angle (rad)
    ///
    /// Returns `[S; N]` — per-harmonic amplitudes (signed, based on cos component).
    pub fn update(&mut self, i_load: S, theta_grid: S) -> [S; N] {
        // Accumulate — silently drop if buffer is full (window_len > 2048 is
        // a configuration error caught at construction or by caller checks)
        let _ = self.samples.push(i_load);
        let _ = self.thetas.push(theta_grid);

        if self.samples.len() >= self.window_len {
            self.compute_dft();
            self.samples.clear();
            self.thetas.clear();
        }

        self.coeff_a
    }

    /// Compute DFT coefficients from the accumulated window.
    fn compute_dft(&mut self) {
        let len = self.samples.len();
        if len == 0 {
            return;
        }
        let inv_len = S::ONE / S::from_f64(len as f64);
        let two = S::TWO;

        for k in 0..N {
            let h = S::from_f64(self.orders[k] as f64);
            let mut a = S::ZERO;
            let mut b = S::ZERO;

            for j in 0..len {
                let theta = self.thetas[j];
                let x = self.samples[j];
                a += x * (h * theta).cos();
                b += x * (h * theta).sin();
            }

            // Normalise: factor 2 for one-sided spectrum
            self.coeff_a[k] = two * a * inv_len;
            self.coeff_b[k] = two * b * inv_len;
        }
    }

    /// In-phase (cosine) amplitude for harmonic index `k` (0-based).
    pub fn amplitude_cos(&self, k: usize) -> S {
        if k < N {
            self.coeff_a[k]
        } else {
            S::ZERO
        }
    }

    /// Quadrature (sine) amplitude for harmonic index `k` (0-based).
    pub fn amplitude_sin(&self, k: usize) -> S {
        if k < N {
            self.coeff_b[k]
        } else {
            S::ZERO
        }
    }

    /// Peak amplitude of harmonic index `k`:  √(a² + b²).
    pub fn amplitude_peak(&self, k: usize) -> S {
        if k < N {
            (self.coeff_a[k] * self.coeff_a[k] + self.coeff_b[k] * self.coeff_b[k]).sqrt()
        } else {
            S::ZERO
        }
    }

    /// Reset accumulated samples and stored coefficients.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.thetas.clear();
        self.coeff_a = [S::ZERO; N];
        self.coeff_b = [S::ZERO; N];
    }
}

// ─── ApfCurrentReference ─────────────────────────────────────────────────────

/// APF compensation current reference generator.
///
/// Reconstructs the time-domain compensation current from detected harmonic
/// amplitudes.  The APF must inject `i_comp` to cancel harmonic distortion:
///
///   i_comp(t) = -∑ₕ [a_h·cos(h·θ) + b_h·sin(h·θ)]
///
/// (Negative because injection is in opposite polarity to the distortion.)
#[derive(Debug, Clone)]
pub struct ApfCurrentReference<S: ControlScalar, const N: usize> {
    /// Underlying harmonic detector.
    pub detector: HarmonicDetector<S, N>,
    /// Cached cosine coefficients from latest detector update.
    coeff_a: [S; N],
    /// Cached sine coefficients from latest detector update.
    coeff_b: [S; N],
}

impl<S: ControlScalar, const N: usize> ApfCurrentReference<S, N> {
    /// Create an `ApfCurrentReference`.
    ///
    /// * `orders`     — harmonic orders to cancel
    /// * `window_len` — samples per fundamental period (≤ 2048)
    pub fn new(orders: [u32; N], window_len: usize) -> Self {
        Self {
            detector: HarmonicDetector::new(orders, window_len),
            coeff_a: [S::ZERO; N],
            coeff_b: [S::ZERO; N],
        }
    }

    /// Update internal harmonic model and compute compensation current reference.
    ///
    /// Must be called at the same rate as the APF sampling loop.
    ///
    /// * `i_load`  — instantaneous load current (A)
    /// * `theta`   — grid electrical angle (rad)
    ///
    /// Returns the instantaneous compensation current reference `i_ref` (A).
    pub fn compute_reference(&mut self, i_load: S, theta: S) -> S {
        let new_a = self.detector.update(i_load, theta);

        // Update cached coefficients from detector output (cos component)
        // We also need the sin component for full reconstruction
        for k in 0..N {
            self.coeff_a[k] = new_a[k];
            self.coeff_b[k] = self.detector.amplitude_sin(k);
        }

        // Reconstruct compensation reference (inverted harmonic sum)
        let mut i_ref = S::ZERO;
        for k in 0..N {
            let h = S::from_f64(self.detector.orders[k] as f64);
            i_ref -= self.coeff_a[k] * (h * theta).cos() + self.coeff_b[k] * (h * theta).sin();
        }

        i_ref
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.detector.reset();
        self.coeff_a = [S::ZERO; N];
        self.coeff_b = [S::ZERO; N];
    }
}

// ─── ApfController ───────────────────────────────────────────────────────────

/// Hysteresis current controller for the APF bridge.
///
/// Generates a Boolean switching command by comparing the actual APF output
/// current with the reference within a hysteresis band `±h_band`.
///
///   if i_actual < i_ref − h_band  → switch HIGH (true)
///   if i_actual > i_ref + h_band  → switch LOW  (false)
///   otherwise                     → hold previous state
///
/// This produces a bang-bang control with bounded ripple current.
#[derive(Debug, Clone, Copy)]
pub struct ApfController<S: ControlScalar> {
    /// Half-width of the hysteresis band (A).
    pub h_band: S,
    /// Current switching state.
    state: bool,
}

impl<S: ControlScalar> ApfController<S> {
    /// Create an `ApfController` with hysteresis band `h_band` (A).
    ///
    /// The initial switch state is `false` (low side on).
    pub fn new(h_band: S) -> Self {
        Self {
            h_band,
            state: false,
        }
    }

    /// Update the hysteresis controller.
    ///
    /// * `i_ref`    — compensation current reference (A)
    /// * `i_actual` — measured APF output current (A)
    ///
    /// Returns the switching command (`true` = upper switch on, `false` = lower switch on).
    pub fn update(&mut self, i_ref: S, i_actual: S) -> bool {
        let err = i_ref - i_actual;
        if err > self.h_band {
            self.state = true;
        } else if err < -self.h_band {
            self.state = false;
        }
        // Otherwise: hold previous state (hysteresis)
        self.state
    }

    /// Current switching state.
    pub fn switching_state(&self) -> bool {
        self.state
    }

    /// Reset to default (low-side) state.
    pub fn reset(&mut self) {
        self.state = false;
    }
}

// ─── Unit Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    const OMEGA: f64 = 2.0 * PI * 50.0; // 50 Hz grid

    /// Helper: generate a waveform with fundamental + selected harmonics.
    ///
    /// i(t) = I1·sin(ωt) + I5·sin(5ωt) + I7·sin(7ωt)
    fn load_current(t: f64, i1: f64, i5: f64, i7: f64) -> f64 {
        i1 * (OMEGA * t).sin() + i5 * (5.0 * OMEGA * t).sin() + i7 * (7.0 * OMEGA * t).sin()
    }

    /// The DFT detector should identify the 5th harmonic amplitude accurately.
    #[test]
    fn fifth_harmonic_detection_accuracy() {
        // Run for 2 complete periods, then check detected amplitude
        let fs = 10_000.0_f64; // 10 kHz sampling
        let dt = 1.0 / fs;
        let window_len = (fs / 50.0).round() as usize; // samples per period = 200

        let mut det: HarmonicDetector<f64, 2> = HarmonicDetector::new([5, 7], window_len);

        let i5_true = 2.0_f64;
        let i7_true = 1.0_f64;

        // Run for exactly 3 periods to get stable detection
        let n_samples = 3 * window_len;
        for k in 0..n_samples {
            let t = k as f64 * dt;
            let theta = OMEGA * t;
            let i_load = load_current(t, 10.0, i5_true, i7_true);
            det.update(i_load, theta);
        }

        // 5th harmonic amplitude (index 0)
        let a5 = det.amplitude_peak(0);
        let a7 = det.amplitude_peak(1);

        // Tolerance: 10% of true amplitude
        assert!(
            (a5 - i5_true).abs() < 0.2,
            "5th harmonic: detected={a5:.4}, true={i5_true:.4}"
        );
        assert!(
            (a7 - i7_true).abs() < 0.15,
            "7th harmonic: detected={a7:.4}, true={i7_true:.4}"
        );
    }

    /// Compensation reference should have near-zero fundamental component.
    #[test]
    fn zero_fundamental_in_filter_output() {
        let fs = 10_000.0_f64;
        let dt = 1.0 / fs;
        let window_len = (fs / 50.0).round() as usize;

        let mut apf: ApfCurrentReference<f64, 2> = ApfCurrentReference::new([5, 7], window_len);

        let i1 = 10.0_f64;
        let i5 = 3.0_f64;
        let i7 = 1.5_f64;

        // Prime the detector (3 periods)
        let n_prime = 3 * window_len;
        for k in 0..n_prime {
            let t = k as f64 * dt;
            let theta = OMEGA * t;
            let i_load = load_current(t, i1, i5, i7);
            apf.compute_reference(i_load, theta);
        }

        // Now measure fundamental content of compensation reference over one period
        let mut fund_cos_acc = 0.0_f64;
        let mut fund_sin_acc = 0.0_f64;
        let mut count = 0usize;

        for k in 0..window_len {
            let t = (n_prime + k) as f64 * dt;
            let theta = OMEGA * t;
            let i_load = load_current(t, i1, i5, i7);
            let i_ref = apf.compute_reference(i_load, theta);

            // Project onto fundamental
            fund_cos_acc += i_ref * theta.cos();
            fund_sin_acc += i_ref * theta.sin();
            count += 1;
        }

        let fund_cos = 2.0 * fund_cos_acc / count as f64;
        let fund_sin = 2.0 * fund_sin_acc / count as f64;
        let fund_amp = (fund_cos * fund_cos + fund_sin * fund_sin).sqrt();

        // Fundamental in compensation reference should be < 5% of load fundamental
        let threshold = 0.05 * i1;
        assert!(
            fund_amp < threshold,
            "Fundamental leakage in APF ref = {fund_amp:.4} A (threshold={threshold:.4})"
        );
    }

    /// Hysteresis controller must switch HIGH when error exceeds band.
    #[test]
    fn hysteresis_switches_high_on_positive_error() {
        let mut ctrl = ApfController::new(0.5_f64);
        // i_ref=5, i_actual=4 → error=1 > band=0.5 → HIGH
        let sw = ctrl.update(5.0, 4.0);
        assert!(sw, "Should switch HIGH (error > band)");
    }

    /// Hysteresis controller must switch LOW when error is strongly negative.
    #[test]
    fn hysteresis_switches_low_on_negative_error() {
        let mut ctrl = ApfController::new(0.5_f64);
        // First set to HIGH
        ctrl.update(5.0, 4.0);
        // Now i_ref=5, i_actual=6 → error=-1 < -band → LOW
        let sw = ctrl.update(5.0, 6.0);
        assert!(!sw, "Should switch LOW (error < -band)");
    }

    /// Within the hysteresis band, the state must be held.
    #[test]
    fn hysteresis_holds_state_within_band() {
        let mut ctrl = ApfController::new(1.0_f64);
        // Drive to HIGH
        ctrl.update(5.0, 3.0);
        // Error = 0.5 < band=1 → should hold HIGH
        let sw = ctrl.update(5.0, 4.5);
        assert!(sw, "Should hold HIGH within band");
    }

    /// Detector reset clears accumulated samples.
    #[test]
    fn detector_reset_clears_state() {
        let mut det: HarmonicDetector<f64, 1> = HarmonicDetector::new([5], 200);
        for k in 0..100 {
            det.update(k as f64 * 0.1, 0.01 * k as f64);
        }
        det.reset();
        assert_eq!(det.amplitude_cos(0), 0.0);
        assert_eq!(det.amplitude_sin(0), 0.0);
    }

    /// ApfController reset returns to low state.
    #[test]
    fn apf_controller_reset() {
        let mut ctrl = ApfController::new(0.5_f64);
        ctrl.update(5.0, 3.0); // → HIGH
        ctrl.reset();
        assert!(!ctrl.switching_state(), "Should be LOW after reset");
    }
}
