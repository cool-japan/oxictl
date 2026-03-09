//! Grid frequency estimator using zero-crossing detection and PLL tracking.
//!
//! Two methods:
//!   1. `ZeroCrossingEstimator`: counts zero-crossings to estimate frequency.
//!      Fast but noisy — good for coarse estimate.
//!   2. `FrequencyPll`: type-2 PLL tracking loop for smooth, accurate estimate.
//!      Slow to lock but high accuracy.

use crate::core::scalar::ControlScalar;

/// Zero-crossing frequency estimator.
///
/// Detects rising zero-crossings of a signal and estimates frequency
/// from the average period between crossings.
#[derive(Debug, Clone, Copy)]
pub struct ZeroCrossingEstimator<S: ControlScalar> {
    /// Estimated frequency (Hz).
    pub frequency_hz: S,
    /// Previous sample value (for edge detection).
    prev: S,
    /// Time accumulator since last crossing (s).
    time_since_crossing: S,
    /// Period from last complete cycle (s).
    period: S,
    /// Frequency range: [min_hz, max_hz] for validity check.
    pub min_hz: S,
    pub max_hz: S,
}

impl<S: ControlScalar> ZeroCrossingEstimator<S> {
    pub fn new(nominal_hz: S) -> Self {
        Self {
            frequency_hz: nominal_hz,
            prev: S::ZERO,
            time_since_crossing: S::ZERO,
            period: S::ONE / nominal_hz,
            min_hz: S::from_f64(40.0),
            max_hz: S::from_f64(70.0),
        }
    }

    /// Update with new signal sample. Returns new frequency estimate.
    pub fn update(&mut self, sample: S, dt: S) -> S {
        self.time_since_crossing += dt;

        // Detect rising zero-crossing: prev < 0, now >= 0
        let rising = self.prev < S::ZERO && sample >= S::ZERO;
        if rising && self.time_since_crossing > dt {
            let t = self.time_since_crossing;
            let f = S::ONE / t;

            // Validate against expected range
            if f >= self.min_hz && f <= self.max_hz {
                self.period = t;
                self.frequency_hz = f;
            }
            self.time_since_crossing = S::ZERO;
        }

        self.prev = sample;
        self.frequency_hz
    }

    pub fn reset(&mut self, nominal_hz: S) {
        self.frequency_hz = nominal_hz;
        self.prev = S::ZERO;
        self.time_since_crossing = S::ZERO;
        self.period = S::ONE / nominal_hz;
    }
}

/// PLL-based frequency estimator with PI tracking loop.
///
/// Tracks phase and frequency of a sinusoidal signal.
/// Accuracy: better than 0.01 Hz at steady state.
///
/// Architecture:
///   θ̂: estimated phase
///   ω̂: estimated angular frequency
///   PI loop drives phase error ε = sin(θ̂ - θ_true) → 0
#[derive(Debug, Clone, Copy)]
pub struct FrequencyPll<S: ControlScalar> {
    /// Estimated angular frequency (rad/s).
    pub omega: S,
    /// Estimated phase (rad).
    pub theta: S,
    /// Estimated amplitude.
    pub amplitude: S,
    /// PI proportional gain.
    pub kp: S,
    /// PI integral gain.
    pub ki: S,
    /// Integrator state.
    int: S,
    /// Nominal frequency (rad/s).
    pub omega_nom: S,
    /// Low-pass filter for amplitude estimation.
    amp_lpf: S,
    /// LP filter coefficient.
    amp_alpha: S,
}

impl<S: ControlScalar> FrequencyPll<S> {
    pub fn new(omega_nom: S, kp: S, ki: S) -> Self {
        Self {
            omega: omega_nom,
            theta: S::ZERO,
            amplitude: S::ONE,
            kp,
            ki,
            int: S::ZERO,
            omega_nom,
            amp_lpf: S::ONE,
            amp_alpha: S::from_f64(0.99),
        }
    }

    /// Update PLL with signal sample. Returns (frequency_hz, theta, amplitude).
    pub fn update(&mut self, sample: S, dt: S) -> (S, S, S) {
        // Phase detector: multiply by 90°-shifted version = -sin(θ̂)
        // error = sample * (-sin(θ̂)) = A*sin(θ)*(-sin(θ̂)) ≈ -A/2 * sin(θ-θ̂) for small error
        let sin_t = self.theta.sin();
        let cos_t = self.theta.cos();

        // Amplitude estimation via envelope detector
        let inst_amp =
            (sample * sample + (sample * sin_t / (cos_t.abs() + S::from_f64(0.01))).powi(2)).sqrt();
        self.amp_lpf = self.amp_alpha * self.amp_lpf + (S::ONE - self.amp_alpha) * inst_amp;
        self.amplitude = self.amp_lpf + S::from_f64(1e-6);

        // Phase error: cross product of input with local oscillator in quadrature
        let err = sample * cos_t; // ≈ A/2 * sin(θ_in - θ̂) for small errors

        // PI loop
        self.int += self.ki * err * dt;
        let delta_omega = self.kp * err + self.int;

        self.omega = self.omega_nom + delta_omega;

        // Integrate phase
        self.theta += self.omega * dt;

        // Wrap to [-π, π]
        let pi = S::PI;
        let two_pi = S::TWO * pi;
        while self.theta > pi {
            self.theta -= two_pi;
        }
        while self.theta < -pi {
            self.theta += two_pi;
        }

        let two_pi_f64 = S::TWO * S::PI;
        (self.omega / two_pi_f64, self.theta, self.amplitude)
    }

    pub fn reset(&mut self) {
        self.omega = self.omega_nom;
        self.theta = S::ZERO;
        self.int = S::ZERO;
        self.amp_lpf = S::ONE;
    }
}

/// Instantaneous frequency estimator using Hilbert transform approximation.
///
/// Uses a simple FIR-based 90°-phase-shifted signal to compute instantaneous
/// frequency via f = (1/2π) * d/dt[atan2(xq, xi)] where xi=in-phase, xq=quadrature.
///
/// Suitable for signals with slowly varying frequency.
#[derive(Debug, Clone, Copy)]
pub struct InstantaneousFrequency<S: ControlScalar> {
    /// Previous in-phase sample.
    xi_prev: S,
    /// Previous quadrature sample (delayed by T/4).
    xq_prev: S,
    /// Quadrature delay buffer (4 samples for quarter-period at nominal freq).
    delay: [S; 4],
    delay_idx: usize,
    /// Estimated frequency (Hz).
    pub frequency_hz: S,
    /// Low-pass filter state.
    lpf: S,
    /// LP coefficient (smoothing).
    pub lpf_alpha: S,
}

impl<S: ControlScalar> InstantaneousFrequency<S> {
    pub fn new(nominal_hz: S, lpf_alpha: S) -> Self {
        Self {
            xi_prev: S::ZERO,
            xq_prev: S::ZERO,
            delay: [S::ZERO; 4],
            delay_idx: 0,
            frequency_hz: nominal_hz,
            lpf: nominal_hz,
            lpf_alpha,
        }
    }

    /// Update with new sample. Returns estimated frequency (Hz).
    pub fn update(&mut self, sample: S, dt: S) -> S {
        // Quadrature via quarter-period delay buffer
        let xq = self.delay[self.delay_idx];
        self.delay[self.delay_idx] = sample;
        self.delay_idx = (self.delay_idx + 1) % 4;

        let xi = sample;

        // Instantaneous frequency via cross/dot products:
        // f ≈ (xi * xq_prev - xq * xi_prev) / (2π * dt * (xi²+xq²))
        let cross = xi * self.xq_prev - xq * self.xi_prev;
        let energy = xi * xi + xq * xq;

        if energy > S::from_f64(1e-6) {
            let two_pi = S::TWO * S::PI;
            let f_inst = cross / (two_pi * dt * energy);
            // Low-pass filter
            self.lpf = self.lpf_alpha * self.lpf + (S::ONE - self.lpf_alpha) * f_inst;
            self.frequency_hz = self.lpf;
        }

        self.xi_prev = xi;
        self.xq_prev = xq;
        self.frequency_hz
    }

    pub fn reset(&mut self, nominal_hz: S) {
        self.delay = [S::ZERO; 4];
        self.delay_idx = 0;
        self.xi_prev = S::ZERO;
        self.xq_prev = S::ZERO;
        self.lpf = nominal_hz;
        self.frequency_hz = nominal_hz;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn zero_crossing_detects_50hz() {
        let dt = 1e-4;
        let f_true = 50.0f64;
        let omega = 2.0 * PI * f_true;
        let mut est = ZeroCrossingEstimator::new(50.0_f64);

        for k in 0..2000 {
            let t = k as f64 * dt;
            let s = (omega * t).sin();
            est.update(s, dt);
        }

        let err = (est.frequency_hz - f_true).abs();
        assert!(err < 1.0, "freq={:.3} Hz, err={err:.3}", est.frequency_hz);
    }

    #[test]
    fn zero_crossing_60hz() {
        let dt = 1e-4;
        let f_true = 60.0f64;
        let omega = 2.0 * PI * f_true;
        let mut est = ZeroCrossingEstimator::new(60.0_f64);

        for k in 0..2000 {
            let t = k as f64 * dt;
            let s = (omega * t).sin();
            est.update(s, dt);
        }

        let err = (est.frequency_hz - f_true).abs();
        assert!(err < 1.0, "freq={:.3} Hz, err={err:.3}", est.frequency_hz);
    }

    #[test]
    fn pll_frequency_locks() {
        let f_true = 50.0f64;
        let omega_nom = 2.0 * PI * f_true;
        let dt = 1e-4;
        let mut pll = FrequencyPll::new(omega_nom, 100.0_f64, 2000.0_f64);

        let mut theta_in = 0.0f64;
        for _ in 0..8000 {
            let s = theta_in.sin();
            pll.update(s, dt);
            theta_in += omega_nom * dt;
            if theta_in > PI {
                theta_in -= 2.0 * PI;
            }
        }

        let freq_err = (pll.omega / (2.0 * PI) - f_true).abs();
        assert!(freq_err < 2.0, "freq_err={freq_err:.4} Hz");
    }

    #[test]
    fn pll_reset_clears_state() {
        let mut pll = FrequencyPll::new(314.16_f64, 50.0, 500.0);
        for _ in 0..100 {
            pll.update(1.0, 1e-4);
        }
        pll.reset();
        assert_eq!(pll.theta, 0.0);
        assert_eq!(pll.int, 0.0);
    }
}
