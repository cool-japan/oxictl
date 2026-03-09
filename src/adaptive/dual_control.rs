//! Dual control: balances exploitation (certainty-equivalence) and exploration (probing).
//!
//! A dual controller blends a certainty-equivalence (CE) proportional law with a
//! deterministic probing signal. The probing weight increases when parameter
//! uncertainty (trace of covariance P) exceeds a threshold.

use crate::core::scalar::ControlScalar;

/// Dual controller combining CE control and probing for active parameter learning.
///
/// Control law:
///   u = u_CE + α_probe * u_probe
///
/// where α_probe ∈ [0, 1] is the probing weight determined from trace(P).
#[derive(Debug, Clone, Copy)]
pub struct DualController<S: ControlScalar> {
    /// Certainty-equivalence proportional gain.
    pub k_ce: S,
    /// Probing signal amplitude.
    pub probe_amplitude: S,
    /// Current trace of parameter covariance (uncertainty measure).
    pub trace_p: S,
    /// Uncertainty threshold above which probing is activated.
    pub probe_threshold: S,
    /// Low-pass forgetting factor for uncertainty tracking (0 < alpha ≤ 1).
    pub alpha: S,
    /// Counter for square-wave probing.
    probe_counter: u32,
    /// Half-period of the probing square wave (steps).
    probe_period: u32,
}

impl<S: ControlScalar> DualController<S> {
    /// Create a dual controller.
    ///
    /// # Arguments
    /// - `k_ce`: CE proportional gain (positive)
    /// - `probe_amplitude`: amplitude of probing square wave
    /// - `probe_threshold`: trace(P) threshold above which probing activates
    /// - `probe_period`: half-period of probing square wave in time steps
    pub fn new(k_ce: S, probe_amplitude: S, probe_threshold: S, probe_period: u32) -> Self {
        Self {
            k_ce,
            probe_amplitude,
            trace_p: S::ZERO,
            probe_threshold,
            alpha: S::from_f64(0.99),
            probe_counter: 0,
            probe_period,
        }
    }

    /// Probing weight α_probe = clamp(trace_p / probe_threshold, 0, 1).
    ///
    /// Returns 0 when uncertainty is below threshold and 1 when at/above threshold.
    pub fn alpha_probing(&self) -> S {
        let ratio = self.trace_p / (self.probe_threshold + S::EPSILON);
        ratio.clamp_val(S::ZERO, S::ONE)
    }

    /// Update the uncertainty estimate from an external RLS estimator.
    ///
    /// Uses exponential smoothing: trace_p ← alpha * trace_p + (1-alpha) * new_trace
    pub fn update_uncertainty(&mut self, trace_p: S) {
        self.trace_p = self.alpha * self.trace_p + (S::ONE - self.alpha) * trace_p;
    }

    /// Certainty-equivalence proportional feedback: u_CE = k_ce * (r - y).
    pub fn ce_control(&self, r: S, y: S) -> S {
        self.k_ce * (r - y)
    }

    /// Probing square wave: alternates between +amplitude and -amplitude every probe_period steps.
    pub fn probing_signal(&mut self) -> S {
        let half = self.probe_period;
        let period = half * 2;
        // Sample the sign first, then advance counter
        let sign = if self.probe_counter < half {
            self.probe_amplitude
        } else {
            -self.probe_amplitude
        };
        self.probe_counter += 1;
        if self.probe_counter >= period {
            self.probe_counter = 0;
        }
        sign
    }

    /// Compute dual control signal u = u_CE + α_probe * u_probe.
    ///
    /// - `r`: reference (set-point)
    /// - `y`: current measurement
    pub fn update(&mut self, r: S, y: S) -> S {
        let u_ce = self.ce_control(r, y);
        let alpha = self.alpha_probing();
        let u_probe = self.probing_signal();
        u_ce + alpha * u_probe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_probing_when_certain() {
        let mut ctrl = DualController::new(1.0_f64, 5.0_f64, 10.0_f64, 4);
        // trace_p = 0, so alpha_probing = 0 → output = u_CE only
        let u = ctrl.update(1.0, 0.0);
        // CE: k_ce * (r - y) = 1.0 * 1.0 = 1.0; probe weighted by 0
        assert!((u - 1.0).abs() < 1e-9, "u={u}");
    }

    #[test]
    fn test_full_probing_when_uncertain() {
        let mut ctrl = DualController::new(0.0_f64, 5.0_f64, 1.0_f64, 4);
        // Set trace_p much higher than threshold
        ctrl.trace_p = 100.0;
        let u = ctrl.update(0.0, 0.0);
        // CE = 0; probe weight = 1; square wave = +5 at first step
        assert!((u.abs() - 5.0).abs() < 1e-9, "u={u}");
    }

    #[test]
    fn test_probing_square_wave_period() {
        let mut ctrl = DualController::new(0.0_f64, 1.0_f64, 0.0_f64, 3);
        ctrl.trace_p = 1.0; // saturate alpha at 1
        let signals: Vec<f64> = (0..6).map(|_| ctrl.update(0.0, 0.0)).collect();
        // First 3 steps: +1, next 3: -1
        assert!(signals[0] > 0.0);
        assert!(signals[1] > 0.0);
        assert!(signals[2] > 0.0);
        assert!(signals[3] < 0.0);
        assert!(signals[4] < 0.0);
        assert!(signals[5] < 0.0);
    }

    #[test]
    fn test_update_uncertainty_smoothing() {
        let mut ctrl = DualController::new(1.0_f64, 1.0_f64, 1.0_f64, 4);
        ctrl.alpha = 0.9;
        ctrl.update_uncertainty(10.0);
        // trace_p should be (1-0.9)*10 = 1.0
        assert!(
            (ctrl.trace_p - 1.0).abs() < 1e-9,
            "trace_p={}",
            ctrl.trace_p
        );
    }

    #[test]
    fn test_alpha_probing_clamped() {
        let mut ctrl = DualController::new(1.0_f64, 1.0_f64, 5.0_f64, 4);
        ctrl.trace_p = 0.0;
        assert_eq!(ctrl.alpha_probing(), 0.0);
        ctrl.trace_p = 100.0;
        assert_eq!(ctrl.alpha_probing(), 1.0);
    }
}
