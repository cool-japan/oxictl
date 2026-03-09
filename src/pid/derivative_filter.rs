use crate::core::scalar::ControlScalar;

/// First-order IIR low-pass filter for the derivative term.
/// Implements the incomplete derivative: D(s) = Kd * s / (1 + tau_f * s)
/// Discretized using backward Euler: y[n] = alpha * y[n-1] + (1-alpha) * x[n]
/// where alpha = tau_f / (tau_f + dt)
#[derive(Debug, Clone, Copy)]
pub struct DerivativeFilter<S: ControlScalar> {
    /// Filter time constant (seconds). Larger = more filtering.
    tau_f: S,
    /// Previous filtered output.
    prev_output: S,
    /// Whether the filter has been initialized.
    initialized: bool,
}

impl<S: ControlScalar> DerivativeFilter<S> {
    /// Create a new derivative filter.
    /// `tau_f` is the filter time constant. Typical: tau_f = Kd / (N * Kp) where N = 8..20.
    pub fn new(tau_f: S) -> Self {
        Self {
            tau_f,
            prev_output: S::ZERO,
            initialized: false,
        }
    }

    /// Create from derivative gain and filter coefficient N.
    /// tau_f = Kd / (N * Kp). If Kp is zero, uses a default tau_f.
    pub fn from_gains(kd: S, kp: S, n: S) -> Self {
        let denominator = n * kp;
        let tau_f = if denominator.abs() > S::EPSILON {
            kd / denominator
        } else {
            kd / n
        };
        Self::new(tau_f)
    }

    /// Apply the filter to a raw derivative value.
    pub fn apply(&mut self, raw_derivative: S, dt: S) -> S {
        if !self.initialized {
            self.prev_output = raw_derivative;
            self.initialized = true;
            return raw_derivative;
        }

        if dt <= S::ZERO || self.tau_f <= S::ZERO {
            self.prev_output = raw_derivative;
            return raw_derivative;
        }

        let alpha = self.tau_f / (self.tau_f + dt);
        let filtered = alpha * self.prev_output + (S::ONE - alpha) * raw_derivative;
        self.prev_output = filtered;
        filtered
    }

    pub fn reset(&mut self) {
        self.prev_output = S::ZERO;
        self.initialized = false;
    }

    pub fn tau_f(&self) -> S {
        self.tau_f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_value_passthrough() {
        let mut filt = DerivativeFilter::<f64>::new(0.1);
        let out = filt.apply(5.0, 0.01);
        assert_eq!(out, 5.0);
    }

    #[test]
    fn filter_attenuates_step_change() {
        let mut filt = DerivativeFilter::<f64>::new(0.1);
        filt.apply(0.0, 0.01);
        let out = filt.apply(10.0, 0.01);
        // alpha = 0.1/(0.1+0.01) ≈ 0.909
        // filtered = 0.909*0 + 0.091*10 ≈ 0.909
        assert!(out < 2.0, "Filter should attenuate: got {}", out);
        assert!(out > 0.0, "Filter should be positive: got {}", out);
    }

    #[test]
    fn filter_converges_to_constant() {
        let mut filt = DerivativeFilter::<f64>::new(0.01);
        filt.apply(0.0, 0.001);
        for _ in 0..1000 {
            filt.apply(5.0, 0.001);
        }
        let out = filt.apply(5.0, 0.001);
        assert!(
            (out - 5.0).abs() < 0.01,
            "Should converge to 5.0, got {}",
            out
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut filt = DerivativeFilter::<f64>::new(0.1);
        filt.apply(100.0, 0.01);
        filt.reset();
        let out = filt.apply(0.0, 0.01);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn from_gains_computes_tau() {
        let filt = DerivativeFilter::<f64>::from_gains(1.0, 2.0, 10.0);
        assert!((filt.tau_f() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn noise_attenuation() {
        let mut filt = DerivativeFilter::<f64>::new(0.05);
        let dt = 0.001;
        // Feed alternating noise
        filt.apply(0.0, dt);
        let mut max_output = 0.0_f64;
        for i in 0..100 {
            let noisy = if i % 2 == 0 { 10.0 } else { -10.0 };
            let out = filt.apply(noisy, dt);
            max_output = max_output.max(out.abs());
        }
        // Filtered output should be much smaller than input amplitude
        assert!(
            max_output < 10.0,
            "Filter should attenuate noise: max={}",
            max_output
        );
    }
}
