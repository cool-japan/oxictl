use crate::core::scalar::ControlScalar;

/// Back-EMF zero-crossing detector for sensorless BLDC commutation.
///
/// In sensorless BLDC drives, the undriven phase's back-EMF (BEMF)
/// is monitored. When it crosses the mid-supply voltage (Vdc/2),
/// commutation should occur 30° later (half of the 60° electrical period).
///
/// This detector implements:
///   1. Zero-crossing detection on the floating phase
///   2. 30° delay timer (one half-commutation interval)
///   3. Speed estimation from commutation timing
#[derive(Debug, Clone, Copy)]
pub struct BemfDetector<S: ControlScalar> {
    /// DC bus voltage (V).
    pub v_dc: S,
    /// Last seen BEMF voltage (for edge detection).
    prev_bemf: S,
    /// Timer accumulator for 30° delay.
    delay_acc: S,
    /// Required delay duration (updated from commutation timing).
    delay_target: S,
    /// Accumulated time since last zero crossing.
    period_acc: S,
    /// Estimated electrical period (s).
    pub electrical_period: S,
    /// Whether a commutation event is pending (30° delay elapsed).
    commutation_pending: bool,
    /// Initialized flag.
    initialized: bool,
}

impl<S: ControlScalar> BemfDetector<S> {
    /// Create BEMF detector.
    ///
    /// - `v_dc`: DC bus voltage
    /// - `initial_period`: initial guess for electrical period (s) to seed delay timer
    pub fn new(v_dc: S, initial_period: S) -> Self {
        let delay = initial_period / S::from_f64(12.0); // 30° = period/12
        Self {
            v_dc,
            prev_bemf: S::ZERO,
            delay_acc: S::ZERO,
            delay_target: delay,
            period_acc: S::ZERO,
            electrical_period: initial_period,
            commutation_pending: false,
            initialized: false,
        }
    }

    /// Process one sample of the floating phase BEMF voltage.
    ///
    /// - `bemf_voltage`: measured voltage on the undriven phase (0 to Vdc)
    /// - `dt`: sample time
    ///
    /// Returns `true` when a commutation event should occur.
    pub fn update(&mut self, bemf_voltage: S, dt: S) -> bool {
        let mid = self.v_dc * S::HALF;
        let crossed = if self.initialized {
            // Rising zero crossing: bemf crosses mid from below
            (self.prev_bemf < mid && bemf_voltage >= mid)
                || (self.prev_bemf > mid && bemf_voltage <= mid)
        } else {
            false
        };

        if crossed {
            // Update period estimate from timing
            if self.period_acc > S::ZERO {
                // Each zero crossing = 60° interval (6 per electrical cycle, but we
                // detect alternating rising/falling, so 2 per 60° sector = period/6)
                // Smooth with exponential filter
                let measured = self.period_acc * S::from_f64(6.0);
                let alpha = S::from_f64(0.3);
                self.electrical_period =
                    alpha * measured + (S::ONE - alpha) * self.electrical_period;
                self.delay_target = self.electrical_period / S::from_f64(12.0); // 30°
            }
            self.period_acc = S::ZERO;
            self.delay_acc = S::ZERO;
            self.commutation_pending = true;
        }

        self.period_acc += dt;
        self.prev_bemf = bemf_voltage;
        self.initialized = true;

        // Check if 30° delay has elapsed
        if self.commutation_pending {
            self.delay_acc += dt;
            if self.delay_acc >= self.delay_target {
                self.commutation_pending = false;
                self.delay_acc = S::ZERO;
                return true;
            }
        }

        false
    }

    /// Estimated electrical speed (rad/s).
    pub fn omega_e(&self) -> S {
        if self.electrical_period > S::ZERO {
            S::TWO * S::PI / self.electrical_period
        } else {
            S::ZERO
        }
    }

    pub fn reset(&mut self) {
        self.prev_bemf = S::ZERO;
        self.delay_acc = S::ZERO;
        self.period_acc = S::ZERO;
        self.commutation_pending = false;
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_zero_crossing_and_commutates() {
        let mut det = BemfDetector::new(48.0_f64, 0.01);
        let dt = 0.0001;
        let mut commutations = 0;
        // Simulate sinusoidal BEMF at 100 Hz (period = 0.01s)
        let freq = 100.0_f64;
        let steps = (0.1 / dt) as usize; // 0.1 seconds
        for step in 0..steps {
            let t = step as f64 * dt;
            let bemf = 24.0 + 20.0 * (2.0 * core::f64::consts::PI * freq * t).sin();
            if det.update(bemf, dt) {
                commutations += 1;
            }
        }
        // Should detect roughly 2*freq*0.1 = 20 crossings, each → commutation after 30°
        assert!(
            commutations > 5,
            "Expected commutations, got {}",
            commutations
        );
    }

    #[test]
    fn no_commutation_with_dc_signal() {
        let mut det = BemfDetector::new(48.0_f64, 0.01);
        let dt = 0.001;
        let mut commutations = 0;
        for _ in 0..100 {
            if det.update(30.0, dt) {
                // Constant above mid (24V): no crossing
                commutations += 1;
            }
        }
        assert_eq!(commutations, 0);
    }

    #[test]
    fn omega_estimate_nonzero_after_detection() {
        let mut det = BemfDetector::new(48.0_f64, 0.005);
        let dt = 0.0001;
        let freq = 200.0_f64;
        for step in 0..2000 {
            let t = step as f64 * dt;
            let bemf = 24.0 + 20.0 * (2.0 * core::f64::consts::PI * freq * t).sin();
            det.update(bemf, dt);
        }
        let omega = det.omega_e();
        assert!(omega > 100.0, "omega={:.1}", omega);
    }
}
