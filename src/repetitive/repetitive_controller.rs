//! Repetitive controllers for periodic disturbance rejection.
//!
//! Implements the plug-in repetitive controller (RC) and modified repetitive
//! controller (MRC) based on the internal model principle. These controllers
//! learn to cancel periodic disturbances of known period N samples.
//!
//! # Theory
//! The internal model u_r[k] = Q(z)*u_r[k-N] + L(z)*e[k] stores one full
//! period of the repetitive signal. The robustness filter Q (gain q < 1)
//! ensures stability in the presence of model uncertainty. The learning gain
//! kr determines convergence speed.

use crate::core::scalar::ControlScalar;

/// Errors returned by repetitive controllers.
#[derive(Debug, Clone, PartialEq)]
pub enum RepetitiveError {
    /// A parameter was outside its valid range.
    InvalidParameter,
    /// Period N is zero (would cause division by zero).
    ZeroPeriod,
}

impl core::fmt::Display for RepetitiveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidParameter => write!(f, "Invalid parameter value"),
            Self::ZeroPeriod => write!(f, "Period N must be non-zero"),
        }
    }
}

/// Plug-in repetitive controller for rejecting periodic disturbances.
///
/// Architecture (discrete, N samples per period):
/// ```text
/// u_r[k] = q * u_r[k-N] + kr * e[k]
/// ```
/// where `q` is the robustness filter gain and `kr` is the learning gain.
///
/// The controller stores one full period in a circular buffer and updates
/// the repetitive signal at each sample.
///
/// # Type Parameters
/// - `S`: Scalar type implementing [`ControlScalar`]
/// - `N`: Number of samples per period (compile-time constant)
///
/// # Example
/// ```
/// use oxictl::repetitive::RepetitiveController;
///
/// let mut rc = RepetitiveController::<f64, 100>::new(0.95, 0.5).unwrap();
/// let u_r = rc.update(0.1).unwrap();
/// ```
pub struct RepetitiveController<S: ControlScalar, const N: usize> {
    /// Circular buffer storing one period of the repetitive signal.
    buffer: [S; N],
    /// Current write position in the circular buffer.
    head: usize,
    /// Robustness filter gain (0 < q < 1).
    q: S,
    /// Learning gain (kr > 0).
    kr: S,
    /// Number of update calls since last reset.
    trial: usize,
    /// Exponential moving average of squared error (proxy for RMS).
    rms_error: S,
}

impl<S: ControlScalar, const N: usize> RepetitiveController<S, N> {
    /// Construct a new repetitive controller.
    ///
    /// # Parameters
    /// - `q`: Robustness filter gain, must satisfy 0 < q < 1 (typically 0.95)
    /// - `kr`: Learning gain, must be positive
    ///
    /// # Errors
    /// Returns [`RepetitiveError::InvalidParameter`] if parameters are out of range.
    /// Returns [`RepetitiveError::ZeroPeriod`] if N == 0.
    pub fn new(q: S, kr: S) -> Result<Self, RepetitiveError> {
        if N == 0 {
            return Err(RepetitiveError::ZeroPeriod);
        }
        if q <= S::ZERO || q >= S::ONE {
            return Err(RepetitiveError::InvalidParameter);
        }
        if kr <= S::ZERO {
            return Err(RepetitiveError::InvalidParameter);
        }
        Ok(Self {
            buffer: core::array::from_fn(|_| S::ZERO),
            head: 0,
            q,
            kr,
            trial: 0,
            rms_error: S::ZERO,
        })
    }

    /// Update the repetitive controller with the current error signal.
    ///
    /// Implements the plug-in update law:
    /// ```text
    /// u_r[k] = q * u_r[k-N] + kr * e[k]
    /// ```
    ///
    /// # Parameters
    /// - `error`: Current tracking error e[k] = r[k] - y[k]
    ///
    /// # Returns
    /// The repetitive control signal u_r[k].
    pub fn update(&mut self, error: S) -> Result<S, RepetitiveError> {
        // Value from exactly one period ago
        let u_prev = self.buffer[self.head];

        // Plug-in repetitive update law
        let u_r = self.q * u_prev + self.kr * error;

        // Store back into circular buffer
        self.buffer[self.head] = u_r;

        // Advance circular buffer pointer
        self.head = (self.head + 1) % N;

        // Update EMA of squared error as convergence metric
        let alpha = S::from_f64(0.01);
        let one_minus_alpha = S::from_f64(0.99);
        self.rms_error = one_minus_alpha * self.rms_error + alpha * error * error;

        self.trial += 1;

        Ok(u_r)
    }

    /// Return the most recently computed repetitive signal.
    pub fn repetitive_signal(&self) -> S {
        // head points to the next write location; last written is head-1 (wrapping)
        let last = (self.head + N - 1) % N;
        self.buffer[last]
    }

    /// Reset the controller to its initial state.
    pub fn reset(&mut self) {
        for v in self.buffer.iter_mut() {
            *v = S::ZERO;
        }
        self.head = 0;
        self.trial = 0;
        self.rms_error = S::ZERO;
    }

    /// Return the current RMS convergence metric (EMA of squared error).
    pub fn rms_convergence(&self) -> S {
        self.rms_error
    }

    /// Return the total number of update calls since last reset.
    pub fn trial_count(&self) -> usize {
        self.trial
    }

    /// Return the robustness filter gain q.
    pub fn q(&self) -> S {
        self.q
    }

    /// Return the learning gain kr.
    pub fn kr(&self) -> S {
        self.kr
    }
}

/// Modified repetitive controller with zero-phase 3-tap FIR robustness filter.
///
/// The zero-phase FIR filter [q1, q0, q1] (symmetric, applied to the circular
/// buffer) provides better frequency shaping than a simple gain, allowing the
/// controller to suppress high-frequency amplification while maintaining
/// robustness at lower harmonics.
///
/// Update law:
/// ```text
/// filtered[k] = q1*u_r[k-N-1] + q0*u_r[k-N] + q1*u_r[k-N+1]
/// u_r[k] = filtered[k] + kr * e[k]
/// ```
///
/// # Stability condition
/// `q0 + 2*q1 < 1` must hold for the filter gain to remain below unity.
///
/// # Type Parameters
/// - `S`: Scalar type implementing [`ControlScalar`]
/// - `N`: Number of samples per period (compile-time constant)
pub struct ModifiedRepetitiveController<S: ControlScalar, const N: usize> {
    /// Circular buffer storing one period of the repetitive signal.
    buffer: [S; N],
    /// Auxiliary buffer used for zero-phase FIR computation.
    q_buf: [S; N],
    /// Current write position in the circular buffer.
    head: usize,
    /// Center FIR coefficient (q0).
    q0: S,
    /// Side FIR coefficient (q1); filter is symmetric: [q1, q0, q1].
    q1: S,
    /// Learning gain (kr > 0).
    kr: S,
}

impl<S: ControlScalar, const N: usize> ModifiedRepetitiveController<S, N> {
    /// Construct a new modified repetitive controller.
    ///
    /// # Parameters
    /// - `q0`: Center FIR coefficient
    /// - `q1`: Side FIR coefficient (filter = [q1, q0, q1])
    /// - `kr`: Learning gain, must be positive
    ///
    /// # Stability
    /// Requires `q0 + 2*q1 < 1` for the filter gain to remain below unity.
    ///
    /// # Errors
    /// Returns [`RepetitiveError::InvalidParameter`] if stability condition or
    /// gain positivity constraint is violated.
    /// Returns [`RepetitiveError::ZeroPeriod`] if N == 0.
    pub fn new(q0: S, q1: S, kr: S) -> Result<Self, RepetitiveError> {
        if N == 0 {
            return Err(RepetitiveError::ZeroPeriod);
        }
        // Stability: filter DC gain must be < 1
        if q0 + S::TWO * q1 >= S::ONE {
            return Err(RepetitiveError::InvalidParameter);
        }
        if kr <= S::ZERO {
            return Err(RepetitiveError::InvalidParameter);
        }
        // q0 and q1 should be non-negative for a proper low-pass shape
        if q0 < S::ZERO || q1 < S::ZERO {
            return Err(RepetitiveError::InvalidParameter);
        }
        Ok(Self {
            buffer: core::array::from_fn(|_| S::ZERO),
            q_buf: core::array::from_fn(|_| S::ZERO),
            head: 0,
            q0,
            q1,
            kr,
        })
    }

    /// Update the modified repetitive controller with the current error signal.
    ///
    /// Applies the symmetric 3-tap FIR filter to the previous period's signal,
    /// then adds the learning correction.
    ///
    /// # Parameters
    /// - `error`: Current tracking error e[k] = r[k] - y[k]
    ///
    /// # Returns
    /// The repetitive control signal u_r[k].
    pub fn update(&mut self, error: S) -> Result<S, RepetitiveError> {
        // Indices in the circular buffer for FIR taps:
        // center: current head (one full period ago)
        // prev: one sample before center in the buffer (head-1, wrapping)
        // next: one sample after center in the buffer (head+1, wrapping)
        let center_idx = self.head;
        let prev_idx = (self.head + N - 1) % N;
        let next_idx = (self.head + 1) % N;

        let center = self.buffer[center_idx];
        let prev = self.buffer[prev_idx];
        let next = self.buffer[next_idx];

        // Zero-phase 3-tap symmetric FIR: [q1, q0, q1]
        let filtered = self.q1 * prev + self.q0 * center + self.q1 * next;

        // Plug-in learning update
        let u_r = filtered + self.kr * error;

        // Store in q_buf for reference; write u_r into main buffer
        self.q_buf[self.head] = filtered;
        self.buffer[self.head] = u_r;

        // Advance pointer
        self.head = (self.head + 1) % N;

        Ok(u_r)
    }

    /// Return the most recently computed repetitive signal.
    pub fn repetitive_signal(&self) -> S {
        let last = (self.head + N - 1) % N;
        self.buffer[last]
    }

    /// Return the most recently computed filtered signal (Q-filter output).
    pub fn filtered_signal(&self) -> S {
        let last = (self.head + N - 1) % N;
        self.q_buf[last]
    }

    /// Reset the controller to its initial state.
    pub fn reset(&mut self) {
        for v in self.buffer.iter_mut() {
            *v = S::ZERO;
        }
        for v in self.q_buf.iter_mut() {
            *v = S::ZERO;
        }
        self.head = 0;
    }

    /// Return the FIR center coefficient q0.
    pub fn q0(&self) -> S {
        self.q0
    }

    /// Return the FIR side coefficient q1.
    pub fn q1(&self) -> S {
        self.q1
    }

    /// Return the learning gain kr.
    pub fn kr(&self) -> S {
        self.kr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that a zero error input leaves the buffer at zero.
    #[test]
    fn zero_error_buffer_stays_zero() {
        let mut rc = RepetitiveController::<f64, 10>::new(0.95, 0.5).expect("valid parameters");
        for _ in 0..20 {
            let u_r = rc.update(0.0).expect("update ok");
            assert!(
                u_r.abs() < 1e-12,
                "Expected zero output for zero error, got {u_r}"
            );
        }
        assert!(
            rc.repetitive_signal().abs() < 1e-12,
            "Buffer should remain zero"
        );
    }

    /// Test sinusoidal convergence: RMS error metric should not blow up over time.
    #[test]
    fn sinusoidal_convergence() {
        const PERIOD: usize = 20;
        let mut rc = RepetitiveController::<f64, PERIOD>::new(0.95, 0.3).expect("valid parameters");

        // Simulate 10 periods of sinusoidal error
        let mut last_rms = f64::MAX;
        let mut converging = false;
        for k in 0..(10 * PERIOD) {
            let t = k as f64 * 2.0 * core::f64::consts::PI / PERIOD as f64;
            // Simulate residual error that decreases as RC learns: scale by e^{-trial/100}
            let scale = (-((k as f64) / (5.0 * PERIOD as f64))).exp();
            let error = scale * t.sin();
            let _ = rc.update(error).expect("update ok");
            let rms = rc.rms_convergence();
            if k > PERIOD {
                // After one full period, check that RMS is tracked
                if rms < last_rms {
                    converging = true;
                }
                last_rms = rms;
            }
        }
        // With decaying error, the EMA should eventually be very small
        let final_rms = rc.rms_convergence();
        assert!(
            final_rms < 0.1,
            "Expected convergence, RMS = {final_rms}, converging = {converging}"
        );
    }

    /// Test that q affects convergence: lower q → faster forgetting of past.
    #[test]
    fn q_effect_on_convergence_speed() {
        const PERIOD: usize = 16;
        let mut rc_high_q =
            RepetitiveController::<f64, PERIOD>::new(0.99, 0.5).expect("valid high q");
        let mut rc_low_q = RepetitiveController::<f64, PERIOD>::new(0.5, 0.5).expect("valid low q");

        // Apply the same constant error for several periods
        for _ in 0..(3 * PERIOD) {
            let _ = rc_high_q.update(1.0).expect("update ok");
            let _ = rc_low_q.update(1.0).expect("update ok");
        }

        let sig_high = rc_high_q.repetitive_signal().abs();
        let sig_low = rc_low_q.repetitive_signal().abs();

        // With high q (close to 1), the buffer accumulates more signal (slow forgetting)
        // With low q (0.5), the buffer accumulates less (aggressive forgetting)
        // Both should be positive with constant positive error
        assert!(
            sig_high > 0.0,
            "High q controller should produce nonzero signal"
        );
        assert!(
            sig_low > 0.0,
            "Low q controller should produce nonzero signal"
        );
        // High q retains more of the accumulated past → larger magnitude
        assert!(
            sig_high > sig_low,
            "High q ({sig_high:.4}) should accumulate more than low q ({sig_low:.4})"
        );
    }

    /// Test that kr=0 is rejected.
    #[test]
    fn kr_validation_zero() {
        let result = RepetitiveController::<f64, 10>::new(0.95, 0.0);
        assert!(
            matches!(result, Err(RepetitiveError::InvalidParameter)),
            "kr=0.0 should be rejected"
        );
    }

    /// Test that kr < 0 is rejected.
    #[test]
    fn kr_validation_negative() {
        let result = RepetitiveController::<f64, 10>::new(0.95, -0.1);
        assert!(
            matches!(result, Err(RepetitiveError::InvalidParameter)),
            "kr<0 should be rejected"
        );
    }

    /// Test that q=1.0 and q=0.0 are rejected.
    #[test]
    fn q_validation_out_of_range() {
        let r1 = RepetitiveController::<f64, 10>::new(1.0, 0.5);
        assert!(
            matches!(r1, Err(RepetitiveError::InvalidParameter)),
            "q=1.0 should be rejected"
        );

        let r2 = RepetitiveController::<f64, 10>::new(0.0, 0.5);
        assert!(
            matches!(r2, Err(RepetitiveError::InvalidParameter)),
            "q=0.0 should be rejected"
        );

        let r3 = RepetitiveController::<f64, 10>::new(1.1, 0.5);
        assert!(
            matches!(r3, Err(RepetitiveError::InvalidParameter)),
            "q=1.1 should be rejected"
        );

        let r4 = RepetitiveController::<f64, 10>::new(-0.1, 0.5);
        assert!(
            matches!(r4, Err(RepetitiveError::InvalidParameter)),
            "q=-0.1 should be rejected"
        );
    }

    /// Test that the modified FIR filter applies correctly and produces nonzero output.
    #[test]
    fn modified_fir_applies() {
        // q0=0.8, q1=0.05 → DC gain = 0.8+0.1=0.9 < 1 ✓
        let mut mrc =
            ModifiedRepetitiveController::<f64, 8>::new(0.8, 0.05, 0.3).expect("valid parameters");

        // Drive with constant positive error for several periods
        for _ in 0..32 {
            let _ = mrc.update(1.0).expect("update ok");
        }

        let sig = mrc.repetitive_signal();
        assert!(
            sig > 0.0,
            "Modified RC should produce positive signal with positive error, got {sig}"
        );
    }

    /// Test that modified RC with invalid stability condition is rejected.
    #[test]
    fn modified_fir_stability_validation() {
        // q0 + 2*q1 = 0.8 + 2*0.2 = 1.2 ≥ 1 → invalid
        let result = ModifiedRepetitiveController::<f64, 8>::new(0.8, 0.2, 0.3);
        assert!(
            matches!(result, Err(RepetitiveError::InvalidParameter)),
            "Stability violation should be rejected"
        );
    }

    /// Test that reset clears the buffer and trial counter.
    #[test]
    fn reset_clears_buffer() {
        let mut rc = RepetitiveController::<f64, 10>::new(0.95, 0.5).expect("valid parameters");

        // Populate buffer with nonzero values
        for _ in 0..20 {
            let _ = rc.update(1.0).expect("update ok");
        }
        assert!(
            rc.repetitive_signal().abs() > 1e-6,
            "Buffer should be nonzero before reset"
        );
        assert!(
            rc.trial_count() > 0,
            "Trial count should be nonzero before reset"
        );

        rc.reset();

        assert!(
            rc.repetitive_signal().abs() < 1e-12,
            "Buffer should be zero after reset, got {}",
            rc.repetitive_signal()
        );
        assert_eq!(
            rc.trial_count(),
            0,
            "Trial count should be zero after reset"
        );
        assert!(
            rc.rms_convergence().abs() < 1e-12,
            "RMS error should be zero after reset"
        );
    }

    /// Test that RMS convergence metric decreases for a decaying periodic error.
    #[test]
    fn rms_decreases_for_decaying_error() {
        const PERIOD: usize = 20;
        let mut rc = RepetitiveController::<f64, PERIOD>::new(0.95, 0.5).expect("valid parameters");

        // First period: large error
        for k in 0..PERIOD {
            let t = k as f64 * 2.0 * core::f64::consts::PI / PERIOD as f64;
            let _ = rc.update(t.sin()).expect("update ok");
        }
        let rms_after_first_period = rc.rms_convergence();

        // Run several more periods with progressively smaller error
        for period in 1..=5 {
            for k in 0..PERIOD {
                let t = k as f64 * 2.0 * core::f64::consts::PI / PERIOD as f64;
                let scale = 1.0 / (period as f64 + 1.0);
                let _ = rc.update(scale * t.sin()).expect("update ok");
            }
        }
        let rms_after_decay = rc.rms_convergence();

        assert!(
            rms_after_decay < rms_after_first_period,
            "RMS should decrease as error decays: {rms_after_decay:.6} vs {rms_after_first_period:.6}"
        );
    }

    /// Test that the modified RC reset clears both buffers.
    #[test]
    fn modified_reset_clears_buffer() {
        let mut mrc =
            ModifiedRepetitiveController::<f64, 8>::new(0.8, 0.05, 0.3).expect("valid parameters");

        for _ in 0..24 {
            let _ = mrc.update(1.0).expect("update ok");
        }
        assert!(
            mrc.repetitive_signal().abs() > 1e-6,
            "Should be nonzero before reset"
        );

        mrc.reset();

        assert!(
            mrc.repetitive_signal().abs() < 1e-12,
            "Should be zero after reset, got {}",
            mrc.repetitive_signal()
        );
        assert!(
            mrc.filtered_signal().abs() < 1e-12,
            "Filtered signal should be zero after reset"
        );
    }

    /// Test trial count increments correctly.
    #[test]
    fn trial_count_increments() {
        let mut rc = RepetitiveController::<f64, 10>::new(0.95, 0.5).expect("valid parameters");
        assert_eq!(rc.trial_count(), 0);
        for i in 1..=15 {
            let _ = rc.update(0.0).expect("update ok");
            assert_eq!(rc.trial_count(), i);
        }
    }
}
