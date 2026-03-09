//! Time-delay effects in digital control systems.
//!
//! Provides:
//! - [`DelayBuffer`]: exact D-step ring-buffer delay (const generic).
//! - [`PadeDelay`]: first-order Padé approximation of e^{−τs} as a causal IIR filter.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by delay constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayError {
    /// A parameter (τ or dt) was zero or negative.
    InvalidParameter,
    /// Zero delay was requested (use direct pass-through instead).
    ZeroDelay,
}

// ---------------------------------------------------------------------------
// DelayBuffer
// ---------------------------------------------------------------------------

/// Fixed D-step delay implemented as a circular ring buffer.
///
/// `push(v)` inserts value `v` and returns the value from D steps ago.
/// The buffer is initialised to a given value so the first D outputs are
/// that initial value (no garbage).
///
/// # Type parameters
/// - `S`: scalar type (must implement [`ControlScalar`]).
/// - `D`: delay depth in samples; must be ≥ 1 (enforced at compile time by
///   the array size — a zero-sized array would still compile, but `push`
///   would panic on modulo with 0).
pub struct DelayBuffer<S, const D: usize> {
    buffer: [S; D],
    head: usize,
}

impl<S: ControlScalar, const D: usize> DelayBuffer<S, D> {
    /// Create a delay buffer with all cells initialised to `initial`.
    pub fn new(initial: S) -> Self {
        Self {
            buffer: [initial; D],
            head: 0,
        }
    }

    /// Push a new sample and return the sample from D steps ago.
    ///
    /// The internal head pointer advances after each call.
    pub fn push(&mut self, value: S) -> S {
        // The oldest stored value is at position `head`
        let delayed = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % D;
        delayed
    }

    /// Return the current output (the oldest value) without advancing the buffer.
    pub fn peek_delayed(&self) -> S {
        self.buffer[self.head]
    }
}

// ---------------------------------------------------------------------------
// PadeDelay
// ---------------------------------------------------------------------------

/// First-order Padé approximation of a pure time delay e^{−τs}.
///
/// The continuous-time transfer function
///
/// ```text
/// H(s) = (1 − τs/2) / (1 + τs/2)
/// ```
///
/// is discretised via the bilinear (Tustin) transform s → 2/dt · (z−1)/(z+1),
/// yielding the difference equation
///
/// ```text
/// y[n] = b0·x[n] + b1·x[n−1] − a1·y[n−1]
/// ```
///
/// where
/// ```text
/// r  = τ / (2·dt)
/// b0 = (1−r)/(1+r)
/// b1 = 1
/// a1 = (1−r)/(1+r)   (same as b0)
/// ```
///
/// The group delay of the first-order Padé at low frequencies is τ/2.
pub struct PadeDelay<S> {
    b0: S,
    b1: S,
    a1: S,
    x_prev: S,
    y_prev: S,
    tau: S,
    dt: S,
}

impl<S: ControlScalar> PadeDelay<S> {
    /// Construct a Padé delay filter.
    ///
    /// # Parameters
    /// - `tau`: target delay in seconds (must be > 0).
    /// - `dt`: sample period in seconds (must be > 0).
    ///
    /// # Errors
    /// Returns [`DelayError::InvalidParameter`] if `tau <= 0` or `dt <= 0`.
    pub fn new(tau: S, dt: S) -> Result<Self, DelayError> {
        if tau <= S::ZERO || dt <= S::ZERO {
            return Err(DelayError::InvalidParameter);
        }

        // r = τ / (2·dt)  (bilinear substitution parameter)
        let r = tau / (S::TWO * dt);
        let one_plus_r = S::ONE + r;
        let one_minus_r = S::ONE - r;

        // From the derivation:
        // Y[n]·(1+r) = (1−r)·X[n] + (1+r)·X[n−1] − (1−r)·Y[n−1]
        // → b0 = (1−r)/(1+r),  b1 = 1,  a1 = (1−r)/(1+r)
        let b0 = one_minus_r / one_plus_r;
        let b1 = S::ONE;
        let a1 = one_minus_r / one_plus_r; // same as b0

        Ok(Self {
            b0,
            b1,
            a1,
            x_prev: S::ZERO,
            y_prev: S::ZERO,
            tau,
            dt,
        })
    }

    /// Apply the Padé filter to input sample `u`, returning the filtered output.
    pub fn filter(&mut self, u: S) -> S {
        let y = self.b0 * u + self.b1 * self.x_prev - self.a1 * self.y_prev;
        self.x_prev = u;
        self.y_prev = y;
        y
    }

    /// Approximate group delay of this filter.
    ///
    /// For the first-order Padé approximation the group delay at DC is τ/2.
    pub fn group_delay(&self) -> S {
        self.tau / S::TWO
    }

    /// Target delay τ.
    pub fn tau(&self) -> S {
        self.tau
    }

    /// Sample period dt.
    pub fn dt(&self) -> S {
        self.dt
    }

    /// Reset the filter state (previous input and output).
    pub fn reset(&mut self) {
        self.x_prev = S::ZERO;
        self.y_prev = S::ZERO;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // DelayBuffer tests
    // -----------------------------------------------------------------------

    #[test]
    fn delay_buffer_d1_first_output_is_initial() {
        let mut buf = DelayBuffer::<f64, 1>::new(0.0);
        // First push returns the initial value
        let out = buf.push(5.0);
        assert!(
            (out - 0.0).abs() < 1e-12,
            "Expected 0.0 (initial), got {out}"
        );
    }

    #[test]
    fn delay_buffer_d1_output_is_previous_input() {
        let mut buf = DelayBuffer::<f64, 1>::new(0.0);
        buf.push(5.0); // returns 0.0 (initial)
        let out = buf.push(9.0); // returns 5.0
        assert!((out - 5.0).abs() < 1e-12, "Expected 5.0, got {out}");
    }

    #[test]
    fn delay_buffer_d3_correct_delay() {
        let mut buf = DelayBuffer::<f64, 3>::new(-1.0);
        // First 3 pushes return the initial value (-1.0)
        let o0 = buf.push(10.0);
        let o1 = buf.push(20.0);
        let o2 = buf.push(30.0);
        assert!((o0 - (-1.0)).abs() < 1e-12, "o0 expected -1, got {o0}");
        assert!((o1 - (-1.0)).abs() < 1e-12, "o1 expected -1, got {o1}");
        assert!((o2 - (-1.0)).abs() < 1e-12, "o2 expected -1, got {o2}");
        // 4th push returns the first pushed value (10.0)
        let o3 = buf.push(40.0);
        assert!((o3 - 10.0).abs() < 1e-12, "o3 expected 10, got {o3}");
        // 5th push returns the second pushed value (20.0)
        let o4 = buf.push(50.0);
        assert!((o4 - 20.0).abs() < 1e-12, "o4 expected 20, got {o4}");
    }

    #[test]
    fn delay_buffer_d2_initial_fill() {
        let mut buf = DelayBuffer::<f64, 2>::new(7.0);
        let o0 = buf.push(1.0); // returns 7.0 (initial)
        let o1 = buf.push(2.0); // returns 7.0 (initial)
        let o2 = buf.push(3.0); // returns 1.0 (push #0)
        assert!((o0 - 7.0).abs() < 1e-12, "o0={o0}");
        assert!((o1 - 7.0).abs() < 1e-12, "o1={o1}");
        assert!((o2 - 1.0).abs() < 1e-12, "o2={o2}");
    }

    #[test]
    fn delay_buffer_peek_delayed_matches_push_output() {
        let mut buf = DelayBuffer::<f64, 2>::new(0.0);
        buf.push(1.0);
        buf.push(2.0);
        // peek_delayed should return what the next push would return
        let peeked = buf.peek_delayed();
        let pushed = buf.push(3.0);
        assert!(
            (peeked - pushed).abs() < 1e-12,
            "peek={peeked} != push={pushed}"
        );
    }

    // -----------------------------------------------------------------------
    // PadeDelay tests
    // -----------------------------------------------------------------------

    #[test]
    fn pade_invalid_parameters_rejected() {
        assert!(PadeDelay::<f64>::new(0.0, 0.001).is_err());
        assert!(PadeDelay::<f64>::new(-0.1, 0.001).is_err());
        assert!(PadeDelay::<f64>::new(0.01, 0.0).is_err());
        assert!(PadeDelay::<f64>::new(0.01, -0.001).is_err());
    }

    #[test]
    fn pade_unit_step_response_final_value_is_one() {
        // The Padé approximation is an all-pass filter with unit DC gain,
        // so the final value of the step response must be 1.
        let tau = 0.05_f64;
        let dt = 0.001_f64;
        let mut pade = PadeDelay::new(tau, dt).unwrap();
        let mut y = 0.0_f64;
        for _ in 0..5000 {
            y = pade.filter(1.0);
        }
        assert!(
            (y - 1.0).abs() < 1e-6,
            "Final value of step response = {y}, expected ≈ 1.0"
        );
    }

    #[test]
    fn pade_group_delay_is_half_tau() {
        let tau = 0.1_f64;
        let dt = 0.001_f64;
        let pade = PadeDelay::new(tau, dt).unwrap();
        let gd = pade.group_delay();
        assert!(
            (gd - tau / 2.0).abs() < 1e-12,
            "Group delay = {gd}, expected {}",
            tau / 2.0
        );
    }

    #[test]
    fn pade_step_response_delayed_rise() {
        // The step response of the Padé filter starts at a negative value
        // (characteristic phase-reversal of the non-minimum-phase zero),
        // then rises to 1.  Check that early outputs are < 1 and final is ~1.
        let tau = 0.1_f64;
        let dt = 0.001_f64;
        let mut pade = PadeDelay::new(tau, dt).unwrap();
        let first = pade.filter(1.0);
        // First output will be b0 * 1.0 + b1 * 0.0 - a1 * 0.0 = b0 = (1-r)/(1+r)
        // With r = tau/(2*dt) = 50, b0 = (1-50)/(1+50) = -49/51 ≈ -0.96
        assert!(
            first < 0.0,
            "First Padé output should be negative (non-min-phase), got {first}"
        );
        // Run to steady state
        let mut y = first;
        for _ in 1..10_000 {
            y = pade.filter(1.0);
        }
        assert!(
            (y - 1.0).abs() < 1e-5,
            "Steady-state should be 1.0, got {y}"
        );
    }

    #[test]
    fn pade_reset_clears_state() {
        let tau = 0.01_f64;
        let dt = 0.001_f64;
        let mut pade = PadeDelay::new(tau, dt).unwrap();
        // Run for a while
        for _ in 0..100 {
            pade.filter(1.0);
        }
        pade.reset();
        // After reset, output with zero input should be zero
        let y = pade.filter(0.0);
        assert!(y.abs() < 1e-12, "After reset, y(0) should be 0, got {y}");
    }
}
