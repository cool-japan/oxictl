/// Self-triggered control — precompute the next required sampling instant.
///
/// In contrast to event-triggered control (which monitors the trigger condition
/// continuously or at every sample), self-triggered control computes at each
/// update the *next* time instant at which the controller will need to act.
/// This allows the processor to sleep or perform other tasks in the interim,
/// making it attractive for energy-constrained embedded systems.
///
/// # Theory
/// For a linear time-invariant system ẋ = Ax + Bu with state feedback u = Kx,
/// the closed-loop state evolves as ẋ = (A + BK)x.  Between updates the error
/// e(t) = x(t_k) − x(t) grows at most as
///   ‖e(t)‖ ≤ ‖x(t_k)‖ · (exp(‖A_cl‖ · (t−t_k)) − 1)
/// A conservative upper bound on the next trigger time τ* is then the solution
/// to   exp(‖A_cl‖ · τ) − 1 = σ ,   i.e.   τ* = ln(1 + σ) / ‖A_cl‖
/// (independent of the current state), providing a simple closed-form estimate.
use crate::core::matrix::vec_norm;
use crate::core::scalar::ControlScalar;
use crate::networked::NetworkedError;
use libm::log as libm_log;

// ──────────────────────────────────────────────────────────────────────────────
// SelfTrigger — next-trigger-time estimator
// ──────────────────────────────────────────────────────────────────────────────

/// Self-trigger that precomputes the next required transmission instant.
///
/// Given the current state `x` and the spectral norm (largest singular value)
/// of the closed-loop system matrix `A_cl`, the trigger computes a conservative
/// upper bound on the time until the static trigger condition
///   ‖e(τ)‖ ≥ σ·‖x(τ)‖
/// is first violated, using the ISS Lyapunov analysis bound.
///
/// `N` is the state dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfTrigger<S: ControlScalar, const N: usize> {
    /// Trigger threshold σ ∈ (0, 1).  Smaller values are more conservative
    /// (shorter inter-event times) but retain a tighter ISS bound.
    trigger_sigma: S,

    /// Minimum next-trigger time in seconds, to prevent Zeno behaviour.
    min_next_trigger_s: S,

    /// Maximum next-trigger time in seconds, as a hard cap.
    max_next_trigger_s: S,
}

impl<S: ControlScalar, const N: usize> SelfTrigger<S, N> {
    /// Construct a self-trigger.
    ///
    /// # Errors
    /// Returns [`NetworkedError::InvalidTopology`] if σ ∉ (0, 1), or if
    /// `min_next_trigger_s` ≤ 0, or if `max_next_trigger_s` ≤ `min_next_trigger_s`.
    pub fn new(
        trigger_sigma: S,
        min_next_trigger_s: S,
        max_next_trigger_s: S,
    ) -> Result<Self, NetworkedError> {
        if trigger_sigma <= S::ZERO || trigger_sigma >= S::ONE {
            return Err(NetworkedError::InvalidTopology);
        }
        if min_next_trigger_s <= S::ZERO {
            return Err(NetworkedError::InvalidTopology);
        }
        if max_next_trigger_s <= min_next_trigger_s {
            return Err(NetworkedError::InvalidTopology);
        }
        Ok(Self {
            trigger_sigma,
            min_next_trigger_s,
            max_next_trigger_s,
        })
    }

    /// Estimate the next required transmission time.
    ///
    /// # Arguments
    /// - `x`:       current state (used to confirm the current norm is non-zero).
    /// - `a_norm`:  induced 2-norm (spectral norm) of the closed-loop matrix
    ///   A_cl = A + BK.  Must be > 0.
    ///
    /// # Returns
    /// Conservative upper bound on the time (in seconds) until the trigger
    /// condition `‖e‖ ≥ σ·‖x‖` is first violated.
    ///
    /// # Errors
    /// Returns [`NetworkedError::NumericalError`] if `a_norm` ≤ 0 or if
    /// the current state is zero (no meaningful bound exists).
    pub fn next_trigger_time(&self, x: &[S; N], a_norm: S) -> Result<S, NetworkedError> {
        if a_norm <= S::ZERO {
            return Err(NetworkedError::NumericalError);
        }
        let norm_x = vec_norm(x);
        if norm_x <= S::ZERO {
            // State is at origin; use minimum inter-event time as fallback.
            return Ok(self.min_next_trigger_s);
        }

        // τ* = ln(1 + σ) / ‖A_cl‖
        let sigma_f64 = self.trigger_sigma.to_f64();
        let a_norm_f64 = a_norm.to_f64();
        let tau_f64 = libm_log(1.0 + sigma_f64) / a_norm_f64;
        let tau = S::from_f64(tau_f64);

        // Clamp to [min, max]
        let clamped = if tau < self.min_next_trigger_s {
            self.min_next_trigger_s
        } else if tau > self.max_next_trigger_s {
            self.max_next_trigger_s
        } else {
            tau
        };

        Ok(clamped)
    }

    /// Sigma threshold.
    pub fn trigger_sigma(&self) -> S {
        self.trigger_sigma
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// LQR coefficient storage
// ──────────────────────────────────────────────────────────────────────────────

/// Minimal LQR state-feedback gain storage for a scalar-output system.
///
/// Stores the gain row vector K such that u = −K·x produces a stabilising
/// scalar control input.
#[derive(Debug, Clone, Copy)]
pub struct LqrGain<S: ControlScalar, const N: usize> {
    /// Feedback gain row vector (1 × N).
    pub k: [S; N],
}

impl<S: ControlScalar, const N: usize> LqrGain<S, N> {
    /// Compute the control: u = −K·x.
    pub fn compute(&self, x: &[S; N]) -> S {
        let mut u = S::ZERO;
        for (&k_i, &x_i) in self.k.iter().zip(x.iter()) {
            u -= k_i * x_i;
        }
        u
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SelfTriggeredLqr
// ──────────────────────────────────────────────────────────────────────────────

/// LQR controller with self-triggered updates.
///
/// At each update the controller:
/// 1. Applies the held control value u = −K·x_last.
/// 2. If the current time has reached or passed the scheduled next-trigger
///    instant, resamples the state, updates the held value, and computes a new
///    next-trigger time.
///
/// # Type parameters
/// - `S` — scalar type.
/// - `N` — state dimension.
#[derive(Debug, Clone, Copy)]
pub struct SelfTriggeredLqr<S: ControlScalar, const N: usize> {
    gain: LqrGain<S, N>,
    trigger: SelfTrigger<S, N>,
    /// Spectral norm of the closed-loop matrix, used for next-trigger estimation.
    a_cl_norm: S,
    /// Held state at last transmission.
    x_last: [S; N],
    /// Held control value.
    u_hold: S,
    /// Scheduled next-trigger time in milliseconds.
    next_trigger_ms: S,
    /// Number of control updates (transmissions).
    update_count: u64,
}

impl<S: ControlScalar, const N: usize> SelfTriggeredLqr<S, N> {
    /// Construct the controller.
    ///
    /// # Arguments
    /// - `gain`:       LQR gain vector.
    /// - `trigger`:    self-trigger configuration.
    /// - `a_cl_norm`:  spectral norm of the closed-loop matrix A + BK.
    /// - `x_init`:     initial state estimate.
    ///
    /// # Errors
    /// Returns [`NetworkedError::NumericalError`] if `a_cl_norm` ≤ 0.
    pub fn new(
        gain: LqrGain<S, N>,
        trigger: SelfTrigger<S, N>,
        a_cl_norm: S,
        x_init: [S; N],
    ) -> Result<Self, NetworkedError> {
        if a_cl_norm <= S::ZERO {
            return Err(NetworkedError::NumericalError);
        }
        let u_hold = gain.compute(&x_init);
        Ok(Self {
            gain,
            trigger,
            a_cl_norm,
            x_last: x_init,
            u_hold,
            next_trigger_ms: S::ZERO, // trigger immediately on first call
            update_count: 0,
        })
    }

    /// Advance the self-triggered controller.
    ///
    /// # Arguments
    /// - `x`:        current state measurement.
    /// - `t_now_ms`: current time in milliseconds.
    ///
    /// # Returns
    /// `(u, Some(next_ms))` when a transmission occurred, where `next_ms` is
    /// the next scheduled trigger time in milliseconds.
    /// `(u, None)` when no transmission occurred (zero-order hold).
    ///
    /// # Errors
    /// Returns [`NetworkedError::NumericalError`] if the internal next-trigger
    /// computation fails.
    pub fn update(&mut self, x: &[S; N], t_now_ms: S) -> Result<(S, Option<S>), NetworkedError> {
        if t_now_ms >= self.next_trigger_ms {
            // Transmit: update held state and compute new control.
            self.x_last = *x;
            self.u_hold = self.gain.compute(x);
            self.update_count += 1;

            // Compute next trigger time (in seconds), convert to ms.
            let tau_s = self.trigger.next_trigger_time(x, self.a_cl_norm)?;
            let tau_ms = tau_s * S::from_f64(1000.0);
            let next_ms = t_now_ms + tau_ms;
            self.next_trigger_ms = next_ms;

            Ok((self.u_hold, Some(next_ms)))
        } else {
            Ok((self.u_hold, None))
        }
    }

    /// Number of transmissions that have occurred.
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Currently scheduled next-trigger time in milliseconds.
    pub fn next_trigger_ms(&self) -> S {
        self.next_trigger_ms
    }

    /// Currently held control value.
    pub fn u_hold(&self) -> S {
        self.u_hold
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SelfTrigger ────────────────────────────────────────────────────────────

    #[test]
    fn self_trigger_config_validation() {
        // σ = 0 is invalid
        assert_eq!(
            SelfTrigger::<f64, 2>::new(0.0, 0.001, 1.0),
            Err(NetworkedError::InvalidTopology)
        );
        // min ≤ 0 is invalid
        assert_eq!(
            SelfTrigger::<f64, 2>::new(0.5, 0.0, 1.0),
            Err(NetworkedError::InvalidTopology)
        );
        // max ≤ min is invalid
        assert_eq!(
            SelfTrigger::<f64, 2>::new(0.5, 1.0, 0.5),
            Err(NetworkedError::InvalidTopology)
        );
        // valid
        assert!(SelfTrigger::<f64, 2>::new(0.5, 0.001, 10.0).is_ok());
    }

    #[test]
    fn next_trigger_time_positive() {
        let trigger = SelfTrigger::<f64, 2>::new(0.3, 0.001, 10.0).expect("valid");
        let x = [1.0_f64, 0.5];
        let tau = trigger
            .next_trigger_time(&x, 2.0)
            .expect("no numerical error");
        assert!(tau > 0.0, "next trigger time must be positive, got {tau}");
    }

    #[test]
    fn next_trigger_time_clamp_to_min() {
        // With very large a_norm the raw τ* would be tiny; should be clamped to min.
        let trigger = SelfTrigger::<f64, 2>::new(0.01, 0.1, 10.0).expect("valid");
        let x = [1.0_f64, 0.0];
        let tau = trigger
            .next_trigger_time(&x, 1000.0) // huge a_norm → tiny raw τ
            .expect("no numerical error");
        assert!(tau >= 0.1, "should be clamped to min=0.1, got {tau}");
    }

    #[test]
    fn next_trigger_time_monotone_in_sigma() {
        // Larger σ → longer inter-event times (more conservative bound).
        let t1 = SelfTrigger::<f64, 2>::new(0.1, 0.001, 100.0)
            .expect("valid")
            .next_trigger_time(&[1.0, 0.0], 1.0)
            .expect("ok");
        let t2 = SelfTrigger::<f64, 2>::new(0.5, 0.001, 100.0)
            .expect("valid")
            .next_trigger_time(&[1.0, 0.0], 1.0)
            .expect("ok");
        assert!(
            t2 >= t1,
            "larger sigma should give longer next-trigger time: t1={t1}, t2={t2}"
        );
    }

    #[test]
    fn next_trigger_time_zero_x_returns_min() {
        let trigger = SelfTrigger::<f64, 2>::new(0.3, 0.05, 10.0).expect("valid");
        let x = [0.0_f64, 0.0]; // zero state
        let tau = trigger.next_trigger_time(&x, 2.0).expect("no error");
        assert!(
            (tau - 0.05).abs() < 1e-10,
            "zero state should return min={}, got {tau}",
            0.05
        );
    }

    #[test]
    fn next_trigger_satisfies_trigger_condition() {
        // Verify the bound: ‖e(τ*)‖ ≤ σ·‖x‖ for simple scalar system
        // ẋ = −a·x  → x(t) = x₀ e^{−at}
        // e(t) = x₀ − x₀ e^{−at} = x₀(1 − e^{−at})
        // ‖e(τ*)‖/‖x₀‖ = 1 − e^{−a·τ*}
        // The conservative bound uses a_norm = a in the closed-loop norm.
        let sigma = 0.3_f64;
        let trigger = SelfTrigger::<f64, 1>::new(sigma, 0.001, 100.0).expect("valid");
        let x = [2.0_f64];
        let a_norm = 1.5_f64;
        let tau = trigger.next_trigger_time(&x, a_norm).expect("ok");

        // Check: e(tau)/‖x‖ = 1 − e^{−a·τ} should be ≤ σ
        // (the bound is conservative, so equality holds at the limit)
        let ratio = 1.0 - (-a_norm * tau).exp();
        assert!(
            ratio <= sigma + 1e-9,
            "ratio={ratio} should be ≤ sigma={sigma}"
        );
    }

    // ── SelfTriggeredLqr ───────────────────────────────────────────────────────

    #[test]
    fn self_triggered_lqr_initial_update() {
        let gain = LqrGain::<f64, 2> { k: [1.0, 0.5] };
        let trigger = SelfTrigger::<f64, 2>::new(0.3, 0.001, 10.0).expect("valid");
        let x_init = [1.0_f64, 0.5];
        let mut ctrl = SelfTriggeredLqr::new(gain, trigger, 1.0, x_init).expect("valid");

        let x = [1.0_f64, 0.5];
        let (u, next) = ctrl.update(&x, 0.0).expect("ok");

        // u = -(1.0*1.0 + 0.5*0.5) = -1.25
        assert!((u - (-1.25)).abs() < 1e-10, "u={u}");
        assert!(next.is_some(), "first call should always transmit");
        assert!(next.expect("some") > 0.0);
    }

    #[test]
    fn self_triggered_lqr_holds_between_events() {
        let gain = LqrGain::<f64, 2> { k: [1.0, 0.0] };
        let trigger = SelfTrigger::<f64, 2>::new(0.5, 0.001, 100.0).expect("valid");
        let x_init = [1.0_f64, 0.0];
        let mut ctrl = SelfTriggeredLqr::new(gain, trigger, 0.5, x_init).expect("valid");

        // First call: triggers and returns next time
        let x = [1.0_f64, 0.0];
        let (u0, next0) = ctrl.update(&x, 0.0).expect("ok");
        assert!(next0.is_some());
        let next_ms = next0.expect("some");

        // Call at t < next_ms: should NOT trigger, returns held u
        let (u1, next1) = ctrl.update(&x, next_ms * 0.5).expect("ok");
        assert!(next1.is_none(), "should hold between events");
        assert!((u1 - u0).abs() < 1e-10, "held value unchanged");
    }

    #[test]
    fn self_triggered_lqr_update_count() {
        let gain = LqrGain::<f64, 2> { k: [2.0, 1.0] };
        let trigger = SelfTrigger::<f64, 2>::new(0.3, 0.001, 1000.0).expect("valid");
        let x_init = [1.0_f64, 0.0];
        let mut ctrl = SelfTriggeredLqr::new(gain, trigger, 1.0, x_init).expect("valid");

        // First update always triggers
        ctrl.update(&x_init, 0.0).expect("ok");
        let count_after_first = ctrl.update_count();
        assert_eq!(count_after_first, 1);

        // Skip well past the scheduled time
        ctrl.update(&x_init, 10_000.0).expect("ok");
        assert_eq!(ctrl.update_count(), 2);
    }
}
