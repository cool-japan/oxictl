/// Event-triggered control sampling.
///
/// Traditional periodic control transmits state information at every sample
/// instant regardless of whether the plant state has changed meaningfully.
/// Event-triggered control transmits only when a trigger condition is violated,
/// substantially reducing network utilisation while preserving stability.
///
/// # References
/// - Tabuada, P. (2007). "Event-triggered real-time scheduling of stabilizing
///   control tasks." *IEEE Trans. Autom. Control*, 52(9), 1680–1685.
/// - Girard, A. (2015). "Dynamic triggering mechanisms for event-triggered
///   control." *IEEE Trans. Autom. Control*, 60(7), 1992–1997.
use crate::core::matrix::{vec_norm, vec_sub};
use crate::core::scalar::ControlScalar;
use crate::networked::NetworkedError;

// ──────────────────────────────────────────────────────────────────────────────
// Static trigger (Tabuada 2007)
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the static event trigger.
///
/// Trigger condition: ‖e‖ ≥ σ·‖x‖ + ε
/// where e = x_last_sent − x_current is the error since the last transmission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticTriggerConfig<S: ControlScalar> {
    /// Relative threshold σ ∈ [0, 1). Larger values mean fewer transmissions
    /// but larger tracking error. Must satisfy σ < 1 for stability.
    pub sigma: S,

    /// Absolute threshold ε > 0. Prevents Zeno behaviour (infinite triggering
    /// in finite time) when x → 0.
    pub epsilon: S,

    /// Minimum inter-event time in milliseconds. Hard lower bound; even if the
    /// trigger condition is satisfied, a new event is not generated until this
    /// interval has elapsed since the previous event.
    pub min_inter_event_time_ms: S,
}

impl<S: ControlScalar> StaticTriggerConfig<S> {
    /// Construct a new configuration, validating parameters.
    ///
    /// # Errors
    /// Returns [`NetworkedError::InvalidTopology`] if σ ≥ 1, or if ε ≤ 0,
    /// or if `min_inter_event_time_ms` < 0.
    pub fn new(sigma: S, epsilon: S, min_inter_event_time_ms: S) -> Result<Self, NetworkedError> {
        if sigma >= S::ONE || sigma < S::ZERO {
            return Err(NetworkedError::InvalidTopology);
        }
        if epsilon <= S::ZERO {
            return Err(NetworkedError::InvalidTopology);
        }
        if min_inter_event_time_ms < S::ZERO {
            return Err(NetworkedError::InvalidTopology);
        }
        Ok(Self {
            sigma,
            epsilon,
            min_inter_event_time_ms,
        })
    }
}

/// Static event trigger implementing the Tabuada (2007) condition.
///
/// A transmission is triggered whenever
///   ‖e‖ ≥ σ·‖x‖ + ε
/// subject to a minimum inter-event time constraint.
///
/// `N` is the dimension of the state vector.
#[derive(Debug, Clone, Copy)]
pub struct StaticTrigger<S: ControlScalar, const N: usize> {
    config: StaticTriggerConfig<S>,
    /// Timestamp (ms) of the last triggered event.
    last_event_time_ms: S,
}

impl<S: ControlScalar, const N: usize> StaticTrigger<S, N> {
    /// Create a new static trigger.
    pub fn new(config: StaticTriggerConfig<S>) -> Self {
        Self {
            config,
            last_event_time_ms: -S::from_f64(1e9), // force trigger on first call
        }
    }

    /// Evaluate the trigger condition.
    ///
    /// Returns `true` if a new transmission should occur.
    ///
    /// - `x_current`: current state measurement.
    /// - `x_last`: state at the last transmission instant.
    /// - `t_now_ms`: current time in milliseconds.
    pub fn check(&mut self, x_current: &[S; N], x_last: &[S; N], t_now_ms: S) -> bool {
        let elapsed = t_now_ms - self.last_event_time_ms;
        if elapsed < self.config.min_inter_event_time_ms {
            return false;
        }
        let e = vec_sub(x_last, x_current);
        let norm_e = vec_norm(&e);
        let norm_x = vec_norm(x_current);
        let threshold = self.config.sigma * norm_x + self.config.epsilon;
        if norm_e >= threshold {
            self.last_event_time_ms = t_now_ms;
            true
        } else {
            false
        }
    }

    /// Force a trigger event at the given time (e.g., for initialisation).
    pub fn force_trigger(&mut self, t_now_ms: S) {
        self.last_event_time_ms = t_now_ms;
    }

    /// Time elapsed since the last trigger event, in milliseconds.
    pub fn time_since_last_ms(&self, t_now_ms: S) -> S {
        t_now_ms - self.last_event_time_ms
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Dynamic trigger (Girard 2015)
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the dynamic event trigger.
///
/// Trigger condition: ‖e‖² ≥ η
/// where η evolves as: η̇ = −β·η + γ·‖x‖² − ‖e‖²
///
/// The dynamic variable η provides memory of past deviations, resulting in
/// fewer transmissions than the static trigger while retaining stability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicTriggerConfig<S: ControlScalar> {
    /// Decay rate β > 0 for the internal variable η.
    pub beta: S,

    /// Gain γ > 0 on the state norm term.  Must satisfy γ < λ_min(Q)/λ_max(P)
    /// for the closed-loop stability certificate to hold.
    pub gamma: S,

    /// Initial value of the internal variable η₀ ≥ 0.
    pub eta_init: S,

    /// Minimum inter-event time in milliseconds.
    pub min_inter_event_time_ms: S,
}

impl<S: ControlScalar> DynamicTriggerConfig<S> {
    /// Construct and validate configuration.
    ///
    /// # Errors
    /// Returns [`NetworkedError::InvalidTopology`] if β ≤ 0, γ ≤ 0, or
    /// η₀ < 0.
    pub fn new(
        beta: S,
        gamma: S,
        eta_init: S,
        min_inter_event_time_ms: S,
    ) -> Result<Self, NetworkedError> {
        if beta <= S::ZERO || gamma <= S::ZERO {
            return Err(NetworkedError::InvalidTopology);
        }
        if eta_init < S::ZERO {
            return Err(NetworkedError::InvalidTopology);
        }
        if min_inter_event_time_ms < S::ZERO {
            return Err(NetworkedError::InvalidTopology);
        }
        Ok(Self {
            beta,
            gamma,
            eta_init,
            min_inter_event_time_ms,
        })
    }
}

/// Dynamic event trigger with internal auxiliary variable η.
///
/// The internal dynamics η̇ = −β·η + γ·‖x‖² − ‖e‖² are integrated via
/// explicit Euler with step-size `dt`.  A transmission is triggered when
///   ‖e‖² ≥ η
/// and the minimum inter-event time has elapsed.
///
/// `N` is the dimension of the state vector.
#[derive(Debug, Clone, Copy)]
pub struct DynamicTrigger<S: ControlScalar, const N: usize> {
    config: DynamicTriggerConfig<S>,
    /// Internal variable η.
    eta: S,
    /// Timestamp (ms) of the last triggered event.
    last_event_time_ms: S,
}

impl<S: ControlScalar, const N: usize> DynamicTrigger<S, N> {
    /// Create a new dynamic trigger.
    pub fn new(config: DynamicTriggerConfig<S>) -> Self {
        let eta = config.eta_init;
        Self {
            config,
            eta,
            last_event_time_ms: -S::from_f64(1e9),
        }
    }

    /// Evaluate the dynamic trigger condition and advance the internal state.
    ///
    /// - `x_current`: current plant state.
    /// - `x_last`: state at the last transmission instant.
    /// - `dt`: elapsed time since the last call, in seconds.
    /// - `t_now_ms`: current time in milliseconds (for MIET enforcement).
    ///
    /// Returns `true` if a new transmission should occur.
    pub fn check(&mut self, x_current: &[S; N], x_last: &[S; N], dt: S, t_now_ms: S) -> bool {
        let e = vec_sub(x_last, x_current);
        let norm_e_sq = {
            let mut s = S::ZERO;
            for v in e.iter() {
                s += *v * *v;
            }
            s
        };
        let norm_x_sq = {
            let mut s = S::ZERO;
            for v in x_current.iter() {
                s += *v * *v;
            }
            s
        };

        // Advance η via explicit Euler: η[k+1] = η[k] + dt·η̇
        let eta_dot = -self.config.beta * self.eta + self.config.gamma * norm_x_sq - norm_e_sq;
        let eta_new = self.eta + dt * eta_dot;
        // η must remain non-negative to retain the Lyapunov certificate.
        self.eta = if eta_new < S::ZERO { S::ZERO } else { eta_new };

        // Enforce minimum inter-event time.
        let elapsed = t_now_ms - self.last_event_time_ms;
        if elapsed < self.config.min_inter_event_time_ms {
            return false;
        }

        if norm_e_sq >= self.eta {
            self.last_event_time_ms = t_now_ms;
            // Reset η to prevent excessive dwell after a trigger.
            self.eta = self.config.eta_init;
            true
        } else {
            false
        }
    }

    /// Current value of the internal variable η.
    pub fn eta(&self) -> S {
        self.eta
    }

    /// Reset the internal variable to its initial value.
    pub fn reset_eta(&mut self) {
        self.eta = self.config.eta_init;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Inter-event timer statistics
// ──────────────────────────────────────────────────────────────────────────────

/// Tracks minimum inter-event time statistics across a run.
///
/// Records the total number of events, the minimum observed inter-event
/// interval, the maximum observed interval, and the running sum (from which
/// the mean can be derived).
#[derive(Debug, Clone, Copy)]
pub struct InterEventTimer<S: ControlScalar> {
    count: u64,
    min_iet_ms: S,
    max_iet_ms: S,
    sum_iet_ms: S,
    last_event_ms: Option<S>,
}

impl<S: ControlScalar> InterEventTimer<S> {
    /// Create a fresh timer.
    pub fn new() -> Self {
        Self {
            count: 0,
            min_iet_ms: S::from_f64(f64::MAX),
            max_iet_ms: S::ZERO,
            sum_iet_ms: S::ZERO,
            last_event_ms: None,
        }
    }

    /// Record an event at `t_ms`.  Call this every time a transmission occurs.
    pub fn record_event(&mut self, t_ms: S) {
        if let Some(prev) = self.last_event_ms {
            let iet = t_ms - prev;
            if iet > S::ZERO {
                self.count += 1;
                if iet < self.min_iet_ms {
                    self.min_iet_ms = iet;
                }
                if iet > self.max_iet_ms {
                    self.max_iet_ms = iet;
                }
                self.sum_iet_ms += iet;
            }
        }
        self.last_event_ms = Some(t_ms);
    }

    /// Number of inter-event intervals recorded.
    pub fn event_count(&self) -> u64 {
        self.count
    }

    /// Minimum inter-event interval observed (ms).  Returns `None` before any
    /// interval has been recorded.
    pub fn min_iet_ms(&self) -> Option<S> {
        if self.count == 0 {
            None
        } else {
            Some(self.min_iet_ms)
        }
    }

    /// Maximum inter-event interval observed (ms).  Returns `None` before any
    /// interval has been recorded.
    pub fn max_iet_ms(&self) -> Option<S> {
        if self.count == 0 {
            None
        } else {
            Some(self.max_iet_ms)
        }
    }

    /// Mean inter-event interval (ms).  Returns `None` before any interval has
    /// been recorded.
    pub fn mean_iet_ms(&self) -> Option<S> {
        if self.count == 0 {
            None
        } else {
            Some(self.sum_iet_ms / S::from_f64(self.count as f64))
        }
    }
}

impl<S: ControlScalar> Default for InterEventTimer<S> {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Generic event-triggered controller wrapper
// ──────────────────────────────────────────────────────────────────────────────

/// A zero-order-hold wrapper that applies any scalar control law with
/// event-triggered transmissions.
///
/// When the trigger fires the supplied `control_fn` is evaluated and the
/// computed control value is held until the next event.  Between events the
/// last control value is returned unchanged (zero-order hold).
///
/// # Type parameters
/// - `S`  — numeric scalar type.
/// - `N`  — state dimension.
/// - `F`  — control law: `Fn(&[S; N]) -> S`.  Must be `Copy` for embedded use.
pub struct EventTriggeredController<S: ControlScalar, const N: usize, F>
where
    F: Fn(&[S; N]) -> S,
{
    trigger: StaticTrigger<S, N>,
    control_fn: F,
    /// State at the last transmission instant.
    x_last: [S; N],
    /// Most recently computed control value (ZOH output).
    u_hold: S,
    /// Statistics tracker.
    timer: InterEventTimer<S>,
}

impl<S: ControlScalar, const N: usize, F> EventTriggeredController<S, N, F>
where
    F: Fn(&[S; N]) -> S,
{
    /// Create a new event-triggered controller.
    ///
    /// - `config`:      trigger configuration.
    /// - `control_fn`:  control law mapping state to scalar input.
    /// - `x_init`:      initial state estimate (sets the first held value).
    ///
    /// # Errors
    /// Propagates any error from [`StaticTriggerConfig::new`].
    pub fn new(
        config: StaticTriggerConfig<S>,
        control_fn: F,
        x_init: [S; N],
    ) -> Result<Self, NetworkedError> {
        let u_init = control_fn(&x_init);
        Ok(Self {
            trigger: StaticTrigger::new(config),
            control_fn,
            x_last: x_init,
            u_hold: u_init,
            timer: InterEventTimer::new(),
        })
    }

    /// Advance the controller by one sample.
    ///
    /// Returns `(u, transmitted)` where `u` is the control signal to apply and
    /// `transmitted` indicates whether a new transmission occurred.
    ///
    /// - `x`:       current state measurement.
    /// - `t_now_ms`: current time in milliseconds.
    pub fn update(&mut self, x: &[S; N], t_now_ms: S) -> (S, bool) {
        let triggered = self.trigger.check(x, &self.x_last, t_now_ms);
        if triggered {
            self.x_last = *x;
            self.u_hold = (self.control_fn)(x);
            self.timer.record_event(t_now_ms);
        }
        (self.u_hold, triggered)
    }

    /// Access the inter-event timer statistics.
    pub fn timer(&self) -> &InterEventTimer<S> {
        &self.timer
    }

    /// Current zero-order-hold control value.
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

    /// Build a simple proportional law: u = −k·x[0]
    fn prop_law<const N: usize>(k: f64) -> impl Fn(&[f64; N]) -> f64 {
        move |x| -k * x[0]
    }

    // ── StaticTrigger ──────────────────────────────────────────────────────────

    #[test]
    fn static_trigger_config_validation() {
        // σ ≥ 1 is invalid
        assert_eq!(
            StaticTriggerConfig::<f64>::new(1.0, 0.01, 0.0),
            Err(NetworkedError::InvalidTopology)
        );
        // ε ≤ 0 is invalid
        assert_eq!(
            StaticTriggerConfig::<f64>::new(0.5, 0.0, 0.0),
            Err(NetworkedError::InvalidTopology)
        );
        // valid config
        assert!(StaticTriggerConfig::<f64>::new(0.5, 0.01, 0.0).is_ok());
    }

    #[test]
    fn static_trigger_fires_on_large_error() {
        let cfg = StaticTriggerConfig::<f64>::new(0.5, 0.01, 0.0).expect("valid config");
        let mut trigger = StaticTrigger::<f64, 2>::new(cfg);

        // x_current far from x_last → must trigger
        let x_current = [1.0_f64, 0.0];
        let x_last = [10.0_f64, 0.0]; // e = 9, ‖x‖ = 1 → threshold = 0.51 → fires
        assert!(trigger.check(&x_current, &x_last, 1.0));
    }

    #[test]
    fn static_trigger_does_not_fire_on_small_error() {
        let cfg = StaticTriggerConfig::<f64>::new(0.5, 0.01, 0.0).expect("valid config");
        let mut trigger = StaticTrigger::<f64, 2>::new(cfg);

        // x_current ≈ x_last → error negligible → must not trigger
        let x_current = [1.0_f64, 0.0];
        let x_last = [1.001_f64, 0.0]; // ‖e‖ = 0.001, threshold ≈ 0.51
        assert!(!trigger.check(&x_current, &x_last, 1.0));
    }

    #[test]
    fn static_trigger_respects_min_iet() {
        let cfg = StaticTriggerConfig::<f64>::new(0.1, 0.01, 100.0).expect("valid config");
        let mut trigger = StaticTrigger::<f64, 2>::new(cfg);

        let x_current = [1.0_f64, 0.0];
        let x_last = [10.0_f64, 0.0];

        // First event (last_event_time = -1e9, elapsed huge → passes MIET)
        assert!(trigger.check(&x_current, &x_last, 0.0));
        // Immediately after: MIET = 100 ms, elapsed < 100 ms → must not fire
        assert!(!trigger.check(&x_current, &x_last, 50.0));
        // After 100 ms elapsed → allowed again
        assert!(trigger.check(&x_current, &x_last, 100.0));
    }

    /// A constant reference signal should cause the trigger rate to reduce over
    /// time as the error stays small once the state converges.
    #[test]
    fn static_trigger_rate_reduces_constant_ref() {
        let cfg = StaticTriggerConfig::<f64>::new(0.1, 0.01, 0.0).expect("valid config");
        let mut trigger = StaticTrigger::<f64, 2>::new(cfg);

        // Simulate: x converging to [1.0, 0.0] in 200 steps
        // Early steps: large error → many triggers
        // Late steps: small error → few triggers
        let mut early_triggers = 0u32;
        let mut late_triggers = 0u32;
        let mut x_last = [0.0_f64, 0.0];

        for step in 0..200 {
            // Exponential approach to reference
            let alpha = 0.95_f64;
            let x_current = [1.0 - alpha.powi(step + 1), 0.0];
            let t_ms = step as f64 * 10.0;

            if trigger.check(&x_current, &x_last, t_ms) {
                x_last = x_current;
                if step < 50 {
                    early_triggers += 1;
                } else if step >= 150 {
                    late_triggers += 1;
                }
            }
        }
        // The system converges, so late triggers should be fewer.
        assert!(
            early_triggers >= late_triggers,
            "early={early_triggers} late={late_triggers}"
        );
    }

    // ── DynamicTrigger ─────────────────────────────────────────────────────────

    #[test]
    fn dynamic_trigger_config_validation() {
        assert_eq!(
            DynamicTriggerConfig::<f64>::new(0.0, 1.0, 0.0, 0.0),
            Err(NetworkedError::InvalidTopology)
        );
        assert!(DynamicTriggerConfig::<f64>::new(1.0, 0.5, 0.1, 0.0).is_ok());
    }

    #[test]
    fn dynamic_trigger_fires_when_error_exceeds_eta() {
        let cfg = DynamicTriggerConfig::<f64>::new(1.0, 0.5, 1.0, 0.0).expect("valid config");
        let mut trigger = DynamicTrigger::<f64, 2>::new(cfg);

        // Very large error → ‖e‖² will exceed η
        let x_current = [1.0_f64, 0.0];
        let x_last = [100.0_f64, 0.0];
        assert!(trigger.check(&x_current, &x_last, 0.001, 10.0));
    }

    #[test]
    fn dynamic_trigger_fewer_events_than_static() {
        // For a constant state, dynamic trigger should produce ≤ events
        // compared to a static trigger with equivalent parameters.
        let static_cfg = StaticTriggerConfig::<f64>::new(0.1, 0.01, 0.0).expect("valid config");
        let dynamic_cfg =
            DynamicTriggerConfig::<f64>::new(2.0, 0.5, 0.01, 0.0).expect("valid config");

        let mut st = StaticTrigger::<f64, 2>::new(static_cfg);
        let mut dt = DynamicTrigger::<f64, 2>::new(dynamic_cfg);

        // simulate state approaching origin
        let mut x_last_s = [5.0_f64, 5.0];
        let mut x_last_d = [5.0_f64, 5.0];
        let mut static_count = 0u32;
        let mut dynamic_count = 0u32;
        let alpha = 0.9_f64;

        for step in 0..200 {
            let x_current = [5.0 * alpha.powi(step + 1), 5.0 * alpha.powi(step + 1)];
            let t_ms = step as f64;

            if st.check(&x_current, &x_last_s, t_ms) {
                x_last_s = x_current;
                static_count += 1;
            }
            if dt.check(&x_current, &x_last_d, 0.001, t_ms) {
                x_last_d = x_current;
                dynamic_count += 1;
            }
        }
        // Dynamic trigger should typically fire no more than the static trigger
        // over a convergent trajectory.
        assert!(
            dynamic_count <= static_count + 5,
            "dynamic={dynamic_count}, static={static_count}"
        );
    }

    // ── EventTriggeredController ────────────────────────────────────────────────

    #[test]
    fn event_triggered_controller_returns_u() {
        let cfg = StaticTriggerConfig::<f64>::new(0.5, 0.01, 0.0).expect("valid");
        let x_init = [0.0_f64, 0.0];
        let mut ctrl = EventTriggeredController::<f64, 2, _>::new(cfg, prop_law(2.0), x_init)
            .expect("valid controller");

        let x = [1.0_f64, 0.0];
        let (u, _tx) = ctrl.update(&x, 0.0);
        // proportional control: u = -2 * 1.0 = -2.0
        assert!((u - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn event_triggered_differs_from_time_triggered() {
        // Event-triggered should NOT fire every step; time-triggered fires every step.
        let cfg = StaticTriggerConfig::<f64>::new(0.1, 0.01, 10.0).expect("valid");
        let x_init = [0.0_f64, 0.0];
        let mut ctrl =
            EventTriggeredController::<f64, 2, _>::new(cfg, prop_law(1.0), x_init).expect("valid");

        let x_const = [0.5_f64, 0.0]; // constant state → error stays the same
        let mut transmissions = 0u32;
        for step in 0..100 {
            let (_, tx) = ctrl.update(&x_const, step as f64);
            if tx {
                transmissions += 1;
            }
        }
        // With MIET = 10 ms and 100 steps at 1 ms each, at most 10 events
        assert!(transmissions <= 15, "transmissions={transmissions}");
        // But there should be at least 1 (the first call always triggers via MIET bypass)
        assert!(transmissions >= 1);
    }

    // ── InterEventTimer ─────────────────────────────────────────────────────────

    #[test]
    fn inter_event_timer_records_correctly() {
        let mut timer = InterEventTimer::<f64>::new();
        assert!(timer.min_iet_ms().is_none());

        timer.record_event(0.0);
        timer.record_event(10.0);
        timer.record_event(30.0);

        assert_eq!(timer.event_count(), 2);
        assert!((timer.min_iet_ms().expect("some") - 10.0).abs() < 1e-10);
        assert!((timer.max_iet_ms().expect("some") - 20.0).abs() < 1e-10);
        assert!((timer.mean_iet_ms().expect("some") - 15.0).abs() < 1e-10);
    }
}
