//! Event-triggered sampling with threshold and hysteresis.
//!
//! An `EventTrigger` fires when the absolute error exceeds `threshold`, with
//! hysteresis to prevent chatter and an enforced minimum inter-event interval.
//! An optional debounce counter requires the condition to hold for
//! `required_debounce` consecutive calls before the event is declared.
use crate::core::scalar::ControlScalar;

/// Event trigger for a control error signal.
#[derive(Debug, Clone, Copy)]
pub struct EventTrigger<S: ControlScalar> {
    /// Absolute error threshold that activates an event.
    pub threshold: S,
    /// Hysteresis band: event clears when |error| < threshold - hysteresis.
    pub hysteresis: S,
    /// Minimum time between consecutive events (seconds).
    pub min_inter_event: S,
    /// Time elapsed since the last triggered event.
    time_since_event: S,
    /// Consecutive samples above threshold (for debounce).
    debounce_count: u32,
    /// Number of consecutive above-threshold samples required before firing.
    pub required_debounce: u32,
    /// Whether the trigger is currently in the "triggered" (latched) state.
    triggered_state: bool,
}

impl<S: ControlScalar> EventTrigger<S> {
    /// Create a new event trigger.
    ///
    /// * `threshold`       – absolute error level that causes a trigger.
    /// * `hysteresis`      – dead-band below threshold before de-triggering.
    /// * `min_inter_event` – minimum seconds between successive events.
    pub fn new(threshold: S, hysteresis: S, min_inter_event: S) -> Self {
        Self {
            threshold,
            hysteresis,
            min_inter_event,
            time_since_event: min_inter_event, // allow immediate first event
            debounce_count: 0,
            required_debounce: 0,
            triggered_state: false,
        }
    }

    /// Set the number of consecutive above-threshold samples required before firing.
    pub fn with_debounce(mut self, count: u32) -> Self {
        self.required_debounce = count;
        self
    }

    /// Evaluate whether a control update event should fire for `error`.
    ///
    /// Advances the inter-event timer by `dt` and applies threshold + hysteresis
    /// logic with debounce.  Returns `true` when an event is declared.
    pub fn should_fire(&mut self, error: S, dt: S) -> bool {
        self.time_since_event += dt;

        let abs_err = error.abs();
        let clear_threshold = (self.threshold - self.hysteresis).max(S::ZERO);

        // State transitions
        if !self.triggered_state {
            if abs_err >= self.threshold {
                self.debounce_count += 1;
            } else {
                self.debounce_count = 0;
            }

            if self.debounce_count > self.required_debounce
                && self.time_since_event >= self.min_inter_event
            {
                self.triggered_state = true;
                self.time_since_event = S::ZERO;
                self.debounce_count = 0;
                return true;
            }
        } else {
            // In triggered state: wait for error to drop below clear threshold.
            if abs_err < clear_threshold {
                self.triggered_state = false;
                self.debounce_count = 0;
            }
        }

        false
    }

    /// Manually advance the inter-event timer without evaluating error.
    pub fn advance_time(&mut self, dt: S) {
        self.time_since_event += dt;
    }

    /// Seconds elapsed since the last event.
    pub fn time_since_last_event(&self) -> S {
        self.time_since_event
    }

    /// Whether the trigger is currently in the latched (high-error) state.
    pub fn is_in_triggered_state(&self) -> bool {
        self.triggered_state
    }

    /// Reset all state (except configuration parameters).
    pub fn reset(&mut self) {
        self.time_since_event = self.min_inter_event;
        self.debounce_count = 0;
        self.triggered_state = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_when_error_exceeds_threshold() {
        let mut trig = EventTrigger::new(0.1_f64, 0.02, 0.0);
        // Below threshold: no fire
        assert!(!trig.should_fire(0.05, 0.01));
        // Above threshold: fires
        assert!(trig.should_fire(0.2, 0.01));
    }

    #[test]
    fn hysteresis_prevents_immediate_re_fire() {
        let mut trig = EventTrigger::new(0.1_f64, 0.05, 0.0);
        // Trigger
        assert!(trig.should_fire(0.15, 0.0));
        assert!(trig.is_in_triggered_state());
        // Error drops to 0.07 — above clear_threshold (0.05) → stays triggered, no new event
        assert!(!trig.should_fire(0.07, 0.0));
        assert!(trig.is_in_triggered_state());
        // Error drops below clear_threshold → de-triggers
        assert!(!trig.should_fire(0.02, 0.0));
        assert!(!trig.is_in_triggered_state());
    }

    #[test]
    fn min_inter_event_enforced() {
        let mut trig = EventTrigger::new(0.1_f64, 0.0, 0.5);
        // First event fires (time_since_event initialized to min_inter_event)
        assert!(trig.should_fire(0.2, 0.0));
        // Immediately try again after de-triggering: inter-event not yet elapsed
        trig.should_fire(0.0, 0.0); // de-trigger
        assert!(!trig.should_fire(0.2, 0.1)); // only 0.1 s elapsed
                                              // After full 0.5 s gap it fires again
        assert!(trig.should_fire(0.2, 0.4));
    }

    #[test]
    fn debounce_requires_consecutive_samples() {
        let mut trig = EventTrigger::new(0.1_f64, 0.0, 0.0).with_debounce(2);
        // 1st sample above: debounce_count=1, required=2 → no fire
        assert!(!trig.should_fire(0.5, 0.0));
        // 2nd sample above: debounce_count=2, still not > 2 → no fire
        assert!(!trig.should_fire(0.5, 0.0));
        // 3rd sample: debounce_count=3 > 2 → fires
        assert!(trig.should_fire(0.5, 0.0));
    }
}
