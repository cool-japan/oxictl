use crate::core::scalar::ControlScalar;

/// Auto-tuning state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoTuneState {
    Idle,
    RelayRunning,
    Done,
}

/// Ziegler-Nichols tuning rule selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZnRule {
    /// P controller: Kp = 0.5 * Ku
    P,
    /// PI controller: Kp = 0.45*Ku, Ti = 0.83*Tu
    Pi,
    /// PID controller: Kp = 0.6*Ku, Ti = 0.5*Tu, Td = 0.125*Tu
    Pid,
    /// "No overshoot" PID: more conservative
    PidNoOvershoot,
}

/// Relay feedback auto-tuner (Åström-Hägglund method).
///
/// Applies a relay (±relay_amplitude) to the output while measuring
/// the system oscillation. Extracts ultimate gain Ku and period Tu,
/// then computes PID gains via Ziegler-Nichols rules.
#[derive(Debug, Clone)]
pub struct RelayAutoTuner<S: ControlScalar> {
    /// Relay output amplitude.
    relay_amplitude: S,
    /// Target setpoint during tuning.
    setpoint: S,
    /// Hysteresis band (prevents chattering).
    hysteresis: S,
    state: AutoTuneState,
    relay_output: S,
    /// Peak values for amplitude estimation.
    peak_high: S,
    peak_low: S,
    /// Timestamps of relay crossings.
    last_crossing_time: S,
    half_period_sum: S,
    half_period_count: usize,
    /// Require this many half-periods before declaring done.
    min_cycles: usize,
    /// Elapsed time.
    elapsed: S,
    /// Results.
    ku: S,
    tu: S,
    tuning_done: bool,
}

impl<S: ControlScalar> RelayAutoTuner<S> {
    pub fn new(relay_amplitude: S, setpoint: S, hysteresis: S) -> Self {
        Self {
            relay_amplitude,
            setpoint,
            hysteresis,
            state: AutoTuneState::Idle,
            relay_output: relay_amplitude,
            peak_high: S::ZERO,
            peak_low: S::ZERO,
            last_crossing_time: S::ZERO,
            half_period_sum: S::ZERO,
            half_period_count: 0,
            min_cycles: 4,
            elapsed: S::ZERO,
            ku: S::ZERO,
            tu: S::ZERO,
            tuning_done: false,
        }
    }

    /// Set minimum number of half-periods before auto-tuning completes.
    pub fn with_min_cycles(mut self, n: usize) -> Self {
        self.min_cycles = n;
        self
    }

    /// Start the tuning process.
    pub fn start(&mut self) {
        self.state = AutoTuneState::RelayRunning;
        self.relay_output = self.relay_amplitude;
        self.peak_high = S::ZERO;
        self.peak_low = S::ZERO;
        self.last_crossing_time = S::ZERO;
        self.half_period_sum = S::ZERO;
        self.half_period_count = 0;
        self.elapsed = S::ZERO;
        self.tuning_done = false;
    }

    /// Process one control step. Returns relay output.
    pub fn update(&mut self, measurement: S, dt: S) -> S {
        if self.state != AutoTuneState::RelayRunning {
            return S::ZERO;
        }

        self.elapsed += dt;
        let error = self.setpoint - measurement;

        // Track peaks
        if measurement > self.peak_high {
            self.peak_high = measurement;
        }
        if measurement < self.peak_low || self.peak_low == S::ZERO {
            self.peak_low = measurement;
        }

        // Relay with hysteresis
        let _prev_output = self.relay_output;
        if error > self.hysteresis && self.relay_output < S::ZERO {
            self.relay_output = self.relay_amplitude;
            self.record_crossing();
        } else if error < -self.hysteresis && self.relay_output > S::ZERO {
            self.relay_output = -self.relay_amplitude;
            self.record_crossing();
        }

        // Check completion
        if self.half_period_count >= self.min_cycles {
            self.finish();
        }

        self.relay_output
    }

    fn record_crossing(&mut self) {
        if self.last_crossing_time > S::ZERO {
            let half_period = self.elapsed - self.last_crossing_time;
            if half_period > S::ZERO {
                self.half_period_sum += half_period;
                self.half_period_count += 1;
            }
        }
        self.last_crossing_time = self.elapsed;
    }

    fn finish(&mut self) {
        if self.half_period_count == 0 {
            return;
        }
        // Average half-period
        let avg_half = self.half_period_sum / S::from_f64(self.half_period_count as f64);
        self.tu = avg_half * S::TWO;

        // Oscillation amplitude
        let amplitude = (self.peak_high - self.peak_low) * S::HALF;
        if amplitude > S::ZERO {
            // Ku = 4*d / (π*a) where d = relay_amplitude, a = oscillation amplitude
            self.ku = S::from_f64(4.0) * self.relay_amplitude / (S::PI * amplitude);
        }

        self.state = AutoTuneState::Done;
        self.tuning_done = true;
    }

    /// Compute PID gains using Ziegler-Nichols rules.
    /// Returns (kp, ki, kd).
    pub fn gains(&self, rule: ZnRule) -> Option<(S, S, S)> {
        if !self.tuning_done || self.tu <= S::ZERO || self.ku <= S::ZERO {
            return None;
        }
        let ku = self.ku;
        let tu = self.tu;
        match rule {
            ZnRule::P => Some((ku * S::from_f64(0.5), S::ZERO, S::ZERO)),
            ZnRule::Pi => {
                let kp = ku * S::from_f64(0.45);
                let ti = tu * S::from_f64(0.83);
                let ki = if ti > S::ZERO { kp / ti } else { S::ZERO };
                Some((kp, ki, S::ZERO))
            }
            ZnRule::Pid => {
                let kp = ku * S::from_f64(0.6);
                let ti = tu * S::from_f64(0.5);
                let td = tu * S::from_f64(0.125);
                let ki = if ti > S::ZERO { kp / ti } else { S::ZERO };
                let kd = kp * td;
                Some((kp, ki, kd))
            }
            ZnRule::PidNoOvershoot => {
                let kp = ku * S::from_f64(0.2);
                let ti = tu * S::from_f64(0.5);
                let td = tu * S::from_f64(0.333);
                let ki = if ti > S::ZERO { kp / ti } else { S::ZERO };
                let kd = kp * td;
                Some((kp, ki, kd))
            }
        }
    }

    pub fn state(&self) -> AutoTuneState {
        self.state
    }

    pub fn is_done(&self) -> bool {
        self.tuning_done
    }

    pub fn ultimate_gain(&self) -> S {
        self.ku
    }

    pub fn ultimate_period(&self) -> S {
        self.tu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_relay_on_first_order(
        kp: f64,
        tau: f64,
        relay_amp: f64,
        setpoint: f64,
    ) -> RelayAutoTuner<f64> {
        let mut tuner = RelayAutoTuner::new(relay_amp, setpoint, 0.001).with_min_cycles(6);
        tuner.start();

        let dt = 0.001;
        let mut plant = 0.0_f64;

        for _ in 0..100_000 {
            if tuner.is_done() {
                break;
            }
            let u = tuner.update(plant, dt);
            // First-order plant: dy/dt = (kp*u - y) / tau
            let dy = (kp * u - plant) / tau;
            plant += dy * dt;
        }
        tuner
    }

    #[test]
    fn tuner_completes() {
        let tuner = run_relay_on_first_order(2.0, 0.1, 1.0, 0.0);
        assert!(tuner.is_done(), "Tuner should complete");
        assert_eq!(tuner.state(), AutoTuneState::Done);
    }

    #[test]
    fn ultimate_gain_and_period_positive() {
        let tuner = run_relay_on_first_order(2.0, 0.1, 1.0, 0.0);
        if tuner.is_done() {
            assert!(tuner.ultimate_gain() > 0.0);
            assert!(tuner.ultimate_period() > 0.0);
        }
    }

    #[test]
    fn zn_gains_valid() {
        let tuner = run_relay_on_first_order(2.0, 0.1, 1.0, 0.0);
        if tuner.is_done() {
            for rule in &[ZnRule::P, ZnRule::Pi, ZnRule::Pid, ZnRule::PidNoOvershoot] {
                let gains = tuner.gains(*rule);
                assert!(gains.is_some(), "Gains should be available for {:?}", rule);
                let (kp, ki, kd) = gains.unwrap();
                assert!(kp > 0.0, "kp should be positive");
                assert!(ki >= 0.0);
                assert!(kd >= 0.0);
            }
        }
    }

    #[test]
    fn gains_none_when_not_done() {
        let tuner = RelayAutoTuner::<f64>::new(1.0, 0.0, 0.001);
        assert!(tuner.gains(ZnRule::Pid).is_none());
    }

    #[test]
    fn idle_returns_zero() {
        let mut tuner = RelayAutoTuner::<f64>::new(1.0, 0.0, 0.001);
        // Not started, should return 0
        let u = tuner.update(0.0, 0.01);
        assert_eq!(u, 0.0);
    }
}
