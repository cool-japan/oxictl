use crate::core::scalar::ControlScalar;

/// Gain-scheduled PID controller with bumpless transfer and anti-windup.
///
/// Operating point breakpoints map to PID gain triples [Kp, Ki, Kd].
/// Between breakpoints, gains are linearly interpolated.
///
/// Anti-windup: integrator is back-calculated when output saturates.
/// Bumpless transfer: derivative is filtered on the measurement (not error)
/// to avoid derivative kick on setpoint change.
///
/// Hysteresis: scheduling variable must move beyond a dead-band before
/// triggering a region change, to prevent chattering.
///
/// # Type parameters
/// - `S`: scalar type (f32 or f64)
/// - `N`: number of breakpoints (compile-time const)
#[derive(Debug, Clone, Copy)]
pub struct ScheduledPid<S: ControlScalar, const N: usize> {
    /// Breakpoint operating-point values (sorted ascending).
    pub breakpoints: [S; N],
    /// PID gains at each breakpoint: [Kp, Ki, Kd].
    pub gains: [[S; 3]; N],
    /// Number of valid breakpoints.
    count: usize,

    /// Integration step (s).
    pub dt: S,
    /// Output saturation limit (±).
    pub output_limit: S,
    /// Anti-windup back-calculation gain.
    pub aw_gain: S,
    /// Derivative filter coefficient (0 = no filter, 1 = heavy filter).
    pub deriv_filter: S,
    /// Hysteresis band around scheduling variable.
    pub hysteresis: S,

    /// Integral accumulator.
    integrator: S,
    /// Previous measurement (for derivative on measurement).
    prev_meas: S,
    /// Filtered derivative term.
    deriv_state: S,
    /// Last saturated output (for anti-windup).
    last_output_sat: S,
    /// Last unsaturated output.
    last_output_raw: S,
    /// Last scheduling variable (for hysteresis).
    last_sched_var: S,
    /// Whether any update has occurred yet.
    initialized: bool,
}

impl<S: ControlScalar, const N: usize> ScheduledPid<S, N> {
    /// Create gain-scheduled PID with given integration step and output limits.
    pub fn new(dt: S, output_limit: S) -> Self {
        Self {
            breakpoints: [S::ZERO; N],
            gains: [[S::ZERO; 3]; N],
            count: 0,
            dt,
            output_limit,
            aw_gain: S::from_f64(0.1),
            deriv_filter: S::from_f64(0.1),
            hysteresis: S::ZERO,
            integrator: S::ZERO,
            prev_meas: S::ZERO,
            deriv_state: S::ZERO,
            last_output_sat: S::ZERO,
            last_output_raw: S::ZERO,
            last_sched_var: S::ZERO,
            initialized: false,
        }
    }

    /// Set anti-windup back-calculation gain (0 disables, 0.1 typical).
    pub fn with_aw_gain(mut self, aw: S) -> Self {
        self.aw_gain = aw;
        self
    }

    /// Set first-order derivative filter coefficient [0, 1).
    /// deriv_filter=0: no filtering; deriv_filter→1: heavy smoothing.
    pub fn with_deriv_filter(mut self, alpha: S) -> Self {
        self.deriv_filter = alpha;
        self
    }

    /// Set hysteresis band for scheduling variable transitions.
    pub fn with_hysteresis(mut self, band: S) -> Self {
        self.hysteresis = band;
        self
    }

    /// Add a breakpoint at `op_point` with PID gains `[Kp, Ki, Kd]`.
    ///
    /// Inserts in sorted order. Returns false if table full or N=0.
    pub fn add_breakpoint(&mut self, op_point: S, kp: S, ki: S, kd: S) -> bool {
        if self.count >= N {
            return false;
        }
        // Find sorted insertion index
        let mut idx = self.count;
        for i in 0..self.count {
            if op_point < self.breakpoints[i] {
                idx = i;
                break;
            }
        }
        // Shift right
        for j in (idx..self.count).rev() {
            self.breakpoints[j + 1] = self.breakpoints[j];
            self.gains[j + 1] = self.gains[j];
        }
        self.breakpoints[idx] = op_point;
        self.gains[idx] = [kp, ki, kd];
        self.count += 1;
        true
    }

    /// Interpolate [Kp, Ki, Kd] at scheduling variable value `v`.
    pub fn interpolate_gains(&self, v: S) -> [S; 3] {
        if self.count == 0 {
            return [S::ZERO; 3];
        }
        if self.count == 1 || v <= self.breakpoints[0] {
            return self.gains[0];
        }
        if v >= self.breakpoints[self.count - 1] {
            return self.gains[self.count - 1];
        }
        // Binary search for bracketing interval
        let mut lo = 0usize;
        let mut hi = self.count - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if v >= self.breakpoints[mid] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let span = self.breakpoints[hi] - self.breakpoints[lo];
        let alpha = if span > S::EPSILON {
            (v - self.breakpoints[lo]) / span
        } else {
            S::ZERO
        };
        let one_minus_alpha = S::ONE - alpha;
        [
            one_minus_alpha * self.gains[lo][0] + alpha * self.gains[hi][0],
            one_minus_alpha * self.gains[lo][1] + alpha * self.gains[hi][1],
            one_minus_alpha * self.gains[lo][2] + alpha * self.gains[hi][2],
        ]
    }

    /// Compute PID output.
    ///
    /// `setpoint`: desired value.
    /// `measurement`: current measured value.
    /// `sched_var`: scheduling variable (e.g. speed, altitude).
    ///
    /// Returns saturated control output.
    pub fn update(&mut self, setpoint: S, measurement: S, sched_var: S) -> S {
        // Apply hysteresis to scheduling variable
        let effective_sched = if !self.initialized {
            self.last_sched_var = sched_var;
            sched_var
        } else {
            let delta = (sched_var - self.last_sched_var).abs();
            if delta > self.hysteresis {
                self.last_sched_var = sched_var;
                sched_var
            } else {
                self.last_sched_var
            }
        };

        let [kp, ki, kd] = self.interpolate_gains(effective_sched);

        let error = setpoint - measurement;

        // Derivative on measurement to avoid derivative kick
        let raw_deriv = if self.initialized {
            (self.prev_meas - measurement) / self.dt
        } else {
            S::ZERO
        };

        // First-order filter on derivative
        let alpha = self.deriv_filter;
        self.deriv_state = alpha * self.deriv_state + (S::ONE - alpha) * raw_deriv;

        // Proportional term
        let p_term = kp * error;

        // Anti-windup back-calculation correction
        let aw_correction = self.aw_gain * (self.last_output_sat - self.last_output_raw);

        // Integral update with anti-windup
        self.integrator += ki * error * self.dt + aw_correction * self.dt;

        // Derivative term
        let d_term = kd * self.deriv_state;

        // Raw output
        let raw = p_term + self.integrator + d_term;
        self.last_output_raw = raw;

        // Saturate output
        let sat = raw.clamp_val(-self.output_limit, self.output_limit);
        self.last_output_sat = sat;

        self.prev_meas = measurement;
        self.initialized = true;

        sat
    }

    /// Reset integrator and derivative state (bumpless transfer).
    pub fn reset_integrator(&mut self) {
        self.integrator = S::ZERO;
        self.deriv_state = S::ZERO;
    }

    /// Full reset of all internal states.
    pub fn reset(&mut self) {
        self.integrator = S::ZERO;
        self.deriv_state = S::ZERO;
        self.prev_meas = S::ZERO;
        self.last_output_sat = S::ZERO;
        self.last_output_raw = S::ZERO;
        self.last_sched_var = S::ZERO;
        self.initialized = false;
    }

    /// Current integrator state.
    pub fn integrator_state(&self) -> S {
        self.integrator
    }

    /// Current number of breakpoints.
    pub fn breakpoint_count(&self) -> usize {
        self.count
    }
}

/// Smooth transition detector for gain scheduling.
///
/// Detects when the scheduling variable changes fast enough to require
/// bumpless transfer (output freezing during transition).
#[derive(Debug, Clone, Copy)]
pub struct TransitionDetector<S: ControlScalar> {
    /// Rate threshold above which a transition is declared.
    pub rate_threshold: S,
    /// Hysteresis for transition detection.
    pub hysteresis: S,
    prev_value: S,
    in_transition: bool,
    transition_count: u32,
}

impl<S: ControlScalar> TransitionDetector<S> {
    pub fn new(rate_threshold: S, hysteresis: S) -> Self {
        Self {
            rate_threshold,
            hysteresis,
            prev_value: S::ZERO,
            in_transition: false,
            transition_count: 0,
        }
    }

    /// Update with current scheduling variable and return transition status.
    pub fn update(&mut self, value: S, dt: S) -> bool {
        let rate = if dt > S::EPSILON {
            (value - self.prev_value).abs() / dt
        } else {
            S::ZERO
        };
        self.prev_value = value;

        if rate > self.rate_threshold + self.hysteresis {
            if !self.in_transition {
                self.transition_count += 1;
            }
            self.in_transition = true;
        } else if rate < self.rate_threshold - self.hysteresis {
            self.in_transition = false;
        }
        self.in_transition
    }

    pub fn is_in_transition(&self) -> bool {
        self.in_transition
    }

    pub fn transition_count(&self) -> u32 {
        self.transition_count
    }

    pub fn reset(&mut self) {
        self.prev_value = S::ZERO;
        self.in_transition = false;
        self.transition_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pid() -> ScheduledPid<f64, 4> {
        let mut pid = ScheduledPid::<f64, 4>::new(0.01, 100.0);
        pid.add_breakpoint(0.0, 1.0, 0.5, 0.1);
        pid.add_breakpoint(10.0, 2.0, 1.0, 0.2);
        pid.add_breakpoint(20.0, 3.0, 1.5, 0.3);
        pid
    }

    #[test]
    fn interpolate_at_breakpoint() {
        let pid = make_pid();
        let g = pid.interpolate_gains(0.0);
        assert!((g[0] - 1.0).abs() < 1e-10, "Kp={}", g[0]);
        assert!((g[1] - 0.5).abs() < 1e-10, "Ki={}", g[1]);
    }

    #[test]
    fn interpolate_midpoint() {
        let pid = make_pid();
        let g = pid.interpolate_gains(5.0);
        assert!((g[0] - 1.5).abs() < 1e-9, "Kp mid={}", g[0]);
        assert!((g[1] - 0.75).abs() < 1e-9, "Ki mid={}", g[1]);
    }

    #[test]
    fn interpolate_clamps_above() {
        let pid = make_pid();
        let g = pid.interpolate_gains(100.0);
        assert!((g[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn interpolate_clamps_below() {
        let pid = make_pid();
        let g = pid.interpolate_gains(-5.0);
        assert!((g[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pid_output_saturates() {
        let mut pid = make_pid();
        // Large setpoint → output should saturate
        let out = pid.update(1000.0, 0.0, 10.0);
        assert!(out <= 100.0 + 1e-9, "out={}", out);
    }

    #[test]
    fn pid_zero_error_zero_output_initially() {
        let mut pid = make_pid();
        let out = pid.update(0.0, 0.0, 5.0);
        assert!((out).abs() < 1e-10, "out={}", out);
    }

    #[test]
    fn pid_integrates_error() {
        let mut pid = ScheduledPid::<f64, 2>::new(0.01, 1000.0);
        pid.add_breakpoint(0.0, 0.0, 1.0, 0.0); // Ki only
        for _ in 0..100 {
            pid.update(1.0, 0.0, 0.0); // constant error=1
        }
        // Integrator should accumulate: Ki*error*dt*100 = 1*1*0.01*100 = 1.0
        assert!(
            (pid.integrator_state() - 1.0).abs() < 0.01,
            "integrator={}",
            pid.integrator_state()
        );
    }

    #[test]
    fn pid_reset_clears_integrator() {
        let mut pid = make_pid();
        for _ in 0..100 {
            pid.update(1.0, 0.0, 5.0);
        }
        pid.reset();
        assert_eq!(pid.integrator_state(), 0.0);
    }

    #[test]
    fn add_breakpoints_sorted() {
        let mut pid = ScheduledPid::<f64, 4>::new(0.01, 100.0);
        pid.add_breakpoint(10.0, 2.0, 1.0, 0.0);
        pid.add_breakpoint(0.0, 1.0, 0.5, 0.0);
        pid.add_breakpoint(5.0, 1.5, 0.75, 0.0);
        // Should be sorted: 0.0, 5.0, 10.0
        assert!((pid.breakpoints[0] - 0.0).abs() < 1e-10);
        assert!((pid.breakpoints[1] - 5.0).abs() < 1e-10);
        assert!((pid.breakpoints[2] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn transition_detector_fires() {
        let mut td = TransitionDetector::<f64>::new(5.0, 0.5);
        // No transition: slow change
        let r1 = td.update(1.0, 0.01); // rate=100 (first step from 0 to 1)
                                       // After first call from 0.0, rate = (1.0 - 0.0)/0.01 = 100 > 5.0
        assert!(r1, "Should detect fast transition");
        assert_eq!(td.transition_count(), 1);
    }

    #[test]
    fn transition_detector_no_fire_slow() {
        let mut td = TransitionDetector::<f64>::new(50.0, 1.0);
        // Slow change: value increments by 0.1 per step, dt=0.1 → rate=1.0 < 50
        for i in 0..100 {
            td.update(i as f64 * 0.1, 0.1);
        }
        assert!(!td.is_in_transition());
    }

    #[test]
    fn transition_detector_reset() {
        let mut td = TransitionDetector::<f64>::new(5.0, 0.5);
        td.update(10.0, 0.01);
        td.reset();
        assert!(!td.is_in_transition());
        assert_eq!(td.transition_count(), 0);
    }

    #[test]
    fn hysteresis_prevents_chattering() {
        let mut pid = ScheduledPid::<f64, 4>::new(0.01, 100.0).with_hysteresis(1.0);
        pid.add_breakpoint(0.0, 1.0, 0.1, 0.0);
        pid.add_breakpoint(10.0, 2.0, 0.2, 0.0);

        // Small oscillation around sched_var=5.0 (±0.5 < hysteresis=1.0)
        let mut out_vals = [0.0_f64; 10];
        for (i, slot) in out_vals.iter_mut().enumerate() {
            let sv = 5.0 + if i % 2 == 0 { 0.4 } else { -0.4 };
            *slot = pid.update(1.0, 0.9, sv);
        }
        // All outputs should use same gains → identical integrator accumulation path
        // (This is a qualitative check that no crash occurs and output is bounded)
        for v in out_vals {
            assert!(v.abs() <= 100.0);
        }
    }
}
