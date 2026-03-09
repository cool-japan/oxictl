use crate::core::scalar::ControlScalar;
use crate::core::signal::{ControlOutput, Feedback, Setpoint};
use crate::core::traits::Controller;
use crate::pid::standard::{Pid, PidConfig};

/// A single gain table entry.
#[derive(Debug, Clone, Copy)]
pub struct GainEntry<S: ControlScalar> {
    /// Scheduling variable value at this breakpoint.
    pub schedule_var: S,
    pub kp: S,
    pub ki: S,
    pub kd: S,
}

impl<S: ControlScalar> GainEntry<S> {
    pub fn new(schedule_var: S, kp: S, ki: S, kd: S) -> Self {
        Self {
            schedule_var,
            kp,
            ki,
            kd,
        }
    }
}

/// Gain-scheduled PID controller.
///
/// Linearly interpolates between gain table entries based on a
/// scheduling variable (e.g., speed, temperature, altitude).
///
/// N = number of gain table entries.
pub struct GainScheduledPid<S: ControlScalar, const N: usize> {
    /// Gain table, sorted by schedule_var ascending.
    table: [GainEntry<S>; N],
    /// The underlying PID controller.
    pid: Pid<S>,
}

impl<S: ControlScalar, const N: usize> GainScheduledPid<S, N> {
    /// Create a gain-scheduled PID.
    /// `table` must be sorted by `schedule_var` ascending.
    /// `base_config` provides output limits and anti-windup settings.
    pub fn new(table: [GainEntry<S>; N], base_config: PidConfig<S>) -> Self {
        Self {
            table,
            pid: base_config.build(),
        }
    }

    /// Interpolate gains from the table at the given scheduling variable.
    pub fn interpolate_gains(&self, var: S) -> (S, S, S) {
        if N == 0 {
            return (S::ZERO, S::ZERO, S::ZERO);
        }
        if N == 1 {
            let e = &self.table[0];
            return (e.kp, e.ki, e.kd);
        }
        // Clamp to table bounds
        if var <= self.table[0].schedule_var {
            let e = &self.table[0];
            return (e.kp, e.ki, e.kd);
        }
        if var >= self.table[N - 1].schedule_var {
            let e = &self.table[N - 1];
            return (e.kp, e.ki, e.kd);
        }
        // Find bracketing entries
        for i in 0..(N - 1) {
            let lo = &self.table[i];
            let hi = &self.table[i + 1];
            if var >= lo.schedule_var && var <= hi.schedule_var {
                let span = hi.schedule_var - lo.schedule_var;
                if span <= S::ZERO {
                    return (lo.kp, lo.ki, lo.kd);
                }
                let alpha = (var - lo.schedule_var) / span;
                let kp = lo.kp + alpha * (hi.kp - lo.kp);
                let ki = lo.ki + alpha * (hi.ki - lo.ki);
                let kd = lo.kd + alpha * (hi.kd - lo.kd);
                return (kp, ki, kd);
            }
        }
        let e = &self.table[N - 1];
        (e.kp, e.ki, e.kd)
    }

    /// Update: first interpolate gains, then run PID.
    pub fn update(
        &mut self,
        setpoint: &Setpoint<S>,
        feedback: &Feedback<S>,
        schedule_var: S,
        dt: S,
    ) -> ControlOutput<S> {
        let (kp, ki, kd) = self.interpolate_gains(schedule_var);
        self.pid.set_gains(kp, ki, kd);
        self.pid.update(setpoint, feedback, dt)
    }

    pub fn reset(&mut self) {
        self.pid.reset();
    }

    pub fn pid(&self) -> &Pid<S> {
        &self.pid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_table() -> [GainEntry<f64>; 3] {
        [
            GainEntry::new(0.0, 1.0, 0.5, 0.1),
            GainEntry::new(50.0, 2.0, 1.0, 0.2),
            GainEntry::new(100.0, 3.0, 2.0, 0.5),
        ]
    }

    #[test]
    fn interpolate_below_min() {
        let table = build_table();
        let pid = GainScheduledPid::new(table, PidConfig::p(1.0_f64));
        let (kp, ki, _kd) = pid.interpolate_gains(-10.0);
        assert_eq!(kp, 1.0);
        assert_eq!(ki, 0.5);
    }

    #[test]
    fn interpolate_above_max() {
        let table = build_table();
        let pid = GainScheduledPid::new(table, PidConfig::p(1.0_f64));
        let (kp, _ki, kd) = pid.interpolate_gains(200.0);
        assert_eq!(kp, 3.0);
        assert_eq!(kd, 0.5);
    }

    #[test]
    fn interpolate_midpoint() {
        let table = build_table();
        let pid = GainScheduledPid::new(table, PidConfig::p(1.0_f64));
        let (kp, ki, kd) = pid.interpolate_gains(25.0); // midpoint of 0..50
        assert!((kp - 1.5).abs() < 1e-10, "kp={}", kp);
        assert!((ki - 0.75).abs() < 1e-10, "ki={}", ki);
        assert!((kd - 0.15).abs() < 1e-10, "kd={}", kd);
    }

    #[test]
    fn interpolate_exact_breakpoint() {
        let table = build_table();
        let pid = GainScheduledPid::new(table, PidConfig::p(1.0_f64));
        let (kp, _, _) = pid.interpolate_gains(50.0);
        assert!((kp - 2.0).abs() < 1e-10);
    }

    #[test]
    fn update_applies_interpolated_gains() {
        let table = build_table();
        let mut pid = GainScheduledPid::new(table, PidConfig::p(1.0_f64));
        let out = pid.update(
            &Setpoint::new(10.0_f64),
            &Feedback::new(5.0_f64),
            50.0,
            0.01,
        );
        // Gains at var=50: kp=2.0, ki=1.0, kd=0.2
        // P = 2.0 * 5.0 = 10.0 (first step, no integral/derivative)
        assert!((out.value() - 10.0).abs() < 0.1, "got {}", out.value());
    }

    #[test]
    fn single_entry_table() {
        let table = [GainEntry::new(0.0_f64, 5.0, 1.0, 0.0)];
        let pid = GainScheduledPid::new(table, PidConfig::p(1.0_f64));
        let (kp, _ki, _kd) = pid.interpolate_gains(100.0);
        assert_eq!(kp, 5.0);
    }
}
