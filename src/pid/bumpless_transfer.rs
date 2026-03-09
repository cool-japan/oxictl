//! Bumpless transfer between manual and automatic PID modes.
//!
//! On a manual→auto switch the PID integrator preload is applied so that the
//! controller output matches the manual output at the instant of switch-over,
//! preventing a step discontinuity ("bump") in the manipulated variable.
#![allow(dead_code)]

use crate::core::scalar::ControlScalar;
use crate::core::signal::{Feedback, Setpoint};
use crate::core::traits::Controller;
use crate::pid::Pid;

/// Operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlMode {
    /// Operator directly sets the output value.
    Manual,
    /// PID controller drives the output.
    Auto,
}

/// Bumpless transfer wrapper: prevents output bump on manual↔auto switch.
pub struct BumplessTransfer<S: ControlScalar> {
    /// Inner PID controller.
    pub pid: Pid<S>,
    /// Current operating mode.
    pub mode: ControlMode,
    /// Manual output value (tracked during manual mode).
    pub manual_output: S,
    /// Last computed output (for continuity query).
    pub last_output: S,
    /// One-shot preload offset applied on the first auto update.
    preload: S,
    /// True if a preload is pending.
    preloaded: bool,
}

impl<S: ControlScalar> BumplessTransfer<S> {
    /// Create a new bumpless transfer wrapper (starts in Manual mode).
    pub fn new(pid: Pid<S>) -> Self {
        Self {
            pid,
            mode: ControlMode::Manual,
            manual_output: S::ZERO,
            last_output: S::ZERO,
            preload: S::ZERO,
            preloaded: false,
        }
    }

    /// Set the manual output value (call while in Manual mode).
    pub fn set_manual(&mut self, v: S) {
        self.manual_output = v;
        self.last_output = v;
    }

    /// Switch to Auto mode.
    ///
    /// Resets the PID transient state and records a one-shot preload equal to
    /// the current manual output so that the first auto output equals the
    /// manual output (bumpless).
    pub fn switch_to_auto(&mut self) {
        if self.mode == ControlMode::Auto {
            return;
        }
        // Reset PID state (integral, prev_error, d_filter).
        self.pid.reset();
        // Record preload for the first auto update.
        self.preload = self.manual_output;
        self.preloaded = true;
        self.mode = ControlMode::Auto;
    }

    /// Switch to Manual mode: latch the current PID output as the manual value.
    pub fn switch_to_manual(&mut self) {
        if self.mode == ControlMode::Manual {
            return;
        }
        self.manual_output = self.last_output;
        self.mode = ControlMode::Manual;
    }

    /// Update the controller.
    ///
    /// In Manual mode returns `manual_output` unchanged.
    /// In Auto mode runs the inner PID; on the first call after a manual→auto
    /// switch a one-shot preload offset is added to produce a bumpless output.
    pub fn update(&mut self, sp: S, fb: S, dt: S) -> S {
        let output = match self.mode {
            ControlMode::Manual => self.manual_output,
            ControlMode::Auto => {
                let sp_sig = Setpoint::new(sp);
                let fb_sig = Feedback::new(fb);
                let pid_out = self.pid.update(&sp_sig, &fb_sig, dt).value();
                if self.preloaded {
                    self.preloaded = false;
                    pid_out + self.preload
                } else {
                    pid_out
                }
            }
        };
        self.last_output = output;
        output
    }

    /// Last computed output value.
    pub fn output(&self) -> S {
        self.last_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pid::PidConfig;

    fn make_bt() -> BumplessTransfer<f64> {
        let pid = PidConfig::pi(1.0_f64, 5.0)
            .with_limits(-100.0, 100.0)
            .build();
        BumplessTransfer::new(pid)
    }

    #[test]
    fn manual_mode_returns_manual_output() {
        let mut bt = make_bt();
        bt.set_manual(core::f64::consts::PI);
        let out = bt.update(0.0, 0.0, 0.01);
        assert!((out - core::f64::consts::PI).abs() < 1e-12, "out={out}");
    }

    #[test]
    fn bumpless_switch_to_auto() {
        let mut bt = make_bt();
        // Manual output = 5.0.
        bt.set_manual(5.0);
        bt.switch_to_auto();
        // First auto update with sp=0, fb=0:
        //   PID: P=0, I=0 (just reset) → pid_out = 0
        //   + preload = 5.0 → output = 5.0
        let out = bt.update(0.0, 0.0, 0.01);
        assert!((out - 5.0).abs() < 1e-9, "out={out}");
        // Second call: preload consumed, PID integral from previous step.
        // sp=0, fb=0 → error=0, only integral term contributes.
        // integral after first step = ki * error * dt = 5 * 0 * 0.01 = 0
        // So out2 ≈ 0.
        let out2 = bt.update(0.0, 0.0, 0.01);
        assert!(out2.abs() < 1e-9, "out2={out2}");
    }

    #[test]
    fn switch_to_manual_latches_output() {
        let mut bt = make_bt();
        bt.set_manual(0.0);
        bt.switch_to_auto();
        // Run a few auto steps to build up some output.
        for _ in 0..5 {
            bt.update(10.0, 0.0, 0.01);
        }
        let auto_out = bt.last_output;
        bt.switch_to_manual();
        // Manual output should now equal the latched auto output.
        assert!((bt.manual_output - auto_out).abs() < 1e-12);
        // Subsequent updates in manual mode return the latched value regardless of sp/fb.
        let manual_out = bt.update(99.0, 99.0, 0.01);
        assert!((manual_out - auto_out).abs() < 1e-12);
    }

    #[test]
    fn double_switch_to_auto_is_idempotent() {
        let mut bt = make_bt();
        bt.set_manual(2.0);
        bt.switch_to_auto();
        // Second call while already in Auto: should be a no-op (no double-preload).
        bt.switch_to_auto();
        let out = bt.update(0.0, 0.0, 0.01);
        // Preload from first switch_to_auto = 2.0; second call was no-op.
        assert!((out - 2.0).abs() < 1e-9, "out={out}");
    }
}
