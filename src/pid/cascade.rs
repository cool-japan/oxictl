use crate::core::scalar::ControlScalar;
use crate::core::signal::{ControlOutput, Feedback, Setpoint};
use crate::core::traits::Controller;
use crate::pid::standard::{Pid, PidConfig};

/// Cascade (inner/outer loop) PID controller.
///
/// The outer loop produces a setpoint for the inner loop.
/// Typical use: outer = position/speed loop, inner = current/torque loop.
pub struct CascadePid<S: ControlScalar> {
    outer: Pid<S>,
    inner: Pid<S>,
}

impl<S: ControlScalar> CascadePid<S> {
    /// Create a cascade controller from outer and inner PidConfigs.
    pub fn new(outer_config: PidConfig<S>, inner_config: PidConfig<S>) -> Self {
        Self {
            outer: outer_config.build(),
            inner: inner_config.build(),
        }
    }

    /// Update the cascade with outer setpoint and both feedbacks.
    /// - `outer_setpoint`: high-level reference (e.g., position)
    /// - `outer_feedback`: outer loop measurement (e.g., position)
    /// - `inner_feedback`: inner loop measurement (e.g., velocity/current)
    /// - `dt`: time step
    pub fn update(
        &mut self,
        outer_setpoint: &Setpoint<S>,
        outer_feedback: &Feedback<S>,
        inner_feedback: &Feedback<S>,
        dt: S,
    ) -> ControlOutput<S> {
        let outer_out = self.outer.update(outer_setpoint, outer_feedback, dt);
        let inner_sp = Setpoint::new(outer_out.value());
        self.inner.update(&inner_sp, inner_feedback, dt)
    }

    /// Update with separate dt for outer/inner (different sampling rates).
    pub fn update_multirate(
        &mut self,
        outer_setpoint: &Setpoint<S>,
        outer_feedback: &Feedback<S>,
        inner_feedback: &Feedback<S>,
        outer_dt: S,
        inner_dt: S,
    ) -> ControlOutput<S> {
        let outer_out = self.outer.update(outer_setpoint, outer_feedback, outer_dt);
        let inner_sp = Setpoint::new(outer_out.value());
        self.inner.update(&inner_sp, inner_feedback, inner_dt)
    }

    pub fn reset(&mut self) {
        self.outer.reset();
        self.inner.reset();
    }

    pub fn outer(&self) -> &Pid<S> {
        &self.outer
    }

    pub fn inner(&self) -> &Pid<S> {
        &self.inner
    }

    pub fn outer_mut(&mut self) -> &mut Pid<S> {
        &mut self.outer
    }

    pub fn inner_mut(&mut self) -> &mut Pid<S> {
        &mut self.inner
    }

    pub fn is_saturated(&self) -> bool {
        self.inner.is_saturated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pid::anti_windup::AntiWindupMethod;

    #[test]
    fn cascade_basic_operation() {
        let outer = PidConfig::p(1.0_f64);
        let inner = PidConfig::p(2.0_f64);
        let mut cascade = CascadePid::new(outer, inner);

        let sp = Setpoint::new(10.0);
        let outer_fb = Feedback::new(8.0);
        let inner_fb = Feedback::new(0.0);

        let out = cascade.update(&sp, &outer_fb, &inner_fb, 0.01);
        // Outer: 1.0 * (10-8) = 2.0 → inner setpoint = 2.0
        // Inner: 2.0 * (2.0 - 0.0) = 4.0
        assert!((out.value() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn cascade_reset() {
        let outer = PidConfig::pi(1.0_f64, 1.0);
        let inner = PidConfig::pi(1.0_f64, 1.0);
        let mut cascade = CascadePid::new(outer, inner);

        for _ in 0..10 {
            cascade.update(
                &Setpoint::new(10.0),
                &Feedback::new(0.0),
                &Feedback::new(0.0),
                0.01,
            );
        }
        cascade.reset();
        // After reset, integral should be 0
        let out = cascade.update(
            &Setpoint::new(0.0),
            &Feedback::new(0.0),
            &Feedback::new(0.0),
            0.01,
        );
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn cascade_converges_simulated_plant() {
        // Outer: position PI, Inner: velocity PI
        // Plant: dx/dt = v, dv/dt = u - v
        let outer = PidConfig::pi(5.0_f64, 2.0)
            .with_limits(-50.0, 50.0)
            .with_anti_windup(AntiWindupMethod::Clamping);
        let inner = PidConfig::pi(2.0_f64, 10.0)
            .with_limits(-100.0, 100.0)
            .with_anti_windup(AntiWindupMethod::Clamping);

        let mut cascade = CascadePid::new(outer, inner);

        let mut pos = 0.0_f64;
        let mut vel = 0.0_f64;
        let dt = 0.001;
        let target_pos = 1.0;

        for _ in 0..20_000 {
            let sp = Setpoint::new(target_pos);
            let outer_fb = Feedback::new(pos);
            let inner_fb = Feedback::new(vel);
            let u = cascade.update(&sp, &outer_fb, &inner_fb, dt);

            let dv = u.value() - vel;
            vel += dv * dt;
            pos += vel * dt;
        }

        assert!(
            (pos - target_pos).abs() < 0.15,
            "Should converge: pos={:.4}",
            pos
        );
    }
}
