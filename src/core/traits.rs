use crate::core::scalar::{ControlScalar, PidScalar};
use crate::core::signal::{ControlOutput, Feedback, Setpoint};

/// A generic controller that computes control output from setpoint and feedback.
pub trait Controller<S: PidScalar> {
    /// Compute one control step.
    /// `dt` is the time step in seconds.
    fn update(&mut self, setpoint: &Setpoint<S>, feedback: &Feedback<S>, dt: S)
        -> ControlOutput<S>;

    /// Reset internal state (integrators, filters, etc.)
    fn reset(&mut self);

    /// Returns true if the controller output is currently saturated.
    fn is_saturated(&self) -> bool;
}

/// State estimator (Kalman filter, observer, etc.)
/// `S` is the scalar type, `N` is the state dimension.
pub trait Estimator<S: ControlScalar, const N: usize> {
    /// Predict state forward by dt.
    fn predict(&mut self, dt: S);

    /// Correct state estimate with measurement.
    fn correct(&mut self, measurement: &[S; N]);

    /// Current state estimate.
    fn state(&self) -> &[S; N];

    /// Current covariance diagonal (simplified).
    fn covariance(&self) -> &[S; N];
}

/// A dynamic system (plant) that can be stepped forward in time.
pub trait Plant<S: ControlScalar> {
    /// Advance the plant by dt with control input u.
    fn step(&mut self, u: S, dt: S);

    /// Current plant output (measured variable).
    fn output(&self) -> S;

    /// Current internal state as a slice.
    fn state(&self) -> &[S];
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockController {
        saturated: bool,
    }

    impl Controller<f64> for MockController {
        fn update(
            &mut self,
            sp: &Setpoint<f64>,
            fb: &Feedback<f64>,
            _dt: f64,
        ) -> ControlOutput<f64> {
            ControlOutput::new(sp.value() - fb.value())
        }

        fn reset(&mut self) {
            self.saturated = false;
        }

        fn is_saturated(&self) -> bool {
            self.saturated
        }
    }

    #[test]
    fn mock_controller_computes_error() {
        let mut ctrl = MockController { saturated: false };
        let sp = Setpoint::new(10.0);
        let fb = Feedback::new(7.0);
        let out = ctrl.update(&sp, &fb, 0.01);
        assert!((out.value() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn mock_controller_reset() {
        let mut ctrl = MockController { saturated: true };
        assert!(ctrl.is_saturated());
        ctrl.reset();
        assert!(!ctrl.is_saturated());
    }
}
