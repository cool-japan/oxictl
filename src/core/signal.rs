use crate::core::scalar::PidScalar;

/// A desired target value for the controller.
#[derive(Debug, Clone, Copy)]
pub struct Setpoint<S: PidScalar> {
    value: S,
}

impl<S: PidScalar> Setpoint<S> {
    pub fn new(value: S) -> Self {
        Self { value }
    }

    pub fn value(&self) -> S {
        self.value
    }
}

/// A measured process variable (feedback signal).
#[derive(Debug, Clone, Copy)]
pub struct Feedback<S: PidScalar> {
    value: S,
}

impl<S: PidScalar> Feedback<S> {
    pub fn new(value: S) -> Self {
        Self { value }
    }

    pub fn value(&self) -> S {
        self.value
    }
}

/// The output of a controller (manipulated variable).
#[derive(Debug, Clone, Copy)]
pub struct ControlOutput<S: PidScalar> {
    value: S,
    saturated: bool,
}

impl<S: PidScalar> ControlOutput<S> {
    pub fn new(value: S) -> Self {
        Self {
            value,
            saturated: false,
        }
    }

    pub fn with_saturation(value: S, saturated: bool) -> Self {
        Self { value, saturated }
    }

    pub fn value(&self) -> S {
        self.value
    }

    pub fn is_saturated(&self) -> bool {
        self.saturated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setpoint_stores_value() {
        let sp = Setpoint::new(42.0_f64);
        assert_eq!(sp.value(), 42.0);
    }

    #[test]
    fn feedback_stores_value() {
        let fb = Feedback::new(core::f32::consts::PI);
        assert_eq!(fb.value(), core::f32::consts::PI);
    }

    #[test]
    fn control_output_default_not_saturated() {
        let out = ControlOutput::new(1.0_f64);
        assert!(!out.is_saturated());
        assert_eq!(out.value(), 1.0);
    }

    #[test]
    fn control_output_with_saturation() {
        let out = ControlOutput::with_saturation(5.0_f64, true);
        assert!(out.is_saturated());
        assert_eq!(out.value(), 5.0);
    }
}
