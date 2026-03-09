//! Feedforward controllers for model-following and trajectory tracking.
//!
//! Feedforward control improves tracking performance by computing the control
//! action needed to follow a desired trajectory without relying solely on
//! feedback error. Combined with a feedback controller, it cancels the nominal
//! plant dynamics.
//!
//! # Modules
//! - [`InversionFeedforward`]: Exact inversion of a first-order discrete plant
//! - [`PolynomialFeedforward`]: Velocity and acceleration feedforward

use crate::core::scalar::ControlScalar;

/// Errors returned by feedforward controller operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FeedforwardError {
    /// A parameter was outside its valid range.
    InvalidParameter,
    /// The plant gain b is zero (inversion impossible).
    ZeroGain,
}

impl core::fmt::Display for FeedforwardError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidParameter => write!(f, "Invalid feedforward parameter"),
            Self::ZeroGain => write!(f, "Plant gain b is zero; inversion not possible"),
        }
    }
}

/// Dynamic inversion feedforward for a first-order discrete plant.
///
/// Given the nominal plant model:
/// ```text
/// y[k] = a * y[k-1] + b * u[k-1]
/// ```
/// the inversion feedforward computes the input needed to achieve a desired
/// output at the next time step:
/// ```text
/// u_ff[k] = (y_des[k+1] - a * y[k]) / b
/// ```
///
/// This cancels the plant dynamics exactly when the model is accurate.
/// Combined with a feedback controller, it provides model-following control.
///
/// # Example
/// ```
/// use oxictl::repetitive::InversionFeedforward;
///
/// // Plant: y[k] = 0.9*y[k-1] + 0.5*u[k-1]
/// let mut ff = InversionFeedforward::<f64>::new(0.9, 0.5).unwrap();
/// let u_ff = ff.compute(1.0, 0.5).unwrap(); // Drive to y=1.0 from y=0.5
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct InversionFeedforward<S: ControlScalar> {
    /// Plant pole coefficient a.
    a: S,
    /// Plant gain coefficient b (must be nonzero).
    b: S,
    /// Previous plant output (maintained for reference; not used in compute).
    y_prev: S,
}

impl<S: ControlScalar> InversionFeedforward<S> {
    /// Construct a new inversion feedforward controller.
    ///
    /// # Parameters
    /// - `a`: Plant pole (any value including zero or > 1)
    /// - `b`: Plant gain (must be nonzero, |b| > EPSILON)
    ///
    /// # Errors
    /// Returns [`FeedforwardError::ZeroGain`] if b ≈ 0.
    pub fn new(a: S, b: S) -> Result<Self, FeedforwardError> {
        if b.abs() <= S::EPSILON {
            return Err(FeedforwardError::ZeroGain);
        }
        Ok(Self {
            a,
            b,
            y_prev: S::ZERO,
        })
    }

    /// Compute the feedforward input to drive the plant toward the desired next output.
    ///
    /// Implements the model inversion:
    /// ```text
    /// u_ff = (y_desired_next - a * y_current) / b
    /// ```
    ///
    /// # Parameters
    /// - `y_desired_next`: Desired plant output at step k+1
    /// - `y_current`: Current plant output at step k
    ///
    /// # Returns
    /// The feedforward control signal u_ff[k].
    pub fn compute(&mut self, y_desired_next: S, y_current: S) -> Result<S, FeedforwardError> {
        let u_ff = (y_desired_next - self.a * y_current) / self.b;
        self.y_prev = y_current;
        Ok(u_ff)
    }

    /// Reset the internal state.
    ///
    /// # Parameters
    /// - `y0`: Initial plant output to set as y_prev
    pub fn reset(&mut self, y0: S) {
        self.y_prev = y0;
    }

    /// Return the previously recorded plant output.
    pub fn y_prev(&self) -> S {
        self.y_prev
    }

    /// Return the plant pole coefficient a.
    pub fn a(&self) -> S {
        self.a
    }

    /// Return the plant gain coefficient b.
    pub fn b(&self) -> S {
        self.b
    }
}

/// Polynomial feedforward controller for smooth reference trajectories.
///
/// Computes feedforward based on the velocity and acceleration of the
/// desired trajectory:
/// ```text
/// u_ff = kv * ṙ + ka * r̈
/// ```
///
/// This is appropriate for motion control applications where the reference
/// trajectory is known in advance (position, velocity, acceleration).
///
/// # Example
/// ```
/// use oxictl::repetitive::PolynomialFeedforward;
///
/// let ff = PolynomialFeedforward::<f64>::new(1.5, 0.3);
/// let u_ff = ff.compute(2.0, 0.5); // u_ff = 1.5*2.0 + 0.3*0.5 = 3.15
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialFeedforward<S: ControlScalar> {
    /// Velocity feedforward gain.
    kv: S,
    /// Acceleration feedforward gain.
    ka: S,
}

impl<S: ControlScalar> PolynomialFeedforward<S> {
    /// Construct a new polynomial feedforward controller.
    ///
    /// # Parameters
    /// - `kv`: Velocity feedforward gain (can be zero or negative)
    /// - `ka`: Acceleration feedforward gain (can be zero or negative)
    pub fn new(kv: S, ka: S) -> Self {
        Self { kv, ka }
    }

    /// Compute the feedforward output.
    ///
    /// # Parameters
    /// - `r_dot`: Reference velocity ṙ
    /// - `r_ddot`: Reference acceleration r̈
    ///
    /// # Returns
    /// Feedforward signal u_ff = kv * ṙ + ka * r̈
    pub fn compute(&self, r_dot: S, r_ddot: S) -> S {
        self.kv * r_dot + self.ka * r_ddot
    }

    /// Update the feedforward gains.
    ///
    /// # Parameters
    /// - `kv`: New velocity feedforward gain
    /// - `ka`: New acceleration feedforward gain
    pub fn set_gains(&mut self, kv: S, ka: S) {
        self.kv = kv;
        self.ka = ka;
    }

    /// Return the velocity feedforward gain.
    pub fn kv(&self) -> S {
        self.kv
    }

    /// Return the acceleration feedforward gain.
    pub fn ka(&self) -> S {
        self.ka
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that perfect tracking is achieved when feedforward drives the model.
    ///
    /// Simulate the plant y[k] = a*y[k-1] + b*u[k-1] with the inversion
    /// feedforward. After one step, the plant output should match y_desired.
    #[test]
    fn perfect_tracking_first_order() {
        let a = 0.9_f64;
        let b = 0.5_f64;
        let mut ff = InversionFeedforward::<f64>::new(a, b).expect("valid params");

        let y_desired = 1.0_f64;
        let mut y_current = 0.0_f64;

        // Run several steps tracking a constant reference
        for _ in 0..20 {
            let u_ff = ff.compute(y_desired, y_current).expect("compute ok");
            // Simulate plant
            y_current = a * y_current + b * u_ff;
        }

        assert!(
            (y_current - y_desired).abs() < 1e-8,
            "Inversion feedforward should achieve perfect tracking, got y={y_current:.8}"
        );
    }

    /// Test that b=0 returns ZeroGain error.
    #[test]
    fn zero_b_returns_error() {
        let result = InversionFeedforward::<f64>::new(0.9, 0.0);
        assert!(
            matches!(result, Err(FeedforwardError::ZeroGain)),
            "b=0.0 should return ZeroGain error"
        );
    }

    /// Test that b close to zero (within EPSILON) returns ZeroGain error.
    #[test]
    fn near_zero_b_returns_error() {
        let result = InversionFeedforward::<f64>::new(0.9, f64::EPSILON * 0.5);
        assert!(
            matches!(result, Err(FeedforwardError::ZeroGain)),
            "b≈0 should return ZeroGain error"
        );
    }

    /// Test polynomial feedforward correct linear combination.
    #[test]
    fn polynomial_linear_combination() {
        let ff = PolynomialFeedforward::<f64>::new(2.0, 3.0);
        let u_ff = ff.compute(1.0, 2.0);
        // u_ff = 2.0*1.0 + 3.0*2.0 = 2.0 + 6.0 = 8.0
        assert!((u_ff - 8.0).abs() < 1e-10, "Expected 8.0, got {u_ff}");
    }

    /// Test reset clears internal state and sets y_prev to given value.
    #[test]
    fn reset_sets_y_prev() {
        let mut ff = InversionFeedforward::<f64>::new(0.5, 1.0).expect("valid params");

        let _ = ff.compute(1.0, 0.7).expect("compute ok");
        assert!(
            (ff.y_prev() - 0.7).abs() < 1e-10,
            "y_prev should be 0.7 after compute"
        );

        ff.reset(5.0);
        assert!(
            (ff.y_prev() - 5.0).abs() < 1e-10,
            "y_prev should be 5.0 after reset, got {}",
            ff.y_prev()
        );
    }

    /// Test exact inversion formula: u_ff = (y_des - a*y_cur) / b.
    #[test]
    fn inversion_feedforward_exact() {
        let a = 0.5_f64;
        let b = 1.0_f64;
        let mut ff = InversionFeedforward::<f64>::new(a, b).expect("valid params");

        // u_ff = (1.0 - 0.5 * 0.0) / 1.0 = 1.0
        let u_ff = ff.compute(1.0, 0.0).expect("compute ok");
        assert!((u_ff - 1.0).abs() < 1e-10, "Expected u_ff=1.0, got {u_ff}");

        // u_ff = (2.0 - 0.5 * 4.0) / 1.0 = 0.0
        let u_ff2 = ff.compute(2.0, 4.0).expect("compute ok");
        assert!(u_ff2.abs() < 1e-10, "Expected u_ff=0.0, got {u_ff2}");
    }

    /// Test polynomial feedforward with zero gains.
    #[test]
    fn polynomial_zero_gains() {
        let ff = PolynomialFeedforward::<f64>::new(0.0, 0.0);
        let u_ff = ff.compute(100.0, 200.0);
        assert!(
            u_ff.abs() < 1e-10,
            "Zero gains should give zero output, got {u_ff}"
        );
    }

    /// Test polynomial feedforward set_gains.
    #[test]
    fn polynomial_set_gains() {
        let mut ff = PolynomialFeedforward::<f64>::new(1.0, 1.0);
        ff.set_gains(2.0, 3.0);

        assert!((ff.kv() - 2.0).abs() < 1e-10, "kv should be updated");
        assert!((ff.ka() - 3.0).abs() < 1e-10, "ka should be updated");

        let u_ff = ff.compute(1.0, 1.0);
        // u_ff = 2.0*1.0 + 3.0*1.0 = 5.0
        assert!((u_ff - 5.0).abs() < 1e-10, "Expected 5.0, got {u_ff}");
    }

    /// Test inversion with negative b (still invertible).
    #[test]
    fn negative_b_inversion() {
        let a = 0.8_f64;
        let b = -0.5_f64;
        let mut ff = InversionFeedforward::<f64>::new(a, b).expect("valid params");

        // u_ff = (y_des - a*y_cur) / b = (1.0 - 0.8*0.0) / (-0.5) = -2.0
        let u_ff = ff.compute(1.0, 0.0).expect("compute ok");
        assert!(
            (u_ff - (-2.0)).abs() < 1e-10,
            "Expected u_ff=-2.0, got {u_ff}"
        );
    }
}
