//! 2-DOF (Two Degree-of-Freedom) controller and reference prefilter.
//!
//! The 2-DOF controller separates reference tracking performance from
//! disturbance rejection. Following the ISA standard formulation:
//!
//! ```text
//! u = Kp*(b*r - y) + Ki*∫(r-y)dt + Kd*(c*ṙ - ẏ)
//! ```
//!
//! where `b ∈ [0,1]` is the proportional setpoint weight and `c ∈ [0,1]` is
//! the derivative setpoint weight. Setting b=1, c=1 recovers the standard PID.
//! Setting b=0 eliminates proportional kick on reference steps. Setting c=0
//! eliminates derivative kick on reference steps.

use crate::core::scalar::ControlScalar;

/// Errors returned by 2-DOF controller operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TwoDofError {
    /// A parameter was outside its valid range.
    InvalidParameter,
}

impl core::fmt::Display for TwoDofError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Invalid 2-DOF controller parameter")
    }
}

/// ISA standard 2-DOF PID controller.
///
/// The 2-DOF formulation allows independent tuning of setpoint tracking
/// and disturbance rejection. The setpoint weights b and c shape the
/// reference response without affecting the disturbance rejection loop.
///
/// # Control Law
/// ```text
/// P = Kp * (b*r - y)
/// I = Ki * ∫(r - y) dt
/// D = Kd * (c*(r - r_prev)/dt - (y - y_prev)/dt)
/// u = clamp(P + I + D, u_min, u_max)
/// ```
///
/// Anti-windup: integration only proceeds when the output is not saturated,
/// or when integration would reduce (not increase) saturation.
///
/// # Example
/// ```
/// use oxictl::repetitive::TwoDofController;
///
/// let mut ctrl = TwoDofController::<f64>::new(
///     2.0, 0.5, 0.1,  // Kp, Ki, Kd
///     1.0, 0.0,        // b, c
///     -10.0, 10.0,     // u_min, u_max
///     0.01,            // dt
/// ).unwrap();
/// let u = ctrl.update(1.0, 0.0).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TwoDofController<S: ControlScalar> {
    /// Proportional gain.
    kp: S,
    /// Integral gain.
    ki: S,
    /// Derivative gain.
    kd: S,
    /// Reference weight for proportional term (0=output-feedback only, 1=full error).
    b: S,
    /// Reference weight for derivative term (0=output-derivative only, 1=full error).
    c: S,
    /// Integral accumulator.
    integral: S,
    /// Previous plant output (for derivative on output).
    y_prev: S,
    /// Previous reference (for derivative on reference).
    r_prev: S,
    /// Sample time [s].
    dt: S,
    /// Minimum output limit.
    u_min: S,
    /// Maximum output limit.
    u_max: S,
}

impl<S: ControlScalar> TwoDofController<S> {
    /// Construct a new 2-DOF PID controller.
    ///
    /// # Parameters
    /// - `kp`: Proportional gain (must be > 0)
    /// - `ki`: Integral gain (must be ≥ 0)
    /// - `kd`: Derivative gain (must be ≥ 0)
    /// - `b`: Proportional setpoint weight in [0, 1]
    /// - `c`: Derivative setpoint weight in [0, 1]
    /// - `u_min`: Minimum output (must be < u_max)
    /// - `u_max`: Maximum output
    /// - `dt`: Sample time in seconds (must be > 0)
    ///
    /// # Errors
    /// Returns [`TwoDofError::InvalidParameter`] if any parameter is invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kp: S,
        ki: S,
        kd: S,
        b: S,
        c: S,
        u_min: S,
        u_max: S,
        dt: S,
    ) -> Result<Self, TwoDofError> {
        if kp <= S::ZERO {
            return Err(TwoDofError::InvalidParameter);
        }
        if ki < S::ZERO || kd < S::ZERO {
            return Err(TwoDofError::InvalidParameter);
        }
        if b < S::ZERO || b > S::ONE {
            return Err(TwoDofError::InvalidParameter);
        }
        if c < S::ZERO || c > S::ONE {
            return Err(TwoDofError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(TwoDofError::InvalidParameter);
        }
        if u_min >= u_max {
            return Err(TwoDofError::InvalidParameter);
        }

        Ok(Self {
            kp,
            ki,
            kd,
            b,
            c,
            integral: S::ZERO,
            y_prev: S::ZERO,
            r_prev: S::ZERO,
            dt,
            u_min,
            u_max,
        })
    }

    /// Compute the control output for the given reference and measurement.
    ///
    /// # Parameters
    /// - `r`: Reference (setpoint) signal
    /// - `y`: Plant output (measurement)
    ///
    /// # Returns
    /// Control output u, clamped to [u_min, u_max].
    pub fn update(&mut self, r: S, y: S) -> Result<S, TwoDofError> {
        let error = r - y;

        // Proportional term: acts on weighted error (b*r - y)
        let p_term = self.kp * (self.b * r - y);

        // Derivative term: c*ṙ - ẏ (finite differences)
        let r_dot = (r - self.r_prev) / self.dt;
        let y_dot = (y - self.y_prev) / self.dt;
        let d_term = self.kd * (self.c * r_dot - y_dot);

        // Unsaturated output (before integral contribution)
        let u_pd = p_term + d_term;
        let u_unsat = u_pd + self.ki * self.integral;

        // Saturate output
        let u = u_unsat.clamp_val(self.u_min, self.u_max);

        // Anti-windup: conditional integration
        // Integrate if: (1) output is not saturated, OR
        //               (2) the error would drive the output back from saturation
        let saturated_high = u_unsat > self.u_max;
        let saturated_low = u_unsat < self.u_min;
        let should_integrate = !saturated_high && !saturated_low
            || (saturated_high && error < S::ZERO)
            || (saturated_low && error > S::ZERO);

        if should_integrate {
            self.integral += error * self.dt;
        }

        // Update state for next step
        self.y_prev = y;
        self.r_prev = r;

        Ok(u)
    }

    /// Return the current integral accumulator state.
    pub fn integral_state(&self) -> S {
        self.integral
    }

    /// Reset all internal state (integral, previous values).
    pub fn reset(&mut self) {
        self.integral = S::ZERO;
        self.y_prev = S::ZERO;
        self.r_prev = S::ZERO;
    }

    /// Update the output saturation limits.
    ///
    /// # Errors
    /// Returns [`TwoDofError::InvalidParameter`] if u_min >= u_max.
    pub fn set_limits(&mut self, u_min: S, u_max: S) -> Result<(), TwoDofError> {
        if u_min >= u_max {
            return Err(TwoDofError::InvalidParameter);
        }
        self.u_min = u_min;
        self.u_max = u_max;
        Ok(())
    }

    /// Return the proportional gain.
    pub fn kp(&self) -> S {
        self.kp
    }

    /// Return the integral gain.
    pub fn ki(&self) -> S {
        self.ki
    }

    /// Return the derivative gain.
    pub fn kd(&self) -> S {
        self.kd
    }

    /// Return the proportional setpoint weight b.
    pub fn b(&self) -> S {
        self.b
    }

    /// Return the derivative setpoint weight c.
    pub fn c(&self) -> S {
        self.c
    }
}

/// First-order reference prefilter for smoothing setpoint steps.
///
/// Implements a discrete first-order low-pass filter:
/// ```text
/// y[k] = y[k-1] + (dt/tau) * (r[k] - y[k-1])
/// ```
///
/// This reduces the bandwidth of the reference signal, attenuating
/// high-frequency content and reducing overshoot from step references.
///
/// # Example
/// ```
/// use oxictl::repetitive::ReferencePrefilter;
///
/// let mut pf = ReferencePrefilter::<f64>::new(0.1, 0.01).unwrap();
/// let r_filt = pf.filter(1.0);
/// ```
#[derive(Debug, Clone)]
pub struct ReferencePrefilter<S: ControlScalar> {
    /// Prefilter time constant [s].
    tau: S,
    /// Current filtered output.
    y: S,
    /// Sample time [s].
    dt: S,
}

impl<S: ControlScalar> ReferencePrefilter<S> {
    /// Construct a new reference prefilter.
    ///
    /// # Parameters
    /// - `tau`: Time constant in seconds (must be > 0)
    /// - `dt`: Sample time in seconds (must be > 0)
    ///
    /// # Errors
    /// Returns [`TwoDofError::InvalidParameter`] if tau ≤ 0 or dt ≤ 0.
    pub fn new(tau: S, dt: S) -> Result<Self, TwoDofError> {
        if tau <= S::ZERO {
            return Err(TwoDofError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(TwoDofError::InvalidParameter);
        }
        Ok(Self {
            tau,
            y: S::ZERO,
            dt,
        })
    }

    /// Apply the prefilter to the input reference.
    ///
    /// # Parameters
    /// - `r`: Raw reference input
    ///
    /// # Returns
    /// Filtered reference output.
    pub fn filter(&mut self, r: S) -> S {
        let alpha = self.dt / self.tau;
        self.y += alpha * (r - self.y);
        self.y
    }

    /// Reset the filter state to zero.
    pub fn reset(&mut self) {
        self.y = S::ZERO;
    }

    /// Return the current filter output.
    pub fn output(&self) -> S {
        self.y
    }

    /// Return the filter time constant.
    pub fn tau(&self) -> S {
        self.tau
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With b=1, the proportional term is Kp*(r - y) = Kp*error (standard PID P-term).
    #[test]
    fn b1_is_standard_error_proportional() {
        let kp = 3.0_f64;
        let mut ctrl = TwoDofController::<f64>::new(
            kp, 0.0, 0.0, // Kp, Ki=0, Kd=0
            1.0, 0.0, // b=1, c=0
            -100.0, 100.0, 0.01,
        )
        .expect("valid params");

        let r = 1.0_f64;
        let y = 0.4_f64;
        let u = ctrl.update(r, y).expect("update ok");

        // P = Kp*(1.0*r - y) = Kp*(r-y) = 3.0 * 0.6 = 1.8
        let expected = kp * (r - y);
        assert!((u - expected).abs() < 1e-10, "Expected {expected}, got {u}");
    }

    /// With b=0, the proportional term is Kp*(0 - y) = -Kp*y (no reference in P-term).
    #[test]
    fn b0_no_proportional_on_reference() {
        let kp = 2.0_f64;
        let mut ctrl = TwoDofController::<f64>::new(
            kp, 0.0, 0.0, // Kp, Ki=0, Kd=0
            0.0, 0.0, // b=0, c=0
            -100.0, 100.0, 0.01,
        )
        .expect("valid params");

        // y=0, r=1: P = Kp*(0*r - y) = Kp*(0-0) = 0
        let u = ctrl.update(1.0, 0.0).expect("update ok");
        assert!(
            u.abs() < 1e-10,
            "With b=0, y=0, r=1: u should be 0 (no P-kick on reference), got {u}"
        );
    }

    /// Test that anti-windup prevents integral from growing when output is saturated.
    #[test]
    fn anti_windup_clamps_output() {
        let mut ctrl = TwoDofController::<f64>::new(
            1.0, 100.0, 0.0, // Large Ki to force windup
            1.0, 0.0, -1.0, 1.0, // Tight limits
            0.01,
        )
        .expect("valid params");

        // Apply large positive error for many steps
        for _ in 0..200 {
            let u = ctrl.update(10.0, 0.0).expect("update ok");
            assert!(
                u <= 1.0 + 1e-10,
                "Output must not exceed u_max=1.0, got {u}"
            );
        }

        // With anti-windup, integral should not have grown to infinity
        let integral = ctrl.integral_state();
        assert!(
            integral.abs() < 1000.0,
            "Integral should be bounded by anti-windup, got {integral}"
        );
    }

    /// Test reference prefilter time constant effect.
    #[test]
    fn prefilter_time_constant() {
        // tau=1.0, dt=0.01 → after 100 steps (1 time constant), output ≈ 1-e^{-1} ≈ 0.632
        let mut pf = ReferencePrefilter::<f64>::new(1.0, 0.01).expect("valid params");

        let mut y = 0.0_f64;
        for _ in 0..100 {
            y = pf.filter(1.0);
        }

        let expected_approx = 1.0 - (-1.0_f64).exp(); // ≈ 0.6321
        assert!(
            (y - expected_approx).abs() < 0.01,
            "After 1 time constant, prefilter output should be ≈{expected_approx:.4}, got {y:.4}"
        );
    }

    /// Test that reset zeroes all controller states.
    #[test]
    fn reset_zeroes_states() {
        let mut ctrl = TwoDofController::<f64>::new(1.0, 1.0, 0.1, 1.0, 0.0, -10.0, 10.0, 0.01)
            .expect("valid params");

        // Drive to build up state
        for _ in 0..50 {
            let _ = ctrl.update(1.0, 0.5).expect("update ok");
        }
        assert!(
            ctrl.integral_state().abs() > 1e-6,
            "Integral should be nonzero before reset"
        );

        ctrl.reset();
        assert_eq!(
            ctrl.integral_state(),
            0.0,
            "Integral should be zero after reset"
        );
    }

    /// Test that prefilter reset zeroes filtered output.
    #[test]
    fn prefilter_reset() {
        let mut pf = ReferencePrefilter::<f64>::new(1.0, 0.01).expect("valid params");

        for _ in 0..50 {
            pf.filter(1.0);
        }
        assert!(pf.output() > 0.1, "Output should be nonzero before reset");

        pf.reset();
        assert_eq!(pf.output(), 0.0, "Output should be zero after reset");
    }

    /// Test step tracking convergence with PID gains.
    #[test]
    fn step_tracking_converges() {
        // Simple first-order plant: y[k] = 0.9*y[k-1] + 0.1*u[k-1]
        let mut ctrl = TwoDofController::<f64>::new(5.0, 2.0, 0.0, 1.0, 0.0, -50.0, 50.0, 0.01)
            .expect("valid params");

        let setpoint = 1.0_f64;
        let mut y = 0.0_f64;

        for _ in 0..500 {
            let u = ctrl.update(setpoint, y).expect("update ok");
            y = 0.9 * y + 0.1 * u;
        }

        assert!(
            (y - setpoint).abs() < 0.05,
            "System should converge to setpoint, got y={y:.4}"
        );
    }

    /// Test validation: kp must be positive.
    #[test]
    fn kp_validation() {
        let r = TwoDofController::<f64>::new(0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.01);
        assert!(
            matches!(r, Err(TwoDofError::InvalidParameter)),
            "kp=0.0 should be rejected"
        );

        let r2 = TwoDofController::<f64>::new(-1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.01);
        assert!(
            matches!(r2, Err(TwoDofError::InvalidParameter)),
            "kp<0 should be rejected"
        );
    }

    /// Test validation: b and c must be in [0,1].
    #[test]
    fn bc_range_validation() {
        // b > 1
        let r1 = TwoDofController::<f64>::new(1.0, 0.0, 0.0, 1.5, 0.0, -1.0, 1.0, 0.01);
        assert!(
            matches!(r1, Err(TwoDofError::InvalidParameter)),
            "b>1 should be rejected"
        );

        // c < 0
        let r2 = TwoDofController::<f64>::new(1.0, 0.0, 0.0, 1.0, -0.1, -1.0, 1.0, 0.01);
        assert!(
            matches!(r2, Err(TwoDofError::InvalidParameter)),
            "c<0 should be rejected"
        );
    }

    /// Test that u_min >= u_max is rejected.
    #[test]
    fn limit_validation() {
        let r = TwoDofController::<f64>::new(1.0, 0.0, 0.0, 1.0, 0.0, 5.0, 1.0, 0.01);
        assert!(
            matches!(r, Err(TwoDofError::InvalidParameter)),
            "u_min>=u_max should be rejected"
        );
    }
}
