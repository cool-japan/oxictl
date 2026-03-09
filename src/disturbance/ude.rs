//! Uncertainty and Disturbance Estimator (UDE) — scalar version.
//!
//! The UDE method estimates lumped uncertainties and disturbances acting on a
//! linear plant by passing the residual between the measured output dynamics
//! and the nominal model prediction through a stable reference filter.
//!
//! For the scalar plant:
//! ```text
//!   ẋ = a·x + b·(u + d)
//! ```
//! with the first-order reference filter H_f(s) = a_f / (s + a_f), the UDE
//! estimator satisfies:
//! ```text
//!   d̂_dot = −a_f · d̂ + a_f · (ẏ − a·y − b·u)
//! ```
//! where ẏ is approximated by a backward-difference of consecutive samples.
//!
//! References:
//! - Zhong, Q.-C. & Rees, D. (2004). "Control of uncertain LTI systems
//!   based on an uncertainty and disturbance estimator."
//!   ASME Journal of Dynamic Systems, Measurement, and Control, 126(4), 905–910.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when constructing or updating a [`UdeController`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UdeError {
    /// Filter bandwidth a_f must be strictly positive.
    NonPositiveFilterBandwidth,
    /// Sampling period dt must be strictly positive.
    NonPositiveDt,
}

impl core::fmt::Display for UdeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            UdeError::NonPositiveFilterBandwidth => {
                f.write_str("UDE filter bandwidth a_f must be strictly positive")
            }
            UdeError::NonPositiveDt => f.write_str("sampling period dt must be strictly positive"),
        }
    }
}

// ---------------------------------------------------------------------------
// Controller struct
// ---------------------------------------------------------------------------

/// Uncertainty and Disturbance Estimator for a scalar LTI plant.
///
/// # Discrete update law (Euler forward)
/// ```text
///   ẏ[k]  ≈ (y[k] − y[k-1]) / dt
///   ξ[k]   = ẏ[k] − a·y[k] − b·u[k]          (residual)
///   d̂_dot  = −a_f · d̂[k] + a_f · ξ[k]
///   d̂[k+1] = d̂[k] + dt · d̂_dot
/// ```
///
/// On the very first call (`initialized == false`) no derivative can be
/// computed, so the estimator is initialised and returns the current estimate
/// (zero by default) unchanged.
///
/// # Type parameters
/// - `S`: numeric scalar type implementing [`ControlScalar`] (f32 or f64).
#[derive(Debug, Clone, Copy)]
pub struct UdeController<S: ControlScalar> {
    /// Current disturbance estimate d̂.
    d_hat: S,
    /// Previous plant output, used for finite-difference derivative.
    y_prev: S,
    /// Reference filter bandwidth a_f (> 0).
    af: S,
    /// Nominal plant pole (coefficient of x in ẋ = a·x + b·(u+d)).
    a: S,
    /// Nominal plant gain (coefficient of (u+d)).
    b: S,
    /// Sampling period.
    dt: S,
    /// Whether the first sample has been processed.
    initialized: bool,
}

impl<S: ControlScalar> UdeController<S> {
    /// Construct a new [`UdeController`].
    ///
    /// # Arguments
    /// - `af`: reference filter bandwidth a_f (rad/s). Must be > 0.
    /// - `a`:  nominal plant pole. Can be any real value.
    /// - `b`:  nominal plant gain. Can be any non-zero real value.
    /// - `dt`: sampling period (seconds). Must be > 0.
    ///
    /// # Errors
    /// Returns [`UdeError`] if `af ≤ 0` or `dt ≤ 0`.
    pub fn new(af: S, a: S, b: S, dt: S) -> Result<Self, UdeError> {
        if af <= S::ZERO {
            return Err(UdeError::NonPositiveFilterBandwidth);
        }
        if dt <= S::ZERO {
            return Err(UdeError::NonPositiveDt);
        }
        Ok(Self {
            d_hat: S::ZERO,
            y_prev: S::ZERO,
            af,
            a,
            b,
            dt,
            initialized: false,
        })
    }

    /// Run one discrete estimation step.
    ///
    /// # Arguments
    /// - `y`: current plant output (= state x for the scalar system).
    /// - `u`: plant control input applied at the current step.
    ///
    /// # Returns
    /// `Ok(d_hat)` — the updated disturbance estimate.
    pub fn update(&mut self, y: S, u: S) -> Result<S, UdeError> {
        // On the first call we have no previous y; initialise and return zero.
        if !self.initialized {
            self.y_prev = y;
            self.initialized = true;
            return Ok(self.d_hat);
        }

        // Backward-difference derivative approximation
        let y_dot = (y - self.y_prev) / self.dt;
        self.y_prev = y;

        // Residual between measured dynamics and nominal model
        let residual = y_dot - self.a * y - self.b * u;

        // Euler integration of the UDE filter equation:
        //   d̂_dot = −a_f · d̂ + a_f · residual
        let d_hat_dot = -(self.af * self.d_hat) + self.af * residual;
        self.d_hat += d_hat_dot * self.dt;

        Ok(self.d_hat)
    }

    /// Return the most recently computed disturbance estimate.
    #[inline]
    pub fn estimate(&self) -> S {
        self.d_hat
    }

    /// Reset the estimator to its initial (zero) state.
    pub fn reset(&mut self) {
        self.d_hat = S::ZERO;
        self.y_prev = S::ZERO;
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate the scalar plant y[k+1] = y[k] + dt·(a·y[k] + b·(u + d))
    fn simulate_plant(a: f64, b: f64, dt: f64, u: f64, d: f64, y0: f64, steps: usize) -> Vec<f64> {
        let mut y = y0;
        let mut ys = Vec::with_capacity(steps + 1);
        ys.push(y);
        for _ in 0..steps {
            y += dt * (a * y + b * (u + d));
            ys.push(y);
        }
        ys
    }

    // -----------------------------------------------------------------------
    // 1. Step disturbance: d_hat should converge close to true d
    // -----------------------------------------------------------------------
    #[test]
    fn test_step_disturbance() {
        let a = -1.0_f64;
        let b = 1.0_f64;
        let af = 10.0_f64;
        let dt = 0.001_f64;
        let u = 0.0_f64;
        let d = 1.0_f64;
        let y0 = 0.0_f64;
        let steps = 3000usize;

        let ys = simulate_plant(a, b, dt, u, d, y0, steps);
        let mut ude = UdeController::new(af, a, b, dt).expect("valid");

        let mut d_hat = 0.0_f64;
        for &y_k in &ys[..steps] {
            d_hat = ude.update(y_k, u).expect("update ok");
        }

        let err = (d_hat - d).abs();
        assert!(
            err < 0.2 * d,
            "UDE: expected d_hat ≈ {d}, got {d_hat} (err={err})"
        );
    }

    // -----------------------------------------------------------------------
    // 2. Zero disturbance: d_hat should remain near zero
    // -----------------------------------------------------------------------
    #[test]
    fn test_zero_disturbance() {
        let a = -1.0_f64;
        let b = 1.0_f64;
        let af = 10.0_f64;
        let dt = 0.001_f64;
        let u = 0.0_f64;
        let d = 0.0_f64;
        let y0 = 0.5_f64;
        let steps = 600usize;

        let ys = simulate_plant(a, b, dt, u, d, y0, steps);
        let mut ude = UdeController::new(af, a, b, dt).expect("valid");

        let mut d_hat = 0.0_f64;
        for &y_k in &ys[..steps] {
            d_hat = ude.update(y_k, u).expect("update ok");
        }

        assert!(d_hat.abs() < 0.05, "UDE: expected d_hat ≈ 0, got {d_hat}");
    }

    // -----------------------------------------------------------------------
    // 3. Bandwidth effect: higher a_f → faster convergence (smaller error at N steps)
    // -----------------------------------------------------------------------
    #[test]
    fn test_bandwidth_effect() {
        let a = -1.0_f64;
        let b = 1.0_f64;
        let dt = 0.001_f64;
        let u = 0.0_f64;
        let d = 1.0_f64;
        let y0 = 0.0_f64;
        let steps = 600usize;

        let ys = simulate_plant(a, b, dt, u, d, y0, steps);

        let mut ude_slow = UdeController::new(5.0, a, b, dt).expect("slow valid");
        let mut ude_fast = UdeController::new(20.0, a, b, dt).expect("fast valid");

        let mut d_slow = 0.0_f64;
        let mut d_fast = 0.0_f64;
        for &y_k in &ys[..steps] {
            d_slow = ude_slow.update(y_k, u).expect("slow update ok");
            d_fast = ude_fast.update(y_k, u).expect("fast update ok");
        }

        let err_slow = (d_slow - d).abs();
        let err_fast = (d_fast - d).abs();
        assert!(
            err_fast < err_slow,
            "Higher bandwidth (af=20) should converge faster: err_fast={err_fast} err_slow={err_slow}"
        );
    }

    // -----------------------------------------------------------------------
    // 4. reset() clears the estimate and re-initialises the state machine
    // -----------------------------------------------------------------------
    #[test]
    fn test_reset() {
        let a = -1.0_f64;
        let b = 1.0_f64;
        let dt = 0.001_f64;
        let mut ude = UdeController::new(10.0, a, b, dt).expect("valid");

        // Drive the estimator with a disturbance for a while
        let ys = simulate_plant(a, b, dt, 0.0, 2.0, 0.0, 300);
        for &y_k in &ys[..300] {
            let _ = ude.update(y_k, 0.0);
        }

        ude.reset();
        assert_eq!(ude.estimate(), 0.0);

        // After reset the very first update should return 0 (initialisation)
        let first = ude.update(0.0, 0.0).expect("post-reset update ok");
        assert_eq!(first, 0.0);
    }

    // -----------------------------------------------------------------------
    // 5. Invalid configuration
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalid_af() {
        let result = UdeController::<f64>::new(0.0, -1.0, 1.0, 0.01);
        assert_eq!(result.unwrap_err(), UdeError::NonPositiveFilterBandwidth);

        let result2 = UdeController::<f64>::new(-5.0, -1.0, 1.0, 0.01);
        assert_eq!(result2.unwrap_err(), UdeError::NonPositiveFilterBandwidth);
    }

    #[test]
    fn test_invalid_dt() {
        let result = UdeController::<f64>::new(10.0, -1.0, 1.0, 0.0);
        assert_eq!(result.unwrap_err(), UdeError::NonPositiveDt);
    }

    // -----------------------------------------------------------------------
    // 6. estimate() accessor is consistent with update() return value
    // -----------------------------------------------------------------------
    #[test]
    fn test_estimate_accessor() {
        let mut ude = UdeController::new(10.0_f64, -1.0, 1.0, 0.001).expect("valid");
        // skip first call (initialisation)
        let _ = ude.update(0.0, 0.0).expect("init ok");
        // second call produces a real estimate
        let ret = ude.update(0.1, 0.05).expect("update ok");
        assert_eq!(ret, ude.estimate());
    }
}
