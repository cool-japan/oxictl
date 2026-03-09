//! Nonlinear Disturbance Observer (NDOB) — scalar version.
//!
//! Based on the formulation by Chen et al.:
//! - Chen, W.-H., Ballance, D. J., Gawthrop, P. J., & O'Reilly, J. (2000).
//!   "A nonlinear disturbance observer for robotic manipulators."
//!   IEEE Transactions on Industrial Electronics, 47(4), 932–938.
//!
//! For a scalar system:
//! ```text
//!   ẋ = f(x) + g(x)·(u + d)
//! ```
//! with the nonlinear gain choice `p(x) = l·x`, the NDOB internal state
//! equation is:
//! ```text
//!   ż = −l·g(x)·z − l·[g(x)·l·x + f(x) + g(x)·u]
//!   d̂ = z + l·x
//! ```
//! where `l > 0` is the (constant scalar) observer gain.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when constructing or updating a [`NonlinearDob`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdobError {
    /// Observer gain l must be strictly positive.
    NonPositiveGain,
    /// Sampling period dt must be strictly positive.
    NonPositiveDt,
}

impl core::fmt::Display for NdobError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NdobError::NonPositiveGain => f.write_str("observer gain l must be strictly positive"),
            NdobError::NonPositiveDt => f.write_str("sampling period dt must be strictly positive"),
        }
    }
}

// ---------------------------------------------------------------------------
// Observer struct
// ---------------------------------------------------------------------------

/// Nonlinear Disturbance Observer (NDOB) for a scalar nonlinear system.
///
/// # Observer equations (Euler discretisation)
/// ```text
///   z_dot = −l·g(x)·z − l·[g(x)·l·x + f(x) + g(x)·u]
///   z[k+1] = z[k] + dt·z_dot
///   d̂[k]  = z[k] + l·x[k]
/// ```
///
/// # Type parameters
/// - `S`: numeric scalar type implementing [`ControlScalar`] (f32 or f64).
#[derive(Debug, Clone, Copy)]
pub struct NonlinearDob<S: ControlScalar> {
    /// Internal observer state z.
    z: S,
    /// Observer gain l (> 0).
    gain: S,
    /// Sampling period.
    dt: S,
    /// Most recent disturbance estimate d̂.
    d_hat: S,
}

impl<S: ControlScalar> NonlinearDob<S> {
    /// Create a new [`NonlinearDob`].
    ///
    /// # Arguments
    /// - `gain`: observer gain `l > 0`.
    /// - `dt`: sampling period (seconds, > 0).
    /// - `x0`: initial state of the plant (used to initialise `z` so that
    ///   the initial disturbance estimate is zero).
    ///
    /// # Errors
    /// Returns [`NdobError`] if `gain ≤ 0` or `dt ≤ 0`.
    pub fn new(gain: S, dt: S, x0: S) -> Result<Self, NdobError> {
        if gain <= S::ZERO {
            return Err(NdobError::NonPositiveGain);
        }
        if dt <= S::ZERO {
            return Err(NdobError::NonPositiveDt);
        }
        // Initialise z so that d̂_0 = z + l·x0 = 0  →  z = −l·x0
        let z = -(gain * x0);
        Ok(Self {
            z,
            gain,
            dt,
            d_hat: S::ZERO,
        })
    }

    /// Run one discrete update step.
    ///
    /// # Arguments
    /// - `x`:   current plant state (scalar).
    /// - `u`:   current plant input (known).
    /// - `f_x`: value of f(x) at the current state.
    /// - `g_x`: value of g(x) at the current state.
    ///
    /// # Returns
    /// `Ok(d_hat)` — the updated disturbance estimate.
    pub fn update(&mut self, x: S, u: S, f_x: S, g_x: S) -> Result<S, NdobError> {
        // z_dot = −l·g(x)·z − l·[g(x)·l·x + f(x) + g(x)·u]
        let l = self.gain;
        let z_dot = -(l * g_x * self.z) - l * (g_x * l * x + f_x + g_x * u);

        // Euler forward integration
        self.z += z_dot * self.dt;

        // Disturbance estimate
        self.d_hat = self.z + l * x;

        Ok(self.d_hat)
    }

    /// Return the most recently computed disturbance estimate.
    #[inline]
    pub fn estimate(&self) -> S {
        self.d_hat
    }

    /// Reset the internal state, treating `x` as the current plant state
    /// (so the initial estimate is zero).
    pub fn reset(&mut self, x: S) {
        self.z = -(self.gain * x);
        self.d_hat = S::ZERO;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. Zero disturbance: x forced constant at 0, d=0 → d_hat → 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_zero_disturbance() {
        // f(x)=0, g(x)=1, u=0 → ẋ = 0, so x stays at 0 with d=0.
        let mut obs = NonlinearDob::new(2.0_f64, 0.005, 0.0).expect("valid");
        let x = 0.0_f64;
        let u = 0.0_f64;
        let f_x = 0.0_f64;
        let g_x = 1.0_f64;

        let mut d_hat = 0.0_f64;
        for _ in 0..400 {
            d_hat = obs.update(x, u, f_x, g_x).expect("update ok");
        }

        assert!(
            d_hat.abs() < 0.05,
            "Expected d_hat ≈ 0 for zero disturbance, got {d_hat}"
        );
    }

    // -----------------------------------------------------------------------
    // 2. Constant disturbance recovery
    //    Plant: ẋ = f(x) + g(x)·(u + d), f=0, g=1, u=0, d=2
    //    Integrate plant with Euler; feed x to observer (observer unaware of d).
    // -----------------------------------------------------------------------
    #[test]
    fn test_constant_disturbance() {
        let gain = 3.0_f64;
        let dt = 0.005_f64;
        let d_true = 2.0_f64;
        let f_x = 0.0_f64;
        let g_x = 1.0_f64;
        let u = 0.0_f64;

        let mut obs = NonlinearDob::new(gain, dt, 0.0_f64).expect("valid");
        let mut x = 0.0_f64;
        let mut d_hat = 0.0_f64;

        for _ in 0..600 {
            // Plant: ẋ = g_x*(u + d_true) = 2.0
            x += dt * (f_x + g_x * (u + d_true));
            d_hat = obs.update(x, u, f_x, g_x).expect("update ok");
        }

        let err = (d_hat - d_true).abs();
        assert!(
            err < 0.5 * d_true,
            "Expected d_hat ≈ {d_true}, got {d_hat} (err={err})"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Negative gain must be rejected
    // -----------------------------------------------------------------------
    #[test]
    fn test_negative_gain_error() {
        let result = NonlinearDob::<f64>::new(-1.0, 0.01, 0.0);
        assert_eq!(result.unwrap_err(), NdobError::NonPositiveGain);
    }

    // -----------------------------------------------------------------------
    // 4. Non-positive dt must be rejected
    // -----------------------------------------------------------------------
    #[test]
    fn test_nonpositive_dt_error() {
        let result = NonlinearDob::<f64>::new(1.0, 0.0, 0.0);
        assert_eq!(result.unwrap_err(), NdobError::NonPositiveDt);
    }

    // -----------------------------------------------------------------------
    // 5. Convergence speed: higher gain → smaller error after N steps
    // -----------------------------------------------------------------------
    #[test]
    fn test_convergence_speed() {
        let dt = 0.005_f64;
        let d_true = 1.0_f64;
        let f_x = 0.0_f64;
        let g_x = 1.0_f64;
        let u = 0.0_f64;
        let steps = 150usize;

        let mut obs_slow = NonlinearDob::new(1.0_f64, dt, 0.0).expect("slow valid");
        let mut obs_fast = NonlinearDob::new(5.0_f64, dt, 0.0).expect("fast valid");

        let mut x_slow = 0.0_f64;
        let mut x_fast = 0.0_f64;
        let mut d_slow = 0.0_f64;
        let mut d_fast = 0.0_f64;

        for _ in 0..steps {
            x_slow += dt * (f_x + g_x * (u + d_true));
            x_fast += dt * (f_x + g_x * (u + d_true));
            d_slow = obs_slow.update(x_slow, u, f_x, g_x).expect("slow update");
            d_fast = obs_fast.update(x_fast, u, f_x, g_x).expect("fast update");
        }

        let err_slow = (d_slow - d_true).abs();
        let err_fast = (d_fast - d_true).abs();

        assert!(
            err_fast < err_slow,
            "Higher-gain observer should converge faster: err_fast={err_fast} err_slow={err_slow}"
        );
    }

    // -----------------------------------------------------------------------
    // 6. estimate() accessor returns same value as update()
    // -----------------------------------------------------------------------
    #[test]
    fn test_estimate_accessor() {
        let mut obs = NonlinearDob::new(2.0_f64, 0.01, 0.0).expect("valid");
        let ret = obs.update(0.5, 0.0, 0.0, 1.0).expect("update ok");
        assert_eq!(ret, obs.estimate());
    }

    // -----------------------------------------------------------------------
    // 7. reset() clears state and estimate
    // -----------------------------------------------------------------------
    #[test]
    fn test_reset() {
        let dt = 0.005_f64;
        let mut obs = NonlinearDob::new(2.0, dt, 0.0).expect("valid");

        let mut x = 0.0_f64;
        for _ in 0..200 {
            x += dt * 1.0;
            let _ = obs.update(x, 0.0, 0.0, 1.0);
        }

        obs.reset(0.0);
        assert_eq!(obs.estimate(), 0.0);
    }
}
