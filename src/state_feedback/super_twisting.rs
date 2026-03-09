//! Super-Twisting Algorithm (STA) — 2nd-order sliding mode controller.
//!
//! The super-twisting algorithm (Levant 1993) is a second-order sliding mode
//! controller that drives both the sliding variable σ and its derivative σ̇ to
//! zero in finite time despite bounded matched disturbances.
//!
//! Algorithm (continuous form):
//! ```text
//!   u  = -k1·|σ|^(1/2)·sign(σ) + v
//!   v̇  = -k2·sign(σ)
//! ```
//!
//! Discrete implementation (Euler for v):
//! ```text
//!   u[n]   = -k1·|σ[n]|^(1/2)·sign(σ[n]) + v[n]
//!   v[n+1] = v[n] - k2·sign(σ[n])·dt
//! ```
//!
//! Stability condition (with perturbation bound W):
//!   k1 > 2·√W,  k2 > W + k1²/2
//!
//! References:
//! - Levant, A. (1993). "Sliding order and sliding accuracy in sliding mode
//!   control." International Journal of Control, 58(6), 1247–1263.
//! - Moreno, J.A. & Osorio, M. (2012). "Strict Lyapunov functions for the
//!   super-twisting algorithm." IEEE TAC, 57(4), 1035–1040.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in super-twisting construction or operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuperTwistingError {
    /// All gains (k1, k2) must be strictly positive.
    InvalidGain,
    /// Sampling period dt must be strictly positive.
    InvalidDt,
}

impl core::fmt::Display for SuperTwistingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SuperTwistingError::InvalidGain => {
                f.write_str("super-twisting gains k1 and k2 must be strictly positive")
            }
            SuperTwistingError::InvalidDt => {
                f.write_str("sampling period dt must be strictly positive")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: sign function
// ---------------------------------------------------------------------------

/// Discontinuous sign function: +1 if x > 0, -1 if x < 0, 0 if x = 0.
#[inline]
fn sign<S: ControlScalar>(x: S) -> S {
    if x > S::ZERO {
        S::ONE
    } else if x < S::ZERO {
        -S::ONE
    } else {
        S::ZERO
    }
}

// ---------------------------------------------------------------------------
// SuperTwistingController
// ---------------------------------------------------------------------------

/// Super-Twisting Algorithm controller (2nd-order sliding mode, SISO).
///
/// Drives σ and σ̇ to zero in finite time.  Accepts the sliding variable σ
/// at each step and returns the control effort u.
///
/// # Type parameters
/// - `S`: scalar type implementing [`ControlScalar`] (f32 or f64).
#[derive(Debug, Clone, Copy)]
pub struct SuperTwistingController<S: ControlScalar> {
    /// Proportional gain on |σ|^(1/2).
    k1: S,
    /// Integral (sign) gain.
    k2: S,
    /// Internal integral state v.
    v: S,
    /// Sampling period.
    dt: S,
    /// Previous sliding variable (retained for diagnostics / extensions).
    sigma_prev: S,
}

impl<S: ControlScalar> SuperTwistingController<S> {
    /// Construct a new super-twisting controller.
    ///
    /// # Arguments
    /// - `k1`: gain on |σ|^(1/2)·sign(σ); must be > 0.
    /// - `k2`: integral sign gain; must be > 0.
    /// - `dt`: sampling period; must be > 0.
    pub fn new(k1: S, k2: S, dt: S) -> Result<Self, SuperTwistingError> {
        if k1 <= S::ZERO || k2 <= S::ZERO {
            return Err(SuperTwistingError::InvalidGain);
        }
        if dt <= S::ZERO {
            return Err(SuperTwistingError::InvalidDt);
        }
        Ok(Self {
            k1,
            k2,
            v: S::ZERO,
            dt,
            sigma_prev: S::ZERO,
        })
    }

    /// Compute the control output for the current sliding variable.
    ///
    /// Updates the internal integral state v and returns:
    /// ```text
    ///   u[n] = -k1·|σ|^(1/2)·sign(σ) + v[n]
    /// ```
    /// then advances:
    /// ```text
    ///   v[n+1] = v[n] - k2·sign(σ)·dt
    /// ```
    pub fn update(&mut self, sigma: S) -> Result<S, SuperTwistingError> {
        let half = S::from_f64(0.5);
        let abs_sigma = sigma.abs();
        let sqrt_sigma = abs_sigma.powf(half);

        // u = -k1·|σ|^(1/2)·sign(σ) + v
        let u = -self.k1 * sqrt_sigma * sign(sigma) + self.v;

        // v[n+1] = v[n] - k2·sign(σ)·dt
        self.v -= self.k2 * sign(sigma) * self.dt;

        self.sigma_prev = sigma;
        Ok(u)
    }

    /// Reset internal state (integral v and sigma_prev) to zero.
    pub fn reset(&mut self) {
        self.v = S::ZERO;
        self.sigma_prev = S::ZERO;
    }

    /// Return the current value of the internal integral state v.
    pub fn integral_state(&self) -> S {
        self.v
    }

    /// Return the previous sliding variable (last value passed to `update`).
    pub fn sigma_prev(&self) -> S {
        self.sigma_prev
    }
}

// ---------------------------------------------------------------------------
// AdaptiveSuperTwisting
// ---------------------------------------------------------------------------

/// Adaptive Super-Twisting controller (Modified STA) with gain adaptation.
///
/// Extends [`SuperTwistingController`] with an adaptive k1 that grows when
/// |σ| exceeds the dead-zone threshold ε and shrinks (floored at k1_min)
/// when |σ| ≤ ε.  This prevents over-amplification in the absence of
/// large disturbances while remaining responsive to them.
///
/// Adaptation law:
/// ```text
///   if |σ| > ε:  k1 ← k1 + k1_dot·dt
///   else:         k1 ← max(k1 - k1_dot·dt, k1_min)
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveSuperTwisting<S: ControlScalar> {
    /// Underlying super-twisting controller (k1 is overwritten each step).
    base: SuperTwistingController<S>,
    /// Current adaptive k1.
    k1_adapt: S,
    /// Rate of k1 adaptation.
    k1_dot: S,
    /// Dead-zone threshold for adaptation.
    epsilon: S,
    /// Minimum value for k1 (= initial k1).
    k1_min: S,
}

impl<S: ControlScalar> AdaptiveSuperTwisting<S> {
    /// Construct an adaptive super-twisting controller.
    ///
    /// # Arguments
    /// - `k1_init`: initial (and minimum) k1; must be > 0.
    /// - `k2`: integral sign gain; must be > 0.
    /// - `k1_dot`: adaptation rate; must be > 0.
    /// - `epsilon`: dead-zone size; must be > 0.
    /// - `dt`: sampling period; must be > 0.
    pub fn new(
        k1_init: S,
        k2: S,
        k1_dot: S,
        epsilon: S,
        dt: S,
    ) -> Result<Self, SuperTwistingError> {
        if k1_init <= S::ZERO || k2 <= S::ZERO || k1_dot <= S::ZERO || epsilon <= S::ZERO {
            return Err(SuperTwistingError::InvalidGain);
        }
        if dt <= S::ZERO {
            return Err(SuperTwistingError::InvalidDt);
        }
        let base = SuperTwistingController::new(k1_init, k2, dt)?;
        Ok(Self {
            base,
            k1_adapt: k1_init,
            k1_dot,
            epsilon,
            k1_min: k1_init,
        })
    }

    /// Compute control output with adaptive gain.
    ///
    /// Adjusts k1 based on |σ| vs. the dead-zone ε, then delegates to the
    /// base super-twisting update.
    pub fn update(&mut self, sigma: S) -> Result<S, SuperTwistingError> {
        // Adapt k1
        if sigma.abs() > self.epsilon {
            self.k1_adapt += self.k1_dot * self.base.dt;
        } else {
            let candidate = self.k1_adapt - self.k1_dot * self.base.dt;
            self.k1_adapt = if candidate > self.k1_min {
                candidate
            } else {
                self.k1_min
            };
        }
        // Propagate updated gain to base controller
        self.base.k1 = self.k1_adapt;
        self.base.update(sigma)
    }

    /// Return the current adaptive k1.
    pub fn adaptive_gain(&self) -> S {
        self.k1_adapt
    }

    /// Reset internal state and restore k1 to k1_min.
    pub fn reset(&mut self) {
        self.base.reset();
        self.k1_adapt = self.k1_min;
        self.base.k1 = self.k1_min;
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.001;

    // 1. Zero sigma → zero output (v=0 initially, |0|^0.5 = 0)
    #[test]
    fn zero_sigma_zero_output() {
        let mut ctrl = SuperTwistingController::<f64>::new(2.0, 5.0, DT).expect("valid params");
        let u = ctrl.update(0.0).expect("update ok");
        assert!(u.abs() < 1e-14, "u should be zero for sigma=0, got {}", u);
    }

    // 2. Non-zero sigma → non-zero output
    #[test]
    fn nonzero_sigma_nonzero_output() {
        let mut ctrl = SuperTwistingController::<f64>::new(2.0, 5.0, DT).expect("valid params");
        let u = ctrl.update(1.0).expect("update ok");
        assert!(
            u.abs() > 1e-10,
            "u should be non-zero for sigma=1, got {}",
            u
        );
    }

    // 3. Invalid k1=0 → error
    #[test]
    fn invalid_k1_returns_error() {
        let res = SuperTwistingController::<f64>::new(0.0, 5.0, DT);
        assert!(
            matches!(res, Err(SuperTwistingError::InvalidGain)),
            "expected InvalidGain, got {:?}",
            res.err()
        );
    }

    // 4. Invalid dt=0 → error
    #[test]
    fn invalid_dt_returns_error() {
        let res = SuperTwistingController::<f64>::new(2.0, 5.0, 0.0);
        assert!(
            matches!(res, Err(SuperTwistingError::InvalidDt)),
            "expected InvalidDt, got {:?}",
            res.err()
        );
    }

    // 5. Integral state decreases for positive sigma (k2·sign(+1)·dt > 0)
    #[test]
    fn integral_state_decreases_for_positive_sigma() {
        let mut ctrl = SuperTwistingController::<f64>::new(2.0, 5.0, DT).expect("valid params");
        let v0 = ctrl.integral_state();
        ctrl.update(1.0).expect("update ok");
        let v1 = ctrl.integral_state();
        assert!(
            v1 < v0,
            "v should decrease for positive sigma: v0={}, v1={}",
            v0,
            v1
        );
    }

    // 6. Known formula: sigma=1, k1=3, k2=5 → u = -3·1^0.5·1 + 0 = -3
    #[test]
    fn formula_verification_unit_sigma() {
        let k1 = 3.0_f64;
        let k2 = 5.0_f64;
        let mut ctrl = SuperTwistingController::<f64>::new(k1, k2, DT).expect("valid params");
        let u = ctrl.update(1.0).expect("update ok");
        let expected = -k1; // -k1 * sqrt(1) * sign(1) + v(=0)
        assert!(
            (u - expected).abs() < 1e-12,
            "expected u={}, got u={}",
            expected,
            u
        );
    }

    // 7. Adaptive gain grows with persistent large sigma
    #[test]
    fn adaptive_gain_grows_with_large_sigma() {
        let mut ctrl =
            AdaptiveSuperTwisting::<f64>::new(1.0, 5.0, 0.5, 0.1, DT).expect("valid params");
        let g0 = ctrl.adaptive_gain();
        // Large sigma (> epsilon=0.1) triggers gain increase every step
        for _ in 0..100 {
            ctrl.update(10.0).expect("update ok");
        }
        let g1 = ctrl.adaptive_gain();
        assert!(
            g1 > g0,
            "gain should grow with large sigma: g0={}, g1={}",
            g0,
            g1
        );
    }

    // 8. STA drives sliding variable magnitude downward on a double integrator
    #[test]
    fn sta_drives_sigma_to_zero() {
        // Double integrator: ẍ = u  (plant)
        // Sliding surface: σ = ẋ + 5·x
        let k1 = 3.0_f64;
        let k2 = 5.0_f64;
        let c = 5.0_f64;
        let mut ctrl = SuperTwistingController::<f64>::new(k1, k2, DT).expect("valid params");

        let mut x = 1.0_f64;
        let mut xdot = 0.0_f64;

        let sigma_init = xdot + c * x;

        for _ in 0..5000 {
            let sigma = xdot + c * x;
            let u = ctrl.update(sigma).expect("update ok");
            // Euler integration of double integrator
            x += DT * xdot;
            xdot += DT * u;
        }

        let sigma_final = xdot + c * x;
        assert!(
            sigma_final.abs() < sigma_init.abs(),
            "sigma should decrease: sigma_init={:.4}, sigma_final={:.4}",
            sigma_init,
            sigma_final
        );
    }
}
