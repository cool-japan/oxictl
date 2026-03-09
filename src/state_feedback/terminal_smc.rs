//! Terminal Sliding Mode Control (TSMC).
//!
//! Achieves finite-time convergence via a fractional-power sliding surface:
//! ```text
//!   σ = ẋ + β·|x|^(p/q)·sign(x)   (0 < p/q < 1)
//! ```
//! Because the exponent p/q < 1, the vector field on the sliding manifold
//! points toward the origin with finite-time guarantees (as opposed to the
//! asymptotic convergence of linear sliding surfaces).
//!
//! For a double-integrator plant `ẍ = u + d(t)`:
//! ```text
//!   σ̇ = ẍ + β·(p/q)·|x|^(p/q−1)·ẋ
//!   u  = −k·|σ|^r·sign(σ) − β·(p/q)·|x|^(p/q−1)·ẋ
//! ```
//! The coupling term cancels the nonlinear part of σ̇, leaving
//! `σ̇ = −k·|σ|^r·sign(σ) + d` which converges in finite time when k > |d|.
//!
//! References:
//! - Man, Z., Paplinski, A. & Wu, H. (1994). "A robust MIMO terminal sliding
//!   mode control scheme for rigid robotic manipulators." IEEE TAC 39(12).
//! - Yu, X. & Zhihong, M. (2002). "Fast terminal sliding-mode control design
//!   for nonlinear dynamical systems." IEEE TAC, 47(4), 656–660.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in terminal SMC construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalSmcError {
    /// Exponent must satisfy 0 < p < q (fractional power, finite-time guarantee).
    InvalidExponent,
    /// Control gain k must be > 0 and power r must satisfy 0 < r ≤ 1.
    InvalidGain,
    /// Sampling period dt must be strictly positive.
    InvalidDt,
}

impl core::fmt::Display for TerminalSmcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TerminalSmcError::InvalidExponent => {
                f.write_str("exponent must satisfy 0 < p < q for fractional terminal surface")
            }
            TerminalSmcError::InvalidGain => {
                f.write_str("k must be positive and r must be in (0, 1]")
            }
            TerminalSmcError::InvalidDt => f.write_str("dt must be strictly positive"),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: sign function
// ---------------------------------------------------------------------------

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
// TerminalSmc
// ---------------------------------------------------------------------------

/// Terminal Sliding Mode Controller for a double-integrator plant.
///
/// Finite-time convergence is guaranteed by the fractional power p/q < 1 in
/// the sliding surface definition.
///
/// # Type parameters
/// - `S`: scalar type implementing [`ControlScalar`] (f32 or f64).
#[derive(Debug, Clone, Copy)]
pub struct TerminalSmc<S: ControlScalar> {
    /// Surface coefficient β.
    beta: S,
    /// Numerator of fractional exponent.
    p: S,
    /// Denominator of fractional exponent (p/q < 1).
    q: S,
    /// Control gain k.
    k: S,
    /// Power of |σ| in the control law (0 < r ≤ 1).
    r: S,
    /// Sampling period.
    dt: S,
    /// Current position state.
    x: S,
    /// Current velocity state.
    xdot: S,
}

impl<S: ControlScalar> TerminalSmc<S> {
    /// Construct a new terminal sliding mode controller.
    ///
    /// # Arguments
    /// - `beta`: sliding surface coefficient β; determines the convergence rate
    ///   on the manifold.
    /// - `p`, `q`: fractional exponent numerator/denominator; must satisfy
    ///   0 < p < q (ensures 0 < p/q < 1).
    /// - `k`: control gain; must be > 0.
    /// - `r`: power of |σ| in control law; must be in (0, 1].
    /// - `dt`: sampling period; must be > 0.
    pub fn new(beta: S, p: S, q: S, k: S, r: S, dt: S) -> Result<Self, TerminalSmcError> {
        if p <= S::ZERO || q <= S::ZERO || p >= q {
            return Err(TerminalSmcError::InvalidExponent);
        }
        if k <= S::ZERO || r <= S::ZERO || r > S::ONE {
            return Err(TerminalSmcError::InvalidGain);
        }
        if dt <= S::ZERO {
            return Err(TerminalSmcError::InvalidDt);
        }
        Ok(Self {
            beta,
            p,
            q,
            k,
            r,
            dt,
            x: S::ZERO,
            xdot: S::ZERO,
        })
    }

    /// Compute the current sliding variable σ = ẋ + β·|x|^(p/q)·sign(x).
    pub fn sliding_variable(&self) -> S {
        let exponent = self.p / self.q;
        let abs_x = self.x.abs();
        let powered = if abs_x < S::EPSILON {
            S::ZERO
        } else {
            abs_x.powf(exponent)
        };
        self.xdot + self.beta * powered * sign(self.x)
    }

    /// Compute control input for the current (x, ẋ) state.
    ///
    /// Sets internal state to (x, xdot), evaluates the sliding variable,
    /// and returns:
    /// ```text
    ///   u = −k·|σ|^r·sign(σ) − β·(p/q)·|x|^(p/q−1)·ẋ
    /// ```
    pub fn update(&mut self, x: S, xdot: S) -> Result<S, TerminalSmcError> {
        self.x = x;
        self.xdot = xdot;

        let sigma = self.sliding_variable();
        let exponent = self.p / self.q;
        let coupling_exp = exponent - S::ONE; // negative since p/q < 1
        let abs_x = x.abs();

        // Coupling term β·(p/q)·|x|^(p/q−1)·ẋ
        // Singularity at x = 0 is guarded: as x → 0 and x^(p/q-1) → ∞,
        // but xdot·|x|^(p/q-1) → 0 along typical trajectories.  We guard
        // with an explicit epsilon check to avoid numerical blow-up.
        let coupling = if abs_x < S::EPSILON {
            S::ZERO
        } else {
            self.beta * exponent * abs_x.powf(coupling_exp) * xdot
        };

        // Control law: u = −k·|σ|^r·sign(σ) − coupling
        let abs_sigma = sigma.abs();
        let u = -self.k * abs_sigma.powf(self.r) * sign(sigma) - coupling;
        Ok(u)
    }

    /// Upper bound on remaining settling time (Lyapunov-based).
    ///
    /// For σ̇ = −k·|σ|^r·sign(σ), the finite-time bound is:
    /// ```text
    ///   T ≤ |σ_0|^(1−r) / (k·(1−r))
    /// ```
    /// Returns zero if σ is already at the origin or if r = 1 (linear decay).
    pub fn finite_time_estimate(&self) -> S {
        if self.r >= S::ONE {
            // r=1 gives exponential (not finite-time) convergence; return zero
            return S::ZERO;
        }
        let sigma = self.sliding_variable();
        let abs_sigma = sigma.abs();
        if abs_sigma < S::EPSILON {
            return S::ZERO;
        }
        let one_minus_r = S::ONE - self.r;
        abs_sigma.powf(one_minus_r) / (self.k * one_minus_r)
    }

    /// Reset state to (x, xdot).
    pub fn reset(&mut self, x: S, xdot: S) {
        self.x = x;
        self.xdot = xdot;
    }

    /// Return sampling period.
    pub fn dt(&self) -> S {
        self.dt
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.001;

    // 1. Invalid p >= q → InvalidExponent
    #[test]
    fn invalid_exponent_p_geq_q() {
        let res = TerminalSmc::<f64>::new(1.0, 2.0, 1.0, 5.0, 0.5, DT);
        assert!(
            matches!(res, Err(TerminalSmcError::InvalidExponent)),
            "expected InvalidExponent, got {:?}",
            res.err()
        );
    }

    // 2. p = q (equal) → InvalidExponent
    #[test]
    fn invalid_exponent_p_eq_q() {
        let res = TerminalSmc::<f64>::new(1.0, 3.0, 3.0, 5.0, 0.5, DT);
        assert!(
            matches!(res, Err(TerminalSmcError::InvalidExponent)),
            "expected InvalidExponent, got {:?}",
            res.err()
        );
    }

    // 3. k=0 → InvalidGain
    #[test]
    fn invalid_k_zero_returns_error() {
        let res = TerminalSmc::<f64>::new(1.0, 1.0, 2.0, 0.0, 0.5, DT);
        assert!(
            matches!(res, Err(TerminalSmcError::InvalidGain)),
            "expected InvalidGain, got {:?}",
            res.err()
        );
    }

    // 4. r > 1 → InvalidGain
    #[test]
    fn invalid_r_gt_one_returns_error() {
        let res = TerminalSmc::<f64>::new(1.0, 1.0, 2.0, 5.0, 1.5, DT);
        assert!(
            matches!(res, Err(TerminalSmcError::InvalidGain)),
            "expected InvalidGain, got {:?}",
            res.err()
        );
    }

    // 5. dt=0 → InvalidDt
    #[test]
    fn invalid_dt_returns_error() {
        let res = TerminalSmc::<f64>::new(1.0, 1.0, 2.0, 5.0, 0.5, 0.0);
        assert!(
            matches!(res, Err(TerminalSmcError::InvalidDt)),
            "expected InvalidDt, got {:?}",
            res.err()
        );
    }

    // 6. Zero state → zero control (σ=0 at origin, coupling=0)
    #[test]
    fn zero_state_zero_control() {
        let mut ctrl = TerminalSmc::<f64>::new(1.0, 1.0, 2.0, 5.0, 0.5, DT).expect("valid params");
        let u = ctrl.update(0.0, 0.0).expect("update ok");
        assert!(u.abs() < 1e-14, "u should be zero at origin, got {}", u);
    }

    // 7. Positive sliding variable → negative control (correct sign)
    // x=0 (coupling=0), xdot=1 → σ = 1 → u = -k*1^r*1 < 0
    #[test]
    fn positive_sigma_gives_negative_control() {
        let mut ctrl = TerminalSmc::<f64>::new(1.0, 1.0, 2.0, 5.0, 0.5, DT).expect("valid params");
        // x=0 so coupling term is zero; xdot=1 so sigma=1 > 0
        let u = ctrl.update(0.0, 1.0).expect("update ok");
        assert!(
            u < 0.0,
            "u should be negative for positive sigma, got {}",
            u
        );
    }

    // 8. Power formula check:
    //    beta=1, p=1, q=2, k=2, r=1, x=4, xdot=0
    //    sigma = 0 + 1*4^0.5*1 = 2
    //    coupling = 1*(1/2)*4^(-0.5)*0 = 0
    //    u = -2*2^1*1 - 0 = -4
    #[test]
    fn formula_verification() {
        let beta = 1.0_f64;
        let p = 1.0_f64;
        let q = 2.0_f64;
        let k = 2.0_f64;
        let r = 1.0_f64;
        let mut ctrl = TerminalSmc::<f64>::new(beta, p, q, k, r, DT).expect("valid params");
        let u = ctrl.update(4.0, 0.0).expect("update ok");
        let expected = -4.0_f64;
        assert!(
            (u - expected).abs() < 1e-10,
            "expected u={}, got u={}",
            expected,
            u
        );
    }

    // 9. Sliding variable decreases over time on a double integrator
    #[test]
    fn sliding_variable_converges() {
        // beta=1, p=1, q=2 (exponent=0.5), k=5, r=0.5
        let mut ctrl = TerminalSmc::<f64>::new(1.0, 1.0, 2.0, 5.0, 0.5, DT).expect("valid params");

        let mut x = 1.0_f64;
        let mut xdot = 0.0_f64;

        ctrl.reset(x, xdot);
        let sigma_init = ctrl.sliding_variable().abs();

        for _ in 0..2000 {
            let u = ctrl.update(x, xdot).expect("update ok");
            // Euler integration: double integrator ẍ = u
            x += DT * xdot;
            xdot += DT * u;
        }

        let sigma_final = ctrl.sliding_variable().abs();
        assert!(
            sigma_final < sigma_init,
            "sliding variable should decrease: sigma_init={:.4}, sigma_final={:.4}",
            sigma_init,
            sigma_final
        );
    }

    // 10. finite_time_estimate returns non-negative value
    #[test]
    fn finite_time_estimate_nonnegative() {
        let mut ctrl = TerminalSmc::<f64>::new(1.0, 1.0, 2.0, 5.0, 0.5, DT).expect("valid params");
        ctrl.update(1.0, 0.5).expect("update ok");
        let t_est = ctrl.finite_time_estimate();
        assert!(
            t_est >= 0.0,
            "finite time estimate must be non-negative: {}",
            t_est
        );
    }
}
