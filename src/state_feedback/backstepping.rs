//! Recursive backstepping control for strict-feedback nonlinear systems.
//!
//! Backstepping is a Lyapunov-based recursive design methodology for
//! strict-feedback systems of the form:
//! ```text
//!   ẋ₁ = f₁(x₁) + g₁(x₁)·x₂
//!   ẋ₂ = f₂(x₁,x₂) + g₂(x₁,x₂)·x₃
//!    ⋮
//!   ẋₙ = fₙ(x) + gₙ(x)·u
//! ```
//!
//! At each step `i`, a *virtual control* α_i is designed so that the
//! sub-system (x₁,…,xᵢ) is rendered stable by treating xᵢ₊₁ = α_i as a
//! fictitious input. The true input `u` is computed at the final step.
//!
//! A Control Lyapunov Function (CLF) `V_i = ½ z_i²` is used at each step,
//! where `z_i = x_i - α_{i-1}` is the tracking error on the virtual state.
//!
//! References:
//! - Krstić, M., Kanellakopoulos, I. & Kokotović, P. (1995).
//!   *Nonlinear and Adaptive Control Design*. Wiley-Interscience.
//! - Khalil, H.K. (2002). *Nonlinear Systems* (3rd ed.), Prentice Hall.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that may occur in backstepping controller construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BacksteppingError {
    /// All gains c_i must be strictly positive.
    NonPositiveGain,
    /// g_i (virtual input gain) must not be zero at the operating point.
    ZeroInputGain,
    /// Sampling period must be strictly positive.
    NonPositiveDt,
}

impl core::fmt::Display for BacksteppingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BacksteppingError::NonPositiveGain => {
                f.write_str("all backstepping gains c_i must be positive")
            }
            BacksteppingError::ZeroInputGain => {
                f.write_str("virtual input gain g_i must be non-zero")
            }
            BacksteppingError::NonPositiveDt => f.write_str("dt must be positive"),
        }
    }
}

// ---------------------------------------------------------------------------
// 2nd-order backstepping (linearised strict-feedback)
// ---------------------------------------------------------------------------

/// Second-order backstepping controller for a strict-feedback system:
/// ```text
///   ẋ₁ = f₁(x₁) + g₁·x₂
///   ẋ₂ = f₂(x₁,x₂) + g₂·u
/// ```
/// where `f₁`, `f₂` are provided as closures and `g₁`, `g₂` are constant
/// gains evaluated at design time (gain-scheduling extensions can be built
/// on top of this structure).
///
/// **Design**
///
/// Step 1 — define z₁ = x₁ - x₁_ref. Virtual control:
/// ```text
///   α₁ = (1/g₁)·(-f₁(x₁) + ẋ₁_ref - c₁·z₁)
/// ```
///
/// Step 2 — define z₂ = x₂ - α₁. True control:
/// ```text
///   u = (1/g₂)·(-f₂(x₁,x₂) + α̇₁ - c₂·z₂ - g₁·z₁)
/// ```
/// where α̇₁ is approximated numerically from the previous step.
///
/// # Type parameters
/// - `S`: scalar type.
pub struct SecondOrderBackstepping<S: ControlScalar> {
    /// Backstepping gain c₁ for the first subsystem.
    c1: S,
    /// Backstepping gain c₂ for the second subsystem.
    c2: S,
    /// Constant virtual input gain g₁.
    g1: S,
    /// Constant input gain g₂.
    g2: S,
    /// Sampling period (for numerical differentiation of α₁).
    dt: S,
    /// Stored α₁ from the previous step (for α̇₁ approximation).
    alpha1_prev: S,
    /// Whether alpha1_prev is valid.
    has_prev: bool,
}

impl<S: ControlScalar> SecondOrderBackstepping<S> {
    /// Construct a second-order backstepping controller.
    ///
    /// # Arguments
    /// - `c1`, `c2`: CLF damping gains (both must be > 0).
    /// - `g1`, `g2`: virtual / true input gains (both must be ≠ 0).
    /// - `dt`: sampling period.
    pub fn new(c1: S, c2: S, g1: S, g2: S, dt: S) -> Result<Self, BacksteppingError> {
        if c1 <= S::ZERO || c2 <= S::ZERO {
            return Err(BacksteppingError::NonPositiveGain);
        }
        if g1 == S::ZERO || g2 == S::ZERO {
            return Err(BacksteppingError::ZeroInputGain);
        }
        if dt <= S::ZERO {
            return Err(BacksteppingError::NonPositiveDt);
        }

        Ok(Self {
            c1,
            c2,
            g1,
            g2,
            dt,
            alpha1_prev: S::ZERO,
            has_prev: false,
        })
    }

    /// Reset internal state (α₁ memory).
    pub fn reset(&mut self) {
        self.alpha1_prev = S::ZERO;
        self.has_prev = false;
    }

    /// Compute the virtual control α₁ for step 1.
    ///
    /// # Arguments
    /// - `x1`: first state.
    /// - `x1_ref`: reference for x₁.
    /// - `dx1_ref`: reference derivative ẋ₁_ref.
    /// - `f1`: value of f₁(x₁) at the current state.
    pub fn virtual_control(&self, x1: S, x1_ref: S, dx1_ref: S, f1: S) -> S {
        let z1 = x1 - x1_ref;
        // α₁ = (1/g₁)·(-f₁ + ẋ₁_ref - c₁·z₁)
        (-f1 + dx1_ref - self.c1 * z1) / self.g1
    }

    /// Compute the true control input `u`.
    ///
    /// # Arguments
    /// - `x1`, `x2`: states.
    /// - `x1_ref`, `dx1_ref`: reference and its derivative.
    /// - `f1`: f₁(x₁) evaluated at current state.
    /// - `f2`: f₂(x₁,x₂) evaluated at current state.
    ///
    /// # Returns
    /// True control input u.
    pub fn update(&mut self, x1: S, x2: S, x1_ref: S, dx1_ref: S, f1: S, f2: S) -> S {
        let alpha1 = self.virtual_control(x1, x1_ref, dx1_ref, f1);

        // Numerical derivative of α₁
        let dalpha1 = if self.has_prev {
            (alpha1 - self.alpha1_prev) / self.dt
        } else {
            S::ZERO
        };
        self.alpha1_prev = alpha1;
        self.has_prev = true;

        let z1 = x1 - x1_ref;
        let z2 = x2 - alpha1;

        // u = (1/g₂)·(-f₂ + α̇₁ - c₂·z₂ - g₁·z₁)
        (-f2 + dalpha1 - self.c2 * z2 - self.g1 * z1) / self.g2
    }
}

// ---------------------------------------------------------------------------
// 3rd-order backstepping
// ---------------------------------------------------------------------------

/// Third-order backstepping controller for:
/// ```text
///   ẋ₁ = f₁(x₁) + g₁·x₂
///   ẋ₂ = f₂(x₁,x₂) + g₂·x₃
///   ẋ₃ = f₃(x₁,x₂,x₃) + g₃·u
/// ```
///
/// Virtual controls are:
/// ```text
///   α₁ = (1/g₁)·(-f₁ + ẋ₁_ref - c₁·z₁)
///   α₂ = (1/g₂)·(-f₂ + α̇₁ - c₂·z₂ - g₁·z₁)
///   u  = (1/g₃)·(-f₃ + α̇₂ - c₃·z₃ - g₂·z₂)
/// ```
///
/// Numerical differentiation is used for α̇₁ and α̇₂.
///
/// # Type parameters
/// - `S`: scalar type.
pub struct ThirdOrderBackstepping<S: ControlScalar> {
    c1: S,
    c2: S,
    c3: S,
    g1: S,
    g2: S,
    g3: S,
    dt: S,
    alpha1_prev: S,
    alpha2_prev: S,
    has_prev: bool,
}

impl<S: ControlScalar> ThirdOrderBackstepping<S> {
    /// Construct a third-order backstepping controller.
    ///
    /// # Arguments
    /// - `c1`, `c2`, `c3`: CLF gains (all must be > 0).
    /// - `g1`, `g2`, `g3`: virtual / true input gains (all must be ≠ 0).
    /// - `dt`: sampling period.
    #[allow(clippy::too_many_arguments)]
    pub fn new(c1: S, c2: S, c3: S, g1: S, g2: S, g3: S, dt: S) -> Result<Self, BacksteppingError> {
        if c1 <= S::ZERO || c2 <= S::ZERO || c3 <= S::ZERO {
            return Err(BacksteppingError::NonPositiveGain);
        }
        if g1 == S::ZERO || g2 == S::ZERO || g3 == S::ZERO {
            return Err(BacksteppingError::ZeroInputGain);
        }
        if dt <= S::ZERO {
            return Err(BacksteppingError::NonPositiveDt);
        }

        Ok(Self {
            c1,
            c2,
            c3,
            g1,
            g2,
            g3,
            dt,
            alpha1_prev: S::ZERO,
            alpha2_prev: S::ZERO,
            has_prev: false,
        })
    }

    /// Reset stored virtual control history.
    pub fn reset(&mut self) {
        self.alpha1_prev = S::ZERO;
        self.alpha2_prev = S::ZERO;
        self.has_prev = false;
    }

    /// Compute true control input u.
    ///
    /// # Arguments
    /// - `x1`, `x2`, `x3`: system states.
    /// - `x1_ref`, `dx1_ref`: reference and its derivative.
    /// - `f1`: f₁(x₁).
    /// - `f2`: f₂(x₁,x₂).
    /// - `f3`: f₃(x₁,x₂,x₃).
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, x1: S, x2: S, x3: S, x1_ref: S, dx1_ref: S, f1: S, f2: S, f3: S) -> S {
        // Step 1
        let z1 = x1 - x1_ref;
        let alpha1 = (-f1 + dx1_ref - self.c1 * z1) / self.g1;

        // Step 2
        let z2 = x2 - alpha1;
        let dalpha1 = if self.has_prev {
            (alpha1 - self.alpha1_prev) / self.dt
        } else {
            S::ZERO
        };
        let alpha2 = (-f2 + dalpha1 - self.c2 * z2 - self.g1 * z1) / self.g2;

        // Step 3
        let z3 = x3 - alpha2;
        let dalpha2 = if self.has_prev {
            (alpha2 - self.alpha2_prev) / self.dt
        } else {
            S::ZERO
        };

        // True control
        let u = (-f3 + dalpha2 - self.c3 * z3 - self.g2 * z2) / self.g3;

        // Store for next step
        self.alpha1_prev = alpha1;
        self.alpha2_prev = alpha2;
        self.has_prev = true;

        u
    }
}

// ---------------------------------------------------------------------------
// Helper: simple backstepping for integrator chains (linear case)
// ---------------------------------------------------------------------------

/// Backstepping controller specialised for a chain of integrators:
/// ```text
///   ẋ₁ = x₂
///   ẋ₂ = x₃    (for 3rd-order)
///    ⋮
///   ẋₙ = u
/// ```
/// This is the linear case where all f_i = 0 and all g_i = 1, which
/// reduces backstepping to a simple pole-placement-like law.
///
/// The resulting gains are equivalent to choosing all virtual poles at
/// `-c_i` in continuous time.
///
/// Supports 2-state and 3-state systems via associated methods.
#[derive(Debug, Clone, Copy)]
pub struct IntegratorChainBackstepping<S: ControlScalar, const N: usize> {
    /// CLF gains [c₁, …, cₙ].
    gains: [S; N],
}

impl<S: ControlScalar, const N: usize> IntegratorChainBackstepping<S, N> {
    /// Construct from a gain array.  All gains must be positive.
    pub fn new(gains: [S; N]) -> Result<Self, BacksteppingError> {
        for &g in &gains {
            if g <= S::ZERO {
                return Err(BacksteppingError::NonPositiveGain);
            }
        }
        Ok(Self { gains })
    }
}

impl<S: ControlScalar> IntegratorChainBackstepping<S, 2> {
    /// Compute control for a 2nd-order integrator chain (ẍ = u).
    ///
    /// # Arguments
    /// - `x`: [x₁, x₂] state.
    /// - `x1_ref`: reference for x₁.
    /// - `dx1_ref`: reference velocity.
    pub fn control(&self, x: &[S; 2], x1_ref: S, dx1_ref: S) -> S {
        let c1 = self.gains[0];
        let c2 = self.gains[1];

        let z1 = x[0] - x1_ref;
        // α₁ = ẋ₁_ref - c₁·z₁  (since f₁=0, g₁=1)
        let alpha1 = dx1_ref - c1 * z1;

        let z2 = x[1] - alpha1;
        // u = -c₂·z₂ - z₁  (since f₂=0, g₂=1, α̇₁ ≈ 0 or handled externally)
        -c2 * z2 - z1
    }
}

impl<S: ControlScalar> IntegratorChainBackstepping<S, 3> {
    /// Compute control for a 3rd-order integrator chain (x⃛ = u).
    ///
    /// Uses `alpha1_dot` for the derivative of α₁ (pass zero if not available).
    ///
    /// # Arguments
    /// - `x`: [x₁, x₂, x₃] state.
    /// - `x1_ref`: reference.
    /// - `dx1_ref`: ẋ₁_ref.
    /// - `alpha1_dot`: numerical derivative of α₁ from the previous step.
    pub fn control(&self, x: &[S; 3], x1_ref: S, dx1_ref: S, alpha1_dot: S) -> S {
        let c1 = self.gains[0];
        let c2 = self.gains[1];
        let c3 = self.gains[2];

        let z1 = x[0] - x1_ref;
        let alpha1 = dx1_ref - c1 * z1;

        let z2 = x[1] - alpha1;
        let alpha2 = alpha1_dot - c2 * z2 - z1;

        let z3 = x[2] - alpha2;
        -c3 * z3 - z2
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.001;

    // Simulate: ẋ₁ = x₂,  ẋ₂ = u  (double integrator)
    fn step_double_integrator(state: [f64; 2], u: f64) -> [f64; 2] {
        [state[0] + DT * state[1], state[1] + DT * u]
    }

    // Triple integrator: ẋ₁=x₂, ẋ₂=x₃, ẋ₃=u
    fn step_triple_integrator(state: [f64; 3], u: f64) -> [f64; 3] {
        [
            state[0] + DT * state[1],
            state[1] + DT * state[2],
            state[2] + DT * u,
        ]
    }

    #[test]
    fn backstepping_2nd_order_invalid_gains() {
        assert!(SecondOrderBackstepping::<f64>::new(0.0, 2.0, 1.0, 1.0, DT).is_err());
        assert!(SecondOrderBackstepping::<f64>::new(1.0, 0.0, 1.0, 1.0, DT).is_err());
        assert!(SecondOrderBackstepping::<f64>::new(1.0, 2.0, 0.0, 1.0, DT).is_err());
        assert!(SecondOrderBackstepping::<f64>::new(1.0, 2.0, 1.0, 0.0, DT).is_err());
    }

    #[test]
    fn backstepping_2nd_order_stabilises_double_integrator() {
        // ẋ₁ = x₂ (f₁=0, g₁=1), ẋ₂ = u (f₂=0, g₂=1)
        let mut ctrl =
            SecondOrderBackstepping::<f64>::new(5.0, 5.0, 1.0, 1.0, DT).expect("valid params");

        let r = 1.0_f64;
        let mut state = [0.0_f64; 2];

        for _ in 0..4000 {
            let u = ctrl.update(state[0], state[1], r, 0.0, 0.0, 0.0);
            state = step_double_integrator(state, u);
        }

        assert!(
            (state[0] - r).abs() < 0.05,
            "x₁={:.4} should converge to r={}",
            state[0],
            r
        );
    }

    #[test]
    fn backstepping_3rd_order_stabilises_triple_integrator() {
        let mut ctrl = ThirdOrderBackstepping::<f64>::new(4.0, 4.0, 4.0, 1.0, 1.0, 1.0, DT)
            .expect("valid params");

        let r = 1.0_f64;
        let mut state = [0.0_f64; 3];

        for _ in 0..8000 {
            let u = ctrl.update(state[0], state[1], state[2], r, 0.0, 0.0, 0.0, 0.0);
            state = step_triple_integrator(state, u);
        }

        assert!(
            (state[0] - r).abs() < 0.05,
            "x₁={:.4} should converge to r={}",
            state[0],
            r
        );
    }

    #[test]
    fn integrator_chain_2nd_order_converges() {
        let ctrl = IntegratorChainBackstepping::<f64, 2>::new([3.0, 5.0]).expect("valid gains");

        let r = 2.0_f64;
        let mut state = [0.0_f64; 2];

        for _ in 0..5000 {
            let u = ctrl.control(&state, r, 0.0);
            state = step_double_integrator(state, u);
        }

        assert!(
            (state[0] - r).abs() < 0.1,
            "x₁={:.4} should track r={}",
            state[0],
            r
        );
    }

    #[test]
    fn integrator_chain_3rd_order_converges() {
        let ctrl =
            IntegratorChainBackstepping::<f64, 3>::new([3.0, 4.0, 5.0]).expect("valid gains");

        let r = 1.5_f64;
        let mut state = [0.0_f64; 3];
        let mut alpha1_prev = 0.0_f64;

        for _ in 0..10000 {
            let c1 = 3.0_f64;
            let z1 = state[0] - r;
            let alpha1 = -c1 * z1; // ẋ₁_ref=0
            let alpha1_dot = (alpha1 - alpha1_prev) / DT;
            alpha1_prev = alpha1;

            let u = ctrl.control(&state, r, 0.0, alpha1_dot);
            state = step_triple_integrator(state, u);
        }

        assert!(
            (state[0] - r).abs() < 0.1,
            "x₁={:.4} should track r={}",
            state[0],
            r
        );
    }

    #[test]
    fn integrator_chain_zero_gain_rejected() {
        let res = IntegratorChainBackstepping::<f64, 2>::new([3.0, 0.0]);
        assert!(res.is_err());
    }

    #[test]
    fn third_order_backstepping_invalid() {
        assert!(ThirdOrderBackstepping::<f64>::new(-1.0, 2.0, 2.0, 1.0, 1.0, 1.0, DT).is_err());
        assert!(ThirdOrderBackstepping::<f64>::new(1.0, 2.0, 2.0, 0.0, 1.0, 1.0, DT).is_err());
    }

    #[test]
    fn second_order_backstepping_reset() {
        let mut ctrl = SecondOrderBackstepping::<f64>::new(2.0, 3.0, 1.0, 1.0, DT).expect("valid");
        // Run a few steps to populate internal state
        for _ in 0..10 {
            let _ = ctrl.update(0.5, 0.1, 1.0, 0.0, 0.0, 0.0);
        }
        ctrl.reset();
        assert!(!ctrl.has_prev);
    }
}
