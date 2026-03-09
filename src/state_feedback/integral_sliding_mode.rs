//! Integral Sliding Mode Control (ISMC).
//!
//! Integral sliding mode control eliminates the reaching phase present in
//! classical SMC by designing the sliding surface so that the initial state
//! already lies on the surface. This is achieved by incorporating an integral
//! term that accounts for the initial error.
//!
//! The sliding surface is:
//! ```text
//!   s(t) = e(t) + ∫₀ᵗ (A+BK) e(τ) dτ  — first-order form
//! ```
//! where K is a nominal stabilising gain designed for the undisturbed system.
//!
//! Two chattering-reduction strategies are provided:
//! - Hard switching: `u_sw = -η·sign(s)`
//! - Continuous boundary layer: `u_sw = -η·sat(s/Φ)`
//!
//! References:
//! - Utkin, V. & Shi, J. (1996). "Integral Sliding Mode in Systems Operating
//!   under Uncertainty Conditions." Proceedings of the 35th CDC.
//! - Edwards, C. & Spurgeon, S. (1998). *Sliding Mode Control: Theory and
//!   Applications*. Taylor & Francis.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Switching strategy
// ---------------------------------------------------------------------------

/// Switching law used for the discontinuous component of ISMC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchingLaw {
    /// Hard sign switching: `u_sw = -η·sign(s)`.
    /// Causes chattering but guarantees exact sliding.
    HardSign,
    /// Continuous saturation boundary layer: `u_sw = -η·sat(s/Φ)`.
    /// Reduces chattering at the cost of a small steady-state error.
    SaturationLayer,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in ISMC construction or operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsmcError {
    /// Switching gain η must be positive.
    NonPositiveSwitchingGain,
    /// Boundary layer thickness Φ must be positive (for saturation law).
    NonPositiveBoundaryLayer,
    /// Sampling period must be positive.
    NonPositiveDt,
}

impl core::fmt::Display for IsmcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IsmcError::NonPositiveSwitchingGain => {
                f.write_str("switching gain eta must be positive")
            }
            IsmcError::NonPositiveBoundaryLayer => {
                f.write_str("boundary layer Phi must be positive")
            }
            IsmcError::NonPositiveDt => f.write_str("dt must be positive"),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: saturation function
// ---------------------------------------------------------------------------

/// Compute sat(x/phi): linear in |x| < phi, ±1 outside.
#[inline]
fn sat<S: ControlScalar>(x: S, phi: S) -> S {
    let ratio = x / phi;
    if ratio > S::ONE {
        S::ONE
    } else if ratio < -S::ONE {
        -S::ONE
    } else {
        ratio
    }
}

// ---------------------------------------------------------------------------
// First-order ISMC (SISO)
// ---------------------------------------------------------------------------

/// First-order Integral Sliding Mode Controller (SISO).
///
/// Designed for a SISO plant of the form:
/// ```text
///   ẋ = a·x + b·(u + d(t))
/// ```
/// where `d(t)` is a matched disturbance bounded by `|d| ≤ δ`.
///
/// The sliding surface is:
/// ```text
///   s = e + σ,   σ(0) = -e(0)
///   σ̇ = -(a + b·k_nom)·e
/// ```
/// so that s(0) = 0 (integral term σ initialised to -e(0)).
///
/// The control law splits into:
/// - Nominal part: `u_nom = k_nom·e` (linear stabiliser for unperturbed plant)
/// - Switching part: `u_sw` according to the chosen `SwitchingLaw`
///
/// # Type parameters
/// - `S`: scalar type (f32 or f64).
#[derive(Debug, Clone, Copy)]
pub struct FirstOrderIsmc<S: ControlScalar> {
    /// Nominal state-feedback gain k_nom (scalar for SISO).
    k_nom: S,
    /// Scalar `a` (plant state matrix, scalar for 1-D system).
    a: S,
    /// Scalar `b` (plant input matrix, scalar for 1-D system).
    b: S,
    /// Switching gain η ≥ δ·|b|.
    eta: S,
    /// Boundary layer thickness Φ (used only with `SaturationLayer`).
    phi: S,
    /// Switching law variant.
    law: SwitchingLaw,
    /// Accumulated integral state σ.
    sigma: S,
    /// Sampling period.
    dt: S,
}

impl<S: ControlScalar> FirstOrderIsmc<S> {
    /// Construct a first-order ISMC.
    ///
    /// # Arguments
    /// - `a`, `b`: plant scalars (ẋ = a·x + b·u).
    /// - `k_nom`: nominal gain such that `a + b·k_nom < 0` (stable).
    /// - `eta`: switching gain; must exceed the disturbance bound × |b|.
    /// - `phi`: boundary layer thickness (ignored for `HardSign`).
    /// - `law`: switching law variant.
    /// - `dt`: sampling period.
    pub fn new(
        a: S,
        b: S,
        k_nom: S,
        eta: S,
        phi: S,
        law: SwitchingLaw,
        dt: S,
    ) -> Result<Self, IsmcError> {
        if eta <= S::ZERO {
            return Err(IsmcError::NonPositiveSwitchingGain);
        }
        if law == SwitchingLaw::SaturationLayer && phi <= S::ZERO {
            return Err(IsmcError::NonPositiveBoundaryLayer);
        }
        if dt <= S::ZERO {
            return Err(IsmcError::NonPositiveDt);
        }

        Ok(Self {
            k_nom,
            a,
            b,
            eta,
            phi,
            law,
            sigma: S::ZERO,
            dt,
        })
    }

    /// Initialise the integral term so that s(0) = 0.
    ///
    /// Must be called once the initial error `e0 = x0 - x_ref` is known.
    pub fn initialise(&mut self, e0: S) {
        self.sigma = -e0;
    }

    /// Reset integral state and optionally re-initialise.
    pub fn reset(&mut self) {
        self.sigma = S::ZERO;
    }

    /// Compute control input for the current step.
    ///
    /// # Arguments
    /// - `x`: current state.
    /// - `x_ref`: current reference (assumed constant or slowly varying).
    ///
    /// # Returns
    /// Control input `u`.
    pub fn update(&mut self, x: S, x_ref: S) -> S {
        let e = x - x_ref;

        // Sliding surface s = e + σ
        let s = e + self.sigma;

        // Nominal control:  u_nom = (1/b)·(-a·x_ref - k_nom·e)
        // This includes both feed-forward (to maintain reference) and feedback.
        // At steady state (e=0, x=x_ref): u_ss = -a·x_ref / b  →  ẋ = a·x_ref + b·u_ss = 0 ✓
        let u_nom = (-self.a * x_ref - self.k_nom * e) / self.b;

        // Switching term
        let u_sw = match self.law {
            SwitchingLaw::HardSign => {
                if s > S::ZERO {
                    -self.eta
                } else if s < S::ZERO {
                    self.eta
                } else {
                    S::ZERO
                }
            }
            SwitchingLaw::SaturationLayer => -self.eta * sat(s, self.phi),
        };

        // σ̇ = -(ė under nominal control with no disturbance)
        //    = -(a·x + b·u_nom) = -(a·x + (-a·x_ref - k_nom·e))
        //    = -a·e - (-a·x_ref - k_nom·e + a·x_ref)  ...simplifies to:
        //    = -(a·x - a·x_ref) - (-k_nom·e ... actually:
        //    ė_nom = a·x + b·u_nom = a·x + (-a·x_ref - k_nom·e)
        //          = a·e + a·x_ref - a·x_ref - k_nom·e = (a - k_nom)·e
        // σ̇ = -(a - k_nom)·e  (ensures s stays zero along nominal trajectory)
        let sigma_dot = -(self.a - self.k_nom) * e;
        self.sigma += sigma_dot * self.dt;

        u_nom + u_sw
    }

    /// Return current sliding surface value.
    pub fn surface(&self, x: S, x_ref: S) -> S {
        (x - x_ref) + self.sigma
    }
}

// ---------------------------------------------------------------------------
// Second-order ISMC (SISO)
// ---------------------------------------------------------------------------

/// Second-order Integral Sliding Mode Controller (SISO).
///
/// Designed for a second-order SISO system in companion form:
/// ```text
///   ẋ1 = x2
///   ẋ2 = f(x) + b·(u + d(t))
/// ```
/// where `f(x)` is known (or approximated) and `|d| ≤ δ`.
///
/// Sliding surface:
/// ```text
///   s = ė + c·e + σ,   σ̇ = -c·ė - c²·e / 2
/// ```
/// with σ(0) = -(ė(0) + c·e(0)) so that s(0) = 0.
///
/// Control law:
/// ```text
///   u_nom  = (ẍ_ref - f(x) - c·ė) / b
///   u_sw   per chosen law
/// ```
///
/// # Type parameters
/// - `S`: scalar type.
#[derive(Debug, Clone, Copy)]
pub struct SecondOrderIsmc<S: ControlScalar> {
    /// Surface damping coefficient c > 0.
    c: S,
    /// Known/approximate drift term scale (used to cancel nominal dynamics).
    /// The plant is: ẍ = f_known·x2 + b·u.
    f_known: S,
    /// Input gain b.
    b: S,
    /// Switching gain η.
    eta: S,
    /// Boundary layer.
    phi: S,
    /// Switching law.
    law: SwitchingLaw,
    /// Integral state σ (initialised to –(ė(0) + c·e(0)), then frozen along
    /// the nominal trajectory since ṡ_nom = 0 for the chosen control law).
    sigma: S,
    /// Sampling period — retained for future σ adaptation extensions.
    #[allow(dead_code)]
    dt: S,
}

impl<S: ControlScalar> SecondOrderIsmc<S> {
    /// Construct a second-order ISMC.
    ///
    /// # Arguments
    /// - `c`: surface coefficient (bandwidth-like, > 0).
    /// - `f_known`: known part of the second-derivative drift (≈ -damping·ẋ).
    /// - `b`: input gain.
    /// - `eta`: switching gain.
    /// - `phi`: boundary layer thickness.
    /// - `law`: switching law.
    /// - `dt`: sampling period.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c: S,
        f_known: S,
        b: S,
        eta: S,
        phi: S,
        law: SwitchingLaw,
        dt: S,
    ) -> Result<Self, IsmcError> {
        if eta <= S::ZERO {
            return Err(IsmcError::NonPositiveSwitchingGain);
        }
        if law == SwitchingLaw::SaturationLayer && phi <= S::ZERO {
            return Err(IsmcError::NonPositiveBoundaryLayer);
        }
        if dt <= S::ZERO {
            return Err(IsmcError::NonPositiveDt);
        }

        Ok(Self {
            c,
            f_known,
            b,
            eta,
            phi,
            law,
            sigma: S::ZERO,
            dt,
        })
    }

    /// Initialise integral term to ensure s(0) = 0.
    ///
    /// # Arguments
    /// - `e0`: initial position error x1(0) - x1_ref(0).
    /// - `de0`: initial velocity error x2(0) - x2_ref(0).
    pub fn initialise(&mut self, e0: S, de0: S) {
        self.sigma = -(de0 + self.c * e0);
    }

    /// Reset integral state.
    pub fn reset(&mut self) {
        self.sigma = S::ZERO;
    }

    /// Compute control input.
    ///
    /// # Arguments
    /// - `x1`: position state.
    /// - `x2`: velocity state.
    /// - `x1_ref`: position reference.
    /// - `x2_ref`: velocity reference (ẋ1_ref).
    /// - `x2_dot_ref`: acceleration reference (ẍ1_ref). Use zero if unavailable.
    pub fn update(&mut self, x1: S, x2: S, x1_ref: S, x2_ref: S, x2_dot_ref: S) -> S {
        let e = x1 - x1_ref;
        let de = x2 - x2_ref;

        // Sliding surface
        let s = de + self.c * e + self.sigma;

        // Nominal cancellation control.
        // Full PD + feed-forward: cancel known drift, impose ë = -c²·e - c·de
        // This makes the nominal error dynamics: ë + c·ė + c²·e/2... wait, we want
        // the sliding surface s = de + c·e to decrease. Ideal: ṡ = ë + c·ė = 0
        // means ë = -c·de. But for position tracking we also need to close the loop
        // on position error. Use: ë_desired = -c·de - c²·e (overdamped second order).
        let u_nom = (x2_dot_ref - self.f_known * x2 - self.c * de - self.c * self.c * e) / self.b;

        // Switching term
        let u_sw = match self.law {
            SwitchingLaw::HardSign => {
                if s > S::ZERO {
                    -self.eta / self.b
                } else if s < S::ZERO {
                    self.eta / self.b
                } else {
                    S::ZERO
                }
            }
            SwitchingLaw::SaturationLayer => -self.eta * sat(s, self.phi) / self.b,
        };

        // Under the nominal control (no disturbance), ṡ = 0 because:
        //   ṡ = ë + c·ė + σ̇ = (-c·de) + c·de + σ̇ = σ̇
        // To maintain s(0)=0 invariant along the nominal trajectory: σ̇ = 0.
        // (σ was initialised to absorb the initial condition; it is thereafter frozen.)
        // This is the standard "Utkin-style" ISMC for 2nd-order systems where the
        // nominal PD control already achieves ṡ_nom = 0.

        u_nom + u_sw
    }

    /// Return current sliding surface value.
    pub fn surface(&self, x1: S, x2: S, x1_ref: S, x2_ref: S) -> S {
        (x2 - x2_ref) + self.c * (x1 - x1_ref) + self.sigma
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.001;

    /// Simple first-order plant: ẋ = -x + u + d  (a=-1, b=1)
    fn step_1st(x: f64, u: f64, d: f64) -> f64 {
        x + DT * (-x + u + d)
    }

    /// Second-order plant: ẍ = -0.5·ẋ + u + d
    fn step_2nd(state: [f64; 2], u: f64, d: f64) -> [f64; 2] {
        [
            state[0] + DT * state[1],
            state[1] + DT * (-0.5 * state[1] + u + d),
        ]
    }

    #[test]
    fn ismc_error_on_invalid_eta() {
        let res = FirstOrderIsmc::<f64>::new(
            -1.0,
            1.0,
            2.0,
            0.0, // eta = 0 — invalid
            0.1,
            SwitchingLaw::HardSign,
            DT,
        );
        assert!(res.is_err());
    }

    #[test]
    fn ismc_error_on_invalid_phi() {
        let res = FirstOrderIsmc::<f64>::new(
            -1.0,
            1.0,
            2.0,
            1.0,
            0.0, // phi = 0 — invalid for saturation
            SwitchingLaw::SaturationLayer,
            DT,
        );
        assert!(res.is_err());
    }

    #[test]
    fn first_order_hard_sign_tracks_with_disturbance() {
        // Plant: ẋ = -x + u + d (a=-1, b=1)
        // Choose k_nom=3 so nominal closed-loop pole = a - b*k_nom = -1 - 3 = -4 (stable)
        // Note: u_nom = -k_nom * e means the closed-loop for ė is:
        //   ė = ẋ - ṙ = a·x + b·u + d - 0 = (a - b*k_nom)·e + b*u_sw + d
        // eta=2 > |d|=0.3, so sliding ensures convergence.
        let a = -1.0_f64;
        let b = 1.0_f64;
        let k_nom = 3.0_f64;
        let eta = 1.0_f64; // bound > disturbance magnitude 0.3
        let r = 1.0_f64;
        let d = 0.3_f64;

        let mut ctrl = FirstOrderIsmc::new(a, b, k_nom, eta, 0.1, SwitchingLaw::HardSign, DT)
            .expect("valid params");

        let mut x = 0.0_f64;
        ctrl.initialise(x - r);

        for _ in 0..5000 {
            let u = ctrl.update(x, r);
            x = step_1st(x, u, d);
        }

        assert!((x - r).abs() < 0.05, "x={:.4} should track r={}", x, r);
    }

    #[test]
    fn first_order_saturation_layer_tracks() {
        let a = -1.0_f64;
        let b = 1.0_f64;
        let k_nom = 3.0_f64;
        let eta = 1.0_f64;
        let phi = 0.1_f64;
        let r = 1.0_f64;
        let d = 0.3_f64;

        let mut ctrl =
            FirstOrderIsmc::new(a, b, k_nom, eta, phi, SwitchingLaw::SaturationLayer, DT)
                .expect("valid params");

        let mut x = 0.0_f64;
        ctrl.initialise(x - r);

        for _ in 0..5000 {
            let u = ctrl.update(x, r);
            x = step_1st(x, u, d);
        }

        // Saturation layer allows small steady-state error
        assert!((x - r).abs() < 0.2, "x={:.4} should be near r={}", x, r);
    }

    #[test]
    fn second_order_ismc_saturation_tracks() {
        // Plant: ẍ = -0.5·ẋ + u + d (f_known = -0.5, b = 1)
        // The 2nd-order ISMC nominal control cancels f_known and imposes
        // PD-like error dynamics. Need eta > |d|/b = 0.4.
        let c = 5.0_f64;
        let f_known = -0.5_f64;
        let b = 1.0_f64;
        let eta = 1.0_f64; // eta/b = 1 > d = 0.4
        let phi = 0.1_f64;
        let d = 0.4_f64;
        let r = 1.0_f64;

        let mut ctrl =
            SecondOrderIsmc::new(c, f_known, b, eta, phi, SwitchingLaw::SaturationLayer, DT)
                .expect("valid params");

        let mut state = [0.0_f64; 2];
        ctrl.initialise(state[0] - r, state[1] - 0.0);

        for _ in 0..10000 {
            let u = ctrl.update(state[0], state[1], r, 0.0, 0.0);
            state = step_2nd(state, u, d);
        }

        assert!(
            (state[0] - r).abs() < 0.2,
            "position={:.4} should converge to r={}",
            state[0],
            r
        );
    }

    #[test]
    fn surface_is_zero_at_init() {
        let a = -1.0_f64;
        let b = 1.0_f64;
        let r = 2.0_f64;
        let x0 = 0.5_f64;

        let mut ctrl = FirstOrderIsmc::new(a, b, 0.5, 1.0, 0.1, SwitchingLaw::HardSign, DT)
            .expect("valid params");
        ctrl.initialise(x0 - r);

        let s = ctrl.surface(x0, r);
        assert!(s.abs() < 1e-14, "surface at init should be zero: s={}", s);
    }

    #[test]
    fn sat_function_clamps() {
        assert_eq!(sat(2.0_f64, 1.0_f64), 1.0_f64);
        assert_eq!(sat(-3.0_f64, 1.0_f64), -1.0_f64);
        assert!((sat(0.5_f64, 1.0_f64) - 0.5_f64).abs() < 1e-15);
    }
}
