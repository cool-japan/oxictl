//! Active Disturbance Rejection Control (ADRC).
//!
//! ADRC estimates the total disturbance (including unmodeled dynamics and external
//! disturbances) through an Extended State Observer (ESO), then cancels it via
//! feed-forward. The remaining system behaves as a chain of integrators that is
//! easily controlled by a PD/proportional law.
//!
//! References:
//! - Han, J. (2009). "From PID to Active Disturbance Rejection Control."
//!   IEEE Transactions on Industrial Electronics, 56(3), 900-906.
//! - Gao, Z. (2006). "Scaling and Bandwidth-Parameterization Based Controller
//!   Tuning." Proceedings of the American Control Conference.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when constructing or updating ADRC controllers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdrcError {
    /// Observer bandwidth must be strictly positive.
    NonPositiveObserverBandwidth,
    /// Controller bandwidth must be strictly positive.
    NonPositiveControllerBandwidth,
    /// The control gain parameter α must be strictly positive.
    NonPositiveAlpha,
    /// Sampling period `dt` must be strictly positive.
    NonPositiveDt,
}

impl core::fmt::Display for AdrcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AdrcError::NonPositiveObserverBandwidth => {
                f.write_str("observer bandwidth must be positive")
            }
            AdrcError::NonPositiveControllerBandwidth => {
                f.write_str("controller bandwidth must be positive")
            }
            AdrcError::NonPositiveAlpha => f.write_str("alpha must be positive"),
            AdrcError::NonPositiveDt => f.write_str("dt must be positive"),
        }
    }
}

// ---------------------------------------------------------------------------
// Second-order ADRC
// ---------------------------------------------------------------------------

/// Second-order Active Disturbance Rejection Controller.
///
/// Plant model (assumed): ÿ = f(y, ẏ, d, t) + b·u
/// where `f` is the total disturbance (unknown) and `b` is the approximate
/// high-frequency gain.
///
/// The ESO augments the state with `z3 = f`, yielding:
/// ```text
///   ż1 = z2
///   ż2 = z3 + b·u
///   ż3 = ḟ   (treated as zero — slowly varying)
/// ```
/// with observer gains tuned by bandwidth ω_o: β1 = 3ω_o, β2 = 3ω_o², β3 = ω_o³.
///
/// The control law is:
/// ```text
///   u0 = kp·(r - z1) + kd·(ṙ - z2)   with kp = ω_c², kd = 2ω_c
///   u  = (u0 - z3) / b
/// ```
///
/// # Type parameters
/// - `S`: numeric scalar type (f32 or f64).
#[derive(Debug, Clone, Copy)]
pub struct SecondOrderAdrc<S: ControlScalar> {
    /// Observer state: estimated output z1.
    pub z1: S,
    /// Observer state: estimated output derivative z2.
    pub z2: S,
    /// Observer state: estimated total disturbance z3.
    pub z3: S,

    /// Observer bandwidth ω_o (rad/s).
    omega_o: S,
    /// Controller bandwidth ω_c (rad/s).
    omega_c: S,
    /// Approximate high-frequency gain b.
    b: S,

    // Pre-computed observer gains
    beta1: S,
    beta2: S,
    beta3: S,
    // Pre-computed controller gains
    kp: S,
    kd: S,

    /// Sampling period (s).
    dt: S,
}

impl<S: ControlScalar> SecondOrderAdrc<S> {
    /// Construct a second-order ADRC.
    ///
    /// # Arguments
    /// - `omega_o`: ESO bandwidth (rad/s). Typically 3–10× `omega_c`.
    /// - `omega_c`: closed-loop controller bandwidth (rad/s).
    /// - `b`: approximate high-frequency gain of the plant.
    /// - `dt`: discrete sampling period (s).
    pub fn new(omega_o: S, omega_c: S, b: S, dt: S) -> Result<Self, AdrcError> {
        if omega_o <= S::ZERO {
            return Err(AdrcError::NonPositiveObserverBandwidth);
        }
        if omega_c <= S::ZERO {
            return Err(AdrcError::NonPositiveControllerBandwidth);
        }
        if b <= S::ZERO {
            return Err(AdrcError::NonPositiveAlpha);
        }
        if dt <= S::ZERO {
            return Err(AdrcError::NonPositiveDt);
        }

        let three = S::from_f64(3.0);
        let beta1 = three * omega_o;
        let beta2 = three * omega_o * omega_o;
        let beta3 = omega_o * omega_o * omega_o;

        let kp = omega_c * omega_c;
        let kd = S::TWO * omega_c;

        Ok(Self {
            z1: S::ZERO,
            z2: S::ZERO,
            z3: S::ZERO,
            omega_o,
            omega_c,
            b,
            beta1,
            beta2,
            beta3,
            kp,
            kd,
            dt,
        })
    }

    /// Reset observer states to zero.
    pub fn reset(&mut self) {
        self.z1 = S::ZERO;
        self.z2 = S::ZERO;
        self.z3 = S::ZERO;
    }

    /// Reset observer states to given initial conditions.
    pub fn reset_to(&mut self, y0: S, dy0: S) {
        self.z1 = y0;
        self.z2 = dy0;
        self.z3 = S::ZERO;
    }

    /// Run one update step.
    ///
    /// # Arguments
    /// - `y`: current plant output (measured).
    /// - `r`: reference (set-point).
    /// - `dr`: reference derivative (set to `S::ZERO` if not available).
    /// - `u_prev`: control input applied at the *previous* step.
    ///
    /// # Returns
    /// The control input `u` to apply at the current step.
    pub fn update(&mut self, y: S, r: S, dr: S, u_prev: S) -> S {
        // ESO prediction error
        let e_obs = self.z1 - y;

        // Euler integration of ESO
        let dz1 = self.z2 - self.beta1 * e_obs;
        let dz2 = self.z3 + self.b * u_prev - self.beta2 * e_obs;
        let dz3 = -self.beta3 * e_obs;

        self.z1 += dz1 * self.dt;
        self.z2 += dz2 * self.dt;
        self.z3 += dz3 * self.dt;

        // PD control on virtual (disturbance-free) double integrator
        let u0 = self.kp * (r - self.z1) + self.kd * (dr - self.z2);

        // Cancel estimated disturbance
        (u0 - self.z3) / self.b
    }

    /// Observer bandwidth.
    pub fn omega_o(&self) -> S {
        self.omega_o
    }

    /// Controller bandwidth.
    pub fn omega_c(&self) -> S {
        self.omega_c
    }

    /// Estimated total disturbance (z3).
    pub fn disturbance_estimate(&self) -> S {
        self.z3
    }
}

// ---------------------------------------------------------------------------
// First-order ADRC
// ---------------------------------------------------------------------------

/// First-order Active Disturbance Rejection Controller.
///
/// Plant model (assumed): ẏ = f(y, d, t) + b·u
///
/// ESO (2 states):
/// ```text
///   ż1 = z2 + b·u   (estimated output)
///   ż2 = ḟ          (estimated total disturbance)
/// ```
/// Observer gains: β1 = 2ω_o, β2 = ω_o².
///
/// Control law (P + disturbance cancellation):
/// ```text
///   u0 = ω_c · (r - z1)
///   u  = (u0 - z2) / b
/// ```
///
/// # Type parameters
/// - `S`: numeric scalar type (f32 or f64).
#[derive(Debug, Clone, Copy)]
pub struct FirstOrderAdrc<S: ControlScalar> {
    /// Observer state: estimated output z1.
    pub z1: S,
    /// Observer state: estimated total disturbance z2.
    pub z2: S,

    /// Observer bandwidth ω_o (rad/s).
    omega_o: S,
    /// Controller bandwidth ω_c (rad/s).
    omega_c: S,
    /// Approximate high-frequency gain b.
    b: S,

    // Pre-computed gains
    beta1: S,
    beta2: S,

    /// Sampling period (s).
    dt: S,
}

impl<S: ControlScalar> FirstOrderAdrc<S> {
    /// Construct a first-order ADRC.
    ///
    /// # Arguments
    /// - `omega_o`: ESO bandwidth (rad/s).
    /// - `omega_c`: closed-loop bandwidth (rad/s).
    /// - `b`: approximate high-frequency gain of the plant.
    /// - `dt`: discrete sampling period (s).
    pub fn new(omega_o: S, omega_c: S, b: S, dt: S) -> Result<Self, AdrcError> {
        if omega_o <= S::ZERO {
            return Err(AdrcError::NonPositiveObserverBandwidth);
        }
        if omega_c <= S::ZERO {
            return Err(AdrcError::NonPositiveControllerBandwidth);
        }
        if b <= S::ZERO {
            return Err(AdrcError::NonPositiveAlpha);
        }
        if dt <= S::ZERO {
            return Err(AdrcError::NonPositiveDt);
        }

        let two = S::TWO;
        let beta1 = two * omega_o;
        let beta2 = omega_o * omega_o;

        Ok(Self {
            z1: S::ZERO,
            z2: S::ZERO,
            omega_o,
            omega_c,
            b,
            beta1,
            beta2,
            dt,
        })
    }

    /// Reset observer states to zero.
    pub fn reset(&mut self) {
        self.z1 = S::ZERO;
        self.z2 = S::ZERO;
    }

    /// Reset observer to a known initial output.
    pub fn reset_to(&mut self, y0: S) {
        self.z1 = y0;
        self.z2 = S::ZERO;
    }

    /// Run one update step.
    ///
    /// # Arguments
    /// - `y`: measured output.
    /// - `r`: reference.
    /// - `u_prev`: control input applied at the previous step.
    ///
    /// # Returns
    /// Control input for the current step.
    pub fn update(&mut self, y: S, r: S, u_prev: S) -> S {
        let e_obs = self.z1 - y;

        let dz1 = self.z2 + self.b * u_prev - self.beta1 * e_obs;
        let dz2 = -self.beta2 * e_obs;

        self.z1 += dz1 * self.dt;
        self.z2 += dz2 * self.dt;

        let u0 = self.omega_c * (r - self.z1);
        (u0 - self.z2) / self.b
    }

    /// Observer bandwidth.
    pub fn omega_o(&self) -> S {
        self.omega_o
    }

    /// Controller bandwidth.
    pub fn omega_c(&self) -> S {
        self.omega_c
    }

    /// Estimated total disturbance.
    pub fn disturbance_estimate(&self) -> S {
        self.z2
    }
}

// ---------------------------------------------------------------------------
// ESO standalone
// ---------------------------------------------------------------------------

/// Standalone second-order Extended State Observer.
///
/// Useful when you want to separate observation from control, or use the
/// disturbance estimate in a different control scheme.
#[derive(Debug, Clone, Copy)]
pub struct ExtendedStateObserver<S: ControlScalar> {
    /// Estimated output.
    pub z1: S,
    /// Estimated output derivative.
    pub z2: S,
    /// Estimated total disturbance.
    pub z3: S,

    beta1: S,
    beta2: S,
    beta3: S,
    b: S,
    dt: S,
}

impl<S: ControlScalar> ExtendedStateObserver<S> {
    /// Create a second-order ESO parameterised by observer bandwidth ω_o.
    pub fn new(omega_o: S, b: S, dt: S) -> Result<Self, AdrcError> {
        if omega_o <= S::ZERO {
            return Err(AdrcError::NonPositiveObserverBandwidth);
        }
        if b <= S::ZERO {
            return Err(AdrcError::NonPositiveAlpha);
        }
        if dt <= S::ZERO {
            return Err(AdrcError::NonPositiveDt);
        }

        let three = S::from_f64(3.0);
        Ok(Self {
            z1: S::ZERO,
            z2: S::ZERO,
            z3: S::ZERO,
            beta1: three * omega_o,
            beta2: three * omega_o * omega_o,
            beta3: omega_o * omega_o * omega_o,
            b,
            dt,
        })
    }

    /// Update ESO with current measurement and applied input.
    pub fn update(&mut self, y: S, u: S) {
        let e = self.z1 - y;
        let dz1 = self.z2 - self.beta1 * e;
        let dz2 = self.z3 + self.b * u - self.beta2 * e;
        let dz3 = -self.beta3 * e;
        self.z1 += dz1 * self.dt;
        self.z2 += dz2 * self.dt;
        self.z3 += dz3 * self.dt;
    }

    /// Returns (z1, z2, z3).
    pub fn states(&self) -> (S, S, S) {
        (self.z1, self.z2, self.z3)
    }

    /// Reset all states to zero.
    pub fn reset(&mut self) {
        self.z1 = S::ZERO;
        self.z2 = S::ZERO;
        self.z3 = S::ZERO;
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.001;

    /// Simple first-order plant: ẏ = -y + u + disturbance
    /// Discretised (Euler): y[k+1] = y[k] + dt*(-y[k] + u[k] + d)
    fn step_first_order(y: f64, u: f64, d: f64) -> f64 {
        y + DT * (-y + u + d)
    }

    /// Second-order plant: ÿ = -0.5ẏ + u + d
    /// State: [y, ẏ]
    fn step_second_order(state: [f64; 2], u: f64, d: f64) -> [f64; 2] {
        let dy = state[1];
        let ddy = -0.5 * state[1] + u + d;
        [state[0] + DT * dy, state[1] + DT * ddy]
    }

    #[test]
    fn first_order_adrc_invalid_params() {
        assert!(FirstOrderAdrc::<f64>::new(0.0, 10.0, 1.0, DT).is_err());
        assert!(FirstOrderAdrc::<f64>::new(50.0, 0.0, 1.0, DT).is_err());
        assert!(FirstOrderAdrc::<f64>::new(50.0, 10.0, 0.0, DT).is_err());
        assert!(FirstOrderAdrc::<f64>::new(50.0, 10.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn second_order_adrc_invalid_params() {
        assert!(SecondOrderAdrc::<f64>::new(0.0, 10.0, 1.0, DT).is_err());
        assert!(SecondOrderAdrc::<f64>::new(100.0, 0.0, 1.0, DT).is_err());
        assert!(SecondOrderAdrc::<f64>::new(100.0, 10.0, -1.0, DT).is_err());
        assert!(SecondOrderAdrc::<f64>::new(100.0, 10.0, 1.0, -DT).is_err());
    }

    #[test]
    fn first_order_adrc_tracks_reference_with_disturbance() {
        let mut ctrl = FirstOrderAdrc::<f64>::new(50.0, 10.0, 1.0, DT).expect("valid params");
        let r = 1.0_f64;
        let disturbance = 0.3_f64; // constant disturbance

        let mut y = 0.0_f64;
        let mut u = 0.0_f64;
        for _ in 0..5000 {
            let u_new = ctrl.update(y, r, u);
            y = step_first_order(y, u, disturbance);
            u = u_new;
        }

        // Should track reference within 5% despite disturbance
        assert!(
            (y - r).abs() < 0.05,
            "output={:.4} should be near reference={}",
            y,
            r
        );
    }

    #[test]
    fn second_order_adrc_tracks_reference() {
        let mut ctrl = SecondOrderAdrc::<f64>::new(80.0, 10.0, 1.0, DT).expect("valid params");
        let r = 1.0_f64;
        let disturbance = 0.5_f64;

        let mut state = [0.0_f64; 2];
        let mut u = 0.0_f64;
        for _ in 0..8000 {
            let u_new = ctrl.update(state[0], r, 0.0, u);
            state = step_second_order(state, u, disturbance);
            u = u_new;
        }

        assert!(
            (state[0] - r).abs() < 0.05,
            "output={:.4} should converge to reference={}",
            state[0],
            r
        );
    }

    #[test]
    fn eso_disturbance_estimate_converges() {
        // Use ADRC first-order controller with a disturbance:
        // Plant: ẏ = d (pure integrator with disturbance d=2).
        // The first-order ADRC (which uses a 2-state ESO) should estimate d.
        let omega_o = 80.0_f64;
        let omega_c = 20.0_f64;
        let b = 1.0_f64;
        let d = 2.0_f64; // constant disturbance to estimate

        let mut ctrl = FirstOrderAdrc::<f64>::new(omega_o, omega_c, b, DT).expect("valid params");

        let mut y = 0.0_f64;
        let mut u = 0.0_f64;
        let r = 1.0_f64;

        for _ in 0..6000 {
            let u_new = ctrl.update(y, r, u);
            // Plant: ẏ = b·u + d
            y += DT * (b * u + d);
            u = u_new;
        }

        // After many steps the ADRC should have converged; output near reference
        assert!(
            (y - r).abs() < 0.1,
            "output y={:.4} should be near r={} despite disturbance d={}",
            y,
            r,
            d
        );
        // ESO second state (z2) estimates the lumped disturbance
        let d_est = ctrl.disturbance_estimate();
        assert!(
            d_est.abs() > 0.5,
            "disturbance estimate z2={:.4} should be significantly non-zero",
            d_est
        );
    }

    #[test]
    fn second_order_adrc_reset() {
        let mut ctrl = SecondOrderAdrc::<f64>::new(100.0, 20.0, 1.0, DT).expect("valid params");

        // Run for a while
        for i in 0..100 {
            let _ = ctrl.update(i as f64 * 0.01, 1.0, 0.0, 0.0);
        }

        ctrl.reset();
        assert_eq!(ctrl.z1, 0.0);
        assert_eq!(ctrl.z2, 0.0);
        assert_eq!(ctrl.z3, 0.0);
    }

    #[test]
    fn first_order_adrc_f32() {
        let mut ctrl =
            FirstOrderAdrc::<f32>::new(50.0, 10.0, 1.0, 0.001).expect("valid f32 params");
        let u = ctrl.update(0.0_f32, 1.0_f32, 0.0_f32);
        // Should produce a non-NaN finite control signal
        assert!(u.is_finite());
    }
}
