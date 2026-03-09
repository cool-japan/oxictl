//! Ultra-local model-based (model-free) control — Fliess & Join approach.
//!
//! The plant is approximated by the ultra-local model:
//! ```text
//!   ẏ = F + α·u
//! ```
//! where:
//! - `F` is the *ultra-local* lumped term (everything except the direct effect of `u`),
//!   treated as slowly varying and estimated online.
//! - `α` is a user-selected constant (≠ 0) that roughly captures the gain magnitude.
//!
//! **Algebraic estimation of F̂**
//!
//! Over a short sliding window of `N` samples the algebraic estimator computes:
//! ```text
//!   F̂ ≈ (Δy/Δt) - α·ū
//! ```
//! where Δy and ū are computed from the buffered output and input samples using
//! a least-squares-derived algebraic formula (finite-difference of the output
//! minus the direct contribution of the input).
//!
//! **Intelligent PID (iPID)**
//!
//! The control law is:
//! ```text
//!   u = (ẏ_ref - F̂ + Kp·e + Ki·∫e dt + Kd·ė) / α
//! ```
//! which renders the closed-loop dynamics ≈ ÿ_err + Kd·ė + Kp·e + Ki·∫e = 0.
//!
//! References:
//! - Fliess, M. & Join, C. (2013). "Model-free control." *International Journal
//!   of Control*, 86(12), 2228–2252.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that may occur in model-free control construction or operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MfcError {
    /// α must not be zero.
    ZeroAlpha,
    /// Sampling period must be strictly positive.
    NonPositiveDt,
    /// Window size must be at least 2.
    WindowTooSmall,
}

impl core::fmt::Display for MfcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MfcError::ZeroAlpha => f.write_str("alpha must be non-zero"),
            MfcError::NonPositiveDt => f.write_str("dt must be positive"),
            MfcError::WindowTooSmall => f.write_str("window size must be >= 2"),
        }
    }
}

// ---------------------------------------------------------------------------
// Sliding window buffer (heap-free, const-generic)
// ---------------------------------------------------------------------------

/// Fixed-size circular buffer for the sliding estimation window.
///
/// `N` is the maximum number of samples stored.
#[derive(Debug, Clone, Copy)]
pub struct SlidingWindow<S: ControlScalar, const N: usize> {
    data: [S; N],
    head: usize,
    count: usize,
}

impl<S: ControlScalar, const N: usize> SlidingWindow<S, N> {
    /// Create an empty window (all zeros).
    pub fn new() -> Self {
        Self {
            data: [S::ZERO; N],
            head: 0,
            count: 0,
        }
    }

    /// Push a new sample into the window, overwriting the oldest if full.
    pub fn push(&mut self, value: S) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % N;
        if self.count < N {
            self.count += 1;
        }
    }

    /// Number of valid samples currently held.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true when no samples have been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Access sample by age: `age=0` → newest, `age=count-1` → oldest.
    pub fn get(&self, age: usize) -> Option<S> {
        if age >= self.count {
            return None;
        }
        // head points one past the newest element
        let idx = (self.head + N - 1 - age) % N;
        Some(self.data[idx])
    }

    /// Oldest sample in the window (or None if empty).
    pub fn oldest(&self) -> Option<S> {
        self.get(self.count.saturating_sub(1))
    }

    /// Newest sample in the window (or None if empty).
    pub fn newest(&self) -> Option<S> {
        self.get(0)
    }

    /// Compute the mean of all samples currently in the window.
    pub fn mean(&self) -> Option<S> {
        if self.count == 0 {
            return None;
        }
        let mut sum = S::ZERO;
        for i in 0..self.count {
            sum += self.data[i];
        }
        Some(sum / S::from_f64(self.count as f64))
    }

    /// Reset the window to empty.
    pub fn reset(&mut self) {
        self.data = [S::ZERO; N];
        self.head = 0;
        self.count = 0;
    }
}

impl<S: ControlScalar, const N: usize> Default for SlidingWindow<S, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Algebraic F estimator
// ---------------------------------------------------------------------------

/// Online algebraic estimator of the ultra-local disturbance term F̂.
///
/// Uses a sliding window of output samples `y` and input samples `u` to
/// compute:
/// ```text
///   Δy = y_newest - y_oldest
///   ū  = mean(u_window)
///   F̂  = Δy / (W·dt) - α·ū
/// ```
/// where `W` is the window length in samples.
///
/// # Type parameters
/// - `S`: scalar type.
/// - `W`: window length (number of samples ≥ 2).
#[derive(Debug, Clone, Copy)]
pub struct AlgebraicFEstimator<S: ControlScalar, const W: usize> {
    y_buf: SlidingWindow<S, W>,
    u_buf: SlidingWindow<S, W>,
    alpha: S,
    dt: S,
    /// Last computed estimate of F.
    f_hat: S,
}

impl<S: ControlScalar, const W: usize> AlgebraicFEstimator<S, W> {
    /// Create a new estimator.
    ///
    /// # Arguments
    /// - `alpha`: ultra-local model gain (must be ≠ 0).
    /// - `dt`: sampling period (s).
    pub fn new(alpha: S, dt: S) -> Result<Self, MfcError> {
        if alpha == S::ZERO {
            return Err(MfcError::ZeroAlpha);
        }
        if dt <= S::ZERO {
            return Err(MfcError::NonPositiveDt);
        }
        if W < 2 {
            return Err(MfcError::WindowTooSmall);
        }

        Ok(Self {
            y_buf: SlidingWindow::new(),
            u_buf: SlidingWindow::new(),
            alpha,
            dt,
            f_hat: S::ZERO,
        })
    }

    /// Push a new (y, u) observation and update the estimate of F̂.
    ///
    /// The estimate is updated only once the window holds at least 2 samples.
    /// Before that the estimate stays at zero.
    pub fn update(&mut self, y: S, u: S) {
        self.y_buf.push(y);
        self.u_buf.push(u);

        let count = self.y_buf.len();
        if count < 2 {
            return;
        }

        // Δy over the window span
        let y_new = match self.y_buf.newest() {
            Some(v) => v,
            None => return,
        };
        let y_old = match self.y_buf.oldest() {
            Some(v) => v,
            None => return,
        };
        let delta_y = y_new - y_old;
        let t_span = S::from_f64((count - 1) as f64) * self.dt;

        if t_span <= S::ZERO {
            return;
        }

        let dy_dt = delta_y / t_span;

        // Mean input over the window
        let u_mean = self.u_buf.mean().unwrap_or(S::ZERO);

        self.f_hat = dy_dt - self.alpha * u_mean;
    }

    /// Current estimate of F̂.
    pub fn f_hat(&self) -> S {
        self.f_hat
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.y_buf.reset();
        self.u_buf.reset();
        self.f_hat = S::ZERO;
    }
}

// ---------------------------------------------------------------------------
// Intelligent PID (iPID)
// ---------------------------------------------------------------------------

/// Intelligent PID controller based on the ultra-local model.
///
/// Control law:
/// ```text
///   u = (ẏ_ref - F̂ + Kp·e + Ki·∫e + Kd·ė) / α
/// ```
///
/// The term `F̂` is updated from the algebraic estimator every step,
/// making the controller effectively model-free.
///
/// # Type parameters
/// - `S`: scalar type.
/// - `W`: estimator window size (number of samples, ≥ 2).
#[derive(Debug, Clone, Copy)]
pub struct IPid<S: ControlScalar, const W: usize> {
    /// Ultra-local model gain α.
    alpha: S,
    /// Proportional gain.
    kp: S,
    /// Integral gain.
    ki: S,
    /// Derivative gain.
    kd: S,
    /// Sampling period.
    dt: S,
    /// Algebraic F estimator.
    estimator: AlgebraicFEstimator<S, W>,
    /// Accumulated integral of error.
    integral: S,
    /// Previous error (for derivative term).
    prev_error: S,
    /// Whether previous error is valid.
    has_prev: bool,
    /// Anti-windup saturation limit for integral term (None = no limit).
    integral_limit: Option<S>,
}

impl<S: ControlScalar, const W: usize> IPid<S, W> {
    /// Construct an iPID controller.
    ///
    /// # Arguments
    /// - `alpha`: ultra-local gain (non-zero).
    /// - `kp`, `ki`, `kd`: PID gains applied to the error in the inner loop.
    /// - `dt`: sampling period.
    /// - `integral_limit`: optional anti-windup saturation limit (absolute value).
    pub fn new(
        alpha: S,
        kp: S,
        ki: S,
        kd: S,
        dt: S,
        integral_limit: Option<S>,
    ) -> Result<Self, MfcError> {
        let estimator = AlgebraicFEstimator::new(alpha, dt)?;
        Ok(Self {
            alpha,
            kp,
            ki,
            kd,
            dt,
            estimator,
            integral: S::ZERO,
            prev_error: S::ZERO,
            has_prev: false,
            integral_limit,
        })
    }

    /// Reset controller state (integral, derivative memory, estimator).
    pub fn reset(&mut self) {
        self.integral = S::ZERO;
        self.prev_error = S::ZERO;
        self.has_prev = false;
        self.estimator.reset();
    }

    /// Run one control step.
    ///
    /// # Arguments
    /// - `y`: measured output.
    /// - `r`: reference (set-point).
    /// - `dr`: reference derivative ẏ_ref (set to `S::ZERO` if unavailable).
    /// - `u_prev`: control applied at the previous step (fed to estimator).
    ///
    /// # Returns
    /// Control input for the current step.
    pub fn update(&mut self, y: S, r: S, dr: S, u_prev: S) -> S {
        // Update disturbance estimate
        self.estimator.update(y, u_prev);
        let f_hat = self.estimator.f_hat();

        let error = r - y;

        // Integral with optional anti-windup
        let raw_integral = self.integral + error * self.dt;
        self.integral = match self.integral_limit {
            Some(lim) => raw_integral.saturate(lim),
            None => raw_integral,
        };

        // Derivative (backward difference)
        let derivative = if self.has_prev {
            (error - self.prev_error) / self.dt
        } else {
            S::ZERO
        };
        self.prev_error = error;
        self.has_prev = true;

        let pid_term = self.kp * error + self.ki * self.integral + self.kd * derivative;

        (dr - f_hat + pid_term) / self.alpha
    }

    /// Return the current F̂ estimate.
    pub fn f_hat(&self) -> S {
        self.estimator.f_hat()
    }

    /// Return accumulated integral state.
    pub fn integral_state(&self) -> S {
        self.integral
    }
}

// ---------------------------------------------------------------------------
// Simple iP (Intelligent Proportional) — no I or D terms
// ---------------------------------------------------------------------------

/// Intelligent proportional controller (iP).
///
/// Simplest model-free law:
/// ```text
///   u = (ẏ_ref - F̂ + Kp·e) / α
/// ```
///
/// # Type parameters
/// - `S`: scalar type.
/// - `W`: estimator window size.
#[derive(Debug, Clone, Copy)]
pub struct IP<S: ControlScalar, const W: usize> {
    alpha: S,
    kp: S,
    estimator: AlgebraicFEstimator<S, W>,
}

impl<S: ControlScalar, const W: usize> IP<S, W> {
    /// Construct an iP controller.
    pub fn new(alpha: S, kp: S, dt: S) -> Result<Self, MfcError> {
        let estimator = AlgebraicFEstimator::new(alpha, dt)?;
        Ok(Self {
            alpha,
            kp,
            estimator,
        })
    }

    /// Run one step.
    pub fn update(&mut self, y: S, r: S, dr: S, u_prev: S) -> S {
        self.estimator.update(y, u_prev);
        let f_hat = self.estimator.f_hat();
        let error = r - y;
        (dr - f_hat + self.kp * error) / self.alpha
    }

    /// Current F̂ estimate.
    pub fn f_hat(&self) -> S {
        self.estimator.f_hat()
    }

    /// Reset estimator.
    pub fn reset(&mut self) {
        self.estimator.reset();
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.005;

    /// First-order plant: ẏ = -2·y + 3·u + d
    fn step_plant(y: f64, u: f64, d: f64) -> f64 {
        y + DT * (-2.0 * y + 3.0 * u + d)
    }

    #[test]
    fn mfc_error_zero_alpha() {
        let res = AlgebraicFEstimator::<f64, 10>::new(0.0, DT);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), MfcError::ZeroAlpha);
    }

    #[test]
    fn mfc_error_bad_dt() {
        let res = AlgebraicFEstimator::<f64, 10>::new(3.0, -0.001);
        assert!(res.is_err());
    }

    #[test]
    fn sliding_window_push_and_get() {
        let mut w = SlidingWindow::<f64, 4>::new();
        w.push(1.0);
        w.push(2.0);
        w.push(3.0);

        assert_eq!(w.len(), 3);
        assert_eq!(w.newest(), Some(3.0));
        assert_eq!(w.oldest(), Some(1.0));
    }

    #[test]
    fn sliding_window_wraps_correctly() {
        let mut w = SlidingWindow::<f64, 3>::new();
        w.push(10.0);
        w.push(20.0);
        w.push(30.0);
        w.push(40.0); // overwrites 10.0

        assert_eq!(w.len(), 3);
        assert_eq!(w.newest(), Some(40.0));
        assert_eq!(w.oldest(), Some(20.0));
    }

    #[test]
    fn sliding_window_mean() {
        let mut w = SlidingWindow::<f64, 4>::new();
        w.push(1.0);
        w.push(2.0);
        w.push(3.0);
        w.push(4.0);
        let m = w.mean().expect("mean of non-empty window");
        assert!((m - 2.5).abs() < 1e-12, "mean={}", m);
    }

    #[test]
    fn ipid_tracks_reference_with_disturbance() {
        // Plant: ẏ = -2y + 3u + d.  α ≈ 3 (input gain), disturbance d=1
        let mut ctrl = IPid::<f64, 20>::new(
            3.0, // alpha
            5.0, // kp
            1.0, // ki
            0.1, // kd
            DT,
            Some(50.0), // integral limit
        )
        .expect("valid params");

        let r = 2.0_f64;
        let d = 1.0_f64;
        let mut y = 0.0_f64;
        let mut u = 0.0_f64;

        for _ in 0..3000 {
            let u_new = ctrl.update(y, r, 0.0, u);
            y = step_plant(y, u, d);
            u = u_new;
        }

        assert!(
            (y - r).abs() < 0.1,
            "output y={:.4} should track r={}",
            y,
            r
        );
    }

    #[test]
    fn ip_produces_finite_output() {
        let mut ctrl = IP::<f64, 8>::new(3.0, 5.0, DT).expect("valid params");
        let mut y = 0.0_f64;
        let mut u = 0.0_f64;
        for _ in 0..100 {
            let u_new = ctrl.update(y, 1.0, 0.0, u);
            y = step_plant(y, u, 0.0);
            u = u_new;
            assert!(u.is_finite(), "u should be finite: {}", u);
        }
    }

    #[test]
    fn ipid_reset_clears_state() {
        let mut ctrl = IPid::<f64, 10>::new(1.0, 2.0, 0.5, 0.1, DT, None).expect("valid params");
        for _ in 0..50 {
            let _ = ctrl.update(0.5, 1.0, 0.0, 0.0);
        }
        ctrl.reset();
        assert_eq!(ctrl.integral_state(), 0.0);
        assert_eq!(ctrl.f_hat(), 0.0);
    }

    #[test]
    fn ipid_f32_compiles() {
        let mut ctrl =
            IPid::<f32, 6>::new(1.0_f32, 2.0, 0.1, 0.01, 0.005, None).expect("f32 valid");
        let u = ctrl.update(0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32);
        assert!(u.is_finite());
    }
}
