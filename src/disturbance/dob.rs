//! Q-filter Disturbance Observer (DOB) for SISO systems.
//!
//! Models the nominal plant as P_n(s) = b_n / (s + a_n) (first-order).
//! The Q-filter is Q(s) = 1/(τs+1)^n for n=1 or n=2.
//!
//! Disturbance estimate: d_hat = Q(s) * [u - P_n^{-1}(s) * y]
//!
//! References:
//! - Ohnishi, K. (1987). "A new servo method in mechatronics."
//!   Trans. Japanese Society Electrical Engineering, 107-D, 83–86.
//! - Sariyildiz, E. & Ohnishi, K. (2015). "Stability and robustness of
//!   disturbance-observer-based motion control systems." IEEE TIE, 62(9).

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when constructing or updating a [`DisturbanceObserver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DobError {
    /// Q-filter time constant τ must be strictly positive.
    NonPositiveTau,
    /// Sampling period dt must be strictly positive.
    NonPositiveDt,
    /// Nominal plant gain b_n must be non-zero.
    ZeroBn,
    /// Q-filter order must be 1 or 2.
    InvalidOrder,
}

impl core::fmt::Display for DobError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DobError::NonPositiveTau => f.write_str("Q-filter time constant tau must be positive"),
            DobError::NonPositiveDt => f.write_str("sampling period dt must be positive"),
            DobError::ZeroBn => f.write_str("nominal plant gain b_n must be non-zero"),
            DobError::InvalidOrder => f.write_str("Q-filter order must be 1 or 2"),
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a [`DisturbanceObserver`].
///
/// The nominal plant is modelled as a first-order system:
/// ```text
///   P_n(s) = b_n / (s + a_n)
/// ```
/// Its inverse is:
/// ```text
///   P_n^{-1}(s) = (s + a_n) / b_n
/// ```
/// which in the time domain acts on y as:
/// ```text
///   v(t) = (ẏ(t) + a_n · y(t)) / b_n
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DisturbanceObserverConfig<S> {
    /// Nominal plant pole (a_n ≥ 0 for stable plant; a_n > 0 for strictly stable).
    pub a_n: S,
    /// Nominal plant DC gain numerator (must be non-zero).
    pub b_n: S,
    /// Q-filter time constant τ (bandwidth = 1/τ rad/s). Must be > 0.
    pub tau: S,
    /// Q-filter order: 1 (first-order low-pass) or 2 (second-order cascade).
    pub order: u8,
    /// Discrete sampling period in seconds. Must be > 0.
    pub dt: S,
}

// ---------------------------------------------------------------------------
// Observer struct
// ---------------------------------------------------------------------------

/// Q-filter Disturbance Observer for a first-order SISO nominal plant.
///
/// # Algorithm
///
/// At each sample:
/// 1. Compute the (approximate) inverse-plant output:
///    ```text
///    v[k] = (y[k] - y[k-1]) / dt + a_n · y[k]) / b_n
///    ```
/// 2. Compute the DOB filter input: `e[k] = u[k] - v[k]`
/// 3. Pass `e[k]` through the Q-filter (1st- or 2nd-order Euler IIR) to
///    obtain `d_hat[k]`.
///
/// # Type parameters
/// - `S`: numeric scalar type implementing [`ControlScalar`] (f32 or f64).
#[derive(Debug, Clone, Copy)]
pub struct DisturbanceObserver<S: ControlScalar> {
    /// First Q-filter state (also serves as the sole state for order-1).
    q1: S,
    /// Second Q-filter state (only used when order == 2).
    q2: S,
    /// Previous plant output — used for backward-difference derivative.
    y_prev: S,
    /// Nominal plant pole.
    a_n: S,
    /// Nominal plant gain (non-zero).
    b_n: S,
    /// Q-filter time constant.
    tau: S,
    /// Q-filter order (1 or 2).
    order: u8,
    /// Sampling period.
    dt: S,
    /// Most recent disturbance estimate.
    d_hat: S,
    /// Whether the observer has received its first sample yet.
    initialized: bool,
}

impl<S: ControlScalar> DisturbanceObserver<S> {
    /// Create a new [`DisturbanceObserver`] from the given configuration.
    ///
    /// # Errors
    /// Returns [`DobError`] if any configuration parameter is invalid.
    pub fn new(config: DisturbanceObserverConfig<S>) -> Result<Self, DobError> {
        if config.tau <= S::ZERO {
            return Err(DobError::NonPositiveTau);
        }
        if config.dt <= S::ZERO {
            return Err(DobError::NonPositiveDt);
        }
        // b_n must be non-zero (can be negative for non-minimum-phase plants).
        if config.b_n == S::ZERO {
            return Err(DobError::ZeroBn);
        }
        if config.order != 1 && config.order != 2 {
            return Err(DobError::InvalidOrder);
        }

        Ok(Self {
            q1: S::ZERO,
            q2: S::ZERO,
            y_prev: S::ZERO,
            a_n: config.a_n,
            b_n: config.b_n,
            tau: config.tau,
            order: config.order,
            dt: config.dt,
            d_hat: S::ZERO,
            initialized: false,
        })
    }

    /// Run one discrete update step.
    ///
    /// # Arguments
    /// - `u`: plant input applied at the current step.
    /// - `y`: plant output measured at the current step.
    ///
    /// # Returns
    /// `Ok(d_hat)` — the current disturbance estimate.
    pub fn update(&mut self, u: S, y: S) -> Result<S, DobError> {
        // On the first call we have no previous y, so we initialise and
        // return zero (cannot compute a derivative yet).
        if !self.initialized {
            self.y_prev = y;
            self.initialized = true;
            return Ok(self.d_hat);
        }

        // 1. Inverse-plant output: v = (ẏ + a_n·y) / b_n
        //    ẏ ≈ (y[k] - y[k-1]) / dt   (backward Euler finite difference)
        let y_dot = (y - self.y_prev) / self.dt;
        let inv_plant_out = (y_dot + self.a_n * y) / self.b_n;
        self.y_prev = y;

        // 2. DOB filter input
        // Classical DOB: d_hat = Q * (P_n^{-1} * y - u)
        let e = inv_plant_out - u;

        // 3. Q-filter (Euler forward IIR)
        //    First-order:  q1[k] = q1[k-1] + (dt/τ)·(e[k] - q1[k-1])
        //    Second-order: cascade of two first-order sections
        let alpha = self.dt / self.tau;

        self.q1 = self.q1 + alpha * (e - self.q1);

        self.d_hat = if self.order == 2 {
            self.q2 = self.q2 + alpha * (self.q1 - self.q2);
            self.q2
        } else {
            self.q1
        };

        Ok(self.d_hat)
    }

    /// Return the most recently computed disturbance estimate without
    /// advancing the internal state.
    #[inline]
    pub fn disturbance_estimate(&self) -> S {
        self.d_hat
    }

    /// Reset all internal states to zero.
    pub fn reset(&mut self) {
        self.q1 = S::ZERO;
        self.q2 = S::ZERO;
        self.y_prev = S::ZERO;
        self.d_hat = S::ZERO;
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a default valid config with order 1.
    fn cfg_order1() -> DisturbanceObserverConfig<f64> {
        DisturbanceObserverConfig {
            a_n: 1.0,
            b_n: 2.0,
            tau: 0.05,
            order: 1,
            dt: 0.005,
        }
    }

    /// Helper: build a default valid config with order 2.
    fn cfg_order2() -> DisturbanceObserverConfig<f64> {
        DisturbanceObserverConfig {
            a_n: 1.0,
            b_n: 2.0,
            tau: 0.05,
            order: 2,
            dt: 0.005,
        }
    }

    /// Simulate the nominal plant (no disturbance):
    ///   y[k+1] = (1 - a_n·dt)·y[k] + b_n·dt·u
    fn simulate_nominal(a_n: f64, b_n: f64, dt: f64, u: f64, y0: f64, steps: usize) -> Vec<f64> {
        let mut y = y0;
        let mut ys = Vec::with_capacity(steps + 1);
        ys.push(y);
        for _ in 0..steps {
            y = (1.0 - a_n * dt) * y + b_n * dt * u;
            ys.push(y);
        }
        ys
    }

    /// Simulate the plant with an additive disturbance d on the input:
    ///   y[k+1] = (1 - a_n·dt)·y[k] + b_n·dt·(u + d)
    fn simulate_disturbed(
        a_n: f64,
        b_n: f64,
        dt: f64,
        u: f64,
        d: f64,
        y0: f64,
        steps: usize,
    ) -> Vec<f64> {
        let mut y = y0;
        let mut ys = Vec::with_capacity(steps + 1);
        ys.push(y);
        for _ in 0..steps {
            y = (1.0 - a_n * dt) * y + b_n * dt * (u + d);
            ys.push(y);
        }
        ys
    }

    // -----------------------------------------------------------------------
    // 1. Zero disturbance, order 1 — d_hat should converge to ≈ 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_zero_disturbance_order1() {
        let cfg = cfg_order1();
        let a_n = cfg.a_n;
        let b_n = cfg.b_n;
        let dt = cfg.dt;
        let u = 0.5_f64;
        let y0 = 1.0_f64;
        let steps = 800usize;

        let mut dob = DisturbanceObserver::new(cfg).expect("valid config");
        let ys = simulate_nominal(a_n, b_n, dt, u, y0, steps);

        let mut d_hat = 0.0_f64;
        for &y_k in &ys[..steps] {
            d_hat = dob.update(u, y_k).expect("update ok");
        }

        assert!(
            d_hat.abs() < 0.05,
            "Expected d_hat ≈ 0 for zero disturbance, got {d_hat}"
        );
    }

    // -----------------------------------------------------------------------
    // 2. Step disturbance, order 1 — d_hat should converge toward d
    // -----------------------------------------------------------------------
    #[test]
    fn test_step_disturbance_order1() {
        let cfg = cfg_order1();
        let a_n = cfg.a_n;
        let b_n = cfg.b_n;
        let dt = cfg.dt;
        let u = 0.5_f64;
        let d = 1.0_f64;
        let y0 = 0.0_f64;
        let steps = 800usize;

        let mut dob = DisturbanceObserver::new(cfg).expect("valid config");
        let ys = simulate_disturbed(a_n, b_n, dt, u, d, y0, steps);

        let mut d_hat = 0.0_f64;
        for &y_k in &ys[..steps] {
            d_hat = dob.update(u, y_k).expect("update ok");
        }

        let err = (d_hat - d).abs();
        assert!(
            err < 0.25 * d,
            "Expected d_hat ≈ {d}, got {d_hat} (err={err})"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Zero disturbance, order 2
    // -----------------------------------------------------------------------
    #[test]
    fn test_zero_disturbance_order2() {
        let cfg = cfg_order2();
        let a_n = cfg.a_n;
        let b_n = cfg.b_n;
        let dt = cfg.dt;
        let u = 0.3_f64;
        let y0 = 0.5_f64;
        let steps = 1000usize;

        let mut dob = DisturbanceObserver::new(cfg).expect("valid config");
        let ys = simulate_nominal(a_n, b_n, dt, u, y0, steps);

        let mut d_hat = 0.0_f64;
        for &y_k in &ys[..steps] {
            d_hat = dob.update(u, y_k).expect("update ok");
        }

        assert!(
            d_hat.abs() < 0.05,
            "Expected d_hat ≈ 0 for zero disturbance (order 2), got {d_hat}"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Step disturbance, order 2
    // -----------------------------------------------------------------------
    #[test]
    fn test_step_disturbance_order2() {
        let cfg = cfg_order2();
        let a_n = cfg.a_n;
        let b_n = cfg.b_n;
        let dt = cfg.dt;
        let u = 0.0_f64;
        let d = 2.0_f64;
        let y0 = 0.0_f64;
        let steps = 1200usize;

        let mut dob = DisturbanceObserver::new(cfg).expect("valid config");
        let ys = simulate_disturbed(a_n, b_n, dt, u, d, y0, steps);

        let mut d_hat = 0.0_f64;
        for &y_k in &ys[..steps] {
            d_hat = dob.update(u, y_k).expect("update ok");
        }

        let err = (d_hat - d).abs();
        assert!(
            err < 0.5 * d,
            "Expected d_hat ≈ {d} (order 2), got {d_hat} (err={err})"
        );
    }

    // -----------------------------------------------------------------------
    // 5. Bandwidth effect: smaller τ → faster convergence
    // -----------------------------------------------------------------------
    #[test]
    fn test_bandwidth_effect() {
        let a_n = 1.0_f64;
        let b_n = 2.0_f64;
        let dt = 0.005_f64;
        let u = 0.0_f64;
        let d = 1.0_f64;
        let y0 = 0.0_f64;
        let steps = 200usize;

        let ys = simulate_disturbed(a_n, b_n, dt, u, d, y0, steps);

        // Fast filter: τ = 0.01
        let mut dob_fast = DisturbanceObserver::new(DisturbanceObserverConfig {
            a_n,
            b_n,
            tau: 0.01,
            order: 1,
            dt,
        })
        .expect("valid fast config");

        // Slow filter: τ = 0.1
        let mut dob_slow = DisturbanceObserver::new(DisturbanceObserverConfig {
            a_n,
            b_n,
            tau: 0.1,
            order: 1,
            dt,
        })
        .expect("valid slow config");

        let mut d_fast = 0.0_f64;
        let mut d_slow = 0.0_f64;
        for &y_k in &ys[..steps] {
            d_fast = dob_fast.update(u, y_k).expect("fast update ok");
            d_slow = dob_slow.update(u, y_k).expect("slow update ok");
        }

        // Faster filter should be closer to the true disturbance after N steps
        let err_fast = (d_fast - d).abs();
        let err_slow = (d_slow - d).abs();
        assert!(
            err_fast < err_slow,
            "Fast DOB (τ=0.01) err={err_fast} should be < slow DOB (τ=0.1) err={err_slow}"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Invalid configurations
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalid_config_tau() {
        let mut cfg = cfg_order1();
        cfg.tau = -0.1;
        assert_eq!(
            DisturbanceObserver::new(cfg).unwrap_err(),
            DobError::NonPositiveTau
        );

        cfg.tau = 0.0;
        assert_eq!(
            DisturbanceObserver::new(cfg).unwrap_err(),
            DobError::NonPositiveTau
        );
    }

    #[test]
    fn test_invalid_config_dt() {
        let mut cfg = cfg_order1();
        cfg.dt = 0.0;
        assert_eq!(
            DisturbanceObserver::new(cfg).unwrap_err(),
            DobError::NonPositiveDt
        );
    }

    #[test]
    fn test_invalid_config_bn_zero() {
        let mut cfg = cfg_order1();
        cfg.b_n = 0.0;
        assert_eq!(DisturbanceObserver::new(cfg).unwrap_err(), DobError::ZeroBn);
    }

    #[test]
    fn test_invalid_config_order() {
        let mut cfg = cfg_order1();
        cfg.order = 3;
        assert_eq!(
            DisturbanceObserver::new(cfg).unwrap_err(),
            DobError::InvalidOrder
        );

        cfg.order = 0;
        assert_eq!(
            DisturbanceObserver::new(cfg).unwrap_err(),
            DobError::InvalidOrder
        );
    }

    // -----------------------------------------------------------------------
    // 7. disturbance_estimate() matches last update() return value
    // -----------------------------------------------------------------------
    #[test]
    fn test_estimate_accessor() {
        let cfg = cfg_order1();
        let mut dob = DisturbanceObserver::new(cfg).expect("valid config");
        let ret = dob.update(0.5, 0.3).expect("update ok");
        assert_eq!(ret, dob.disturbance_estimate());
    }

    // -----------------------------------------------------------------------
    // 8. reset() clears state
    // -----------------------------------------------------------------------
    #[test]
    fn test_reset() {
        let cfg = cfg_order1();
        let a_n = cfg.a_n;
        let b_n = cfg.b_n;
        let dt = cfg.dt;
        let mut dob = DisturbanceObserver::new(cfg).expect("valid config");

        // Drive with a disturbance
        let ys = simulate_disturbed(a_n, b_n, dt, 0.0, 3.0, 0.0, 300);
        for &y_k in &ys[..300] {
            let _ = dob.update(0.0, y_k);
        }

        dob.reset();
        assert_eq!(dob.disturbance_estimate(), 0.0);
        // After reset, the first update should act like the first call (no derivative)
        let d = dob.update(0.0, 0.0).expect("post-reset update ok");
        assert_eq!(d, 0.0);
    }
}
