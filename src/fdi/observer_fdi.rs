//! Observer-Based Fault Detection & Isolation
//!
//! Implements a Luenberger observer residual generator for fault detection and
//! isolation. The observer is:
//!
//!   x̂_dot = A·x̂ + B·u + L·(y − C·x̂)
//!
//! Discretised with forward Euler at step `dt`:
//!
//!   x̂[k+1] = x̂[k] + dt·(A·x̂[k] + B·u[k] + L·innovation[k])
//!
//! where `innovation[k] = y[k] − C·x̂[k]`.
//! The post-update residual r[k] = y[k] − C·x̂[k+1] converges to zero when
//! the model matches the plant.  A sensor/actuator fault shifts the residual.
//! Per-channel thresholding enables isolation of which output is affected.

#![cfg_attr(not(feature = "std"), no_std)]

use crate::core::scalar::ControlScalar;
use crate::fdi::parity_space::FdiError;

/// Per-step result from the observer-based FDI update.
#[derive(Debug, Clone, Copy)]
pub struct FaultIsolationResult<S: ControlScalar, const M: usize> {
    /// Residual vector r = y − C·x̂ (post-update).
    pub residual: [S; M],
    /// Boolean mask: `threshold_exceeded[i]` is `true` if |r[i]| > threshold.
    pub threshold_exceeded: [bool; M],
    /// Index of the output channel with the largest |r[i]| that exceeded the
    /// threshold, or `None` if no channel exceeded.
    pub largest_channel: Option<usize>,
}

/// Luenberger observer-based fault detector/isolator.
///
/// # Type Parameters
/// * `S` — scalar type implementing [`ControlScalar`].
/// * `N` — state dimension.
/// * `M` — output (measurement) dimension.
/// * `I` — input dimension.
///
/// # Observer Design
/// The observer gain matrix `L` (N×M) must be chosen so that `A − L·C` is
/// stable (all eigenvalues inside the unit disk for discrete time, or with
/// negative real parts for the continuous-time design used here).
#[derive(Debug, Clone)]
pub struct ObserverFdi<S: ControlScalar, const N: usize, const M: usize, const I: usize> {
    /// System matrix A (N×N).
    a: [[S; N]; N],
    /// Input matrix B (N×I).
    b: [[S; N]; I],
    /// Output matrix C (M×N): y = C·x.
    c: [[S; M]; N],
    /// Observer gain L (N×M): correction term.
    l: [[S; N]; M],
    /// Observer state estimate x̂.
    x_hat: [S; N],
    /// Most recently computed residual.
    residual: [S; M],
    /// Per-channel absolute threshold for fault declaration.
    threshold: S,
    /// Euler integration step size (seconds).
    dt: S,
    /// Exponential moving average of the residual (for trend tracking).
    r_ema: [S; M],
    /// EMA forgetting factor α ∈ (0, 1).
    alpha: S,
}

impl<S: ControlScalar, const N: usize, const M: usize, const I: usize> ObserverFdi<S, N, M, I> {
    /// Construct a new [`ObserverFdi`].
    ///
    /// # Arguments
    /// * `a`         — system matrix A (N×N).
    /// * `b`         — input matrix B (N×I).
    /// * `c`         — output matrix C (M×N).
    /// * `l`         — observer gain L (N×M).
    /// * `threshold` — positive per-channel residual threshold.
    /// * `dt`        — positive Euler integration step size (s).
    /// * `alpha`     — EMA factor, strictly in (0, 1).
    ///
    /// # Errors
    /// Returns [`FdiError::InvalidParameter`] if any of the following hold:
    /// - `threshold ≤ 0`
    /// - `dt ≤ 0`
    /// - `alpha ≤ 0` or `alpha ≥ 1`
    pub fn new(
        a: [[S; N]; N],
        b: [[S; N]; I],
        c: [[S; M]; N],
        l: [[S; N]; M],
        threshold: S,
        dt: S,
        alpha: S,
    ) -> Result<Self, FdiError> {
        if threshold <= S::ZERO || dt <= S::ZERO || alpha <= S::ZERO || alpha >= S::ONE {
            return Err(FdiError::InvalidParameter);
        }
        Ok(Self {
            a,
            b,
            c,
            l,
            x_hat: [S::ZERO; N],
            residual: [S::ZERO; M],
            threshold,
            dt,
            r_ema: [S::ZERO; M],
            alpha,
        })
    }

    /// Process one measurement sample and return isolation results.
    ///
    /// Steps performed:
    /// 1. Compute pre-update innovation: `innovation = y − C·x̂`
    /// 2. Euler observer update: `x̂ += dt·(A·x̂ + B·u + L·innovation)`
    /// 3. Compute post-update residual: `r = y − C·x̂`
    /// 4. Update per-channel EMA: `r_ema = α·r_ema + (1−α)·r`
    /// 5. Build [`FaultIsolationResult`]
    #[allow(clippy::needless_range_loop)]
    pub fn update(
        &mut self,
        u: &[S; I],
        y: &[S; M],
    ) -> Result<FaultIsolationResult<S, M>, FdiError> {
        // Step 1: innovation = y − C·x̂ (pre-update)
        let mut innovation = [S::ZERO; M];
        for i in 0..M {
            let mut cx_i = S::ZERO;
            for j in 0..N {
                cx_i += self.c[j][i] * self.x_hat[j];
            }
            innovation[i] = y[i] - cx_i;
        }

        // Step 2: Euler update
        // dx[j] = A[j][k]·x̂[k] + B[j][k]·u[k] + L[j][m]·innovation[m]
        let mut dx = [S::ZERO; N];
        for j in 0..N {
            for k in 0..N {
                dx[j] += self.a[j][k] * self.x_hat[k];
            }
            for k in 0..I {
                dx[j] += self.b[k][j] * u[k];
            }
            for m in 0..M {
                dx[j] += self.l[m][j] * innovation[m];
            }
        }
        for j in 0..N {
            self.x_hat[j] += self.dt * dx[j];
        }

        // Step 3: post-update residual
        let mut r = [S::ZERO; M];
        for i in 0..M {
            let mut cx_i = S::ZERO;
            for j in 0..N {
                cx_i += self.c[j][i] * self.x_hat[j];
            }
            r[i] = y[i] - cx_i;
        }
        self.residual = r;

        // Step 4: EMA update
        let one_minus_alpha = S::ONE - self.alpha;
        for i in 0..M {
            self.r_ema[i] = self.alpha * self.r_ema[i] + one_minus_alpha * r[i];
        }

        // Step 5: build result
        let mut threshold_exceeded = [false; M];
        let mut max_abs = S::ZERO;
        let mut largest_channel: Option<usize> = None;

        for i in 0..M {
            let abs_ri = if r[i] < S::ZERO { -r[i] } else { r[i] };
            if abs_ri > self.threshold {
                threshold_exceeded[i] = true;
                if abs_ri > max_abs {
                    max_abs = abs_ri;
                    largest_channel = Some(i);
                }
            }
        }

        Ok(FaultIsolationResult {
            residual: r,
            threshold_exceeded,
            largest_channel,
        })
    }

    /// Return a reference to the most recently computed residual vector.
    pub fn residual(&self) -> &[S; M] {
        &self.residual
    }

    /// Return the EMA residual vector.
    pub fn residual_ema(&self) -> &[S; M] {
        &self.r_ema
    }

    /// Reset the observer state.
    ///
    /// Sets x̂ = `x0` and clears residual and EMA arrays.
    pub fn reset(&mut self, x0: [S; N]) {
        self.x_hat = x0;
        self.residual = [S::ZERO; M];
        self.r_ema = [S::ZERO; M];
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple scalar observer (N=1, M=1, I=1).
    /// System: dx/dt = -a*x + u, y = x.
    /// Observer gain L chosen for fast convergence.
    fn make_scalar_observer(
        a_val: f64,
        l_val: f64,
        threshold: f64,
        dt: f64,
        alpha: f64,
    ) -> ObserverFdi<f64, 1, 1, 1> {
        let a = [[-a_val]];
        let b = [[1.0_f64]];
        let c = [[1.0_f64]];
        let l = [[l_val]];
        ObserverFdi::new(a, b, c, l, threshold, dt, alpha).expect("valid params")
    }

    #[test]
    fn perfect_model_zero_residual() {
        // System A=-1, B=0, C=1, L=5 — stable observer.
        // x_true starts at 0, u=0, y=0 → residual → 0.
        let mut obs = make_scalar_observer(1.0, 5.0, 0.5, 0.01, 0.9);
        for _ in 0..200 {
            let res = obs.update(&[0.0], &[0.0]).expect("ok");
            // After convergence the residual should remain near zero
            assert!(
                !res.threshold_exceeded[0],
                "residual should not exceed threshold"
            );
        }
        assert!(obs.residual()[0].abs() < 1e-6);
    }

    #[test]
    fn step_fault_detected_channel0() {
        // Inject a step fault on y[0]: true system outputs 0 but we report 5.
        // Observer expects 0 → residual ≈ 5 → detected on channel 0.
        let mut obs = make_scalar_observer(1.0, 5.0, 0.5, 0.01, 0.5);
        // Converge first
        for _ in 0..200 {
            obs.update(&[0.0], &[0.0]).expect("ok");
        }
        // Inject fault
        let res = obs.update(&[0.0], &[5.0]).expect("ok");
        assert_eq!(
            res.largest_channel,
            Some(0),
            "fault should show on channel 0"
        );
        assert!(res.threshold_exceeded[0]);
    }

    #[test]
    fn observer_convergence() {
        // Observer error dynamics: de/dt = (A - L*C)*e.
        // With A=0, C=1, L=10: de/dt = -10*e → error decays exponentially to 0.
        // True plant: x_true = 1 (constant), y = 1.  Observer starts at x_hat=0.
        // After convergence, x_hat → 1 and residual → 0.
        let a = [[0.0_f64]]; // A = 0
        let b = [[0.0_f64]];
        let c = [[1.0_f64]];
        let l = [[10.0_f64]]; // large gain → fast convergence
        let mut obs: ObserverFdi<f64, 1, 1, 1> =
            ObserverFdi::new(a, b, c, l, 0.5, 0.01, 0.5).expect("ok");
        let mut last_res = 0.0_f64;
        for _ in 0..500 {
            let res = obs.update(&[0.0], &[1.0]).expect("ok");
            last_res = res.residual[0];
        }
        // After 500 steps at dt=0.01 with eigenvalue -10, residual should be < 0.001
        assert!(
            last_res.abs() < 0.01,
            "observer should have converged, residual={last_res}"
        );
    }

    #[test]
    fn reset_clears_observer_state() {
        let mut obs = make_scalar_observer(1.0, 5.0, 0.5, 0.01, 0.5);
        // Drive x_hat away from zero
        for _ in 0..50 {
            obs.update(&[0.0], &[10.0]).expect("ok");
        }
        // Reset
        obs.reset([0.0]);
        // Immediately after reset x_hat = 0 → with y=0, residual ≈ 0
        let res = obs.update(&[0.0], &[0.0]).expect("ok");
        assert!(
            !res.threshold_exceeded[0],
            "after reset residual should be near zero"
        );
    }

    #[test]
    fn threshold_per_channel_2d() {
        // 2D output: only inject fault on channel 1.
        let a = [[-1.0_f64, 0.0], [0.0, -1.0]];
        let b = [[0.0_f64, 0.0]]; // 2×1 → stored as b[input][state]
        let c = [[1.0_f64, 0.0], [0.0, 1.0]];
        let l = [[5.0_f64, 0.0], [0.0, 5.0]]; // l[output][state]
        let mut obs: ObserverFdi<f64, 2, 2, 1> =
            ObserverFdi::new(a, b, c, l, 0.5, 0.01, 0.5).expect("ok");
        // Converge
        for _ in 0..300 {
            obs.update(&[0.0], &[0.0, 0.0]).expect("ok");
        }
        // Inject fault only on channel 1
        let res = obs.update(&[0.0], &[0.0, 5.0]).expect("ok");
        assert!(
            !res.threshold_exceeded[0],
            "channel 0 should NOT be flagged"
        );
        assert!(res.threshold_exceeded[1], "channel 1 SHOULD be flagged");
        assert_eq!(res.largest_channel, Some(1));
    }

    #[test]
    fn alpha_bounds_validation() {
        let a = [[-1.0_f64]];
        let b = [[0.0_f64]];
        let c = [[1.0_f64]];
        let l = [[5.0_f64]];
        // alpha = 0.0 → invalid
        assert!(
            ObserverFdi::<f64, 1, 1, 1>::new(a, b, c, l, 1.0, 0.01, 0.0).is_err(),
            "alpha=0 should be invalid"
        );
        // alpha = 1.0 → invalid
        assert!(
            ObserverFdi::<f64, 1, 1, 1>::new(a, b, c, l, 1.0, 0.01, 1.0).is_err(),
            "alpha=1 should be invalid"
        );
        // alpha = 0.5 → valid
        assert!(ObserverFdi::<f64, 1, 1, 1>::new(a, b, c, l, 1.0, 0.01, 0.5).is_ok());
    }

    #[test]
    fn ema_tracks_residual_trend() {
        // Inject a persistent fault and verify EMA grows (rather than fluctuates).
        let mut obs = make_scalar_observer(1.0, 2.0, 0.5, 0.01, 0.1);
        // Converge
        for _ in 0..300 {
            obs.update(&[0.0], &[0.0]).expect("ok");
        }
        let ema_before = obs.residual_ema()[0];
        // Inject fault
        for _ in 0..50 {
            obs.update(&[0.0], &[3.0]).expect("ok");
        }
        let ema_after = obs.residual_ema()[0];
        assert!(
            ema_after.abs() > ema_before.abs(),
            "EMA should grow under persistent fault: before={ema_before}, after={ema_after}"
        );
    }
}
