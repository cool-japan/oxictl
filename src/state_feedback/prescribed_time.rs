//! Prescribed-Time Control.
//!
//! Achieves exact convergence to zero by a user-specified deadline T*,
//! using a time-varying gain μ(t) that diverges as t → T*:
//! ```text
//!   μ(t) = 1 / (T* − t + ε)
//! ```
//! where ε > 0 is a small regularisation constant that prevents a hard
//! singularity at t = T*.
//!
//! For a scalar system `ẋ = u + d(t)` the control law is:
//! ```text
//!   u(t) = −k · μ(t)^λ · x
//! ```
//! The state transformation ξ = μ^λ · x satisfies `ξ̇ = (λ/(T*−t))·ξ + μ^λ·d`,
//! and with k > λ the closed-loop ξ-system decays to zero before T*.
//!
//! References:
//! - Krishnamurthy, P., Khorrami, F. & Bhatt, R. (2020). "A dynamic high-gain
//!   design for prescribed-time regulation of nonlinear systems." Automatica
//!   115, 108860.
//! - Song, Y., Wang, Y. & Holloway, J. (2017). "Time-varying feedback for
//!   regulation of normal-form nonlinear systems in prescribed finite time."
//!   Automatica 83, 243–251.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in prescribed-time control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrescribedTimeError {
    /// target_time, epsilon, or dt is non-positive.
    InvalidTime,
    /// k must be positive; lambda must be ≥ 1.
    InvalidGain,
    /// Current simulation time has reached or exceeded target_time − epsilon.
    TimeExpired,
}

impl core::fmt::Display for PrescribedTimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PrescribedTimeError::InvalidTime => {
                f.write_str("target_time, epsilon and dt must all be strictly positive")
            }
            PrescribedTimeError::InvalidGain => {
                f.write_str("k must be positive and lambda must be >= 1")
            }
            PrescribedTimeError::TimeExpired => {
                f.write_str("prescribed time has expired (t >= target_time - epsilon)")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PrescribedTimeController
// ---------------------------------------------------------------------------

/// Prescribed-Time Controller for scalar systems.
///
/// Guarantees convergence to zero by the user-specified deadline T*.
/// The control gain grows from a finite value at t = 0 toward infinity as
/// t → T*, ensuring the state is driven to zero before the deadline.
///
/// # Type parameters
/// - `S`: scalar type implementing [`ControlScalar`] (f32 or f64).
#[derive(Debug, Clone, Copy)]
pub struct PrescribedTimeController<S: ControlScalar> {
    /// Prescribed convergence deadline T*.
    target_time: S,
    /// Regularisation constant ε (prevents 1/(T*−t) singularity at t = T*).
    epsilon: S,
    /// Linear feedback gain k.
    k: S,
    /// State-scaling power λ ≥ 1.
    lambda: S,
    /// Current simulation time t (advanced by dt on each call to `update`).
    t: S,
    /// Sampling period.
    dt: S,
}

impl<S: ControlScalar> PrescribedTimeController<S> {
    /// Construct a new prescribed-time controller.
    ///
    /// # Arguments
    /// - `target_time`: convergence deadline T* > 0.
    /// - `epsilon`: regularisation offset ε > 0 (typically 1% of T*).
    /// - `k`: feedback gain; must be > 0.  For robustness, choose k > λ.
    /// - `lambda`: state-scaling power; must be ≥ 1.
    /// - `dt`: sampling period; must be > 0.
    pub fn new(
        target_time: S,
        epsilon: S,
        k: S,
        lambda: S,
        dt: S,
    ) -> Result<Self, PrescribedTimeError> {
        if target_time <= S::ZERO || epsilon <= S::ZERO || dt <= S::ZERO {
            return Err(PrescribedTimeError::InvalidTime);
        }
        if k <= S::ZERO || lambda < S::ONE {
            return Err(PrescribedTimeError::InvalidGain);
        }
        Ok(Self {
            target_time,
            epsilon,
            k,
            lambda,
            t: S::ZERO,
            dt,
        })
    }

    /// Compute the control output for the current state `x`.
    ///
    /// Returns `Err(TimeExpired)` when `t ≥ target_time − epsilon`.
    ///
    /// Otherwise computes:
    /// ```text
    ///   μ = 1 / (T* − t + ε)
    ///   u = −k · μ^λ · x
    /// ```
    /// and advances the internal clock by dt.
    pub fn update(&mut self, x: S) -> Result<S, PrescribedTimeError> {
        if self.t >= self.target_time - self.epsilon {
            return Err(PrescribedTimeError::TimeExpired);
        }
        let remaining = self.target_time - self.t + self.epsilon;
        let mu = S::ONE / remaining;
        let u = -self.k * mu.powf(self.lambda) * x;
        self.t += self.dt;
        Ok(u)
    }

    /// Return time remaining until T* (clamped to zero when past deadline).
    pub fn time_remaining(&self) -> S {
        let remaining = self.target_time - self.t;
        if remaining < S::ZERO {
            S::ZERO
        } else {
            remaining
        }
    }

    /// Reset the internal clock to zero.
    pub fn reset(&mut self) {
        self.t = S::ZERO;
    }

    /// Return the current internal time.
    pub fn current_time(&self) -> S {
        self.t
    }

    /// Return the prescribed deadline T*.
    pub fn target_time(&self) -> S {
        self.target_time
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.001;

    // 1. Zero state → zero control (u = -k * mu^lambda * 0 = 0)
    #[test]
    fn zero_state_zero_control() {
        let mut ctrl =
            PrescribedTimeController::<f64>::new(1.0, 0.01, 2.0, 1.5, DT).expect("valid params");
        let u = ctrl.update(0.0).expect("update ok");
        assert!(u.abs() < 1e-14, "u should be zero for x=0, got {}", u);
    }

    // 2. Control grows as t → T*: compare |u| at t=0 vs after 900 steps
    //    (mu increases monotonically, so |u| grows for constant x)
    #[test]
    fn control_grows_near_target_time() {
        // T*=1.0, eps=0.01, k=1.0, lambda=2.0, dt=0.001
        let mut ctrl =
            PrescribedTimeController::<f64>::new(1.0, 0.01, 1.0, 2.0, DT).expect("valid params");
        let u0 = ctrl.update(1.0).expect("first update").abs();
        // Advance to t ≈ 0.9 (900 more steps)
        for _ in 0..899 {
            ctrl.update(1.0).expect("update ok");
        }
        let u_late = ctrl.update(1.0).expect("late update").abs();
        assert!(
            u_late > u0,
            "control magnitude should grow near deadline: u0={:.6}, u_late={:.6}",
            u0,
            u_late
        );
    }

    // 3. TimeExpired after t reaches T* - epsilon
    #[test]
    fn time_expiry_returns_error() {
        // T*=0.1, eps=0.01, dt=0.001 → expires after ~90 steps
        let mut ctrl =
            PrescribedTimeController::<f64>::new(0.1, 0.01, 1.0, 1.0, DT).expect("valid params");
        let mut expired = false;
        for _ in 0..200 {
            match ctrl.update(0.5) {
                Ok(_) => {}
                Err(PrescribedTimeError::TimeExpired) => {
                    expired = true;
                    break;
                }
                Err(e) => panic!("unexpected error: {}", e),
            }
        }
        assert!(expired, "controller should have expired");
    }

    // 4. Known gain formula at t=0:
    //    T*=1.0, eps=0.01, k=1.0, lambda=1.0
    //    mu = 1/(1.0 - 0 + 0.01) = 1/1.01
    //    u = -1.0 * (1/1.01)^1.0 * 2.0 = -2.0/1.01
    #[test]
    fn known_gain_formula() {
        let mut ctrl =
            PrescribedTimeController::<f64>::new(1.0, 0.01, 1.0, 1.0, DT).expect("valid params");
        let x = 2.0_f64;
        let u = ctrl.update(x).expect("update ok");
        let expected = -2.0_f64 / 1.01_f64;
        assert!(
            (u - expected).abs() < 1e-10,
            "expected u={:.8}, got u={:.8}",
            expected,
            u
        );
    }

    // 5. Controller reduces state magnitude over 500 steps
    //    plant: x[n+1] = x[n] + dt*u[n]  (single integrator)
    #[test]
    fn controller_reduces_state() {
        // T*=2.0, eps=0.01, k=2.0, lambda=1.5, dt=0.001
        let mut ctrl =
            PrescribedTimeController::<f64>::new(2.0, 0.01, 2.0, 1.5, DT).expect("valid params");
        let mut x = 1.0_f64;
        let x_init = x;
        for _ in 0..500 {
            match ctrl.update(x) {
                Ok(u) => {
                    // Single integrator: ẋ = u
                    x += DT * u;
                }
                Err(_) => break,
            }
        }
        assert!(
            x.abs() < x_init.abs(),
            "state should decrease: x_init={:.4}, x_final={:.4}",
            x_init,
            x
        );
    }

    // 6. Invalid target_time = 0 → InvalidTime
    #[test]
    fn invalid_target_time_returns_error() {
        let res = PrescribedTimeController::<f64>::new(0.0, 0.01, 1.0, 1.0, DT);
        assert!(
            matches!(res, Err(PrescribedTimeError::InvalidTime)),
            "expected InvalidTime, got {:?}",
            res.err()
        );
    }

    // 7. time_remaining() decreases monotonically with each update
    #[test]
    fn time_remaining_decreases() {
        let mut ctrl =
            PrescribedTimeController::<f64>::new(1.0, 0.01, 1.0, 1.0, DT).expect("valid params");
        let r0 = ctrl.time_remaining();
        ctrl.update(1.0).expect("update ok");
        let r1 = ctrl.time_remaining();
        assert!(
            r1 < r0,
            "time_remaining should decrease: r0={:.6}, r1={:.6}",
            r0,
            r1
        );
    }

    // 8. reset() restores internal clock to zero
    #[test]
    fn reset_restores_clock() {
        let mut ctrl =
            PrescribedTimeController::<f64>::new(1.0, 0.01, 1.0, 1.0, DT).expect("valid params");
        for _ in 0..100 {
            ctrl.update(0.5).expect("update ok");
        }
        ctrl.reset();
        assert!(
            ctrl.current_time().abs() < 1e-14,
            "time should be zero after reset: t={}",
            ctrl.current_time()
        );
    }

    // 9. lambda < 1 → InvalidGain
    #[test]
    fn invalid_lambda_returns_error() {
        let res = PrescribedTimeController::<f64>::new(1.0, 0.01, 1.0, 0.5, DT);
        assert!(
            matches!(res, Err(PrescribedTimeError::InvalidGain)),
            "expected InvalidGain, got {:?}",
            res.err()
        );
    }
}
