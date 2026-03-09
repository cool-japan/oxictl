// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright COOLJAPAN OU (Team Kitasan)
//
// D-type Iterative Learning Control (derivative-based).
// Update law: u_{k+1}[n] = u_k[n] + L · (e_k[n+1] - e_k[n]) / dt
// The forward-difference approximation of ė_k drives the feedforward
// update, making D-type ILC particularly effective for ramp references.

use crate::core::scalar::ControlScalar;

use super::IlcError;

/// D-type ILC for SISO systems.
///
/// Uses the forward-difference derivative of the tracking error as the
/// learning signal:
///
/// ```text
/// de_k[n] = (e_k[n+1] - e_k[n]) / dt,   n = 0 … TRIAL_LEN-2
/// de_k[TRIAL_LEN-1] = 0                  (boundary condition)
/// u_{k+1}[n] = u_k[n] + L · de_k[n]
/// ```
pub struct DTypeIlc<S, const TRIAL_LEN: usize> {
    /// Current feedforward signal.
    u_ff: [S; TRIAL_LEN],
    /// Tracking error from the most recently completed trial.
    e_prev: [S; TRIAL_LEN],
    /// Scalar learning gain L.
    learning_gain: S,
    /// Sampling period (seconds).
    dt: S,
    /// Number of completed trials.
    trial: usize,
}

impl<S: ControlScalar, const TRIAL_LEN: usize> DTypeIlc<S, TRIAL_LEN> {
    /// Create a new D-type ILC controller.
    ///
    /// # Errors
    /// Returns [`IlcError::InvalidGain`] when `dt` is not strictly positive
    /// or `learning_gain` is non-finite.
    pub fn new(learning_gain: S, dt: S) -> Result<Self, IlcError> {
        if !dt.is_finite() || dt <= S::ZERO {
            return Err(IlcError::InvalidGain);
        }
        if !learning_gain.is_finite() {
            return Err(IlcError::InvalidGain);
        }
        Ok(Self {
            u_ff: [S::ZERO; TRIAL_LEN],
            e_prev: [S::ZERO; TRIAL_LEN],
            learning_gain,
            dt,
            trial: 0,
        })
    }

    /// Apply the D-type learning update using the error from the completed trial.
    ///
    /// # Errors
    /// Returns [`IlcError::NotConverged`] if any updated feedforward value is
    /// non-finite.
    pub fn update(&mut self, error: &[S; TRIAL_LEN]) -> Result<&[S; TRIAL_LEN], IlcError> {
        for n in 0..TRIAL_LEN {
            let de = if n < TRIAL_LEN - 1 {
                (error[n + 1] - error[n]) / self.dt
            } else {
                S::ZERO // boundary condition
            };
            let new_u = self.u_ff[n] + self.learning_gain * de;
            if !new_u.is_finite() {
                return Err(IlcError::NotConverged);
            }
            self.u_ff[n] = new_u;
            self.e_prev[n] = error[n];
        }
        self.trial += 1;
        Ok(&self.u_ff)
    }

    /// Return a reference to the current feedforward signal.
    #[inline]
    pub fn feedforward(&self) -> &[S; TRIAL_LEN] {
        &self.u_ff
    }

    /// Return the number of completed trials.
    #[inline]
    pub fn trial_count(&self) -> usize {
        self.trial
    }

    /// Reset the controller to its initial (zero) state.
    pub fn reset(&mut self) {
        self.u_ff = [S::ZERO; TRIAL_LEN];
        self.e_prev = [S::ZERO; TRIAL_LEN];
        self.trial = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 10;

    /// Zero error → derivative is zero → u_ff unchanged (all zeros).
    #[test]
    fn zero_error_no_update() {
        let mut ilc: DTypeIlc<f64, N> = DTypeIlc::new(1.0, 0.01).unwrap();
        let zero_err = [0.0_f64; N];
        let u = ilc.update(&zero_err).unwrap();
        for &v in u.iter() {
            assert_eq!(v, 0.0);
        }
    }

    /// A linear ramp error [0, dt, 2dt, …] has constant forward difference = 1.
    /// With dt=0.1 and L=0.5: de[n] = (0.1*(n+1) - 0.1*n) / 0.1 = 1.0
    /// So u[n] += 0.5 * 1.0 = 0.5 for all n except the last.
    #[test]
    fn ramp_tracking_uniform_update() {
        const DT: f64 = 0.1;
        let mut ilc: DTypeIlc<f64, N> = DTypeIlc::new(0.5, DT).unwrap();
        let mut ramp = [0.0_f64; N];
        for (n, v) in ramp.iter_mut().enumerate() {
            *v = DT * n as f64;
        }
        let u = ilc.update(&ramp).unwrap();
        // For n=0..N-2: de = (ramp[n+1]-ramp[n])/DT = 1.0; update = 0.5*1.0 = 0.5
        for (n, &u_n) in u.iter().enumerate().take(N - 1) {
            assert!((u_n - 0.5).abs() < 1e-10, "u[{n}] = {}, expected 0.5", u_n);
        }
        // Last element: boundary condition de=0, so update = 0
        assert!(
            u[N - 1].abs() < 1e-10,
            "u[N-1] = {}, expected 0.0",
            u[N - 1]
        );
    }

    /// dt ≤ 0 or non-finite should return InvalidGain.
    #[test]
    fn invalid_dt_rejected() {
        assert!(DTypeIlc::<f64, N>::new(1.0, 0.0).is_err());
        assert!(DTypeIlc::<f64, N>::new(1.0, -0.01).is_err());
        assert!(DTypeIlc::<f64, N>::new(1.0, f64::NAN).is_err());
        assert!(DTypeIlc::<f64, N>::new(f64::NAN, 0.01).is_err());
    }

    /// Trial counter should increment on each update call.
    #[test]
    fn trial_count_increments() {
        let mut ilc: DTypeIlc<f64, N> = DTypeIlc::new(0.5, 0.01).unwrap();
        let err = [0.1_f64; N];
        for k in 1..=4 {
            ilc.update(&err).unwrap();
            assert_eq!(ilc.trial_count(), k);
        }
    }

    /// Boundary condition: last element of de should be zero, leaving u[N-1] = 0
    /// when starting from zero feedforward.
    #[test]
    fn boundary_condition_last_element() {
        let mut ilc: DTypeIlc<f64, N> = DTypeIlc::new(2.0, 0.1).unwrap();
        // All-ones error: de[n] = (1.0 - 1.0)/0.1 = 0 for all interior points
        let err = [1.0_f64; N];
        let u = ilc.update(&err).unwrap();
        // de[n] = (1.0 - 1.0)/0.1 = 0 for all n (including last via boundary)
        for &v in u.iter() {
            assert!(v.abs() < 1e-10, "expected 0.0, got {v}");
        }
    }

    /// reset() should zero everything and reset trial count.
    #[test]
    fn reset_clears_state() {
        let mut ilc: DTypeIlc<f64, N> = DTypeIlc::new(0.5, 0.01).unwrap();
        let mut ramp = [0.0_f64; N];
        for (n, v) in ramp.iter_mut().enumerate() {
            *v = 0.01 * n as f64;
        }
        ilc.update(&ramp).unwrap();
        assert_eq!(ilc.trial_count(), 1);

        ilc.reset();

        assert_eq!(ilc.trial_count(), 0);
        for &v in ilc.feedforward().iter() {
            assert_eq!(v, 0.0);
        }
    }

    /// Zero learning gain should leave u_ff unchanged regardless of error shape.
    #[test]
    fn zero_learning_gain_no_update() {
        let mut ilc: DTypeIlc<f64, N> = DTypeIlc::new(0.0, 0.1).unwrap();
        let mut ramp = [0.0_f64; N];
        for (n, v) in ramp.iter_mut().enumerate() {
            *v = 0.1 * n as f64;
        }
        let u = ilc.update(&ramp).unwrap();
        for &v in u.iter() {
            assert_eq!(v, 0.0);
        }
    }
}
