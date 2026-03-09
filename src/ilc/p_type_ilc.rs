// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright COOLJAPAN OU (Team Kitasan)
//
// P-type Iterative Learning Control (Arimoto 1984).
// Update law: u_{k+1}[n] = u_k[n] + L * e_k[n]
// Convergence condition: |1 - L*B| < 1 where B is the plant Markov parameter.

use crate::core::scalar::ControlScalar;

use super::IlcError;

/// P-type ILC for SISO systems.
///
/// Learns a feedforward signal over repeated trials of fixed length `TRIAL_LEN`.
/// After each trial the update law
///
/// ```text
/// u_{k+1}[n] = u_k[n] + L · e_k[n],  n = 0 … TRIAL_LEN-1
/// ```
///
/// drives the tracking error to zero provided `|1 - L·B| < 1`.
pub struct PTypeIlc<S, const TRIAL_LEN: usize> {
    /// Current feedforward signal (applied during the next trial).
    u_ff: [S; TRIAL_LEN],
    /// Tracking error recorded during the most recent trial.
    e_prev: [S; TRIAL_LEN],
    /// Scalar learning gain L.
    learning_gain: S,
    /// Number of trials completed so far.
    trial: usize,
}

impl<S: ControlScalar, const TRIAL_LEN: usize> PTypeIlc<S, TRIAL_LEN> {
    /// Create a new P-type ILC controller.
    ///
    /// # Errors
    /// Returns [`IlcError::InvalidGain`] when `learning_gain` is zero, negative,
    /// or non-finite.
    pub fn new(learning_gain: S) -> Result<Self, IlcError> {
        if !learning_gain.is_finite() || learning_gain <= S::ZERO {
            return Err(IlcError::InvalidGain);
        }
        Ok(Self {
            u_ff: [S::ZERO; TRIAL_LEN],
            e_prev: [S::ZERO; TRIAL_LEN],
            learning_gain,
            trial: 0,
        })
    }

    /// Apply the P-type learning update using the error from the completed trial.
    ///
    /// Computes `u_{k+1}[n] = u_k[n] + L · error[n]`, stores the error, increments
    /// the trial counter, and returns a reference to the updated feedforward signal.
    ///
    /// # Errors
    /// Returns [`IlcError::NotConverged`] if any element of the updated feedforward
    /// is non-finite (numerical blow-up detected).
    pub fn update(&mut self, error: &[S; TRIAL_LEN]) -> Result<&[S; TRIAL_LEN], IlcError> {
        for (n, &e) in error.iter().enumerate() {
            let new_u = self.u_ff[n] + self.learning_gain * e;
            if !new_u.is_finite() {
                return Err(IlcError::NotConverged);
            }
            self.u_ff[n] = new_u;
            self.e_prev[n] = e;
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

    /// Compute the RMS difference between `error` and the previous trial's error.
    ///
    /// This measures how much the error profile changed between trials and can be
    /// used as a convergence criterion.
    pub fn convergence_error(&self, error: &[S; TRIAL_LEN]) -> S {
        if TRIAL_LEN == 0 {
            return S::ZERO;
        }
        let mut sum_sq = S::ZERO;
        for (&e, &ep) in error.iter().zip(self.e_prev.iter()) {
            let diff = e - ep;
            sum_sq += diff * diff;
        }
        let mean_sq = sum_sq / S::from_f64(TRIAL_LEN as f64);
        mean_sq.sqrt()
    }

    /// Reset the controller: zero feedforward, zero stored error, reset trial counter.
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

    const N: usize = 16;

    /// Helper: compute RMS of an array.
    fn rms(arr: &[f64; N]) -> f64 {
        let mut s = 0.0_f64;
        for &v in arr.iter() {
            s += v * v;
        }
        (s / N as f64).sqrt()
    }

    /// Zero error should leave u_ff unchanged (all zeros).
    #[test]
    fn zero_error_no_update() {
        let mut ilc: PTypeIlc<f64, N> = PTypeIlc::new(0.5).unwrap();
        let zero_err = [0.0_f64; N];
        let u = ilc.update(&zero_err).unwrap();
        for &v in u.iter() {
            assert_eq!(v, 0.0);
        }
    }

    /// With plant gain = 1 and L = 0.5, tracking error should converge to zero.
    ///
    /// Simulation: y_k[n] = u_ff[n] (plant = identity), reference = 1.0.
    /// Error: e_k[n] = ref - y_k[n] = 1 - u_ff[n].
    /// Update: u_{k+1} = u_k + 0.5*(1 - u_k) → converges to 1.
    #[test]
    fn constant_tracking_convergence() {
        const REF: f64 = 1.0;
        let mut ilc: PTypeIlc<f64, N> = PTypeIlc::new(0.5).unwrap();

        for _ in 0..20 {
            let u_ff = *ilc.feedforward();
            let mut error = [0.0_f64; N];
            for (n, e) in error.iter_mut().enumerate() {
                *e = REF - u_ff[n];
            }
            ilc.update(&error).unwrap();
        }
        // After 20 trials error should be very small
        let u_ff = *ilc.feedforward();
        let mut final_err = [0.0_f64; N];
        for (n, e) in final_err.iter_mut().enumerate() {
            *e = REF - u_ff[n];
        }
        assert!(
            rms(&final_err) < 1e-5,
            "ILC should converge, got rms={}",
            rms(&final_err)
        );
    }

    /// RMS error decreases monotonically over trials.
    #[test]
    fn rms_error_decreases() {
        const REF: f64 = 2.0;
        let mut ilc: PTypeIlc<f64, N> = PTypeIlc::new(0.4).unwrap();
        let mut prev_rms = f64::INFINITY;

        for trial in 0..15 {
            let u_ff = *ilc.feedforward();
            let mut error = [0.0_f64; N];
            for (n, e) in error.iter_mut().enumerate() {
                *e = REF - u_ff[n];
            }
            let cur_rms = rms(&error);
            // After the first trial we expect monotone decrease
            if trial > 0 {
                assert!(
                    cur_rms < prev_rms + 1e-12,
                    "trial {trial}: rms did not decrease ({cur_rms} >= {prev_rms})"
                );
            }
            prev_rms = cur_rms;
            ilc.update(&error).unwrap();
        }
    }

    /// Invalid (non-positive / non-finite) learning gain should return an error.
    #[test]
    fn invalid_gain_rejected() {
        assert!(PTypeIlc::<f64, N>::new(0.0).is_err());
        assert!(PTypeIlc::<f64, N>::new(-1.0).is_err());
        assert!(PTypeIlc::<f64, N>::new(f64::NAN).is_err());
        assert!(PTypeIlc::<f64, N>::new(f64::INFINITY).is_err());
    }

    /// After reset, state should be zeroed and trial count should be 0.
    #[test]
    fn reset_clears_state() {
        let mut ilc: PTypeIlc<f64, N> = PTypeIlc::new(0.5).unwrap();
        let ones = [1.0_f64; N];
        ilc.update(&ones).unwrap();
        ilc.update(&ones).unwrap();
        assert_eq!(ilc.trial_count(), 2);

        ilc.reset();

        assert_eq!(ilc.trial_count(), 0);
        for &v in ilc.feedforward().iter() {
            assert_eq!(v, 0.0);
        }
    }

    /// Trial counter should increment by 1 on each call to update.
    #[test]
    fn trial_count_increments() {
        let mut ilc: PTypeIlc<f64, N> = PTypeIlc::new(0.3).unwrap();
        let err = [0.1_f64; N];
        for k in 1..=5 {
            ilc.update(&err).unwrap();
            assert_eq!(ilc.trial_count(), k);
        }
    }

    /// convergence_error should return 0 when error profile is unchanged.
    #[test]
    fn convergence_error_zero_on_repeat() {
        let mut ilc: PTypeIlc<f64, N> = PTypeIlc::new(0.5).unwrap();
        let err = [0.5_f64; N];
        ilc.update(&err).unwrap();
        // Same error again → convergence_error = 0
        let ce = ilc.convergence_error(&err);
        assert!(ce.abs() < 1e-12, "convergence_error should be 0, got {ce}");
    }

    /// f32 precision sanity check.
    #[test]
    fn f32_basic() {
        let mut ilc: PTypeIlc<f32, 8> = PTypeIlc::new(0.5_f32).unwrap();
        let err = [1.0_f32; 8];
        let u = ilc.update(&err).unwrap();
        for &v in u.iter() {
            assert!((v - 0.5_f32).abs() < 1e-6);
        }
    }
}
