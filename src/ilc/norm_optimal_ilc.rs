// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright COOLJAPAN OU (Team Kitasan)
//
// Norm-Optimal Iterative Learning Control (Rogers et al.).
//
// Minimises the per-trial cost:
//   J = e_{k+1}^T Q e_{k+1} + Δu_k^T R Δu_k
//
// Diagonal approximation of the full Toeplitz plant model:
//   G^T Q G  ≈  g² · q · I,  where g = Markov gain (DC gain approximation)
//
// Optimal update (element-wise):
//   Δu_k[n] = l_opt · e_k[n]
//   l_opt   = g·q / (r + g²·q)
//
// This is the ILC analogue of gradient descent on the quadratic cost surface.

use crate::core::scalar::ControlScalar;

use super::IlcError;

/// Norm-Optimal ILC for SISO systems using a diagonal plant approximation.
///
/// Precomputes the optimal learning gain at construction time and applies
/// it element-wise — identical in structure to P-type ILC, but with the
/// gain derived from the Q/R cost weights and the plant Markov parameter.
pub struct NormOptimalIlc<S, const TRIAL_LEN: usize> {
    /// Current feedforward signal.
    u_ff: [S; TRIAL_LEN],
    /// Tracking error from the most recently completed trial.
    e_prev: [S; TRIAL_LEN],
    /// Precomputed optimal learning gain: l_opt = g·q / (r + g²·q).
    l_opt: S,
    /// Tracking weight Q (> 0).
    q_weight: S,
    /// Control effort weight R (> 0).
    r_weight: S,
    /// Last control update Δu (used in cost computation).
    delta_u: [S; TRIAL_LEN],
    /// Number of completed trials.
    trial: usize,
}

impl<S: ControlScalar, const TRIAL_LEN: usize> NormOptimalIlc<S, TRIAL_LEN> {
    /// Create a new Norm-Optimal ILC controller.
    ///
    /// # Parameters
    /// - `markov_gain` — DC/Markov gain `g` of the plant (must be non-zero and finite).
    /// - `q` — Tracking error weight (must be > 0).
    /// - `r` — Control effort weight (must be > 0).
    ///
    /// # Errors
    /// Returns [`IlcError::InvalidGain`] if any parameter is invalid.
    pub fn new(markov_gain: S, q: S, r: S) -> Result<Self, IlcError> {
        if !markov_gain.is_finite() || markov_gain.abs() <= S::EPSILON {
            return Err(IlcError::InvalidGain);
        }
        if !q.is_finite() || q <= S::ZERO {
            return Err(IlcError::InvalidGain);
        }
        if !r.is_finite() || r <= S::ZERO {
            return Err(IlcError::InvalidGain);
        }

        // l_opt = g·q / (r + g²·q)
        let g2 = markov_gain * markov_gain;
        let l_opt = markov_gain * q / (r + g2 * q);

        Ok(Self {
            u_ff: [S::ZERO; TRIAL_LEN],
            e_prev: [S::ZERO; TRIAL_LEN],
            l_opt,
            q_weight: q,
            r_weight: r,
            delta_u: [S::ZERO; TRIAL_LEN],
            trial: 0,
        })
    }

    /// Apply the norm-optimal learning update.
    ///
    /// `u_{k+1}[n] = u_k[n] + l_opt · error[n]`
    ///
    /// # Errors
    /// Returns [`IlcError::NotConverged`] on numerical blow-up.
    pub fn update(&mut self, error: &[S; TRIAL_LEN]) -> Result<&[S; TRIAL_LEN], IlcError> {
        for (n, &e) in error.iter().enumerate() {
            let du = self.l_opt * e;
            let new_u = self.u_ff[n] + du;
            if !new_u.is_finite() {
                return Err(IlcError::NotConverged);
            }
            self.delta_u[n] = du;
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

    /// Return the precomputed optimal learning gain `l_opt`.
    #[inline]
    pub fn optimal_gain(&self) -> S {
        self.l_opt
    }

    /// Compute the ILC cost for a given error profile.
    ///
    /// `J = q · ‖error‖² + r · ‖Δu‖²`
    ///
    /// Uses the stored `delta_u` from the most recent `update` call.
    pub fn cost(&self, error: &[S; TRIAL_LEN]) -> S {
        let mut e_sq = S::ZERO;
        let mut du_sq = S::ZERO;
        for (&e, &du) in error.iter().zip(self.delta_u.iter()) {
            e_sq += e * e;
            du_sq += du * du;
        }
        self.q_weight * e_sq + self.r_weight * du_sq
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
        self.delta_u = [S::ZERO; TRIAL_LEN];
        self.trial = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 20;

    /// Stability condition: l_opt < 1/g (for g > 0, q, r > 0).
    ///
    /// Proof: l_opt·g = g²·q / (r + g²·q) < 1  iff  r > 0 ✓
    #[test]
    fn optimal_gain_stability() {
        // g=1, q=1, r=0.1: l_opt = 1/1.1 ≈ 0.909, 1/g = 1.0
        let ilc: NormOptimalIlc<f64, N> = NormOptimalIlc::new(1.0, 1.0, 0.1).unwrap();
        assert!(
            ilc.optimal_gain() < 1.0,
            "l_opt={} should be < 1/g=1.0",
            ilc.optimal_gain()
        );

        // g=2, q=3, r=1: l_opt = 6/13 ≈ 0.461, 1/g = 0.5
        let ilc2: NormOptimalIlc<f64, N> = NormOptimalIlc::new(2.0, 3.0, 1.0).unwrap();
        assert!(
            ilc2.optimal_gain() < 0.5,
            "l_opt={} should be < 0.5",
            ilc2.optimal_gain()
        );
    }

    /// Cost should decrease each trial when simulating a plant with gain=markov_gain.
    #[test]
    fn cost_decreases_each_trial() {
        const G: f64 = 1.0;
        const REF: f64 = 1.0;
        let mut ilc: NormOptimalIlc<f64, N> = NormOptimalIlc::new(G, 1.0, 0.5).unwrap();
        let mut prev_cost = f64::MAX;

        for k in 0..10_usize {
            let u_ff = *ilc.feedforward();
            let mut error = [0.0_f64; N];
            for (n, e) in error.iter_mut().enumerate() {
                *e = REF - G * u_ff[n];
            }
            let cost = ilc.cost(&error);
            if k > 0 {
                assert!(
                    cost <= prev_cost + 1e-12,
                    "cost did not decrease at trial {k}: {cost} > {prev_cost}"
                );
            }
            prev_cost = cost;
            ilc.update(&error).unwrap();
        }
    }

    /// High Q (tracking emphasis) → l_opt approaches 1/g.
    #[test]
    fn high_q_approaches_one_over_g() {
        let g = 1.0_f64;
        // q >> r: l_opt = g*q/(r + g²*q) → 1/g as q→∞
        let ilc: NormOptimalIlc<f64, N> = NormOptimalIlc::new(g, 1e8, 1.0).unwrap();
        let expected = 1.0 / g;
        assert!(
            (ilc.optimal_gain() - expected).abs() < 1e-4,
            "l_opt={} should approach 1/g={expected}",
            ilc.optimal_gain()
        );
    }

    /// High R (control effort emphasis) → l_opt approaches 0.
    #[test]
    fn high_r_approaches_zero() {
        // r >> q: l_opt → 0 as r→∞
        let ilc: NormOptimalIlc<f64, N> = NormOptimalIlc::new(1.0, 1.0, 1e8).unwrap();
        assert!(
            ilc.optimal_gain() < 1e-4,
            "l_opt={} should approach 0",
            ilc.optimal_gain()
        );
    }

    /// Convergence within 10 trials on a step-response plant.
    #[test]
    fn convergence_within_ten_trials() {
        const G: f64 = 1.0;
        const REF: f64 = 1.0;
        let mut ilc: NormOptimalIlc<f64, N> = NormOptimalIlc::new(G, 10.0, 0.1).unwrap();

        for _ in 0..10 {
            let u_ff = *ilc.feedforward();
            let mut error = [0.0_f64; N];
            for (n, e) in error.iter_mut().enumerate() {
                *e = REF - G * u_ff[n];
            }
            ilc.update(&error).unwrap();
        }

        let u_ff = *ilc.feedforward();
        let mut final_err = [0.0_f64; N];
        for (n, e) in final_err.iter_mut().enumerate() {
            *e = REF - G * u_ff[n];
        }
        let mut sq = 0.0_f64;
        for &v in final_err.iter() {
            sq += v * v;
        }
        let err_norm = sq.sqrt();
        assert!(
            err_norm < 0.1 * (N as f64).sqrt(),
            "Error norm {err_norm} too large after 10 trials"
        );
    }

    /// Invalid parameters should return errors.
    #[test]
    fn invalid_params_rejected() {
        assert!(NormOptimalIlc::<f64, N>::new(1.0, 0.0, 1.0).is_err());
        assert!(NormOptimalIlc::<f64, N>::new(1.0, 1.0, 0.0).is_err());
        assert!(NormOptimalIlc::<f64, N>::new(0.0, 1.0, 1.0).is_err());
        assert!(NormOptimalIlc::<f64, N>::new(1.0, -1.0, 1.0).is_err());
        assert!(NormOptimalIlc::<f64, N>::new(f64::NAN, 1.0, 1.0).is_err());
    }

    /// reset() should zero state and reset trial count.
    #[test]
    fn reset_clears_state() {
        let mut ilc: NormOptimalIlc<f64, N> = NormOptimalIlc::new(1.0, 1.0, 1.0).unwrap();
        let err = [0.5_f64; N];
        ilc.update(&err).unwrap();
        assert_eq!(ilc.trial_count(), 1);

        ilc.reset();

        assert_eq!(ilc.trial_count(), 0);
        for &v in ilc.feedforward().iter() {
            assert_eq!(v, 0.0);
        }
    }
}
