#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
use crate::core::matrix::{matmul, matvec, outer, Matrix};
use crate::core::scalar::ControlScalar;

/// Rao-Blackwellized (Marginalized) Particle Filter.
///
/// Exploits a **mixed linear/nonlinear structure** in the state space to achieve
/// a significant variance reduction over a pure particle filter of the same
/// computational cost.
///
/// **State partition**:
/// ```text
///   x = [x_nl (NL-dim)]    ← nonlinear part, tracked by particles
///       [x_lin (NX-dim)]   ← linear part, marginalized analytically via KF
/// ```
///
/// Each particle carries its own Kalman filter for the linear substate
/// conditioned on the nonlinear particle trajectory.
///
/// **Algorithm** (Schön et al., 2005):
/// 1. **Predict** each particle through the nonlinear dynamics.
/// 2. **Predict** each particle's KF for the linear substate.
/// 3. **Weight** each particle by the likelihood of the observation given its
///    nonlinear state (Gaussian, evaluated from the KF predictive distribution).
/// 4. **Resample** using systematic resampling when the Effective Sample Size
///    drops below a threshold.
///
/// Deterministic weight computation — no `rand` crate.  Systematic resampling
/// uses a deterministic LCG to generate a single uniform start point.
///
/// # Type Parameters
/// * `S`  – scalar type (`f32` or `f64`)
/// * `NL` – nonlinear state dimension (particle state)
/// * `NX` – linear state dimension (KF state per particle)
/// * `NM` – measurement dimension
/// * `P`  – number of particles (must be ≥ 2)
#[derive(Debug, Clone)]
pub struct MarginalizedParticleFilter<
    S: ControlScalar,
    const NL: usize,
    const NX: usize,
    const NM: usize,
    const P: usize,
> {
    /// Nonlinear dynamics: f_nl(x_nl, u) -> x_nl_next.
    f_nl: fn(&[S; NL], &[S; 1]) -> [S; NL],
    /// Linear state transition (NX×NX) — may depend on x_nl conceptually,
    /// but is fixed here for the marginalization to remain tractable.
    pub a_lin: Matrix<S, NX, NX>,
    /// Control input matrix for linear substate (NX×1).
    pub b_lin: Matrix<S, NX, 1>,
    /// Measurement matrix for linear substate (NM×NX).
    pub h_lin: Matrix<S, NM, NX>,
    /// Process noise covariance for linear substate (NX×NX).
    pub q_lin: Matrix<S, NX, NX>,
    /// Measurement noise covariance (NM×NM).
    pub r: Matrix<S, NM, NM>,
    /// Nonlinear particle states.
    x_nl: [[S; NL]; P],
    /// Per-particle KF state estimate for linear substate.
    x_lin: [[S; NX]; P],
    /// Per-particle KF covariance for linear substate.
    p_lin: [Matrix<S, NX, NX>; P],
    /// Normalized particle weights.
    weights: [S; P],
    /// Effective sample size threshold for resampling (fraction of P).
    pub ess_threshold: S,
    /// LCG state for deterministic resampling perturbation.
    lcg_state: u64,
}

/// Error type for the Marginalized Particle Filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpfError {
    /// Particle count is too small (need P ≥ 2).
    TooFewParticles,
    /// Innovation covariance is singular for all particles.
    SingularInnovationCovariance,
    /// All particle weights collapsed to zero (weight degeneracy).
    WeightDegeneracy,
}

impl core::fmt::Display for MpfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MpfError::TooFewParticles => write!(f, "MPF: need P ≥ 2 particles"),
            MpfError::SingularInnovationCovariance => {
                write!(f, "MPF: innovation covariance singular")
            }
            MpfError::WeightDegeneracy => write!(f, "MPF: all particle weights are zero"),
        }
    }
}

/// LCG step — Knuth multiplier.
#[inline]
fn lcg_step(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Map u64 to uniform [0, 1).
#[inline]
fn lcg_uniform<S: ControlScalar>(state: &mut u64) -> S {
    let raw = lcg_step(state);
    S::from_f64((raw >> 11) as f64 / (1u64 << 53) as f64)
}

impl<S: ControlScalar, const NL: usize, const NX: usize, const NM: usize, const P: usize>
    MarginalizedParticleFilter<S, NL, NX, NM, P>
{
    /// Create a Marginalized Particle Filter.
    ///
    /// All particles are initialized at `x_nl0` with linear substate `x_lin0`
    /// and covariance `p_lin0`.
    ///
    /// Returns `Err` if `P < 2`.
    pub fn new(
        f_nl: fn(&[S; NL], &[S; 1]) -> [S; NL],
        a_lin: Matrix<S, NX, NX>,
        b_lin: Matrix<S, NX, 1>,
        h_lin: Matrix<S, NM, NX>,
        q_lin: Matrix<S, NX, NX>,
        r: Matrix<S, NM, NM>,
        x_nl0: [S; NL],
        x_lin0: [S; NX],
        p_lin0: Matrix<S, NX, NX>,
        seed: u64,
    ) -> Result<Self, MpfError> {
        if P < 2 {
            return Err(MpfError::TooFewParticles);
        }

        let uniform_weight = S::ONE / S::from_f64(P as f64);
        Ok(Self {
            f_nl,
            a_lin,
            b_lin,
            h_lin,
            q_lin,
            r,
            x_nl: [x_nl0; P],
            x_lin: [x_lin0; P],
            p_lin: [p_lin0; P],
            weights: [uniform_weight; P],
            ess_threshold: S::from_f64(0.5),
            lcg_state: seed ^ 0xCAFE_BABE_1234_5678,
        })
    }

    /// **Predict step**.
    ///
    /// 1. Propagate nonlinear particle state through `f_nl`.
    /// 2. Propagate each particle's KF: `x_lin ← A·x_lin + B·u`, `P ← A·P·Aᵀ + Q`.
    pub fn predict(&mut self, u: S) {
        let u_arr = [u];
        for p_idx in 0..P {
            // Nonlinear prediction
            self.x_nl[p_idx] = (self.f_nl)(&self.x_nl[p_idx], &u_arr);

            // Linear KF predict
            let ax = matvec(&self.a_lin, &self.x_lin[p_idx]);
            let bu = matvec(&self.b_lin, &u_arr);
            self.x_lin[p_idx] = core::array::from_fn(|i| ax[i] + bu[i]);

            let ap = matmul(&self.a_lin, &self.p_lin[p_idx]);
            let at = self.a_lin.transpose();
            let apat = matmul(&ap, &at);
            self.p_lin[p_idx] = apat.add_mat(&self.q_lin);
        }
    }

    /// **Update step** — weight + KF update + optional resample.
    ///
    /// For each particle:
    /// 1. Compute predicted measurement and innovation covariance from its KF.
    /// 2. Evaluate Gaussian log-likelihood of the observation.
    /// 3. Perform the KF update for the linear substate.
    /// 4. Multiply weight by likelihood.
    ///
    /// Weights are then normalized.  If ESS < threshold · P, systematic
    /// resampling is triggered.
    ///
    /// Returns the weighted-mean innovation, or `Err` on numerical failure.
    pub fn update(&mut self, z: &[S; NM]) -> Result<[S; NM], MpfError> {
        let mut log_weights = [S::ZERO; P];
        let mut any_valid = false;

        for p_idx in 0..P {
            // Predicted measurement: ẑ = H · x_lin
            let z_pred = matvec(&self.h_lin, &self.x_lin[p_idx]);

            // Innovation
            let innovation: [S; NM] = core::array::from_fn(|j| z[j] - z_pred[j]);

            // Innovation covariance: S = H·P·Hᵀ + R
            let hp = matmul(&self.h_lin, &self.p_lin[p_idx]);
            let ht = self.h_lin.transpose();
            let hpht = matmul(&hp, &ht);
            let s_mat = hpht.add_mat(&self.r);

            let s_inv = match s_mat.inv() {
                Some(inv) => inv,
                None => {
                    // Degenerate particle: set very low weight
                    log_weights[p_idx] = S::from_f64(-1e30_f64);
                    continue;
                }
            };

            // Log Gaussian likelihood: -0.5 · (νᵀ·S⁻¹·ν + log|2π·S|)
            // We use a simplified version: -0.5 · νᵀ·S⁻¹·ν (ignore det term,
            // which is particle-independent when R and H are shared)
            let sinv_nu = matvec(&s_inv, &innovation);
            let maha: S = innovation
                .iter()
                .zip(sinv_nu.iter())
                .map(|(&a, &b)| a * b)
                .fold(S::ZERO, |acc, x| acc + x);
            log_weights[p_idx] = S::from_f64(-0.5_f64) * maha;
            any_valid = true;

            // KF update for linear substate
            let pht = matmul(&self.p_lin[p_idx], &ht);
            let k = matmul(&pht, &s_inv);
            let k_innov = matvec(&k, &innovation);
            for i in 0..NX {
                self.x_lin[p_idx][i] += k_innov[i];
            }

            // Covariance: P ← (I - K·H)·P
            let kh = matmul(&k, &self.h_lin);
            let eye = Matrix::<S, NX, NX>::identity();
            let i_kh = eye.sub_mat(&kh);
            self.p_lin[p_idx] = matmul(&i_kh, &self.p_lin[p_idx]);
        }

        if !any_valid {
            return Err(MpfError::SingularInnovationCovariance);
        }

        // Numerically stable weight normalization via log-sum-exp
        let log_max = log_weights
            .iter()
            .copied()
            .fold(
                S::from_f64(f64::NEG_INFINITY),
                |a, b| if b > a { b } else { a },
            );

        let mut sum_exp = S::ZERO;
        for &lw in &log_weights {
            sum_exp += (lw - log_max).exp();
        }

        if sum_exp <= S::ZERO || !sum_exp.is_finite() {
            return Err(MpfError::WeightDegeneracy);
        }

        for p_idx in 0..P {
            self.weights[p_idx] *= (log_weights[p_idx] - log_max).exp() / sum_exp;
        }

        // Re-normalize weights to sum to 1
        let w_sum: S = self.weights.iter().copied().fold(S::ZERO, |a, b| a + b);
        if w_sum <= S::ZERO {
            return Err(MpfError::WeightDegeneracy);
        }
        let inv_w_sum = S::ONE / w_sum;
        for w in self.weights.iter_mut() {
            *w *= inv_w_sum;
        }

        // Compute weighted-mean innovation for diagnostics
        let x_nl_mean = self.state_nl();
        let x_lin_mean = self.state_lin();
        let z_pred_mean = matvec(&self.h_lin, &x_lin_mean);
        let _ = x_nl_mean; // used for future extensions
        let mean_innov: [S; NM] = core::array::from_fn(|j| z[j] - z_pred_mean[j]);

        // Adaptive resampling: check ESS
        let ess = self.effective_sample_size();
        if ess < self.ess_threshold * S::from_f64(P as f64) {
            self.systematic_resample();
        }

        Ok(mean_innov)
    }

    /// Effective Sample Size: `N_eff = 1 / Σ w_i²`.
    pub fn effective_sample_size(&self) -> S {
        let sum_sq: S = self
            .weights
            .iter()
            .map(|&w| w * w)
            .fold(S::ZERO, |a, b| a + b);
        if sum_sq <= S::ZERO {
            S::ZERO
        } else {
            S::ONE / sum_sq
        }
    }

    /// Weighted mean of the nonlinear particle states.
    pub fn state_nl(&self) -> [S; NL] {
        let mut mean = [S::ZERO; NL];
        for p_idx in 0..P {
            let w = self.weights[p_idx];
            for i in 0..NL {
                mean[i] += w * self.x_nl[p_idx][i];
            }
        }
        mean
    }

    /// Weighted mean of the linear substates.
    pub fn state_lin(&self) -> [S; NX] {
        let mut mean = [S::ZERO; NX];
        for p_idx in 0..P {
            let w = self.weights[p_idx];
            for i in 0..NX {
                mean[i] += w * self.x_lin[p_idx][i];
            }
        }
        mean
    }

    /// Weighted covariance of the linear substate.
    ///
    /// Includes both the within-particle covariance (KF `P`) and the
    /// between-particle variance.
    pub fn covariance_lin(&self) -> Matrix<S, NX, NX> {
        let x_mean = self.state_lin();
        let mut cov = Matrix::<S, NX, NX>::zeros();

        for p_idx in 0..P {
            let w = self.weights[p_idx];
            // Within-particle term: w_i · P_i
            let scaled_p = self.p_lin[p_idx].scale(w);
            cov = cov.add_mat(&scaled_p);

            // Between-particle term: w_i · (x_lin_i - x̄)(x_lin_i - x̄)ᵀ
            let dx: [S; NX] = core::array::from_fn(|i| self.x_lin[p_idx][i] - x_mean[i]);
            let dx_dxt = outer(&dx, &dx).scale(w);
            cov = cov.add_mat(&dx_dxt);
        }
        cov
    }

    /// Raw particle weights.
    pub fn weights(&self) -> &[S; P] {
        &self.weights
    }

    /// Reset all particles to the given state with uniform weights.
    pub fn reset(&mut self, x_nl0: [S; NL], x_lin0: [S; NX], p_lin0: Matrix<S, NX, NX>) {
        let uniform = S::ONE / S::from_f64(P as f64);
        for p_idx in 0..P {
            self.x_nl[p_idx] = x_nl0;
            self.x_lin[p_idx] = x_lin0;
            self.p_lin[p_idx] = p_lin0;
            self.weights[p_idx] = uniform;
        }
    }

    /// **Systematic resampling** (Kitagawa, 1996).
    ///
    /// Generates a single uniform start point via LCG and draws `P` samples
    /// with replacement proportional to the cumulative weight.  After
    /// resampling all weights are reset to `1/P`.
    fn systematic_resample(&mut self) {
        // Build cumulative weight vector
        let mut cumulative = [S::ZERO; P];
        cumulative[0] = self.weights[0];
        for i in 1..P {
            cumulative[i] = cumulative[i - 1] + self.weights[i];
        }

        // Single uniform start point in [0, 1/P)
        let inv_p = S::ONE / S::from_f64(P as f64);
        let u0: S = lcg_uniform::<S>(&mut self.lcg_state) * inv_p;

        // Determine ancestor indices
        let mut ancestors = [0usize; P];
        let mut j = 0usize;
        for k in 0..P {
            let target = u0 + S::from_f64(k as f64) * inv_p;
            while j < P - 1 && cumulative[j] < target {
                j += 1;
            }
            ancestors[k] = j;
        }

        // Copy resampled particles into temporary storage
        let x_nl_copy = self.x_nl;
        let x_lin_copy = self.x_lin;
        let p_lin_copy = self.p_lin;

        for k in 0..P {
            let src = ancestors[k];
            self.x_nl[k] = x_nl_copy[src];
            self.x_lin[k] = x_lin_copy[src];
            self.p_lin[k] = p_lin_copy[src];
            self.weights[k] = inv_p;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Nonlinear part: scalar angle θ, identity dynamics
    fn f_nl_id(x: &[f64; 1], _u: &[f64; 1]) -> [f64; 1] {
        *x
    }

    fn build_mpf() -> MarginalizedParticleFilter<f64, 1, 2, 1, 16> {
        // Linear substate: position-velocity (2D)
        let dt = 0.01_f64;
        let mut a_lin = Matrix::<f64, 2, 2>::identity();
        a_lin.data[0][1] = dt;

        let b_lin = Matrix::<f64, 2, 1>::zeros();

        let mut h_lin = Matrix::<f64, 1, 2>::zeros();
        h_lin.data[0][0] = 1.0;

        let q_lin = Matrix::<f64, 2, 2>::identity().scale(1e-4);
        let r = Matrix::<f64, 1, 1>::identity().scale(0.1);
        let p_lin0 = Matrix::<f64, 2, 2>::identity().scale(10.0);

        MarginalizedParticleFilter::new(
            f_nl_id,
            a_lin,
            b_lin,
            h_lin,
            q_lin,
            r,
            [0.0_f64],
            [0.0_f64; 2],
            p_lin0,
            42,
        )
        .expect("valid MPF")
    }

    #[test]
    fn new_creates_filter() {
        let _mpf = build_mpf();
    }

    #[test]
    fn predict_runs() {
        let mut mpf = build_mpf();
        mpf.predict(0.0);
    }

    #[test]
    fn update_returns_innovation() {
        let mut mpf = build_mpf();
        mpf.predict(0.0);
        let result = mpf.update(&[1.0]);
        assert!(result.is_ok(), "Update failed: {:?}", result);
    }

    #[test]
    fn tracks_constant_position() {
        let mut mpf = build_mpf();
        let true_pos = 4.0_f64;
        for _ in 0..200 {
            mpf.predict(0.0);
            mpf.update(&[true_pos]).expect("update");
        }
        let x_lin = mpf.state_lin();
        assert!(
            (x_lin[0] - true_pos).abs() < 1.0,
            "Expected ~{true_pos}, got {}",
            x_lin[0]
        );
    }

    #[test]
    fn weights_sum_to_one() {
        let mut mpf = build_mpf();
        mpf.predict(0.0);
        mpf.update(&[1.0]).expect("update");
        let sum: f64 = mpf.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Weights sum = {sum}");
    }

    #[test]
    fn effective_sample_size_bounds() {
        let mpf = build_mpf();
        let ess = mpf.effective_sample_size();
        // ESS ∈ [1, P] for normalized weights
        assert!(ess >= 1.0 - 1e-9, "ESS too low: {ess}");
        assert!(ess <= 16.0 + 1e-9, "ESS too high: {ess}");
    }

    #[test]
    fn covariance_lin_is_symmetric() {
        let mut mpf = build_mpf();
        for _ in 0..10 {
            mpf.predict(0.0);
            mpf.update(&[2.0]).expect("update");
        }
        let cov = mpf.covariance_lin();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (cov.data[i][j] - cov.data[j][i]).abs() < 1e-10,
                    "Cov not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn reset_restores_uniform_weights() {
        let mut mpf = build_mpf();
        for _ in 0..30 {
            mpf.predict(0.0);
            mpf.update(&[3.0]).expect("update");
        }
        let p_lin0 = Matrix::<f64, 2, 2>::identity().scale(10.0);
        mpf.reset([0.0_f64], [0.0_f64; 2], p_lin0);
        let sum: f64 = mpf.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Weights after reset sum = {sum}");
        let expected_w = 1.0 / 16.0;
        for &w in mpf.weights() {
            assert!(
                (w - expected_w).abs() < 1e-12,
                "Non-uniform weight after reset: {w}"
            );
        }
    }
}
