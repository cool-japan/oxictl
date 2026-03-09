//! Model Predictive Path Integral (MPPI) controller.
//!
//! Sampling-based stochastic optimal control using importance-weighted
//! trajectory rollouts. Suitable for nonlinear systems with complex cost
//! functions including obstacle avoidance, contact dynamics, etc.
//!
//! Algorithm (Williams et al. 2016):
//! 1. Generate K noisy control trajectories: v^i = u + ε^i where ε ~ N(0, Σ)
//! 2. Roll out each trajectory through the dynamics for H steps
//! 3. Compute total cost S^i for each trajectory
//! 4. Compute importance sampling weights: w^i = exp(-(S^i - S_min)/λ)
//! 5. Update optimal control: u* = Σ w^i · v^i / Σ w^i  (weighted average)
//! 6. Apply u*[0], shift horizon (receding horizon / warm start)
//!
//! Implementation notes:
//! - No heap allocation: all arrays are stack-allocated via const generics.
//! - No `rand` crate: deterministic LCG + Box-Muller for Gaussian noise.
//! - `no_std` compatible: uses `core::` and `libm` throughout.
//! - Two-pass approach for numerical stability: collect all K costs first,
//!   then shift by S_min before computing exp.
//! - Control cost term λ · u^T · Σ^{-1} · ε follows the IS formulation.
#![allow(
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::type_complexity
)]
use crate::core::scalar::ControlScalar;
use crate::mpc::stochastic_mpc::Lcg;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during MPPI operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MppiError {
    /// Temperature λ must be strictly positive.
    InvalidTemperature,
    /// At least one sample (K ≥ 1) is required.
    ZeroSamples,
    /// The nominal trajectory provided for warm-starting has wrong dimensions.
    DimensionMismatch,
}

impl core::fmt::Display for MppiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MppiError::InvalidTemperature => {
                write!(f, "MPPI: temperature λ must be strictly positive")
            }
            MppiError::ZeroSamples => write!(f, "MPPI: K must be ≥ 1"),
            MppiError::DimensionMismatch => {
                write!(f, "MPPI: warm-start trajectory dimension mismatch")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MppiConfig
// ---------------------------------------------------------------------------

/// Configuration for the MPPI controller.
///
/// Type parameters:
/// - `S`: scalar type (`f32` or `f64`).
/// - `N`: state dimension.
/// - `I`: input (control) dimension.
#[derive(Clone, Copy, Debug)]
pub struct MppiConfig<S: ControlScalar, const N: usize, const I: usize> {
    /// Temperature λ — lower values make the update more greedy (exploit best
    /// trajectory), higher values blend trajectories more uniformly.
    pub temperature: S,

    /// Per-input noise standard deviation σ_j for each control channel j.
    /// Perturbations are drawn as ε_j ~ N(0, σ_j²) independently.
    pub sigma: [S; I],

    /// Lower bound on each control input (element-wise).
    pub u_min: [S; I],

    /// Upper bound on each control input (element-wise).
    pub u_max: [S; I],

    /// Discount factor γ applied to stage costs (typically 1.0 for undiscounted).
    pub gamma: S,

    /// Seed for the internal LCG pseudo-random number generator.
    pub lcg_seed: u64,

    // Phantom to carry N (state dim is not used in config directly but is part
    // of the generic signature so the user can write MppiConfig<f64, 4, 2>).
    _phantom: core::marker::PhantomData<[S; N]>,
}

impl<S: ControlScalar, const N: usize, const I: usize> MppiConfig<S, N, I> {
    /// Create a new `MppiConfig`.
    ///
    /// # Errors
    /// Returns [`MppiError::InvalidTemperature`] if `temperature ≤ 0`.
    pub fn new(
        temperature: S,
        sigma: [S; I],
        u_min: [S; I],
        u_max: [S; I],
        gamma: S,
        lcg_seed: u64,
    ) -> Result<Self, MppiError> {
        if temperature <= S::ZERO {
            return Err(MppiError::InvalidTemperature);
        }
        Ok(Self {
            temperature,
            sigma,
            u_min,
            u_max,
            gamma,
            lcg_seed,
            _phantom: core::marker::PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// MppiStats
// ---------------------------------------------------------------------------

/// Diagnostic statistics computed over a set of K trajectory costs.
#[derive(Clone, Copy, Debug)]
pub struct MppiStats<S: ControlScalar> {
    /// Number of samples used in this update step.
    pub n_samples: usize,
    /// Minimum trajectory cost encountered.
    pub min_cost: S,
    /// Maximum trajectory cost encountered.
    pub max_cost: S,
    /// Mean trajectory cost (unweighted sample average).
    pub mean_cost: S,
    /// Effective Sample Size: (Σ w_i)² / Σ w_i²
    /// Measures how many samples contribute effectively (1 ≤ ESS ≤ K).
    pub effective_sample_size: S,
}

// ---------------------------------------------------------------------------
// Mppi — main controller
// ---------------------------------------------------------------------------

/// Model Predictive Path Integral (MPPI) controller.
///
/// Type parameters:
/// - `S`  : scalar type (`f32` or `f64`).
/// - `N`  : state dimension.
/// - `I`  : input (control) dimension.
/// - `H`  : prediction horizon (number of steps).
/// - `K`  : number of Monte Carlo samples per update call.
pub struct Mppi<S: ControlScalar, const N: usize, const I: usize, const H: usize, const K: usize> {
    /// Controller configuration.
    config: MppiConfig<S, N, I>,

    /// Current nominal (warm-started) control sequence u[0..H][0..I].
    /// Updated at the end of each `update` call via the MPPI weighted mean.
    u_nominal: [[S; I]; H],

    /// LCG state — advances deterministically across `update` calls for
    /// reproducibility while still varying each call.
    lcg_state: u64,

    /// Perturbed trajectories for all K samples: v_samples[k][h][i].
    /// Stored as a flat array [K × H × I] using a 3-level nested array.
    /// K=100, H=30, I=4 → 12 000 floats × 8 bytes = 96 KiB on stack;
    /// acceptable for embedded targets; reduce K/H if stack is limited.
    v_samples: [[[S; I]; H]; K],

    /// Total trajectory costs S^k for k = 0..K (two-pass stability).
    sample_costs: [S; K],
}

impl<S: ControlScalar, const N: usize, const I: usize, const H: usize, const K: usize>
    Mppi<S, N, I, H, K>
{
    // ------------------------------------------------------------------
    // Construction / initialisation
    // ------------------------------------------------------------------

    /// Create a new MPPI controller.
    ///
    /// The nominal trajectory is initialised to all-zeros.
    /// The LCG is seeded from `config.lcg_seed`.
    ///
    /// # Errors
    /// Returns [`MppiError::ZeroSamples`] if `K == 0`.
    pub fn new(config: MppiConfig<S, N, I>) -> Result<Self, MppiError> {
        if K == 0 {
            return Err(MppiError::ZeroSamples);
        }
        let lcg_seed = config.lcg_seed;
        Ok(Self {
            config,
            u_nominal: [[S::ZERO; I]; H],
            lcg_state: lcg_seed,
            v_samples: [[[S::ZERO; I]; H]; K],
            sample_costs: [S::ZERO; K],
        })
    }

    /// Reset the nominal trajectory to all-zeros and re-seed the LCG.
    pub fn reset(&mut self) {
        self.u_nominal = [[S::ZERO; I]; H];
        self.lcg_state = self.config.lcg_seed;
    }

    /// Warm-start the nominal trajectory from an externally provided sequence.
    ///
    /// This is useful when a good prior solution is known (e.g., from a
    /// previous planning step or a reference trajectory).
    pub fn set_nominal(&mut self, u: &[[S; I]; H]) {
        self.u_nominal = *u;
    }

    /// Read-only access to the current nominal trajectory.
    pub fn nominal(&self) -> &[[S; I]; H] {
        &self.u_nominal
    }

    // ------------------------------------------------------------------
    // Internal LCG helpers
    // ------------------------------------------------------------------

    /// Advance the internal LCG and return the next uniform sample in [0, 1).
    fn next_uniform(&mut self) -> f64 {
        // NR parameters (same as Lcg::new in stochastic_mpc.rs)
        const A: u64 = 1_664_525;
        const C: u64 = 1_013_904_223;
        const M: u64 = 1u64 << 32;
        self.lcg_state = A.wrapping_mul(self.lcg_state).wrapping_add(C) & (M - 1);
        self.lcg_state as f64 / M as f64
    }

    /// Box-Muller transform: two uniform samples → one standard-normal sample.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform().max(1e-15_f64); // guard against log(0)
        let u2 = self.next_uniform();
        let two_pi = 2.0_f64 * core::f64::consts::PI;
        libm::sqrt(-2.0_f64 * libm::log(u1)) * libm::cos(two_pi * u2)
    }

    // ------------------------------------------------------------------
    // Perturbation generation
    // ------------------------------------------------------------------

    /// Generate a single perturbation vector for horizon step `h` into `eps`.
    ///
    /// Each control channel `j` is drawn independently from N(0, σ_j²).
    fn sample_perturbation(&mut self, eps: &mut [S; I]) {
        for j in 0..I {
            let z = self.next_normal();
            let sigma_j = self.config.sigma[j];
            eps[j] = sigma_j * S::from_f64(z);
        }
    }

    // ------------------------------------------------------------------
    // Control cost term
    // ------------------------------------------------------------------

    /// Compute the IS control cost addition for one (h, k) step:
    ///   λ · u_h^T · Σ^{-1} · ε_h^k
    ///
    /// With diagonal Σ = diag(σ_j²), Σ^{-1} = diag(1/σ_j²), so:
    ///   cost = λ · Σ_j  u_j · ε_j / σ_j²
    fn control_cost_term(&self, u_h: &[S; I], eps_h: &[S; I]) -> S {
        let lambda = self.config.temperature;
        let mut acc = S::ZERO;
        for j in 0..I {
            let sigma_j = self.config.sigma[j];
            // Avoid division by zero: if sigma is zero, the channel has no
            // noise and contributes no IS cost.
            if sigma_j > S::ZERO {
                acc += u_h[j] * eps_h[j] / (sigma_j * sigma_j);
            }
        }
        lambda * acc
    }

    // ------------------------------------------------------------------
    // Core MPPI update
    // ------------------------------------------------------------------

    /// Run one MPPI update step.
    ///
    /// # Arguments
    /// - `x0`      : current system state (N-vector).
    /// - `dynamics`: one-step discrete dynamics f(x, u) → x_next.
    /// - `cost`    : stage / terminal cost g(x, u, is_terminal) → scalar.
    ///
    /// # Returns
    /// The first element of the updated optimal control sequence u*[0],
    /// or [`MppiError`] if the controller is misconfigured.
    ///
    /// # Side effects
    /// - Updates `self.u_nominal` (warm-start for next call).
    /// - Advances the internal LCG state.
    pub fn update<F, G>(&mut self, x0: [S; N], dynamics: F, cost: G) -> Result<[S; I], MppiError>
    where
        F: Fn(&[S; N], &[S; I]) -> [S; N],
        G: Fn(&[S; N], &[S; I], bool) -> S,
    {
        // --- Pass 1: generate K perturbed trajectories and compute costs ---
        for k in 0..K {
            let mut x = x0;
            let mut total_cost = S::ZERO;
            let mut discount = S::ONE;

            for h in 0..H {
                // Sample perturbation ε^k_h ~ N(0, Σ)
                let mut eps = [S::ZERO; I];
                self.sample_perturbation(&mut eps);

                // Perturbed control: v^k_h = clamp(u_h + ε^k_h, u_min, u_max)
                let mut v_h = [S::ZERO; I];
                for j in 0..I {
                    let raw = self.u_nominal[h][j] + eps[j];
                    v_h[j] = raw.clamp_val(self.config.u_min[j], self.config.u_max[j]);
                }
                self.v_samples[k][h] = v_h;

                // Stage cost (with discount)
                let is_terminal = h == H - 1;
                let stage = discount * cost(&x, &v_h, is_terminal);

                // IS control cost term: λ · u^T Σ^{-1} ε
                let ctrl_cost = discount * self.control_cost_term(&self.u_nominal[h], &eps);

                total_cost += stage + ctrl_cost;
                discount *= self.config.gamma;

                // Roll out dynamics (skip at terminal step — no next state needed)
                if !is_terminal {
                    x = dynamics(&x, &v_h);
                }
            }

            self.sample_costs[k] = total_cost;
        }

        // --- Pass 2: compute importance weights (shifted for stability) ---
        let lambda = self.config.temperature;

        // Find minimum cost for numerical stability (prevent exp overflow)
        let mut s_min = self.sample_costs[0];
        for k in 1..K {
            if self.sample_costs[k] < s_min {
                s_min = self.sample_costs[k];
            }
        }

        // Compute unnormalised log-weights: -( S^k - S_min ) / λ
        // then exponentiate: w^k = exp(-(S^k - S_min)/λ)
        let mut weights = [S::ZERO; K];
        let mut weight_sum = S::ZERO;
        for k in 0..K {
            let shifted = self.sample_costs[k] - s_min;
            // exp(-shifted / λ); use libm via num_traits Float::exp
            let log_w = -(shifted / lambda);
            let w = log_w.exp();
            weights[k] = w;
            weight_sum += w;
        }

        // Normalise weights
        let inv_weight_sum = if weight_sum > S::ZERO {
            S::ONE / weight_sum
        } else {
            // Degenerate case: all costs identical — uniform weights
            S::ONE / S::from_f64(K as f64)
        };

        // --- Compute weighted mean of perturbed trajectories ---
        let mut u_star = [[S::ZERO; I]; H];
        for k in 0..K {
            let w_norm = weights[k] * inv_weight_sum;
            for h in 0..H {
                for j in 0..I {
                    u_star[h][j] += w_norm * self.v_samples[k][h][j];
                }
            }
        }

        // Clamp u_star to control bounds
        for h in 0..H {
            for j in 0..I {
                u_star[h][j] = u_star[h][j].clamp_val(self.config.u_min[j], self.config.u_max[j]);
            }
        }

        // --- Warm-start: shift horizon (receding horizon principle) ---
        // u_nominal[0..H-2] ← u_star[1..H-1], u_nominal[H-1] ← u_star[H-1]
        for h in 0..H - 1 {
            self.u_nominal[h] = u_star[h + 1];
        }
        self.u_nominal[H - 1] = u_star[H - 1];

        Ok(u_star[0])
    }

    // ------------------------------------------------------------------
    // Utility methods
    // ------------------------------------------------------------------

    /// Compute the total cost of the current nominal trajectory (no noise).
    ///
    /// Useful for monitoring convergence or comparing warm-start quality.
    pub fn baseline_cost<F, G>(&self, x0: [S; N], dynamics: F, cost: G) -> S
    where
        F: Fn(&[S; N], &[S; I]) -> [S; N],
        G: Fn(&[S; N], &[S; I], bool) -> S,
    {
        let mut x = x0;
        let mut total = S::ZERO;
        let mut discount = S::ONE;

        for h in 0..H {
            let is_terminal = h == H - 1;
            total += discount * cost(&x, &self.u_nominal[h], is_terminal);
            discount *= self.config.gamma;
            if !is_terminal {
                x = dynamics(&x, &self.u_nominal[h]);
            }
        }
        total
    }

    /// Compute diagnostic statistics over the most recently computed K costs.
    ///
    /// Call this after `update` to inspect sampling quality.
    pub fn compute_stats(&self) -> MppiStats<S> {
        let lambda = self.config.temperature;

        // Basic statistics
        let mut s_min = self.sample_costs[0];
        let mut s_max = self.sample_costs[0];
        let mut s_sum = S::ZERO;

        for k in 0..K {
            let s = self.sample_costs[k];
            if s < s_min {
                s_min = s;
            }
            if s > s_max {
                s_max = s;
            }
            s_sum += s;
        }
        let mean_cost = s_sum / S::from_f64(K as f64);

        // Effective sample size: ESS = (Σ w_i)² / Σ w_i²
        // (uses shifted weights for numerical stability)
        let mut w_sum = S::ZERO;
        let mut w_sq_sum = S::ZERO;
        for k in 0..K {
            let shifted = self.sample_costs[k] - s_min;
            let w = (-(shifted / lambda)).exp();
            w_sum += w;
            w_sq_sum += w * w;
        }
        let ess = if w_sq_sum > S::ZERO {
            (w_sum * w_sum) / w_sq_sum
        } else {
            S::from_f64(K as f64)
        };

        MppiStats {
            n_samples: K,
            min_cost: s_min,
            max_cost: s_max,
            mean_cost,
            effective_sample_size: ess,
        }
    }

    /// Access the last computed sample costs (for external analysis).
    pub fn sample_costs(&self) -> &[S; K] {
        &self.sample_costs
    }

    /// Access the MPPI configuration.
    pub fn config(&self) -> &MppiConfig<S, N, I> {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Builder helper
// ---------------------------------------------------------------------------

/// Convenience builder for [`MppiConfig`] with sensible defaults.
pub struct MppiConfigBuilder<S: ControlScalar, const N: usize, const I: usize> {
    temperature: S,
    sigma: [S; I],
    u_min: [S; I],
    u_max: [S; I],
    gamma: S,
    lcg_seed: u64,
}

impl<S: ControlScalar, const N: usize, const I: usize> MppiConfigBuilder<S, N, I> {
    /// Start building a config.
    /// Defaults: temperature=1.0, sigma=0.1 per channel, bounds ±1, gamma=1.0, seed=42.
    pub fn new() -> Self {
        let sigma_val = S::from_f64(0.1);
        let u_min_val = S::from_f64(-1.0);
        let u_max_val = S::ONE;
        Self {
            temperature: S::ONE,
            sigma: [sigma_val; I],
            u_min: [u_min_val; I],
            u_max: [u_max_val; I],
            gamma: S::ONE,
            lcg_seed: 42,
        }
    }

    /// Set temperature λ.
    pub fn temperature(mut self, t: S) -> Self {
        self.temperature = t;
        self
    }

    /// Set per-channel noise standard deviation (same for all channels).
    pub fn sigma_uniform(mut self, s: S) -> Self {
        self.sigma = [s; I];
        self
    }

    /// Set per-channel noise standard deviation individually.
    pub fn sigma(mut self, s: [S; I]) -> Self {
        self.sigma = s;
        self
    }

    /// Set symmetric control bounds ±bound for all channels.
    pub fn bounds_symmetric(mut self, bound: S) -> Self {
        self.u_min = [-bound; I];
        self.u_max = [bound; I];
        self
    }

    /// Set individual lower and upper bounds.
    pub fn bounds(mut self, u_min: [S; I], u_max: [S; I]) -> Self {
        self.u_min = u_min;
        self.u_max = u_max;
        self
    }

    /// Set cost discount factor γ.
    pub fn gamma(mut self, g: S) -> Self {
        self.gamma = g;
        self
    }

    /// Set LCG seed for reproducibility.
    pub fn lcg_seed(mut self, seed: u64) -> Self {
        self.lcg_seed = seed;
        self
    }

    /// Finalise the configuration.
    ///
    /// # Errors
    /// Returns [`MppiError::InvalidTemperature`] if temperature ≤ 0.
    pub fn build(self) -> Result<MppiConfig<S, N, I>, MppiError> {
        MppiConfig::new(
            self.temperature,
            self.sigma,
            self.u_min,
            self.u_max,
            self.gamma,
            self.lcg_seed,
        )
    }
}

impl<S: ControlScalar, const N: usize, const I: usize> Default for MppiConfigBuilder<S, N, I> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Standalone Box-Muller helper using an external Lcg
// (exposed for users who manage their own LCG, mirrors stochastic_mpc pattern)
// ---------------------------------------------------------------------------

/// Generate a standard-normal sample using Box-Muller transform from an [`Lcg`].
///
/// Convenience wrapper exposed for reuse outside the controller struct.
pub fn box_muller_normal(lcg: &mut Lcg) -> f64 {
    lcg.next_normal_f64()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper: build a default f64 MPPI for an integrator (N=2, I=1)
    // ------------------------------------------------------------------

    /// Double integrator: x_{t+1} = A x + B u
    /// x = [position, velocity], u = acceleration
    /// dt = 0.1 s
    fn integrator_dynamics(x: &[f64; 2], u: &[f64; 1]) -> [f64; 2] {
        let dt = 0.1_f64;
        [x[0] + dt * x[1] + 0.5 * dt * dt * u[0], x[1] + dt * u[0]]
    }

    /// Quadratic tracking cost toward origin: x^T x + 0.01 u^T u
    fn quadratic_cost(x: &[f64; 2], u: &[f64; 1], _terminal: bool) -> f64 {
        x[0] * x[0] + x[1] * x[1] + 0.01 * u[0] * u[0]
    }

    /// Zero cost — all trajectories equally good.
    fn zero_cost(_x: &[f64; 2], _u: &[f64; 1], _terminal: bool) -> f64 {
        0.0_f64
    }

    type Mppi2_1_10_50 = Mppi<f64, 2, 1, 10, 50>;

    fn make_mppi() -> Mppi2_1_10_50 {
        let config = MppiConfigBuilder::<f64, 2, 1>::new()
            .temperature(1.0)
            .sigma_uniform(0.5)
            .bounds_symmetric(5.0)
            .gamma(1.0)
            .lcg_seed(12345)
            .build()
            .expect("valid config");
        Mppi::new(config).expect("valid mppi")
    }

    // ------------------------------------------------------------------
    // Configuration tests
    // ------------------------------------------------------------------

    #[test]
    fn invalid_temperature_returns_error() {
        let result = MppiConfig::<f64, 2, 1>::new(-1.0, [0.1], [-5.0], [5.0], 1.0, 42);
        assert!(
            matches!(result, Err(MppiError::InvalidTemperature)),
            "expected InvalidTemperature"
        );
    }

    #[test]
    fn zero_temperature_returns_error() {
        let result = MppiConfig::<f64, 2, 1>::new(0.0, [0.1], [-5.0], [5.0], 1.0, 42);
        assert!(matches!(result, Err(MppiError::InvalidTemperature)));
    }

    #[test]
    fn valid_config_constructs_ok() {
        let result = MppiConfig::<f64, 2, 1>::new(1.0, [0.5], [-5.0], [5.0], 1.0, 42);
        assert!(result.is_ok());
    }

    // ------------------------------------------------------------------
    // Zero-cost test: weights must be uniform → u* ≈ u_nominal (all zeros)
    // ------------------------------------------------------------------

    #[test]
    fn zero_cost_uniform_weights_u_star_near_nominal() {
        let mut ctrl = make_mppi();
        let x0 = [0.0_f64; 2];

        // With zero cost, all K trajectories have equal cost.
        // The weighted average of K random perturbations around zero
        // nominal should remain near zero (within noise σ/sqrt(K)).
        let u0 = ctrl
            .update(x0, integrator_dynamics, zero_cost)
            .expect("update ok");

        // K=50, sigma=0.5 → std of mean ≈ 0.5/sqrt(50) ≈ 0.07
        // Allow generous margin for LCG finite-sample variance
        assert!(
            u0[0].abs() < 1.5,
            "with zero cost u* should be near 0; got {}",
            u0[0]
        );
    }

    // ------------------------------------------------------------------
    // Quadratic cost + integrator: u* should push state toward origin
    // ------------------------------------------------------------------

    #[test]
    fn quadratic_cost_drives_state_toward_origin() {
        let mut ctrl = make_mppi();

        // Start far from origin
        let x0 = [2.0_f64, 0.0_f64];

        let u0 = ctrl
            .update(x0, integrator_dynamics, quadratic_cost)
            .expect("update ok");

        // For x[0] > 0, the optimal control should be negative (braking / pulling back)
        assert!(
            u0[0] < 0.5,
            "expected control to reduce positive position, got u={}",
            u0[0]
        );
    }

    #[test]
    fn quadratic_cost_negative_position_positive_control() {
        let mut ctrl = make_mppi();

        let x0 = [-2.0_f64, 0.0_f64];

        let u0 = ctrl
            .update(x0, integrator_dynamics, quadratic_cost)
            .expect("update ok");

        // For x[0] < 0, optimal control should be positive
        assert!(
            u0[0] > -0.5,
            "expected control to reduce negative position, got u={}",
            u0[0]
        );
    }

    // ------------------------------------------------------------------
    // ESS tests
    // ------------------------------------------------------------------

    #[test]
    fn ess_le_k_with_zero_cost() {
        let mut ctrl = make_mppi();
        let x0 = [0.0_f64; 2];
        ctrl.update(x0, integrator_dynamics, zero_cost)
            .expect("update ok");
        let stats = ctrl.compute_stats();
        // With zero cost all weights equal → ESS = K exactly
        let k_f64 = 50.0_f64;
        assert!(
            (stats.effective_sample_size - k_f64).abs() < 1e-6,
            "ESS should equal K={} with uniform weights, got {}",
            k_f64,
            stats.effective_sample_size
        );
    }

    #[test]
    fn ess_lt_k_with_high_cost_variance() {
        // Use a very sharp (low temperature) controller so one trajectory dominates.
        let config = MppiConfigBuilder::<f64, 2, 1>::new()
            .temperature(0.01) // very greedy
            .sigma_uniform(2.0)
            .bounds_symmetric(10.0)
            .gamma(1.0)
            .lcg_seed(999)
            .build()
            .expect("config");
        let mut ctrl: Mppi<f64, 2, 1, 10, 50> = Mppi::new(config).expect("mppi");

        let x0 = [3.0_f64, 0.0_f64];
        ctrl.update(x0, integrator_dynamics, quadratic_cost)
            .expect("update ok");

        let stats = ctrl.compute_stats();
        // ESS should be well below K=50 when one trajectory dominates
        assert!(
            stats.effective_sample_size <= 50.0,
            "ESS must be ≤ K, got {}",
            stats.effective_sample_size
        );
    }

    // ------------------------------------------------------------------
    // Reset test
    // ------------------------------------------------------------------

    #[test]
    fn reset_clears_nominal_trajectory() {
        let mut ctrl = make_mppi();
        let x0 = [1.0_f64, 0.0_f64];

        // Run update to populate u_nominal with non-zero values
        ctrl.update(x0, integrator_dynamics, quadratic_cost)
            .expect("update ok");

        // Reset should zero out nominal
        ctrl.reset();

        let nom = ctrl.nominal();
        for h in 0..10 {
            for j in 0..1 {
                assert_eq!(
                    nom[h][j], 0.0,
                    "after reset nominal[{}][{}] should be 0, got {}",
                    h, j, nom[h][j]
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Stats invariants
    // ------------------------------------------------------------------

    #[test]
    fn stats_min_le_mean_le_max() {
        let mut ctrl = make_mppi();
        let x0 = [1.5_f64, -0.5_f64];
        ctrl.update(x0, integrator_dynamics, quadratic_cost)
            .expect("update ok");

        let stats = ctrl.compute_stats();
        assert!(
            stats.min_cost <= stats.mean_cost,
            "min {} > mean {}",
            stats.min_cost,
            stats.mean_cost
        );
        assert!(
            stats.mean_cost <= stats.max_cost,
            "mean {} > max {}",
            stats.mean_cost,
            stats.max_cost
        );
        assert_eq!(stats.n_samples, 50);
    }

    #[test]
    fn stats_ess_positive_and_finite() {
        let mut ctrl = make_mppi();
        let x0 = [0.5_f64, 0.1_f64];
        ctrl.update(x0, integrator_dynamics, quadratic_cost)
            .expect("update ok");

        let stats = ctrl.compute_stats();
        assert!(stats.effective_sample_size > 0.0, "ESS must be positive");
        assert!(
            stats.effective_sample_size.is_finite(),
            "ESS must be finite"
        );
    }

    // ------------------------------------------------------------------
    // Warm-start test: warm-starting from a good trajectory should yield
    // lower baseline cost than cold start (zero nominal).
    // ------------------------------------------------------------------

    #[test]
    fn warm_start_reduces_baseline_cost() {
        let ctrl_cold = make_mppi();
        let mut ctrl_warm = make_mppi();

        let x0 = [2.0_f64, 0.0_f64];

        // Run multiple update steps so warm controller has a good nominal
        for _ in 0..5 {
            ctrl_warm
                .update(x0, integrator_dynamics, quadratic_cost)
                .expect("warm update ok");
        }

        // Baseline cost of warm vs cold nominal
        let warm_cost = ctrl_warm.baseline_cost(x0, integrator_dynamics, quadratic_cost);
        let cold_cost = ctrl_cold.baseline_cost(x0, integrator_dynamics, quadratic_cost);

        assert!(
            warm_cost <= cold_cost,
            "warm_cost={} should be ≤ cold_cost={}",
            warm_cost,
            cold_cost
        );
    }

    // ------------------------------------------------------------------
    // set_nominal test
    // ------------------------------------------------------------------

    #[test]
    fn set_nominal_takes_effect() {
        let mut ctrl = make_mppi();

        // Build a handcrafted trajectory that drives x0=[1,0] toward origin
        // using constant deceleration input.
        let u_good: [[f64; 1]; 10] = [[-0.5]; 10];
        ctrl.set_nominal(&u_good);

        let nom = ctrl.nominal();
        for h in 0..10 {
            assert!(
                (nom[h][0] - (-0.5_f64)).abs() < 1e-12,
                "set_nominal not reflected at h={}",
                h
            );
        }
    }

    // ------------------------------------------------------------------
    // Determinism: same seed → same u* output
    // ------------------------------------------------------------------

    #[test]
    fn deterministic_with_same_seed() {
        let x0 = [1.0_f64, 0.5_f64];

        let mut ctrl1 = make_mppi();
        let mut ctrl2 = make_mppi();

        let u1 = ctrl1
            .update(x0, integrator_dynamics, quadratic_cost)
            .expect("ctrl1 update");
        let u2 = ctrl2
            .update(x0, integrator_dynamics, quadratic_cost)
            .expect("ctrl2 update");

        assert_eq!(u1, u2, "same seed must produce same output");
    }

    // ------------------------------------------------------------------
    // Baseline cost is finite and non-negative for zero nominal on +ve state
    // ------------------------------------------------------------------

    #[test]
    fn baseline_cost_finite_and_nonneg() {
        let ctrl = make_mppi();
        let x0 = [1.0_f64, 0.5_f64];
        let bc = ctrl.baseline_cost(x0, integrator_dynamics, quadratic_cost);
        assert!(bc.is_finite(), "baseline cost must be finite, got {}", bc);
        assert!(bc >= 0.0, "quadratic cost is non-negative, got {}", bc);
    }

    // ------------------------------------------------------------------
    // Control bounds are respected in u*
    // ------------------------------------------------------------------

    #[test]
    fn output_respects_control_bounds() {
        // Use narrow bounds [−0.3, 0.3]
        let config = MppiConfigBuilder::<f64, 2, 1>::new()
            .temperature(1.0)
            .sigma_uniform(2.0) // large noise to stress-test clamping
            .bounds([-0.3], [0.3])
            .gamma(1.0)
            .lcg_seed(77)
            .build()
            .expect("config");
        let mut ctrl: Mppi<f64, 2, 1, 10, 30> = Mppi::new(config).expect("mppi");

        let x0 = [5.0_f64, 0.0_f64];
        let u0 = ctrl
            .update(x0, integrator_dynamics, quadratic_cost)
            .expect("update ok");

        assert!(
            u0[0] >= -0.3 - 1e-9 && u0[0] <= 0.3 + 1e-9,
            "u0={} out of [-0.3, 0.3]",
            u0[0]
        );
    }

    // ------------------------------------------------------------------
    // Builder default → valid config
    // ------------------------------------------------------------------

    #[test]
    fn builder_default_produces_valid_config() {
        let result = MppiConfigBuilder::<f64, 2, 1>::default().build();
        assert!(
            result.is_ok(),
            "default builder should produce valid config"
        );
    }

    // ------------------------------------------------------------------
    // Multi-step receding horizon: state should trend toward origin
    // ------------------------------------------------------------------

    #[test]
    fn receding_horizon_reduces_state_norm() {
        let mut ctrl = make_mppi();
        let mut x = [3.0_f64, 0.0_f64];

        let initial_norm = x[0] * x[0] + x[1] * x[1];

        for _step in 0..15 {
            let u = ctrl
                .update(x, integrator_dynamics, quadratic_cost)
                .expect("update ok");
            x = integrator_dynamics(&x, &u);
        }

        let final_norm = x[0] * x[0] + x[1] * x[1];
        assert!(
            final_norm < initial_norm,
            "state norm should decrease: initial={}, final={}",
            initial_norm,
            final_norm
        );
    }
}
