/// Particle Filter (Sequential Monte Carlo) for nonlinear/non-Gaussian estimation.
///
/// Algorithm: Bootstrap Particle Filter (SIR — Sequential Importance Resampling)
///
/// Cycle:
///   1. Predict: propagate particles through nonlinear state transition f(x, u, noise)
///   2. Update: weight particles by likelihood p(y | x)
///   3. Resample: resample N particles from weighted distribution
///
/// State dim N, observation dim M. `P` = number of particles.
///
/// Only available with `std` feature (requires Vec/allocation).
use std::vec::Vec;

/// A single weighted particle.
#[derive(Debug, Clone)]
pub struct Particle<const N: usize> {
    pub state: [f64; N],
    pub weight: f64,
}

/// Particle filter for nonlinear state estimation.
///
/// - `N` = state dimension
/// - `M` = observation dimension
pub struct ParticleFilter<const N: usize, const M: usize> {
    particles: Vec<Particle<N>>,
    /// Process noise standard deviation (per state dimension).
    pub process_noise: [f64; N],
    /// Observation noise standard deviation (per observation dimension).
    pub obs_noise: [f64; M],
    /// State transition: f(state, u) → predicted state (before noise).
    pub transition_fn: fn(&[f64; N], &[f64; M]) -> [f64; N],
    /// Observation likelihood: log p(y | x). Used as log weight.
    pub likelihood_fn: fn(&[f64; N], &[f64; M]) -> f64,
    /// Simple LCG RNG seed (reproducible, no-std compatible in principle).
    rng_state: u64,
}

impl<const N: usize, const M: usize> ParticleFilter<N, M> {
    /// Create a new particle filter with `n_particles` particles.
    ///
    /// Initial particles are drawn from a Gaussian centered at `x0`
    /// with standard deviation `init_std` per dimension.
    pub fn new(
        n_particles: usize,
        x0: [f64; N],
        init_std: f64,
        process_noise: [f64; N],
        obs_noise: [f64; M],
        transition_fn: fn(&[f64; N], &[f64; M]) -> [f64; N],
        likelihood_fn: fn(&[f64; N], &[f64; M]) -> f64,
    ) -> Self {
        let mut pf = Self {
            particles: Vec::with_capacity(n_particles),
            process_noise,
            obs_noise,
            transition_fn,
            likelihood_fn,
            rng_state: 12345,
        };

        // Initialize particles with Gaussian noise around x0
        let w = 1.0 / n_particles as f64;
        for _ in 0..n_particles {
            let state: [f64; N] = core::array::from_fn(|i| x0[i] + pf.randn() * init_std);
            pf.particles.push(Particle { state, weight: w });
        }
        pf
    }

    /// Predict step: propagate particles through state transition + noise.
    pub fn predict(&mut self, u: &[f64; M]) {
        let transition_fn = self.transition_fn;
        let process_noise = self.process_noise;
        // Pre-generate noise to avoid borrow conflict with iter_mut
        let noise: Vec<[f64; N]> = (0..self.particles.len())
            .map(|_| core::array::from_fn(|_| self.randn()))
            .collect();
        for (p, noise) in self.particles.iter_mut().zip(noise.iter()) {
            let x_pred = transition_fn(&p.state, u);
            p.state = core::array::from_fn(|i| x_pred[i] + noise[i] * process_noise[i]);
        }
    }

    /// Update step: weight particles by measurement likelihood.
    pub fn update(&mut self, y: &[f64; M]) {
        // Compute log weights — extract fn ptr to avoid borrow conflict
        let likelihood_fn = self.likelihood_fn;
        let log_weights: Vec<f64> = self
            .particles
            .iter()
            .map(|p| likelihood_fn(&p.state, y))
            .collect();

        // LogSumExp for numerical stability
        let max_lw = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_exp = log_weights
            .iter()
            .map(|&lw| (lw - max_lw).exp())
            .sum::<f64>();
        let log_sum = max_lw + sum_exp.ln();

        // Normalize weights
        for (p, &lw) in self.particles.iter_mut().zip(log_weights.iter()) {
            p.weight = (lw - log_sum).exp();
        }
    }

    /// Resample using systematic resampling (O(N), low variance).
    pub fn resample(&mut self) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }

        // Build cumulative weight array
        let mut cumsum = Vec::with_capacity(n);
        let mut acc = 0.0;
        for p in &self.particles {
            acc += p.weight;
            cumsum.push(acc);
        }

        // Systematic resampling
        let step = 1.0 / n as f64;
        let u0 = self.rand_uniform() * step;
        let w_new = 1.0 / n as f64;

        let mut new_particles = Vec::with_capacity(n);
        let mut j = 0usize;
        for i in 0..n {
            let threshold = u0 + i as f64 * step;
            while j < n - 1 && cumsum[j] < threshold {
                j += 1;
            }
            new_particles.push(Particle {
                state: self.particles[j].state,
                weight: w_new,
            });
        }
        self.particles = new_particles;
    }

    /// Compute the weighted mean state estimate.
    pub fn mean(&self) -> [f64; N] {
        let mut mu = [0.0f64; N];
        for p in &self.particles {
            for (i, mi) in mu.iter_mut().enumerate() {
                *mi += p.weight * p.state[i];
            }
        }
        mu
    }

    /// Compute the weighted variance per state dimension.
    pub fn variance(&self) -> [f64; N] {
        let mu = self.mean();
        let mut var = [0.0f64; N];
        for p in &self.particles {
            for (i, vi) in var.iter_mut().enumerate() {
                let d = p.state[i] - mu[i];
                *vi += p.weight * d * d;
            }
        }
        var
    }

    /// Effective sample size (ESS). Low ESS → resampling needed.
    pub fn effective_sample_size(&self) -> f64 {
        let sum_sq: f64 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        if sum_sq < 1e-300 {
            return 0.0;
        }
        1.0 / sum_sq
    }

    /// Number of particles.
    pub fn n_particles(&self) -> usize {
        self.particles.len()
    }

    /// Run full filter step: predict + update + resample (if ESS low).
    pub fn step(&mut self, u: &[f64; M], y: &[f64; M]) -> [f64; N] {
        self.predict(u);
        self.update(y);
        let ess = self.effective_sample_size();
        if ess < self.particles.len() as f64 / 2.0 {
            self.resample();
        }
        self.mean()
    }

    // --- Simple LCG PRNG ---

    fn rand_u64(&mut self) -> u64 {
        // Xorshift64
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        self.rng_state
    }

    fn rand_uniform(&mut self) -> f64 {
        self.rand_u64() as f64 / u64::MAX as f64
    }

    fn randn(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.rand_uniform().max(1e-15);
        let u2 = self.rand_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f64::consts::PI * u2;
        r * theta.cos()
    }
}

/// Gaussian log-likelihood: log p(y|x) assuming independent Gaussian noise.
///
/// Useful as a building block for `likelihood_fn` in particle filters.
pub fn gaussian_log_likelihood<const N: usize>(residual: &[f64; N], std_dev: &[f64; N]) -> f64 {
    let mut ll = 0.0;
    for (&r, &s) in residual.iter().zip(std_dev.iter()) {
        let s2 = s * s;
        ll -= 0.5 * (r * r / s2 + (2.0 * core::f64::consts::PI * s2).ln());
    }
    ll
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1D integrator: x_{k+1} = x_k + 0.1 * u[0]
    fn integrator_transition(x: &[f64; 1], u: &[f64; 1]) -> [f64; 1] {
        [x[0] + 0.1 * u[0]]
    }

    // Gaussian likelihood: p(y|x) = N(y; x[0], σ=1)
    fn position_likelihood(x: &[f64; 1], y: &[f64; 1]) -> f64 {
        let r = y[0] - x[0];
        -0.5 * r * r // log likelihood (up to constant)
    }

    #[test]
    fn particle_filter_tracks_integrator() {
        let mut pf = ParticleFilter::new(
            200,
            [0.0f64],
            0.5,
            [0.05],
            [0.1],
            integrator_transition,
            position_likelihood,
        );

        let mut x_true = 0.0f64;
        for _ in 0..50 {
            let u = [1.0f64]; // constant velocity
            x_true += 0.1 * u[0];
            let y = [x_true + 0.0]; // perfect measurement
            pf.step(&u, &y);
        }

        let est = pf.mean();
        assert!(
            (est[0] - x_true).abs() < 0.5,
            "est={:.3}, true={:.3}",
            est[0],
            x_true
        );
    }

    #[test]
    fn particle_filter_mean_initialized_near_x0() {
        let pf = ParticleFilter::new(
            500,
            [3.0f64],
            0.01,
            [0.0],
            [0.1],
            integrator_transition,
            position_likelihood,
        );
        let mu = pf.mean();
        assert!((mu[0] - 3.0).abs() < 0.1, "mu={:.4}", mu[0]);
    }

    #[test]
    fn effective_sample_size_uniform_is_n() {
        let pf = ParticleFilter::new(
            100,
            [0.0f64],
            0.1,
            [0.1],
            [0.1],
            integrator_transition,
            position_likelihood,
        );
        let ess = pf.effective_sample_size();
        assert!((ess - 100.0).abs() < 1.0, "ESS={:.2}", ess);
    }

    #[test]
    fn gaussian_log_likelihood_at_zero_residual() {
        let residual = [0.0f64];
        let std = [1.0f64];
        let ll = gaussian_log_likelihood(&residual, &std);
        // -0.5 * ln(2π) ≈ -0.9189
        assert!(
            (ll - (-0.5 * (2.0 * core::f64::consts::PI).ln())).abs() < 1e-6,
            "ll={ll:.6}"
        );
    }

    #[test]
    fn resample_preserves_count() {
        let mut pf = ParticleFilter::new(
            50,
            [0.0f64],
            1.0,
            [0.1],
            [0.1],
            integrator_transition,
            position_likelihood,
        );
        pf.update(&[0.0]);
        pf.resample();
        assert_eq!(pf.n_particles(), 50);
    }
}
