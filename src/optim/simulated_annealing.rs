//! Simulated Annealing (SA) for continuous minimisation.
//!
//! Uses a geometric cooling schedule T[k] = T0 · αᵏ and Box-Muller
//! Gaussian proposals.  All randomness from an LCG (no rand crate).
//! Uses `libm` for transcendental functions (no_std compatible).

use crate::core::scalar::ControlScalar;
use crate::optim::particle_swarm::OptimError;

// ---------------------------------------------------------------------------
// LCG helper
// ---------------------------------------------------------------------------

#[inline]
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// Box-Muller normal sample
// ---------------------------------------------------------------------------

/// Returns one standard-normal sample via the Box-Muller transform.
///
/// Uses `libm::sqrt`, `libm::log`, `libm::cos` for no_std compatibility.
fn normal_sample(lcg: &mut u64) -> f64 {
    let u1 = lcg_next(lcg).max(1e-10);
    let u2 = lcg_next(lcg);
    libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * core::f64::consts::PI * u2)
}

// ---------------------------------------------------------------------------
// SimulatedAnnealing
// ---------------------------------------------------------------------------

/// Simulated Annealing optimiser in `D` dimensions.
///
/// Minimises an objective function `f: R^D → R` using a geometric cooling
/// schedule and Gaussian proposals.
pub struct SimulatedAnnealing<S, const D: usize> {
    x: [S; D],
    f_x: S,
    best_x: [S; D],
    best_f: S,
    temperature: S,
    t0: S,
    alpha: S,
    sigma: S,
    bounds_min: [S; D],
    bounds_max: [S; D],
    lcg: u64,
    iteration: usize,
    n_accepted: usize,
}

impl<S: ControlScalar, const D: usize> SimulatedAnnealing<S, D> {
    /// Create a new SA instance.
    ///
    /// * `x0`         — initial solution.
    /// * `t0`         — initial temperature (must be `> 0`).
    /// * `alpha`      — geometric cooling rate; must be in `(0, 1)`.
    /// * `sigma`      — proposal step size (must be `> 0`).
    /// * `bounds_min` / `bounds_max` — search space bounds.
    /// * `seed`       — LCG seed.
    pub fn new(
        x0: [S; D],
        t0: S,
        alpha: S,
        sigma: S,
        bounds_min: [S; D],
        bounds_max: [S; D],
        seed: u64,
    ) -> Result<Self, OptimError> {
        if t0 <= S::ZERO {
            return Err(OptimError::InvalidParameter);
        }
        if alpha <= S::ZERO || alpha >= S::ONE {
            return Err(OptimError::InvalidParameter);
        }
        if sigma <= S::ZERO {
            return Err(OptimError::InvalidParameter);
        }

        Ok(Self {
            x: x0,
            f_x: S::infinity(),
            best_x: x0,
            best_f: S::infinity(),
            temperature: t0,
            t0,
            alpha,
            sigma,
            bounds_min,
            bounds_max,
            lcg: seed,
            iteration: 0,
            n_accepted: 0,
        })
    }

    /// Perform one SA step.
    ///
    /// * Generates a Gaussian proposal centred at the current solution.
    /// * Accepts or rejects using the Metropolis criterion.
    /// * Multiplies temperature by `alpha`.
    pub fn step<F: Fn(&[S; D]) -> S>(&mut self, f: &F) -> Result<(), OptimError> {
        // Lazily evaluate the current position on the first step.
        if self.f_x.is_infinite() {
            self.f_x = f(&self.x);
            if self.f_x < self.best_f {
                self.best_f = self.f_x;
                self.best_x = self.x;
            }
        }

        // ---- generate proposal ----
        let sigma_f64 = self.sigma.to_f64();
        let mut x_new = self.x;
        for (d, gene) in x_new.iter_mut().enumerate() {
            let noise = S::from_f64(sigma_f64 * normal_sample(&mut self.lcg));
            *gene = (self.x[d] + noise).clamp_val(self.bounds_min[d], self.bounds_max[d]);
        }

        let f_new = f(&x_new);

        // ---- Metropolis acceptance ----
        let accepted = if f_new < self.f_x {
            true
        } else {
            let delta = (f_new - self.f_x).to_f64();
            let t = self.temperature.to_f64();
            // Avoid division by zero if temperature has collapsed
            let p = if t > 0.0 { libm::exp(-delta / t) } else { 0.0 };
            lcg_next(&mut self.lcg) < p
        };

        if accepted {
            self.x = x_new;
            self.f_x = f_new;
            self.n_accepted += 1;

            if self.f_x < self.best_f {
                self.best_f = self.f_x;
                self.best_x = self.x;
            }
        }

        // ---- geometric cooling ----
        self.temperature *= self.alpha;
        self.iteration += 1;

        Ok(())
    }

    /// Run SA for `max_iter` steps.
    ///
    /// Returns `(best_f, best_x)`.
    pub fn optimize<F: Fn(&[S; D]) -> S>(
        &mut self,
        f: &F,
        max_iter: usize,
    ) -> Result<(S, [S; D]), OptimError> {
        for _ in 0..max_iter {
            self.step(f)?;
        }
        Ok((self.best_f, self.best_x))
    }

    /// Fraction of accepted moves so far.
    ///
    /// Returns `0` if no steps have been taken.
    #[inline]
    pub fn acceptance_rate(&self) -> S {
        if self.iteration == 0 {
            return S::ZERO;
        }
        S::from_f64(self.n_accepted as f64 / self.iteration as f64)
    }

    /// Initial temperature (set at construction).
    #[inline]
    pub fn initial_temperature(&self) -> S {
        self.t0
    }

    /// Current temperature.
    #[inline]
    pub fn temperature(&self) -> S {
        self.temperature
    }

    /// Returns a reference to the current (not necessarily best) position.
    #[inline]
    pub fn current_position(&self) -> &[S; D] {
        &self.x
    }

    /// Number of completed steps.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.iteration
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Minimize f(x) = x^2 (1D) → best < 1.0 after 2000 steps
    #[test]
    fn sa_minimize_x_squared() {
        let mut sa =
            SimulatedAnnealing::<f64, 1>::new([3.0_f64], 10.0, 0.99, 0.5, [-5.0], [5.0], 42)
                .expect("valid params");

        let (best, _) = sa.optimize(&|x: &[f64; 1]| x[0] * x[0], 2000).expect("ok");
        assert!(best < 1.0, "best={best}");
    }

    // 2. Temperature after N steps = t0 * alpha^N
    #[test]
    fn sa_temperature_decreases() {
        let t0 = 5.0_f64;
        let alpha = 0.95_f64;
        let n = 50;

        let mut sa = SimulatedAnnealing::<f64, 1>::new([0.0_f64], t0, alpha, 0.1, [-1.0], [1.0], 1)
            .expect("valid params");

        for _ in 0..n {
            sa.step(&|_: &[f64; 1]| 0.0).expect("step");
        }

        let expected = t0 * alpha.powi(n);
        let got = sa.temperature();
        assert!(
            (got - expected).abs() < 1e-10,
            "expected {expected}, got {got}"
        );
    }

    // 3. Acceptance rate > 0 early in optimisation (high temperature)
    #[test]
    fn sa_acceptance_rate_positive_early() {
        let mut sa =
            SimulatedAnnealing::<f64, 1>::new([0.0_f64], 1000.0, 0.99, 1.0, [-10.0], [10.0], 7)
                .expect("valid params");

        for _ in 0..100 {
            sa.step(&|x: &[f64; 1]| x[0] * x[0]).expect("step");
        }

        let rate = sa.acceptance_rate();
        assert!(rate > 0.0, "acceptance_rate={rate}");
    }

    // 4. Invalid parameters return errors
    #[test]
    fn sa_invalid_params() {
        // Negative initial temperature
        assert!(matches!(
            SimulatedAnnealing::<f64, 1>::new([0.0], -1.0, 0.95, 0.1, [-1.0], [1.0], 1),
            Err(OptimError::InvalidParameter)
        ));

        // alpha >= 1.0 (no cooling)
        assert!(matches!(
            SimulatedAnnealing::<f64, 1>::new([0.0], 10.0, 1.0, 0.1, [-1.0], [1.0], 1),
            Err(OptimError::InvalidParameter)
        ));

        // sigma = 0.0 (no movement)
        assert!(matches!(
            SimulatedAnnealing::<f64, 1>::new([0.0], 10.0, 0.95, 0.0, [-1.0], [1.0], 1),
            Err(OptimError::InvalidParameter)
        ));
    }

    // 5. Best solution found lies within the specified bounds
    #[test]
    fn sa_bounds_respected() {
        let lo = [-2.0_f64; 2];
        let hi = [2.0_f64; 2];

        let mut sa = SimulatedAnnealing::<f64, 2>::new([0.5_f64; 2], 5.0, 0.98, 0.3, lo, hi, 99)
            .expect("valid params");

        let (_, best_x) = sa
            .optimize(&|x: &[f64; 2]| x[0] * x[0] + x[1] * x[1], 500)
            .expect("ok");

        for d in 0..2 {
            assert!(
                best_x[d] >= lo[d] && best_x[d] <= hi[d],
                "dim {d} out of bounds: {}",
                best_x[d]
            );
        }
    }
}
