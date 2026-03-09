//! Particle Swarm Optimization (PSO) for minimizing f: R^D → R.
//!
//! Uses LCG-based pseudo-random number generation (no rand crate).
//! Generic over scalar type S implementing ControlScalar.

use crate::core::scalar::ControlScalar;
use core::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by bioinspired optimizers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimError {
    /// A parameter is out of its valid range.
    InvalidParameter,
    /// The optimizer hit the maximum iteration limit without converging.
    MaxIterReached,
}

impl fmt::Display for OptimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimError::InvalidParameter => write!(f, "invalid optimizer parameter"),
            OptimError::MaxIterReached => write!(f, "maximum iterations reached"),
        }
    }
}

// ---------------------------------------------------------------------------
// LCG helper
// ---------------------------------------------------------------------------

/// Linear Congruential Generator returning a uniform float in [0, 1).
#[inline]
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// ParticleSwarm
// ---------------------------------------------------------------------------

/// Particle Swarm Optimizer with `N` particles in `D` dimensions.
///
/// Minimises an objective function `f: R^D → R` using velocity-position
/// updates driven by personal and global bests.
pub struct ParticleSwarm<S, const D: usize, const N: usize> {
    positions: [[S; D]; N],
    velocities: [[S; D]; N],
    pbest_pos: [[S; D]; N],
    pbest_val: [S; N],
    gbest_pos: [S; D],
    gbest_val: S,
    w: S,
    c1: S,
    c2: S,
    bounds_min: [S; D],
    bounds_max: [S; D],
    lcg: u64,
    iteration: usize,
}

impl<S: ControlScalar, const D: usize, const N: usize> ParticleSwarm<S, D, N> {
    /// Create a new PSO instance.
    ///
    /// * `bounds_min` / `bounds_max` — search space bounds (per dimension).
    /// * `w`  — inertia weight; must be in `(0, 1.5)`.
    /// * `c1` — cognitive coefficient (> 0, typically ~2).
    /// * `c2` — social coefficient   (> 0, typically ~2).
    /// * `seed` — LCG seed.
    pub fn new(
        bounds_min: [S; D],
        bounds_max: [S; D],
        w: S,
        c1: S,
        c2: S,
        seed: u64,
    ) -> Result<Self, OptimError> {
        if w <= S::ZERO || w >= S::from_f64(1.5) {
            return Err(OptimError::InvalidParameter);
        }
        if c1 <= S::ZERO || c2 <= S::ZERO {
            return Err(OptimError::InvalidParameter);
        }

        let mut lcg = seed;

        // Initialise positions uniformly in [bounds_min, bounds_max].
        let mut positions = [[S::ZERO; D]; N];
        for particle in positions.iter_mut() {
            for d in 0..D {
                let r = S::from_f64(lcg_next(&mut lcg));
                particle[d] = bounds_min[d] + r * (bounds_max[d] - bounds_min[d]);
            }
        }

        let velocities = [[S::ZERO; D]; N];
        let pbest_pos = positions;
        let pbest_val = [S::infinity(); N];
        let gbest_pos = positions[0];
        let gbest_val = S::infinity();

        Ok(Self {
            positions,
            velocities,
            pbest_pos,
            pbest_val,
            gbest_pos,
            gbest_val,
            w,
            c1,
            c2,
            bounds_min,
            bounds_max,
            lcg,
            iteration: 0,
        })
    }

    /// Perform one PSO iteration.
    ///
    /// Evaluates `f` for every particle, updates personal/global bests,
    /// then integrates the velocity-position equations.
    pub fn step<F: Fn(&[S; D]) -> S>(&mut self, f: &F) -> Result<(), OptimError> {
        for i in 0..N {
            // ---- evaluate ----
            let val = f(&self.positions[i]);

            // ---- update personal best ----
            if val < self.pbest_val[i] {
                self.pbest_val[i] = val;
                self.pbest_pos[i] = self.positions[i];
            }

            // ---- update global best ----
            if val < self.gbest_val {
                self.gbest_val = val;
                self.gbest_pos = self.positions[i];
            }

            // ---- velocity / position update ----
            let r1 = S::from_f64(lcg_next(&mut self.lcg));
            let r2 = S::from_f64(lcg_next(&mut self.lcg));

            for d in 0..D {
                let cognitive = self.c1 * r1 * (self.pbest_pos[i][d] - self.positions[i][d]);
                let social = self.c2 * r2 * (self.gbest_pos[d] - self.positions[i][d]);

                self.velocities[i][d] = self.w * self.velocities[i][d] + cognitive + social;

                let new_pos = self.positions[i][d] + self.velocities[i][d];
                self.positions[i][d] = new_pos.clamp_val(self.bounds_min[d], self.bounds_max[d]);
            }
        }

        self.iteration += 1;
        Ok(())
    }

    /// Run the optimiser for `max_iter` iterations.
    ///
    /// Returns `(best_value, best_position)`.
    pub fn optimize<F: Fn(&[S; D]) -> S>(
        &mut self,
        f: &F,
        max_iter: usize,
    ) -> Result<(S, [S; D]), OptimError> {
        for _ in 0..max_iter {
            self.step(f)?;
        }
        Ok((self.gbest_val, self.gbest_pos))
    }

    /// Returns a reference to the current global-best position.
    #[inline]
    pub fn best_position(&self) -> &[S; D] {
        &self.gbest_pos
    }

    /// Returns the current global-best objective value.
    #[inline]
    pub fn best_value(&self) -> S {
        self.gbest_val
    }

    /// Returns the number of completed iterations.
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

    // ---- helper: Rosenbrock (2D) ------------------------------------------
    fn rosenbrock(x: &[f64; 2]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    // 1. Minimize f(x) = x^2 (1D) → should converge to 0
    #[test]
    fn pso_minimize_x_squared() {
        let bounds_min = [-5.0_f64];
        let bounds_max = [5.0_f64];
        let mut pso = ParticleSwarm::<f64, 1, 20>::new(bounds_min, bounds_max, 0.7, 2.0, 2.0, 42)
            .expect("valid params");

        let (best_val, _) = pso
            .optimize(&|x: &[f64; 1]| x[0] * x[0], 300)
            .expect("optimize ok");

        assert!(best_val < 0.01, "best_val={best_val}");
    }

    // 2. Minimize f(x,y) = (x-2)^2 + (y+1)^2 → converges to (2, -1)
    #[test]
    fn pso_minimize_2d() {
        let bounds_min = [-5.0_f64; 2];
        let bounds_max = [5.0_f64; 2];
        let mut pso = ParticleSwarm::<f64, 2, 30>::new(bounds_min, bounds_max, 0.7, 2.0, 2.0, 123)
            .expect("valid params");

        let obj = |x: &[f64; 2]| {
            let dx = x[0] - 2.0;
            let dy = x[1] + 1.0;
            dx * dx + dy * dy
        };
        let (_, best_pos) = pso.optimize(&obj, 500).expect("optimize ok");
        assert!((best_pos[0] - 2.0).abs() < 0.1, "x={}", best_pos[0]);
        assert!((best_pos[1] + 1.0).abs() < 0.1, "y={}", best_pos[1]);
    }

    // 3. Invalid inertia weight w=0.0 → InvalidParameter
    #[test]
    fn pso_invalid_w_zero() {
        let result = ParticleSwarm::<f64, 1, 10>::new([-1.0], [1.0], 0.0, 2.0, 2.0, 1);
        assert!(matches!(result, Err(OptimError::InvalidParameter)));
    }

    // 4. Best value must be non-increasing over steps
    #[test]
    fn pso_best_value_decreases() {
        let mut pso = ParticleSwarm::<f64, 2, 20>::new([-5.0; 2], [5.0; 2], 0.7, 2.0, 2.0, 7)
            .expect("valid params");

        let obj = |x: &[f64; 2]| x[0] * x[0] + x[1] * x[1];
        let mut prev = f64::INFINITY;
        for _ in 0..20 {
            pso.step(&obj).expect("step ok");
            let cur = pso.best_value();
            assert!(cur <= prev + 1e-12, "best went up: {prev} → {cur}");
            prev = cur;
        }
    }

    // 5. All particle positions must remain within bounds after many steps
    #[test]
    fn pso_bounds_respected() {
        let lo = [-3.0_f64; 2];
        let hi = [3.0_f64; 2];
        let mut pso =
            ParticleSwarm::<f64, 2, 20>::new(lo, hi, 0.7, 2.0, 2.0, 99).expect("valid params");

        let obj = |x: &[f64; 2]| x[0] * x[0] + x[1] * x[1];
        pso.optimize(&obj, 100).expect("optimize ok");

        // gbest must be inside bounds
        let pos = pso.best_position();
        for d in 0..2 {
            assert!(pos[d] >= lo[d] && pos[d] <= hi[d], "dim {d} out of bounds");
        }
    }

    // 6. Rosenbrock: minimum at (1, 1), value 0
    #[test]
    fn pso_rosenbrock() {
        let bounds_min = [-2.0_f64; 2];
        let bounds_max = [2.0_f64; 2];
        let mut pso = ParticleSwarm::<f64, 2, 40>::new(bounds_min, bounds_max, 0.7, 1.5, 1.5, 314)
            .expect("valid params");

        let (_, best_pos) = pso.optimize(&rosenbrock, 800).expect("optimize ok");
        assert!(
            (best_pos[0] - 1.0).abs() < 0.5 && (best_pos[1] - 1.0).abs() < 0.5,
            "best={best_pos:?}"
        );
    }

    // 7. Same seed → identical results
    #[test]
    fn pso_seed_reproducibility() {
        let make =
            || ParticleSwarm::<f64, 1, 10>::new([-5.0], [5.0], 0.7, 2.0, 2.0, 55).expect("valid");
        let mut a = make();
        let mut b = make();
        let obj = |x: &[f64; 1]| x[0] * x[0];
        let (va, _) = a.optimize(&obj, 50).expect("ok");
        let (vb, _) = b.optimize(&obj, 50).expect("ok");
        assert!((va - vb).abs() < 1e-15, "seeds differ: {va} vs {vb}");
    }
}
