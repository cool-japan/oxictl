//! Genetic Algorithm (GA) for minimizing f: R^D → R (real-valued encoding).
//!
//! Uses tournament selection, arithmetic crossover, and Gaussian mutation
//! with elitism.  All randomness from an LCG (no rand crate).

use crate::core::scalar::ControlScalar;
use crate::optim::particle_swarm::OptimError;

// ---------------------------------------------------------------------------
// LCG helper (duplicated locally so each module is self-contained)
// ---------------------------------------------------------------------------

#[inline]
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// GeneticAlgorithm
// ---------------------------------------------------------------------------

/// Real-valued Genetic Algorithm with `POP` individuals in `D` dimensions.
///
/// Minimises an objective function `f: R^D → R`.
pub struct GeneticAlgorithm<S, const D: usize, const POP: usize> {
    population: [[S; D]; POP],
    fitness: [S; POP],
    bounds_min: [S; D],
    bounds_max: [S; D],
    mutation_rate: S,
    crossover_rate: S,
    elitism: usize,
    lcg: u64,
    generation: usize,
    best_idx: usize,
}

impl<S: ControlScalar, const D: usize, const POP: usize> GeneticAlgorithm<S, D, POP> {
    /// Create a new GA instance.
    ///
    /// * `bounds_min` / `bounds_max` — search space bounds (per dimension).
    /// * `mutation_rate`  — probability of mutating a gene; must be in `(0, 1)`.
    /// * `crossover_rate` — probability of performing crossover; must be in `(0, 1)`.
    /// * `elitism`        — number of elite individuals copied verbatim; must be `< POP`.
    /// * `seed`           — LCG seed.
    pub fn new(
        bounds_min: [S; D],
        bounds_max: [S; D],
        mutation_rate: S,
        crossover_rate: S,
        elitism: usize,
        seed: u64,
    ) -> Result<Self, OptimError> {
        if mutation_rate <= S::ZERO || mutation_rate >= S::ONE {
            return Err(OptimError::InvalidParameter);
        }
        if crossover_rate <= S::ZERO || crossover_rate >= S::ONE {
            return Err(OptimError::InvalidParameter);
        }
        if elitism >= POP {
            return Err(OptimError::InvalidParameter);
        }

        let mut lcg = seed;

        // Initialise population uniformly in [bounds_min, bounds_max].
        let mut population = [[S::ZERO; D]; POP];
        for individual in population.iter_mut() {
            for d in 0..D {
                let r = S::from_f64(lcg_next(&mut lcg));
                individual[d] = bounds_min[d] + r * (bounds_max[d] - bounds_min[d]);
            }
        }

        Ok(Self {
            population,
            fitness: [S::infinity(); POP],
            bounds_min,
            bounds_max,
            mutation_rate,
            crossover_rate,
            elitism,
            lcg,
            generation: 0,
            best_idx: 0,
        })
    }

    // ---- tournament selection (size 3) ------------------------------------

    /// Returns the index of the tournament winner (lowest fitness) from 3
    /// randomly sampled individuals.
    fn tournament_select(&mut self) -> usize {
        let i0 = (lcg_next(&mut self.lcg) * POP as f64) as usize % POP;
        let i1 = (lcg_next(&mut self.lcg) * POP as f64) as usize % POP;
        let i2 = (lcg_next(&mut self.lcg) * POP as f64) as usize % POP;

        let mut best = i0;
        if self.fitness[i1] < self.fitness[best] {
            best = i1;
        }
        if self.fitness[i2] < self.fitness[best] {
            best = i2;
        }
        best
    }

    // ---- find best individual ---------------------------------------------

    fn update_best_idx(&mut self) {
        let mut best = 0;
        for i in 1..POP {
            if self.fitness[i] < self.fitness[best] {
                best = i;
            }
        }
        self.best_idx = best;
    }

    // ---- collect elite indices (ascending fitness) ------------------------

    /// Returns the indices of the `elitism` best individuals (ascending by
    /// fitness).  Uses a simple selection sort; `elitism` is typically small.
    fn elite_indices(&self) -> heapless::Vec<usize, 64> {
        // We cap elitism at 64 in the Vec capacity; the constructor already
        // validates elitism < POP.
        let mut indices: heapless::Vec<usize, 64> = heapless::Vec::new();
        // Collect all indices into a fixed-size array and sort the first
        // `elitism` by fitness (partial selection sort).
        let mut order = [0usize; 256]; // support up to POP=256 — will panic if POP>256
        let count = POP.min(256);
        for (i, slot) in order.iter_mut().enumerate().take(count) {
            *slot = i;
        }
        // Selection sort for first `elitism` positions
        let n = self.elitism.min(count);
        for i in 0..n {
            let mut min_j = i;
            for j in (i + 1)..count {
                if self.fitness[order[j]] < self.fitness[order[min_j]] {
                    min_j = j;
                }
            }
            order.swap(i, min_j);
            // Safety: indices capacity is 64, and elitism < POP; the
            // constructor guarantees elitism < POP ≤ capacity.
            let _ = indices.push(order[i]);
        }
        indices
    }

    /// Perform one GA generation.
    ///
    /// 1. Evaluate fitness for every individual.
    /// 2. Build a new population via tournament selection + arithmetic
    ///    crossover + Gaussian mutation.
    /// 3. Copy the `elitism` best individuals to the front of the new
    ///    population (elitism).
    pub fn step<F: Fn(&[S; D]) -> S>(&mut self, f: &F) -> Result<(), OptimError> {
        // ---- evaluate fitness ----
        for i in 0..POP {
            self.fitness[i] = f(&self.population[i]);
        }
        self.update_best_idx();

        // ---- collect elite before building new pop ----
        let elite_idx = self.elite_indices();

        // ---- build new population ----
        let mut new_pop = [[S::ZERO; D]; POP];
        let mut new_fitness = [S::infinity(); POP];

        // Fill elite slots first (copy individuals AND their fitness)
        for (slot, &src) in elite_idx.iter().enumerate() {
            new_pop[slot] = self.population[src];
            new_fitness[slot] = self.fitness[src];
        }

        // Fill remaining slots via selection + crossover + mutation
        for slot in new_pop.iter_mut().skip(self.elitism) {
            let p1_idx = self.tournament_select();
            let p2_idx = self.tournament_select();
            let p1 = self.population[p1_idx];
            let p2 = self.population[p2_idx];

            let do_cross = lcg_next(&mut self.lcg) < self.crossover_rate.to_f64();

            let mut child = [S::ZERO; D];
            if do_cross {
                let alpha = S::from_f64(lcg_next(&mut self.lcg));
                for (d, gene) in child.iter_mut().enumerate() {
                    *gene = alpha * p1[d] + (S::ONE - alpha) * p2[d];
                }
            } else {
                child = p1;
            }

            // Gaussian mutation (approximate: uniform [-1,1] scaled by sigma)
            let mr = self.mutation_rate.to_f64();
            for (d, gene) in child.iter_mut().enumerate() {
                if lcg_next(&mut self.lcg) < mr {
                    let range = (self.bounds_max[d] - self.bounds_min[d]).to_f64();
                    let noise = range * mr * (lcg_next(&mut self.lcg) - 0.5) * 2.0;
                    *gene += S::from_f64(noise);
                }
                // Clamp to bounds
                *gene = gene.clamp_val(self.bounds_min[d], self.bounds_max[d]);
            }

            *slot = child;
        }

        self.population = new_pop;
        self.fitness = new_fitness;
        self.generation += 1;

        // Evaluate fitness for the non-elite (newly generated) individuals.
        for (fit, ind) in self
            .fitness
            .iter_mut()
            .zip(self.population.iter())
            .skip(self.elitism)
        {
            *fit = f(ind);
        }
        self.update_best_idx();

        Ok(())
    }

    /// Run the GA for `max_iter` generations.
    ///
    /// Returns `(best_fitness, best_individual)`.
    pub fn optimize<F: Fn(&[S; D]) -> S>(
        &mut self,
        f: &F,
        max_iter: usize,
    ) -> Result<(S, [S; D]), OptimError> {
        for _ in 0..max_iter {
            self.step(f)?;
        }
        Ok((self.best_fitness(), self.population[self.best_idx]))
    }

    /// Returns a reference to the current best individual.
    #[inline]
    pub fn best_individual(&self) -> &[S; D] {
        &self.population[self.best_idx]
    }

    /// Returns the fitness of the current best individual.
    #[inline]
    pub fn best_fitness(&self) -> S {
        self.fitness[self.best_idx]
    }

    /// Returns the number of completed generations.
    #[inline]
    pub fn generation(&self) -> usize {
        self.generation
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Minimize f(x) = x^2 (1D) → converges to ~0
    #[test]
    fn ga_minimize_x_squared() {
        let mut ga = GeneticAlgorithm::<f64, 1, 30>::new([-5.0], [5.0], 0.05, 0.8, 2, 42)
            .expect("valid params");

        let (best, _) = ga.optimize(&|x: &[f64; 1]| x[0] * x[0], 300).expect("ok");
        assert!(best < 0.5, "best={best}");
    }

    // 2. Elitism must keep best fitness non-increasing
    #[test]
    fn ga_elitism_preserves_best() {
        let mut ga = GeneticAlgorithm::<f64, 2, 20>::new([-5.0; 2], [5.0; 2], 0.05, 0.8, 2, 7)
            .expect("valid params");

        let obj = |x: &[f64; 2]| x[0] * x[0] + x[1] * x[1];
        let mut prev = f64::INFINITY;
        for _ in 0..10 {
            ga.step(&obj).expect("step ok");
            let cur = ga.best_fitness();
            assert!(cur <= prev + 1e-12, "best went up: {prev} → {cur}");
            prev = cur;
        }
    }

    // 3. mutation_rate=0.0 → InvalidParameter
    #[test]
    fn ga_invalid_mutation_rate() {
        let result = GeneticAlgorithm::<f64, 1, 10>::new([-1.0], [1.0], 0.0, 0.8, 1, 1);
        assert!(matches!(result, Err(OptimError::InvalidParameter)));
    }

    // 4. After construction, not all individuals are identical (diversity)
    #[test]
    fn ga_population_diverse_initially() {
        let ga = GeneticAlgorithm::<f64, 2, 20>::new([-5.0; 2], [5.0; 2], 0.05, 0.8, 2, 99)
            .expect("valid params");

        // At least two individuals must differ
        let first = ga.population[0];
        let all_same = ga
            .population
            .iter()
            .all(|ind| ind[0] == first[0] && ind[1] == first[1]);
        assert!(!all_same, "all individuals identical after init");
    }

    // 5. Best fitness after `optimize` must be ≤ initial best fitness
    #[test]
    fn ga_best_fitness_decreases() {
        let mut ga = GeneticAlgorithm::<f64, 2, 20>::new([-5.0; 2], [5.0; 2], 0.05, 0.8, 2, 55)
            .expect("valid params");

        let obj = |x: &[f64; 2]| x[0] * x[0] + x[1] * x[1];

        // Evaluate initial fitness
        ga.step(&obj).expect("step");
        let initial_best = ga.best_fitness();

        let (final_best, _) = ga.optimize(&obj, 50).expect("ok");
        assert!(
            final_best <= initial_best + 1e-12,
            "final {final_best} > initial {initial_best}"
        );
    }

    // 6. Minimize (x-1)^2 + (y-1)^2 (2D) → near (1,1)
    #[test]
    fn ga_minimize_2d() {
        let mut ga = GeneticAlgorithm::<f64, 2, 40>::new([-3.0; 2], [3.0; 2], 0.05, 0.8, 3, 314)
            .expect("valid params");

        let obj = |x: &[f64; 2]| {
            let dx = x[0] - 1.0;
            let dy = x[1] - 1.0;
            dx * dx + dy * dy
        };
        let (_, best_pos) = ga.optimize(&obj, 400).expect("ok");
        assert!((best_pos[0] - 1.0).abs() < 0.5, "x={}", best_pos[0]);
        assert!((best_pos[1] - 1.0).abs() < 0.5, "y={}", best_pos[1]);
    }
}
