//! Bioinspired optimisation algorithms.
//!
//! Provides metaheuristic solvers for minimising continuous objective functions
//! `f: R^D → R` without requiring gradient information.
//!
//! # Algorithms
//!
//! | Module                  | Algorithm                      |
//! |-------------------------|--------------------------------|
//! [`particle_swarm`]        | Particle Swarm Optimisation    |
//! [`genetic_algorithm`]     | Genetic Algorithm (real-coded) |
//! [`simulated_annealing`]   | Simulated Annealing            |
//!
//! All implementations:
//! - are `no_std` compatible (use `libm` for transcendental functions),
//! - use an LCG for pseudo-randomness (no `rand` crate),
//! - are generic over `S: ControlScalar` (supports `f32` and `f64`).

pub mod genetic_algorithm;
pub mod particle_swarm;
pub mod simulated_annealing;

pub use genetic_algorithm::GeneticAlgorithm;
pub use particle_swarm::{OptimError, ParticleSwarm};
pub use simulated_annealing::SimulatedAnnealing;
