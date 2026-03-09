pub mod adaptive_filters;
pub mod discretize;
pub mod filters;
pub mod frequency_domain;
pub mod linearization;
pub mod matrix;
pub mod saturation;
pub mod scalar;
pub mod signal;
pub mod state_space;
pub mod traits;
pub mod transfer_fn;

pub use discretize::{discretize_euler, discretize_tustin, discretize_zoh};
pub use linearization::{
    controllability_rank, is_controllable, linearize, linearize_discrete, DiscreteLinearizedSystem,
    LinearizationError, LinearizedSystem,
};
