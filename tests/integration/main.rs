// Each scenario exercises a different feature-gated module; compile only the
// ones whose features are enabled so the target builds in every configuration.
#[cfg(feature = "protocol")]
mod ethercat_sim;
#[cfg(feature = "motor")]
mod foc_speed_control;
#[cfg(feature = "kinematics")]
mod kinematics_integration;
#[cfg(feature = "mpc")]
mod mpc_motor;
#[cfg(feature = "state_feedback")]
mod state_feedback_integration;
