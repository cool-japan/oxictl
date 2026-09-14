// Each property suite exercises a different feature-gated module; compile only
// the ones whose features are enabled so the target builds in every configuration.
#[cfg(feature = "estimator")]
mod estimator_props;
#[cfg(feature = "kinematics")]
mod kinematics_props;
mod matrix_props;
#[cfg(feature = "motor")]
mod motor_props;
#[cfg(feature = "pid")]
mod pid_props;
#[cfg(feature = "trajectory")]
mod trajectory_props;
