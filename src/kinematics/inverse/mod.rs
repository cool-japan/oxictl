//! Inverse kinematics (IK) solvers for serial robot arms.
//!
//! This module provides two complementary approaches to solving the inverse
//! kinematics problem (finding joint angles that realise a desired
//! end-effector pose):
//!
//! - **Geometric / closed-form** ([`geometric_6dof`]): Pieper solution for
//!   standard 6R manipulators with a spherical wrist.  Returns up to 8
//!   candidate solutions analytically with no iteration.
//!
//! - **Numerical** ([`numerical_ik()`]): Levenberg-Marquardt / damped
//!   least-squares iteration for arbitrary N-DOF robots.  Handles joint
//!   limits via null-space projection.

pub mod geometric_6dof;
pub mod numerical_ik;

pub use geometric_6dof::{closest_solution, geometric_ik_6dof, IkError, IkSolution};
pub use numerical_ik::{
    numerical_ik, NumericalIkConfig, NumericalIkError, NumericalIkResult, NumericalIkRobot,
    Robot6DofAdapter,
};
