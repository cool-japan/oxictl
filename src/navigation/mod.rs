//! Navigation module for OxiCtl.
//!
//! Provides algorithms for robot localisation and mapping:
//!
//! - [`dead_reckoning`] — wheel odometry + IMU complementary-filter fusion.
//! - [`ekf_slam_2d`]    — 2D Extended Kalman Filter SLAM (range–bearing landmarks).
//! - [`pose_graph`]     — linear pose-graph optimisation with loop-closure.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod dead_reckoning;
pub mod ekf_slam_2d;
pub mod pose_graph;

pub use dead_reckoning::{DeadReckoning, NavigationError};
pub use ekf_slam_2d::EkfSlam2D;
pub use pose_graph::PoseGraph;
pub use pose_graph::PoseGraphError;
