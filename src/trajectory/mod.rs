pub mod bezier;
pub mod bspline;
pub mod clothoid;
pub mod cubic_spline;
pub mod dubins;
pub mod jerk_limited;
pub mod polynomial;
pub mod quintic;
pub mod rrt;
pub mod time_optimal;
pub mod trapezoidal;

pub use bezier::{BezierCurve, BezierPath};
pub use bspline::BSpline;
pub use clothoid::ClothoidSegment;
pub use cubic_spline::CubicSpline;
pub use dubins::{DubinsPath, DubinsPathType};
pub use jerk_limited::JerkLimitedProfile;
pub use polynomial::{MinJerkTrajectory, MinSnapTrajectory, PolynomialPath};
pub use quintic::QuinticPolynomial;
pub use rrt::RrtPlanner;
pub use time_optimal::{plan_1dof, sample_at, total_time, TimeOptimalProfile, TimeOptimalSegment};
pub use trapezoidal::{TrapezoidalMotion, TrapezoidalProfile};

/// Errors that can be returned by trajectory planners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryError {
    /// No feasible path exists within the given constraints or iteration budget.
    NoPathFound,
    /// A fixed-size buffer (heapless::Vec) is full.
    BufferFull,
    /// One or more input parameters are invalid (e.g. rho ≤ 0, v_max ≤ 0).
    InvalidParameter,
}
