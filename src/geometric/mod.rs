pub mod geometric_control;
pub mod quaternion;
pub mod se3;
/// Geometric control on SO(3)/SE(3).
///
/// Provides:
/// - `SO3<S>`  — rotation group operations and Lie algebra maps (so3)
/// - `UnitQuat<S>` / `QuatKinematics` — unit quaternion attitude representation (quaternion)
/// - `SE3<S>`, `Twist<S>`, `Wrench<S>` — rigid body transforms (se3)
/// - `GeometricController<S>` — geometric PD tracking controller (Lee 2010) (geometric_control)
///
/// All structures are generic over `S: ControlScalar` (f32 or f64) and require
/// no heap allocation (`no_std` compatible).
pub mod so3;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors arising from geometric operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeoError {
    /// A matrix or vector is singular (e.g., zero-norm axis).
    Singular,
    /// The supplied matrix or quaternion does not satisfy SO(3)/unit constraints.
    InvalidRotation,
    /// An iterative algorithm failed to converge.
    NotConverged,
}

impl core::fmt::Display for GeoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Singular => write!(f, "GeoError: singular matrix or zero-norm vector"),
            Self::InvalidRotation => {
                write!(f, "GeoError: matrix/quaternion is not a valid rotation")
            }
            Self::NotConverged => write!(f, "GeoError: iterative algorithm did not converge"),
        }
    }
}

// ── Public re-exports ─────────────────────────────────────────────────────────

// SO(3)
pub use so3::{hat, rotation_error, vee, SO3};

// Unit quaternion
pub use quaternion::{QuatKinematics, UnitQuat};

// SE(3) / Twist / Wrench
pub use se3::{transform_twist, transform_wrench, Twist, Wrench, SE3};

// Geometric controller
pub use geometric_control::{
    GeometricConfig, GeometricController, GeometricRef, QuadRotorGeomState,
};
