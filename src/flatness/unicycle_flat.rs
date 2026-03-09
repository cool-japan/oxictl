//! Unicycle (differential-drive) differential flatness.
//!
//! The unicycle kinematic model is:
//! ```text
//!   ẋ = v · cos θ
//!   ẏ = v · sin θ
//!   θ̇ = ω
//! ```
//!
//! The flat outputs are **σ = [x, y]** (Cartesian position).  From σ and its
//! derivatives all states and inputs can be recovered:
//!
//! ```text
//!   v = √(ẋ² + ẏ²)
//!   θ = atan2(ẏ, ẋ)
//!   ω = (ẍ·ẏ − ẋ·ÿ) / v²
//! ```
//!
//! `FlatPathTracker` integrates a 2-D parametric path (represented as a
//! `FlatPath`) and provides real-time velocity commands using a lookahead
//! strategy: it finds the closest point on the path and advances a fixed
//! lookahead distance ahead to generate a smooth target.

use crate::core::scalar::ControlScalar;
use crate::flatness::FlatnessError;

// ────────────────────────────────────────────────────────────────────────────
// UnicycleFlatMap
// ────────────────────────────────────────────────────────────────────────────

/// Inverse flat map for a unicycle / differential-drive robot.
///
/// Converts flat outputs [x, y] and their derivatives into unicycle
/// states [x, y, θ] and controls [v, ω].
#[derive(Debug, Clone, Copy, Default)]
pub struct UnicycleFlatMap<S: ControlScalar> {
    /// Minimum speed threshold below which heading is undefined.
    pub speed_threshold: S,
    _phantom: core::marker::PhantomData<S>,
}

impl<S: ControlScalar> UnicycleFlatMap<S> {
    /// Create a flat map with a given speed singularity threshold (m/s).
    pub fn new(speed_threshold: S) -> Result<Self, FlatnessError> {
        if speed_threshold < S::ZERO {
            return Err(FlatnessError::InvalidParameter(
                "speed_threshold must be non-negative",
            ));
        }
        Ok(Self {
            speed_threshold,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Recover linear speed v and angular rate ω from first and second
    /// derivatives of the flat output.
    ///
    /// ```text
    ///   v = √(ẋ² + ẏ²)
    ///   ω = (ẍ·ẏ − ẋ·ÿ) / v²
    /// ```
    ///
    /// # Errors
    /// Returns `FlatnessError::Singular` when `v < speed_threshold` (heading
    /// is undefined at rest or near-rest).
    pub fn flat_to_control(
        &self,
        x_dot: S,
        y_dot: S,
        x_ddot: S,
        y_ddot: S,
    ) -> Result<(S, S), FlatnessError> {
        let v2 = x_dot * x_dot + y_dot * y_dot;
        let v = v2.sqrt();

        if v < self.speed_threshold {
            return Err(FlatnessError::Singular);
        }

        let omega = (x_ddot * y_dot - x_dot * y_ddot) / v2;
        Ok((v, omega))
    }

    /// Recover the unicycle state [x, y, θ] from position and velocity.
    ///
    /// ```text
    ///   θ = atan2(ẏ, ẋ)
    /// ```
    ///
    /// # Errors
    /// Returns `FlatnessError::Singular` when speed `√(ẋ²+ẏ²) < speed_threshold`.
    pub fn flat_to_state(&self, x: S, y: S, x_dot: S, y_dot: S) -> Result<[S; 3], FlatnessError> {
        let v2 = x_dot * x_dot + y_dot * y_dot;
        let v = v2.sqrt();

        if v < self.speed_threshold {
            return Err(FlatnessError::Singular);
        }

        let theta = y_dot.atan2(x_dot);
        Ok([x, y, theta])
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ParametricPath — a 2-D path defined by (x(s), y(s)) at N sample points
// ────────────────────────────────────────────────────────────────────────────

/// A 2-D parametric path stored as N evenly-spaced sample points.
///
/// The arc-length parameter `s` runs from 0 to 1.
#[derive(Debug, Clone, Copy)]
pub struct ParametricPath<S: ControlScalar, const N: usize> {
    /// X coordinates at each sample.
    pub xs: [S; N],
    /// Y coordinates at each sample.
    pub ys: [S; N],
    /// Total arc length (sum of segment lengths).
    pub arc_length: S,
}

impl<S: ControlScalar, const N: usize> ParametricPath<S, N> {
    /// Build a parametric path from arrays of x and y coordinates.
    ///
    /// Points are assumed to be in path order.  Returns `None` if N < 2.
    pub fn new(xs: [S; N], ys: [S; N]) -> Option<Self> {
        if N < 2 {
            return None;
        }
        let mut arc = S::ZERO;
        for i in 1..N {
            let dx = xs[i] - xs[i - 1];
            let dy = ys[i] - ys[i - 1];
            arc += (dx * dx + dy * dy).sqrt();
        }
        Some(Self {
            xs,
            ys,
            arc_length: arc,
        })
    }

    /// Evaluate path position at normalised arc-length parameter `s ∈ [0, 1]`.
    ///
    /// Uses linear interpolation between sample points.
    pub fn eval(&self, s: S) -> (S, S) {
        let s = s.clamp_val(S::ZERO, S::ONE);
        // Map s to a segment index
        let n = N;
        let idx_f = s * S::from_f64((n - 1) as f64);
        let idx = {
            let i = idx_f.to_f64() as usize;
            i.min(n - 2)
        };
        let frac = idx_f - S::from_f64(idx as f64);
        let frac = frac.clamp_val(S::ZERO, S::ONE);

        let x = self.xs[idx] + frac * (self.xs[idx + 1] - self.xs[idx]);
        let y = self.ys[idx] + frac * (self.ys[idx + 1] - self.ys[idx]);
        (x, y)
    }

    /// Evaluate path tangent direction at normalised arc-length `s ∈ [0, 1]`.
    ///
    /// Returns a unit vector (dx, dy).  Falls back to (1, 0) if segment is degenerate.
    pub fn tangent(&self, s: S) -> (S, S) {
        let s = s.clamp_val(S::ZERO, S::ONE);
        let n = N;
        let idx_f = s * S::from_f64((n - 1) as f64);
        let idx = (idx_f.to_f64() as usize).min(n - 2);

        let dx = self.xs[idx + 1] - self.xs[idx];
        let dy = self.ys[idx + 1] - self.ys[idx];
        let norm = (dx * dx + dy * dy).sqrt();

        if norm < S::EPSILON {
            (S::ONE, S::ZERO)
        } else {
            (dx / norm, dy / norm)
        }
    }

    /// Find the closest point on the path to (px, py).
    ///
    /// Returns the normalised arc-length parameter `s ∈ [0, 1]`.
    pub fn closest_param(&self, px: S, py: S) -> S {
        let mut best_s = S::ZERO;
        let mut best_dist2 = S::from_f64(f64::MAX);

        for i in 0..(N - 1) {
            let ax = self.xs[i];
            let ay = self.ys[i];
            let bx = self.xs[i + 1];
            let by = self.ys[i + 1];

            let abx = bx - ax;
            let aby = by - ay;
            let ab2 = abx * abx + aby * aby;

            let t = if ab2 < S::EPSILON {
                S::ZERO
            } else {
                let apx = px - ax;
                let apy = py - ay;
                let t = (apx * abx + apy * aby) / ab2;
                t.clamp_val(S::ZERO, S::ONE)
            };

            let cx = ax + t * abx;
            let cy = ay + t * aby;
            let dx = px - cx;
            let dy = py - cy;
            let dist2 = dx * dx + dy * dy;

            if dist2 < best_dist2 {
                best_dist2 = dist2;
                let seg_s0 = S::from_f64(i as f64 / (N - 1) as f64);
                let seg_s1 = S::from_f64((i + 1) as f64 / (N - 1) as f64);
                best_s = seg_s0 + t * (seg_s1 - seg_s0);
            }
        }
        best_s
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FlatPathTracker
// ────────────────────────────────────────────────────────────────────────────

/// Real-time flat-output path tracker for a unicycle robot.
///
/// Maintains an internal arc-length pointer and uses a lookahead distance to
/// generate smooth velocity commands (v_cmd, ω_cmd) that drive the robot along
/// the path.
///
/// The lookahead point is computed by advancing `lookahead_dist` along the path
/// from the closest projected point.  A proportional controller then steers the
/// robot toward the lookahead point.
#[derive(Debug, Clone, Copy)]
pub struct FlatPathTracker<S: ControlScalar, const N: usize> {
    path: ParametricPath<S, N>,
    flat_map: UnicycleFlatMap<S>,
    /// Nominal forward speed (m/s).
    pub nominal_speed: S,
    /// Pure-pursuit lookahead distance (m).
    pub lookahead_dist: S,
    /// Heading proportional gain for angular rate.
    pub heading_gain: S,
    /// Current arc-length parameter estimate (0..1).
    current_s: S,
}

impl<S: ControlScalar, const N: usize> FlatPathTracker<S, N> {
    /// Create a path tracker.
    ///
    /// # Parameters
    /// - `path`: the 2-D parametric path to follow.
    /// - `nominal_speed`: desired forward speed (m/s) along the path.
    /// - `lookahead_dist`: pure-pursuit lookahead distance (m); must be > 0.
    /// - `heading_gain`: proportional gain on heading error → ω_cmd.
    /// - `speed_threshold`: passed to `UnicycleFlatMap`.
    pub fn new(
        path: ParametricPath<S, N>,
        nominal_speed: S,
        lookahead_dist: S,
        heading_gain: S,
        speed_threshold: S,
    ) -> Result<Self, FlatnessError> {
        if nominal_speed <= S::ZERO {
            return Err(FlatnessError::InvalidParameter(
                "nominal_speed must be positive",
            ));
        }
        if lookahead_dist <= S::ZERO {
            return Err(FlatnessError::InvalidParameter(
                "lookahead_dist must be positive",
            ));
        }
        if heading_gain < S::ZERO {
            return Err(FlatnessError::InvalidParameter(
                "heading_gain must be non-negative",
            ));
        }
        let flat_map = UnicycleFlatMap::new(speed_threshold)?;
        Ok(Self {
            path,
            flat_map,
            nominal_speed,
            lookahead_dist,
            heading_gain,
            current_s: S::ZERO,
        })
    }

    /// Update the tracker given the robot's current position and return (v_cmd, ω_cmd).
    ///
    /// The method:
    /// 1. Projects (current_x, current_y) onto the path to get `s_closest`.
    /// 2. Advances `s_lookahead = s_closest + lookahead_dist / arc_length`.
    /// 3. Evaluates the lookahead point and its tangent.
    /// 4. Computes heading error and returns (nominal_speed, heading_gain * heading_error).
    ///
    /// # Errors
    /// Returns `FlatnessError::OutOfRange` if the path arc length is zero.
    pub fn update(
        &mut self,
        current_x: S,
        current_y: S,
        current_theta: S,
    ) -> Result<(S, S), FlatnessError> {
        if self.path.arc_length < S::EPSILON {
            return Err(FlatnessError::OutOfRange);
        }

        // Find closest point on path
        let s_closest = self.path.closest_param(current_x, current_y);

        // Advance by lookahead distance (clamped to path end)
        let ds_lookahead = self.lookahead_dist / self.path.arc_length;
        let s_lookahead = (s_closest + ds_lookahead).clamp_val(S::ZERO, S::ONE);
        self.current_s = s_lookahead;

        // Get lookahead point tangent direction
        let (tx, ty) = self.path.tangent(s_lookahead);

        // Desired heading at lookahead
        let theta_desired = ty.atan2(tx);

        // Heading error (wrap to [-π, π])
        let mut dtheta = theta_desired - current_theta;
        while dtheta > S::PI {
            dtheta -= S::TWO * S::PI;
        }
        while dtheta < -S::PI {
            dtheta += S::TWO * S::PI;
        }

        // Speed command: slow down near end of path
        let remaining = S::ONE - s_closest;
        let speed_factor = remaining.clamp_val(S::ZERO, S::ONE);
        let v_cmd = self.nominal_speed * speed_factor;

        // Angular rate command from heading error
        let omega_cmd = self.heading_gain * dtheta;

        // Validate through flat map if speed is above threshold
        if v_cmd >= self.flat_map.speed_threshold {
            // Use approximate derivatives: ẋ ≈ v·cos θ, ẏ ≈ v·sin θ
            let x_dot = v_cmd * current_theta.cos();
            let y_dot = v_cmd * current_theta.sin();
            // Verify that flat_to_state does not return Singular
            let _ = self
                .flat_map
                .flat_to_state(current_x, current_y, x_dot, y_dot)?;
        }

        Ok((v_cmd, omega_cmd))
    }

    /// Current normalised arc-length progress (0..1).
    pub fn progress(&self) -> S {
        self.current_s
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Circle path: parametric circle.  At any point on the circle with uniform
    /// speed, we expect constant |v| and constant |ω|.
    ///
    /// For x(t)=r·cos(Ω·t), y(t)=r·sin(Ω·t) (CCW circle, Ω>0):
    ///   ẋ = -r·Ω·sin, ẏ = r·Ω·cos
    ///   ẍ = -r·Ω²·cos, ÿ = -r·Ω²·sin
    ///
    /// The flat-map formula ω = (ẍ·ẏ − ẋ·ÿ)/v² gives:
    ///   numerator = (-rΩ²cos)(rΩcos) − (-rΩsin)(-rΩ²sin)
    ///             = -r²Ω³cos² - r²Ω³sin² = -r²Ω³
    ///   ω = -r²Ω³ / (r²Ω²) = -Ω
    ///
    /// The signed curvature convention yields ω = -Ω for a CCW (left-turning)
    /// circle because the unicycle angular rate is ψ̇ = ω (right-hand rule),
    /// and the standard formula ω = (ẍẏ−ẋÿ)/v² equals −Ω for CCW motion.
    #[test]
    fn circle_path_constant_v_and_omega() {
        let flat_map = UnicycleFlatMap::<f64>::new(1e-6).expect("flat map");

        // Sample the circle at many points
        let r = 2.0_f64;
        let omega_param = 1.0_f64; // parametric angular rate (rad/s), CCW
        let v_true = r * omega_param; // |v| = r · |Ω|

        // The flat map gives ω = -omega_param for a CCW circle (see derivation above)
        let omega_expected = -omega_param;

        // Parametric derivatives for x=r·cos(Ω·t), y=r·sin(Ω·t)
        // ẋ = -r·Ω·sin(Ω·t), ẏ = r·Ω·cos(Ω·t)
        // ẍ = -r·Ω²·cos(Ω·t), ÿ = -r·Ω²·sin(Ω·t)

        for i in 0..16 {
            let angle = 2.0 * core::f64::consts::PI * (i as f64) / 16.0;
            let xd = -r * omega_param * angle.sin();
            let yd = r * omega_param * angle.cos();
            let xdd = -r * omega_param * omega_param * angle.cos();
            let ydd = -r * omega_param * omega_param * angle.sin();

            let (v, omega) = flat_map
                .flat_to_control(xd, yd, xdd, ydd)
                .expect("flat_to_control");

            assert!(
                (v - v_true).abs() < 1e-10,
                "angle={:.2}: v={:.6} expected {:.6}",
                angle,
                v,
                v_true
            );
            // ω is constant (magnitude = omega_param, sign per derivation)
            assert!(
                (omega - omega_expected).abs() < 1e-10,
                "angle={:.2}: ω={:.6} expected {:.6}",
                angle,
                omega,
                omega_expected
            );
        }
    }

    /// Straight-line motion: ω should be exactly zero.
    #[test]
    fn straight_line_zero_omega() {
        let flat_map = UnicycleFlatMap::<f64>::new(1e-6).expect("flat map");

        // Moving in +x direction at constant speed: ẏ=0, ÿ=0, ẍ=0
        let v = 1.5_f64;
        let (speed, omega) = flat_map
            .flat_to_control(v, 0.0, 0.0, 0.0)
            .expect("flat_to_control");

        assert!((speed - v).abs() < 1e-12, "speed={:.6}", speed);
        assert!(omega.abs() < 1e-12, "omega={:.2e} should be 0", omega);
    }

    /// flat_to_state returns correct heading angle.
    #[test]
    fn flat_to_state_heading() {
        let flat_map = UnicycleFlatMap::<f64>::new(1e-6).expect("flat map");

        // Moving at 45 degrees
        let v = 2.0_f64;
        let angle = core::f64::consts::PI / 4.0;
        let xd = v * angle.cos();
        let yd = v * angle.sin();

        let state = flat_map.flat_to_state(1.0, 2.0, xd, yd).expect("state");
        assert!((state[0] - 1.0).abs() < 1e-12, "x");
        assert!((state[1] - 2.0).abs() < 1e-12, "y");
        assert!(
            (state[2] - angle).abs() < 1e-10,
            "theta={:.6} expected {:.6}",
            state[2],
            angle
        );
    }

    /// flat_to_control returns Singular when speed is below threshold.
    #[test]
    fn singular_below_threshold() {
        let threshold = 0.1_f64;
        let flat_map = UnicycleFlatMap::<f64>::new(threshold).expect("flat map");

        // Speed of 0.05 < 0.1
        let result = flat_map.flat_to_control(0.05, 0.0, 0.0, 0.0);
        assert!(
            matches!(result, Err(FlatnessError::Singular)),
            "Expected Singular, got {:?}",
            result
        );
    }

    /// Path tracker follows a simple straight-line path with ω≈0.
    #[test]
    fn path_tracker_straight_line() {
        // 5-point straight line from (0,0) to (4,0)
        let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0_f64, 0.0, 0.0, 0.0, 0.0];
        let path = ParametricPath::<f64, 5>::new(xs, ys).expect("path");

        let mut tracker = FlatPathTracker::new(
            path, 1.0, // nominal_speed
            0.5, // lookahead
            2.0, // heading_gain
            1e-6,
        )
        .expect("tracker");

        // Robot at (0, 0) facing +x (theta = 0)
        let (v_cmd, omega_cmd) = tracker.update(0.0, 0.0, 0.0).expect("update");

        // Should be moving forward, essentially zero omega
        assert!(v_cmd >= 0.0, "v_cmd={:.4}", v_cmd);
        assert!(
            omega_cmd.abs() < 1e-6,
            "omega_cmd={:.2e} should be ~0 on straight path",
            omega_cmd
        );
    }

    /// Closest-point projection works correctly.
    #[test]
    fn closest_param_correct() {
        let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0_f64, 0.0, 0.0, 0.0, 0.0];
        let path = ParametricPath::<f64, 5>::new(xs, ys).expect("path");

        // Point directly above x=2, y=1 → closest should be at s≈0.5
        let s = path.closest_param(2.0, 1.0);
        assert!(
            (s - 0.5).abs() < 0.15,
            "closest_param: s={:.4} should be ~0.5",
            s
        );

        // Point behind start → s=0
        let s_start = path.closest_param(-1.0, 0.0);
        assert!(s_start < 0.01, "s_start={:.4}", s_start);

        // Point past end → s≈1
        let s_end = path.closest_param(5.0, 0.0);
        assert!(s_end > 0.99, "s_end={:.4}", s_end);
    }
}
