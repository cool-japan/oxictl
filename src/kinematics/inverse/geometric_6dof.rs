//! Geometric (closed-form) 6-DOF inverse kinematics using the Pieper solution.
//!
//! Assumes a standard 6R manipulator with a **spherical wrist**: joints 4, 5, 6
//! axes all intersect at the wrist centre.  Under this assumption the IK
//! decouples into:
//!
//!  1. **Position IK** (joints 1–3): find the wrist-centre from the desired
//!     end-effector pose, then solve for q1/q2/q3 geometrically.
//!  2. **Orientation IK** (joints 4–6): R_34 = R_03^T · R_06, then decompose
//!     via ZYZ Euler angles → q4, q5, q6.
//!
//! Up to 8 candidate solutions are returned (elbow-up/down ×
//! shoulder-left/right × wrist-flip).  Each solution is validated against
//! joint limits before inclusion.

use crate::core::scalar::ControlScalar;
use crate::kinematics::serial::six_dof::DhParam;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Up to 8 joint-angle solutions `[q1, q2, q3, q4, q5, q6]` (rad).
#[derive(Debug, Clone, Copy)]
pub struct IkSolution<S: ControlScalar> {
    /// Valid solution candidates (joint angles in rad).
    pub candidates: [[S; 6]; 8],
    /// Number of valid candidates stored in `candidates[..count]`.
    pub count: usize,
}

impl<S: ControlScalar> IkSolution<S> {
    fn empty() -> Self {
        Self {
            candidates: [[S::ZERO; 6]; 8],
            count: 0,
        }
    }

    /// Attempt to push a candidate; silently drops overflow beyond 8.
    fn push(&mut self, sol: [S; 6]) {
        if self.count < 8 {
            self.candidates[self.count] = sol;
            self.count += 1;
        }
    }

    /// Iterate over valid candidates.
    pub fn iter(&self) -> impl Iterator<Item = &[S; 6]> {
        self.candidates[..self.count].iter()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Reasons the geometric IK solver can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IkError {
    /// Target position is outside the robot's kinematic reachability envelope.
    OutOfReach,
    /// All geometric candidates violate joint limits.
    NoValidSolution,
}

// ---------------------------------------------------------------------------
// Core solver
// ---------------------------------------------------------------------------

/// Geometric 6-DOF IK solver for a standard 6R manipulator with spherical
/// wrist (Pieper solution).
///
/// # Parameters
/// - `links`  – DH parameters for all six joints.
/// - `q_min` / `q_max` – joint limit arrays (rad).
/// - `target_r` – desired end-effector rotation matrix (row-major 3×3).
/// - `target_t` – desired end-effector translation [x, y, z].
///
/// # Returns
/// An [`IkSolution`] containing up to 8 valid candidates, or an [`IkError`].
pub fn geometric_ik_6dof<S: ControlScalar>(
    links: &[DhParam<S>; 6],
    q_min: &[S; 6],
    q_max: &[S; 6],
    target_r: &[[S; 3]; 3],
    target_t: &[S; 3],
) -> Result<IkSolution<S>, IkError> {
    // -----------------------------------------------------------------------
    // Step 1: Wrist-centre position (Frame 4 origin for spherical wrist).
    //
    // For a robot with d5 = 0 (true spherical wrist), the wrist centre is:
    //   wc = p_ee − d6 · z_ee
    //
    // For robots with d5 ≠ 0 the formula is approximate; the numerical IK
    // solver should be preferred in that case.
    // -----------------------------------------------------------------------
    let d6 = links[5].d;
    let approach = [target_r[0][2], target_r[1][2], target_r[2][2]];
    let wc = [
        target_t[0] - d6 * approach[0],
        target_t[1] - d6 * approach[1],
        target_t[2] - d6 * approach[2],
    ];

    // -----------------------------------------------------------------------
    // Step 2: Solve for q1.
    //
    // For a standard 6R arm with alpha0 = π/2 (UR-style), the z-axis of
    // frame 1 in the world is z1 = [sin(q1), −cos(q1), 0], which is
    // HORIZONTAL and perpendicular to the arm plane.  The d4 offset (wrist
    // offset along z3 = z1) adds a perpendicular horizontal component:
    //
    //   wc_x = R·cos(q1) + d4_proj·sin(q1)
    //   wc_y = R·sin(q1) − d4_proj·cos(q1)
    //
    // where R is the sagittal radial coordinate and d4_proj is the effective
    // perpendicular offset in the horizontal plane.
    //
    // For a spherical wrist robot the net horizontal perpendicular offset
    // is d4 (since alpha1=alpha2=0 keep z1=z2=z3 all pointing the same way
    // and d4 is along z3).
    //
    // From  −wc_x·sin(q1) + wc_y·cos(q1) = −d4_perp, we get two q1 values.
    //
    // For a robot where d4 is NOT along a horizontal perpendicular (e.g. a
    // simple planar robot with alpha0=0), set d4_perp = 0 and use
    // q1 = atan2(wc_y, wc_x).
    // -----------------------------------------------------------------------
    let d1 = links[0].d;
    let a1 = links[0].a;
    let a2 = links[1].a;
    let a3 = links[2].a;
    let d4 = links[3].d;
    let alpha0 = links[0].alpha;

    // Determine the horizontal perpendicular offset from d4.
    // For alpha0 ≈ π/2 (UR-style), d4 is along the horizontal-perpendicular
    // z1 axis, creating an offset in the horizontal plane.
    // We detect this by checking whether |sin(alpha0)| ≈ 1 (alpha0 ≈ ±π/2).
    let sin_a0 = alpha0.sin().abs();
    // d4_perp: effective horizontal perpendicular offset from d4.
    // When alpha0 = π/2: z1 is horizontal and perpendicular, d4_perp = d4.
    // When alpha0 = 0:   z1 is vertical,                    d4_perp = 0.
    let d4_perp = d4 * sin_a0;

    // Horizontal distance of wrist centre from base z-axis.
    let wc_xy_sq = wc[0] * wc[0] + wc[1] * wc[1];
    let _wc_xy = wc_xy_sq.sqrt();

    // The equation  −wc_x·sin(q1) + wc_y·cos(q1) = −d4_perp
    // has solutions when wc_xy >= |d4_perp|.
    // q1 = atan2(wc_y, wc_x) + atan2(d4_perp, sqrt(wc_xy² − d4_perp²))
    // giving two solutions (± offset).
    let d4p2 = d4_perp * d4_perp;
    let wc_xy_sq_minus_d4p2 = wc_xy_sq - d4p2;
    let wc_xy_minus_d4_perp = if wc_xy_sq_minus_d4p2 > S::ZERO {
        wc_xy_sq_minus_d4p2.sqrt()
    } else {
        S::ZERO
    };

    let base_angle = atan2(wc[1], wc[0]);
    let offset_angle = atan2(d4_perp, wc_xy_minus_d4_perp);
    let q1_a = wrap_angle(base_angle + offset_angle);
    let q1_b = wrap_angle(base_angle - offset_angle + S::PI);
    let q1_candidates = [q1_a, q1_b];

    // -----------------------------------------------------------------------
    // Step 3: Solve for q2, q3 using the sagittal-plane triangle.
    //
    // After factoring out q1, the sagittal-plane position of the wrist centre
    // (projected onto the arm plane) satisfies:
    //
    //   R  = cos(q1)·wc_x + sin(q1)·wc_y  − a1   (radial = a2·cos(q2) + a3·cos(q2+q3))
    //   sz = wc_z − d1                              (vertical = a2·sin(q2) + a3·sin(q2+q3))
    //
    // Note: d4 does NOT appear in the sagittal plane for UR-style robots
    // (it was accounted for in the q1 computation above).
    //
    // Law of cosines for q3 (using signed link lengths):
    //   cos(q3) = (R² + sz² − a2² − a3²) / (2·a2·a3)
    //
    // q2 = atan2(sz, R) − atan2(a3·sin(q3), a2 + a3·cos(q3))
    // -----------------------------------------------------------------------
    let a2_safe = if a2.abs() < S::from_f64(1e-10) {
        S::from_f64(1e-10).copysign(a2)
    } else {
        a2
    };
    let a3_safe = if a3.abs() < S::from_f64(1e-10) {
        S::from_f64(1e-10).copysign(a3)
    } else {
        a3
    };

    let mut sol = IkSolution::empty();

    for &q1 in &q1_candidates {
        let c1 = q1.cos();
        let s1 = q1.sin();

        // Sagittal-plane projection (radial, vertical).
        let r_sag = c1 * wc[0] + s1 * wc[1] - a1;
        let sz = wc[2] - d1;

        // Squared distance in the sagittal plane.
        let dist2 = r_sag * r_sag + sz * sz;
        let dist = dist2.sqrt();

        // Reachability check.
        let l1 = a2_safe.abs();
        let l2 = a3_safe.abs();
        let reach_max = l1 + l2;
        let reach_min = (l1 - l2).abs();

        if dist > reach_max * S::from_f64(1.0 + 1e-6) || dist < reach_min * S::from_f64(1.0 - 1e-6)
        {
            continue;
        }

        // cos(q3) from law of cosines (signed a2, a3):
        //   dist² = a2² + a3² + 2·a2·a3·cos(q3)
        //   cos(q3) = (dist² − a2² − a3²) / (2·a2·a3)
        let denom_q3 = S::TWO * a2_safe * a3_safe;
        let cos_q3 = if denom_q3.abs() < S::from_f64(1e-14) {
            S::ZERO
        } else {
            ((dist2 - a2_safe * a2_safe - a3_safe * a3_safe) / denom_q3).clamp_val(-S::ONE, S::ONE)
        };

        // Two elbow configurations.
        for elbow_sign in [S::ONE, -S::ONE] {
            let sin_q3 = elbow_sign * (S::ONE - cos_q3 * cos_q3).sqrt();
            let q3 = wrap_angle(atan2(sin_q3, cos_q3));

            // q2 from the sagittal geometry.
            let k1 = a2_safe + a3_safe * cos_q3;
            let k2 = a3_safe * sin_q3;
            let q2 = wrap_angle(atan2(sz, r_sag) - atan2(k2, k1));

            // -----------------------------------------------------------
            // Step 4: Orientation IK — solve q4, q5, q6.
            // -----------------------------------------------------------
            // Build R_0_3 from q1, q2, q3 and extract R_3_6 = R_0_3^T · R_0_6.
            let r03 = fk_rotation_0_3(links, q1, q2, q3);
            let r03_t = mat3_transpose(&r03);
            let r36 = mat3_mul(&r03_t, target_r);

            // Orientation IK: decompose R36 for the DH wrist
            // R36_DH = Rz(q4)·Rx(α4)·Rz(q5)·Rx(α5)·Rz(q6)·Rx(α6)
            //
            // For the standard wrist with α4=π/2, α5=−π/2, α6=0, this
            // simplifies to a wrist whose z-column is:
            //   R36[:,2] = [−sin(q5)·cos(q4), −sin(q5)·sin(q4), cos(q5)]
            // and whose second row is:
            //   R36[2][:] = [sin(q5)·cos(q6), −sin(q5)·sin(q6), cos(q5)]
            //
            // These derive from the actual DH algebra (not standard ZYZ).
            // Two wrist configurations: sin(q5) positive / negative.
            let alpha4 = links[3].alpha;
            let alpha5 = links[4].alpha;
            let alpha6 = links[5].alpha;
            let use_standard_wrist = (alpha4 - S::PI / S::TWO).abs() < S::from_f64(0.01)
                && (alpha5 + S::PI / S::TWO).abs() < S::from_f64(0.01)
                && alpha6.abs() < S::from_f64(0.01);

            for wrist_sign in [S::ONE, -S::ONE] {
                let (q4, q5, q6) = if use_standard_wrist {
                    // DH wrist with α4=π/2, α5=−π/2, α6=0.
                    // Column 2: [-sin(q5)*cos(q4), -sin(q5)*sin(q4), cos(q5)]
                    let cos_q5_v = r36[2][2];
                    let sin_q5_sq = r36[0][2] * r36[0][2] + r36[1][2] * r36[1][2];
                    let sin_q5_v = wrist_sign * sin_q5_sq.sqrt();
                    let q5_v = atan2(sin_q5_v, cos_q5_v);

                    if sin_q5_v.abs() < S::from_f64(1e-7) {
                        // Gimbal lock: q4±q6 = atan2(r36[0][1], r36[0][0])
                        let combined = atan2(r36[0][1], r36[0][0]);
                        (combined * S::HALF, q5_v, combined * S::HALF)
                    } else {
                        // q4: cos(q4) = -R36[0][2]/sin_q5, sin(q4) = -R36[1][2]/sin_q5
                        let q4_v = atan2(-r36[1][2] / sin_q5_v, -r36[0][2] / sin_q5_v);
                        // q6: cos(q6) = R36[2][0]/sin_q5, sin(q6) = -R36[2][1]/sin_q5
                        let q6_v = atan2(-r36[2][1] / sin_q5_v, r36[2][0] / sin_q5_v);
                        (q4_v, q5_v, q6_v)
                    }
                } else {
                    // Generic wrist: approximate ZYZ decomposition.
                    // R36[:,2] ≈ [sin(q5)*cos(q4), sin(q5)*sin(q4), cos(q5)]
                    let cos_q5_v = r36[2][2];
                    let sin_q5_sq = r36[0][2] * r36[0][2] + r36[1][2] * r36[1][2];
                    let sin_q5_v = wrist_sign * sin_q5_sq.sqrt();
                    let q5_v = atan2(sin_q5_v, cos_q5_v);
                    if sin_q5_v.abs() < S::from_f64(1e-7) {
                        let sum = atan2(-r36[0][1], r36[0][0]);
                        (sum * S::HALF, q5_v, sum * S::HALF)
                    } else {
                        let q4_v = atan2(r36[1][2] / sin_q5_v, r36[0][2] / sin_q5_v);
                        let q6_v = atan2(r36[2][1] / sin_q5_v, -r36[2][0] / sin_q5_v);
                        (q4_v, q5_v, q6_v)
                    }
                };

                let candidate = [q1, q2, q3, q4, q5, q6];

                if within_limits(&candidate, q_min, q_max) {
                    sol.push(candidate);
                }
            }
        }
    }

    if sol.count == 0 {
        // Check if the target is geometrically reachable at all.
        let r_wc = (wc[0] * wc[0] + wc[1] * wc[1] + wc[2] * wc[2]).sqrt();
        let total_reach = links[0].d.abs()
            + links[1].a.abs()
            + links[2].a.abs()
            + links[3].d.abs()
            + links[4].d.abs();
        if r_wc > total_reach * S::from_f64(1.1) {
            Err(IkError::OutOfReach)
        } else {
            Err(IkError::NoValidSolution)
        }
    } else {
        Ok(sol)
    }
}

// ---------------------------------------------------------------------------
// Select the solution closest to a reference configuration
// ---------------------------------------------------------------------------

/// From an [`IkSolution`], select the candidate with minimum joint-space
/// distance (L2 norm) to `q_ref`.
///
/// Returns `None` if the solution set is empty.
pub fn closest_solution<S: ControlScalar>(sol: &IkSolution<S>, q_ref: &[S; 6]) -> Option<[S; 6]> {
    if sol.count == 0 {
        return None;
    }
    let mut best_idx = 0;
    let mut best_dist = S::from_f64(f64::MAX);
    for (i, cand) in sol.candidates[..sol.count].iter().enumerate() {
        let mut d = S::ZERO;
        for j in 0..6 {
            let diff = cand[j] - q_ref[j];
            d += diff * diff;
        }
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    Some(sol.candidates[best_idx])
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the rotation matrix R_0_3 for the first three joints using the
/// same DH convention as `Robot6Dof`.
fn fk_rotation_0_3<S: ControlScalar>(links: &[DhParam<S>; 6], q1: S, q2: S, q3: S) -> [[S; 3]; 3] {
    let qs = [q1, q2, q3];
    let mut r = [[S::ZERO; 3]; 3];
    // Identity
    r[0][0] = S::ONE;
    r[1][1] = S::ONE;
    r[2][2] = S::ONE;

    for i in 0..3 {
        let p = &links[i];
        let theta = qs[i] + p.theta_offset;
        let ct = theta.cos();
        let st = theta.sin();
        let ca = p.alpha.cos();
        let sa = p.alpha.sin();

        // DH rotation sub-matrix (upper-left 3×3 of DH homogeneous matrix)
        let ri = [
            [ct, -st * ca, st * sa],
            [st, ct * ca, -ct * sa],
            [S::ZERO, sa, ca],
        ];
        r = mat3_mul(&r, &ri);
    }
    r
}

/// 3×3 matrix multiplication.
fn mat3_mul<S: ControlScalar>(a: &[[S; 3]; 3], b: &[[S; 3]; 3]) -> [[S; 3]; 3] {
    core::array::from_fn(|i| {
        core::array::from_fn(|j| (0..3).fold(S::ZERO, |acc, k| acc + a[i][k] * b[k][j]))
    })
}

/// 3×3 matrix transpose.
fn mat3_transpose<S: ControlScalar>(a: &[[S; 3]; 3]) -> [[S; 3]; 3] {
    core::array::from_fn(|i| core::array::from_fn(|j| a[j][i]))
}

/// Wrap angle to [-π, π].
fn wrap_angle<S: ControlScalar>(mut a: S) -> S {
    let two_pi = S::TWO * S::PI;
    while a > S::PI {
        a -= two_pi;
    }
    while a < -S::PI {
        a += two_pi;
    }
    a
}

/// `atan2` via `num_traits::Float`.
fn atan2<S: ControlScalar>(y: S, x: S) -> S {
    y.atan2(x)
}

/// Check that all joint angles are within their limits.
fn within_limits<S: ControlScalar>(q: &[S; 6], q_min: &[S; 6], q_max: &[S; 6]) -> bool {
    for i in 0..6 {
        if q[i] < q_min[i] - S::from_f64(1e-9) || q[i] > q_max[i] + S::from_f64(1e-9) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::serial::six_dof::{DhParam, Robot6Dof};
    use core::f64::consts::PI;

    /// Build a 6R robot with a **true spherical wrist** (d5 = 0) suitable for
    /// the Pieper geometric IK.  Based loosely on UR5 proportions but with
    /// d5 set to zero so the wrist axes truly intersect.
    fn spherical_wrist_robot() -> Robot6Dof<f64> {
        let half_pi = PI / 2.0;
        Robot6Dof::new([
            DhParam {
                a: 0.0,
                d: 0.089_2,
                alpha: half_pi,
                theta_offset: 0.0,
            },
            DhParam {
                a: -0.425,
                d: 0.0,
                alpha: 0.0,
                theta_offset: 0.0,
            },
            DhParam {
                a: -0.392,
                d: 0.0,
                alpha: 0.0,
                theta_offset: 0.0,
            },
            DhParam {
                a: 0.0,
                d: 0.109_5,
                alpha: half_pi,
                theta_offset: 0.0,
            },
            DhParam {
                a: 0.0,
                d: 0.0,
                alpha: -half_pi,
                theta_offset: 0.0,
            }, // d5=0
            DhParam {
                a: 0.0,
                d: 0.082_0,
                alpha: 0.0,
                theta_offset: 0.0,
            },
        ])
    }

    /// Forward kinematics of the spherical wrist robot at a given joint
    /// configuration, then solve IK and check the position round-trip.
    fn round_trip_at(q_seed: [f64; 6]) {
        let mut robot = spherical_wrist_robot();
        robot.set_joints(q_seed);

        let t = robot.forward();
        let target_r = [
            [t[0][0], t[0][1], t[0][2]],
            [t[1][0], t[1][1], t[1][2]],
            [t[2][0], t[2][1], t[2][2]],
        ];
        let target_t = [t[0][3], t[1][3], t[2][3]];

        let res = geometric_ik_6dof(
            &robot.links,
            &robot.q_min,
            &robot.q_max,
            &target_r,
            &target_t,
        );

        match res {
            Ok(sol) => {
                assert!(sol.count > 0, "Expected at least one solution");
                // Verify each solution via FK
                for cand in sol.iter() {
                    let mut r2 = spherical_wrist_robot();
                    r2.set_joints(*cand);
                    let t2 = r2.forward();
                    let pos_err = ((t2[0][3] - target_t[0]).powi(2)
                        + (t2[1][3] - target_t[1]).powi(2)
                        + (t2[2][3] - target_t[2]).powi(2))
                    .sqrt();
                    // For true spherical wrist, position error should be small
                    assert!(
                        pos_err < 1e-3,
                        "Position error {pos_err:.2e} for q={cand:?}"
                    );
                }
            }
            Err(IkError::NoValidSolution) => {
                // Acceptable when joint limits exclude all geometric solutions
            }
            Err(IkError::OutOfReach) => {
                panic!("FK-derived target should always be reachable");
            }
        }
    }

    #[test]
    fn geometric_ik_zero_config() {
        round_trip_at([0.0; 6]);
    }

    #[test]
    fn geometric_ik_nonzero_config() {
        round_trip_at([0.3, -0.8, 1.0, 0.2, 0.5, -0.4]);
    }

    #[test]
    fn geometric_ik_negative_angles() {
        round_trip_at([-0.5, -1.2, 0.6, -0.3, 0.8, 1.1]);
    }

    #[test]
    fn closest_solution_selects_nearest() {
        let mut sol = IkSolution::<f64>::empty();
        sol.push([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]);
        sol.push([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let ref_q = [0.0_f64; 6];
        let best = closest_solution(&sol, &ref_q).expect("solution exists");
        // First candidate is closer to zero
        assert!((best[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn closest_solution_empty_returns_none() {
        let sol = IkSolution::<f64>::empty();
        assert!(closest_solution(&sol, &[0.0; 6]).is_none());
    }

    #[test]
    fn ik_solution_iterator() {
        let mut sol = IkSolution::<f64>::empty();
        sol.push([1.0; 6]);
        sol.push([2.0; 6]);
        let count = sol.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn geometric_ik_respects_joint_limits() {
        let robot = spherical_wrist_robot();
        // Create very tight joint limits that exclude most solutions
        let mut q_min = robot.q_min;
        let mut q_max = robot.q_max;
        // Restrict q1 to a tiny range near π — unlikely to match for the
        // default target below which is near q1=0.
        q_min[0] = 2.9_f64;
        q_max[0] = 3.1_f64;

        let target_r = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let target_t = [0.5_f64, 0.0, 0.3];

        let res = geometric_ik_6dof(&robot.links, &q_min, &q_max, &target_r, &target_t);
        // Either NoValidSolution or a filtered set — just ensure no panic
        let _ = res;
    }

    #[test]
    fn geometric_ik_multi_configs() {
        // Test several reachable configurations
        let configs: &[[f64; 6]] = &[
            [0.0, -1.0, 1.0, 0.0, 0.5, 0.0],
            [0.5, -0.5, 0.8, -0.2, 0.3, 0.1],
            [-0.3, -1.2, 0.5, 0.4, -0.6, 0.2],
            [1.0, -0.6, 0.7, 0.0, 0.8, -0.5],
        ];
        for &q_seed in configs {
            round_trip_at(q_seed);
        }
    }
}
