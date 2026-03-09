use crate::core::scalar::ControlScalar;

/// Minimum-jerk polynomial trajectory (5th-order).
///
/// Given boundary conditions at t=0 and t=T:
///   p(0)=p0,  ṗ(0)=v0,  p̈(0)=a0
///   p(T)=p1,  ṗ(T)=v1,  p̈(T)=a1
///
/// The solution minimizes ∫₀ᵀ (d³p/dt³)² dt.
///
/// Closed-form solution via 6×6 Vandermonde system:
///   p(t) = c0 + c1*τ + c2*τ² + c3*τ³ + c4*τ⁴ + c5*τ⁵
///   where τ = t/T ∈ [0, 1]
///
/// Generic over scalar S.
#[derive(Debug, Clone, Copy)]
pub struct MinJerkTrajectory<S: ControlScalar> {
    /// Normalized polynomial coefficients [c0, c1, c2, c3, c4, c5].
    pub coeffs: [S; 6],
    /// Duration T.
    pub duration: S,
    /// Start time offset.
    pub t_start: S,
}

impl<S: ControlScalar> MinJerkTrajectory<S> {
    /// Compute minimum-jerk trajectory from boundary conditions.
    ///
    /// Boundary conditions: positions, velocities, accelerations at t=0 and t=T.
    /// Returns `None` if T ≤ 0.
    pub fn new(p0: S, v0: S, a0: S, p1: S, v1: S, a1: S, duration: S) -> Option<Self> {
        if duration <= S::ZERO {
            return None;
        }

        let t = duration;
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let _t5 = t4 * t;

        // Solve the 6x6 Vandermonde system analytically.
        // c0 = p0, c1 = v0, c2 = a0/2
        let c0 = p0;
        let c1 = v0;
        let c2 = a0 * S::HALF;

        // The remaining 3 equations:
        // p(T) = c0 + c1*T + c2*T² + c3*T³ + c4*T⁴ + c5*T⁵ = p1
        // ṗ(T) = c1 + 2c2*T + 3c3*T² + 4c4*T³ + 5c5*T⁴ = v1
        // p̈(T) = 2c2 + 6c3*T + 12c4*T² + 20c5*T³ = a1
        //
        // Rewriting in terms of unknowns [c3, c4, c5]:
        //   T³*c3 + T⁴*c4 + T⁵*c5 = p1 - c0 - c1*T - c2*T²   := d0
        //   3T²*c3 + 4T³*c4 + 5T⁴*c5 = v1 - c1 - 2c2*T         := d1
        //   6T*c3 + 12T²*c4 + 20T³*c5 = a1 - 2c2                := d2
        //
        // Divide to normalize:
        let d0 = p1 - c0 - c1 * t - c2 * t2;
        let d1 = v1 - c1 - S::TWO * c2 * t;
        let d2 = a1 - S::TWO * c2;

        // 3x3 system (matrix divided by common factors):
        // [T³   T⁴    T⁵  ] [c3]   [d0]
        // [3T²  4T³   5T⁴ ] [c4] = [d1]
        // [6T   12T²  20T³] [c5]   [d2]
        //
        // Divide row 0 by T³, row 1 by T², row 2 by T:
        // [1     T     T²   ] [c3]   [d0/T³]
        // [3     4T    5T²  ] [c4] = [d1/T²]
        // [6     12T   20T² ] [c5]   [d2/T ]
        //
        // Cramer's rule on the divided system:
        // det = |1  T  T²  |
        //       |3  4T  5T²|
        //       |6  12T 20T²|
        //
        // = 1*(4T*20T² - 5T²*12T) - T*(3*20T² - 5T²*6) + T²*(3*12T - 4T*6)
        // = 1*(80T³ - 60T³) - T*(60T² - 30T²) + T²*(36T - 24T)
        // = 20T³ - 30T³ + 12T³ = 2T³

        if t3.abs() < S::EPSILON {
            return None;
        }

        let det = S::TWO * t3;
        let r0 = d0 / t3;
        let r1 = d1 / t2;
        let r2 = d2 / t;

        // Numerators for c3, c4, c5 via Cramer's rule:
        // c3: replace col0 with rhs
        // det_c3 = |r0  T  T² |
        //          |r1  4T  5T²|
        //          |r2  12T 20T²|
        // = r0*(80T³-60T³) - T*(r1*20T²-5T²*r2) + T²*(r1*12T-4T*r2)
        // = 20T³*r0 - T*(20T²*r1 - 5T²*r2) + T²*(12T*r1 - 4T*r2)
        // = 20T³*r0 - 20T³*r1 + 5T³*r2 + 12T³*r1 - 4T³*r2
        // = T³*(20r0 - 8r1 + r2)
        let num_c3 = t3 * (S::from_f64(20.0) * r0 - S::from_f64(8.0) * r1 + r2);

        // c4: replace col1 with rhs
        // det_c4 = |1  r0  T² |
        //          |3  r1  5T²|
        //          |6  r2  20T²|
        // = 1*(r1*20T² - 5T²*r2) - r0*(3*20T² - 5T²*6) + T²*(3*r2 - 6*r1)
        // = 20T²*r1 - 5T²*r2 - r0*(60T² - 30T²) + T²*(3r2 - 6r1)
        // = 20T²*r1 - 5T²*r2 - 30T²*r0 + 3T²*r2 - 6T²*r1
        // = T²*(-30r0 + 14r1 - 2r2)
        let num_c4 = t2 * (S::from_f64(-30.0) * r0 + S::from_f64(14.0) * r1 - S::TWO * r2);

        // c5: replace col2 with rhs
        // det_c5 = |1  T  r0|
        //          |3  4T r1|
        //          |6  12T r2|
        // = 1*(4T*r2 - r1*12T) - T*(3*r2 - r1*6) + r0*(3*12T - 4T*6)
        // = 4T*r2 - 12T*r1 - T*(3r2 - 6r1) + r0*(36T - 24T)
        // = 4T*r2 - 12T*r1 - 3T*r2 + 6T*r1 + 12T*r0
        // = T*(12r0 - 6r1 + r2)
        let num_c5 = t * (S::from_f64(12.0) * r0 - S::from_f64(6.0) * r1 + r2);

        let c3 = num_c3 / det;
        let c4 = num_c4 / det;
        let c5 = num_c5 / det;

        Some(Self {
            coeffs: [c0, c1, c2, c3, c4, c5],
            duration,
            t_start: S::ZERO,
        })
    }

    /// Set start time (shifts the trajectory in time).
    pub fn with_start_time(mut self, t_start: S) -> Self {
        self.t_start = t_start;
        self
    }

    /// Evaluate position at absolute time `t`.
    pub fn position(&self, t: S) -> S {
        let tau = (t - self.t_start).clamp_val(S::ZERO, self.duration);
        let [c0, c1, c2, c3, c4, c5] = self.coeffs;
        c0 + tau * (c1 + tau * (c2 + tau * (c3 + tau * (c4 + tau * c5))))
    }

    /// Evaluate velocity (first derivative) at absolute time `t`.
    pub fn velocity(&self, t: S) -> S {
        let tau = (t - self.t_start).clamp_val(S::ZERO, self.duration);
        let [_, c1, c2, c3, c4, c5] = self.coeffs;
        let four = S::from_f64(4.0);
        let five = S::from_f64(5.0);
        c1 + tau
            * (S::TWO * c2 + tau * (S::from_f64(3.0) * c3 + tau * (four * c4 + tau * five * c5)))
    }

    /// Evaluate acceleration (second derivative) at absolute time `t`.
    pub fn acceleration(&self, t: S) -> S {
        let tau = (t - self.t_start).clamp_val(S::ZERO, self.duration);
        let [_, _, c2, c3, c4, c5] = self.coeffs;
        let six = S::from_f64(6.0);
        let twelve = S::from_f64(12.0);
        let twenty = S::from_f64(20.0);
        S::TWO * c2 + tau * (six * c3 + tau * (twelve * c4 + tau * twenty * c5))
    }

    /// Evaluate jerk (third derivative) at absolute time `t`.
    pub fn jerk(&self, t: S) -> S {
        let tau = (t - self.t_start).clamp_val(S::ZERO, self.duration);
        let [_, _, _, c3, c4, c5] = self.coeffs;
        let six = S::from_f64(6.0);
        let twenty_four = S::from_f64(24.0);
        let sixty = S::from_f64(60.0);
        six * c3 + tau * (twenty_four * c4 + tau * sixty * c5)
    }

    /// Whether the trajectory has completed at time `t`.
    pub fn is_complete(&self, t: S) -> bool {
        t - self.t_start >= self.duration
    }
}

/// Minimum-snap polynomial trajectory (7th-order).
///
/// Minimizes ∫₀ᵀ (d⁴p/dt⁴)² dt.
///
/// Boundary conditions (8): p, ṗ, p̈, p⃛ at t=0 and t=T.
///
/// p(t) = Σ_{i=0}^{7} c_i * τ^i,  τ = t ∈ [0, T].
#[derive(Debug, Clone, Copy)]
pub struct MinSnapTrajectory<S: ControlScalar> {
    /// Polynomial coefficients [c0, …, c7].
    pub coeffs: [S; 8],
    /// Duration T.
    pub duration: S,
    /// Start time offset.
    pub t_start: S,
}

impl<S: ControlScalar> MinSnapTrajectory<S> {
    /// Compute minimum-snap trajectory.
    ///
    /// `j0`, `j1`: initial/final jerk (d³p/dt³).
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub fn new(
        p0: S,
        v0: S,
        a0: S,
        j0: S,
        p1: S,
        v1: S,
        a1: S,
        j1: S,
        duration: S,
    ) -> Option<Self> {
        if duration <= S::ZERO {
            return None;
        }

        let t = duration;
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;
        let t6 = t5 * t;
        let _t7 = t6 * t;

        // First 4 coefficients from t=0 BCs:
        let c0 = p0;
        let c1 = v0;
        let c2 = a0 * S::HALF;
        let c3 = j0 / S::from_f64(6.0);

        // 4 unknowns c4, c5, c6, c7 from t=T BCs:
        // p(T) = sum = p1 → d0
        // ṗ(T) = sum = v1 → d1
        // p̈(T) = sum = a1 → d2
        // p⃛(T) = sum = j1 → d3
        let d0 = p1 - c0 - c1 * t - c2 * t2 - c3 * t3;
        let d1 = v1 - c1 - S::TWO * c2 * t - S::from_f64(3.0) * c3 * t2;
        let d2 = a1 - S::TWO * c2 - S::from_f64(6.0) * c3 * t;
        let d3 = j1 - S::from_f64(6.0) * c3;

        // System (normalized):
        // [T⁴   T⁵   T⁶   T⁷  ] [c4]   [d0]
        // [4T³  5T⁴  6T⁵  7T⁶ ] [c5] = [d1]
        // [12T² 20T³ 30T⁴ 42T⁵] [c6]   [d2]
        // [24T  60T² 120T³ 210T⁴][c7]   [d3]
        //
        // Factor: row i / T^{4-i} etc. Use Gaussian elimination.

        if t4.abs() < S::EPSILON {
            return None;
        }

        // Row normalize: divide row k by common power to get simpler numbers
        // r0: divide by T^4 → [1, T, T², T³, d0/T^4]
        // r1: divide by T^3 → [4, 5T, 6T², 7T³, d1/T^3]
        // r2: divide by T^2 → [12, 20T, 30T², 42T³, d2/T^2]
        // r3: divide by T   → [24, 60T, 120T², 210T³, d3/T]

        let mut mat = [
            [S::ONE, t, t2, t3, d0 / t4],
            [
                S::from_f64(4.0),
                S::from_f64(5.0) * t,
                S::from_f64(6.0) * t2,
                S::from_f64(7.0) * t3,
                d1 / t3,
            ],
            [
                S::from_f64(12.0),
                S::from_f64(20.0) * t,
                S::from_f64(30.0) * t2,
                S::from_f64(42.0) * t3,
                d2 / t2,
            ],
            [
                S::from_f64(24.0),
                S::from_f64(60.0) * t,
                S::from_f64(120.0) * t2,
                S::from_f64(210.0) * t3,
                d3 / t,
            ],
        ];

        // Gaussian elimination with partial pivoting (4x4)
        for col in 0..4 {
            // Find pivot
            let mut max_row = col;
            let mut max_val = mat[col][col].abs();
            for row in (col + 1)..4 {
                if mat[row][col].abs() > max_val {
                    max_val = mat[row][col].abs();
                    max_row = row;
                }
            }
            mat.swap(col, max_row);

            let pivot = mat[col][col];
            if pivot.abs() < S::EPSILON {
                return None;
            }

            for row in (col + 1)..4 {
                let factor = mat[row][col] / pivot;
                for k in col..5 {
                    let sub = factor * mat[col][k];
                    mat[row][k] -= sub;
                }
            }
        }

        // Back substitution
        let mut sol = [S::ZERO; 4];
        for i in (0..4).rev() {
            let mut sum = mat[i][4];
            for j in (i + 1)..4 {
                sum -= mat[i][j] * sol[j];
            }
            if mat[i][i].abs() < S::EPSILON {
                return None;
            }
            sol[i] = sum / mat[i][i];
        }

        // sol[k] are coefficients after normalization: c4 = sol[0], etc.
        let c4 = sol[0];
        let c5 = sol[1];
        let c6 = sol[2];
        let c7 = sol[3];

        Some(Self {
            coeffs: [c0, c1, c2, c3, c4, c5, c6, c7],
            duration,
            t_start: S::ZERO,
        })
    }

    pub fn with_start_time(mut self, t_start: S) -> Self {
        self.t_start = t_start;
        self
    }

    fn tau(&self, t: S) -> S {
        (t - self.t_start).clamp_val(S::ZERO, self.duration)
    }

    /// Evaluate position at absolute time `t`.
    pub fn position(&self, t: S) -> S {
        let tau = self.tau(t);
        let [c0, c1, c2, c3, c4, c5, c6, c7] = self.coeffs;
        c0 + tau * (c1 + tau * (c2 + tau * (c3 + tau * (c4 + tau * (c5 + tau * (c6 + tau * c7))))))
    }

    /// Evaluate velocity at absolute time `t`.
    pub fn velocity(&self, t: S) -> S {
        let tau = self.tau(t);
        let [_, c1, c2, c3, c4, c5, c6, c7] = self.coeffs;
        c1 + tau
            * (S::TWO * c2
                + tau
                    * (S::from_f64(3.0) * c3
                        + tau
                            * (S::from_f64(4.0) * c4
                                + tau
                                    * (S::from_f64(5.0) * c5
                                        + tau
                                            * (S::from_f64(6.0) * c6
                                                + tau * S::from_f64(7.0) * c7)))))
    }

    /// Evaluate acceleration at absolute time `t`.
    pub fn acceleration(&self, t: S) -> S {
        let tau = self.tau(t);
        let [_, _, c2, c3, c4, c5, c6, c7] = self.coeffs;
        S::TWO * c2
            + tau
                * (S::from_f64(6.0) * c3
                    + tau
                        * (S::from_f64(12.0) * c4
                            + tau
                                * (S::from_f64(20.0) * c5
                                    + tau
                                        * (S::from_f64(30.0) * c6 + tau * S::from_f64(42.0) * c7))))
    }

    /// Evaluate snap (4th derivative) at absolute time `t`.
    pub fn snap(&self, t: S) -> S {
        let tau = self.tau(t);
        let [_, _, _, _, c4, c5, c6, c7] = self.coeffs;
        S::from_f64(24.0) * c4
            + tau
                * (S::from_f64(120.0) * c5
                    + tau * (S::from_f64(360.0) * c6 + tau * S::from_f64(840.0) * c7))
    }

    pub fn is_complete(&self, t: S) -> bool {
        t - self.t_start >= self.duration
    }
}

/// Multi-segment polynomial trajectory stitcher.
///
/// Stitches up to `SEG` MinJerkTrajectory segments with C2-continuous joints.
///
/// Each segment starts immediately after the previous one ends.
#[derive(Debug, Clone, Copy)]
pub struct PolynomialPath<S: ControlScalar, const SEG: usize> {
    segments: [Option<MinJerkTrajectory<S>>; SEG],
    seg_count: usize,
    total_duration: S,
}

impl<S: ControlScalar, const SEG: usize> PolynomialPath<S, SEG> {
    /// Create empty path.
    pub fn new() -> Self {
        Self {
            segments: [None; SEG],
            seg_count: 0,
            total_duration: S::ZERO,
        }
    }

    /// Append a segment reaching `p_end` with `v_end`, `a_end` in `duration`.
    ///
    /// The start conditions of the new segment match the end conditions of the previous.
    /// Returns false if the path is full or duration ≤ 0.
    pub fn append_segment(&mut self, p_end: S, v_end: S, a_end: S, duration: S) -> bool {
        if self.seg_count >= SEG || duration <= S::ZERO {
            return false;
        }

        // Start conditions from previous segment end (or zeros if first)
        let (p0, v0, a0) = if self.seg_count == 0 {
            (S::ZERO, S::ZERO, S::ZERO)
        } else {
            let prev_idx = self.seg_count - 1;
            if let Some(prev) = &self.segments[prev_idx] {
                let t_end = prev.t_start + prev.duration;
                (
                    prev.position(t_end),
                    prev.velocity(t_end),
                    prev.acceleration(t_end),
                )
            } else {
                (S::ZERO, S::ZERO, S::ZERO)
            }
        };

        let t_start = self.total_duration;
        let seg = match MinJerkTrajectory::new(p0, v0, a0, p_end, v_end, a_end, duration) {
            Some(s) => s.with_start_time(t_start),
            None => return false,
        };

        self.segments[self.seg_count] = Some(seg);
        self.seg_count += 1;
        self.total_duration += duration;
        true
    }

    /// Evaluate position at absolute time `t`.
    pub fn position(&self, t: S) -> S {
        let seg = self.find_segment(t);
        seg.map(|s| s.position(t)).unwrap_or(S::ZERO)
    }

    /// Evaluate velocity at absolute time `t`.
    pub fn velocity(&self, t: S) -> S {
        let seg = self.find_segment(t);
        seg.map(|s| s.velocity(t)).unwrap_or(S::ZERO)
    }

    /// Evaluate acceleration at absolute time `t`.
    pub fn acceleration(&self, t: S) -> S {
        let seg = self.find_segment(t);
        seg.map(|s| s.acceleration(t)).unwrap_or(S::ZERO)
    }

    fn find_segment(&self, t: S) -> Option<&MinJerkTrajectory<S>> {
        for i in 0..self.seg_count {
            if let Some(seg) = &self.segments[i] {
                let t_end = seg.t_start + seg.duration;
                if t <= t_end {
                    return Some(seg);
                }
            }
        }
        // Return last segment if t is beyond end
        self.segments[self.seg_count.saturating_sub(1)].as_ref()
    }

    pub fn total_duration(&self) -> S {
        self.total_duration
    }

    pub fn segment_count(&self) -> usize {
        self.seg_count
    }
}

impl<S: ControlScalar, const SEG: usize> Default for PolynomialPath<S, SEG> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_jerk_boundary_conditions() {
        let traj = MinJerkTrajectory::<f64>::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
        assert!(
            (traj.position(0.0) - 0.0).abs() < 1e-10,
            "p(0)={}",
            traj.position(0.0)
        );
        assert!(
            (traj.position(1.0) - 1.0).abs() < 1e-9,
            "p(T)={}",
            traj.position(1.0)
        );
        assert!(
            (traj.velocity(0.0)).abs() < 1e-10,
            "v(0)={}",
            traj.velocity(0.0)
        );
        assert!(
            (traj.velocity(1.0)).abs() < 1e-10,
            "v(T)={}",
            traj.velocity(1.0)
        );
        assert!((traj.acceleration(0.0)).abs() < 1e-10);
        assert!((traj.acceleration(1.0)).abs() < 1e-9);
    }

    #[test]
    fn min_jerk_nonzero_velocities() {
        let traj = MinJerkTrajectory::<f64>::new(0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0).unwrap();
        assert!((traj.position(0.0) - 0.0).abs() < 1e-10);
        assert!((traj.position(2.0) - 2.0).abs() < 1e-9);
        assert!((traj.velocity(0.0) - 1.0).abs() < 1e-10);
        assert!((traj.velocity(2.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn min_jerk_constant_position() {
        // p0=p1=1, v0=v1=0, a0=a1=0 → constant trajectory
        let traj = MinJerkTrajectory::<f64>::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
        for i in 0..=10 {
            let t = i as f64 * 0.1;
            assert!(
                (traj.position(t) - 1.0).abs() < 1e-9,
                "t={}, p={}",
                t,
                traj.position(t)
            );
        }
    }

    #[test]
    fn min_jerk_zero_duration_returns_none() {
        assert!(MinJerkTrajectory::<f64>::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn min_jerk_jerk_finite() {
        let traj = MinJerkTrajectory::<f64>::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
        let j_mid = traj.jerk(0.5);
        assert!(j_mid.is_finite(), "jerk at mid={}", j_mid);
    }

    #[test]
    fn min_snap_boundary_conditions() {
        let traj =
            MinSnapTrajectory::<f64>::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).unwrap();
        assert!((traj.position(0.0) - 0.0).abs() < 1e-9);
        assert!(
            (traj.position(1.0) - 1.0).abs() < 1e-9,
            "p(T)={}",
            traj.position(1.0)
        );
        assert!((traj.velocity(0.0)).abs() < 1e-9);
        assert!(
            (traj.velocity(1.0)).abs() < 1e-9,
            "v(T)={}",
            traj.velocity(1.0)
        );
        assert!((traj.acceleration(0.0)).abs() < 1e-9);
        assert!((traj.acceleration(1.0)).abs() < 1e-9);
    }

    #[test]
    fn min_snap_zero_duration_returns_none() {
        assert!(
            MinSnapTrajectory::<f64>::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0).is_none()
        );
    }

    #[test]
    fn polynomial_path_two_segments() {
        let mut path = PolynomialPath::<f64, 4>::new();
        assert!(path.append_segment(1.0, 0.0, 0.0, 1.0)); // 0→1 in 1s
        assert!(path.append_segment(2.0, 0.0, 0.0, 1.0)); // 1→2 in 1s
        assert!((path.total_duration() - 2.0).abs() < 1e-10);

        // Start at 0
        assert!((path.position(0.0) - 0.0).abs() < 1e-9);
        // End at 2
        assert!(
            (path.position(2.0) - 2.0).abs() < 1e-9,
            "end={}",
            path.position(2.0)
        );
        // Midpoint of first segment: t=0.5 → between 0 and 1
        let p_mid = path.position(0.5);
        assert!(p_mid > 0.0 && p_mid < 1.0, "p(0.5)={}", p_mid);
    }

    #[test]
    fn polynomial_path_continuous_velocity_at_joint() {
        let mut path = PolynomialPath::<f64, 4>::new();
        path.append_segment(1.0, 0.0, 0.0, 1.0);
        path.append_segment(2.0, 0.0, 0.0, 1.0);
        // Velocity should be zero at joint (boundary condition of both segments)
        let v_joint = path.velocity(1.0);
        assert!(v_joint.abs() < 1e-6, "v at joint={}", v_joint);
    }

    #[test]
    fn polynomial_path_full_returns_false() {
        let mut path = PolynomialPath::<f64, 2>::new();
        assert!(path.append_segment(1.0, 0.0, 0.0, 1.0));
        assert!(path.append_segment(2.0, 0.0, 0.0, 1.0));
        assert!(!path.append_segment(3.0, 0.0, 0.0, 1.0)); // path is full
    }
}
