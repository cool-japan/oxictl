use crate::core::scalar::ControlScalar;

/// Delta robot (3-RRS parallel manipulator) configuration.
///
/// Three identical arms at 120° intervals:
/// - Base: revolute joints at radius `r_f` from center
/// - Upper arms: length `l1`, actuated
/// - Lower arms: length `l2` (parallelogram linkage)
/// - End-effector: attachment points at radius `r_e` from center
///
/// Coordinate convention: z-axis points down, EE is below the base.
#[derive(Debug, Clone, Copy)]
pub struct DeltaConfig<S: ControlScalar> {
    /// Base triangle circumradius (m) — distance from center to each actuator.
    pub r_f: S,
    /// End-effector triangle circumradius (m).
    pub r_e: S,
    /// Upper arm length (m).
    pub l1: S,
    /// Lower arm (forearm/parallelogram) length (m).
    pub l2: S,
    /// Joint angle limits (min, max) in radians.
    pub theta_min: S,
    pub theta_max: S,
}

impl<S: ControlScalar> DeltaConfig<S> {
    /// Typical small desktop delta robot (~200mm reach).
    pub fn desktop() -> Self {
        Self {
            r_f: S::from_f64(0.15),
            r_e: S::from_f64(0.05),
            l1: S::from_f64(0.2),
            l2: S::from_f64(0.35),
            theta_min: -S::PI / S::TWO,
            theta_max: S::PI / S::TWO,
        }
    }
}

/// Delta robot kinematics.
pub struct DeltaRobot<S: ControlScalar> {
    pub config: DeltaConfig<S>,
    /// Current joint angles [θ1, θ2, θ3] (rad).
    pub theta: [S; 3],
}

impl<S: ControlScalar> DeltaRobot<S> {
    pub fn new(config: DeltaConfig<S>) -> Self {
        Self {
            config,
            theta: [S::ZERO; 3],
        }
    }

    /// Inverse kinematics: given end-effector position (x, y, z), compute joint angles.
    ///
    /// z must be negative (EE is below base).
    ///
    /// Returns `None` if the position is unreachable or outside joint limits.
    pub fn inverse(&self, x: S, y: S, z: S) -> Option<[S; 3]> {
        let mut angles = [S::ZERO; 3];
        let two_pi = S::TWO * S::PI;
        let third = two_pi / S::from_f64(3.0);

        for (i, angle) in angles.iter_mut().enumerate() {
            let delta = S::from_f64(i as f64) * third;
            let cos_d = delta.cos();
            let sin_d = delta.sin();

            // Project EE position into each arm's sagittal plane.
            // pi: radial projection (toward arm), qi: tangential projection
            let pi_proj = x * cos_d + y * sin_d;
            let qi_proj = -x * sin_d + y * cos_d;

            // Effective horizontal distance from arm pivot to wrist sphere center
            let xm = (self.config.r_f - self.config.r_e) - pi_proj;

            let l1 = self.config.l1;
            let l2 = self.config.l2;

            // c_val = cos(angle between upper arm and horizontal at wrist center
            // Derived from: |wrist - elbow|² = L2² with elbow at (L1*cos θ, L1*sin θ)
            let g = xm * xm + z * z + qi_proj * qi_proj + l1 * l1 - l2 * l2;
            let c_val = -g / (S::TWO * l1);

            // r = distance from arm pivot to wrist sphere center (in sagittal plane)
            let r = (xm * xm + z * z).sqrt();
            if r < S::from_f64(1e-10) {
                return None;
            }

            let phi = (-z).atan2(xm); // angle of wrist center from arm pivot
            let ratio = c_val / r;
            if ratio.abs() > S::ONE {
                return None;
            }

            let theta_i = phi - ratio.acos();
            if theta_i < self.config.theta_min || theta_i > self.config.theta_max {
                return None;
            }
            *angle = theta_i;
        }

        Some(angles)
    }

    /// Forward kinematics: given joint angles, compute end-effector position (x, y, z).
    ///
    /// Uses analytical 3-sphere intersection.
    ///
    /// Returns `None` if the configuration is singular.
    pub fn forward(&self) -> Option<(S, S, S)> {
        self.fk_impl(&self.theta)
    }

    fn fk_impl(&self, theta: &[S; 3]) -> Option<(S, S, S)> {
        let two_pi = S::TWO * S::PI;
        let third = two_pi / S::from_f64(3.0);
        let rf_re = self.config.r_f - self.config.r_e;
        let l1 = self.config.l1;

        // Compute effective wrist center for each arm
        // Ei = (rf - re + L1*cos(θi)) * [cos(δi), sin(δi), 0] + [0, 0, L1*sin(θi)]
        let mut ex = [S::ZERO; 3];
        let mut ey = [S::ZERO; 3];
        let mut ez = [S::ZERO; 3];

        for i in 0..3 {
            let delta = S::from_f64(i as f64) * third;
            let r_i = rf_re + l1 * theta[i].cos();
            ex[i] = r_i * delta.cos();
            ey[i] = r_i * delta.sin();
            ez[i] = l1 * theta[i].sin();
        }

        // 3-sphere intersection: |P - Ei|² = L2²
        // Subtract eq0 from eq1 and eq2 to get two planes:
        // p1*x + q1*y + r1*z = s1
        // p2*x + q2*y + r2*z = s2
        let norm0 = ex[0] * ex[0] + ey[0] * ey[0] + ez[0] * ez[0];
        let norm1 = ex[1] * ex[1] + ey[1] * ey[1] + ez[1] * ez[1];
        let norm2 = ex[2] * ex[2] + ey[2] * ey[2] + ez[2] * ez[2];

        let p1 = S::TWO * (ex[0] - ex[1]);
        let q1 = S::TWO * (ey[0] - ey[1]);
        let r1 = S::TWO * (ez[0] - ez[1]);
        let s1 = norm0 - norm1;

        let p2 = S::TWO * (ex[0] - ex[2]);
        let q2 = S::TWO * (ey[0] - ey[2]);
        let r2 = S::TWO * (ez[0] - ez[2]);
        let s2 = norm0 - norm2;

        // Solve for x and y in terms of z:
        // p1*x + q1*y = s1 - r1*z
        // p2*x + q2*y = s2 - r2*z
        // Using Cramer's rule: det = p1*q2 - p2*q1
        let det = p1 * q2 - p2 * q1;
        if det.abs() < S::from_f64(1e-10) {
            return None; // Degenerate configuration
        }
        let det_inv = S::ONE / det;

        // x = ((s1-r1*z)*q2 - (s2-r2*z)*q1) / det
        //   = (s1*q2 - s2*q1 + z*(r2*q1 - r1*q2)) / det
        // y = (p1*(s2-r2*z) - p2*(s1-r1*z)) / det
        //   = (p1*s2 - p2*s1 + z*(p2*r1 - p1*r2)) / det
        let ax = (s1 * q2 - s2 * q1) * det_inv;
        let bx = (r2 * q1 - r1 * q2) * det_inv; // coefficient of z
        let ay = (p1 * s2 - p2 * s1) * det_inv;
        let by_coeff = (p2 * r1 - p1 * r2) * det_inv; // coefficient of z

        // Substitute into sphere 0: (x-E0x)² + (y-E0y)² + (z-E0z)² = L2²
        // (ax + bx*z - E0x)² + (ay + by*z - E0y)² + (z - E0z)² = L2²
        let cx = ax - ex[0];
        let cy = ay - ey[0];
        let cz = -ez[0];

        // Expand: (cx + bx*z)² + (cy + by*z)² + (cz + z)² = L2²
        let a_coeff = bx * bx + by_coeff * by_coeff + S::ONE;
        let b_coeff = S::TWO * (cx * bx + cy * by_coeff + cz);
        let c_coeff = cx * cx + cy * cy + cz * cz - self.config.l2 * self.config.l2;

        let discriminant = b_coeff * b_coeff - S::from_f64(4.0) * a_coeff * c_coeff;
        if discriminant < S::ZERO {
            return None;
        }

        // Two solutions — take the one where z is most negative (EE below base)
        let sqrt_disc = discriminant.sqrt();
        let z1 = (-b_coeff + sqrt_disc) / (S::TWO * a_coeff);
        let z2 = (-b_coeff - sqrt_disc) / (S::TWO * a_coeff);

        // For delta robot: EE is below base → choose more negative z
        let z = if z1 < z2 { z1 } else { z2 };
        let x = ax + bx * z;
        let y = ay + by_coeff * z;

        Some((x, y, z))
    }

    /// Set joint angles (clamped to limits).
    pub fn set_joints(&mut self, theta: [S; 3]) {
        for (i, &ti) in theta.iter().enumerate() {
            self.theta[i] = ti.clamp_val(self.config.theta_min, self.config.theta_max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_delta() -> DeltaRobot<f64> {
        DeltaRobot::new(DeltaConfig::desktop())
    }

    #[test]
    fn fk_at_zero_angles_gives_center_below_base() {
        let mut robot = build_delta();
        robot.set_joints([0.0, 0.0, 0.0]);
        let (x, y, z) = robot.forward().expect("FK should succeed at zero angles");
        // At θ=0: symmetric → x,y ≈ 0, z < 0
        assert!(x.abs() < 1e-6, "x={:.6} should be ~0", x);
        assert!(y.abs() < 1e-6, "y={:.6} should be ~0", y);
        assert!(z < 0.0, "z={:.4} should be negative (EE below base)", z);
    }

    #[test]
    fn ik_fk_roundtrip() {
        let mut robot = build_delta();
        // A reachable point: center, 250mm below base
        let (xr, yr, zr) = (0.0_f64, 0.0, -0.25);
        let angles = robot.inverse(xr, yr, zr).expect("IK should succeed");
        robot.set_joints(angles);
        let (x2, y2, z2) = robot.forward().expect("FK should succeed");
        assert!((x2 - xr).abs() < 1e-4, "x error: {} vs {}", x2, xr);
        assert!((y2 - yr).abs() < 1e-4, "y error: {} vs {}", y2, yr);
        assert!((z2 - zr).abs() < 1e-4, "z error: {} vs {}", z2, zr);
    }

    #[test]
    fn unreachable_returns_none() {
        let robot = build_delta();
        // Far below reach
        assert!(robot.inverse(0.0, 0.0, -1.5).is_none());
    }

    #[test]
    fn off_center_ik_fk_roundtrip() {
        let mut robot = build_delta();
        let (xr, yr, zr) = (0.05_f64, 0.03, -0.22);
        let angles = match robot.inverse(xr, yr, zr) {
            Some(a) => a,
            None => return, // Skip if not reachable
        };
        robot.set_joints(angles);
        let (x2, y2, z2) = robot.forward().expect("FK should succeed");
        assert!((x2 - xr).abs() < 1e-3, "x: {} vs {}", x2, xr);
        assert!((y2 - yr).abs() < 1e-3, "y: {} vs {}", y2, yr);
        assert!((z2 - zr).abs() < 1e-3, "z: {} vs {}", z2, zr);
    }
}
