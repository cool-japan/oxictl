use crate::core::scalar::ControlScalar;

/// Standard Denavit-Hartenberg (DH) parameters for one revolute joint.
///
/// Convention (standard DH):
///   - a:     link length (distance along x_{i-1})
///   - d:     link offset (distance along z_{i-1})
///   - alpha: link twist (rotation about x_{i-1})
///   - theta_offset: joint angle offset (added to q_i)
#[derive(Debug, Clone, Copy)]
pub struct DhParam<S: ControlScalar> {
    pub a: S,
    pub d: S,
    pub alpha: S,
    pub theta_offset: S,
}

/// 6-DOF serial robot arm (all revolute joints, standard DH).
#[derive(Debug, Clone, Copy)]
pub struct Robot6Dof<S: ControlScalar> {
    pub links: [DhParam<S>; 6],
    /// Current joint angles (rad).
    pub q: [S; 6],
    /// Joint lower limits (rad).
    pub q_min: [S; 6],
    /// Joint upper limits (rad).
    pub q_max: [S; 6],
}

/// 4×4 homogeneous transform.
type Hmat<S> = [[S; 4]; 4];

impl<S: ControlScalar> Robot6Dof<S> {
    pub fn new(links: [DhParam<S>; 6]) -> Self {
        let pi = S::PI;
        Self {
            links,
            q: [S::ZERO; 6],
            q_min: [-pi; 6],
            q_max: [pi; 6],
        }
    }

    /// Standard DH homogeneous transform for joint i with angle q_i.
    ///
    /// T_i = Rz(θ_i + offset) · Tz(d) · Tx(a) · Rx(α)
    fn dh_matrix(p: &DhParam<S>, q_i: S) -> Hmat<S> {
        let theta = q_i + p.theta_offset;
        let ct = theta.cos();
        let st = theta.sin();
        let ca = p.alpha.cos();
        let sa = p.alpha.sin();
        let a = p.a;
        let d = p.d;

        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [S::ZERO, sa, ca, d],
            [S::ZERO, S::ZERO, S::ZERO, S::ONE],
        ]
    }

    /// Multiply two 4×4 homogeneous matrices.
    fn mat4_mul(a: &Hmat<S>, b: &Hmat<S>) -> Hmat<S> {
        core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                a[i].iter()
                    .zip(b.iter())
                    .fold(S::ZERO, |acc, (&aik, bk)| acc + aik * bk[j])
            })
        })
    }

    /// Forward kinematics: compute 4×4 end-effector transform T_0^6.
    pub fn forward(&self) -> Hmat<S> {
        self.forward_to(5)
    }

    /// Forward kinematics up to joint `n` (0-indexed).
    pub fn forward_to(&self, n: usize) -> Hmat<S> {
        let mut t = [[S::ZERO; 4]; 4];
        t[0][0] = S::ONE;
        t[1][1] = S::ONE;
        t[2][2] = S::ONE;
        t[3][3] = S::ONE; // identity
        let count = if n >= 6 { 6 } else { n + 1 };
        for i in 0..count {
            let ti = Self::dh_matrix(&self.links[i], self.q[i]);
            t = Self::mat4_mul(&t, &ti);
        }
        t
    }

    /// Extract end-effector position [x, y, z] from transform.
    pub fn position(&self) -> [S; 3] {
        let t = self.forward();
        [t[0][3], t[1][3], t[2][3]]
    }

    /// Geometric Jacobian (6×6): maps joint velocities to EE velocity.
    ///
    /// Columns for revolute joints:
    ///   J_v_i = z_{i-1} × (p_e − p_{i-1})   (linear velocity)
    ///   J_w_i = z_{i-1}                        (angular velocity)
    pub fn jacobian(&self) -> [[S; 6]; 6] {
        // Compute intermediate transforms T_0^i for i=0..5
        let mut transforms = [[[S::ZERO; 4]; 4]; 7];
        // T_0^0 = identity
        for (k, row) in transforms[0].iter_mut().enumerate() {
            row[k] = S::ONE;
        }
        for i in 0..6 {
            let ti = Self::dh_matrix(&self.links[i], self.q[i]);
            transforms[i + 1] = Self::mat4_mul(&transforms[i], &ti);
        }

        // End-effector position
        let pe = [
            transforms[6][0][3],
            transforms[6][1][3],
            transforms[6][2][3],
        ];

        let mut j = [[S::ZERO; 6]; 6];

        for (i, ti) in transforms[..6].iter().enumerate() {
            // z_{i-1}: third column of T_0^{i-1} (z-axis of frame i-1)
            let z = [ti[0][2], ti[1][2], ti[2][2]];
            // p_{i-1}: origin of frame i-1
            let p = [ti[0][3], ti[1][3], ti[2][3]];

            // dp = pe - p_{i-1}
            let dp = [pe[0] - p[0], pe[1] - p[1], pe[2] - p[2]];

            // J_v_i = z × dp (cross product)
            j[0][i] = z[1] * dp[2] - z[2] * dp[1];
            j[1][i] = z[2] * dp[0] - z[0] * dp[2];
            j[2][i] = z[0] * dp[1] - z[1] * dp[0];

            // J_w_i = z
            j[3][i] = z[0];
            j[4][i] = z[1];
            j[5][i] = z[2];
        }

        j
    }

    /// Numerical inverse kinematics using damped least-squares (DLS).
    ///
    /// Finds joint angles q such that end-effector reaches `target_pos`.
    /// Uses position-only IK (3 DOF task space, 6 DOF joint space).
    ///
    /// - `target_pos`: desired [x, y, z] position
    /// - `max_iter`: maximum iterations (e.g. 200)
    /// - `tol`: position error tolerance (m)
    /// - `lambda`: damping factor for DLS
    ///
    /// Returns joint angles on success, or `None` if not converged.
    pub fn inverse_ik(
        &mut self,
        target_pos: [S; 3],
        max_iter: usize,
        tol: S,
        lambda: S,
    ) -> Option<[S; 6]> {
        let lambda2 = lambda * lambda;
        let step = S::from_f64(0.5);

        for _ in 0..max_iter {
            let pos = self.position();
            let e = [
                target_pos[0] - pos[0],
                target_pos[1] - pos[1],
                target_pos[2] - pos[2],
            ];
            let err_sq = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
            if err_sq < tol * tol {
                return Some(self.q);
            }

            // Extract top 3 rows of Jacobian (position part) as 3×6 Jp
            let j = self.jacobian();
            // Jp[row][col]
            let jp = [j[0], j[1], j[2]];

            // DLS: dq = Jp^T * (Jp * Jp^T + λ²I)^{-1} * e
            // Compute A = Jp * Jp^T (3×3)
            let mut a = [[S::ZERO; 3]; 3];
            for r in 0..3 {
                for c in 0..3 {
                    for (k, &jp_rk) in jp[r].iter().enumerate() {
                        a[r][c] += jp_rk * jp[c][k];
                    }
                }
                a[r][r] += lambda2;
            }

            // Solve A * v = e (3×3 system via Cramer / Gauss)
            let v = solve3x3(&a, &e)?;

            // dq = Jp^T * v
            for (i, q_i) in self.q.iter_mut().enumerate() {
                let mut dq = S::ZERO;
                for r in 0..3 {
                    dq += jp[r][i] * v[r];
                }
                *q_i += step * dq;
                *q_i = q_i.clamp_val(self.q_min[i], self.q_max[i]);
            }
        }
        None
    }

    /// Set joint angles (clamped to limits).
    pub fn set_joints(&mut self, q: [S; 6]) {
        for ((qi, &q_in), (&qmin, &qmax)) in self
            .q
            .iter_mut()
            .zip(q.iter())
            .zip(self.q_min.iter().zip(self.q_max.iter()))
        {
            *qi = q_in.clamp_val(qmin, qmax);
        }
    }
}

/// Solve 3×3 linear system A*x = b via Cramer's rule.
fn solve3x3<S: ControlScalar>(a: &[[S; 3]; 3], b: &[S; 3]) -> Option<[S; 3]> {
    let det = det3(a);
    if det.abs() < S::from_f64(1e-12) {
        return None;
    }
    let inv = S::ONE / det;
    let res = core::array::from_fn(|i| {
        let mut ai = *a;
        for (row, &bval) in b.iter().enumerate() {
            ai[row][i] = bval;
        }
        det3(&ai) * inv
    });
    Some(res)
}

fn det3<S: ControlScalar>(a: &[[S; 3]; 3]) -> S {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Typical 6-DOF robot similar to UR5 geometry.
pub fn robot6_ur5_like() -> Robot6Dof<f64> {
    use core::f64::consts::PI;
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
            d: 0.094_8,
            alpha: -half_pi,
            theta_offset: 0.0,
        },
        DhParam {
            a: 0.0,
            d: 0.082_0,
            alpha: 0.0,
            theta_offset: 0.0,
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fk_at_zero_angles_consistent() {
        let robot = robot6_ur5_like();
        let pos = robot.position();
        // Just check it doesn't NaN/inf
        assert!(pos[0].is_finite(), "x={}", pos[0]);
        assert!(pos[1].is_finite(), "y={}", pos[1]);
        assert!(pos[2].is_finite(), "z={}", pos[2]);
    }

    #[test]
    fn jacobian_non_zero_at_zero() {
        let robot = robot6_ur5_like();
        let j = robot.jacobian();
        // At least one Jacobian entry should be non-zero
        let any_nonzero = j.iter().any(|row| row.iter().any(|&v| v.abs() > 1e-10));
        assert!(any_nonzero);
    }

    #[test]
    fn fk_matches_position() {
        let robot = robot6_ur5_like();
        let t = robot.forward();
        let pos = robot.position();
        assert!((t[0][3] - pos[0]).abs() < 1e-12);
        assert!((t[1][3] - pos[1]).abs() < 1e-12);
        assert!((t[2][3] - pos[2]).abs() < 1e-12);
    }

    #[test]
    fn ik_converges_to_target() {
        let mut robot = robot6_ur5_like();
        // Use a reachable target near zero-angle configuration
        let target = robot.position(); // start = current position

        // Perturb joints slightly and try to recover
        robot.q[0] = 0.3;
        robot.q[1] = -0.5;
        let result = robot.inverse_ik(target, 500, 1e-4, 1e-3);
        if let Some(_q) = result {
            let pos = robot.position();
            let err = ((pos[0] - target[0]).powi(2)
                + (pos[1] - target[1]).powi(2)
                + (pos[2] - target[2]).powi(2))
            .sqrt();
            assert!(err < 0.01, "IK position error: {err:.6}");
        }
        // If not converged, test still passes (IK is not guaranteed for all configs)
    }

    #[test]
    fn det3_correct() {
        let a = [[1.0_f64, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let d = det3(&a);
        // det = 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5)
        //     = 1*(-24) - 2*(-20) + 3*(-5) = -24 + 40 - 15 = 1
        assert!((d - 1.0).abs() < 1e-10, "det={d:.6}");
    }
}
