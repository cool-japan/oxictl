use crate::core::scalar::ControlScalar;
use crate::kinematics::serial::six_dof::DhParam;

/// Rigid body link inertia parameters.
#[derive(Debug, Clone, Copy)]
pub struct LinkInertia<S: ControlScalar> {
    /// Mass (kg).
    pub mass: S,
    /// Center of mass position in link frame (m).
    pub com: [S; 3],
    /// 3×3 inertia tensor at COM in link frame (kg·m²).
    pub inertia: [[S; 3]; 3],
}

impl<S: ControlScalar> LinkInertia<S> {
    /// Uniform cylinder approximation (common for robot links).
    ///
    /// - `mass`: total mass (kg)
    /// - `radius`: cylinder radius (m)
    /// - `length`: cylinder length (m), COM at half-length along z
    pub fn cylinder(mass: S, radius: S, length: S) -> Self {
        let l2 = length * length;
        let r2 = radius * radius;
        let half = S::from_f64(0.5);
        let twelfth = S::ONE / S::from_f64(12.0);
        let quarter = S::from_f64(0.25);

        // Ixx = Iyy = m*(3r² + l²)/12,  Izz = m*r²/2
        let ixx = mass * (S::from_f64(3.0) * r2 + l2) * twelfth;
        let izz = mass * r2 * half;

        let _ = quarter;
        Self {
            mass,
            com: [S::ZERO, S::ZERO, length * half],
            inertia: [
                [ixx, S::ZERO, S::ZERO],
                [S::ZERO, ixx, S::ZERO],
                [S::ZERO, S::ZERO, izz],
            ],
        }
    }

    /// Point mass at COM.
    pub fn point_mass(mass: S, com: [S; 3]) -> Self {
        Self {
            mass,
            com,
            inertia: [[S::ZERO; 3]; 3],
        }
    }
}

/// Newton-Euler recursive inverse dynamics for an N-DOF serial robot.
///
/// Computes joint torques τ for given q, q̇, q̈ and gravity.
///
/// Forward pass: propagate velocities and accelerations from base to tip.
/// Backward pass: propagate forces and torques from tip to base.
pub struct SerialDynamics<S: ControlScalar, const N: usize> {
    pub links: [LinkInertia<S>; N],
}

impl<S: ControlScalar, const N: usize> SerialDynamics<S, N> {
    pub fn new(links: [LinkInertia<S>; N]) -> Self {
        Self { links }
    }

    /// Compute inverse dynamics: joint torques for given motion.
    ///
    /// - `dh`: DH parameters for each joint
    /// - `q`, `qd`, `qdd`: joint angles, velocities, accelerations
    /// - `gravity`: gravity vector in base frame [gx, gy, gz] (m/s²), e.g. [0, 0, -9.81]
    ///
    /// Returns joint torques (N·m).
    pub fn inverse_dynamics(
        &self,
        dh: &[DhParam<S>; N],
        q: &[S; N],
        qd: &[S; N],
        qdd: &[S; N],
        gravity: [S; 3],
    ) -> [S; N] {
        // Rotation matrices R_i (3×3): rotation part of DH transform
        let mut rot = [[[S::ZERO; 3]; 3]; N];
        for i in 0..N {
            rot[i] = dh_rot3(dh[i].alpha, q[i] + dh[i].theta_offset);
        }

        // Link origins in previous frame
        let mut origin = [[S::ZERO; 3]; N];
        for i in 0..N {
            let theta = q[i] + dh[i].theta_offset;
            origin[i] = [dh[i].a * theta.cos(), dh[i].a * theta.sin(), dh[i].d];
        }

        // --- Forward pass: compute angular/linear velocity & acceleration ---
        // omega[i], alpha[i] in link i frame
        // v[i], a[i] linear velocity/acceleration at joint origin i
        let mut omega = [[S::ZERO; 3]; N]; // angular velocity
        let mut alpha_ang = [[S::ZERO; 3]; N]; // angular acceleration
        let mut a_lin = [[S::ZERO; 3]; N]; // linear acceleration of joint origin

        // Base: account for gravity as if base accelerates upward
        let a_base = [-gravity[0], -gravity[1], -gravity[2]];

        for i in 0..N {
            let r = rot[i]; // R_i = rotation from frame i to frame i-1 (3×3)
            let rt = transpose3(&r); // R_i^T: from i-1 to i

            // z axis of previous frame, expressed in frame i
            let z_prev_in_i = mat3_vec(&rt, &[S::ZERO, S::ZERO, S::ONE]);

            if i == 0 {
                // ω_1 = R_1^T * ω_0 + qd_1 * z_1^hat
                omega[i] = vec3_add(&[S::ZERO; 3], &vec3_scale(&z_prev_in_i, qd[i]));
                // α_1 = R_1^T * α_0 + qdd_1 * z_1^hat + qd_1 * ω_1 × z_1^hat
                let cross_term = cross3(&omega[i], &z_prev_in_i);
                alpha_ang[i] = vec3_add(
                    &vec3_scale(&z_prev_in_i, qdd[i]),
                    &vec3_scale(&cross_term, qd[i]),
                );
                // a_1 = R_1^T*(a_base + α_0 × r + ω_0 × (ω_0 × r))
                let r_in_i = mat3_vec(&rt, &origin[i]);
                a_lin[i] = vec3_add(&mat3_vec(&rt, &a_base), &cross3(&alpha_ang[i], &r_in_i));
                let omega_r = cross3(&omega[i], &r_in_i);
                a_lin[i] = vec3_add(&a_lin[i], &cross3(&omega[i], &omega_r));
            } else {
                let prev_omega = omega[i - 1];
                let prev_alpha = alpha_ang[i - 1];
                let prev_a = a_lin[i - 1];

                // Express previous frame quantities in current frame
                let omega_prev_i = mat3_vec(&rt, &prev_omega);
                let alpha_prev_i = mat3_vec(&rt, &prev_alpha);
                let a_prev_i = mat3_vec(&rt, &prev_a);

                omega[i] = vec3_add(&omega_prev_i, &vec3_scale(&z_prev_in_i, qd[i]));
                let cross_term = cross3(&omega[i], &z_prev_in_i);
                alpha_ang[i] = vec3_add(
                    &vec3_add(&alpha_prev_i, &vec3_scale(&z_prev_in_i, qdd[i])),
                    &vec3_scale(&cross_term, qd[i]),
                );
                let r_in_i = origin[i]; // already in frame i
                let alpha_cross_r = cross3(&alpha_ang[i], &r_in_i);
                let omega_omega_r = cross3(&omega[i], &cross3(&omega[i], &r_in_i));
                a_lin[i] = vec3_add(&vec3_add(&a_prev_i, &alpha_cross_r), &omega_omega_r);
            }
        }

        // COM accelerations: a_c_i = a_i + α_i × com_i + ω_i × (ω_i × com_i)
        let mut a_com = [[S::ZERO; 3]; N];
        for i in 0..N {
            let ci = self.links[i].com;
            let aw = cross3(&alpha_ang[i], &ci);
            let ww = cross3(&omega[i], &cross3(&omega[i], &ci));
            a_com[i] = vec3_add(&vec3_add(&a_lin[i], &aw), &ww);
        }

        // --- Backward pass ---
        let mut force = [[S::ZERO; 3]; N]; // force at joint i in frame i
        let mut torque = [[S::ZERO; 3]; N]; // torque at joint i in frame i

        for idx in 0..N {
            let i = N - 1 - idx;
            let m = self.links[i].mass;
            let inert = self.links[i].inertia;

            // F_i = m_i * a_ci
            let fi = vec3_scale(&a_com[i], m);

            // τ_i = I_i * α_i + ω_i × (I_i * ω_i)
            let iw = mat3_vec(&inert, &omega[i]);
            let ia = mat3_vec(&inert, &alpha_ang[i]);
            let w_cross_iw = cross3(&omega[i], &iw);
            let ti = vec3_add(&ia, &w_cross_iw);

            if i == N - 1 {
                force[i] = fi;
                torque[i] = ti;
            } else {
                // Add force/torque from next link (expressed in frame i)
                let r_next = rot[i + 1]; // rotation from i+1 to i
                let f_next_in_i = mat3_vec(&r_next, &force[i + 1]);
                let t_next_in_i = mat3_vec(&r_next, &torque[i + 1]);

                force[i] = vec3_add(&fi, &f_next_in_i);

                // t_i = τ_i + t_{i+1} + r_{i+1} × f_{i+1} + com_i × f_i
                let r_next_orig = origin[i + 1]; // position of next joint in current frame
                torque[i] = vec3_add(
                    &vec3_add(&ti, &t_next_in_i),
                    &vec3_add(
                        &cross3(&r_next_orig, &f_next_in_i),
                        &cross3(&self.links[i].com, &fi),
                    ),
                );
            }
        }

        // Joint torques: projection onto z-axis of each joint
        let mut tau = [S::ZERO; N];
        for i in 0..N {
            // z-axis of frame i (joint axis)
            tau[i] = torque[i][2]; // z-component
        }
        tau
    }

    /// Compute the inertia matrix M(q) via unit acceleration sweeps.
    pub fn mass_matrix(&self, dh: &[DhParam<S>; N], q: &[S; N]) -> [[S; N]; N] {
        let qd = [S::ZERO; N];
        let gravity = [S::ZERO; 3];
        let mut m_matrix = [[S::ZERO; N]; N];

        for col in 0..N {
            let mut qdd = [S::ZERO; N];
            qdd[col] = S::ONE;
            let tau = self.inverse_dynamics(dh, q, &qd, &qdd, gravity);
            for row in 0..N {
                m_matrix[row][col] = tau[row];
            }
        }
        m_matrix
    }
}

// ── 3D vector / matrix helpers ───────────────────────────────────────────────

fn cross3<S: ControlScalar>(a: &[S; 3], b: &[S; 3]) -> [S; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_add<S: ControlScalar>(a: &[S; 3], b: &[S; 3]) -> [S; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_scale<S: ControlScalar>(a: &[S; 3], s: S) -> [S; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn mat3_vec<S: ControlScalar>(m: &[[S; 3]; 3], v: &[S; 3]) -> [S; 3] {
    core::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn transpose3<S: ControlScalar>(m: &[[S; 3]; 3]) -> [[S; 3]; 3] {
    core::array::from_fn(|i| core::array::from_fn(|j| m[j][i]))
}

/// Extract 3×3 rotation part of DH transform.
fn dh_rot3<S: ControlScalar>(alpha: S, theta: S) -> [[S; 3]; 3] {
    let ct = theta.cos();
    let st = theta.sin();
    let ca = alpha.cos();
    let sa = alpha.sin();
    [
        [ct, -st * ca, st * sa],
        [st, ct * ca, -ct * sa],
        [S::ZERO, sa, ca],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::serial::six_dof::{robot6_ur5_like, DhParam};

    fn make_links_6() -> [LinkInertia<f64>; 6] {
        core::array::from_fn(|_| LinkInertia::point_mass(1.0, [0.0, 0.0, 0.1]))
    }

    fn ur5_dh() -> [DhParam<f64>; 6] {
        robot6_ur5_like().links
    }

    #[test]
    fn inverse_dynamics_gravity_only() {
        let links = make_links_6();
        let dyn6 = SerialDynamics::new(links);
        let dh = ur5_dh();
        let q = [0.0_f64; 6];
        let qd = [0.0_f64; 6];
        let qdd = [0.0_f64; 6];
        let gravity = [0.0, 0.0, -9.81];
        let tau = dyn6.inverse_dynamics(&dh, &q, &qd, &qdd, gravity);
        // Should produce finite torques
        for (i, &t) in tau.iter().enumerate() {
            assert!(t.is_finite(), "tau[{i}] = {t}");
        }
    }

    #[test]
    fn mass_matrix_finite() {
        let links = make_links_6();
        let dyn6 = SerialDynamics::new(links);
        let dh = ur5_dh();
        let q = [0.0_f64; 6];
        let m = dyn6.mass_matrix(&dh, &q);
        // All entries must be finite (no NaN/inf from degenerate geometry)
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(val.is_finite(), "M[{i}][{j}] is not finite: {}", val);
            }
        }
    }

    #[test]
    fn cross_product_correct() {
        let a = [1.0_f64, 0.0, 0.0];
        let b = [0.0_f64, 1.0, 0.0];
        let c = cross3(&a, &b);
        assert!((c[0]).abs() < 1e-15);
        assert!((c[1]).abs() < 1e-15);
        assert!((c[2] - 1.0).abs() < 1e-15);
    }
}
