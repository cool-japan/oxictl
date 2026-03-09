use super::so3::SO3;
/// SE(3) — Special Euclidean Group of rigid body transforms.
///
/// SE(3) = SO(3) ⋉ ℝ³ is the group of orientation-preserving isometries of ℝ³.
/// An element g = (R, t) acts on a point p as g·p = R·p + t.
///
/// The adjoint representation maps twists/wrenches between frames.
use crate::core::scalar::ControlScalar;

// ─── SE(3) element ────────────────────────────────────────────────────────────

/// Rigid body transform: rotation R ∈ SO(3) and translation t ∈ ℝ³.
///
/// Composition: (R₁,t₁)·(R₂,t₂) = (R₁·R₂, R₁·t₂ + t₁)
/// Inverse:     (R,t)⁻¹ = (Rᵀ, −Rᵀ·t)
#[derive(Debug, Clone, Copy)]
pub struct SE3<S: ControlScalar> {
    /// Rotational part.
    pub rotation: SO3<S>,
    /// Translational part (origin of body frame in world frame).
    pub translation: [S; 3],
}

impl<S: ControlScalar> SE3<S> {
    /// Identity transform (R=I, t=0).
    pub fn identity() -> Self {
        Self {
            rotation: SO3::identity(),
            translation: [S::ZERO; 3],
        }
    }

    /// Construct from an SO3 rotation and a translation vector.
    pub fn from_rt(r: SO3<S>, t: [S; 3]) -> Self {
        Self {
            rotation: r,
            translation: t,
        }
    }

    /// Compose two transforms: `self ∘ other`.
    ///
    /// (R₁,t₁)·(R₂,t₂) = (R₁·R₂, R₁·t₂ + t₁)
    pub fn multiply(&self, other: &SE3<S>) -> SE3<S> {
        let r_new = self.rotation.multiply(&other.rotation);
        let t_new = mat3_vec3_add(self.rotation.apply(other.translation), self.translation);
        SE3 {
            rotation: r_new,
            translation: t_new,
        }
    }

    /// Inverse transform.
    ///
    /// (R,t)⁻¹ = (Rᵀ, −Rᵀ·t)
    pub fn inverse(&self) -> SE3<S> {
        let rt = self.rotation.transpose();
        let t_inv = rt.apply(self.translation);
        SE3 {
            rotation: rt,
            translation: [-t_inv[0], -t_inv[1], -t_inv[2]],
        }
    }

    /// Apply transform to a point: p' = R·p + t.
    pub fn apply(&self, p: [S; 3]) -> [S; 3] {
        mat3_vec3_add(self.rotation.apply(p), self.translation)
    }

    /// 6×6 adjoint matrix `Ad_g` for transforming twists and wrenches.
    ///
    /// For a twist (v, ω) in the body frame of `other`, the twist in the
    /// frame of `self` is `Ad_{g} · [v; ω]` where g = self.
    ///
    /// Layout (row-major 6×6, [0..3] = linear, [3..6] = angular):
    /// ```text
    ///   Ad_g = [ R   t×R ]
    ///          [ 0   R   ]
    /// ```
    /// where `t×R` is the cross-product matrix of t times R.
    pub fn adjoint(&self) -> [[S; 6]; 6] {
        let r = self.rotation.mat;
        let t = self.translation;

        // t× (skew-symmetric matrix of t)
        let tx = [
            [S::ZERO, -t[2], t[1]],
            [t[2], S::ZERO, -t[0]],
            [-t[1], t[0], S::ZERO],
        ];

        // t× · R  (3×3)
        let mut txr = [[S::ZERO; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    txr[i][j] += tx[i][k] * r[k][j];
                }
            }
        }

        // Assemble 6×6
        let mut adj = [[S::ZERO; 6]; 6];
        // Top-left: R
        for i in 0..3 {
            for j in 0..3 {
                adj[i][j] = r[i][j];
            }
        }
        // Top-right: t×R
        for i in 0..3 {
            for j in 0..3 {
                adj[i][j + 3] = txr[i][j];
            }
        }
        // Bottom-right: R
        for i in 0..3 {
            for j in 0..3 {
                adj[i + 3][j + 3] = r[i][j];
            }
        }
        // Bottom-left: 0 (already zero)
        adj
    }
}

// ─── Twist & Wrench ───────────────────────────────────────────────────────────

/// A 6-D velocity screw (twist) [v, ω] ∈ ℝ⁶.
///
/// `v` = linear velocity, `ω` = angular velocity.
#[derive(Debug, Clone, Copy)]
pub struct Twist<S: ControlScalar> {
    /// Linear velocity component [m/s].
    pub v: [S; 3],
    /// Angular velocity component [rad/s].
    pub omega: [S; 3],
}

impl<S: ControlScalar> Twist<S> {
    /// Zero twist.
    pub fn zero() -> Self {
        Self {
            v: [S::ZERO; 3],
            omega: [S::ZERO; 3],
        }
    }

    /// Construct from linear and angular velocity arrays.
    pub fn new(v: [S; 3], omega: [S; 3]) -> Self {
        Self { v, omega }
    }

    /// Represent as a 6-element array [v0,v1,v2,ω0,ω1,ω2].
    pub fn as_array(&self) -> [S; 6] {
        [
            self.v[0],
            self.v[1],
            self.v[2],
            self.omega[0],
            self.omega[1],
            self.omega[2],
        ]
    }

    /// Construct from a 6-element array.
    pub fn from_array(a: [S; 6]) -> Self {
        Self {
            v: [a[0], a[1], a[2]],
            omega: [a[3], a[4], a[5]],
        }
    }
}

/// A 6-D force screw (wrench) [f, τ] ∈ ℝ⁶.
///
/// `f` = force [N], `tau` = torque [N·m].
#[derive(Debug, Clone, Copy)]
pub struct Wrench<S: ControlScalar> {
    /// Force component [N].
    pub force: [S; 3],
    /// Torque component [N·m].
    pub tau: [S; 3],
}

impl<S: ControlScalar> Wrench<S> {
    /// Zero wrench.
    pub fn zero() -> Self {
        Self {
            force: [S::ZERO; 3],
            tau: [S::ZERO; 3],
        }
    }

    /// Construct from force and torque arrays.
    pub fn new(force: [S; 3], tau: [S; 3]) -> Self {
        Self { force, tau }
    }

    /// Represent as a 6-element array [f0,f1,f2,τ0,τ1,τ2].
    pub fn as_array(&self) -> [S; 6] {
        [
            self.force[0],
            self.force[1],
            self.force[2],
            self.tau[0],
            self.tau[1],
            self.tau[2],
        ]
    }
}

// ─── Adjoint wrench transform ─────────────────────────────────────────────────

/// Transform a wrench from the body frame of `g` to the world frame.
///
/// The wrench dual transform is Ad_g^{-T}·w, which for the wrench (force/torque)
/// maps as:
///   f_world = R · f_body
///   τ_world = R · τ_body + t × (R · f_body)
pub fn transform_wrench<S: ControlScalar>(g: &SE3<S>, w: &Wrench<S>) -> Wrench<S> {
    let f_w = g.rotation.apply(w.force);
    let tau_from_r = g.rotation.apply(w.tau);
    let t = g.translation;
    // t × f_w
    let t_cross_f = [
        t[1] * f_w[2] - t[2] * f_w[1],
        t[2] * f_w[0] - t[0] * f_w[2],
        t[0] * f_w[1] - t[1] * f_w[0],
    ];
    Wrench {
        force: f_w,
        tau: [
            tau_from_r[0] + t_cross_f[0],
            tau_from_r[1] + t_cross_f[1],
            tau_from_r[2] + t_cross_f[2],
        ],
    }
}

/// Transform a twist using the adjoint Ad_g.
///
/// For a twist expressed in frame A, compute the equivalent twist in frame B
/// where g = T_{BA} (transform from A to B):
///   ξ_B = Ad_g · ξ_A
pub fn transform_twist<S: ControlScalar>(g: &SE3<S>, tw: &Twist<S>) -> Twist<S> {
    let adj = g.adjoint();
    let arr = tw.as_array();
    let mut out = [S::ZERO; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i] += adj[i][j] * arr[j];
        }
    }
    Twist::from_array(out)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Element-wise add of two 3-vectors.
#[inline]
fn mat3_vec3_add<S: ControlScalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;
    const EPS_MED: f64 = 1e-8;

    fn assert_se3_nearly_equal(a: &SE3<f64>, b: &SE3<f64>, tol: f64) {
        for i in 0..3 {
            for j in 0..3 {
                let diff = (a.rotation.mat[i][j] - b.rotation.mat[i][j]).abs();
                assert!(
                    diff < tol,
                    "R[{},{}]: {} vs {}",
                    i,
                    j,
                    a.rotation.mat[i][j],
                    b.rotation.mat[i][j]
                );
            }
            let diff = (a.translation[i] - b.translation[i]).abs();
            assert!(
                diff < tol,
                "t[{}]: {} vs {}",
                i,
                a.translation[i],
                b.translation[i]
            );
        }
    }

    #[test]
    fn identity_apply_point() {
        let g = SE3::<f64>::identity();
        let p = [1.0, 2.0, 3.0];
        let q = g.apply(p);
        for i in 0..3 {
            assert!((q[i] - p[i]).abs() < EPS);
        }
    }

    #[test]
    fn inverse_compose_identity() {
        let r = SO3::<f64>::from_axis_angle([0.0, 0.0, 1.0], 0.7).unwrap();
        let g = SE3::from_rt(r, [1.0, 2.0, 3.0]);
        let gi = g.inverse();
        let eye = g.multiply(&gi);
        let expected = SE3::<f64>::identity();
        assert_se3_nearly_equal(&eye, &expected, EPS_MED);
    }

    #[test]
    fn compose_inverse_identity() {
        let r = SO3::<f64>::from_axis_angle([1.0, 0.0, 0.0], 1.0).unwrap();
        let g = SE3::from_rt(r, [-1.0, 0.5, 2.0]);
        let gi = g.inverse();
        let eye = gi.multiply(&g);
        let expected = SE3::<f64>::identity();
        assert_se3_nearly_equal(&eye, &expected, EPS_MED);
    }

    #[test]
    fn group_associativity() {
        let r1 = SO3::<f64>::from_axis_angle([1.0, 0.0, 0.0], 0.3).unwrap();
        let r2 = SO3::<f64>::from_axis_angle([0.0, 1.0, 0.0], 0.5).unwrap();
        let r3 = SO3::<f64>::from_axis_angle([0.0, 0.0, 1.0], 0.7).unwrap();
        let g1 = SE3::from_rt(r1, [1.0, 0.0, 0.0]);
        let g2 = SE3::from_rt(r2, [0.0, 1.0, 0.0]);
        let g3 = SE3::from_rt(r3, [0.0, 0.0, 1.0]);
        let left = g1.multiply(&g2).multiply(&g3);
        let right = g1.multiply(&g2.multiply(&g3));
        assert_se3_nearly_equal(&left, &right, EPS_MED);
    }

    #[test]
    fn apply_pure_translation() {
        let g = SE3::from_rt(SO3::<f64>::identity(), [5.0, 0.0, 0.0]);
        let p = [1.0, 2.0, 3.0];
        let q = g.apply(p);
        assert!((q[0] - 6.0).abs() < EPS);
        assert!((q[1] - 2.0).abs() < EPS);
        assert!((q[2] - 3.0).abs() < EPS);
    }

    #[test]
    fn apply_pure_rotation() {
        // 90° rotation about z — maps [1,0,0] to [0,1,0]
        let r = SO3::<f64>::from_axis_angle([0.0, 0.0, 1.0], core::f64::consts::FRAC_PI_2).unwrap();
        let g = SE3::from_rt(r, [0.0; 3]);
        let p = [1.0, 0.0, 0.0];
        let q = g.apply(p);
        assert!((q[0]).abs() < 1e-10);
        assert!((q[1] - 1.0).abs() < 1e-10);
        assert!((q[2]).abs() < 1e-10);
    }

    #[test]
    fn wrench_transform_identity() {
        let g = SE3::<f64>::identity();
        let w = Wrench::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        let w2 = transform_wrench(&g, &w);
        for i in 0..3 {
            assert!((w2.force[i] - w.force[i]).abs() < EPS);
            assert!((w2.tau[i] - w.tau[i]).abs() < EPS);
        }
    }

    #[test]
    fn adjoint_6x6_structure() {
        // For pure rotation, top-left and bottom-right 3×3 should equal R
        // and top-right should be zero (t=0).
        let r = SO3::<f64>::from_axis_angle([0.0, 0.0, 1.0], 0.5).unwrap();
        let g = SE3::from_rt(r, [0.0; 3]);
        let adj = g.adjoint();
        // Top-left == R
        for (i, adj_row) in adj.iter().enumerate().take(3) {
            for (j, &val) in adj_row.iter().enumerate().take(3) {
                assert!((val - r.mat[i][j]).abs() < EPS, "adj TL[{},{}]", i, j);
            }
        }
        // Top-right == 0 (t=0)
        for (i, adj_row) in adj.iter().enumerate().take(3) {
            for (j, &val) in adj_row.iter().enumerate().skip(3).take(3) {
                assert!(val.abs() < EPS, "adj TR[{},{}] = {}", i, j - 3, val);
            }
        }
        // Bottom-left == 0
        for (i, adj_row) in adj.iter().enumerate().skip(3).take(3) {
            for (j, &val) in adj_row.iter().enumerate().take(3) {
                assert!(val.abs() < EPS, "adj BL[{},{}] = {}", i - 3, j, val);
            }
        }
        // Bottom-right == R
        for (i, adj_row) in adj.iter().enumerate().skip(3).take(3) {
            for (j, &val) in adj_row.iter().enumerate().skip(3).take(3) {
                assert!(
                    (val - r.mat[i - 3][j - 3]).abs() < EPS,
                    "adj BR[{},{}]",
                    i - 3,
                    j - 3
                );
            }
        }
    }
}
