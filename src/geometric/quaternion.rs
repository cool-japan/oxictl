use super::so3::{vec3_norm, SO3};
/// Unit quaternion attitude representation and kinematics.
///
/// A unit quaternion `q = [w, x, y, z]` with ‖q‖ = 1 represents an SO(3)
/// rotation equivalent to axis-angle (n̂, θ):
///   q = [cos(θ/2), sin(θ/2)·n̂]
///
/// The Hamilton product is used for composition; conjugation for inversion.
/// Quaternion kinematics:
///   q̇ = 0.5 · q ⊗ [0, ω]
use crate::core::scalar::ControlScalar;

/// Unit quaternion [w, x, y, z] with ‖q‖ = 1 invariant.
#[derive(Debug, Clone, Copy)]
pub struct UnitQuat<S: ControlScalar> {
    /// w component (scalar part).
    pub w: S,
    /// x component (first vector part).
    pub x: S,
    /// y component (second vector part).
    pub y: S,
    /// z component (third vector part).
    pub z: S,
}

// ─── Construction ────────────────────────────────────────────────────────────

impl<S: ControlScalar> UnitQuat<S> {
    /// Identity quaternion q = [1, 0, 0, 0] (zero rotation).
    #[inline]
    pub fn identity() -> Self {
        Self {
            w: S::ONE,
            x: S::ZERO,
            y: S::ZERO,
            z: S::ZERO,
        }
    }

    /// Construct from axis-angle.  `axis` is normalised internally.
    ///
    /// If ‖axis‖ < ε returns the identity quaternion.
    pub fn from_axis_angle(axis: [S; 3], angle: S) -> Self {
        let n = vec3_norm(axis);
        if n < S::EPSILON * S::from_f64(1e3) {
            return Self::identity();
        }
        let inv_n = S::ONE / n;
        let half = angle * S::HALF;
        let s = half.sin();
        let c = half.cos();
        Self {
            w: c,
            x: axis[0] * inv_n * s,
            y: axis[1] * inv_n * s,
            z: axis[2] * inv_n * s,
        }
    }

    /// Internal constructor (no normalisation check — caller must guarantee unit).
    #[inline]
    fn raw(w: S, x: S, y: S, z: S) -> Self {
        Self { w, x, y, z }
    }

    /// Re-normalise to compensate numerical drift.
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < S::EPSILON * S::from_f64(1e3) {
            return Self::identity();
        }
        let inv = S::ONE / n;
        Self {
            w: self.w * inv,
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }

    /// ‖q‖.
    #[inline]
    fn norm(&self) -> S {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

// ─── Group operations ─────────────────────────────────────────────────────────

impl<S: ControlScalar> UnitQuat<S> {
    /// Hamilton product: `self ⊗ other`.
    pub fn multiply(&self, other: &UnitQuat<S>) -> UnitQuat<S> {
        let (w1, x1, y1, z1) = (self.w, self.x, self.y, self.z);
        let (w2, x2, y2, z2) = (other.w, other.x, other.y, other.z);
        UnitQuat::raw(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    }

    /// Conjugate: q* = [w, -x, -y, -z] = q⁻¹ for unit quaternions.
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Rotate vector v using q ⊗ [0,v] ⊗ q*.
    pub fn rotate_vector(&self, v: [S; 3]) -> [S; 3] {
        // Efficient formula: v' = v + 2w(q_vec × v) + 2(q_vec × (q_vec × v))
        let qv = [self.x, self.y, self.z];
        let t = [
            S::TWO * (qv[1] * v[2] - qv[2] * v[1]),
            S::TWO * (qv[2] * v[0] - qv[0] * v[2]),
            S::TWO * (qv[0] * v[1] - qv[1] * v[0]),
        ];
        [
            v[0] + self.w * t[0] + (qv[1] * t[2] - qv[2] * t[1]),
            v[1] + self.w * t[1] + (qv[2] * t[0] - qv[0] * t[2]),
            v[2] + self.w * t[2] + (qv[0] * t[1] - qv[1] * t[0]),
        ]
    }

    /// Attitude error: q_e = q_d* ⊗ q.
    ///
    /// The vector part of q_e encodes the rotation error in the desired frame.
    pub fn error(&self, q_d: &UnitQuat<S>) -> UnitQuat<S> {
        q_d.conjugate().multiply(self)
    }

    /// Spherical linear interpolation (SLERP) from `self` to `other` at `t ∈ [0,1]`.
    pub fn slerp(&self, other: &UnitQuat<S>, t: S) -> UnitQuat<S> {
        let mut cos_half =
            self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        // Ensure shortest path
        let other_flip;
        let other_ref: &UnitQuat<S>;
        if cos_half < S::ZERO {
            other_flip = UnitQuat::raw(-other.w, -other.x, -other.y, -other.z);
            cos_half = -cos_half;
            other_ref = &other_flip;
        } else {
            other_ref = other;
        }

        let (s0, s1);
        if cos_half > S::from_f64(0.9995) {
            // Nearly parallel — use linear interpolation to avoid divide-by-zero
            s0 = S::ONE - t;
            s1 = t;
        } else {
            let half_theta = cos_half.clamp_val(S::from_f64(-1.0), S::ONE).acos();
            let sin_half = half_theta.sin();
            s0 = ((S::ONE - t) * half_theta).sin() / sin_half;
            s1 = (t * half_theta).sin() / sin_half;
        }

        UnitQuat::raw(
            s0 * self.w + s1 * other_ref.w,
            s0 * self.x + s1 * other_ref.x,
            s0 * self.y + s1 * other_ref.y,
            s0 * self.z + s1 * other_ref.z,
        )
    }
}

// ─── Conversion to/from SO3 ───────────────────────────────────────────────────

impl<S: ControlScalar> UnitQuat<S> {
    /// Convert this quaternion to an SO(3) rotation matrix.
    pub fn to_so3(&self) -> SO3<S> {
        let q = [self.w, self.x, self.y, self.z];
        // We know the quaternion is unit by invariant, so use unchecked
        SO3::from_quaternion(q).unwrap_or_else(|_| SO3::identity())
    }

    /// Build a unit quaternion from an SO(3) rotation matrix.
    pub fn from_so3(r: &SO3<S>) -> UnitQuat<S> {
        let q = r.to_quaternion();
        UnitQuat::raw(q[0], q[1], q[2], q[3])
    }

    /// Expose as [w, x, y, z] array.
    #[inline]
    pub fn as_array(&self) -> [S; 4] {
        [self.w, self.x, self.y, self.z]
    }
}

// ─── Quaternion kinematics ────────────────────────────────────────────────────

/// Quaternion kinematic integrator.
///
/// Integrates `q̇ = 0.5 · q ⊗ [0, ω]` over a small timestep `dt` using a
/// first-order matrix exponential approximation that preserves unit-norm better
/// than naive Euler integration.
pub struct QuatKinematics;

impl QuatKinematics {
    /// Integrate attitude one step.
    ///
    /// `q`     — current unit quaternion (body orientation)
    /// `omega` — angular velocity in body frame [rad/s]
    /// `dt`    — timestep [s]
    ///
    /// Returns the updated quaternion, re-normalised to resist drift.
    pub fn integrate<S: ControlScalar>(q: &UnitQuat<S>, omega: [S; 3], dt: S) -> UnitQuat<S> {
        let theta = vec3_norm(omega) * dt;
        if theta < S::from_f64(1e-10) {
            return q.normalize();
        }

        // Exact integration via rotation quaternion for the rotation ω*dt:
        //   axis = ω/‖ω‖,  angle = ‖ω‖·dt = theta
        // Δq = [cos(theta/2), sin(theta/2)·axis]
        // q(t+dt) = q(t) ⊗ Δq   (body-frame right-multiplication)
        let delta = UnitQuat::from_axis_angle(omega, theta);
        q.multiply(&delta).normalize()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;
    const EPS_MED: f64 = 1e-7;

    fn nearly_equal(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn quat_nearly_equal(a: &UnitQuat<f64>, b: &UnitQuat<f64>, tol: f64) -> bool {
        // Allow sign flip (q and -q represent the same rotation)
        let same = nearly_equal(a.w, b.w, tol)
            && nearly_equal(a.x, b.x, tol)
            && nearly_equal(a.y, b.y, tol)
            && nearly_equal(a.z, b.z, tol);
        let flip = nearly_equal(a.w, -b.w, tol)
            && nearly_equal(a.x, -b.x, tol)
            && nearly_equal(a.y, -b.y, tol)
            && nearly_equal(a.z, -b.z, tol);
        same || flip
    }

    #[test]
    fn identity_rotate_vector() {
        let q = UnitQuat::<f64>::identity();
        let v = [1.0, 2.0, 3.0];
        let rv = q.rotate_vector(v);
        for i in 0..3 {
            assert!(
                nearly_equal(rv[i], v[i], EPS),
                "rv[{}]={} v[{}]={}",
                i,
                rv[i],
                i,
                v[i]
            );
        }
    }

    #[test]
    fn multiply_associativity() {
        let q1 = UnitQuat::<f64>::from_axis_angle([1.0, 0.0, 0.0], 0.5);
        let q2 = UnitQuat::<f64>::from_axis_angle([0.0, 1.0, 0.0], 0.8);
        let q3 = UnitQuat::<f64>::from_axis_angle([0.0, 0.0, 1.0], 1.2);
        let left = q1.multiply(&q2).multiply(&q3);
        let right = q1.multiply(&q2.multiply(&q3));
        assert!(quat_nearly_equal(&left, &right, EPS_MED));
    }

    #[test]
    fn rotate_then_un_rotate() {
        let axis = [1.0_f64 / 3.0_f64.sqrt(); 3];
        let angle = 1.1_f64;
        let q = UnitQuat::<f64>::from_axis_angle(axis, angle);
        let q_c = q.conjugate();
        let v = [2.0, -1.0, 3.0];
        let rv = q.rotate_vector(v);
        let v2 = q_c.rotate_vector(rv);
        for i in 0..3 {
            assert!(
                nearly_equal(v2[i], v[i], EPS_MED),
                "v2[{}]={} v[{}]={}",
                i,
                v2[i],
                i,
                v[i]
            );
        }
    }

    #[test]
    fn conjugate_is_inverse() {
        let q = UnitQuat::<f64>::from_axis_angle([0.0, 1.0, 0.0], 1.0);
        let qc = q.conjugate();
        let eye = q.multiply(&qc);
        assert!(nearly_equal(eye.w, 1.0, EPS_MED));
        assert!(nearly_equal(eye.x, 0.0, EPS_MED));
        assert!(nearly_equal(eye.y, 0.0, EPS_MED));
        assert!(nearly_equal(eye.z, 0.0, EPS_MED));
    }

    #[test]
    fn to_so3_from_so3_roundtrip() {
        let q1 = UnitQuat::<f64>::from_axis_angle([1.0, 1.0, 0.0], 0.9).normalize();
        let r = q1.to_so3();
        let q2 = UnitQuat::<f64>::from_so3(&r);
        assert!(quat_nearly_equal(&q1, &q2, 1e-9));
    }

    #[test]
    fn slerp_endpoints() {
        let q1 = UnitQuat::<f64>::from_axis_angle([0.0, 0.0, 1.0], 0.0);
        let q2 = UnitQuat::<f64>::from_axis_angle([0.0, 0.0, 1.0], 1.0);
        let s0 = q1.slerp(&q2, 0.0);
        let s1 = q1.slerp(&q2, 1.0);
        assert!(quat_nearly_equal(&s0, &q1, EPS_MED));
        assert!(quat_nearly_equal(&s1, &q2, EPS_MED));
    }

    #[test]
    fn kinematics_integrate_small_step() {
        // Integrate a 90° rotation about z-axis in 1000 steps of 90°/1000
        let n = 1000_usize;
        let total_angle = core::f64::consts::FRAC_PI_2;
        let omega_z = total_angle; // 1 rad/s * 1 s total
        let dt = 1.0 / (n as f64);
        let mut q = UnitQuat::<f64>::identity();
        for _ in 0..n {
            q = QuatKinematics::integrate(&q, [0.0, 0.0, omega_z], dt);
        }
        // Expected: q = [cos(45°), 0, 0, sin(45°)]
        let expected_w = (total_angle / 2.0).cos();
        let expected_z = (total_angle / 2.0).sin();
        let expected = UnitQuat::<f64>::raw(expected_w, 0.0, 0.0, expected_z);
        assert!(
            quat_nearly_equal(&q, &expected, 1e-4),
            "q=[{},{},{},{}] expected=[{},{},{},{}]",
            q.w,
            q.x,
            q.y,
            q.z,
            expected_w,
            0.0,
            0.0,
            expected_z
        );
    }

    #[test]
    fn normalize_corrects_drift() {
        let mut q = UnitQuat::<f64>::from_axis_angle([1.0, 0.0, 0.0], 0.5);
        // Inject some drift
        q.w += 0.1;
        let qn = q.normalize();
        let norm_sq = qn.w * qn.w + qn.x * qn.x + qn.y * qn.y + qn.z * qn.z;
        assert!(nearly_equal(norm_sq, 1.0, EPS_MED));
    }
}
