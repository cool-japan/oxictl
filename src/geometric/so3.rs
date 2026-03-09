use super::GeoError;
/// SO(3) — Special Orthogonal Group (3D rotations).
///
/// Represents the Lie group SO(3) = {R ∈ ℝ³×³ : RᵀR = I, det(R) = 1}.
/// The Lie algebra so(3) consists of skew-symmetric 3×3 matrices, identified
/// with ℝ³ via the hat/vee isomorphism.
///
/// The exponential map is implemented via the Rodrigues formula:
///   exp(ω×) = I + sin(θ)·ω̂× + (1-cos(θ))·ω̂×²
/// where θ = ‖ω‖ and ω̂ = ω/θ.
use crate::core::scalar::ControlScalar;

/// Rotation matrix element (row, col) indexing helpers.
///
/// Internal storage: `mat[row][col]`, row-major.
#[derive(Debug, Clone, Copy)]
pub struct SO3<S: ControlScalar> {
    /// Rotation matrix stored row-major: `mat[row][col]`.
    pub(crate) mat: [[S; 3]; 3],
}

// ─── Construction ────────────────────────────────────────────────────────────

impl<S: ControlScalar> SO3<S> {
    /// Identity rotation R = I₃.
    #[inline]
    pub fn identity() -> Self {
        Self {
            mat: [
                [S::ONE, S::ZERO, S::ZERO],
                [S::ZERO, S::ONE, S::ZERO],
                [S::ZERO, S::ZERO, S::ONE],
            ],
        }
    }

    /// Construct from a raw 3×3 row-major array **without** enforcing SO(3)
    /// invariants.  Use only when the caller guarantees orthogonality and det=+1.
    #[inline]
    pub fn from_matrix_unchecked(mat: [[S; 3]; 3]) -> Self {
        Self { mat }
    }

    /// Build an SO(3) element from a unit axis and angle via the Rodrigues formula.
    ///
    /// `axis` need not be unit-length; it is normalised internally.
    /// Returns `Err(GeoError::Singular)` if `‖axis‖ < ε`.
    pub fn from_axis_angle(axis: [S; 3], angle: S) -> Result<Self, GeoError> {
        let n = vec3_norm(axis);
        if n < S::EPSILON * S::from_f64(1e3) {
            return Err(GeoError::Singular);
        }
        let inv_n = S::ONE / n;
        let u = [axis[0] * inv_n, axis[1] * inv_n, axis[2] * inv_n];

        let s = angle.sin();
        let c = angle.cos();
        let t = S::ONE - c; // 1 - cos(θ)

        // R = cos(θ)·I + (1-cos(θ))·u·uᵀ + sin(θ)·u×
        // Expanded entry by entry:
        let mat = [
            [
                t * u[0] * u[0] + c,
                t * u[0] * u[1] - s * u[2],
                t * u[0] * u[2] + s * u[1],
            ],
            [
                t * u[1] * u[0] + s * u[2],
                t * u[1] * u[1] + c,
                t * u[1] * u[2] - s * u[0],
            ],
            [
                t * u[2] * u[0] - s * u[1],
                t * u[2] * u[1] + s * u[0],
                t * u[2] * u[2] + c,
            ],
        ];
        Ok(Self { mat })
    }

    /// Build SO(3) from ZYX Euler angles (roll φ, pitch θ, yaw ψ).
    ///
    /// Convention: R = Rz(ψ)·Ry(θ)·Rx(φ)
    pub fn from_euler_zyx(roll: S, pitch: S, yaw: S) -> Self {
        let (sr, cr) = (roll.sin(), roll.cos());
        let (sp, cp) = (pitch.sin(), pitch.cos());
        let (sy, cy) = (yaw.sin(), yaw.cos());

        Self {
            mat: [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
        }
    }

    /// Build SO(3) from a unit quaternion `q = [w, x, y, z]`.
    ///
    /// Returns `Err(GeoError::InvalidRotation)` if `‖q‖` is not near 1.
    pub fn from_quaternion(q: [S; 4]) -> Result<Self, GeoError> {
        let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        if (norm_sq - S::ONE).abs() > S::from_f64(1e-3) {
            return Err(GeoError::InvalidRotation);
        }
        let w = q[0];
        let x = q[1];
        let y = q[2];
        let z = q[3];
        let two = S::TWO;
        let mat = [
            [
                S::ONE - two * (y * y + z * z),
                two * (x * y - w * z),
                two * (x * z + w * y),
            ],
            [
                two * (x * y + w * z),
                S::ONE - two * (x * x + z * z),
                two * (y * z - w * x),
            ],
            [
                two * (x * z - w * y),
                two * (y * z + w * x),
                S::ONE - two * (x * x + y * y),
            ],
        ];
        Ok(Self { mat })
    }
}

// ─── Extraction ──────────────────────────────────────────────────────────────

impl<S: ControlScalar> SO3<S> {
    /// Extract ZYX Euler angles [roll, pitch, yaw] from the rotation matrix.
    ///
    /// Pitch is clamped to [-π/2, π/2] via arcsin; gimbal-lock regions (|pitch| ≈ π/2)
    /// will produce numerically degraded roll/yaw.
    pub fn to_euler_zyx(&self) -> [S; 3] {
        let r = &self.mat;
        // r[2][0] = -sin(pitch)
        let sp = -r[2][0];
        let pitch = sp.clamp_val(S::from_f64(-1.0), S::ONE).asin();

        let cp = pitch.cos();
        let (roll, yaw);
        if cp.abs() < S::from_f64(1e-6) {
            // Gimbal lock: set roll=0, recover yaw from r[0][1]/r[1][1].
            roll = S::ZERO;
            yaw = r[1][1].atan2(r[0][1]);
        } else {
            roll = r[2][1].atan2(r[2][2]);
            yaw = r[1][0].atan2(r[0][0]);
        }
        [roll, pitch, yaw]
    }

    /// Convert to unit quaternion [w, x, y, z].
    ///
    /// Uses Shepperd's method for numerical stability.
    pub fn to_quaternion(&self) -> [S; 4] {
        let r = &self.mat;
        let trace = r[0][0] + r[1][1] + r[2][2];
        let quarter = S::from_f64(0.25);

        if trace > S::ZERO {
            let s = (trace + S::ONE).sqrt() * S::TWO; // 4w
            let inv_s = S::ONE / s;
            let w = quarter * s;
            let x = (r[2][1] - r[1][2]) * inv_s;
            let y = (r[0][2] - r[2][0]) * inv_s;
            let z = (r[1][0] - r[0][1]) * inv_s;
            [w, x, y, z]
        } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
            let s = (S::ONE + r[0][0] - r[1][1] - r[2][2]).sqrt() * S::TWO; // 4x
            let inv_s = S::ONE / s;
            let w = (r[2][1] - r[1][2]) * inv_s;
            let x = quarter * s;
            let y = (r[0][1] + r[1][0]) * inv_s;
            let z = (r[0][2] + r[2][0]) * inv_s;
            [w, x, y, z]
        } else if r[1][1] > r[2][2] {
            let s = (S::ONE + r[1][1] - r[0][0] - r[2][2]).sqrt() * S::TWO; // 4y
            let inv_s = S::ONE / s;
            let w = (r[0][2] - r[2][0]) * inv_s;
            let x = (r[0][1] + r[1][0]) * inv_s;
            let y = quarter * s;
            let z = (r[1][2] + r[2][1]) * inv_s;
            [w, x, y, z]
        } else {
            let s = (S::ONE + r[2][2] - r[0][0] - r[1][1]).sqrt() * S::TWO; // 4z
            let inv_s = S::ONE / s;
            let w = (r[1][0] - r[0][1]) * inv_s;
            let x = (r[0][2] + r[2][0]) * inv_s;
            let y = (r[1][2] + r[2][1]) * inv_s;
            let z = quarter * s;
            [w, x, y, z]
        }
    }

    /// Extract axis-angle representation `([axis; 3], angle)`.
    ///
    /// For the identity (angle ≈ 0) the axis is returned as [1,0,0] by convention.
    pub fn to_axis_angle(&self) -> ([S; 3], S) {
        let omega = self.log();
        let theta = vec3_norm(omega);
        if theta < S::EPSILON * S::from_f64(1e3) {
            return ([S::ONE, S::ZERO, S::ZERO], S::ZERO);
        }
        let inv = S::ONE / theta;
        ([omega[0] * inv, omega[1] * inv, omega[2] * inv], theta)
    }

    /// Return the underlying row-major matrix `[[S;3];3]`.
    #[inline]
    pub fn as_matrix(&self) -> [[S; 3]; 3] {
        self.mat
    }
}

// ─── Group operations ─────────────────────────────────────────────────────────

impl<S: ControlScalar> SO3<S> {
    /// Compose two rotations: `R_self · R_other`.
    pub fn multiply(&self, other: &SO3<S>) -> SO3<S> {
        let a = &self.mat;
        let b = &other.mat;
        let mut c = [[S::ZERO; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        SO3 { mat: c }
    }

    /// Inverse / transpose: R⁻¹ = Rᵀ for SO(3).
    pub fn transpose(&self) -> SO3<S> {
        let r = &self.mat;
        SO3 {
            mat: [
                [r[0][0], r[1][0], r[2][0]],
                [r[0][1], r[1][1], r[2][1]],
                [r[0][2], r[1][2], r[2][2]],
            ],
        }
    }

    /// Rotate a vector: `R · v`.
    pub fn apply(&self, v: [S; 3]) -> [S; 3] {
        let r = &self.mat;
        [
            r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
        ]
    }
}

// ─── Lie algebra (so(3) ↔ ℝ³) ────────────────────────────────────────────────

impl<S: ControlScalar> SO3<S> {
    /// Logarithmic map: SO(3) → so(3) ≅ ℝ³.
    ///
    /// Returns the rotation vector ω such that `exp(hat(ω)) = R`.
    /// For R ≈ I the result is ≈ 0.  Uses the standard formula:
    ///   θ  = arccos((trace(R)-1)/2)
    ///   ω  = θ/(2·sin(θ)) · vee(R - Rᵀ)
    pub fn log(&self) -> [S; 3] {
        let r = &self.mat;
        let trace = r[0][0] + r[1][1] + r[2][2];
        // Clamp to valid arccos range
        let cos_theta = ((trace - S::ONE) * S::HALF).clamp_val(S::from_f64(-1.0), S::ONE);
        let theta = cos_theta.acos();

        if theta.abs() < S::from_f64(1e-7) {
            // Near identity — first-order approximation
            return [
                S::HALF * (r[2][1] - r[1][2]),
                S::HALF * (r[0][2] - r[2][0]),
                S::HALF * (r[1][0] - r[0][1]),
            ];
        }
        if (theta - S::PI).abs() < S::from_f64(1e-4) {
            // Near π — use diagonal elements
            let half_pi = S::from_f64(core::f64::consts::FRAC_PI_2);
            let _ = half_pi;
            // R + I = 2·(1+cos θ)·(v·vᵀ) so v = sqrt((r_ii+1)/2)
            let vx = ((r[0][0] + S::ONE) * S::HALF).max(S::ZERO).sqrt();
            let vy_sign = if r[0][1] >= S::ZERO { S::ONE } else { -S::ONE };
            let vz_sign = if r[0][2] >= S::ZERO { S::ONE } else { -S::ONE };
            let vy = ((r[1][1] + S::ONE) * S::HALF).max(S::ZERO).sqrt() * vy_sign;
            let vz = ((r[2][2] + S::ONE) * S::HALF).max(S::ZERO).sqrt() * vz_sign;
            return [vx * S::PI, vy * S::PI, vz * S::PI];
        }

        let factor = theta / (S::TWO * theta.sin());
        [
            factor * (r[2][1] - r[1][2]),
            factor * (r[0][2] - r[2][0]),
            factor * (r[1][0] - r[0][1]),
        ]
    }

    /// Exponential map: ℝ³ → SO(3).
    ///
    /// Computes R = exp(ω×) via the Rodrigues formula:
    ///   R = I + sin(θ)·ω̂× + (1-cos(θ))·ω̂×²
    /// where θ = ‖ω‖ and ω̂ = ω/θ.
    pub fn exp(omega: [S; 3]) -> SO3<S> {
        let theta = vec3_norm(omega);
        if theta < S::from_f64(1e-10) {
            return SO3::identity();
        }
        // This is the same as from_axis_angle but with guaranteed finite norm.
        SO3::from_axis_angle(omega, theta).unwrap_or_else(|_| SO3::identity())
    }
}

// ─── Utility free functions ───────────────────────────────────────────────────

/// Skew-symmetric (hat) operator: ω → ω× ∈ so(3).
///
/// Returns the 3×3 matrix `[[0,-ωz,ωy],[ωz,0,-ωx],[-ωy,ωx,0]]`.
pub fn hat<S: ControlScalar>(v: [S; 3]) -> [[S; 3]; 3] {
    let z = S::ZERO;
    [[z, -v[2], v[1]], [v[2], z, -v[0]], [-v[1], v[0], z]]
}

/// Vee operator (inverse of hat): extracts ω from a skew-symmetric matrix.
///
/// For a matrix A the result is [A[2][1], A[0][2], A[1][0]].
pub fn vee<S: ControlScalar>(m: [[S; 3]; 3]) -> [S; 3] {
    [m[2][1], m[0][2], m[1][0]]
}

/// Attitude error on SO(3) (Lee et al. 2010):
///   e_R = 0.5 · vee(R_dᵀ·R − Rᵀ·R_d)
///
/// `R_d` is the desired rotation, `R` is the current rotation.
pub fn rotation_error<S: ControlScalar>(r_d: &SO3<S>, r: &SO3<S>) -> [S; 3] {
    // Rdt = R_dᵀ,  Rt = Rᵀ
    let rdt = r_d.transpose();
    let rt = r.transpose();

    // A = R_dᵀ·R  (desired-to-current)
    let a = rdt.multiply(r);
    // B = Rᵀ·R_d  (current-to-desired)
    let b = rt.multiply(r_d);

    // A - B (skew part, both are skew-symmetric individually when A=Bᵀ)
    let diff = [
        [
            a.mat[0][0] - b.mat[0][0],
            a.mat[0][1] - b.mat[0][1],
            a.mat[0][2] - b.mat[0][2],
        ],
        [
            a.mat[1][0] - b.mat[1][0],
            a.mat[1][1] - b.mat[1][1],
            a.mat[1][2] - b.mat[1][2],
        ],
        [
            a.mat[2][0] - b.mat[2][0],
            a.mat[2][1] - b.mat[2][1],
            a.mat[2][2] - b.mat[2][2],
        ],
    ];

    let raw = vee(diff);
    [raw[0] * S::HALF, raw[1] * S::HALF, raw[2] * S::HALF]
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Euclidean norm of a 3-vector.
#[inline]
pub(crate) fn vec3_norm<S: ControlScalar>(v: [S; 3]) -> S {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Cross product of two 3-vectors.
#[inline]
pub(crate) fn vec3_cross<S: ControlScalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-vectors.
#[inline]
pub(crate) fn vec3_dot<S: ControlScalar>(a: [S; 3], b: [S; 3]) -> S {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;
    const EPS_MED: f64 = 1e-7;

    // Check that a matrix is orthogonal (RᵀR = I) within tolerance.
    fn assert_orthogonal(r: &SO3<f64>, tol: f64) {
        let rt = r.transpose();
        let prod = rt.multiply(r);
        let eye = SO3::<f64>::identity();
        for i in 0..3 {
            for j in 0..3 {
                let diff = (prod.mat[i][j] - eye.mat[i][j]).abs();
                assert!(
                    diff < tol,
                    "RᵀR[{},{}] = {:.2e}, expected {}",
                    i,
                    j,
                    prod.mat[i][j],
                    eye.mat[i][j]
                );
            }
        }
    }

    #[test]
    fn identity_is_identity() {
        let r = SO3::<f64>::identity();
        let v = [1.0, 2.0, 3.0];
        let rv = r.apply(v);
        for i in 0..3 {
            assert!((rv[i] - v[i]).abs() < EPS);
        }
    }

    #[test]
    fn identity_orthogonal() {
        let r = SO3::<f64>::identity();
        assert_orthogonal(&r, EPS);
    }

    #[test]
    fn axis_angle_orthogonal() {
        let r = SO3::<f64>::from_axis_angle([0.0, 0.0, 1.0], 1.2).unwrap();
        assert_orthogonal(&r, EPS);
    }

    #[test]
    fn rodrigues_euler_roundtrip() {
        let roll = 0.3_f64;
        let pitch = 0.2_f64;
        let yaw = 1.1_f64;
        let r = SO3::<f64>::from_euler_zyx(roll, pitch, yaw);
        let [r2, p2, y2] = r.to_euler_zyx();
        assert!(
            (r2 - roll).abs() < EPS_MED,
            "roll  mismatch: {} vs {}",
            r2,
            roll
        );
        assert!(
            (p2 - pitch).abs() < EPS_MED,
            "pitch mismatch: {} vs {}",
            p2,
            pitch
        );
        assert!(
            (y2 - yaw).abs() < EPS_MED,
            "yaw   mismatch: {} vs {}",
            y2,
            yaw
        );
    }

    #[test]
    fn group_property_compose_inverse() {
        let r1 = SO3::<f64>::from_axis_angle([1.0, 0.0, 0.0], 0.5).unwrap();
        let r2 = SO3::<f64>::from_axis_angle([0.0, 1.0, 0.0], 0.8).unwrap();
        let prod = r1.multiply(&r2);
        let inv = prod.transpose();
        let eye = prod.multiply(&inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (eye.mat[i][j] - expected).abs() < EPS,
                    "eye[{},{}] = {:.2e}",
                    i,
                    j,
                    eye.mat[i][j]
                );
            }
        }
    }

    #[test]
    fn log_exp_roundtrip() {
        let omega = [0.1_f64, -0.2, 0.3];
        let r = SO3::<f64>::exp(omega);
        let omega2 = r.log();
        for i in 0..3 {
            assert!(
                (omega2[i] - omega[i]).abs() < EPS_MED,
                "omega[{}]: {} vs {}",
                i,
                omega2[i],
                omega[i]
            );
        }
    }

    #[test]
    fn hat_vee_roundtrip() {
        let v = [1.0_f64, -2.0, 3.0];
        let m = hat(v);
        let v2 = vee(m);
        for i in 0..3 {
            assert!((v2[i] - v[i]).abs() < EPS);
        }
    }

    #[test]
    fn hat_is_skew_symmetric() {
        let m = hat([1.0_f64, 2.0, 3.0]);
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!((val + m[j][i]).abs() < EPS);
            }
        }
    }

    #[test]
    fn rotation_error_at_identity() {
        let r_d = SO3::<f64>::identity();
        let r = SO3::<f64>::identity();
        let err = rotation_error(&r_d, &r);
        for (i, &e) in err.iter().enumerate() {
            assert!(e.abs() < EPS, "err[{}] = {:.2e}", i, e);
        }
    }

    #[test]
    fn quaternion_roundtrip() {
        let r1 = SO3::<f64>::from_euler_zyx(0.3, -0.1, 0.7);
        let q = r1.to_quaternion();
        let r2 = SO3::<f64>::from_quaternion(q).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (r1.mat[i][j] - r2.mat[i][j]).abs() < 1e-10,
                    "R1[{},{}]={} R2[{},{}]={}",
                    i,
                    j,
                    r1.mat[i][j],
                    i,
                    j,
                    r2.mat[i][j]
                );
            }
        }
    }

    #[test]
    fn axis_angle_roundtrip() {
        let axis = [1.0_f64 / 3.0_f64.sqrt(); 3];
        let angle = 1.2_f64;
        let r = SO3::<f64>::from_axis_angle(axis, angle).unwrap();
        let (axis2, angle2) = r.to_axis_angle();
        assert!(
            (angle2 - angle).abs() < 1e-8,
            "angle: {} vs {}",
            angle2,
            angle
        );
        for i in 0..3 {
            assert!(
                (axis2[i] - axis[i]).abs() < 1e-8,
                "axis[{}]: {} vs {}",
                i,
                axis2[i],
                axis[i]
            );
        }
    }

    #[test]
    fn singular_axis_returns_error() {
        let result = SO3::<f64>::from_axis_angle([0.0, 0.0, 0.0], 1.0);
        assert!(matches!(result, Err(GeoError::Singular)));
    }
}
