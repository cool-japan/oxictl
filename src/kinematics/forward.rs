use crate::core::scalar::ControlScalar;

/// 2D rigid body transform: (x, y, θ).
#[derive(Debug, Clone, Copy)]
pub struct Transform2D<S: ControlScalar> {
    pub x: S,
    pub y: S,
    /// Rotation angle (rad).
    pub theta: S,
}

impl<S: ControlScalar> Transform2D<S> {
    pub fn new(x: S, y: S, theta: S) -> Self {
        Self { x, y, theta }
    }

    pub fn identity() -> Self {
        Self {
            x: S::ZERO,
            y: S::ZERO,
            theta: S::ZERO,
        }
    }

    /// Compose: self ∘ other (apply other in self's frame).
    pub fn compose(&self, other: &Self) -> Self {
        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();
        Self {
            x: self.x + cos_t * other.x - sin_t * other.y,
            y: self.y + sin_t * other.x + cos_t * other.y,
            theta: self.theta + other.theta,
        }
    }

    /// Transform a point from local to world frame.
    pub fn transform_point(&self, px: S, py: S) -> (S, S) {
        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();
        (
            self.x + cos_t * px - sin_t * py,
            self.y + sin_t * px + cos_t * py,
        )
    }

    /// Inverse transform.
    pub fn inverse(&self) -> Self {
        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();
        let inv_theta = -self.theta;
        let inv_x = -(cos_t * self.x + sin_t * self.y);
        let inv_y = -(-sin_t * self.x + cos_t * self.y);
        Self {
            x: inv_x,
            y: inv_y,
            theta: inv_theta,
        }
    }
}

impl<S: ControlScalar> Default for Transform2D<S> {
    fn default() -> Self {
        Self::identity()
    }
}

/// 3D homogeneous transform using 4×4 matrix representation.
/// Stored as rotation (row-major 3×3) + translation [3].
#[derive(Debug, Clone, Copy)]
pub struct Transform3D<S: ControlScalar> {
    /// Rotation matrix R (row-major).
    pub r: [[S; 3]; 3],
    /// Translation vector.
    pub t: [S; 3],
}

impl<S: ControlScalar> Transform3D<S> {
    pub fn identity() -> Self {
        Self {
            r: [
                [S::ONE, S::ZERO, S::ZERO],
                [S::ZERO, S::ONE, S::ZERO],
                [S::ZERO, S::ZERO, S::ONE],
            ],
            t: [S::ZERO; 3],
        }
    }

    /// Rotation around Z-axis by angle θ.
    pub fn rot_z(theta: S) -> Self {
        let c = theta.cos();
        let s = theta.sin();
        Self {
            r: [
                [c, -s, S::ZERO],
                [s, c, S::ZERO],
                [S::ZERO, S::ZERO, S::ONE],
            ],
            t: [S::ZERO; 3],
        }
    }

    /// Rotation around Y-axis by angle θ.
    pub fn rot_y(theta: S) -> Self {
        let c = theta.cos();
        let s = theta.sin();
        Self {
            r: [
                [c, S::ZERO, s],
                [S::ZERO, S::ONE, S::ZERO],
                [-s, S::ZERO, c],
            ],
            t: [S::ZERO; 3],
        }
    }

    /// Rotation around X-axis by angle θ.
    pub fn rot_x(theta: S) -> Self {
        let c = theta.cos();
        let s = theta.sin();
        Self {
            r: [
                [S::ONE, S::ZERO, S::ZERO],
                [S::ZERO, c, -s],
                [S::ZERO, s, c],
            ],
            t: [S::ZERO; 3],
        }
    }

    /// Pure translation.
    pub fn translate(dx: S, dy: S, dz: S) -> Self {
        let mut tf = Self::identity();
        tf.t = [dx, dy, dz];
        tf
    }

    /// Compose: self * other (other applied first).
    pub fn compose(&self, other: &Self) -> Self {
        // R_new = self.R * other.R
        let r_new: [[S; 3]; 3] = core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                (0..3)
                    .map(|k| self.r[i][k] * other.r[k][j])
                    .fold(S::ZERO, |a, b| a + b)
            })
        });
        // t_new = self.R * other.t + self.t
        let t_new: [S; 3] = core::array::from_fn(|i| {
            (0..3)
                .map(|k| self.r[i][k] * other.t[k])
                .fold(S::ZERO, |a, b| a + b)
                + self.t[i]
        });
        Self { r: r_new, t: t_new }
    }

    /// Transform a 3D point.
    pub fn transform_point(&self, p: [S; 3]) -> [S; 3] {
        core::array::from_fn(|i| {
            self.r[i]
                .iter()
                .zip(p.iter())
                .map(|(&r, &pj)| r * pj)
                .fold(S::ZERO, |a, b| a + b)
                + self.t[i]
        })
    }

    /// Inverse (R^T, -R^T * t).
    pub fn inverse(&self) -> Self {
        let rt = [
            [self.r[0][0], self.r[1][0], self.r[2][0]],
            [self.r[0][1], self.r[1][1], self.r[2][1]],
            [self.r[0][2], self.r[1][2], self.r[2][2]],
        ];
        let mut t_inv = [S::ZERO; 3];
        for i in 0..3 {
            for (k, rt_row_k) in rt[i].iter().enumerate() {
                t_inv[i] -= *rt_row_k * self.t[k];
            }
        }
        Self { r: rt, t: t_inv }
    }

    /// Extract translation.
    pub fn position(&self) -> [S; 3] {
        self.t
    }
}

impl<S: ControlScalar> Default for Transform3D<S> {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform2d_compose_identity() {
        let t = Transform2D::new(1.0_f64, 2.0, 0.5);
        let id = Transform2D::identity();
        let composed = t.compose(&id);
        assert!((composed.x - t.x).abs() < 1e-10);
        assert!((composed.y - t.y).abs() < 1e-10);
        assert!((composed.theta - t.theta).abs() < 1e-10);
    }

    #[test]
    fn transform2d_inverse() {
        let t = Transform2D::new(3.0_f64, 1.0, core::f64::consts::PI / 4.0);
        let inv = t.inverse();
        let composed = t.compose(&inv);
        assert!(composed.x.abs() < 1e-10, "x={}", composed.x);
        assert!(composed.y.abs() < 1e-10, "y={}", composed.y);
    }

    #[test]
    fn transform3d_rot_z_90() {
        let t = Transform3D::rot_z(core::f64::consts::PI / 2.0);
        let p = t.transform_point([1.0, 0.0, 0.0]);
        assert!(p[0].abs() < 1e-10, "x={}", p[0]);
        assert!((p[1] - 1.0).abs() < 1e-10, "y={}", p[1]);
    }

    #[test]
    fn transform3d_compose_identity() {
        let t = Transform3D::rot_z(1.0_f64).compose(&Transform3D::translate(1.0, 2.0, 0.0));
        let inv = t.inverse();
        let c = t.compose(&inv);
        assert!(c.t[0].abs() < 1e-10);
        assert!(c.t[1].abs() < 1e-10);
        // Check R ≈ I
        assert!((c.r[0][0] - 1.0).abs() < 1e-10);
        assert!((c.r[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn transform3d_translate_point() {
        let t = Transform3D::translate(1.0_f64, 2.0, 3.0);
        let p = t.transform_point([0.0, 0.0, 0.0]);
        assert!((p[0] - 1.0).abs() < 1e-10);
        assert!((p[1] - 2.0).abs() < 1e-10);
        assert!((p[2] - 3.0).abs() < 1e-10);
    }
}
