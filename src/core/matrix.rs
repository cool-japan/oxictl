use crate::core::scalar::ControlScalar;

/// Fixed-size matrix backed by a 2D array. R = rows, C = cols.
/// All operations are O(R*C) or O(R*C*K) — no heap allocation.
#[derive(Debug, Clone, Copy)]
pub struct Matrix<S: ControlScalar, const R: usize, const C: usize> {
    pub data: [[S; C]; R],
}

impl<S: ControlScalar, const R: usize, const C: usize> Matrix<S, R, C> {
    /// Zero matrix.
    pub fn zeros() -> Self {
        Self {
            data: core::array::from_fn(|_| core::array::from_fn(|_| S::ZERO)),
        }
    }

    /// Matrix filled with a constant.
    pub fn filled(val: S) -> Self {
        Self {
            data: core::array::from_fn(|_| core::array::from_fn(|_| val)),
        }
    }

    /// Transpose: R×C → C×R.
    pub fn transpose(&self) -> Matrix<S, C, R> {
        Matrix {
            data: core::array::from_fn(|c| core::array::from_fn(|r| self.data[r][c])),
        }
    }

    /// Element-wise add.
    pub fn add_mat(&self, rhs: &Self) -> Self {
        Self {
            data: core::array::from_fn(|r| {
                core::array::from_fn(|c| self.data[r][c] + rhs.data[r][c])
            }),
        }
    }

    /// Element-wise subtract.
    pub fn sub_mat(&self, rhs: &Self) -> Self {
        Self {
            data: core::array::from_fn(|r| {
                core::array::from_fn(|c| self.data[r][c] - rhs.data[r][c])
            }),
        }
    }

    /// Scale by scalar.
    pub fn scale(&self, s: S) -> Self {
        Self {
            data: core::array::from_fn(|r| core::array::from_fn(|c| self.data[r][c] * s)),
        }
    }

    /// Negate.
    pub fn neg(&self) -> Self {
        self.scale(-S::ONE)
    }

    /// Get element.
    pub fn get(&self, r: usize, c: usize) -> S {
        self.data[r][c]
    }

    /// Set element.
    pub fn set(&mut self, r: usize, c: usize, val: S) {
        self.data[r][c] = val;
    }

    /// Frobenius norm (for convergence checks).
    pub fn frob_norm(&self) -> S {
        let mut sum = S::ZERO;
        for r in 0..R {
            for c in 0..C {
                sum += self.data[r][c] * self.data[r][c];
            }
        }
        sum.sqrt()
    }
}

impl<S: ControlScalar, const N: usize> Matrix<S, N, N> {
    /// Identity matrix (square only).
    pub fn identity() -> Self {
        Self {
            data: core::array::from_fn(|r| {
                core::array::from_fn(|c| if r == c { S::ONE } else { S::ZERO })
            }),
        }
    }

    /// Trace (sum of diagonal elements).
    pub fn trace(&self) -> S {
        let mut t = S::ZERO;
        for i in 0..N {
            t += self.data[i][i];
        }
        t
    }

    /// Cholesky decomposition: L such that L * L^T = self (lower-triangular).
    /// Returns None if the matrix is not positive definite.
    pub fn cholesky(&self) -> Option<Self> {
        let mut l = Self::zeros();
        for i in 0..N {
            for j in 0..=i {
                let mut sum = S::ZERO;
                for k in 0..j {
                    sum += l.data[i][k] * l.data[j][k];
                }
                if i == j {
                    let d = self.data[i][i] - sum;
                    if d <= S::ZERO {
                        return None;
                    }
                    l.data[i][j] = d.sqrt();
                } else {
                    l.data[i][j] = (self.data[i][j] - sum) / l.data[j][j];
                }
            }
        }
        Some(l)
    }

    /// Matrix inverse via Gaussian elimination with partial pivoting.
    /// Returns None if the matrix is singular.
    pub fn inv(&self) -> Option<Self> {
        let mut a = self.data;
        let mut inv: [[S; N]; N] = core::array::from_fn(|r| {
            core::array::from_fn(|c| if r == c { S::ONE } else { S::ZERO })
        });

        for col in 0..N {
            // Find pivot
            let mut max_row = col;
            let mut max_val = a[col][col].abs();
            for (row, row_data) in a.iter().enumerate().skip(col + 1) {
                if row_data[col].abs() > max_val {
                    max_val = row_data[col].abs();
                    max_row = row;
                }
            }

            if max_val < S::EPSILON * S::from_f64(1e6) {
                return None; // singular
            }

            // Swap rows
            if max_row != col {
                a.swap(max_row, col);
                inv.swap(max_row, col);
            }

            let pivot = a[col][col];
            let inv_pivot = S::ONE / pivot;

            // Scale pivot row
            for c in 0..N {
                a[col][c] *= inv_pivot;
                inv[col][c] *= inv_pivot;
            }

            // Eliminate column
            for row in 0..N {
                if row == col {
                    continue;
                }
                let factor = a[row][col];
                for c in 0..N {
                    a[row][c] -= factor * a[col][c];
                    inv[row][c] -= factor * inv[col][c];
                }
            }
        }

        Some(Self { data: inv })
    }
}

/// Matrix multiplication: (R×K) × (K×C) → (R×C)
pub fn matmul<S: ControlScalar, const R: usize, const K: usize, const C: usize>(
    a: &Matrix<S, R, K>,
    b: &Matrix<S, K, C>,
) -> Matrix<S, R, C> {
    Matrix {
        data: core::array::from_fn(|r| {
            core::array::from_fn(|c| {
                let mut sum = S::ZERO;
                for k in 0..K {
                    sum += a.data[r][k] * b.data[k][c];
                }
                sum
            })
        }),
    }
}

/// Matrix-vector product: (R×C) × (C×1) → (R×1), represented as arrays.
pub fn matvec<S: ControlScalar, const R: usize, const C: usize>(
    a: &Matrix<S, R, C>,
    v: &[S; C],
) -> [S; R] {
    core::array::from_fn(|r| {
        let mut sum = S::ZERO;
        for (c, v_c) in v.iter().enumerate() {
            sum += a.data[r][c] * *v_c;
        }
        sum
    })
}

/// Outer product: v (R×1) * w^T (1×C) → R×C
pub fn outer<S: ControlScalar, const R: usize, const C: usize>(
    v: &[S; R],
    w: &[S; C],
) -> Matrix<S, R, C> {
    Matrix {
        data: core::array::from_fn(|r| core::array::from_fn(|c| v[r] * w[c])),
    }
}

/// vec_add: element-wise add of two arrays.
pub fn vec_add<S: ControlScalar, const N: usize>(a: &[S; N], b: &[S; N]) -> [S; N] {
    core::array::from_fn(|i| a[i] + b[i])
}

/// vec_sub: element-wise sub of two arrays.
pub fn vec_sub<S: ControlScalar, const N: usize>(a: &[S; N], b: &[S; N]) -> [S; N] {
    core::array::from_fn(|i| a[i] - b[i])
}

/// vec_scale: multiply array by scalar.
pub fn vec_scale<S: ControlScalar, const N: usize>(a: &[S; N], s: S) -> [S; N] {
    core::array::from_fn(|i| a[i] * s)
}

/// vec_dot: dot product.
pub fn vec_dot<S: ControlScalar, const N: usize>(a: &[S; N], b: &[S; N]) -> S {
    let mut sum = S::ZERO;
    for i in 0..N {
        sum += a[i] * b[i];
    }
    sum
}

/// vec_norm: Euclidean norm.
pub fn vec_norm<S: ControlScalar, const N: usize>(a: &[S; N]) -> S {
    vec_dot(a, a).sqrt()
}

impl<S: ControlScalar, const R: usize, const C: usize> PartialEq for Matrix<S, R, C> {
    fn eq(&self, other: &Self) -> bool {
        for r in 0..R {
            for c in 0..C {
                if self.data[r][c] != other.data[r][c] {
                    return false;
                }
            }
        }
        true
    }
}

impl<S: ControlScalar, const R: usize, const C: usize> Default for Matrix<S, R, C> {
    fn default() -> Self {
        Self::zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros() {
        let m = Matrix::<f64, 2, 2>::zeros();
        assert_eq!(m.data[0][0], 0.0);
        assert_eq!(m.data[1][1], 0.0);
    }

    #[test]
    fn identity() {
        let m = Matrix::<f64, 3, 3>::identity();
        assert_eq!(m.data[0][0], 1.0);
        assert_eq!(m.data[1][1], 1.0);
        assert_eq!(m.data[2][2], 1.0);
        assert_eq!(m.data[0][1], 0.0);
    }

    #[test]
    fn transpose_2x3() {
        let mut m = Matrix::<f64, 2, 3>::zeros();
        m.data[0][1] = 5.0;
        m.data[1][2] = 3.0;
        let t = m.transpose();
        assert_eq!(t.data[1][0], 5.0);
        assert_eq!(t.data[2][1], 3.0);
    }

    #[test]
    fn matmul_identity() {
        let a = Matrix::<f64, 3, 3>::identity();
        let b = Matrix::<f64, 3, 3>::identity();
        let c = matmul(&a, &b);
        assert_eq!(c, Matrix::identity());
    }

    #[test]
    fn matmul_2x2() {
        let mut a = Matrix::<f64, 2, 2>::zeros();
        a.data[0][0] = 1.0;
        a.data[0][1] = 2.0;
        a.data[1][0] = 3.0;
        a.data[1][1] = 4.0;
        let result = matmul(&a, &a);
        // [[1,2],[3,4]] * [[1,2],[3,4]] = [[7,10],[15,22]]
        assert!((result.data[0][0] - 7.0).abs() < 1e-10);
        assert!((result.data[0][1] - 10.0).abs() < 1e-10);
        assert!((result.data[1][0] - 15.0).abs() < 1e-10);
        assert!((result.data[1][1] - 22.0).abs() < 1e-10);
    }

    #[test]
    fn matmul_2x3_3x2() {
        let mut a = Matrix::<f64, 2, 3>::zeros();
        a.data[0] = [1.0, 2.0, 3.0];
        a.data[1] = [4.0, 5.0, 6.0];
        let b = a.transpose();
        let c = matmul(&a, &b); // 2x2
                                // [1,2,3;4,5,6] * [1,4;2,5;3,6] = [14,32;32,77]
        assert!((c.data[0][0] - 14.0).abs() < 1e-10);
        assert!((c.data[0][1] - 32.0).abs() < 1e-10);
        assert!((c.data[1][0] - 32.0).abs() < 1e-10);
        assert!((c.data[1][1] - 77.0).abs() < 1e-10);
    }

    #[test]
    fn inv_2x2() {
        let mut m = Matrix::<f64, 2, 2>::zeros();
        m.data[0] = [1.0, 2.0];
        m.data[1] = [3.0, 4.0];
        let inv = m.inv().expect("should be invertible");
        let prod = matmul(&m, &inv);
        let eye = Matrix::<f64, 2, 2>::identity();
        for r in 0..2 {
            for c in 0..2 {
                assert!((prod.data[r][c] - eye.data[r][c]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn inv_1x1() {
        let mut m = Matrix::<f64, 1, 1>::zeros();
        m.data[0][0] = 4.0;
        let inv = m.inv().unwrap();
        assert!((inv.data[0][0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn inv_singular_returns_none() {
        let m = Matrix::<f64, 2, 2>::zeros();
        assert!(m.inv().is_none());
    }

    #[test]
    fn matvec_basic() {
        let mut m = Matrix::<f64, 2, 2>::zeros();
        m.data[0] = [1.0, 0.0];
        m.data[1] = [0.0, 2.0];
        let v = [3.0, 4.0];
        let r = matvec(&m, &v);
        assert_eq!(r, [3.0, 8.0]);
    }

    #[test]
    fn vec_ops() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let sum = vec_add(&a, &b);
        assert_eq!(sum, [5.0, 7.0, 9.0]);
        let diff = vec_sub(&b, &a);
        assert_eq!(diff, [3.0, 3.0, 3.0]);
        let dot = vec_dot(&a, &b);
        assert!((dot - 32.0).abs() < 1e-10);
    }

    #[test]
    fn scale_and_neg() {
        let eye = Matrix::<f64, 2, 2>::identity();
        let s = eye.scale(3.0);
        assert_eq!(s.data[0][0], 3.0);
        let n = eye.neg();
        assert_eq!(n.data[0][0], -1.0);
    }
}
