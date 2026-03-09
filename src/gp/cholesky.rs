//! Cholesky factorization and triangular solvers for the GP module.
//!
//! Operates on fixed-size `[[S; N]; N]` arrays (no heap allocation), making
//! it suitable for `no_std` embedded targets.

#![allow(clippy::needless_range_loop)]

use super::GpError;
use crate::core::scalar::ControlScalar;

// ──────────────────────────────────────────────────────────────────────────────
// Cholesky decomposition  (Cholesky-Banachiewicz)
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the lower-triangular Cholesky factor **L** such that A = L · Lᵀ.
///
/// `A` must be symmetric and positive definite.  Only the lower-triangular
/// part (including the diagonal) of `A` is read; the upper-triangular entries
/// are ignored.
///
/// Returns `Err(GpError::NotPositiveDefinite)` if any diagonal element of L
/// would be ≤ 0 (i.e. A is not positive definite).
pub fn cholesky<S: ControlScalar, const N: usize>(a: &[[S; N]; N]) -> Result<[[S; N]; N], GpError> {
    let mut l = [[S::ZERO; N]; N];

    for j in 0..N {
        // Diagonal element: L[j][j] = sqrt(A[j][j] - Σ_{k<j} L[j][k]²)
        let mut diag = a[j][j];
        for k in 0..j {
            diag -= l[j][k] * l[j][k];
        }
        if diag.to_f64() <= 0.0 {
            return Err(GpError::NotPositiveDefinite);
        }
        l[j][j] = S::from_f64(libm::sqrt(diag.to_f64()));

        // Sub-diagonal elements: L[i][j] = (A[i][j] - Σ_{k<j} L[i][k]*L[j][k]) / L[j][j]
        for i in (j + 1)..N {
            let mut off = a[i][j];
            for k in 0..j {
                off -= l[i][k] * l[j][k];
            }
            l[i][j] = off / l[j][j];
        }
    }

    Ok(l)
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward substitution  L x = b  (L lower-triangular)
// ──────────────────────────────────────────────────────────────────────────────

/// Solve the lower-triangular system **L** x = b by forward substitution.
///
/// Returns `Err(GpError::NumericalError)` if any diagonal of `L` is zero.
pub fn forward_sub<S: ControlScalar, const N: usize>(
    l: &[[S; N]; N],
    b: &[S; N],
) -> Result<[S; N], GpError> {
    let mut x = [S::ZERO; N];
    for i in 0..N {
        if l[i][i].to_f64() == 0.0 {
            return Err(GpError::NumericalError);
        }
        let mut acc = b[i];
        for j in 0..i {
            acc -= l[i][j] * x[j];
        }
        x[i] = acc / l[i][i];
    }
    Ok(x)
}

// ──────────────────────────────────────────────────────────────────────────────
// Backward substitution  Lᵀ x = b  (L lower-triangular → Lᵀ upper-triangular)
// ──────────────────────────────────────────────────────────────────────────────

/// Solve the upper-triangular system Lᵀ x = b by backward substitution.
///
/// Returns `Err(GpError::NumericalError)` if any diagonal of `L` is zero.
pub fn backward_sub<S: ControlScalar, const N: usize>(
    l: &[[S; N]; N],
    b: &[S; N],
) -> Result<[S; N], GpError> {
    let mut x = [S::ZERO; N];
    // i goes from N-1 down to 0
    for step in 0..N {
        let i = N - 1 - step;
        if l[i][i].to_f64() == 0.0 {
            return Err(GpError::NumericalError);
        }
        let mut acc = b[i];
        // Lᵀ[i][j] = L[j][i] for j > i
        for j in (i + 1)..N {
            acc -= l[j][i] * x[j];
        }
        x[i] = acc / l[i][i];
    }
    Ok(x)
}

// ──────────────────────────────────────────────────────────────────────────────
// Combined solve  A x = b  via Cholesky
// ──────────────────────────────────────────────────────────────────────────────

/// Solve the symmetric positive definite system **A** x = b via Cholesky
/// factorization followed by forward and backward substitution.
///
/// Internally computes L = chol(A), then solves L y = b and Lᵀ x = y.
pub fn cholesky_solve<S: ControlScalar, const N: usize>(
    a: &[[S; N]; N],
    b: &[S; N],
) -> Result<[S; N], GpError> {
    let l = cholesky(a)?;
    let y = forward_sub(&l, b)?;
    backward_sub(&l, &y)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── cholesky ──────────────────────────────────────────────────────────────

    #[test]
    fn cholesky_identity() {
        let a = [[1.0_f64, 0.0], [0.0, 1.0]];
        let l = cholesky(&a).expect("cholesky of identity must succeed");
        // L should be identity
        assert!(approx_eq(l[0][0], 1.0, TOL));
        assert!(approx_eq(l[1][0], 0.0, TOL));
        assert!(approx_eq(l[1][1], 1.0, TOL));
    }

    #[test]
    fn cholesky_known_2x2() {
        // A = [[4, 2], [2, 3]]
        // L = [[2, 0], [1, sqrt(2)]]
        let a = [[4.0_f64, 2.0], [2.0, 3.0]];
        let l = cholesky(&a).expect("cholesky must succeed");
        assert!(approx_eq(l[0][0], 2.0, TOL));
        assert!(approx_eq(l[1][0], 1.0, TOL));
        assert!(approx_eq(l[1][1], libm::sqrt(2.0_f64), TOL));
    }

    #[test]
    fn cholesky_not_positive_definite() {
        // A = [[-1, 0], [0, 1]]  — not PD
        let a = [[-1.0_f64, 0.0], [0.0, 1.0]];
        assert_eq!(cholesky(&a), Err(GpError::NotPositiveDefinite));
    }

    #[test]
    fn cholesky_round_trip() {
        // Verify L * Lᵀ ≈ A for a 3×3 SPD matrix
        let a = [[6.0_f64, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]];
        let l = cholesky(&a).expect("cholesky must succeed");
        // Reconstruct A' = L * Lᵀ
        for i in 0..3 {
            for j in 0..3 {
                let mut entry = 0.0_f64;
                for k in 0..3 {
                    entry += l[i][k] * l[j][k];
                }
                assert!(approx_eq(entry, a[i][j], 1e-12), "A'[{i}][{j}] mismatch");
            }
        }
    }

    // ── forward_sub ───────────────────────────────────────────────────────────

    #[test]
    fn forward_sub_identity() {
        let l = [[1.0_f64, 0.0], [0.0, 1.0]];
        let b = [3.0_f64, 5.0];
        let x = forward_sub(&l, &b).expect("forward_sub must succeed");
        assert!(approx_eq(x[0], 3.0, TOL));
        assert!(approx_eq(x[1], 5.0, TOL));
    }

    #[test]
    fn forward_sub_known() {
        // L = [[2,0],[1,sqrt(2)]], b = [4, 5]
        // y[0] = 4/2 = 2
        // y[1] = (5 - 1*2) / sqrt(2) = 3/sqrt(2)
        let sqrt2 = libm::sqrt(2.0_f64);
        let l = [[2.0_f64, 0.0], [1.0, sqrt2]];
        let b = [4.0_f64, 5.0];
        let x = forward_sub(&l, &b).expect("forward_sub must succeed");
        assert!(approx_eq(x[0], 2.0, TOL));
        assert!(approx_eq(x[1], 3.0 / sqrt2, TOL));
    }

    // ── backward_sub ─────────────────────────────────────────────────────────

    #[test]
    fn backward_sub_identity() {
        // Lᵀ = Iᵀ = I
        let l = [[1.0_f64, 0.0], [0.0, 1.0]];
        let b = [7.0_f64, -2.0];
        let x = backward_sub(&l, &b).expect("backward_sub must succeed");
        assert!(approx_eq(x[0], 7.0, TOL));
        assert!(approx_eq(x[1], -2.0, TOL));
    }

    // ── cholesky_solve ────────────────────────────────────────────────────────

    #[test]
    fn cholesky_solve_residual() {
        // A = [[4,2],[2,3]], b = [1,2]
        // Expected: x such that A*x ≈ b
        let a = [[4.0_f64, 2.0], [2.0, 3.0]];
        let b = [1.0_f64, 2.0];
        let x = cholesky_solve(&a, &b).expect("cholesky_solve must succeed");
        // Check A*x ≈ b
        let r0 = a[0][0] * x[0] + a[0][1] * x[1];
        let r1 = a[1][0] * x[0] + a[1][1] * x[1];
        assert!(approx_eq(r0, b[0], 1e-12), "residual[0] failed: got {r0}");
        assert!(approx_eq(r1, b[1], 1e-12), "residual[1] failed: got {r1}");
    }

    #[test]
    fn cholesky_solve_3x3() {
        let a = [[6.0_f64, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]];
        let b = [1.0_f64, 2.0, 3.0];
        let x = cholesky_solve(&a, &b).expect("3x3 solve must succeed");
        // Verify A*x ≈ b
        for i in 0..3 {
            let mut row = 0.0_f64;
            for j in 0..3 {
                row += a[i][j] * x[j];
            }
            assert!(approx_eq(row, b[i], 1e-10), "row {i} residual: got {row}");
        }
    }
}
