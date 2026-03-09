//! Extended Dynamic Mode Decomposition (EDMD).
//!
//! EDMD approximates the **Koopman operator** K from snapshot pairs
//! `{(ψ(x[k]), ψ(x[k+1]))}` by solving the weighted least-squares problem:
//!
//! ```text
//!   K = ( Σ_k  ψ(x[k+1]) · ψ(x[k])ᵀ ) · ( Σ_k  ψ(x[k]) · ψ(x[k])ᵀ )⁻¹
//! ```
//!
//! so that `ψ(x[k+1]) ≈ K · ψ(x[k])` in the lifted space.
//!
//! # Const parameters
//! - `L`    — lifting (observable) dimension.
//! - `DATA` — number of snapshot pairs used for fitting.

use crate::core::scalar::ControlScalar;
use crate::koopman::lifting_functions::KoopmanError;

// ── Gaussian elimination helper ───────────────────────────────────────────────

/// Invert an `L×L` matrix in-place using Gaussian elimination with partial
/// pivoting.  The result is stored back in `mat`; `inv` is initialised to the
/// identity and updated simultaneously.
///
/// Returns `Err(KoopmanError::SingularMatrix)` if a zero pivot is encountered.
#[allow(clippy::needless_range_loop)]
fn mat_inv_inplace<S: ControlScalar, const L: usize>(
    mat: &mut [[S; L]; L],
    inv: &mut [[S; L]; L],
) -> Result<(), KoopmanError> {
    // Initialise `inv` to identity
    for i in 0..L {
        for j in 0..L {
            inv[i][j] = if i == j { S::ONE } else { S::ZERO };
        }
    }

    for col in 0..L {
        // Partial pivot: find row with largest absolute value in this column
        let mut max_val = mat[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..L {
            let v = mat[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        // Check for (near-)singular pivot
        let tol = S::from_f64(1e-14);
        if max_val < tol {
            return Err(KoopmanError::SingularMatrix);
        }

        // Swap rows in both matrices
        if max_row != col {
            mat.swap(col, max_row);
            inv.swap(col, max_row);
        }

        // Scale pivot row
        let pivot = mat[col][col];
        for j in 0..L {
            mat[col][j] = mat[col][j] / pivot;
            inv[col][j] = inv[col][j] / pivot;
        }

        // Eliminate column in all other rows
        for row in 0..L {
            if row == col {
                continue;
            }
            let factor = mat[row][col];
            if factor == S::ZERO {
                continue;
            }
            for j in 0..L {
                let m_col_j = mat[col][j];
                let i_col_j = inv[col][j];
                mat[row][j] -= factor * m_col_j;
                inv[row][j] -= factor * i_col_j;
            }
        }
    }

    Ok(())
}

/// Multiply two `L×L` matrices: `C = A * B`.
fn mat_mul<S: ControlScalar, const L: usize>(a: &[[S; L]; L], b: &[[S; L]; L]) -> [[S; L]; L] {
    let mut c = [[S::ZERO; L]; L];
    for i in 0..L {
        for k in 0..L {
            if a[i][k] == S::ZERO {
                continue;
            }
            for j in 0..L {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

// ── EDMD ──────────────────────────────────────────────────────────────────────

/// EDMD model: approximates the Koopman operator K (L×L) from data.
///
/// # Type parameters
/// - `S`    — scalar type implementing [`ControlScalar`].
/// - `L`    — lifting dimension.
/// - `DATA` — number of snapshot pairs.
#[derive(Clone, Debug)]
pub struct Edmd<S, const L: usize, const DATA: usize> {
    /// Approximated Koopman matrix K ∈ ℝ^{L×L}.
    k_matrix: [[S; L]; L],
    /// Whether `fit` has been called successfully.
    fitted: bool,
}

impl<S: ControlScalar, const L: usize, const DATA: usize> Edmd<S, L, DATA> {
    /// Create an unfitted EDMD model (K is zero, fitted = false).
    pub fn new() -> Self {
        Self {
            k_matrix: [[S::ZERO; L]; L],
            fitted: false,
        }
    }

    /// Fit the Koopman operator from snapshot pairs.
    ///
    /// - `psi_x`      — lifted states `ψ(x[k])`,        shape `[DATA][L]`.
    /// - `psi_x_next` — lifted next-states `ψ(x[k+1])`, shape `[DATA][L]`.
    ///
    /// Requires `DATA ≥ L` for the Gram matrix to be invertible.
    ///
    /// # Errors
    /// - [`KoopmanError::InsufficientData`] if `DATA < L`.
    /// - [`KoopmanError::SingularMatrix`]   if the Gram matrix B is rank-deficient.
    pub fn fit(
        &mut self,
        psi_x: &[[S; L]; DATA],
        psi_x_next: &[[S; L]; DATA],
    ) -> Result<(), KoopmanError> {
        if DATA < L {
            return Err(KoopmanError::InsufficientData);
        }

        // A[i][j] = Σ_k  psi_x_next[k][i] * psi_x[k][j]
        let mut a_mat = [[S::ZERO; L]; L];
        // B[i][j] = Σ_k  psi_x[k][i]      * psi_x[k][j]
        let mut b_mat = [[S::ZERO; L]; L];

        for k in 0..DATA {
            for i in 0..L {
                for j in 0..L {
                    a_mat[i][j] += psi_x_next[k][i] * psi_x[k][j];
                    b_mat[i][j] += psi_x[k][i] * psi_x[k][j];
                }
            }
        }

        // Invert B
        let mut b_copy = b_mat;
        let mut b_inv = [[S::ZERO; L]; L];
        mat_inv_inplace::<S, L>(&mut b_copy, &mut b_inv)?;

        // K = A * B⁻¹
        self.k_matrix = mat_mul::<S, L>(&a_mat, &b_inv);
        self.fitted = true;

        Ok(())
    }

    /// Predict the next lifted state: `ψ(x[k+1]) ≈ K · ψ(x[k])`.
    ///
    /// # Errors
    /// - [`KoopmanError::NotFitted`] if `fit` has not been called.
    #[allow(clippy::needless_range_loop)]
    pub fn predict(&self, psi_x: &[S; L]) -> Result<[S; L], KoopmanError> {
        if !self.fitted {
            return Err(KoopmanError::NotFitted);
        }
        let mut out = [S::ZERO; L];
        for i in 0..L {
            for j in 0..L {
                out[i] += self.k_matrix[i][j] * psi_x[j];
            }
        }
        Ok(out)
    }

    /// Return a reference to the fitted Koopman matrix K.
    pub fn k_matrix(&self) -> &[[S; L]; L] {
        &self.k_matrix
    }

    /// Compute mean squared reconstruction error on the given snapshot pairs.
    ///
    /// `error = (1 / (DATA * L)) * Σ_k ‖K·ψ_x[k] − ψ_x_next[k]‖²`
    ///
    /// # Errors
    /// - [`KoopmanError::NotFitted`] if `fit` has not been called.
    pub fn reconstruction_error(
        &self,
        psi_x: &[[S; L]; DATA],
        psi_x_next: &[[S; L]; DATA],
    ) -> Result<S, KoopmanError> {
        if !self.fitted {
            return Err(KoopmanError::NotFitted);
        }

        let mut total = S::ZERO;
        for k in 0..DATA {
            let pred = self.predict(&psi_x[k])?;
            for i in 0..L {
                let diff = pred[i] - psi_x_next[k][i];
                total += diff * diff;
            }
        }

        let n = S::from_f64((DATA * L) as f64);
        Ok(total / n)
    }

    /// Return a rough spectral approximation: the diagonal entries of K
    /// (convenient as a proxy for Koopman eigenvalues in the scalar / diagonal case).
    #[allow(clippy::needless_range_loop)]
    pub fn eigenvalues_real_part(&self) -> [S; L] {
        let mut ev = [S::ZERO; L];
        for i in 0..L {
            ev[i] = self.k_matrix[i][i];
        }
        ev
    }

    /// Return `true` if the model has been successfully fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl<S: ControlScalar, const L: usize, const DATA: usize> Default for Edmd<S, L, DATA> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build snapshot pairs for the scalar linear map x[k+1] = a * x[k]
    /// using the identity lifting ψ(x) = x  (L=1).
    fn linear_scalar_data<const DATA: usize>(a: f64) -> ([[f64; 1]; DATA], [[f64; 1]; DATA]) {
        let mut psi = [[0.0_f64; 1]; DATA];
        let mut psi_next = [[0.0_f64; 1]; DATA];
        for k in 0..DATA {
            let x = (k as f64) * 0.1 + 0.1; // avoid x=0
            psi[k][0] = x;
            psi_next[k][0] = a * x;
        }
        (psi, psi_next)
    }

    #[test]
    fn edmd_recovers_linear_scalar_map() {
        // ψ = identity (L=1), system x[k+1] = 2*x[k] → K should be [[2]]
        let (psi, psi_next) = linear_scalar_data::<10>(2.0);
        let mut edmd: Edmd<f64, 1, 10> = Edmd::new();
        edmd.fit(&psi, &psi_next).expect("fit failed");
        let k = edmd.k_matrix();
        assert!((k[0][0] - 2.0).abs() < 1e-8, "K[0][0]={}", k[0][0]);
    }

    #[test]
    fn edmd_prediction_error_small_on_training_data() {
        let a = 0.8_f64;
        let (psi, psi_next) = linear_scalar_data::<20>(a);
        let mut edmd: Edmd<f64, 1, 20> = Edmd::new();
        edmd.fit(&psi, &psi_next).expect("fit failed");

        let mut total_err = 0.0_f64;
        for k in 0..20 {
            let pred = edmd.predict(&psi[k]).expect("predict failed");
            let err = (pred[0] - psi_next[k][0]).abs();
            total_err += err;
        }
        assert!(total_err < 1e-8, "total_err={}", total_err);
    }

    #[test]
    fn edmd_not_fitted_returns_error() {
        let edmd: Edmd<f64, 2, 5> = Edmd::new();
        let psi = [1.0_f64, 0.0];
        let result = edmd.predict(&psi);
        assert!(matches!(result, Err(KoopmanError::NotFitted)));
    }

    #[test]
    fn edmd_k_matrix_recovers_half_identity() {
        // L=2: psi_next = 0.5 * psi → K should be 0.5 * I
        let psi: [[f64; 2]; 5] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]];
        let psi_next: [[f64; 2]; 5] = [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]];
        let mut edmd: Edmd<f64, 2, 5> = Edmd::new();
        edmd.fit(&psi, &psi_next).expect("fit failed");
        let k = edmd.k_matrix();
        assert!((k[0][0] - 0.5).abs() < 1e-8, "K[0][0]={}", k[0][0]);
        assert!((k[1][1] - 0.5).abs() < 1e-8, "K[1][1]={}", k[1][1]);
        assert!(k[0][1].abs() < 1e-8);
        assert!(k[1][0].abs() < 1e-8);
    }

    #[test]
    fn edmd_insufficient_data_error() {
        // L=3, DATA=2 < L → InsufficientData
        let psi: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let psi_next: [[f64; 3]; 2] = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]];
        let mut edmd: Edmd<f64, 3, 2> = Edmd::new();
        let result = edmd.fit(&psi, &psi_next);
        assert!(matches!(result, Err(KoopmanError::InsufficientData)));
    }

    #[test]
    fn edmd_singular_gram_matrix_error() {
        // All snapshots identical → B is rank-1, singular
        let psi: [[f64; 2]; 4] = [[1.0, 1.0]; 4];
        let psi_next: [[f64; 2]; 4] = [[2.0, 2.0]; 4];
        let mut edmd: Edmd<f64, 2, 4> = Edmd::new();
        let result = edmd.fit(&psi, &psi_next);
        assert!(matches!(result, Err(KoopmanError::SingularMatrix)));
    }

    #[test]
    fn edmd_reconstruction_error_approx_zero_on_training_data() {
        let a = 1.5_f64;
        let (psi, psi_next) = linear_scalar_data::<15>(a);
        let mut edmd: Edmd<f64, 1, 15> = Edmd::new();
        edmd.fit(&psi, &psi_next).expect("fit failed");
        let err = edmd
            .reconstruction_error(&psi, &psi_next)
            .expect("recon error failed");
        assert!(err < 1e-12, "reconstruction error={}", err);
    }

    #[test]
    fn edmd_reconstruction_error_not_fitted_error() {
        let psi: [[f64; 1]; 5] = [[1.0]; 5];
        let psi_next: [[f64; 1]; 5] = [[2.0]; 5];
        let edmd: Edmd<f64, 1, 5> = Edmd::new();
        let result = edmd.reconstruction_error(&psi, &psi_next);
        assert!(matches!(result, Err(KoopmanError::NotFitted)));
    }

    #[test]
    fn edmd_eigenvalues_real_part_returns_diagonal() {
        let a = 0.5_f64;
        let (psi, psi_next) = linear_scalar_data::<10>(a);
        let mut edmd: Edmd<f64, 1, 10> = Edmd::new();
        edmd.fit(&psi, &psi_next).expect("fit failed");
        let ev = edmd.eigenvalues_real_part();
        assert!((ev[0] - a).abs() < 1e-8, "ev[0]={}", ev[0]);
    }
}
