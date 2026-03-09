//! Weighted Pseudo-Inverse Control Allocation for over-actuated systems.
//!
//! Solves the weighted minimum-norm problem:
//!   min (u - u_pref)^T W (u - u_pref)
//!   s.t. B u = v_des,  u_min ≤ u ≤ u_max
//!
//! where B is the N×M effectiveness matrix (N control objectives, M ≥ N actuators),
//! W = diag(w) is a positive-definite diagonal weight matrix.
//!
//! The unconstrained weighted pseudo-inverse solution is:
//!   u* = u_pref + W^{-1} B^T (B W^{-1} B^T)^{-1} (v_des - B u_pref)
//!
//! Constraint handling uses a single-pass re-allocation: saturated actuators are
//! removed from the free set and the remaining free actuators absorb the residual.
use crate::core::scalar::ControlScalar;

/// Errors that can arise during control allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationError {
    /// The allocation problem has no feasible solution (e.g. v_des outside attainable set).
    Infeasible,
    /// The Gram matrix B W^{-1} B^T is (near-)singular — cannot invert.
    SingularMatrix,
    /// Dimension mismatch between inputs.
    DimensionMismatch,
    /// One or more weight values are non-positive.
    InvalidWeight,
    /// One or more bound pairs are inverted (u_min > u_max).
    InvalidBounds,
}

impl core::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Infeasible => write!(f, "allocation infeasible"),
            Self::SingularMatrix => write!(f, "singular Gram matrix"),
            Self::DimensionMismatch => write!(f, "dimension mismatch"),
            Self::InvalidWeight => write!(f, "non-positive weight"),
            Self::InvalidBounds => write!(f, "invalid bounds (u_min > u_max)"),
        }
    }
}

/// Weighted pseudo-inverse control allocator.
///
/// # Type Parameters
/// - `S`: scalar type (`f32` or `f64`)
/// - `N`: number of control objectives (rows of B)
/// - `M`: number of actuators (columns of B), must satisfy M ≥ N
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedPseudoInverse<S, const N: usize, const M: usize> {
    /// Effectiveness matrix B (N×M).
    b: [[S; M]; N],
    /// Diagonal weight vector w (W = diag(w), w[i] > 0).
    w: [S; M],
    /// Lower actuator bounds.
    u_min: [S; M],
    /// Upper actuator bounds.
    u_max: [S; M],
    /// Preferred actuator state (bias point).
    u_pref: [S; M],
    /// Singularity threshold for Gaussian elimination pivot detection.
    singular_tol: S,
}

impl<S: ControlScalar, const N: usize, const M: usize> WeightedPseudoInverse<S, N, M> {
    /// Construct a new `WeightedPseudoInverse` allocator.
    ///
    /// # Errors
    /// Returns `AllocationError::InvalidWeight` if any `w[i] ≤ 0`.
    /// Returns `AllocationError::InvalidBounds` if any `u_min[i] > u_max[i]`.
    pub fn new(
        b: [[S; M]; N],
        w: [S; M],
        u_min: [S; M],
        u_max: [S; M],
    ) -> Result<Self, AllocationError> {
        for i in 0..M {
            if w[i] <= S::ZERO {
                return Err(AllocationError::InvalidWeight);
            }
            if u_min[i] > u_max[i] {
                return Err(AllocationError::InvalidBounds);
            }
        }
        Ok(Self {
            b,
            w,
            u_min,
            u_max,
            u_pref: [S::ZERO; M],
            singular_tol: S::from_f64(1e-10),
        })
    }

    /// Set the preferred actuator state (bias point for the weighted norm).
    pub fn set_preference(&mut self, u_pref: [S; M]) {
        self.u_pref = u_pref;
    }

    /// Set the singularity detection threshold (default 1e-10).
    pub fn set_singular_tol(&mut self, tol: S) {
        self.singular_tol = tol;
    }

    /// Compute the allocation for the desired virtual control `v_des`.
    ///
    /// Returns the actuator command vector `u` (M×1).
    ///
    /// # Algorithm
    /// 1. Compute W^{-1} (diagonal: 1/w[i]).
    /// 2. Compute the Gram matrix G = B W^{-1} B^T (N×N symmetric).
    /// 3. Compute residual r = v_des - B u_pref (N×1).
    /// 4. Solve G λ = r via Gaussian elimination with partial pivoting.
    /// 5. Compute unconstrained u = u_pref + W^{-1} B^T λ.
    /// 6. Clamp to [u_min, u_max]; perform one re-allocation pass for saturated actuators.
    pub fn allocate(&self, v_des: &[S; N]) -> Result<[S; M], AllocationError> {
        // Step 1: W^{-1} diagonal
        let mut w_inv = [S::ZERO; M];
        for (inv, &wi) in w_inv.iter_mut().zip(self.w.iter()) {
            *inv = S::ONE / wi;
        }

        // Step 2: Gram matrix G = B W^{-1} B^T  (N×N)
        let mut gram = [[S::ZERO; N]; N];
        for (row, gram_row) in gram.iter_mut().enumerate() {
            for (col, gram_val) in gram_row.iter_mut().enumerate() {
                *gram_val = w_inv
                    .iter()
                    .enumerate()
                    .fold(S::ZERO, |acc, (k, &w_inv_k)| {
                        acc + self.b[row][k] * w_inv_k * self.b[col][k]
                    });
            }
        }

        // Step 3: residual r = v_des - B u_pref
        let mut rhs = [S::ZERO; N];
        for (row, rhs_val) in rhs.iter_mut().enumerate() {
            let bu: S = self.b[row]
                .iter()
                .zip(self.u_pref.iter())
                .fold(S::ZERO, |acc, (&bij, &up)| acc + bij * up);
            *rhs_val = v_des[row] - bu;
        }

        // Step 4: Solve G λ = rhs via Gaussian elimination with partial pivoting
        let lambda = gauss_solve_n::<S, N>(&gram, &rhs, self.singular_tol)?;

        // Step 5: u_free = u_pref + W^{-1} B^T λ
        let mut u = [S::ZERO; M];
        for (j, (u_j, (&up_j, &w_inv_j))) in u
            .iter_mut()
            .zip(self.u_pref.iter().zip(w_inv.iter()))
            .enumerate()
        {
            let bt_lam: S = self
                .b
                .iter()
                .zip(lambda.iter())
                .fold(S::ZERO, |acc, (b_row, &lam_i)| acc + b_row[j] * lam_i);
            *u_j = up_j + w_inv_j * bt_lam;
        }

        // Step 6: Clamp and single-pass re-allocation
        self.constrained_realloc(u, v_des)
    }

    /// Clamp actuators to bounds and redistribute residual over free actuators.
    fn constrained_realloc(
        &self,
        mut u: [S; M],
        v_des: &[S; N],
    ) -> Result<[S; M], AllocationError> {
        let mut saturated = [false; M];
        for j in 0..M {
            if u[j] < self.u_min[j] {
                u[j] = self.u_min[j];
                saturated[j] = true;
            } else if u[j] > self.u_max[j] {
                u[j] = self.u_max[j];
                saturated[j] = true;
            }
        }

        let free_count = saturated.iter().filter(|&&s| !s).count();
        if free_count == 0 {
            return Ok(u);
        }

        // Compute residual: v_res = v_des - B u_sat
        let mut v_res = [S::ZERO; N];
        for (row, v_res_val) in v_res.iter_mut().enumerate() {
            let bu: S = self.b[row]
                .iter()
                .zip(u.iter())
                .fold(S::ZERO, |acc, (&bij, &uj)| acc + bij * uj);
            *v_res_val = v_des[row] - bu;
        }

        // Check if residual is already negligible
        let res_norm: S = v_res.iter().fold(S::ZERO, |acc, &v| acc + v * v);
        if res_norm < S::from_f64(1e-20) {
            return Ok(u);
        }

        // Collect free actuator indices
        let mut free_idx = [0usize; M];
        let mut fi = 0usize;
        for (j, &sat) in saturated.iter().enumerate() {
            if !sat {
                free_idx[fi] = j;
                fi += 1;
            }
        }

        // Compute G_free = B_free W_free^{-1} B_free^T (N×N)
        let mut gram_free = [[S::ZERO; N]; N];
        for (row, gram_row) in gram_free.iter_mut().enumerate() {
            for (col, gram_val) in gram_row.iter_mut().enumerate() {
                *gram_val = free_idx[..free_count].iter().fold(S::ZERO, |acc, &k| {
                    acc + self.b[row][k] * (S::ONE / self.w[k]) * self.b[col][k]
                });
            }
        }

        let lambda_free = match gauss_solve_n::<S, N>(&gram_free, &v_res, self.singular_tol) {
            Ok(lam) => lam,
            Err(_) => return Ok(u),
        };

        // Update free actuators
        for &k in free_idx[..free_count].iter() {
            let bt_lam: S = self
                .b
                .iter()
                .zip(lambda_free.iter())
                .fold(S::ZERO, |acc, (b_row, &lam_i)| acc + b_row[k] * lam_i);
            let delta = (S::ONE / self.w[k]) * bt_lam;
            u[k] = (u[k] + delta).clamp_val(self.u_min[k], self.u_max[k]);
        }

        Ok(u)
    }

    /// Evaluate the weighted norm cost for a given actuator vector.
    ///
    /// Returns (u - u_pref)^T W (u - u_pref).
    pub fn weighted_cost(&self, u: &[S; M]) -> S {
        u.iter().zip(self.u_pref.iter()).zip(self.w.iter()).fold(
            S::ZERO,
            |acc, ((&uj, &up), &wj)| {
                let diff = uj - up;
                acc + wj * diff * diff
            },
        )
    }

    /// Compute the virtual control actually produced by actuator vector u.
    pub fn virtual_control(&self, u: &[S; M]) -> [S; N] {
        let mut v = [S::ZERO; N];
        for (row, v_val) in v.iter_mut().enumerate() {
            *v_val = self.b[row]
                .iter()
                .zip(u.iter())
                .fold(S::ZERO, |acc, (&bij, &uj)| acc + bij * uj);
        }
        v
    }

    /// Compute the tracking error ||B u - v_des||.
    pub fn tracking_error(&self, u: &[S; M], v_des: &[S; N]) -> S {
        let v = self.virtual_control(u);
        let sq: S = v.iter().zip(v_des.iter()).fold(S::ZERO, |acc, (&vi, &di)| {
            let d = vi - di;
            acc + d * d
        });
        S::from_f64(libm::sqrt(sq.to_f64()))
    }
}

/// Solve an N×N linear system A x = b using Gaussian elimination with partial pivoting.
///
/// Returns `Err(AllocationError::SingularMatrix)` if the pivot is smaller than `tol`.
#[allow(clippy::needless_range_loop)]
fn gauss_solve_n<S: ControlScalar, const N: usize>(
    a: &[[S; N]; N],
    b: &[S; N],
    tol: S,
) -> Result<[S; N], AllocationError> {
    let mut aug = *a;
    let mut rhs = *b;

    for col in 0..N {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..N {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < tol {
            return Err(AllocationError::SingularMatrix);
        }
        if max_row != col {
            aug.swap(max_row, col);
            rhs.swap(max_row, col);
        }
        let pivot = aug[col][col];
        for row in (col + 1)..N {
            let factor = aug[row][col] / pivot;
            for k in col..N {
                let sub = factor * aug[col][k];
                aug[row][k] -= sub;
            }
            let sub_r = factor * rhs[col];
            rhs[row] -= sub_r;
        }
    }

    // Back substitution
    let mut x = [S::ZERO; N];
    for i in (0..N).rev() {
        let mut acc = rhs[i];
        for j in (i + 1)..N {
            acc -= aug[i][j] * x[j];
        }
        x[i] = acc / aug[i][i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq<const M: usize>(a: &[f64; M], b: &[f64; M], tol: f64) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(&ai, &bi)| (ai - bi).abs() <= tol)
    }

    /// Square (2×2) system — should give exact allocation.
    #[test]
    fn test_square_exact_allocation() {
        let b = [[1.0f64, 0.0], [0.0, 1.0]];
        let w = [1.0f64, 1.0];
        let u_min = [-10.0f64, -10.0];
        let u_max = [10.0f64, 10.0];

        let alloc =
            WeightedPseudoInverse::<f64, 2, 2>::new(b, w, u_min, u_max).expect("valid constructor");
        let v_des = [1.0f64, 2.0];
        let u = alloc.allocate(&v_des).expect("allocation ok");
        assert!(
            approx_eq(&u, &[1.0, 2.0], 1e-9),
            "square identity: got {:?}",
            u
        );
        assert!(alloc.tracking_error(&u, &v_des) < 1e-9);
    }

    /// Over-actuated (2×3): low-weight actuator should carry more authority.
    #[test]
    fn test_over_actuated_weighted() {
        let b: [[f64; 3]; 2] = [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let w = [1.0f64, 1.0, 100.0];
        let u_min = [-10.0f64; 3];
        let u_max = [10.0f64; 3];

        let alloc =
            WeightedPseudoInverse::<f64, 2, 3>::new(b, w, u_min, u_max).expect("valid constructor");
        let v_des = [1.0f64, 1.0];
        let u = alloc.allocate(&v_des).expect("allocation ok");

        let v_actual = alloc.virtual_control(&u);
        assert!(
            (v_actual[0] - 1.0).abs() < 1e-8,
            "v[0] mismatch: {}",
            v_actual[0]
        );
        assert!(
            (v_actual[1] - 1.0).abs() < 1e-8,
            "v[1] mismatch: {}",
            v_actual[1]
        );
        assert!(
            (u[0] - u[1]).abs() < 1e-8,
            "equal weight actuators should share equally: {:?}",
            u
        );
        assert!((u[2] - 1.0).abs() < 1e-8, "u[2] should be 1.0: {}", u[2]);
    }

    /// Low-weight actuator preferred.
    #[test]
    fn test_weight_preference_low_weight_favored() {
        let b: [[f64; 2]; 1] = [[1.0, 1.0]];
        let w = [0.1f64, 10.0];
        let u_min = [-10.0f64; 2];
        let u_max = [10.0f64; 2];

        let alloc =
            WeightedPseudoInverse::<f64, 1, 2>::new(b, w, u_min, u_max).expect("valid constructor");
        let v_des = [1.0f64];
        let u = alloc.allocate(&v_des).expect("allocation ok");

        assert!(
            u[0] > u[1] * 5.0,
            "low-weight actuator should carry more: u={:?}",
            u
        );
        let v_actual = alloc.virtual_control(&u);
        assert!((v_actual[0] - 1.0).abs() < 1e-8);
    }

    /// Saturation test.
    #[test]
    fn test_saturation_clamped() {
        let b = [[1.0f64, 0.0], [0.0, 1.0]];
        let w = [1.0f64, 1.0];
        let u_min = [-2.0f64, -2.0];
        let u_max = [2.0f64, 2.0];

        let alloc =
            WeightedPseudoInverse::<f64, 2, 2>::new(b, w, u_min, u_max).expect("valid constructor");
        let v_des = [5.0f64, 5.0];
        let u = alloc.allocate(&v_des).expect("allocation ok");

        assert!(u[0] <= 2.0 + 1e-12 && u[0] >= -2.0 - 1e-12);
        assert!(u[1] <= 2.0 + 1e-12 && u[1] >= -2.0 - 1e-12);
    }

    /// Preferred state test.
    #[test]
    fn test_preferred_state() {
        let b: [[f64; 2]; 1] = [[1.0, 1.0]];
        let w = [1.0f64, 1.0];
        let u_min = [-10.0f64; 2];
        let u_max = [10.0f64; 2];

        let mut alloc =
            WeightedPseudoInverse::<f64, 1, 2>::new(b, w, u_min, u_max).expect("valid constructor");

        let v_des = [2.0f64];
        let u_no_pref = alloc.allocate(&v_des).expect("ok");
        assert!((u_no_pref[0] - 1.0).abs() < 1e-8);
        assert!((u_no_pref[1] - 1.0).abs() < 1e-8);

        alloc.set_preference([3.0, 0.0]);
        let u_pref = alloc.allocate(&v_des).expect("ok");
        assert!(
            u_pref[0] > 1.0,
            "u[0] should be pulled toward preference: {:?}",
            u_pref
        );
        let v_actual = alloc.virtual_control(&u_pref);
        assert!((v_actual[0] - 2.0).abs() < 1e-8);
    }

    /// Constructor rejects invalid weights.
    #[test]
    fn test_invalid_weight_rejected() {
        let b = [[1.0f64, 0.0], [0.0, 1.0]];
        let w = [0.0f64, 1.0];
        let u_min = [-10.0f64; 2];
        let u_max = [10.0f64; 2];
        assert_eq!(
            WeightedPseudoInverse::<f64, 2, 2>::new(b, w, u_min, u_max),
            Err(AllocationError::InvalidWeight)
        );
    }

    /// Constructor rejects invalid bounds.
    #[test]
    fn test_invalid_bounds_rejected() {
        let b = [[1.0f64, 0.0], [0.0, 1.0]];
        let w = [1.0f64; 2];
        let u_min = [5.0f64, 0.0];
        let u_max = [2.0f64, 10.0];
        assert_eq!(
            WeightedPseudoInverse::<f64, 2, 2>::new(b, w, u_min, u_max),
            Err(AllocationError::InvalidBounds)
        );
    }

    /// Weighted cost reflects the norm correctly.
    #[test]
    fn test_weighted_cost() {
        let b = [[1.0f64, 0.0], [0.0, 1.0]];
        let w = [2.0f64, 3.0];
        let u_min = [-10.0f64; 2];
        let u_max = [10.0f64; 2];
        let alloc =
            WeightedPseudoInverse::<f64, 2, 2>::new(b, w, u_min, u_max).expect("valid constructor");

        let u = [1.0f64, 2.0];
        let cost = alloc.weighted_cost(&u);
        assert!((cost - 14.0).abs() < 1e-10, "cost: {}", cost);
    }
}
