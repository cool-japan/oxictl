//! Bounded Least-Squares Control Allocation via Projected Gradient Descent.
//!
//! Solves the bounded least-squares problem:
//!   min_{u} ||v_des - B u||^2
//!   s.t.    u_min ≤ u ≤ u_max
//!
//! where B is the N×M effectiveness matrix.
//!
//! # Algorithm
//!
//! Projected gradient descent (PGD):
//!   u_{k+1} = clip(u_k - alpha * B^T (B u_k - v_des), u_min, u_max)
//!
//! The step size alpha = 1 / ||B^T B||_inf ensures convergence.
use crate::core::scalar::ControlScalar;

pub use crate::allocation::weighted_pseudo::AllocationError;

/// Bounded least-squares control allocator using projected gradient descent.
///
/// # Type Parameters
/// - `S`: scalar type (`f32` or `f64`)
/// - `N`: number of control objectives (rows of B)
/// - `M`: number of actuators (columns of B)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedLsAllocator<S, const N: usize, const M: usize> {
    /// Effectiveness matrix B (N×M).
    b: [[S; M]; N],
    /// Lower actuator bounds.
    u_min: [S; M],
    /// Upper actuator bounds.
    u_max: [S; M],
    /// Maximum number of gradient iterations.
    max_iter: usize,
    /// Convergence tolerance on the residual ||B u - v_des||.
    tol: S,
    /// Gradient step size = 1 / spectral_norm_est(B^T B).
    step_size: S,
    /// B^T B matrix (M×M), precomputed for efficiency.
    btb: [[S; M]; M],
}

impl<S: ControlScalar, const N: usize, const M: usize> BoundedLsAllocator<S, N, M> {
    /// Construct a new `BoundedLsAllocator`.
    ///
    /// Precomputes B^T B and the step size.
    ///
    /// # Errors
    /// - `AllocationError::InvalidBounds` if any `u_min[i] > u_max[i]`.
    /// - `AllocationError::SingularMatrix` if B is the zero matrix.
    pub fn new(
        b: [[S; M]; N],
        u_min: [S; M],
        u_max: [S; M],
        max_iter: usize,
        tol: S,
    ) -> Result<Self, AllocationError> {
        for i in 0..M {
            if u_min[i] > u_max[i] {
                return Err(AllocationError::InvalidBounds);
            }
        }

        let mut btb = [[S::ZERO; M]; M];
        for i in 0..M {
            for j in 0..M {
                btb[i][j] = b
                    .iter()
                    .fold(S::ZERO, |acc, b_row| acc + b_row[i] * b_row[j]);
            }
        }

        let spectral_est = estimate_spectral_norm::<S, M>(&btb);
        if spectral_est < S::from_f64(1e-14) {
            return Err(AllocationError::SingularMatrix);
        }
        let step_size = S::ONE / spectral_est;

        Ok(Self {
            b,
            u_min,
            u_max,
            max_iter,
            tol,
            step_size,
            btb,
        })
    }

    /// Compute the allocation for the desired virtual control `v_des`.
    ///
    /// Returns actuator vector `u` minimising ||B u - v_des||^2 subject to bounds.
    /// Warm-starts from the midpoint of [u_min, u_max].
    pub fn allocate(&self, v_des: &[S; N]) -> Result<[S; M], AllocationError> {
        let mut u = [S::ZERO; M];
        for (j, u_j) in u.iter_mut().enumerate() {
            *u_j = (self.u_min[j] + self.u_max[j]) * S::HALF;
        }

        let tol_sq = self.tol * self.tol;

        for _iter in 0..self.max_iter {
            // Residual e = B u - v_des
            let mut e = [S::ZERO; N];
            for (i, e_i) in e.iter_mut().enumerate() {
                let bu = self.b[i]
                    .iter()
                    .zip(u.iter())
                    .fold(S::ZERO, |acc, (&bij, &uj)| acc + bij * uj);
                *e_i = bu - v_des[i];
            }

            let e_sq = e.iter().fold(S::ZERO, |acc, &ei| acc + ei * ei);
            if e_sq < tol_sq {
                return Ok(u);
            }

            // Gradient step: u <- clip(u - step * B^T e, u_min, u_max)
            for (j, u_j) in u.iter_mut().enumerate() {
                let g_j = self
                    .b
                    .iter()
                    .zip(e.iter())
                    .fold(S::ZERO, |acc, (b_row, &ei)| acc + b_row[j] * ei);
                *u_j = (*u_j - self.step_size * g_j).clamp_val(self.u_min[j], self.u_max[j]);
            }
        }

        Ok(u)
    }

    /// Compute the residual ||B u - v_des|| for a given actuator vector.
    pub fn residual(&self, u: &[S; M], v_des: &[S; N]) -> S {
        let sq = self
            .b
            .iter()
            .zip(v_des.iter())
            .fold(S::ZERO, |acc, (b_row, &vd)| {
                let bu = b_row
                    .iter()
                    .zip(u.iter())
                    .fold(S::ZERO, |s, (&bij, &uj)| s + bij * uj);
                let d = bu - vd;
                acc + d * d
            });
        S::from_f64(libm::sqrt(sq.to_f64()))
    }

    /// Compute the virtual control B u.
    pub fn virtual_control(&self, u: &[S; M]) -> [S; N] {
        let mut v = [S::ZERO; N];
        for (i, v_i) in v.iter_mut().enumerate() {
            *v_i = self.b[i]
                .iter()
                .zip(u.iter())
                .fold(S::ZERO, |acc, (&bij, &uj)| acc + bij * uj);
        }
        v
    }

    /// Returns the precomputed gradient step size.
    pub fn step_size(&self) -> S {
        self.step_size
    }

    /// Returns the precomputed B^T B matrix.
    pub fn btb(&self) -> &[[S; M]; M] {
        &self.btb
    }

    /// Update the effectiveness matrix B and recompute B^T B and step size.
    ///
    /// # Errors
    /// Returns `AllocationError::SingularMatrix` if the new B is the zero matrix.
    pub fn update_b(&mut self, b: [[S; M]; N]) -> Result<(), AllocationError> {
        self.b = b;
        for i in 0..M {
            for j in 0..M {
                self.btb[i][j] = self
                    .b
                    .iter()
                    .fold(S::ZERO, |acc, b_row| acc + b_row[i] * b_row[j]);
            }
        }
        let spectral_est = estimate_spectral_norm::<S, M>(&self.btb);
        if spectral_est < S::from_f64(1e-14) {
            return Err(AllocationError::SingularMatrix);
        }
        self.step_size = S::ONE / spectral_est;
        Ok(())
    }
}

/// Estimate the spectral radius of a symmetric PSD M×M matrix using the
/// column-sum (infinity) norm: max_j Σ_i |A[i][j]|.
fn estimate_spectral_norm<S: ControlScalar, const M: usize>(a: &[[S; M]; M]) -> S {
    (0..M).fold(S::ZERO, |max_sum, j| {
        let col_sum = a.iter().fold(S::ZERO, |acc, row| {
            let v = row[j];
            acc + if v < S::ZERO { -v } else { v }
        });
        if col_sum > max_sum {
            col_sum
        } else {
            max_sum
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Feasible allocation converges to near-zero residual for exact case.
    #[test]
    fn test_feasible_exact_convergence() {
        let b = [[1.0f64, 0.0], [0.0, 1.0]];
        let u_min = [-10.0f64; 2];
        let u_max = [10.0f64; 2];

        let alloc = BoundedLsAllocator::<f64, 2, 2>::new(b, u_min, u_max, 1000, 1e-10).expect("ok");
        let v_des = [3.0f64, 4.0];
        let u = alloc.allocate(&v_des).expect("ok");

        assert!(approx(u[0], 3.0, 1e-6), "u[0]={}", u[0]);
        assert!(approx(u[1], 4.0, 1e-6), "u[1]={}", u[1]);
        assert!(alloc.residual(&u, &v_des) < 1e-6);
    }

    /// Saturated case: v_des outside attainable set → u clamped at bounds.
    #[test]
    fn test_saturated_case_handled() {
        let b = [[1.0f64, 0.0], [0.0, 1.0]];
        let u_min = [0.0f64; 2];
        let u_max = [2.0f64; 2];

        let alloc = BoundedLsAllocator::<f64, 2, 2>::new(b, u_min, u_max, 500, 1e-8).expect("ok");
        let v_des = [5.0f64, 5.0];
        let u = alloc.allocate(&v_des).expect("ok");

        for (j, &u_j) in u.iter().enumerate().take(2) {
            assert!(
                (-1e-9..=2.0 + 1e-9).contains(&u_j),
                "u[{}]={} out of bounds",
                j,
                u_j
            );
        }
        assert!(approx(u[0], 2.0, 1e-6), "u[0]={}", u[0]);
        assert!(approx(u[1], 2.0, 1e-6), "u[1]={}", u[1]);
    }

    /// Residual with more iterations is no worse than with fewer.
    #[test]
    fn test_residual_decreases() {
        let b: [[f64; 3]; 2] = [[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]];
        let u_min = [-5.0f64; 3];
        let u_max = [5.0f64; 3];
        let v_des = [2.0f64, 1.5];

        let alloc_few =
            BoundedLsAllocator::<f64, 2, 3>::new(b, u_min, u_max, 5, 1e-12).expect("ok");
        let u_few = alloc_few.allocate(&v_des).expect("ok");
        let res_few = alloc_few.residual(&u_few, &v_des);

        let alloc_many =
            BoundedLsAllocator::<f64, 2, 3>::new(b, u_min, u_max, 500, 1e-12).expect("ok");
        let u_many = alloc_many.allocate(&v_des).expect("ok");
        let res_many = alloc_many.residual(&u_many, &v_des);

        assert!(
            res_many <= res_few + 1e-10,
            "more iterations should not increase residual: res_few={} res_many={}",
            res_few,
            res_many
        );
    }

    /// Over-actuated system: feasible solution found within bounds.
    #[test]
    fn test_over_actuated_feasible() {
        let b: [[f64; 3]; 2] = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]];
        let u_min = [0.0f64; 3];
        let u_max = [5.0f64; 3];

        let alloc = BoundedLsAllocator::<f64, 2, 3>::new(b, u_min, u_max, 1000, 1e-8).expect("ok");
        let v_des = [2.0f64, 2.0];
        let u = alloc.allocate(&v_des).expect("ok");

        let res = alloc.residual(&u, &v_des);
        assert!(res < 1e-5, "residual too large: {}", res);
        for (j, &u_j) in u.iter().enumerate().take(3) {
            assert!((-1e-9..=5.0 + 1e-9).contains(&u_j), "u[{}]={}", j, u_j);
        }
    }

    /// Invalid bounds are rejected.
    #[test]
    fn test_invalid_bounds_rejected() {
        let b = [[1.0f64, 0.0], [0.0, 1.0]];
        let u_min = [5.0f64, 0.0];
        let u_max = [2.0f64, 10.0];
        assert_eq!(
            BoundedLsAllocator::<f64, 2, 2>::new(b, u_min, u_max, 100, 1e-8),
            Err(AllocationError::InvalidBounds)
        );
    }

    /// Residual function computes correct Euclidean norm.
    #[test]
    fn test_residual_function() {
        let b = [[1.0f64, 0.0], [0.0, 1.0]];
        let u_min = [-10.0f64; 2];
        let u_max = [10.0f64; 2];
        let alloc = BoundedLsAllocator::<f64, 2, 2>::new(b, u_min, u_max, 100, 1e-10).expect("ok");

        let u = [1.0f64, 2.0];
        let v_des = [2.0f64, 4.0]; // ||B u - v_des|| = ||[-1,-2]|| = sqrt(5)
        let res = alloc.residual(&u, &v_des);
        let expected = 5.0f64.sqrt();
        assert!(
            approx(res, expected, 1e-10),
            "residual={} expected={}",
            res,
            expected
        );
    }

    /// Virtual control computation is correct.
    #[test]
    fn test_virtual_control() {
        let b: [[f64; 2]; 2] = [[2.0, 0.0], [0.0, 3.0]];
        let u_min = [-10.0f64; 2];
        let u_max = [10.0f64; 2];
        let alloc = BoundedLsAllocator::<f64, 2, 2>::new(b, u_min, u_max, 100, 1e-10).expect("ok");

        let u = [1.5f64, 2.0];
        let v = alloc.virtual_control(&u);
        assert!(approx(v[0], 3.0, 1e-10));
        assert!(approx(v[1], 6.0, 1e-10));
    }

    /// update_b recomputes step size and allocation still works.
    #[test]
    fn test_update_b() {
        let b = [[1.0f64, 0.0], [0.0, 1.0]];
        let u_min = [-10.0f64; 2];
        let u_max = [10.0f64; 2];
        let mut alloc =
            BoundedLsAllocator::<f64, 2, 2>::new(b, u_min, u_max, 200, 1e-8).expect("ok");

        alloc.update_b([[2.0f64, 0.0], [0.0, 2.0]]).expect("ok");

        let v_des = [2.0f64, 4.0];
        let u = alloc.allocate(&v_des).expect("ok");
        let res = alloc.residual(&u, &v_des);
        assert!(res < 1e-5, "residual after update: {}", res);
    }
}
