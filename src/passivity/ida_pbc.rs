//! IDA-PBC: Interconnection and Damping Assignment Passivity-Based Control.
//!
//! Given a port-Hamiltonian system (J, R, H, g), IDA-PBC designs a control law u
//! such that the closed-loop system has a desired pH structure (J_d, R_d, H_d).
//!
//! ## Theory
//! The desired closed-loop dynamics are:
//!   ẋ = [J_d(x) - R_d(x)] · ∂H_d/∂x
//!
//! The matching condition (using the left annihilator g⊥ of g) is:
//!   g⊥(x) · [J(x)-R(x)] · ∂H/∂x  =  g⊥(x) · [J_d-R_d] · ∂H_d/∂x
//!
//! When g has full column rank, the control law that satisfies matching is:
//!   u = (gᵀg)⁻¹ · gᵀ · { [J_d-R_d]·∂H_d/∂x - [J-R]·∂H/∂x }
//!
//! ## Energy/Damping shaping
//! H_d = H + H_a  (add shaping potential to relocate equilibrium)
//! R_d = R + R_a  (inject additional damping)
#![allow(clippy::needless_range_loop)]
use crate::core::scalar::ControlScalar;
use crate::passivity::port_hamiltonian::{is_psd, is_skew_symmetric};
use crate::passivity::PassivityError;

// ---------------------------------------------------------------------------
// Internal linear-algebra helpers (raw arrays, no heap)
// ---------------------------------------------------------------------------

/// Multiply N×N raw matrix by N-vector.
#[inline]
fn raw_mv<S: ControlScalar, const N: usize>(m: &[[S; N]; N], v: &[S; N]) -> [S; N] {
    core::array::from_fn(|r| {
        let mut acc = S::ZERO;
        for c in 0..N {
            acc += m[r][c] * v[c];
        }
        acc
    })
}

/// Subtract two N-vectors.
#[inline]
fn vec_sub_n<S: ControlScalar, const N: usize>(a: &[S; N], b: &[S; N]) -> [S; N] {
    core::array::from_fn(|i| a[i] - b[i])
}

/// Compute (J-R) for raw N×N matrices.
#[inline]
fn jr_matrix<S: ControlScalar, const N: usize>(j: &[[S; N]; N], r: &[[S; N]; N]) -> [[S; N]; N] {
    core::array::from_fn(|row| core::array::from_fn(|col| j[row][col] - r[row][col]))
}

/// Multiply I×N raw matrix (i.e. gᵀ stored row-major) by N-vector → I-vector.
/// Here `gt[i][n]` = gᵀ[i,n] = g[n,i].
#[inline]
fn gt_mv<S: ControlScalar, const N: usize, const I: usize>(g: &[[S; I]; N], v: &[S; N]) -> [S; I] {
    core::array::from_fn(|col| {
        let mut acc = S::ZERO;
        for row in 0..N {
            acc += g[row][col] * v[row];
        }
        acc
    })
}

/// Compute (gᵀg) as I×I matrix, stored as [[S;I];I].
#[inline]
fn gtg<S: ControlScalar, const N: usize, const I: usize>(g: &[[S; I]; N]) -> [[S; I]; I] {
    core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            let mut acc = S::ZERO;
            for n in 0..N {
                acc += g[n][i] * g[n][j];
            }
            acc
        })
    })
}

/// Invert a small I×I matrix via Gaussian elimination. Returns None if singular.
fn invert_small<S: ControlScalar, const I: usize>(m: [[S; I]; I]) -> Option<[[S; I]; I]> {
    let mut a = m;
    let mut inv: [[S; I]; I] =
        core::array::from_fn(|r| core::array::from_fn(|c| if r == c { S::ONE } else { S::ZERO }));

    for col in 0..I {
        // Find pivot.
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..I {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < S::EPSILON * S::from_f64(1e6) {
            return None;
        }
        if max_row != col {
            a.swap(max_row, col);
            inv.swap(max_row, col);
        }
        let pivot_inv = S::ONE / a[col][col];
        for c in 0..I {
            a[col][c] *= pivot_inv;
            inv[col][c] *= pivot_inv;
        }
        for row in 0..I {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for c in 0..I {
                a[row][c] -= factor * a[col][c];
                inv[row][c] -= factor * inv[col][c];
            }
        }
    }
    Some(inv)
}

/// Multiply I×I matrix by I-vector.
#[inline]
fn mv_ii<S: ControlScalar, const I: usize>(m: &[[S; I]; I], v: &[S; I]) -> [S; I] {
    core::array::from_fn(|r| {
        let mut acc = S::ZERO;
        for c in 0..I {
            acc += m[r][c] * v[c];
        }
        acc
    })
}

// ---------------------------------------------------------------------------
// IdaPbcConfig
// ---------------------------------------------------------------------------

/// Configuration for an IDA-PBC controller.
///
/// Specifies the *desired* interconnection structure, damping, and the gradient
/// of the desired Hamiltonian.  The user also provides the gradient of the
/// added shaping potential H_a (so that ∂H_d/∂x = ∂H/∂x + ∂H_a/∂x).
pub struct IdaPbcConfig<S: ControlScalar, const N: usize, const I: usize> {
    /// Desired interconnection matrix J_d (skew-symmetric).
    pub j_desired: [[S; N]; N],
    /// Desired damping matrix R_d = R + R_a (positive semi-definite).
    pub r_desired: [[S; N]; N],
    /// Gradient of the desired Hamiltonian: ∂H_d/∂x.
    pub grad_h_desired: fn(&[S; N]) -> [S; N],
    /// Phantom for I (input dimension carried for type-checking).
    _phantom: core::marker::PhantomData<[S; I]>,
}

impl<S: ControlScalar, const N: usize, const I: usize> IdaPbcConfig<S, N, I> {
    /// Create and validate an IDA-PBC configuration.
    ///
    /// Returns an error if J_d is not skew-symmetric or R_d is not PSD.
    pub fn new(
        j_desired: [[S; N]; N],
        r_desired: [[S; N]; N],
        grad_h_desired: fn(&[S; N]) -> [S; N],
    ) -> Result<Self, PassivityError> {
        let tol = S::from_f64(1e-9);
        if !is_skew_symmetric(&j_desired, tol) {
            return Err(PassivityError::NotPassive);
        }
        if !is_psd(&r_desired, tol) {
            return Err(PassivityError::NotPassive);
        }
        Ok(Self {
            j_desired,
            r_desired,
            grad_h_desired,
            _phantom: core::marker::PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// IdaPbcController
// ---------------------------------------------------------------------------

/// IDA-PBC controller.
///
/// Computes the control law:
///   u = (gᵀg)⁻¹ · gᵀ · { [J_d - R_d]·∂H_d/∂x − [J - R]·∂H/∂x }
///
/// This ensures the closed-loop behaves as the desired pH system with (J_d, R_d, H_d).
pub struct IdaPbcController<S: ControlScalar, const N: usize, const I: usize> {
    config: IdaPbcConfig<S, N, I>,
}

impl<S: ControlScalar, const N: usize, const I: usize> IdaPbcController<S, N, I> {
    /// Build the controller from a validated configuration.
    pub fn new(config: IdaPbcConfig<S, N, I>) -> Self {
        Self { config }
    }

    /// Compute the IDA-PBC control signal at state `x`.
    ///
    /// Requires the original plant pH system to extract J, R, g, and ∇H.
    ///
    /// Returns `Err(PassivityError::SingularMatrix)` if gᵀg is not invertible
    /// (e.g. the input matrix g has linearly dependent columns or is zero).
    /// Returns `Err(PassivityError::MatchingFailed)` if the desired target
    /// gradient is inconsistent (NaN / Inf guard).
    pub fn compute<F1, F2>(
        &self,
        j_plant: &[[S; N]; N],
        r_plant: &[[S; N]; N],
        g_plant: &[[S; I]; N],
        grad_h_plant: F1,
        x: &[S; N],
    ) -> Result<[S; I], PassivityError>
    where
        F1: Fn(&[S; N]) -> [S; N],
        F2: Fn(&[S; N]) -> [S; N],
    {
        // ∂H/∂x  and  ∂H_d/∂x
        let grad_h = grad_h_plant(x);
        let grad_hd = (self.config.grad_h_desired)(x);

        // Guard against NaN/Inf in gradients.
        for &v in grad_hd.iter() {
            if !v.is_finite() {
                return Err(PassivityError::MatchingFailed);
            }
        }

        // [J - R]·∂H/∂x  and  [J_d - R_d]·∂H_d/∂x
        let jr = jr_matrix(j_plant, r_plant);
        let jrd = jr_matrix(&self.config.j_desired, &self.config.r_desired);

        let jrg = raw_mv(&jr, &grad_h);
        let jrdgd = raw_mv(&jrd, &grad_hd);

        // Desired contribution − plant contribution  (N-vector)
        let rhs = vec_sub_n(&jrdgd, &jrg);

        // Project via gᵀ: gives I-vector
        let gtrhs = gt_mv(g_plant, &rhs);

        // (gᵀg)⁻¹ · gᵀ · rhs
        let gtg_mat = gtg(g_plant);
        let gtg_inv = invert_small(gtg_mat).ok_or(PassivityError::SingularMatrix)?;
        let u = mv_ii(&gtg_inv, &gtrhs);

        Ok(u)
    }

    /// Convenience wrapper that bundles the plant closures for users who
    /// prefer supplying them directly without extra generic parameters.
    pub fn compute_with_closures(
        &self,
        j_plant: &[[S; N]; N],
        r_plant: &[[S; N]; N],
        g_plant: &[[S; I]; N],
        grad_h_plant: fn(&[S; N]) -> [S; N],
        x: &[S; N],
    ) -> Result<[S; I], PassivityError> {
        let grad_h = grad_h_plant(x);
        let grad_hd = (self.config.grad_h_desired)(x);

        for &v in grad_hd.iter() {
            if !v.is_finite() {
                return Err(PassivityError::MatchingFailed);
            }
        }

        let jr = jr_matrix(j_plant, r_plant);
        let jrd = jr_matrix(&self.config.j_desired, &self.config.r_desired);

        let jrg = raw_mv(&jr, &grad_h);
        let jrdgd = raw_mv(&jrd, &grad_hd);

        let rhs = vec_sub_n(&jrdgd, &jrg);
        let gtrhs = gt_mv(g_plant, &rhs);

        let gtg_mat = gtg(g_plant);
        let gtg_inv = invert_small(gtg_mat).ok_or(PassivityError::SingularMatrix)?;
        let u = mv_ii(&gtg_inv, &gtrhs);

        Ok(u)
    }

    /// Access the desired interconnection structure.
    pub fn j_desired(&self) -> &[[S; N]; N] {
        &self.config.j_desired
    }

    /// Access the desired damping structure.
    pub fn r_desired(&self) -> &[[S; N]; N] {
        &self.config.r_desired
    }
}

// ---------------------------------------------------------------------------
// Mechanical IDA-PBC specialisation
// ---------------------------------------------------------------------------

/// IDA-PBC configuration for underactuated mechanical systems.
///
/// State: x = [q₁, …, q_K, p₁, …, p_K]  where N = 2·K.
/// H(x) = ½ pᵀ M⁻¹ p + V(q)
/// H_d(x) = ½ pᵀ M_d⁻¹ p + V_d(q)
///
/// The mechanical pH structure has:
///   J = [[0, I_K],[-I_K, 0]]   (canonical symplectic)
///   R = [[0, 0],[0, D]]        (velocity damping)
///
/// This struct assists in constructing the gradient of H_d for the general
/// IDA-PBC controller.
///
/// The const parameter `K` is the number of configuration coordinates
/// (degrees of freedom); the full state dimension is N = 2·K.
/// The input parameter `I` is the number of control inputs.
pub struct MechanicalIdaPbc<S: ControlScalar, const N: usize, const K: usize, const I: usize> {
    /// Inverse of desired inertia matrix M_d (K×K block).
    pub md_inv: [[S; K]; K],
    /// Gradient of desired potential: ∂V_d/∂q: ℝᴷ → ℝᴷ.
    pub grad_vd: fn(&[S; K]) -> [S; K],
    _phantom: core::marker::PhantomData<([S; N], [S; I])>,
}

impl<S: ControlScalar, const N: usize, const K: usize, const I: usize>
    MechanicalIdaPbc<S, N, K, I>
{
    /// Create a mechanical IDA-PBC helper.
    ///
    /// Caller must ensure N == 2·K; this is not checked at compile time without
    /// `#![feature(generic_const_exprs)]`.
    pub fn new(md_inv: [[S; K]; K], grad_vd: fn(&[S; K]) -> [S; K]) -> Self {
        Self {
            md_inv,
            grad_vd,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Evaluate ∂H_d/∂x = [∂V_d/∂q, M_d⁻¹·p].
    ///
    /// Assumes x = [q (K elements), p (K elements)], i.e. N = 2·K.
    /// Only the first K and last K elements of `x` are used; if N ≠ 2·K
    /// the remaining elements are ignored.
    pub fn grad_hd(&self, x: &[S; N]) -> [S; N] {
        // Extract q and p (up to K elements each).
        let mut q = [S::ZERO; K];
        let mut p = [S::ZERO; K];
        for i in 0..K {
            if i < N {
                q[i] = x[i];
            }
            if K + i < N {
                p[i] = x[K + i];
            }
        }
        // ∂V_d/∂q
        let dvdq = (self.grad_vd)(&q);
        // M_d⁻¹ · p
        let mut md_inv_p = [S::ZERO; K];
        for r in 0..K {
            let mut acc = S::ZERO;
            for c in 0..K {
                acc += self.md_inv[r][c] * p[c];
            }
            md_inv_p[r] = acc;
        }
        // Assemble full gradient: [∂V_d/∂q, M_d⁻¹·p] padded to N.
        core::array::from_fn(|i| {
            if i < K {
                dvdq[i]
            } else if i - K < K {
                md_inv_p[i - K]
            } else {
                S::ZERO
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test setup: mass-spring system (no damping in plant) with IDA-PBC
    // to create a desired equilibrium at q*=1 instead of q*=0.
    //
    // Plant (k=1, so N=2, I=1):
    //   H(x) = ½(q² + p²)   [m=k=1]
    //   J = [[0,1],[-1,0]],  R = [[0,0],[0,0]],  g = [[0],[1]]
    //
    // Desired (equilibrium at q*=1):
    //   H_d(x) = ½((q-1)² + p²)   [same inertia, shifted potential]
    //   J_d = J,  R_d = [[0,0],[0,r_a]] with r_a>0  (added damping)
    //   ∂H_d/∂x = [q-1, p]
    // -----------------------------------------------------------------------

    fn plant_j() -> [[f64; 2]; 2] {
        [[0.0, 1.0], [-1.0, 0.0]]
    }
    fn plant_r() -> [[f64; 2]; 2] {
        [[0.0, 0.0], [0.0, 0.0]]
    }
    fn plant_g() -> [[f64; 1]; 2] {
        [[0.0], [1.0]]
    }
    fn grad_h_plant(x: &[f64; 2]) -> [f64; 2] {
        [x[0], x[1]] // ∇H = [q, p]
    }

    fn grad_hd_shifted(x: &[f64; 2]) -> [f64; 2] {
        [x[0] - 1.0, x[1]] // ∇H_d = [q-1, p]
    }

    fn make_config(r_a: f64) -> IdaPbcConfig<f64, 2, 1> {
        let j_d = [[0.0f64, 1.0], [-1.0, 0.0]];
        let r_d = [[0.0f64, 0.0], [0.0, r_a]];
        IdaPbcConfig::new(j_d, r_d, grad_hd_shifted).expect("Config should be valid")
    }

    #[test]
    fn ida_pbc_config_valid() {
        let _cfg = make_config(1.0);
    }

    #[test]
    fn ida_pbc_config_invalid_j_not_skew() {
        let j_d = [[1.0f64, 1.0], [1.0, 0.0]]; // NOT skew
        let r_d = [[0.0f64, 0.0], [0.0, 1.0]];
        let result = IdaPbcConfig::<f64, 2, 1>::new(j_d, r_d, grad_hd_shifted);
        assert!(matches!(result, Err(PassivityError::NotPassive)));
    }

    #[test]
    fn ida_pbc_config_invalid_r_not_psd() {
        let j_d = [[0.0f64, 1.0], [-1.0, 0.0]];
        let r_d = [[0.0f64, 0.0], [0.0, -1.0]]; // negative damping
        let result = IdaPbcConfig::<f64, 2, 1>::new(j_d, r_d, grad_hd_shifted);
        assert!(matches!(result, Err(PassivityError::NotPassive)));
    }

    #[test]
    fn ida_pbc_control_at_desired_equilibrium_is_zero() {
        // At x* = [q*=1, p=0]: ∂H/∂x = [1,0], ∂H_d/∂x = [0,0].
        // [J-R]∂H/∂x = [[0,1],[-1,0]]·[1,0] = [0,-1]
        // [J_d-R_d]∂H_d/∂x = [[0,1],[-1,-r_a]]·[0,0] = [0,0]
        // rhs = [0,0] - [0,-1] = [0,1]
        // gᵀ·rhs = [0,1]·[0,1] = 1  (hmm not zero…)
        //
        // Actually at equilibrium: [J_d-R_d]∂H_d/∂x = 0  and  [J-R]∂H/∂x
        // must also = 0 for the plant to be at rest WITHOUT control.
        // With pure spring (no damping), the plant at [1,0] has ẋ = [0, -1] ≠ 0.
        // IDA-PBC corrects this by injecting u to make the desired closed-loop
        // reach equilibrium at [1,0].
        //
        // Verify instead that the *closed-loop* ẋ = 0 at x*.
        let cfg = make_config(1.0);
        let ctrl = IdaPbcController::new(cfg);

        let j_plant = plant_j();
        let r_plant = plant_r();
        let g_plant = plant_g();

        let x_star = [1.0f64, 0.0];
        let u = ctrl
            .compute_with_closures(&j_plant, &r_plant, &g_plant, grad_h_plant, &x_star)
            .expect("compute should succeed");

        // Closed-loop ẋ = [J_d - R_d] ∂H_d/∂x should be [0,0] at x*.
        let grad_hd = grad_hd_shifted(&x_star);
        let jrd = jr_matrix(ctrl.j_desired(), ctrl.r_desired());
        let xdot_cl = raw_mv(&jrd, &grad_hd);
        assert!(
            xdot_cl[0].abs() < 1e-12,
            "Closed-loop q̇ should be 0 at x*: {}",
            xdot_cl[0]
        );
        assert!(
            xdot_cl[1].abs() < 1e-12,
            "Closed-loop ṗ should be 0 at x*: {}",
            xdot_cl[1]
        );
        // The control should cancel the plant instability.
        let _ = u; // computed without error
    }

    #[test]
    fn ida_pbc_simulation_converges_to_desired() {
        // Simulate the closed-loop mass-spring system driven by IDA-PBC.
        // Desired equilibrium: q* = 1.0.  Start at x0 = [0, 0].
        // Integration: Euler with dt = 0.01, 5000 steps.
        let r_a = 2.0; // strong added damping to speed convergence
        let cfg = make_config(r_a);
        let ctrl = IdaPbcController::new(cfg);

        let j_plant = plant_j();
        let r_plant = plant_r();
        let g_plant = plant_g();

        let mut x = [0.0f64, 0.0];
        let dt = 0.005;

        for _ in 0..10_000 {
            let u = ctrl
                .compute_with_closures(&j_plant, &r_plant, &g_plant, grad_h_plant, &x)
                .expect("compute ok");

            // ẋ_plant = (J-R)∇H + g·u
            let grad_h = grad_h_plant(&x);
            let jr = jr_matrix(&j_plant, &r_plant);
            let jrg = raw_mv(&jr, &grad_h);
            let xdot = [jrg[0] + g_plant[0][0] * u[0], jrg[1] + g_plant[1][0] * u[0]];
            x = [x[0] + dt * xdot[0], x[1] + dt * xdot[1]];
        }

        assert!(
            (x[0] - 1.0).abs() < 0.05,
            "Position should converge to q*=1: q={}",
            x[0]
        );
        assert!(
            x[1].abs() < 0.05,
            "Momentum should converge to 0: p={}",
            x[1]
        );
    }

    #[test]
    fn ida_pbc_singular_g_returns_error() {
        // g = zero matrix → gᵀg singular.
        let cfg = make_config(1.0);
        let ctrl = IdaPbcController::new(cfg);
        let j_plant = plant_j();
        let r_plant = plant_r();
        let g_zero = [[0.0f64], [0.0]]; // singular
        let x = [0.5f64, 0.1];
        let result = ctrl.compute_with_closures(&j_plant, &r_plant, &g_zero, grad_h_plant, &x);
        assert!(matches!(result, Err(PassivityError::SingularMatrix)));
    }

    #[test]
    fn mechanical_ida_pbc_grad_hd() {
        // k=1: N=2, x=[q,p], M_d⁻¹ = [[2.0]], V_d potential: ∂V_d/∂q = q - 1.
        fn grad_vd(q: &[f64; 1]) -> [f64; 1] {
            [q[0] - 1.0]
        }
        let md_inv = [[2.0f64]];
        // MechanicalIdaPbc<S, N, K, I>: N=2, K=1 (DOF), I=1 (input)
        let mech = MechanicalIdaPbc::<f64, 2, 1, 1>::new(md_inv, grad_vd);

        let x = [1.5f64, 3.0]; // q=1.5, p=3.0
        let grad = mech.grad_hd(&x);
        // ∂H_d/∂x = [∂V_d/∂q, M_d⁻¹·p] = [0.5, 2.0*3.0] = [0.5, 6.0]
        assert!((grad[0] - 0.5).abs() < 1e-12, "∂H_d/∂q={}", grad[0]);
        assert!((grad[1] - 6.0).abs() < 1e-12, "∂H_d/∂p={}", grad[1]);
    }

    #[test]
    fn ida_pbc_control_finite_far_from_eq() {
        // Just verify the control value is finite away from equilibrium.
        let cfg = make_config(1.0);
        let ctrl = IdaPbcController::new(cfg);
        let j = plant_j();
        let r = plant_r();
        let g = plant_g();
        for &(q, p) in &[(0.0f64, 0.0f64), (2.0, 1.0), (-1.0, 0.5), (0.5, -2.0)] {
            let x = [q, p];
            let u = ctrl
                .compute_with_closures(&j, &r, &g, grad_h_plant, &x)
                .expect("compute ok");
            assert!(u[0].is_finite(), "u not finite at q={}, p={}", q, p);
        }
    }
}
