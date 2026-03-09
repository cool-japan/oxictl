//! Port-Hamiltonian system representation.
//!
//! A port-Hamiltonian (pH) system is defined by:
//!   ẋ = [J(x) - R(x)] · ∂H/∂x + g(x) · u
//!   y  = g(x)ᵀ · ∂H/∂x
//!
//! where:
//!   H(x)  — Hamiltonian (stored energy)
//!   J(x)  — interconnection matrix (skew-symmetric: J = -Jᵀ)
//!   R(x)  — damping matrix (positive semi-definite)
//!   g(x)  — input/output matrix
//!
//! Passivity is guaranteed by construction: Ḣ = uᵀy - (∂H/∂x)ᵀ R ∂H/∂x ≤ uᵀy.
#![allow(clippy::needless_range_loop)]
use crate::core::matrix::{matvec, vec_dot, Matrix};
use crate::core::scalar::ControlScalar;
use crate::passivity::PassivityError;

// ---------------------------------------------------------------------------
// Helper: (N×N) matrix acting on an N-vector producing an N-vector.
// We keep everything in raw [[S;N];N] / [S;N] so no heap is required.
// ---------------------------------------------------------------------------

/// Multiply a raw N×N matrix by an N-vector.
#[inline]
fn raw_matvec<S: ControlScalar, const N: usize>(m: &[[S; N]; N], v: &[S; N]) -> [S; N] {
    core::array::from_fn(|r| {
        let mut s = S::ZERO;
        for c in 0..N {
            s += m[r][c] * v[c];
        }
        s
    })
}

/// Subtract two N×N raw matrices.
#[inline]
fn raw_mat_sub<S: ControlScalar, const N: usize>(a: &[[S; N]; N], b: &[[S; N]; N]) -> [[S; N]; N] {
    core::array::from_fn(|r| core::array::from_fn(|c| a[r][c] - b[r][c]))
}

/// Multiply an N×I raw matrix by an I-vector to produce an N-vector.
#[inline]
fn raw_matvec_ni<S: ControlScalar, const N: usize, const I: usize>(
    m: &[[S; I]; N],
    v: &[S; I],
) -> [S; N] {
    core::array::from_fn(|r| {
        let mut s = S::ZERO;
        for c in 0..I {
            s += m[r][c] * v[c];
        }
        s
    })
}

/// Multiply the transpose of an N×I matrix (i.e. Iˣᴺ) by an N-vector → I-vector.
#[inline]
fn raw_matvec_gt<S: ControlScalar, const N: usize, const I: usize>(
    m: &[[S; I]; N],
    v: &[S; N],
) -> [S; I] {
    core::array::from_fn(|c| {
        let mut s = S::ZERO;
        for r in 0..N {
            s += m[r][c] * v[r];
        }
        s
    })
}

// ---------------------------------------------------------------------------
// Check helpers
// ---------------------------------------------------------------------------

/// Return true when `m` is skew-symmetric: m[i][j] + m[j][i] == 0 ∀ i,j.
/// Uses a relative tolerance based on the maximum absolute element.
pub(crate) fn is_skew_symmetric<S: ControlScalar, const N: usize>(m: &[[S; N]; N], tol: S) -> bool {
    for i in 0..N {
        // Diagonal must be zero.
        if m[i][i].abs() > tol {
            return false;
        }
        for j in (i + 1)..N {
            if (m[i][j] + m[j][i]).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Return true when `m` is positive semi-definite via Cholesky-like check.
///
/// We use the eigenvalue lower-bound: all diagonal elements of L in L·Lᵀ must
/// be real and non-negative.  We accept matrices where the smallest diagonal
/// entry of the Cholesky factor is ≥ -tol (numerically PSD).
pub(crate) fn is_psd<S: ControlScalar, const N: usize>(m: &[[S; N]; N], tol: S) -> bool {
    // Perform Cholesky with tolerance: allow slightly negative intermediate
    // values (up to tol) to handle near-zero PSD matrices.
    let mut l = [[S::ZERO; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut sum = S::ZERO;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let d = m[i][i] - sum;
                if d < -tol {
                    return false;
                }
                // Clamp small negative values to zero before sqrt.
                let d_clamped = if d < S::ZERO { S::ZERO } else { d };
                l[i][j] = d_clamped.sqrt();
            } else {
                let diag = l[j][j];
                if diag.abs() < S::EPSILON * S::from_f64(1e4) {
                    // Near-singular diagonal: only valid if numerator ~ 0.
                    if (m[i][j] - sum).abs() > tol {
                        return false;
                    }
                    l[i][j] = S::ZERO;
                } else {
                    l[i][j] = (m[i][j] - sum) / diag;
                }
            }
        }
    }
    true
}

/// Return true when `m` is positive definite (strictly).
pub(crate) fn is_pd<S: ControlScalar, const N: usize>(m: &[[S; N]; N], tol: S) -> bool {
    let mut l = [[S::ZERO; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut sum = S::ZERO;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let d = m[i][i] - sum;
                if d <= tol {
                    return false;
                }
                l[i][j] = d.sqrt();
            } else {
                let diag = l[j][j];
                if diag.abs() < S::EPSILON * S::from_f64(1e4) {
                    return false;
                }
                l[i][j] = (m[i][j] - sum) / diag;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// PortHamiltonian
// ---------------------------------------------------------------------------

/// Nonlinear port-Hamiltonian system with state dimension N and input dimension I.
///
/// Dynamics:   ẋ = [J(x) - R(x)] · ∂H/∂x + g(x) · u
/// Output:     y  = g(x)ᵀ · ∂H/∂x
///
/// The matrices J, R, g and the functions H and ∇H are provided at construction
/// time and treated as constant (representing a fixed operating point or a
/// linearised / parameter-frozen system).  For truly state-dependent matrices
/// the caller must reconstruct a new `PortHamiltonian` at each time step.
pub struct PortHamiltonian<S: ControlScalar, const N: usize, const I: usize> {
    /// Interconnection matrix J(x) — must be skew-symmetric.
    pub j_matrix: [[S; N]; N],
    /// Damping matrix R(x) — must be positive semi-definite.
    pub r_matrix: [[S; N]; N],
    /// Input matrix g(x) — maps inputs to state space (N×I).
    pub g_matrix: [[S; I]; N],
    /// Hamiltonian H: ℝᴺ → ℝ.
    pub hamiltonian: fn(&[S; N]) -> S,
    /// Gradient ∂H/∂x: ℝᴺ → ℝᴺ.
    pub grad_hamiltonian: fn(&[S; N]) -> [S; N],
}

impl<S: ControlScalar, const N: usize, const I: usize> PortHamiltonian<S, N, I> {
    /// Compute the state derivative ẋ = (J - R)·∂H/∂x + g·u.
    pub fn dynamics(&self, x: &[S; N], u: &[S; I]) -> [S; N] {
        let grad = (self.grad_hamiltonian)(x);
        // (J - R) applied to grad
        let jr = raw_mat_sub(&self.j_matrix, &self.r_matrix);
        let jrg = raw_matvec(&jr, &grad);
        // g·u
        let gu = raw_matvec_ni(&self.g_matrix, u);
        core::array::from_fn(|i| jrg[i] + gu[i])
    }

    /// Compute the output y = gᵀ·∂H/∂x  (I-vector).
    pub fn output(&self, x: &[S; N]) -> [S; I] {
        let grad = (self.grad_hamiltonian)(x);
        raw_matvec_gt(&self.g_matrix, &grad)
    }

    /// Passivity is guaranteed by construction for any valid pH system.
    ///
    /// Specifically Ḣ = uᵀy - (∂H/∂x)ᵀ R (∂H/∂x) ≤ uᵀy.
    /// This method validates J skew-symmetric and R PSD.
    pub fn is_passive(&self) -> bool {
        let tol = S::from_f64(1e-9);
        is_skew_symmetric(&self.j_matrix, tol) && is_psd(&self.r_matrix, tol)
    }

    /// Compute Ḣ = (∂H/∂x)ᵀ · ẋ.
    ///
    /// For a valid pH system this equals uᵀy − (∂H/∂x)ᵀ R ∂H/∂x.
    pub fn storage_function_rate(&self, x: &[S; N], u: &[S; I]) -> S {
        let grad = (self.grad_hamiltonian)(x);
        let xdot = self.dynamics(x, u);
        vec_dot(&grad, &xdot)
    }

    /// Supply rate: uᵀy — how much power is injected from outside.
    pub fn supply_rate(&self, x: &[S; N], u: &[S; I]) -> S {
        let y = self.output(x);
        vec_dot(u, &y)
    }

    /// Dissipation: (∂H/∂x)ᵀ R ∂H/∂x ≥ 0.
    pub fn dissipation_rate(&self, x: &[S; N]) -> S {
        let grad = (self.grad_hamiltonian)(x);
        let rg = raw_matvec(&self.r_matrix, &grad);
        vec_dot(&grad, &rg)
    }
}

// ---------------------------------------------------------------------------
// LinearPh
// ---------------------------------------------------------------------------

/// Linear port-Hamiltonian system with constant J, R, g, Q.
///
/// H(x) = ½ xᵀQx,  ∇H = Qx.
/// Dynamics: ẋ = (J - R)Qx + g·u.
/// Output:   y  = gᵀQx.
pub struct LinearPh<S: ControlScalar, const N: usize, const I: usize> {
    /// Interconnection matrix (skew-symmetric).
    pub j_matrix: Matrix<S, N, N>,
    /// Damping matrix (PSD).
    pub r_matrix: Matrix<S, N, N>,
    /// Energy weighting matrix Q (positive definite).  H = ½ xᵀQx.
    pub q_matrix: Matrix<S, N, N>,
    /// Input matrix (N×I).
    pub g_matrix: Matrix<S, N, I>,
}

impl<S: ControlScalar, const N: usize, const I: usize> LinearPh<S, N, I> {
    /// Construct and validate a linear pH system.
    ///
    /// Validates:
    ///   - R ≥ 0 (positive semi-definite)
    ///   - Q > 0 (positive definite — required for H to be a storage function)
    ///   - J skew-symmetric
    pub fn new(
        j: [[S; N]; N],
        r: [[S; N]; N],
        q: [[S; N]; N],
        g: [[S; I]; N],
    ) -> Result<Self, PassivityError> {
        let tol = S::from_f64(1e-9);
        if !is_skew_symmetric(&j, tol) {
            return Err(PassivityError::NotPassive);
        }
        if !is_psd(&r, tol) {
            return Err(PassivityError::NotPassive);
        }
        if !is_pd(&q, tol) {
            return Err(PassivityError::InvalidHamiltonian);
        }

        let j_mat = Matrix { data: j };
        let r_mat = Matrix { data: r };
        let q_mat = Matrix { data: q };
        let g_mat = Matrix { data: g };

        Ok(Self {
            j_matrix: j_mat,
            r_matrix: r_mat,
            q_matrix: q_mat,
            g_matrix: g_mat,
        })
    }

    /// Evaluate Hamiltonian H(x) = ½ xᵀQx.
    pub fn hamiltonian(&self, x: &[S; N]) -> S {
        let qx = matvec(&self.q_matrix, x);
        S::HALF * vec_dot(x, &qx)
    }

    /// Evaluate ∇H(x) = Qx.
    pub fn grad_hamiltonian(&self, x: &[S; N]) -> [S; N] {
        matvec(&self.q_matrix, x)
    }

    /// Compute ẋ = (J - R)Qx + gu.
    pub fn dynamics(&self, x: &[S; N], u: &[S; I]) -> [S; N] {
        use crate::core::matrix::vec_add;
        let grad = self.grad_hamiltonian(x);
        // (J - R)
        let jr = self.j_matrix.sub_mat(&self.r_matrix);
        let jrg = matvec(&jr, &grad);
        let gu = matvec(&self.g_matrix, u);
        vec_add(&jrg, &gu)
    }

    /// Compute output y = gᵀQx.
    pub fn output(&self, x: &[S; N]) -> [S; I] {
        let grad = self.grad_hamiltonian(x);
        let gt = self.g_matrix.transpose();
        matvec(&gt, &grad)
    }

    /// Convert to nonlinear `PortHamiltonian` representation (using closures
    /// is not possible in no_std with fn pointers, so we return the raw arrays).
    pub fn j_raw(&self) -> [[S; N]; N] {
        self.j_matrix.data
    }

    pub fn r_raw(&self) -> [[S; N]; N] {
        self.r_matrix.data
    }

    pub fn g_raw(&self) -> [[S; I]; N] {
        self.g_matrix.data
    }

    pub fn q_raw(&self) -> [[S; N]; N] {
        self.q_matrix.data
    }

    /// Check that this linear pH system satisfies passivity conditions.
    pub fn is_passive(&self) -> bool {
        let tol = S::from_f64(1e-9);
        is_skew_symmetric(&self.j_matrix.data, tol) && is_psd(&self.r_matrix.data, tol)
    }

    /// Compute Ḣ = (Qx)ᵀ ẋ.
    pub fn storage_function_rate(&self, x: &[S; N], u: &[S; I]) -> S {
        let grad = self.grad_hamiltonian(x);
        let xdot = self.dynamics(x, u);
        vec_dot(&grad, &xdot)
    }

    /// Dissipation rate (Qx)ᵀ R (Qx) ≥ 0.
    pub fn dissipation_rate(&self, x: &[S; N]) -> S {
        let grad = self.grad_hamiltonian(x);
        let rg = matvec(&self.r_matrix, &grad);
        vec_dot(&grad, &rg)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Mass-spring-damper as a linear pH system
    //
    // State: x = [q, p]  (position, momentum)  — N=2, I=1
    // H(x) = p²/(2m) + k·q²/2  →  Q = diag(k, 1/m)  [scaled to unit mass m=1, k=1]
    //        With m=1, k=1: Q = I₂ (identity for simplicity)
    //
    // Dynamics:  q̇ = p/m = p,   ṗ = -k·q - b·p + u
    //
    // pH form with Q = diag(k, 1/m) = I (k=m=1):
    //   J = [[0, 1],[-1, 0]]   (skew-sym)
    //   R = [[0, 0],[0,  b]]   (PSD, b≥0)
    //   g = [[0],[1]]          (force input on momentum)
    //   ∇H = Qx = [q, p]
    //
    //   ẋ = (J-R)∇H + g·u
    //     = [[0,1],[-1,-b]]·[q,p]ᵀ + [0,u]ᵀ
    //     = [p, -q - b·p + u]ᵀ   ✓
    // -----------------------------------------------------------------------

    fn make_msd_linear_ph(b: f64) -> LinearPh<f64, 2, 1> {
        let j = [[0.0f64, 1.0], [-1.0, 0.0]];
        let r = [[0.0f64, 0.0], [0.0, b]];
        let q = [[1.0f64, 0.0], [0.0, 1.0]]; // identity (k=m=1)
        let g = [[0.0f64], [1.0]];
        LinearPh::new(j, r, q, g).expect("Valid mass-spring-damper pH system")
    }

    #[test]
    fn linear_ph_construction_succeeds() {
        let _sys = make_msd_linear_ph(0.5);
    }

    #[test]
    fn linear_ph_invalid_j_not_skew() {
        // J is NOT skew-symmetric
        let j = [[1.0f64, 1.0], [1.0, 0.0]];
        let r = [[0.0f64, 0.0], [0.0, 0.1]];
        let q = [[1.0f64, 0.0], [0.0, 1.0]];
        let g = [[0.0f64], [1.0]];
        let result = LinearPh::<f64, 2, 1>::new(j, r, q, g);
        assert!(matches!(result, Err(PassivityError::NotPassive)));
    }

    #[test]
    fn linear_ph_invalid_r_not_psd() {
        // R has a negative diagonal
        let j = [[0.0f64, 1.0], [-1.0, 0.0]];
        let r = [[0.0f64, 0.0], [0.0, -0.5]]; // negative damping
        let q = [[1.0f64, 0.0], [0.0, 1.0]];
        let g = [[0.0f64], [1.0]];
        let result = LinearPh::<f64, 2, 1>::new(j, r, q, g);
        assert!(matches!(result, Err(PassivityError::NotPassive)));
    }

    #[test]
    fn linear_ph_invalid_q_not_pd() {
        // Q is singular (not positive definite)
        let j = [[0.0f64, 1.0], [-1.0, 0.0]];
        let r = [[0.0f64, 0.0], [0.0, 0.5]];
        let q = [[0.0f64, 0.0], [0.0, 1.0]]; // singular
        let g = [[0.0f64], [1.0]];
        let result = LinearPh::<f64, 2, 1>::new(j, r, q, g);
        assert!(matches!(result, Err(PassivityError::InvalidHamiltonian)));
    }

    #[test]
    fn linear_ph_is_passive() {
        let sys = make_msd_linear_ph(0.5);
        assert!(sys.is_passive(), "MSD pH system should be passive");
    }

    #[test]
    fn linear_ph_hamiltonian_positive() {
        let sys = make_msd_linear_ph(0.5);
        let x = [1.0f64, 2.0]; // q=1, p=2
        let h = sys.hamiltonian(&x);
        // H = 0.5*(1² + 2²) = 2.5
        assert!((h - 2.5).abs() < 1e-12, "H={}", h);
    }

    #[test]
    fn linear_ph_dynamics_correct() {
        let sys = make_msd_linear_ph(0.3);
        let x = [1.0f64, 2.0]; // q=1, p=2
        let u = [0.5f64];
        let xdot = sys.dynamics(&x, &u);
        // ẋ = [p, -q - b·p + u] = [2, -1 - 0.3·2 + 0.5] = [2, -1.1]
        assert!((xdot[0] - 2.0).abs() < 1e-12, "q̇={}", xdot[0]);
        assert!((xdot[1] - (-1.1)).abs() < 1e-12, "ṗ={}", xdot[1]);
    }

    #[test]
    fn linear_ph_output_correct() {
        let sys = make_msd_linear_ph(0.3);
        let x = [1.0f64, 2.0];
        let y = sys.output(&x);
        // y = gᵀQx = gᵀ[q, p] = p = 2.0  (since g = [0; 1])
        assert!((y[0] - 2.0).abs() < 1e-12, "y={}", y[0]);
    }

    #[test]
    fn passivity_inequality_holds() {
        // Check Ḣ ≤ uᵀy at several operating points.
        let b = 0.5;
        let sys = make_msd_linear_ph(b);
        for &(q, p, u_val) in &[
            (1.0, 0.5, 0.0),
            (0.5, 1.0, 1.0),
            (0.0, 0.0, 0.5),
            (-1.0, 2.0, -0.3),
        ] {
            let x = [q, p];
            let u = [u_val];
            let hdot = sys.storage_function_rate(&x, &u);
            let supply = sys.dissipation_rate(&x);
            // Ḣ = uᵀy - (∇H)ᵀR(∇H) → Ḣ + dissipation = uᵀy
            let y = sys.output(&x);
            let uty = u[0] * y[0];
            // Verify Ḣ ≤ uᵀy
            assert!(
                hdot <= uty + 1e-10,
                "Passivity violated: Ḣ={:.6} > uᵀy={:.6} (q={}, p={}, u={})",
                hdot,
                uty,
                q,
                p,
                u_val
            );
            // Verify Ḣ = uᵀy - dissipation
            assert!(
                (hdot - (uty - supply)).abs() < 1e-10,
                "Energy balance off: Ḣ={:.6}, uᵀy-D={:.6}",
                hdot,
                uty - supply
            );
        }
    }

    #[test]
    fn nonlinear_ph_dynamics() {
        // Use the same MSD system but via the nonlinear PortHamiltonian wrapper.
        let j_raw = [[0.0f64, 1.0], [-1.0, 0.0]];
        let r_raw = [[0.0f64, 0.0], [0.0, 0.3]];
        let g_raw = [[0.0f64], [1.0]];

        fn ham(x: &[f64; 2]) -> f64 {
            0.5 * (x[0] * x[0] + x[1] * x[1])
        }
        fn grad_ham(x: &[f64; 2]) -> [f64; 2] {
            [x[0], x[1]]
        }

        let sys = PortHamiltonian {
            j_matrix: j_raw,
            r_matrix: r_raw,
            g_matrix: g_raw,
            hamiltonian: ham,
            grad_hamiltonian: grad_ham,
        };

        assert!(sys.is_passive());
        let x = [1.0f64, 2.0];
        let u = [0.5f64];
        let xdot = sys.dynamics(&x, &u);
        // [p, -q - 0.3p + u] = [2, -1 - 0.6 + 0.5] = [2, -1.1]
        assert!((xdot[0] - 2.0).abs() < 1e-12);
        assert!((xdot[1] - (-1.1)).abs() < 1e-12);
    }

    #[test]
    fn storage_function_rate_zero_input_negative() {
        // With u=0, Ḣ = -(∇H)ᵀR(∇H) ≤ 0 (dissipation only).
        let sys = make_msd_linear_ph(0.5);
        let x = [1.0f64, 2.0];
        let u = [0.0f64];
        let hdot = sys.storage_function_rate(&x, &u);
        assert!(hdot <= 1e-12, "Ḣ with u=0 should be ≤ 0, got {}", hdot);
    }
}
