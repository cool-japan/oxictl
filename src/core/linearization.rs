/// Numerical linearization of nonlinear systems via central finite differences.
///
/// Given a nonlinear continuous-time system:
///   ẋ = f(x, u)
///
/// the linearization around an operating point (x0, u0) produces:
///   δẋ ≈ A·δx + B·δu
///
/// where:
///   A = ∂f/∂x |_(x0,u0)  (N×N Jacobian w.r.t. state)
///   B = ∂f/∂u |_(x0,u0)  (N×I Jacobian w.r.t. input)
///
/// Both Jacobians are computed using second-order central finite differences:
///   ∂f_i/∂x_j ≈ [ f(x0 + ε·ej, u0) - f(x0 - ε·ej, u0) ] / (2ε)
///
/// Output matrix:
///   C = I_N  (full-state output, N×N identity)
///   D = 0    (no direct feedthrough)
///
/// Discrete-time variant (ZOH approximation for small dt):
///   Ad = I + A·dt  (forward Euler on continuous A)
///   Bd = B·dt
///
/// Controllability check (Kalman rank condition):
///   The controllability matrix is:
///     Wc = [B | A·B | A²·B | … | A^{N-1}·B]  (N × N·I)
///   Rank is estimated by counting singular values above a threshold.
///   For no_std compatibility we use a column-orthogonalisation approach
///   (Gram-Schmidt) to estimate the rank numerically.
use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Error type for linearization operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearizationError {
    /// The numerical perturbation `eps` is too small or non-positive.
    InvalidEpsilon,
    /// The timestep `dt` is non-positive for the discrete variant.
    InvalidTimestep,
    /// The controllability check could not be completed (e.g., zero column norms).
    ControllabilityCheckFailed,
}

impl core::fmt::Display for LinearizationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidEpsilon => write!(f, "eps must be strictly positive"),
            Self::InvalidTimestep => write!(f, "dt must be strictly positive"),
            Self::ControllabilityCheckFailed => {
                write!(f, "Controllability check failed (zero column encountered)")
            }
        }
    }
}

/// Result of a linearization: continuous-time Jacobians A and B, with
/// full-state output (C = I, D = 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearizedSystem<S: ControlScalar, const N: usize, const I: usize> {
    /// State Jacobian A = ∂f/∂x.
    pub a: Matrix<S, N, N>,
    /// Input Jacobian B = ∂f/∂u.
    pub b: Matrix<S, N, I>,
    /// Output matrix C (N×N identity for full-state output).
    pub c: Matrix<S, N, N>,
}

impl<S: ControlScalar, const N: usize, const I: usize> LinearizedSystem<S, N, I> {
    /// Construct from Jacobians; C is automatically set to identity.
    pub fn new(a: Matrix<S, N, N>, b: Matrix<S, N, I>) -> Self {
        Self {
            a,
            b,
            c: Matrix::identity(),
        }
    }

    /// Discrete-time approximation using Zero-Order Hold (forward Euler for
    /// small dt):
    ///   Ad = I + A·dt
    ///   Bd = B·dt
    ///
    /// # Errors
    /// Returns `Err(InvalidTimestep)` if `dt <= 0`.
    pub fn discretize(
        &self,
        dt: S,
    ) -> Result<DiscreteLinearizedSystem<S, N, I>, LinearizationError> {
        if dt <= S::ZERO {
            return Err(LinearizationError::InvalidTimestep);
        }
        let eye = Matrix::<S, N, N>::identity();
        let ad = eye.add_mat(&self.a.scale(dt));
        let bd = self.b.scale(dt);
        Ok(DiscreteLinearizedSystem { ad, bd, c: self.c })
    }
}

/// Discrete-time linearized system (ZOH approximation).
#[derive(Debug, Clone, Copy)]
pub struct DiscreteLinearizedSystem<S: ControlScalar, const N: usize, const I: usize> {
    /// Discrete-time state transition matrix Ad.
    pub ad: Matrix<S, N, N>,
    /// Discrete-time input matrix Bd.
    pub bd: Matrix<S, N, I>,
    /// Output matrix C (full-state).
    pub c: Matrix<S, N, N>,
}

/// Linearize a continuous nonlinear system `f: (x, u) → ẋ` around
/// an operating point `(x0, u0)` using central finite differences with
/// perturbation size `eps`.
///
/// # Type parameters
/// - `N` : state dimension
/// - `I` : input dimension
/// - `S` : scalar type (`f32` or `f64`)
///
/// # Arguments
/// - `f`   : the nonlinear dynamics function `f(x, u) -> [S; N]`
/// - `x0`  : nominal state vector
/// - `u0`  : nominal input vector
/// - `eps` : finite-difference perturbation size (should be ~√ε_machine)
///
/// # Errors
/// Returns `Err(InvalidEpsilon)` if `eps <= 0`.
pub fn linearize<S: ControlScalar, const N: usize, const I: usize>(
    f: impl Fn(&[S; N], &[S; I]) -> [S; N],
    x0: &[S; N],
    u0: &[S; I],
    eps: S,
) -> Result<LinearizedSystem<S, N, I>, LinearizationError> {
    if eps <= S::ZERO {
        return Err(LinearizationError::InvalidEpsilon);
    }
    let two_eps = S::TWO * eps;

    // Compute A = ∂f/∂x via central differences
    let mut a_data = [[S::ZERO; N]; N];
    for j in 0..N {
        // Forward perturbation: x0 + eps·ej
        let mut xp = *x0;
        xp[j] += eps;
        let fp = f(&xp, u0);

        // Backward perturbation: x0 - eps·ej
        let mut xm = *x0;
        xm[j] -= eps;
        let fm = f(&xm, u0);

        // Column j of A: (f(x0+eps·ej,u0) - f(x0-eps·ej,u0)) / (2·eps)
        for i in 0..N {
            a_data[i][j] = (fp[i] - fm[i]) / two_eps;
        }
    }

    // Compute B = ∂f/∂u via central differences
    let mut b_data = [[S::ZERO; I]; N];
    for j in 0..I {
        let mut up = *u0;
        up[j] += eps;
        let fp = f(x0, &up);

        let mut um = *u0;
        um[j] -= eps;
        let fm = f(x0, &um);

        for i in 0..N {
            b_data[i][j] = (fp[i] - fm[i]) / two_eps;
        }
    }

    let a = Matrix { data: a_data };
    let b = Matrix { data: b_data };
    Ok(LinearizedSystem::new(a, b))
}

/// Linearize and immediately discretize in one step (ZOH).
///
/// Computes A and B via central finite differences, then applies:
///   Ad = I + A·dt,  Bd = B·dt
///
/// # Errors
/// Returns an error if `eps <= 0` or `dt <= 0`.
pub fn linearize_discrete<S: ControlScalar, const N: usize, const I: usize>(
    f: impl Fn(&[S; N], &[S; I]) -> [S; N],
    x0: &[S; N],
    u0: &[S; I],
    eps: S,
    dt: S,
) -> Result<DiscreteLinearizedSystem<S, N, I>, LinearizationError> {
    let sys = linearize(f, x0, u0, eps)?;
    sys.discretize(dt)
}

/// Estimate the rank of the N×(N·I) controllability matrix
///   Wc = [B | A·B | A²·B | … | A^{N-1}·B]
///
/// using Modified Gram-Schmidt orthogonalisation on the columns of Wc.
/// A column is considered linearly independent if its norm after projection
/// exceeds `tol` (absolute threshold).
///
/// Returns the estimated rank in `[0, N]`.
///
/// # Errors
/// Returns `Err(ControllabilityCheckFailed)` if any column has infinite or NaN
/// norm, indicating a degenerate system.
pub fn controllability_rank<S: ControlScalar, const N: usize, const I: usize>(
    sys: &LinearizedSystem<S, N, I>,
    tol: S,
) -> Result<usize, LinearizationError> {
    // We have at most N*I columns total, but we only need N linearly independent
    // vectors to achieve full controllability.  Build Wc column-by-column.
    let max_cols = N * I; // per Cayley-Hamilton, need only N power-of-A terms × I cols
    let power_count = N; // number of {A^k · B} blocks to compute

    // Store up to N*I orthonormal basis vectors as arrays of length N
    // (heap-free: fixed at compile time via N and I).
    // We use a flat array of up to N vectors (rank cannot exceed N).
    let mut basis = [[S::ZERO; N]; N]; // at most N basis vectors
    let mut rank = 0usize;

    // Precompute A^k · B for k = 0 … N-1
    // Current power of A: starts as I (identity), multiplied by A each iteration
    let mut a_power = Matrix::<S, N, N>::identity();

    'outer: for _k in 0..power_count {
        // Columns of A^k · B form a block in Wc
        let ab = matmul(&a_power, &sys.b);

        for col in 0..I {
            if rank >= N {
                break 'outer; // full rank achieved
            }
            if rank * I + col >= max_cols {
                break 'outer;
            }

            // Extract column `col` of ab as a length-N vector
            let mut v: [S; N] = core::array::from_fn(|row| ab.data[row][col]);

            // Orthogonalise v against existing basis vectors (Modified Gram-Schmidt)
            for basis_vec in basis.iter().take(rank) {
                let dot = dot_n::<S, N>(&v, basis_vec);
                for i in 0..N {
                    v[i] -= dot * basis_vec[i];
                }
            }

            // Compute norm of residual
            let norm = dot_n::<S, N>(&v, &v).sqrt();
            if !norm.is_finite() {
                return Err(LinearizationError::ControllabilityCheckFailed);
            }

            if norm > tol {
                // Normalise and add to basis
                let inv_norm = S::ONE / norm;
                for i in 0..N {
                    basis[rank][i] = v[i] * inv_norm;
                }
                rank += 1;
            }
        }

        // Advance A^k → A^{k+1}
        a_power = matmul(&a_power, &sys.a);
    }

    Ok(rank)
}

/// Check whether the linearized system is controllable (full rank of Wc).
///
/// Returns `true` if the controllability matrix has rank N.
///
/// Uses `tol` as the singular-value threshold for rank estimation.
/// A reasonable default is `tol = 1e-8` for f64.
///
/// # Errors
/// Propagates errors from `controllability_rank`.
pub fn is_controllable<S: ControlScalar, const N: usize, const I: usize>(
    sys: &LinearizedSystem<S, N, I>,
    tol: S,
) -> Result<bool, LinearizationError> {
    let r = controllability_rank(sys, tol)?;
    Ok(r == N)
}

// ─── helper ─────────────────────────────────────────────────────────────────

/// Dot product of two length-N arrays (avoids importing vec_dot from matrix).
#[inline]
fn dot_n<S: ControlScalar, const N: usize>(a: &[S; N], b: &[S; N]) -> S {
    let mut s = S::ZERO;
    for i in 0..N {
        s += a[i] * b[i];
    }
    s
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Linear system: f(x,u) = A·x + B·u where
    ///   A = [[-1, 0], [0, -2]],  B = [[1], [1]].
    /// The analytic Jacobians are A and B themselves.
    /// The numerical linearization should match to high precision.
    #[test]
    fn linearize_matches_analytic_jacobian_linear_system() {
        // f([x0, x1], [u]) = [-x0 + u, -2·x1 + u]
        let f = |x: &[f64; 2], u: &[f64; 1]| -> [f64; 2] { [-x[0] + u[0], -2.0 * x[1] + u[0]] };
        let x0 = [0.5_f64, -0.3];
        let u0 = [1.0_f64];
        let eps = 1e-5_f64;

        let sys = linearize(f, &x0, &u0, eps).expect("linearize should succeed");

        // Analytic: A = [[-1,0],[0,-2]], B = [[1],[1]]
        assert!((sys.a.data[0][0] - (-1.0)).abs() < 1e-6, "A[0][0]");
        assert!((sys.a.data[0][1]).abs() < 1e-6, "A[0][1]");
        assert!((sys.a.data[1][0]).abs() < 1e-6, "A[1][0]");
        assert!((sys.a.data[1][1] - (-2.0)).abs() < 1e-6, "A[1][1]");

        assert!((sys.b.data[0][0] - 1.0).abs() < 1e-6, "B[0][0]");
        assert!((sys.b.data[1][0] - 1.0).abs() < 1e-6, "B[1][0]");
    }

    /// Nonlinear system: simple pendulum without damping.
    ///   x = [θ, θ̇],  u = [torque]
    ///   ẋ = [θ̇,  -g/l·sin(θ) + τ/(m·l²)]
    ///
    /// Analytic Jacobian at θ=0, θ̇=0, τ=0:
    ///   A = [[0, 1], [-g/l, 0]],  B = [[0], [1/(m·l²)]].
    #[test]
    fn linearize_pendulum_at_upright() {
        let g = 9.81_f64;
        let l = 1.0_f64;
        let m = 1.0_f64;

        let f = move |x: &[f64; 2], u: &[f64; 1]| -> [f64; 2] {
            let theta = x[0];
            let theta_dot = x[1];
            let tau = u[0];
            let theta_ddot = -(g / l) * theta.sin() + tau / (m * l * l);
            [theta_dot, theta_ddot]
        };

        let x0 = [0.0_f64, 0.0];
        let u0 = [0.0_f64];
        let eps = 1e-5_f64;

        let sys = linearize(f, &x0, &u0, eps).expect("linearize ok");

        // A[0][0] = 0, A[0][1] = 1 (from θ̇ equation)
        assert!((sys.a.data[0][0]).abs() < 1e-5, "A[0][0] should be 0");
        assert!((sys.a.data[0][1] - 1.0).abs() < 1e-5, "A[0][1] should be 1");
        // A[1][0] = -g/l (derivative of -g/l·sin(θ) at θ=0)
        assert!(
            (sys.a.data[1][0] - (-g / l)).abs() < 1e-4,
            "A[1][0] should be -g/l={:.4}, got {:.4}",
            -g / l,
            sys.a.data[1][0]
        );
        assert!((sys.a.data[1][1]).abs() < 1e-5, "A[1][1] should be 0");

        // B[0][0] = 0, B[1][0] = 1/(m·l²)
        assert!((sys.b.data[0][0]).abs() < 1e-5, "B[0][0] should be 0");
        assert!(
            (sys.b.data[1][0] - 1.0 / (m * l * l)).abs() < 1e-5,
            "B[1][0] should be 1/(m·l²)={:.4}, got {:.4}",
            1.0 / (m * l * l),
            sys.b.data[1][0]
        );
    }

    /// A double integrator ẋ₁=x₂, ẋ₂=u is controllable.
    #[test]
    fn double_integrator_is_controllable() {
        let f = |x: &[f64; 2], u: &[f64; 1]| -> [f64; 2] { [x[1], u[0]] };
        let x0 = [0.0_f64, 0.0];
        let u0 = [0.0_f64];
        let sys = linearize(f, &x0, &u0, 1e-5).expect("ok");
        let ctrl = is_controllable(&sys, 1e-8).expect("no error");
        assert!(ctrl, "double integrator should be controllable");
    }

    /// An uncontrollable system: two decoupled modes, only one driven by input.
    ///   ẋ₁ = -x₁ + u,  ẋ₂ = -x₂  (x₂ is not reachable)
    #[test]
    fn uncontrollable_system_detected() {
        let f = |x: &[f64; 2], u: &[f64; 1]| -> [f64; 2] { [-x[0] + u[0], -x[1]] };
        let x0 = [0.0_f64, 0.0];
        let u0 = [0.0_f64];
        let sys = linearize(f, &x0, &u0, 1e-5).expect("ok");
        let ctrl = is_controllable(&sys, 1e-8).expect("no error");
        assert!(!ctrl, "system with undriven mode should be uncontrollable");
    }

    /// Discrete linearization: step response of linearized double integrator
    /// should match exact Euler integration for small dt.
    #[test]
    fn discrete_linearization_step_response() {
        let f = |x: &[f64; 2], u: &[f64; 1]| -> [f64; 2] { [x[1], u[0]] };
        let x0 = [0.0_f64, 0.0];
        let u0 = [1.0_f64];
        let dt = 0.01_f64;

        let dsys = linearize_discrete(f, &x0, &u0, 1e-5, dt).expect("ok");

        // Step the discrete system with unit input for 10 steps
        let mut x = [0.0_f64, 0.0];
        let u = [1.0_f64];
        for _ in 0..10 {
            let x_new = [
                dsys.ad.data[0][0] * x[0] + dsys.ad.data[0][1] * x[1] + dsys.bd.data[0][0] * u[0],
                dsys.ad.data[1][0] * x[0] + dsys.ad.data[1][1] * x[1] + dsys.bd.data[1][0] * u[0],
            ];
            x = x_new;
        }
        // After 10 steps with dt=0.01 and unit input:
        // x₂(t) = t (linear ramp), x₁(t) = ½t² (quadratic)
        // At t=0.1: x₂≈0.1, x₁≈0.005
        assert!(
            (x[1] - 0.1).abs() < 0.01,
            "velocity should be ≈0.1: got {}",
            x[1]
        );
        assert!(x[0] > 0.0, "position should be positive: got {}", x[0]);
    }

    /// Epsilon validation: non-positive eps should return error.
    #[test]
    fn negative_eps_returns_error() {
        let f = |x: &[f64; 1], _u: &[f64; 1]| -> [f64; 1] { [-x[0]] };
        let result = linearize(f, &[0.0], &[0.0], -1e-5);
        assert!(
            result == Err(LinearizationError::InvalidEpsilon),
            "negative eps should return InvalidEpsilon"
        );
        let result2 = linearize(f, &[0.0], &[0.0], 0.0_f64);
        assert!(
            result2 == Err(LinearizationError::InvalidEpsilon),
            "zero eps should return InvalidEpsilon"
        );
    }

    /// Controllability rank of a single-integrator is 1 (trivially controllable).
    #[test]
    fn single_integrator_rank_one() {
        let f = |_x: &[f64; 1], u: &[f64; 1]| -> [f64; 1] { [u[0]] };
        let sys = linearize(f, &[0.0], &[0.0], 1e-5).expect("ok");
        let rank = controllability_rank(&sys, 1e-8).expect("no error");
        assert_eq!(rank, 1, "single integrator has rank 1");
    }
}
