//! Simplified subspace identification (N4SID-inspired).
#![allow(
    clippy::needless_range_loop,
    clippy::doc_overindented_list_items,
    clippy::manual_memcpy
)]
//!
//! Identifies a discrete-time SISO state-space model:
//! ```text
//!   x(t+1) = A·x(t) + B·u(t)
//!   y(t)   = C·x(t) + D·u(t)
//! ```
//! from input/output data without prior knowledge of noise statistics.
//!
//! # Method
//! A simplified variant of N4SID is implemented using a projection of future
//! outputs onto the past data space. Instead of full SVD, we solve a Least Squares
//! problem for the oblique projection weights (one per future-step), then extract
//! the state sequence from these projections. The system matrices A, B, C, D are
//! recovered by a second LS step on the estimated state sequence.
//!
//! # Const parameters for [`identify`]
//! - `N`   — system order (states).
//! - `I`   — horizon; `config.past_horizon` must be ≤ I (used for [`SubspaceModel`] type).
//! - `PI`  — past-regressor dimension bound = 2 * (past_horizon) ≤ 2 * I.
//!           **Caller must ensure PI == 2 * config.past_horizon**.
//! - `NP1` — N + 1 (**caller must ensure NP1 == N + 1**).

use crate::core::scalar::ControlScalar;
use crate::sysid::SysIdError;

// ── SubspaceIdConfig ──────────────────────────────────────────────────────────

/// Configuration for subspace identification.
#[derive(Debug, Clone, Copy)]
pub struct SubspaceIdConfig<S: ControlScalar> {
    /// System order (number of state variables to identify). Must be ≤ N const param.
    pub order: usize,
    /// Past horizon — number of block rows from the past window.
    pub past_horizon: usize,
    /// Future horizon — kept for compatibility; currently only past_horizon is used.
    pub future_horizon: usize,
    /// Regularisation added to diagonal of the Gram matrix.
    pub regularisation: S,
}

impl<S: ControlScalar> SubspaceIdConfig<S> {
    /// Create a configuration suitable for SISO systems of the given order.
    pub fn siso(order: usize) -> Self {
        Self {
            order,
            past_horizon: order * 2 + 2,
            future_horizon: order * 2 + 2,
            regularisation: S::from_f64(1e-8),
        }
    }
}

// ── SubspaceModel ─────────────────────────────────────────────────────────────

/// Identified SISO state-space model.
///
/// # Const parameters
/// - `N` — system order.
/// - `I` — horizon used during identification.
#[derive(Debug, Clone, Copy)]
pub struct SubspaceModel<S: ControlScalar, const N: usize, const I: usize> {
    /// State transition matrix A (N×N).
    pub a: [[S; N]; N],
    /// Input matrix B (N×1 for SISO).
    pub b: [S; N],
    /// Output matrix C (1×N for SISO).
    pub c: [S; N],
    /// Direct feedthrough scalar D.
    pub d: S,
    /// Number of data samples used.
    pub n_samples: usize,
}

impl<S: ControlScalar, const N: usize, const I: usize> SubspaceModel<S, N, I> {
    /// Create a zero-initialised model.
    pub fn zeros() -> Self {
        Self {
            a: [[S::ZERO; N]; N],
            b: [S::ZERO; N],
            c: [S::ZERO; N],
            d: S::ZERO,
            n_samples: 0,
        }
    }

    /// Simulate the model for `M` steps given an input sequence.
    ///
    /// `u_seq` — input sequence (length M).
    /// `x0`   — initial state.
    pub fn simulate<const M: usize>(
        &self,
        u_seq: &[S; M],
        x0: &[S; N],
    ) -> Result<[S; M], SysIdError> {
        let mut x = *x0;
        let mut y_out = [S::ZERO; M];

        for k in 0..M {
            let mut yk = self.d * u_seq[k];
            for i in 0..N {
                yk += self.c[i] * x[i];
            }
            y_out[k] = yk;

            let mut x_next = [S::ZERO; N];
            for i in 0..N {
                x_next[i] = self.b[i] * u_seq[k];
                for j in 0..N {
                    x_next[i] += self.a[i][j] * x[j];
                }
            }
            x = x_next;
        }

        Ok(y_out)
    }
}

// ── Internal: Cholesky solve ─────────────────────────────────────────────────

fn cholesky_solve<S: ControlScalar, const M: usize>(
    a: &mut [[S; M]; M],
    b: &mut [S; M],
) -> Result<(), SysIdError> {
    for i in 0..M {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            if i == j {
                if s <= S::ZERO {
                    return Err(SysIdError::SingularMatrix);
                }
                a[i][j] = s.sqrt();
            } else {
                let d = a[j][j];
                if d == S::ZERO {
                    return Err(SysIdError::SingularMatrix);
                }
                a[i][j] = s / d;
            }
        }
    }
    for i in 0..M {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i][k] * b[k];
        }
        if a[i][i] == S::ZERO {
            return Err(SysIdError::SingularMatrix);
        }
        b[i] = s / a[i][i];
    }
    let mut i = M;
    while i > 0 {
        i -= 1;
        let mut s = b[i];
        for k in (i + 1)..M {
            s -= a[k][i] * b[k];
        }
        if a[i][i] == S::ZERO {
            return Err(SysIdError::SingularMatrix);
        }
        b[i] = s / a[i][i];
    }
    Ok(())
}

// ── identify ──────────────────────────────────────────────────────────────────

/// Identify a SISO state-space model of order `N` from data.
///
/// # Algorithm (simplified N4SID)
///
/// 1. For each sample column `t`, build a past regressor
///    `p(t) = [y(t-1)…y(t-ip), u(t-1)…u(t-ip)]` (dimension 2·ip) and a
///    future output window `f(t) = [y(t)…y(t+N-1)]` (length N).
/// 2. Regress each future output step `y(t+k)` on the past regressor to obtain
///    projection weights Ψ_k ∈ R^{2·ip}. The estimated states are
///    X̂_k(t) = Ψ_k · p(t).
/// 3. Identify A, B, C, D from {X̂(t), y(t), u(t)} via ordinary least squares.
///
/// # Const parameters
/// - `N`   — system order.
/// - `I`   — horizon type bound; `config.past_horizon` must be ≤ I.
/// - `PI`  — past-regressor array dimension = 2 * config.past_horizon
///           (**caller must ensure PI == 2 * past_horizon**).
/// - `NP1` — N + 1  (**caller must ensure NP1 == N + 1**).
pub fn identify<
    S: ControlScalar,
    const N: usize,
    const I: usize,
    const PI: usize,
    const NP1: usize,
>(
    y: &[S],
    u: &[S],
    config: &SubspaceIdConfig<S>,
) -> Result<SubspaceModel<S, N, I>, SysIdError> {
    let n_data = y.len();
    if u.len() != n_data {
        return Err(SysIdError::InvalidData);
    }
    for &v in y.iter().chain(u.iter()) {
        if !v.is_finite() {
            return Err(SysIdError::InvalidData);
        }
    }

    let i_p = config.past_horizon;
    let order = config.order;

    if order == 0 || order > N {
        return Err(SysIdError::InvalidData);
    }
    if i_p == 0 || i_p > I {
        return Err(SysIdError::InsufficientData);
    }

    let p_dim = 2 * i_p;
    // Runtime guard: the caller must have passed PI == 2 * i_p
    if p_dim > PI {
        return Err(SysIdError::InvalidData);
    }

    let hankel_width = n_data.saturating_sub(i_p + N);
    if hankel_width < N + 2 {
        return Err(SysIdError::InsufficientData);
    }

    // Stack-allocated buffers; J_CAP limits Hankel columns to avoid stack blowout.
    const J_CAP: usize = 512;
    let j_actual = hankel_width.min(J_CAP);

    // past_mat: j_actual × PI  (only first p_dim columns active)
    // fut_mat:  j_actual × N
    let mut past_mat = [[S::ZERO; PI]; J_CAP];
    let mut fut_mat = [[S::ZERO; N]; J_CAP];

    for col in 0..j_actual {
        let t = i_p + col;
        for k in 0..i_p {
            if t > k {
                past_mat[col][k] = y[t - 1 - k];
                past_mat[col][i_p + k] = u[t - 1 - k];
            }
        }
        for k in 0..N {
            if t + k < n_data {
                fut_mat[col][k] = y[t + k];
            }
        }
    }

    // Normal equations ATA (PI × PI), only p_dim × p_dim subblock active
    let mut ata = [[S::ZERO; PI]; PI];
    for t in 0..j_actual {
        for i in 0..p_dim {
            for jj in 0..p_dim {
                ata[i][jj] += past_mat[t][i] * past_mat[t][jj];
            }
        }
    }
    for i in 0..p_dim {
        ata[i][i] += config.regularisation;
    }

    // Cholesky factorisation of ata (restricted to p_dim)
    let mut l_chol = [[S::ZERO; PI]; PI];
    for i in 0..p_dim {
        for j_col in 0..=i {
            let mut s = ata[i][j_col];
            for k in 0..j_col {
                s -= l_chol[i][k] * l_chol[j_col][k];
            }
            if i == j_col {
                if s <= S::ZERO {
                    return Err(SysIdError::SingularMatrix);
                }
                l_chol[i][j_col] = s.sqrt();
            } else {
                let d = l_chol[j_col][j_col];
                if d == S::ZERO {
                    return Err(SysIdError::SingularMatrix);
                }
                l_chol[i][j_col] = s / d;
            }
        }
    }

    // For each future step k=0..N-1, solve ATA·Ψ_k = ATy_k
    // psi[k] ∈ R^{PI}, only first p_dim entries are meaningful
    let mut psi = [[S::ZERO; PI]; N];

    for k in 0..N {
        let mut aty_k = [S::ZERO; PI];
        for t in 0..j_actual {
            for i in 0..p_dim {
                aty_k[i] += past_mat[t][i] * fut_mat[t][k];
            }
        }
        // Forward substitution L·z = aty_k
        let mut z = [S::ZERO; PI];
        for i in 0..p_dim {
            let mut s = aty_k[i];
            for m in 0..i {
                s -= l_chol[i][m] * z[m];
            }
            if l_chol[i][i] == S::ZERO {
                return Err(SysIdError::SingularMatrix);
            }
            z[i] = s / l_chol[i][i];
        }
        // Back substitution Lᵀ·Ψ_k = z
        let mut idx = p_dim;
        while idx > 0 {
            idx -= 1;
            let mut s = z[idx];
            for m in (idx + 1)..p_dim {
                s -= l_chol[m][idx] * psi[k][m];
            }
            if l_chol[idx][idx] == S::ZERO {
                return Err(SysIdError::SingularMatrix);
            }
            psi[k][idx] = s / l_chol[idx][idx];
        }
    }

    // Compute estimated state sequence X̂ (j_actual × N)
    let mut x_hat = [[S::ZERO; N]; J_CAP];
    for col in 0..j_actual {
        for k in 0..N {
            let mut v = S::ZERO;
            for m in 0..p_dim {
                v += psi[k][m] * past_mat[col][m];
            }
            x_hat[col][k] = v;
        }
    }

    // Identify C, D: y(t) = C·x̂(t) + D·u(t)   →   regressor dim NP1 = N + 1
    let mut ata_cd = [[S::ZERO; NP1]; NP1];
    let mut aty_cd = [S::ZERO; NP1];
    for col in 0..j_actual {
        let t = i_p + col;
        let mut phi = [S::ZERO; NP1];
        for k in 0..N {
            phi[k] = x_hat[col][k];
        }
        phi[N] = if t < n_data { u[t] } else { S::ZERO };
        let yt = if t < n_data { y[t] } else { S::ZERO };
        for i in 0..NP1 {
            for j in 0..NP1 {
                ata_cd[i][j] += phi[i] * phi[j];
            }
            aty_cd[i] += phi[i] * yt;
        }
    }
    cholesky_solve::<S, NP1>(&mut ata_cd, &mut aty_cd)?;

    let mut model = SubspaceModel::zeros();
    model.n_samples = n_data;
    for k in 0..N {
        model.c[k] = aty_cd[k];
    }
    model.d = aty_cd[N];

    // Identify A, B: x̂(t+1) = A·x̂(t) + B·u(t)   →   regressor dim NP1
    let n_obs = j_actual.saturating_sub(1);
    if n_obs < NP1 + 1 {
        return Err(SysIdError::InsufficientData);
    }

    let mut phi_ab_buf = [[S::ZERO; NP1]; J_CAP];
    for col in 0..n_obs {
        let t = i_p + col;
        for k in 0..N {
            phi_ab_buf[col][k] = x_hat[col][k];
        }
        phi_ab_buf[col][N] = if t < n_data { u[t] } else { S::ZERO };
    }

    // Pre-compute ATA_ab (shared across state rows)
    let mut ata_ab = [[S::ZERO; NP1]; NP1];
    for col in 0..n_obs {
        for i in 0..NP1 {
            for j in 0..NP1 {
                ata_ab[i][j] += phi_ab_buf[col][i] * phi_ab_buf[col][j];
            }
        }
    }

    for state_k in 0..N {
        let mut aty_ab = [S::ZERO; NP1];
        for col in 0..n_obs {
            for i in 0..NP1 {
                aty_ab[i] += phi_ab_buf[col][i] * x_hat[col + 1][state_k];
            }
        }
        let mut ata_copy = ata_ab;
        cholesky_solve::<S, NP1>(&mut ata_copy, &mut aty_ab)?;

        for j in 0..N {
            model.a[state_k][j] = aty_ab[j];
        }
        model.b[state_k] = aty_ab[N];
    }

    Ok(model)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Identify a first-order integrator: y(t) = y(t-1) + u(t-1).
    /// True model: A = [[1]], B = [1], C = [1], D = 0.
    #[test]
    fn identify_pure_integrator() {
        let n = 300_usize;
        let mut y: heapless::Vec<f64, 512> = heapless::Vec::new();
        let mut u_data: heapless::Vec<f64, 512> = heapless::Vec::new();
        let _ = y.push(0.0);
        let _ = u_data.push(0.0);
        for t in 1..n {
            let ut = libm::sin(0.2 * t as f64) * 0.1 + libm::cos(0.07 * t as f64) * 0.05;
            let yt = y[t - 1] + u_data[t - 1];
            let _ = y.push(yt);
            let _ = u_data.push(ut);
        }

        let config = SubspaceIdConfig::<f64> {
            order: 1,
            past_horizon: 4,
            future_horizon: 4,
            regularisation: 1e-6,
        };

        // N=1, I=8, PI=2*past_horizon=8, NP1=N+1=2
        let model = identify::<f64, 1, 8, 8, 2>(y.as_slice(), u_data.as_slice(), &config)
            .expect("identify should succeed for integrator data");

        let a11 = model.a[0][0];
        let b0 = model.b[0];
        let c0 = model.c[0];
        let d0 = model.d;

        assert!(
            (a11 - 1.0).abs() < 0.3,
            "A[0][0] = {a11:.4}, expected near 1.0"
        );
        assert!(b0.abs() > 0.1, "B[0] = {b0:.4}, expected non-trivial");
        assert!(c0.abs() > 0.1, "C[0] = {c0:.4}, expected non-trivial");
        assert!(
            d0.abs() < 0.5,
            "D = {d0:.4}, expected near 0 for integrator"
        );
    }

    /// Identify a first-order stable system: y(t) = 0.7·y(t-1) + 0.3·u(t-1).
    #[test]
    fn identify_first_order_stable() {
        let n = 350_usize;
        let mut y: heapless::Vec<f64, 512> = heapless::Vec::new();
        let mut u_data: heapless::Vec<f64, 512> = heapless::Vec::new();
        let _ = y.push(0.0);
        let _ = u_data.push(0.0);
        for t in 1..n {
            let ut = libm::sin(0.15 * t as f64) + libm::cos(0.09 * t as f64) * 0.7;
            let yt = 0.7 * y[t - 1] + 0.3 * u_data[t - 1];
            let _ = y.push(yt);
            let _ = u_data.push(ut);
        }

        let config = SubspaceIdConfig::<f64> {
            order: 1,
            past_horizon: 5,
            future_horizon: 5,
            regularisation: 1e-7,
        };

        // N=1, I=10, PI=2*5=10, NP1=2
        let model = identify::<f64, 1, 10, 10, 2>(y.as_slice(), u_data.as_slice(), &config)
            .expect("identify should succeed");

        let u_sim: [f64; 50] = core::array::from_fn(|i| libm::sin(0.15 * i as f64));
        let x0 = [0.0_f64];
        let sim_out = model.simulate(&u_sim, &x0).expect("simulate");
        for &v in sim_out.iter() {
            assert!(v.is_finite(), "simulate produced non-finite output");
        }
    }

    /// Zero-initialised model simulation returns all zeros.
    #[test]
    fn zero_model_simulate_gives_zeros() {
        let model = SubspaceModel::<f64, 2, 4>::zeros();
        let u_seq = [1.0_f64; 10];
        let x0 = [0.0_f64; 2];
        let out = model.simulate(&u_seq, &x0).expect("simulate");
        for &v in out.iter() {
            assert_eq!(v, 0.0);
        }
    }

    /// Insufficient data returns error.
    #[test]
    fn identify_insufficient_data_returns_error() {
        let y = [1.0_f64, 2.0, 3.0];
        let u = [0.5_f64, 0.3, 0.2];
        let config = SubspaceIdConfig::<f64> {
            order: 2,
            past_horizon: 4,
            future_horizon: 4,
            regularisation: 1e-6,
        };
        // N=2, I=8, PI=2*4=8, NP1=3
        let result = identify::<f64, 2, 8, 8, 3>(&y, &u, &config);
        assert!(result.is_err(), "Should fail with insufficient data");
    }
}
