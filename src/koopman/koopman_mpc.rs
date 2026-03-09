//! Single-step greedy Koopman MPC.
//!
//! Operates in the Koopman-lifted space where the dynamics are approximately linear:
//!
//! ```text
//!   ψ[k+1] = K · ψ[k] + B_u · u[k]
//!   y[k]   = cᵀ · ψ[k]
//! ```
//!
//! The greedy (receding-horizon depth-1) optimal control law minimises
//!
//! ```text
//!   J(u) = q · (y_ref − ŷ[k+1])² + Σᵢ r_w · uᵢ²
//! ```
//!
//! resulting in an analytic per-input solution that is then clamped to box
//! constraints `[u_min, u_max]`.

use crate::core::scalar::ControlScalar;
use crate::koopman::lifting_functions::KoopmanError;

// ── KoopmanGreedyMpc ──────────────────────────────────────────────────────────

/// Single-step greedy Koopman MPC controller.
///
/// # Type parameters
/// - `S` — scalar type implementing [`ControlScalar`].
/// - `L` — dimension of the Koopman-lifted state space.
/// - `I` — number of control inputs.
#[derive(Clone, Debug)]
pub struct KoopmanGreedyMpc<S, const L: usize, const I: usize> {
    /// Koopman matrix K ∈ ℝ^{L×L}.
    k_mat: [[S; L]; L],
    /// Input matrix B_u ∈ ℝ^{L×I} (L rows, I columns).
    b_u: [[S; I]; L],
    /// Output selection vector c ∈ ℝ^L;  y = cᵀ · ψ.
    c_out: [S; L],
    /// Tracking weight q > 0.
    q: S,
    /// Effort weight r_w > 0.
    r_w: S,
    /// Lower bound on each control input.
    u_min: [S; I],
    /// Upper bound on each control input.
    u_max: [S; I],
}

impl<S: ControlScalar, const L: usize, const I: usize> KoopmanGreedyMpc<S, L, I> {
    /// Create a new `KoopmanGreedyMpc`.
    ///
    /// # Errors
    /// - [`KoopmanError::InvalidParameter`] if `q <= 0`, `r_w <= 0`, or any
    ///   `u_min[i] > u_max[i]`.
    pub fn new(
        k_mat: [[S; L]; L],
        b_u: [[S; I]; L],
        c_out: [S; L],
        q: S,
        r_w: S,
        u_min: [S; I],
        u_max: [S; I],
    ) -> Result<Self, KoopmanError> {
        if q <= S::ZERO || r_w <= S::ZERO {
            return Err(KoopmanError::InvalidParameter);
        }
        for i in 0..I {
            if u_min[i] > u_max[i] {
                return Err(KoopmanError::InvalidParameter);
            }
        }
        Ok(Self {
            k_mat,
            b_u,
            c_out,
            q,
            r_w,
            u_min,
            u_max,
        })
    }

    /// Compute the greedy optimal control action given the current lifted state `psi`
    /// and reference output `y_ref`.
    ///
    /// For each input channel `i`:
    /// ```text
    ///   CB_i   = cᵀ · B_u[:,i]
    ///   u_i*   = (q · CB_i · error) / (q · CB_i² + r_w),  clipped to [u_min_i, u_max_i]
    /// ```
    /// where `error = y_ref − cᵀ · K · ψ`.
    #[allow(clippy::needless_range_loop)]
    pub fn control(&self, psi: &[S; L], y_ref: S) -> Result<[S; I], KoopmanError> {
        // Compute K·ψ (predicted next lifted state without input contribution)
        let mut k_psi = [S::ZERO; L];
        for row in 0..L {
            for col in 0..L {
                k_psi[row] += self.k_mat[row][col] * psi[col];
            }
        }

        // Predicted output y_pred = cᵀ · (K·ψ)
        let mut y_pred = S::ZERO;
        for l in 0..L {
            y_pred += self.c_out[l] * k_psi[l];
        }

        let error = y_ref - y_pred;

        // Solve analytically for each input channel
        let mut u_out = [S::ZERO; I];
        for i in 0..I {
            // CB_i = cᵀ · B_u[:,i]  (column i of B_u)
            let mut cb_i = S::ZERO;
            for l in 0..L {
                cb_i += self.c_out[l] * self.b_u[l][i];
            }

            // Analytic unconstrained optimum
            let numerator = self.q * cb_i * error;
            let denominator = self.q * cb_i * cb_i + self.r_w;
            // denominator > 0 because r_w > 0
            let u_star = numerator / denominator;

            // Clamp to box constraints
            u_out[i] = u_star.clamp_val(self.u_min[i], self.u_max[i]);
        }

        Ok(u_out)
    }

    /// Return a reference to the Koopman matrix.
    pub fn k_matrix(&self) -> &[[S; L]; L] {
        &self.k_mat
    }

    /// Return a reference to the input matrix B_u.
    pub fn b_matrix(&self) -> &[[S; I]; L] {
        &self.b_u
    }

    /// Return a reference to the output selection vector c.
    pub fn c_out(&self) -> &[S; L] {
        &self.c_out
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: 1-input MPC with identity K, B_u = [[1]], c = [1], q=1, r=1
    fn simple_mpc() -> KoopmanGreedyMpc<f64, 1, 1> {
        let k_mat = [[1.0_f64]];
        let b_u = [[1.0_f64]];
        let c_out = [1.0_f64];
        KoopmanGreedyMpc::new(k_mat, b_u, c_out, 1.0, 1.0, [-10.0], [10.0]).expect("new")
    }

    /// Zero tracking error → control output should be (near) zero.
    #[test]
    fn zero_error_near_zero_control() {
        let mpc = simple_mpc();
        // K·psi = [[1]]*[3] = [3]; y_pred = 1*3 = 3; y_ref = 3 → error = 0
        let psi = [3.0_f64];
        let u = mpc.control(&psi, 3.0).expect("control");
        assert!(u[0].abs() < 1e-12, "u[0]={}", u[0]);
    }

    /// Positive tracking error with positive CB should give positive control.
    #[test]
    fn control_sign_correct() {
        let mpc = simple_mpc();
        // y_ref=5, y_pred = K*psi = 3 → error = 2 > 0
        let psi = [3.0_f64];
        let u = mpc.control(&psi, 5.0).expect("control");
        assert!(u[0] > 0.0, "expected positive control, got {}", u[0]);
    }

    /// When the unconstrained optimum exceeds u_max, output should be clamped.
    #[test]
    fn saturation_clamps_to_u_max() {
        // Use very large q to drive u_star far beyond u_max=1
        let k_mat = [[1.0_f64]];
        let b_u = [[1.0_f64]];
        let c_out = [1.0_f64];
        let mpc = KoopmanGreedyMpc::new(k_mat, b_u, c_out, 1e9, 1.0, [-1.0], [1.0]).expect("new");
        let psi = [0.0_f64];
        let u = mpc.control(&psi, 100.0).expect("control");
        assert!(
            (u[0] - 1.0).abs() < 1e-9,
            "u should be clamped to 1.0, got {}",
            u[0]
        );
    }

    /// When the unconstrained optimum is below u_min, output should be clamped.
    #[test]
    fn saturation_clamps_to_u_min() {
        let k_mat = [[1.0_f64]];
        let b_u = [[1.0_f64]];
        let c_out = [1.0_f64];
        let mpc = KoopmanGreedyMpc::new(k_mat, b_u, c_out, 1e9, 1.0, [-1.0], [1.0]).expect("new");
        let psi = [0.0_f64];
        let u = mpc.control(&psi, -100.0).expect("control");
        assert!(
            (u[0] - (-1.0)).abs() < 1e-9,
            "u should be clamped to -1.0, got {}",
            u[0]
        );
    }

    /// q <= 0 should return InvalidParameter.
    #[test]
    fn invalid_q_returns_error() {
        let result = KoopmanGreedyMpc::<f64, 1, 1>::new(
            [[1.0]],
            [[1.0]],
            [1.0],
            0.0, // invalid q
            1.0,
            [-1.0],
            [1.0],
        );
        assert!(matches!(result, Err(KoopmanError::InvalidParameter)));
    }

    /// r_w <= 0 should return InvalidParameter.
    #[test]
    fn invalid_r_returns_error() {
        let result = KoopmanGreedyMpc::<f64, 1, 1>::new(
            [[1.0]],
            [[1.0]],
            [1.0],
            1.0,
            0.0, // invalid r_w
            [-1.0],
            [1.0],
        );
        assert!(matches!(result, Err(KoopmanError::InvalidParameter)));
    }

    /// u_min > u_max should return InvalidParameter.
    #[test]
    fn u_min_greater_than_u_max_returns_error() {
        let result = KoopmanGreedyMpc::<f64, 1, 1>::new(
            [[1.0]],
            [[1.0]],
            [1.0],
            1.0,
            1.0,
            [5.0], // u_min > u_max
            [1.0],
        );
        assert!(matches!(result, Err(KoopmanError::InvalidParameter)));
    }

    /// Multi-input: verify each channel is solved independently.
    #[test]
    fn multi_input_independent_channels() {
        // L=2, I=2; K=I, B_u=I (2x2 identity), c=[1,0] → only first lifted dim tracked
        let k_mat = [[1.0_f64, 0.0], [0.0, 1.0]];
        let b_u = [[1.0_f64, 0.0], [0.0, 1.0]];
        let c_out = [1.0_f64, 0.0]; // only track first dimension
        let mpc = KoopmanGreedyMpc::new(k_mat, b_u, c_out, 1.0, 1.0, [-10.0, -10.0], [10.0, 10.0])
            .expect("new");
        // psi = [2, 0]; y_pred = 1*2 + 0*0 = 2; y_ref = 4 → error = 2
        // CB_0 = c^T * b_u[:,0] = 1*1 + 0*0 = 1; u_0* = 1*1*2/(1*1+1) = 1
        // CB_1 = c^T * b_u[:,1] = 1*0 + 0*1 = 0; u_1* = 0
        let psi = [2.0_f64, 0.0];
        let u = mpc.control(&psi, 4.0).expect("control");
        assert!((u[0] - 1.0).abs() < 1e-12, "u[0]={}", u[0]);
        assert!(u[1].abs() < 1e-12, "u[1]={}", u[1]);
    }
}
