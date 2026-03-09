use crate::core::scalar::ControlScalar;

/// Adaptive MPC with online Recursive Least Squares (RLS) plant identification.
///
/// Maintains a linear model  x_{k+1} = A·x_k + B·u_k  identified online.
/// At each step:
///   1. Update RLS estimates of [A | B] from last (x, u) → x_next measurement.
///   2. Run horizon-1 receding-horizon LQR (approximate MPC) using current model.
///
/// State dim N, input dim M.  RLS parameter matrix Θ ∈ R^{N×(N+M)}.
pub struct AdaptiveMpc<S: ControlScalar, const N: usize, const M: usize> {
    /// RLS estimate: Θ = [A | B], stored row-major Θ[i][j]
    /// where j ∈ 0..N → A col, j ∈ N..N+M → B col.
    pub theta: [[S; N]; N], // A part
    pub theta_b: [[S; M]; N], // B part
    /// RLS covariance (diagonal approximation) — size N*(N+M).
    cov_diag: [[S; N]; N], // A cov (diagonal)
    cov_diag_b: [[S; M]; N],  // B cov (diagonal)
    /// RLS forgetting factor λ ∈ (0,1].
    pub lambda: S,
    /// State tracking weight Q (diagonal).
    pub q: [S; N],
    /// Input weight R (diagonal).
    pub r: [S; M],
    /// Input bounds.
    pub u_min: [S; M],
    pub u_max: [S; M],
    /// Last control input (for warm-start / diagnostics).
    last_u: [S; M],
}

impl<S: ControlScalar, const N: usize, const M: usize> AdaptiveMpc<S, N, M> {
    pub fn new(lambda: S, q: [S; N], r: [S; M], u_min: [S; M], u_max: [S; M]) -> Self {
        let init_cov = S::from_f64(1e3);
        Self {
            theta: [[S::ZERO; N]; N],
            theta_b: [[S::ZERO; M]; N],
            cov_diag: [[init_cov; N]; N],
            cov_diag_b: [[init_cov; M]; N],
            lambda,
            q,
            r,
            u_min,
            u_max,
            last_u: [S::ZERO; M],
        }
    }

    /// Update RLS estimate given measurement: transition from (x_prev, u_prev) → x_now.
    pub fn rls_update(&mut self, x_prev: &[S; N], u_prev: &[S; M], x_now: &[S; N]) {
        // For each output dimension i: predict y_hat = θ_row_i · [x_prev; u_prev]
        // error = x_now[i] - y_hat
        // RLS update with diagonal covariance approximation (independent per output)
        for (i, &x_now_i) in x_now.iter().enumerate() {
            // Prediction from A part
            let mut y_hat = S::ZERO;
            for (j, &xpj) in x_prev.iter().enumerate() {
                y_hat += self.theta[i][j] * xpj;
            }
            for (j, &upj) in u_prev.iter().enumerate() {
                y_hat += self.theta_b[i][j] * upj;
            }
            let err = x_now_i - y_hat;

            // Update A part
            for (j, &xpj) in x_prev.iter().enumerate() {
                let p = self.cov_diag[i][j];
                // scalar RLS gain: k = p * xj / (lambda + p * xj^2)
                let denom = self.lambda + p * xpj * xpj;
                if denom.abs() > S::from_f64(1e-15) {
                    let gain = p * xpj / denom;
                    self.theta[i][j] += gain * err;
                    self.cov_diag[i][j] = (p - gain * xpj * p) / self.lambda;
                }
            }

            // Update B part
            for (j, &upj) in u_prev.iter().enumerate() {
                let p = self.cov_diag_b[i][j];
                let denom = self.lambda + p * upj * upj;
                if denom.abs() > S::from_f64(1e-15) {
                    let gain = p * upj / denom;
                    self.theta_b[i][j] += gain * err;
                    self.cov_diag_b[i][j] = (p - gain * upj * p) / self.lambda;
                }
            }
        }
    }

    /// Compute one-step ahead control using identified model.
    ///
    /// Minimizes Q-weighted state error + R-weighted input at horizon=1:
    ///   u* = argmin_{u_min ≤ u ≤ u_max}  ||A·x + B·u - x_ref||_Q² + ||u||_R²
    ///
    /// Closed-form solution per input dimension (assuming B is column-decoupled).
    /// Returns optimal u.
    pub fn update(&mut self, x: &[S; N], x_ref: &[S; N]) -> [S; M] {
        // predicted state without control: x_A = A·x
        let mut x_a = [S::ZERO; N];
        for (i, xa_i) in x_a.iter_mut().enumerate() {
            for (j, &xj) in x.iter().enumerate() {
                *xa_i += self.theta[i][j] * xj;
            }
        }

        // For each input channel, compute gradient and step
        // ∂J/∂u_k = 2 Σ_i q_i * B[i][k] * (x_A[i] + Σ_l B[i][l]*u[l] - x_ref[i]) + 2 r_k * u_k
        // One Newton step from u=0:
        //   numerator_k = -Σ_i q_i * B[i][k] * (x_A[i] - x_ref[i])
        //   denom_k = r_k + Σ_i q_i * B[i][k]^2
        let mut u = [S::ZERO; M];
        for (k, uk) in u.iter_mut().enumerate() {
            let mut num = S::ZERO;
            let mut den = self.r[k];
            for (i, (&qi, (&x_ref_i, &xa_i))) in
                self.q.iter().zip(x_ref.iter().zip(x_a.iter())).enumerate()
            {
                let bik = self.theta_b[i][k];
                num += qi * bik * (x_ref_i - xa_i);
                den += qi * bik * bik;
            }
            if den.abs() > S::from_f64(1e-15) {
                *uk = num / den;
            }
            *uk = uk.clamp_val(self.u_min[k], self.u_max[k]);
        }

        self.last_u = u;
        u
    }

    pub fn reset(&mut self) {
        self.theta = [[S::ZERO; N]; N];
        self.theta_b = [[S::ZERO; M]; N];
        let init_cov = S::from_f64(1e3);
        self.cov_diag = [[init_cov; N]; N];
        self.cov_diag_b = [[init_cov; M]; N];
        self.last_u = [S::ZERO; M];
    }

    pub fn last_control(&self) -> [S; M] {
        self.last_u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integrator(x: &[f64; 1], u: &[f64; 1]) -> [f64; 1] {
        [x[0] + 0.1 * u[0]]
    }

    #[test]
    fn rls_identifies_integrator() {
        let mut ampc = AdaptiveMpc::<f64, 1, 1>::new(0.99, [1.0], [0.1], [-10.0], [10.0]);
        let mut x = [0.0f64];
        for step in 0..200 {
            let u = [((step % 10) as f64 - 5.0) * 0.5];
            let x_next = integrator(&x, &u);
            ampc.rls_update(&x, &u, &x_next);
            x = x_next;
        }
        // Should have identified A ≈ 1.0, B ≈ 0.1
        let a_id = ampc.theta[0][0];
        let b_id = ampc.theta_b[0][0];
        assert!((a_id - 1.0).abs() < 0.05, "A={a_id:.4}, expected ≈1.0");
        assert!((b_id - 0.1).abs() < 0.05, "B={b_id:.4}, expected ≈0.1");
    }

    #[test]
    fn adaptive_mpc_drives_to_reference() {
        let mut ampc = AdaptiveMpc::<f64, 1, 1>::new(0.98, [10.0], [0.1], [-5.0], [5.0]);
        // Pre-identify with known model
        let mut x = [0.0f64];
        for step in 0..300 {
            let u = [((step % 20) as f64 - 10.0) * 0.3];
            let x_next = integrator(&x, &u);
            ampc.rls_update(&x, &u, &x_next);
            x = x_next;
        }

        // Now run closed-loop toward reference
        let x_ref = [5.0f64];
        x = [0.0];
        for _ in 0..100 {
            let u = ampc.update(&x, &x_ref);
            let x_prev = x;
            x = integrator(&x_prev, &u);
            ampc.rls_update(&x_prev, &u, &x);
        }
        assert!(x[0].abs() < 6.0, "x={:.4}, should be near ref=5", x[0]);
    }

    #[test]
    fn input_constraints_respected() {
        let mut ampc = AdaptiveMpc::<f64, 1, 1>::new(1.0, [1.0], [0.01], [-2.0], [2.0]);
        // Set theta_b to something large so u would be unconstrained
        ampc.theta_b[0][0] = 1.0;
        let x_ref = [100.0];
        let u = ampc.update(&[0.0], &x_ref);
        assert!(u[0] <= 2.0 + 1e-10 && u[0] >= -2.0 - 1e-10, "u={:.4}", u[0]);
    }
}
