use crate::core::matrix::{matmul, matvec, Matrix};
use crate::core::scalar::ControlScalar;

/// Variational Bayes adaptive noise covariance Kalman filter.
///
/// This filter jointly estimates the hidden state and the noise covariances
/// `Q` (process) and `R` (measurement) in a **variational Bayes** (VB)
/// framework.  Both `Q` and `R` are given **Inverse-Wishart (IW)** conjugate
/// priors; the VB E-step runs a standard KF with the current noise estimates,
/// and the M-step updates the IW hyperparameters from the posterior statistics.
///
/// A **forgetting factor** `ρ ∈ (0, 1]` exponentially discounts older
/// observations, enabling tracking of slowly time-varying noise covariances.
///
/// # Algorithm (one step)
///
/// Given IW hyperparameters `(ν_Q, Ψ_Q)` for Q and `(ν_R, Ψ_R)` for R:
///
/// **E-step (KF)**
/// ```text
///   Q̂ = Ψ_Q / (ν_Q − N − 1)          (IW mean)
///   R̂ = Ψ_R / (ν_R − M − 1)          (IW mean)
///   x_pred = F · x
///   P_pred = F · P · Fᵀ + Q̂
///   ν      = z − H · x_pred
///   S      = H · P_pred · Hᵀ + R̂
///   K      = P_pred · Hᵀ · S⁻¹
///   x      = x_pred + K · ν
///   P      = (I − K · H) · P_pred
/// ```
///
/// **M-step (IW hyperparameter update with forgetting factor ρ)**
/// ```text
///   ε  = z − H · x           (posterior innovation)
///   Ψ_R ← ρ · Ψ_R + ε · εᵀ + H · P · Hᵀ
///   ν_R ← ρ · (ν_R − M − 1) + 1 + M + 1
///
///   d   = x − x_pred         (state correction)
///   Ψ_Q ← ρ · Ψ_Q + d · dᵀ + P + P_pred
///   ν_Q ← ρ · (ν_Q − N − 1) + 1 + N + 1
/// ```
///
/// # Type Parameters
/// * `S` – scalar type (`f32` or `f64`)
/// * `N` – state dimension
/// * `M` – measurement dimension
#[derive(Debug, Clone, Copy)]
pub struct VariationalBayesFilter<S: ControlScalar, const N: usize, const M: usize> {
    /// Current state estimate (N).
    x: [S; N],
    /// Error covariance (N×N).
    p: Matrix<S, N, N>,
    /// IW degrees-of-freedom for Q (must be > N + 1 for well-defined mean).
    nu_q: S,
    /// IW scale matrix for Q (N×N).
    psi_q: Matrix<S, N, N>,
    /// IW degrees-of-freedom for R (must be > M + 1 for well-defined mean).
    nu_r: S,
    /// IW scale matrix for R (M×M).
    psi_r: Matrix<S, M, M>,
    /// Forgetting factor ρ ∈ (0, 1].  Values close to 1 give slow adaptation;
    /// smaller values track faster but are noisier.
    pub rho: S,
}

/// Errors produced by [`VariationalBayesFilter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VbFilterError {
    /// IW degrees-of-freedom are too small to give a valid covariance mean.
    DegreesOfFreedomTooSmall,
    /// Innovation covariance S is singular (cannot compute Kalman gain).
    SingularInnovationCovariance,
    /// Forgetting factor is outside the valid range (0, 1].
    InvalidForgettingFactor,
}

impl core::fmt::Display for VbFilterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VbFilterError::DegreesOfFreedomTooSmall => {
                write!(
                    f,
                    "VariationalBayesFilter: degrees of freedom too small for IW mean"
                )
            }
            VbFilterError::SingularInnovationCovariance => {
                write!(f, "VariationalBayesFilter: singular innovation covariance")
            }
            VbFilterError::InvalidForgettingFactor => {
                write!(
                    f,
                    "VariationalBayesFilter: forgetting factor must be in (0, 1]"
                )
            }
        }
    }
}

impl<S: ControlScalar, const N: usize, const M: usize> VariationalBayesFilter<S, N, M> {
    /// Construct a new Variational Bayes adaptive filter.
    ///
    /// # Arguments
    /// * `x0`     – initial state estimate
    /// * `p0`     – initial error covariance
    /// * `nu_q0`  – initial IW degrees-of-freedom for Q (must be > N + 1)
    /// * `psi_q0` – initial IW scale matrix for Q
    /// * `nu_r0`  – initial IW degrees-of-freedom for R (must be > M + 1)
    /// * `psi_r0` – initial IW scale matrix for R
    /// * `rho`    – forgetting factor in (0, 1]
    pub fn new(
        x0: [S; N],
        p0: Matrix<S, N, N>,
        nu_q0: S,
        psi_q0: Matrix<S, N, N>,
        nu_r0: S,
        psi_r0: Matrix<S, M, M>,
        rho: S,
    ) -> Result<Self, VbFilterError> {
        if rho <= S::ZERO || rho > S::ONE {
            return Err(VbFilterError::InvalidForgettingFactor);
        }
        let n_f = S::from_f64(N as f64);
        let m_f = S::from_f64(M as f64);
        if nu_q0 <= n_f + S::ONE {
            return Err(VbFilterError::DegreesOfFreedomTooSmall);
        }
        if nu_r0 <= m_f + S::ONE {
            return Err(VbFilterError::DegreesOfFreedomTooSmall);
        }
        Ok(Self {
            x: x0,
            p: p0,
            nu_q: nu_q0,
            psi_q: psi_q0,
            nu_r: nu_r0,
            psi_r: psi_r0,
            rho,
        })
    }

    /// Compute the IW mean for Q: `Q̂ = Ψ_Q / (ν_Q − N − 1)`.
    ///
    /// Returns `Err` if `ν_Q ≤ N + 1`.
    pub fn estimated_q(&self) -> Result<Matrix<S, N, N>, VbFilterError> {
        let n_f = S::from_f64(N as f64);
        let denom = self.nu_q - n_f - S::ONE;
        if denom <= S::ZERO {
            return Err(VbFilterError::DegreesOfFreedomTooSmall);
        }
        Ok(self.psi_q.scale(S::ONE / denom))
    }

    /// Compute the IW mean for R: `R̂ = Ψ_R / (ν_R − M − 1)`.
    ///
    /// Returns `Err` if `ν_R ≤ M + 1`.
    pub fn estimated_r(&self) -> Result<Matrix<S, M, M>, VbFilterError> {
        let m_f = S::from_f64(M as f64);
        let denom = self.nu_r - m_f - S::ONE;
        if denom <= S::ZERO {
            return Err(VbFilterError::DegreesOfFreedomTooSmall);
        }
        Ok(self.psi_r.scale(S::ONE / denom))
    }

    /// **Predict step** (E-step, prediction half).
    ///
    /// Propagates state and covariance using the current Q estimate:
    /// ```text
    ///   x_pred = F · x
    ///   P_pred = F · P · Fᵀ + Q̂
    /// ```
    pub fn predict(&mut self, f_mat: &Matrix<S, N, N>) -> Result<(), VbFilterError> {
        let q_hat = self.estimated_q()?;

        // x_pred = F·x
        self.x = matvec(f_mat, &self.x);

        // P_pred = F·P·Fᵀ + Q̂
        let fp = matmul(f_mat, &self.p);
        let ft = f_mat.transpose();
        let fpft = matmul(&fp, &ft);
        self.p = fpft.add_mat(&q_hat);

        Ok(())
    }

    /// **Update step** (E-step update + M-step IW hyperparameter update).
    ///
    /// Runs a standard KF update with the current `R̂`, then updates the IW
    /// hyperparameters `(ν_Q, Ψ_Q)` and `(ν_R, Ψ_R)` using the forgetting
    /// factor `ρ`.
    pub fn update(&mut self, h_mat: &Matrix<S, M, N>, z: &[S; M]) -> Result<(), VbFilterError> {
        let r_hat = self.estimated_r()?;
        let n_f = S::from_f64(N as f64);
        let m_f = S::from_f64(M as f64);

        // --- E-step: KF update ---
        // Save predicted state and covariance for M-step
        let x_pred = self.x;
        let p_pred = self.p;

        // Prior innovation: ν = z − H·x_pred
        let hx_pred: [S; M] = matvec(h_mat, &x_pred);
        let nu_innov: [S; M] = core::array::from_fn(|i| z[i] - hx_pred[i]);

        // Innovation covariance: S = H·P·Hᵀ + R̂
        let ht = h_mat.transpose();
        let ph_t = matmul(&self.p, &ht); // N×M
        let h_p_ht = matmul(h_mat, &ph_t); // M×M
        let s_innov = h_p_ht.add_mat(&r_hat);

        // Kalman gain: K = P·Hᵀ · S⁻¹  (N×M)
        let s_inv = s_innov
            .inv()
            .ok_or(VbFilterError::SingularInnovationCovariance)?;
        let k = matmul(&ph_t, &s_inv);

        // State update: x ← x + K·ν
        let k_nu = matvec(&k, &nu_innov);
        for (x_i, &knu_i) in self.x.iter_mut().zip(k_nu.iter()) {
            *x_i += knu_i;
        }

        // Covariance update: P ← (I − K·H)·P  (standard form)
        let kh = matmul(&k, h_mat); // N×N
        let kh_p = matmul(&kh, &self.p);
        self.p = self.p.sub_mat(&kh_p);

        // --- M-step: update IW hyperparameters ---

        // Posterior innovation: ε = z − H·x_post
        let hx_post: [S; M] = matvec(h_mat, &self.x);
        let eps: [S; M] = core::array::from_fn(|i| z[i] - hx_post[i]);

        // R update:
        //   Ψ_R ← ρ·Ψ_R + ε·εᵀ + H·P·Hᵀ
        //   ν_R ← ρ·(ν_R − M − 1) + 1 + M + 1
        let eps_eps_t = outer_product_mm::<S, M>(&eps, &eps);
        let h_p_post_ht = matmul(h_mat, &matmul(&self.p, &ht)); // M×M
        let psi_r_new = self
            .psi_r
            .scale(self.rho)
            .add_mat(&eps_eps_t)
            .add_mat(&h_p_post_ht);
        let nu_r_new = self.rho * (self.nu_r - m_f - S::ONE) + S::ONE + m_f + S::ONE;

        // Q update:
        //   d   = x_post − x_pred
        //   Ψ_Q ← ρ·Ψ_Q + d·dᵀ + P_post + P_pred
        //   ν_Q ← ρ·(ν_Q − N − 1) + 1 + N + 1
        let d: [S; N] = core::array::from_fn(|i| self.x[i] - x_pred[i]);
        let d_dt = outer_product_nn::<S, N>(&d, &d);
        let psi_q_new = self
            .psi_q
            .scale(self.rho)
            .add_mat(&d_dt)
            .add_mat(&self.p)
            .add_mat(&p_pred);
        let nu_q_new = self.rho * (self.nu_q - n_f - S::ONE) + S::ONE + n_f + S::ONE;

        // Commit updates (only if new DoF remain valid)
        if nu_r_new > m_f + S::ONE {
            self.psi_r = psi_r_new;
            self.nu_r = nu_r_new;
        }
        if nu_q_new > n_f + S::ONE {
            self.psi_q = psi_q_new;
            self.nu_q = nu_q_new;
        }

        Ok(())
    }

    /// Current state estimate.
    pub fn state(&self) -> &[S; N] {
        &self.x
    }

    /// Current error covariance.
    pub fn covariance(&self) -> &Matrix<S, N, N> {
        &self.p
    }

    /// Current IW degrees-of-freedom for Q.
    pub fn nu_q(&self) -> S {
        self.nu_q
    }

    /// Current IW degrees-of-freedom for R.
    pub fn nu_r(&self) -> S {
        self.nu_r
    }

    /// Reset to new state and covariance (hyperparameters unchanged).
    pub fn reset(&mut self, x0: [S; N], p0: Matrix<S, N, N>) {
        self.x = x0;
        self.p = p0;
    }
}

// ─── matrix helpers ─────────────────────────────────────────────────────────

/// Outer product of two M-vectors: result[i][j] = a[i] * b[j].
fn outer_product_mm<S: ControlScalar, const M: usize>(a: &[S; M], b: &[S; M]) -> Matrix<S, M, M> {
    Matrix {
        data: core::array::from_fn(|i| core::array::from_fn(|j| a[i] * b[j])),
    }
}

/// Outer product of two N-vectors: result[i][j] = a[i] * b[j].
fn outer_product_nn<S: ControlScalar, const N: usize>(a: &[S; N], b: &[S; N]) -> Matrix<S, N, N> {
    Matrix {
        data: core::array::from_fn(|i| core::array::from_fn(|j| a[i] * b[j])),
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a scalar 1-state / 1-measurement VB filter with known true noise.
    fn build_scalar_vbf(true_q: f64, true_r: f64, rho: f64) -> VariationalBayesFilter<f64, 1, 1> {
        let x0 = [0.0_f64];
        let p0 = Matrix::<f64, 1, 1> { data: [[100.0]] };
        // Initial Q guess: 10× true value
        let psi_q0 = Matrix::<f64, 1, 1> {
            data: [[true_q * 10.0 * 4.0]],
        }; // ν_Q=4 → mean=true_q*10
        let nu_q0 = 4.0_f64; // > N+1 = 2
                             // Initial R guess: 10× true value
        let psi_r0 = Matrix::<f64, 1, 1> {
            data: [[true_r * 10.0 * 4.0]],
        };
        let nu_r0 = 4.0_f64; // > M+1 = 2
        VariationalBayesFilter::new(x0, p0, nu_q0, psi_q0, nu_r0, psi_r0, rho)
            .expect("valid parameters")
    }

    #[test]
    fn invalid_rho_rejected() {
        let x0 = [0.0_f64];
        let p0 = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let psi = Matrix::<f64, 1, 1> { data: [[1.0]] };
        assert!(VariationalBayesFilter::new(x0, p0, 3.0, psi, 3.0, psi, 0.0).is_err());
        assert!(VariationalBayesFilter::new(x0, p0, 3.0, psi, 3.0, psi, 1.1).is_err());
        assert!(VariationalBayesFilter::new(x0, p0, 3.0, psi, 3.0, psi, -0.1).is_err());
    }

    #[test]
    fn invalid_dof_rejected() {
        let x0 = [0.0_f64];
        let p0 = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let psi = Matrix::<f64, 1, 1> { data: [[1.0]] };
        // nu_q = 2.0 = N + 1 = 2 — NOT strictly greater, so should fail
        assert!(VariationalBayesFilter::new(x0, p0, 2.0, psi, 3.0, psi, 0.99).is_err());
        // nu_r = 2.0 — same issue
        assert!(VariationalBayesFilter::new(x0, p0, 3.0, psi, 2.0, psi, 0.99).is_err());
    }

    #[test]
    fn predict_update_cycle_completes() {
        let mut f = build_scalar_vbf(0.01, 1.0, 0.99);
        let f_mat = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let h_mat = Matrix::<f64, 1, 1> { data: [[1.0]] };
        assert!(f.predict(&f_mat).is_ok());
        assert!(f.update(&h_mat, &[1.0]).is_ok());
    }

    #[test]
    fn filter_converges_on_constant_signal() {
        let mut f = build_scalar_vbf(1e-3, 0.5, 0.99);
        let f_mat = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let h_mat = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let true_val = 3.0_f64;
        for _ in 0..500 {
            f.predict(&f_mat).expect("predict");
            f.update(&h_mat, &[true_val]).expect("update");
        }
        let x = f.state()[0];
        assert!(
            (x - true_val).abs() < 0.5,
            "Filter should converge to {true_val}, got {x}"
        );
    }

    #[test]
    fn noise_covariance_adapts_toward_truth() {
        // True R = 2.0; initial guess R_hat ~ 20.0.
        // After enough observations the estimate should move toward truth.
        let true_r = 2.0_f64;
        let mut f = build_scalar_vbf(1e-3, true_r, 0.95);
        let f_mat = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let h_mat = Matrix::<f64, 1, 1> { data: [[1.0]] };

        let initial_r = f.estimated_r().expect("r mean").data[0][0];

        // Simple deterministic pseudo-measurements (no RNG)
        for k in 0..300 {
            f.predict(&f_mat).expect("predict");
            // Alternate +/- around zero to simulate unit-variance noise
            let sign = if k % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            let z = sign * true_r.sqrt(); // std-dev scaled measurement
            f.update(&h_mat, &[z]).expect("update");
        }

        let final_r = f.estimated_r().expect("r mean").data[0][0];
        // R estimate must have moved from its initial value toward truth
        let initial_dist = (initial_r - true_r).abs();
        let final_dist = (final_r - true_r).abs();
        assert!(
            final_dist < initial_dist,
            "R estimate should move toward truth: initial dist={initial_dist:.4}, final dist={final_dist:.4}"
        );
    }

    #[test]
    fn estimated_q_and_r_positive() {
        let f = build_scalar_vbf(0.01, 1.0, 0.99);
        let q_mean = f.estimated_q().expect("q mean").data[0][0];
        let r_mean = f.estimated_r().expect("r mean").data[0][0];
        assert!(q_mean > 0.0, "Q mean must be positive");
        assert!(r_mean > 0.0, "R mean must be positive");
    }

    #[test]
    fn reset_restores_state() {
        let mut f = build_scalar_vbf(0.01, 1.0, 0.99);
        let f_mat = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let h_mat = Matrix::<f64, 1, 1> { data: [[1.0]] };
        for _ in 0..50 {
            f.predict(&f_mat).expect("predict");
            f.update(&h_mat, &[5.0]).expect("update");
        }
        let p0 = Matrix::<f64, 1, 1> { data: [[100.0]] };
        f.reset([0.0_f64], p0);
        assert!(
            f.state()[0].abs() < 1e-12,
            "State should be 0 after reset, got {}",
            f.state()[0]
        );
    }
}
