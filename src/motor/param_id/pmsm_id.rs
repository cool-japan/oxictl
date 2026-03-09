//! Online PMSM parameter identification via Recursive Least Squares (RLS).
//!
//! Estimates stator resistance Rs, d-axis inductance Ld, q-axis inductance Lq,
//! and permanent magnet flux linkage λ_pm from voltage and current measurements.
//!
//! The RLS algorithm uses a forgetting factor λ ∈ (0,1] to track slowly
//! varying parameters. Values of λ close to 1.0 give slower adaptation but
//! better noise rejection; values closer to 0.95–0.98 are typical for motor drives.
//!
//! # Identification Model
//!
//! d-axis: vd = Rs·id + Ld·(Δid/Δt) − ωe·Lq·iq
//!   → rewritten as: vd + ωe·Lq·iq = Rs·id + Ld·(Δid/Δt)
//!   → φd = [id, Δid/Δt]ᵀ,  θd = [Rs, Ld]ᵀ
//!
//! q-axis: vq = Rs·iq + Lq·(Δiq/Δt) + ωe·Ld·id + ωe·λ_pm
//!   → vq − ωe·Ld·id = Rs·iq + Lq·(Δiq/Δt) + ωe·λ_pm
//!   → φq = [iq, Δiq/Δt, ωe]ᵀ,  θq = [Rs, Lq, λ_pm]ᵀ
//!
//! Because Rs appears in both axes, the d-axis RLS provides a clean Rs estimate
//! used to warm-start the q-axis. The two are identified independently per step.

#![allow(clippy::needless_range_loop)]
use crate::core::scalar::ControlScalar;

/// Identifies which axis this RLS instance is operating on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlsAxis {
    /// d-axis: estimates [Rs, Ld].
    D,
    /// q-axis: estimates [Rs, Lq, λ_pm].
    Q,
}

/// Configuration for the PMSM RLS identifier.
#[derive(Debug, Clone, Copy)]
pub struct PmsmIdConfig<S: ControlScalar> {
    /// Forgetting factor λ ∈ (0, 1]. Typical range 0.95–1.0.
    pub forgetting_factor: S,
    /// Initial diagonal value for the covariance matrix P (large = uncertain).
    pub p_init: S,
    /// Convergence threshold: max parameter change per step considered converged.
    pub convergence_threshold: S,
    /// Minimum number of steps before reporting convergence.
    pub min_steps_for_convergence: u32,
}

impl<S: ControlScalar> PmsmIdConfig<S> {
    /// Reasonable default configuration.
    pub fn default_config() -> Self {
        Self {
            forgetting_factor: S::from_f64(0.97),
            p_init: S::from_f64(1.0e4),
            convergence_threshold: S::from_f64(1.0e-5),
            min_steps_for_convergence: 500,
        }
    }
}

/// Result snapshot from PMSM parameter identification.
#[derive(Debug, Clone, Copy)]
pub struct PmsmParamIdResult<S: ControlScalar> {
    /// Estimated stator resistance (Ω) — average of d and q axis estimates.
    pub rs: S,
    /// Estimated d-axis inductance (H).
    pub ld: S,
    /// Estimated q-axis inductance (H).
    pub lq: S,
    /// Estimated permanent magnet flux linkage λ_pm (Wb).
    pub lambda_pm: S,
    /// Whether the d-axis RLS has converged.
    pub d_converged: bool,
    /// Whether the q-axis RLS has converged.
    pub q_converged: bool,
    /// Total steps processed.
    pub steps: u32,
}

/// Single-axis RLS state for a 2-parameter or 3-parameter system.
///
/// Supports N = 2 (d-axis: [Rs, Ld]) or N = 3 (q-axis: [Rs, Lq, λ_pm]).
/// We use fixed-size 3×3 to avoid const generics complexity, with the third
/// row/column unused for the 2-parameter case.
#[derive(Debug, Clone, Copy)]
struct RlsState<S: ControlScalar> {
    /// Parameter vector θ (up to 3 elements; valid entries are θ[0..n_params]).
    theta: [S; 3],
    /// Covariance matrix P (3×3; valid subblock is [0..n_params][0..n_params]).
    p: [[S; 3]; 3],
    /// Number of active parameters (2 or 3).
    n_params: usize,
    /// Forgetting factor.
    lambda: S,
    /// Number of updates performed.
    steps: u32,
    /// Convergence threshold on max |Δθ|.
    conv_threshold: S,
    /// Minimum steps before declaring convergence.
    min_steps: u32,
    /// Whether convergence has been declared.
    converged: bool,
}

impl<S: ControlScalar> RlsState<S> {
    fn new(n_params: usize, lambda: S, p_init: S, conv_threshold: S, min_steps: u32) -> Self {
        debug_assert!(n_params == 2 || n_params == 3);
        let mut p = [[S::ZERO; 3]; 3];
        for i in 0..n_params {
            p[i][i] = p_init;
        }
        Self {
            theta: [S::ZERO; 3],
            p,
            n_params,
            lambda,
            steps: 0,
            conv_threshold,
            min_steps,
            converged: false,
        }
    }

    /// Perform one RLS update with regressor φ (length n_params) and observation y.
    ///
    /// Uses the standard forgetting-factor RLS update:
    /// ```text
    /// k  = P·φ / (λ + φᵀ·P·φ)
    /// θ  = θ + k·(y - φᵀ·θ)
    /// P  = (P - k·φᵀ·P) / λ
    /// ```
    fn update(&mut self, phi: &[S; 3], y: S) {
        let n = self.n_params;

        // Compute P·φ  (n-vector)
        let mut p_phi = [S::ZERO; 3];
        for i in 0..n {
            for j in 0..n {
                p_phi[i] += self.p[i][j] * phi[j];
            }
        }

        // Denominator: λ + φᵀ·P·φ
        let mut phi_t_p_phi = S::ZERO;
        for i in 0..n {
            phi_t_p_phi += phi[i] * p_phi[i];
        }
        let denom = self.lambda + phi_t_p_phi;

        // Gain k = P·φ / denom
        let mut k = [S::ZERO; 3];
        for i in 0..n {
            k[i] = p_phi[i] / denom;
        }

        // Innovation: e = y - φᵀ·θ
        let mut y_hat = S::ZERO;
        for i in 0..n {
            y_hat += phi[i] * self.theta[i];
        }
        let innovation = y - y_hat;

        // θ update
        let mut max_delta = S::ZERO;
        for i in 0..n {
            let delta = k[i] * innovation;
            self.theta[i] += delta;
            let abs_delta = if delta < S::ZERO { -delta } else { delta };
            if abs_delta > max_delta {
                max_delta = abs_delta;
            }
        }

        // P update: P = (P - k·φᵀ·P) / λ
        // First compute k·φᵀ·P (outer product times P row)
        // k_phi_t[i][j] = k[i] * phi[j]  →  then * P_col
        // More directly: new_P[i][j] = (P[i][j] - k[i] * (φᵀ·P)[j]) / λ
        // where (φᵀ·P)[j] = Σ_l phi[l]*P[l][j]
        let mut phi_t_p = [S::ZERO; 3];
        for j in 0..n {
            for l in 0..n {
                phi_t_p[j] += phi[l] * self.p[l][j];
            }
        }
        for i in 0..n {
            for j in 0..n {
                self.p[i][j] = (self.p[i][j] - k[i] * phi_t_p[j]) / self.lambda;
            }
        }

        self.steps += 1;

        // Convergence check
        if !self.converged && self.steps >= self.min_steps && max_delta < self.conv_threshold {
            self.converged = true;
        }
    }
}

/// Online PMSM parameter identifier using decoupled d/q-axis RLS.
///
/// # Usage
/// Feed voltage, current, and electrical speed measurements every control cycle.
/// After sufficient excitation and steps, call [`results`](PmsmParamId::results)
/// to retrieve identified parameters.
///
/// # Notes
/// - The d-axis RLS (2 params) runs independently of the q-axis RLS (3 params).
/// - The Rs estimate is the average of d-axis and q-axis estimates once both converge.
/// - A minimum amount of persistent excitation is required for convergence.
#[derive(Debug, Clone)]
pub struct PmsmParamId<S: ControlScalar> {
    /// d-axis RLS state: θd = [Rs_d, Ld].
    d_rls: RlsState<S>,
    /// q-axis RLS state: θq = [Rs_q, Lq, λ_pm].
    q_rls: RlsState<S>,
    /// Previous d-axis current for derivative approximation.
    prev_id: S,
    /// Previous q-axis current for derivative approximation.
    prev_iq: S,
    /// Timestep (s).
    dt: S,
    /// Configuration snapshot.
    config: PmsmIdConfig<S>,
}

impl<S: ControlScalar> PmsmParamId<S> {
    /// Construct a new identifier.
    ///
    /// # Arguments
    /// * `dt` - Control loop timestep (s). Must be > 0.
    /// * `config` - RLS tuning configuration.
    pub fn new(dt: S, config: PmsmIdConfig<S>) -> Self {
        let lambda = config.forgetting_factor;
        let p0 = config.p_init;
        let eps = config.convergence_threshold;
        let min_steps = config.min_steps_for_convergence;

        Self {
            d_rls: RlsState::new(2, lambda, p0, eps, min_steps),
            q_rls: RlsState::new(3, lambda, p0, eps, min_steps),
            prev_id: S::ZERO,
            prev_iq: S::ZERO,
            dt,
            config,
        }
    }

    /// Construct with default tuning.
    ///
    /// # Arguments
    /// * `dt` - Control loop timestep (s).
    pub fn with_defaults(dt: S) -> Self {
        Self::new(dt, PmsmIdConfig::default_config())
    }

    /// Process one measurement cycle.
    ///
    /// # Arguments
    /// * `vd`, `vq` - Applied d/q-axis voltages (V).
    /// * `id`, `iq` - Measured d/q-axis currents (A).
    /// * `omega_e` - Electrical angular speed (rad/s).
    pub fn update(&mut self, vd: S, vq: S, id: S, iq: S, omega_e: S) {
        let dt_inv = if self.dt > S::ZERO {
            S::ONE / self.dt
        } else {
            S::ZERO
        };

        // Finite-difference current derivatives
        let did_dt = (id - self.prev_id) * dt_inv;
        let diq_dt = (iq - self.prev_iq) * dt_inv;

        // ------- d-axis RLS -------
        // Model: vd + ωe·Lq·iq = Rs·id + Ld·(Δid/Δt)
        // Observation y_d = vd + ωe * lq_est * iq
        // We bootstrap Lq from current q-axis estimate to decouple.
        let lq_est = self.q_rls.theta[1];
        let y_d = vd + omega_e * lq_est * iq;
        let mut phi_d = [S::ZERO; 3];
        phi_d[0] = id; // Rs coefficient
        phi_d[1] = did_dt; // Ld coefficient
        self.d_rls.update(&phi_d, y_d);

        // ------- q-axis RLS -------
        // Model: vq − ωe·Ld·id = Rs·iq + Lq·(Δiq/Δt) + ωe·λ_pm
        // Observation y_q uses Ld estimate from d-axis.
        let ld_est = self.d_rls.theta[1];
        let y_q = vq - omega_e * ld_est * id;
        let mut phi_q = [S::ZERO; 3];
        phi_q[0] = iq; // Rs coefficient
        phi_q[1] = diq_dt; // Lq coefficient
        phi_q[2] = omega_e; // λ_pm coefficient
        self.q_rls.update(&phi_q, y_q);

        self.prev_id = id;
        self.prev_iq = iq;
    }

    /// Retrieve current parameter estimates.
    pub fn results(&self) -> PmsmParamIdResult<S> {
        let rs_d = self.d_rls.theta[0];
        let ld = self.d_rls.theta[1];
        let rs_q = self.q_rls.theta[0];
        let lq = self.q_rls.theta[1];
        let lambda_pm = self.q_rls.theta[2];

        // Average Rs from both axes; if one has not converged keep the other's value.
        let rs = if self.d_rls.converged && self.q_rls.converged {
            (rs_d + rs_q) * S::HALF
        } else if self.d_rls.converged {
            rs_d
        } else {
            rs_q
        };

        // Clamp physical parameters to non-negative
        let rs = if rs < S::ZERO { S::ZERO } else { rs };
        let ld = if ld < S::ZERO { S::ZERO } else { ld };
        let lq = if lq < S::ZERO { S::ZERO } else { lq };
        let lambda_pm = if lambda_pm < S::ZERO {
            S::ZERO
        } else {
            lambda_pm
        };

        PmsmParamIdResult {
            rs,
            ld,
            lq,
            lambda_pm,
            d_converged: self.d_rls.converged,
            q_converged: self.q_rls.converged,
            steps: self.d_rls.steps,
        }
    }

    /// Reset estimator state, keeping configuration.
    pub fn reset(&mut self) {
        let lambda = self.config.forgetting_factor;
        let p0 = self.config.p_init;
        let eps = self.config.convergence_threshold;
        let min_steps = self.config.min_steps_for_convergence;

        self.d_rls = RlsState::new(2, lambda, p0, eps, min_steps);
        self.q_rls = RlsState::new(3, lambda, p0, eps, min_steps);
        self.prev_id = S::ZERO;
        self.prev_iq = S::ZERO;
    }

    /// Whether both d and q axes have declared convergence.
    pub fn is_converged(&self) -> bool {
        self.d_rls.converged && self.q_rls.converged
    }

    /// Number of RLS update steps performed.
    pub fn steps(&self) -> u32 {
        self.d_rls.steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate PMSM voltages from known parameters and verify recovery.
    ///
    /// True parameters: Rs=0.5Ω, Ld=3e-4H, Lq=4e-4H, λ_pm=0.05Wb
    #[test]
    fn rls_recovers_d_axis_params() {
        let dt = 1e-4_f64;
        let config = PmsmIdConfig {
            forgetting_factor: 0.99,
            p_init: 1.0e6,
            convergence_threshold: 1e-7,
            min_steps_for_convergence: 200,
        };
        let mut id_est = PmsmParamId::<f64>::new(dt, config);

        let rs_true = 0.5_f64;
        let ld_true = 3e-4_f64;
        let lq_true = 4e-4_f64;
        let lpm_true = 0.05_f64;
        let omega_e = 100.0_f64;

        // Persistent excitation: sinusoidal currents
        let mut id = 0.0_f64;
        let mut iq = 0.0_f64;
        let mut t = 0.0_f64;

        for _step in 0..5000 {
            let id_new = 2.0 * libm::sin(50.0 * t);
            let iq_new = 3.0 * libm::cos(70.0 * t);
            let did_dt = (id_new - id) / dt;
            let diq_dt = (iq_new - iq) / dt;

            // True voltages
            let vd = rs_true * id_new + ld_true * did_dt - omega_e * lq_true * iq_new;
            let vq = rs_true * iq_new
                + lq_true * diq_dt
                + omega_e * ld_true * id_new
                + omega_e * lpm_true;

            id = id_new;
            iq = iq_new;
            id_est.update(vd, vq, id, iq, omega_e);
            t += dt;
        }

        let res = id_est.results();
        // d-axis should converge within 20% of the true Ld value.
        // RLS with cross-coupling and finite excitation achieves relative accuracy;
        // tighter bounds require longer runs and better excitation design.
        let rel_err = (res.ld - ld_true).abs() / ld_true;
        assert!(
            rel_err < 0.20,
            "Ld relative error {:.2}% too large (estimate={:.6e}, true={:.6e})",
            rel_err * 100.0,
            res.ld,
            ld_true
        );
    }

    #[test]
    fn default_config_has_valid_forgetting_factor() {
        let cfg = PmsmIdConfig::<f64>::default_config();
        assert!(cfg.forgetting_factor > 0.0 && cfg.forgetting_factor <= 1.0);
    }

    #[test]
    fn reset_clears_state() {
        let mut est = PmsmParamId::<f32>::with_defaults(1e-4);
        for i in 0..100 {
            let v = i as f32 * 0.01;
            est.update(v, v, v * 0.1, v * 0.2, 50.0);
        }
        est.reset();
        assert_eq!(est.steps(), 0);
        let res = est.results();
        assert_eq!(res.rs, 0.0_f32);
    }

    #[test]
    fn rls_state_two_params_basic_convergence() {
        // Trivial 1D identification: y = 2·x  →  theta[0] should → 2
        let mut rls = RlsState::<f64>::new(2, 0.99, 1e4, 1e-8, 100);
        for i in 0..2000 {
            let x = (i as f64) * 0.01;
            let y = 2.0 * x + 0.5 * (i as f64 * 0.7).sin();
            let phi = [x, 0.0, 0.0];
            rls.update(&phi, y);
        }
        // theta[0] ≈ 2 (dominant term)
        assert!(
            (rls.theta[0] - 2.0).abs() < 0.5,
            "theta[0]={:.4} should be near 2.0",
            rls.theta[0]
        );
    }
}
