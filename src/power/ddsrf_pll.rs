use crate::core::scalar::ControlScalar;

/// Decoupled Double Synchronous Reference Frame PLL (DDSRF-PLL).
///
/// Handles unbalanced three-phase grid voltages by separating positive and
/// negative sequence components in dq frames rotating at +ω and −ω.
///
/// Architecture:
///   - Positive sequence frame: rotates at +ω̂
///   - Negative sequence frame: rotates at −ω̂
///   - Cross-coupling cancellation between the two frames
///   - PI controller on positive-sequence q-axis error → frequency estimate
///
/// Reference: Rodríguez et al., "Decoupled Double Synchronous Reference Frame
/// PLL for Power Converters Control" (IEEE Trans. Power Electron., 2007)
#[derive(Debug, Clone, Copy)]
pub struct DdsrfPll<S: ControlScalar> {
    /// Estimated frequency (rad/s).
    pub omega: S,
    /// Estimated positive-sequence angle (rad).
    pub theta_pos: S,
    /// Estimated negative-sequence angle (rad).
    pub theta_neg: S,

    /// Positive-sequence dq estimates (after cross-coupling cancellation).
    pub vd_pos: S,
    pub vq_pos: S,

    /// Negative-sequence dq estimates.
    pub vd_neg: S,
    pub vq_neg: S,

    /// Low-pass filter state for cross-coupling term (positive frame, d).
    lp_pos_d: S,
    /// Low-pass filter state for cross-coupling term (positive frame, q).
    lp_pos_q: S,
    /// Low-pass filter state for cross-coupling term (negative frame, d).
    lp_neg_d: S,
    /// Low-pass filter state for cross-coupling term (negative frame, q).
    lp_neg_q: S,

    /// PI controller: proportional gain.
    pub kp: S,
    /// PI controller: integral gain.
    pub ki: S,
    /// PI integrator state.
    pi_int: S,
    /// Nominal frequency (rad/s).
    pub omega_nom: S,
    /// Low-pass filter bandwidth (rad/s) for cross-coupling.
    pub lp_bw: S,
}

impl<S: ControlScalar> DdsrfPll<S> {
    /// Create a new DDSRF-PLL.
    ///
    /// - `omega_nom`: nominal grid frequency (e.g. 2π·50 ≈ 314.16 rad/s)
    /// - `kp`, `ki`: PI gains for frequency loop
    /// - `lp_bw`: low-pass filter bandwidth (rad/s) for decoupling filters
    pub fn new(omega_nom: S, kp: S, ki: S, lp_bw: S) -> Self {
        Self {
            omega: omega_nom,
            theta_pos: S::ZERO,
            theta_neg: S::ZERO,
            vd_pos: S::ZERO,
            vq_pos: S::ZERO,
            vd_neg: S::ZERO,
            vq_neg: S::ZERO,
            lp_pos_d: S::ZERO,
            lp_pos_q: S::ZERO,
            lp_neg_d: S::ZERO,
            lp_neg_q: S::ZERO,
            kp,
            ki,
            pi_int: S::ZERO,
            omega_nom,
            lp_bw,
        }
    }

    /// Update PLL with three-phase voltages [va, vb, vc] and time step dt.
    ///
    /// Returns (theta_pos, omega): estimated positive-sequence angle and frequency.
    pub fn update(&mut self, v_abc: &[S; 3], dt: S) -> (S, S) {
        // Clarke transform: αβ from abc
        let two_thirds = S::from_f64(2.0 / 3.0);
        let inv_sqrt3 = S::from_f64(1.0 / 1.732_050_808);
        let v_alpha =
            two_thirds * (v_abc[0] - S::from_f64(0.5) * v_abc[1] - S::from_f64(0.5) * v_abc[2]);
        let v_beta = two_thirds * inv_sqrt3 * (v_abc[1] - v_abc[2]);

        // Park transform into positive frame (+θ̂)
        let cos_p = self.theta_pos.cos();
        let sin_p = self.theta_pos.sin();
        let vd_p_raw = cos_p * v_alpha + sin_p * v_beta;
        let vq_p_raw = -sin_p * v_alpha + cos_p * v_beta;

        // Park transform into negative frame (−θ̂, i.e. rotate by −θ̂)
        let cos_n = self.theta_neg.cos();
        let sin_n = self.theta_neg.sin();
        let vd_n_raw = cos_n * v_alpha + sin_n * v_beta;
        let vq_n_raw = -sin_n * v_alpha + cos_n * v_beta;

        // Cross-coupling decoupling via first-order low-pass filters
        // The negative sequence appears as 2ω oscillation in positive frame.
        // Use filtered negative dq estimates to cancel.
        let alpha_lp = S::ONE - (-self.lp_bw * dt).exp();

        // Update LP filters
        self.lp_pos_d += alpha_lp * (vd_p_raw - self.lp_pos_d);
        self.lp_pos_q += alpha_lp * (vq_p_raw - self.lp_pos_q);
        self.lp_neg_d += alpha_lp * (vd_n_raw - self.lp_neg_d);
        self.lp_neg_q += alpha_lp * (vq_n_raw - self.lp_neg_q);

        // Decoupled estimates: subtract cross-frame contributions
        // Positive frame sees negative at 2θ → approximate via LP-filtered neg frame
        let cos_2p = (S::TWO * self.theta_pos).cos();
        let sin_2p = (S::TWO * self.theta_pos).sin();
        self.vd_pos = self.lp_pos_d - (cos_2p * self.lp_neg_d + sin_2p * self.lp_neg_q);
        self.vq_pos = self.lp_pos_q - (-sin_2p * self.lp_neg_d + cos_2p * self.lp_neg_q);

        let cos_2n = (S::TWO * self.theta_neg).cos();
        let sin_2n = (S::TWO * self.theta_neg).sin();
        self.vd_neg = self.lp_neg_d - (cos_2n * self.lp_pos_d + sin_2n * self.lp_pos_q);
        self.vq_neg = self.lp_neg_q - (-sin_2n * self.lp_pos_d + cos_2n * self.lp_pos_q);

        // PI controller: drive vq_pos → 0
        let err = self.vq_pos;
        self.pi_int += self.ki * err * dt;
        let delta_omega = self.kp * err + self.pi_int;

        // Update frequency and angles
        self.omega = self.omega_nom + delta_omega;
        self.theta_pos += self.omega * dt;
        self.theta_neg -= self.omega * dt;

        // Wrap angles to [-π, π]
        let pi = S::PI;
        let two_pi = S::TWO * pi;
        while self.theta_pos > pi {
            self.theta_pos -= two_pi;
        }
        while self.theta_pos < -pi {
            self.theta_pos += two_pi;
        }
        while self.theta_neg > pi {
            self.theta_neg -= two_pi;
        }
        while self.theta_neg < -pi {
            self.theta_neg += two_pi;
        }

        (self.theta_pos, self.omega)
    }

    /// Reset PLL state.
    pub fn reset(&mut self) {
        self.omega = self.omega_nom;
        self.theta_pos = S::ZERO;
        self.theta_neg = S::ZERO;
        self.vd_pos = S::ZERO;
        self.vq_pos = S::ZERO;
        self.vd_neg = S::ZERO;
        self.vq_neg = S::ZERO;
        self.lp_pos_d = S::ZERO;
        self.lp_pos_q = S::ZERO;
        self.lp_neg_d = S::ZERO;
        self.lp_neg_q = S::ZERO;
        self.pi_int = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn balanced_abc(theta: f64, amp: f64) -> [f64; 3] {
        [
            amp * theta.cos(),
            amp * (theta - 2.0 * PI / 3.0).cos(),
            amp * (theta + 2.0 * PI / 3.0).cos(),
        ]
    }

    #[test]
    fn locks_to_balanced_grid() {
        let omega_nom = 2.0 * PI * 50.0;
        let dt = 1e-4;
        let mut pll = DdsrfPll::new(omega_nom, 50.0, 500.0, 200.0);

        let mut theta_grid = 0.0f64;
        for _ in 0..5000 {
            let v = balanced_abc(theta_grid, 1.0);
            pll.update(&v, dt);
            theta_grid += omega_nom * dt;
            // Wrap
            if theta_grid > PI {
                theta_grid -= 2.0 * PI;
            }
        }

        // Frequency should be close to nominal
        let omega_err = (pll.omega - omega_nom).abs();
        assert!(omega_err < 5.0, "omega err={omega_err:.3} rad/s");
    }

    #[test]
    fn handles_unbalanced_grid() {
        let omega_nom = 2.0 * PI * 50.0;
        let dt = 1e-4;
        let mut pll = DdsrfPll::new(omega_nom, 50.0, 500.0, 200.0);

        let mut theta_grid = 0.0f64;
        for _ in 0..8000 {
            // Unbalanced: phase A has 0.8 amplitude
            let v = [
                0.8 * theta_grid.cos(),
                (theta_grid - 2.0 * PI / 3.0).cos(),
                (theta_grid + 2.0 * PI / 3.0).cos(),
            ];
            pll.update(&v, dt);
            theta_grid += omega_nom * dt;
            if theta_grid > PI {
                theta_grid -= 2.0 * PI;
            }
        }

        let omega_err = (pll.omega - omega_nom).abs();
        assert!(
            omega_err < 20.0,
            "omega err={omega_err:.3} rad/s (unbalanced)"
        );
    }

    #[test]
    fn vq_pos_near_zero_at_lock() {
        let omega_nom = 2.0 * PI * 50.0;
        let dt = 1e-4;
        let mut pll = DdsrfPll::new(omega_nom, 50.0, 500.0, 200.0);

        let mut theta_grid = 0.0f64;
        for _ in 0..6000 {
            let v = balanced_abc(theta_grid, 1.0);
            pll.update(&v, dt);
            theta_grid += omega_nom * dt;
            if theta_grid > PI {
                theta_grid -= 2.0 * PI;
            }
        }

        assert!(pll.vq_pos.abs() < 0.1, "vq_pos={:.4}", pll.vq_pos);
    }
}
