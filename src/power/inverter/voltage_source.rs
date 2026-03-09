use crate::core::scalar::ControlScalar;

/// Voltage Source Inverter (VSI) controller in αβ frame.
///
/// Controls output voltage using PR (Proportional-Resonant) controllers
/// in stationary αβ reference frame. No coordinate rotation needed.
///
/// Each axis (α, β) runs identical PR controller:
///   C(s) = Kp + 2Ki*ω₀*s / (s² + ω₀²)
///
/// Discrete implementation via bilinear (Tustin) approximation.
#[derive(Debug, Clone, Copy)]
pub struct PrController<S: ControlScalar> {
    /// Proportional gain.
    pub kp: S,
    /// Resonant gain.
    pub ki: S,
    /// Resonant frequency (rad/s).
    pub omega0: S,
    /// Bilinear integrator state 1.
    s1: S,
    /// Bilinear integrator state 2.
    s2: S,
}

impl<S: ControlScalar> PrController<S> {
    pub fn new(kp: S, ki: S, omega0: S) -> Self {
        Self {
            kp,
            ki,
            omega0,
            s1: S::ZERO,
            s2: S::ZERO,
        }
    }

    /// Update PR controller with error signal and time step.
    pub fn update(&mut self, err: S, dt: S) -> S {
        // Bilinear (Tustin) discretization of resonant term:
        // H(z) = 2Ki*ω₀ * (z-1)/(z+1) / (((z-1)/(z+1))² + ω₀²)
        // Implemented as biquad state variable form.
        // Simple trapezoidal resonator:
        // y'' + ω₀²*y = 2Ki*ω₀ * u'
        // States: [s1=y_res, s2=y_res']
        // s1' = s2
        // s2' = -ω₀²*s1 + 2Ki*ω₀*u
        let ki2 = S::TWO * self.ki * self.omega0;
        let new_s2 = self.s2 + dt * (-self.omega0 * self.omega0 * self.s1 + ki2 * err);
        let new_s1 = self.s1 + dt * self.s2;

        self.s1 = new_s1;
        self.s2 = new_s2;

        self.kp * err + self.s1
    }

    pub fn reset(&mut self) {
        self.s1 = S::ZERO;
        self.s2 = S::ZERO;
    }
}

/// VSI controller: two PR controllers (α and β axes) with output limiting.
#[derive(Debug, Clone, Copy)]
pub struct VsiController<S: ControlScalar> {
    /// PR controller for α axis.
    pub pr_alpha: PrController<S>,
    /// PR controller for β axis.
    pub pr_beta: PrController<S>,
    /// DC bus voltage (used for duty-cycle normalization).
    pub v_dc: S,
    /// Output voltage limit (V).
    pub v_max: S,
}

impl<S: ControlScalar> VsiController<S> {
    pub fn new(kp: S, ki: S, omega0: S, v_dc: S, v_max: S) -> Self {
        Self {
            pr_alpha: PrController::new(kp, ki, omega0),
            pr_beta: PrController::new(kp, ki, omega0),
            v_dc,
            v_max,
        }
    }

    /// Update VSI voltage controller.
    ///
    /// - `v_alpha_ref`, `v_beta_ref`: reference voltage (αβ)
    /// - `v_alpha`, `v_beta`: measured voltage (αβ)
    /// - `dt`: time step
    ///
    /// Returns `(m_alpha, m_beta)`: modulation indices ∈ [-1, 1].
    pub fn update(
        &mut self,
        v_alpha_ref: S,
        v_beta_ref: S,
        v_alpha: S,
        v_beta: S,
        dt: S,
    ) -> (S, S) {
        let err_a = v_alpha_ref - v_alpha;
        let err_b = v_beta_ref - v_beta;

        let u_alpha = self.pr_alpha.update(err_a, dt);
        let u_beta = self.pr_beta.update(err_b, dt);

        // Normalize to modulation index
        let half_vdc = self.v_dc / S::TWO;
        if half_vdc.abs() < S::from_f64(1e-12) {
            return (S::ZERO, S::ZERO);
        }

        let m_a = u_alpha / half_vdc;
        let m_b = u_beta / half_vdc;

        // Vector amplitude limiting
        let amp = (m_a * m_a + m_b * m_b).sqrt();
        let limit = S::ONE;
        if amp > limit {
            let scale = limit / amp;
            (m_a * scale, m_b * scale)
        } else {
            (m_a, m_b)
        }
    }

    pub fn reset(&mut self) {
        self.pr_alpha.reset();
        self.pr_beta.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn pr_controller_tracks_sine() {
        let omega0 = 2.0 * PI * 50.0;
        let dt = 1e-4;
        let mut pr = PrController::new(1.0_f64, 20.0, omega0);

        // Apply sinusoidal reference, controller should produce non-zero output
        let mut max_out = 0.0f64;
        for k in 0..1000 {
            let t = k as f64 * dt;
            let r = (omega0 * t).sin();
            let out = pr.update(r, dt);
            if out.abs() > max_out {
                max_out = out.abs();
            }
        }
        assert!(max_out > 0.1, "PR output should grow: max={max_out:.4}");
    }

    #[test]
    fn vsi_output_bounded() {
        let omega0 = 2.0 * PI * 50.0;
        let dt = 1e-4;
        let mut vsi = VsiController::new(2.0_f64, 50.0, omega0, 400.0, 230.0);

        for k in 0..500 {
            let t = k as f64 * dt;
            let r = 230.0 * (omega0 * t).sin();
            let (ma, mb) = vsi.update(r, 0.0, 0.0, 0.0, dt);
            assert!(ma.abs() <= 1.0 + 1e-9, "m_alpha={ma:.4} out of [-1,1]");
            assert!(mb.abs() <= 1.0 + 1e-9, "m_beta={mb:.4} out of [-1,1]");
        }
    }

    #[test]
    fn pr_reset_clears_state() {
        let mut pr = PrController::new(1.0_f64, 10.0, 314.16);
        for _ in 0..100 {
            pr.update(1.0, 1e-4);
        }
        pr.reset();
        assert_eq!(pr.s1, 0.0);
        assert_eq!(pr.s2, 0.0);
    }
}
