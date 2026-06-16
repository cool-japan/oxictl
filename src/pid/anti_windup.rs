use crate::core::scalar::PidScalar;

/// Anti-windup strategy for PID integral term.
#[derive(Debug, Clone, Copy, Default)]
pub enum AntiWindupMethod<S: PidScalar> {
    /// No anti-windup; integrator runs freely.
    None,
    /// Clamp the integrator when output is saturated.
    #[default]
    Clamping,
    /// Back-calculation with tracking time constant Kb.
    BackCalculation {
        /// Back-calculation gain. Typical: Kb = 1/Ti or Kb = sqrt(Ki*Kd).
        kb: S,
    },
}

impl<S: PidScalar> AntiWindupMethod<S> {
    /// Compute the integrator correction term.
    /// - `output_unlimited`: raw PID output before limiting
    /// - `output_limited`: output after saturation limiter
    /// - `error`: current control error
    /// - `dt`: time step
    ///
    /// Returns the corrected integral increment.
    pub fn correct_integral(
        &self,
        integral_term: S,
        output_unlimited: S,
        output_limited: S,
        error: S,
        ki: S,
        dt: S,
    ) -> S {
        match self {
            AntiWindupMethod::None => integral_term + ki * error * dt,
            AntiWindupMethod::Clamping => {
                let is_saturated = (output_unlimited - output_limited).abs() > S::EPSILON;
                let same_sign = (error * (output_unlimited - output_limited)) > S::ZERO;
                if is_saturated && same_sign {
                    // Don't integrate further in the saturating direction
                    integral_term
                } else {
                    integral_term + ki * error * dt
                }
            }
            AntiWindupMethod::BackCalculation { kb } => {
                let saturation_error = output_limited - output_unlimited;
                integral_term + (ki * error + *kb * saturation_error) * dt
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_always_integrates() {
        let aw = AntiWindupMethod::<f64>::None;
        let result = aw.correct_integral(0.0, 15.0, 10.0, 1.0, 1.0, 0.01);
        assert!((result - 0.01).abs() < 1e-10);
    }

    #[test]
    fn clamping_stops_when_saturated_same_direction() {
        let aw = AntiWindupMethod::<f64>::Clamping;
        // Output is saturated high (15 -> 10), error is positive (pushing higher)
        let result = aw.correct_integral(5.0, 15.0, 10.0, 1.0, 1.0, 0.01);
        // Should NOT integrate further
        assert_eq!(result, 5.0);
    }

    #[test]
    fn clamping_integrates_when_recovering() {
        let aw = AntiWindupMethod::<f64>::Clamping;
        // Output is saturated high (15 -> 10), but error is negative (recovering)
        let result = aw.correct_integral(5.0, 15.0, 10.0, -1.0, 1.0, 0.01);
        // Should integrate (reducing the integral)
        assert!((result - 4.99).abs() < 1e-10);
    }

    #[test]
    fn back_calculation_drives_integral_toward_limit() {
        let aw = AntiWindupMethod::<f64>::BackCalculation { kb: 1.0 };
        // output_limited < output_unlimited → saturation_error is negative
        let result = aw.correct_integral(5.0, 15.0, 10.0, 1.0, 1.0, 0.01);
        // integral + (ki*error + kb*(10-15)) * dt = 5.0 + (1.0 - 5.0)*0.01 = 5.0 - 0.04 = 4.96
        assert!((result - 4.96).abs() < 1e-10);
    }

    #[test]
    fn clamping_integrates_when_not_saturated() {
        let aw = AntiWindupMethod::<f64>::Clamping;
        let result = aw.correct_integral(5.0, 8.0, 8.0, 1.0, 1.0, 0.01);
        assert!((result - 5.01).abs() < 1e-10);
    }
}
