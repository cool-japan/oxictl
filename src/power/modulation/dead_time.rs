use crate::core::scalar::ControlScalar;

/// Dead-time compensator for H-bridge / inverter legs.
///
/// During the dead time (both switches off), the phase current freewheels
/// through the anti-parallel diode. This creates a voltage error:
///   V_err = ±V_dc · t_d / T_sw
/// where the sign depends on the direction of the phase current.
///
/// The compensator corrects the duty cycle to pre-compensate for this error.
#[derive(Debug, Clone, Copy)]
pub struct DeadTimeCompensator<S: ControlScalar> {
    /// Dead time (s).
    pub dead_time: S,
    /// Switching period (s).
    pub period: S,
    /// Current zero-crossing threshold (A) — below this, compensation is reduced.
    pub zero_cross_threshold: S,
}

impl<S: ControlScalar> DeadTimeCompensator<S> {
    pub fn new(dead_time: S, period: S) -> Self {
        Self {
            dead_time,
            period,
            zero_cross_threshold: S::from_f64(0.1),
        }
    }

    /// Compensate a single-phase duty cycle.
    ///
    /// - `duty`: uncompensated duty cycle ∈ [0, 1]
    /// - `current`: phase current (A)  — sign determines compensation direction
    ///
    /// Returns compensated duty cycle (still clamped to [0, 1]).
    pub fn compensate_duty(&self, duty: S, current: S) -> S {
        let correction = self.dead_time / self.period;

        // Smooth sign near zero crossing to avoid chattering
        let sign = smooth_sign(current, self.zero_cross_threshold);
        let compensated = duty + sign * correction;
        compensated.clamp_val(S::ZERO, S::ONE)
    }

    /// Compensate three-phase duties.
    ///
    /// - `duties`: [d_a, d_b, d_c] uncompensated
    /// - `currents`: [i_a, i_b, i_c] phase currents (A)
    pub fn compensate_three_phase(&self, duties: [S; 3], currents: [S; 3]) -> [S; 3] {
        core::array::from_fn(|i| self.compensate_duty(duties[i], currents[i]))
    }

    /// Correction magnitude: t_d / T_sw.
    pub fn correction_factor(&self) -> S {
        self.dead_time / self.period
    }
}

/// Smooth sign function using tanh approximation to avoid chattering near zero.
fn smooth_sign<S: ControlScalar>(x: S, threshold: S) -> S {
    // Use linear interpolation in [-threshold, threshold], ±1 outside
    if threshold <= S::ZERO || threshold.abs() < S::from_f64(1e-12) {
        return if x >= S::ZERO { S::ONE } else { -S::ONE };
    }
    let ratio = x / threshold;
    if ratio > S::ONE {
        S::ONE
    } else if ratio < -S::ONE {
        -S::ONE
    } else {
        ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_current_increases_duty() {
        let comp = DeadTimeCompensator::new(2e-6_f64, 100e-6);
        let d = comp.compensate_duty(0.5, 5.0); // positive current
        assert!(d > 0.5, "d={d:.6}");
    }

    #[test]
    fn negative_current_decreases_duty() {
        let comp = DeadTimeCompensator::new(2e-6_f64, 100e-6);
        let d = comp.compensate_duty(0.5, -5.0); // negative current
        assert!(d < 0.5, "d={d:.6}");
    }

    #[test]
    fn duty_stays_in_range() {
        let comp = DeadTimeCompensator::new(5e-6_f64, 100e-6);
        // Large dead time, edge duties
        let d1 = comp.compensate_duty(0.98, 10.0);
        let d2 = comp.compensate_duty(0.02, -10.0);
        assert!(d1 <= 1.0, "d1={d1:.6}");
        assert!(d2 >= 0.0, "d2={d2:.6}");
    }

    #[test]
    fn correction_factor_correct() {
        let comp = DeadTimeCompensator::new(2e-6_f64, 100e-6);
        assert!((comp.correction_factor() - 0.02).abs() < 1e-10);
    }

    #[test]
    fn three_phase_compensation() {
        let comp = DeadTimeCompensator::new(2e-6_f64, 100e-6);
        let duties = [0.5_f64, 0.5, 0.5];
        let currents = [5.0_f64, -5.0, 0.0];
        let compensated = comp.compensate_three_phase(duties, currents);
        assert!(compensated[0] > 0.5); // positive current → higher duty
        assert!(compensated[1] < 0.5); // negative current → lower duty
    }
}
