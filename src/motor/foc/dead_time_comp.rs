//! Dead-time compensation for 3-phase inverter.
//!
//! Inverter dead time causes a voltage error proportional to:
//!   ΔV = Vdc · td · fsw · sign(i)
//!
//! This module corrects the reference voltage by the estimated error.
#![allow(clippy::excessive_precision)]

use crate::core::scalar::ControlScalar;

/// Polarity-based average voltage correction for dead-time effect.
///
/// The dead-time compensation voltage magnitude is:
/// `Vdt = Vdc * td * fsw`
///
/// The sign of the correction follows the phase current polarity.
/// Near zero current (within `threshold`) polarity is uncertain and
/// a proportional correction is applied instead of a step.
#[derive(Debug, Clone, Copy)]
pub struct DeadTimeCompensator<S: ControlScalar> {
    /// Dead time duration in seconds.
    pub dead_time: S,
    /// DC bus voltage.
    pub v_dc: S,
    /// Current polarity threshold (below this → proportional correction).
    pub threshold: S,
    /// Switching frequency (Hz).
    pub f_sw: S,
}

impl<S: ControlScalar> DeadTimeCompensator<S> {
    /// Create a new dead-time compensator.
    ///
    /// # Arguments
    /// * `dead_time` - Dead-time interval (s), e.g. 500e-9 for 500 ns.
    /// * `v_dc` - DC bus voltage (V).
    /// * `threshold` - Current magnitude below which polarity is uncertain (A).
    /// * `f_sw` - Switching frequency (Hz).
    pub fn new(dead_time: S, v_dc: S, threshold: S, f_sw: S) -> Self {
        Self {
            dead_time,
            v_dc,
            threshold,
            f_sw,
        }
    }

    /// Dead-time voltage error magnitude: Vdt = Vdc · td · fsw.
    ///
    /// This is the average voltage error per switching period caused by
    /// dead time insertion.
    pub fn dead_time_voltage(&self) -> S {
        self.v_dc * self.dead_time * self.f_sw
    }

    /// Compute compensation voltage for one phase.
    ///
    /// The corrected reference is:
    ///   v_comp = v_ref + Vdt · polarity(i_phase)
    ///
    /// Near zero current, a linear interpolation avoids chattering:
    ///   polarity = i_phase / threshold (for |i| < threshold)
    ///
    /// # Arguments
    /// * `v_ref` - Reference voltage for this phase (V).
    /// * `i_phase` - Phase current (A).
    ///
    /// # Returns
    /// Corrected (compensated) phase voltage reference.
    pub fn compensate_phase(&self, v_ref: S, i_phase: S) -> S {
        let vdt = self.dead_time_voltage();
        let abs_i = i_phase.abs();

        let polarity = if abs_i >= self.threshold {
            i_phase.signum()
        } else if self.threshold > S::ZERO {
            // Proportional region to avoid switching noise near zero crossing
            i_phase / self.threshold
        } else {
            S::ZERO
        };

        v_ref + vdt * polarity
    }

    /// Compensate all three phases at once.
    ///
    /// # Arguments
    /// * `v_ref` - Reference voltages [Va, Vb, Vc] (V).
    /// * `i_phase` - Phase currents [Ia, Ib, Ic] (A).
    ///
    /// # Returns
    /// Array of compensated phase voltages.
    pub fn compensate(&self, v_ref: [S; 3], i_phase: [S; 3]) -> [S; 3] {
        [
            self.compensate_phase(v_ref[0], i_phase[0]),
            self.compensate_phase(v_ref[1], i_phase[1]),
            self.compensate_phase(v_ref[2], i_phase[2]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dead_time_voltage_magnitude() {
        // td=500ns, Vdc=400V, fsw=10kHz → Vdt = 400*500e-9*10000 = 2.0V
        let comp = DeadTimeCompensator::<f32>::new(500e-9, 400.0, 0.1, 10_000.0);
        let vdt = comp.dead_time_voltage();
        assert!((vdt - 2.0_f32).abs() < 1e-3, "Vdt={vdt}");
    }

    #[test]
    fn test_compensate_positive_current() {
        // Positive current: compensation adds +Vdt
        let comp = DeadTimeCompensator::<f32>::new(500e-9, 400.0, 0.1, 10_000.0);
        let vdt = comp.dead_time_voltage(); // 2.0V
        let v_in = 100.0_f32;
        let i = 5.0_f32; // well above threshold
        let v_out = comp.compensate_phase(v_in, i);
        assert!((v_out - (v_in + vdt)).abs() < 1e-4, "v_out={v_out}");
    }

    #[test]
    fn test_compensate_negative_current() {
        // Negative current: compensation subtracts Vdt
        let comp = DeadTimeCompensator::<f32>::new(500e-9, 400.0, 0.1, 10_000.0);
        let vdt = comp.dead_time_voltage(); // 2.0V
        let v_in = 100.0_f32;
        let i = -5.0_f32;
        let v_out = comp.compensate_phase(v_in, i);
        assert!((v_out - (v_in - vdt)).abs() < 1e-4, "v_out={v_out}");
    }

    #[test]
    fn test_compensate_three_phase() {
        let comp = DeadTimeCompensator::<f32>::new(500e-9, 400.0, 0.1, 10_000.0);
        let v_ref = [100.0_f32, -50.0, -50.0];
        let i_phase = [5.0_f32, -2.5, -2.5];
        let compensated = comp.compensate(v_ref, i_phase);
        let vdt = comp.dead_time_voltage();
        assert!((compensated[0] - (100.0 + vdt)).abs() < 1e-4);
        assert!((compensated[1] - (-50.0 - vdt)).abs() < 1e-4);
        assert!((compensated[2] - (-50.0 - vdt)).abs() < 1e-4);
    }

    #[test]
    fn test_compensate_near_zero_current() {
        // Current = threshold/2 → polarity = 0.5, correction = 0.5*Vdt
        let comp = DeadTimeCompensator::<f32>::new(500e-9, 400.0, 0.1, 10_000.0);
        let vdt = comp.dead_time_voltage(); // 2.0V
        let v_in = 0.0_f32;
        let i = 0.05_f32; // half of threshold=0.1
        let v_out = comp.compensate_phase(v_in, i);
        let expected = vdt * 0.5;
        assert!(
            (v_out - expected).abs() < 1e-4,
            "v_out={v_out}, expected={expected}"
        );
    }
}
