use crate::core::scalar::ControlScalar;

/// H-Bridge driver for brushed DC motors.
///
/// Maps a signed duty cycle command [-1, +1] to high/low side switch duties
/// for each half-bridge, with configurable dead time insertion.
///
/// H-Bridge legs:
///   - High side A (HA) + Low side B (LB): positive current
///   - High side B (HB) + Low side A (LA): negative current
///
/// Sign convention: positive command → HA on, LB on (motor forward)
///                  negative command → HB on, LA on (motor reverse)
#[derive(Debug, Clone, Copy)]
pub struct HBridge<S: ControlScalar> {
    /// Dead time (fraction of period, 0..0.1 typical).
    pub dead_time: S,
    /// Minimum duty cycle magnitude (dead-band to stop at near-zero commands).
    pub dead_band: S,
    /// Current command (-1..+1).
    command: S,
}

/// H-Bridge PWM output.
#[derive(Debug, Clone, Copy)]
pub struct HBridgeOutput<S: ControlScalar> {
    /// High-side A duty cycle [0, 1].
    pub ha: S,
    /// Low-side A duty cycle [0, 1].
    pub la: S,
    /// High-side B duty cycle [0, 1].
    pub hb: S,
    /// Low-side B duty cycle [0, 1].
    pub lb: S,
    /// Effective duty cycle applied to motor [-1, +1].
    pub effective: S,
}

impl<S: ControlScalar> HBridge<S> {
    /// Create H-Bridge driver.
    ///
    /// - `dead_time`: dead time as fraction of period (e.g. 0.02 for 2%)
    /// - `dead_band`: minimum |command| below which output is clamped to zero
    pub fn new(dead_time: S, dead_band: S) -> Self {
        Self {
            dead_time,
            dead_band,
            command: S::ZERO,
        }
    }

    /// Compute PWM output for the given signed command [-1, +1].
    ///
    /// Returns all four switch duties and the effective applied duty.
    pub fn compute(&mut self, command: S) -> HBridgeOutput<S> {
        let cmd = command.clamp_val(-S::ONE, S::ONE);

        // Apply dead-band
        let effective = if cmd.abs() < self.dead_band {
            S::ZERO
        } else {
            cmd
        };
        self.command = effective;

        let duty = effective.abs();
        let duty_with_dt = (duty - self.dead_time).clamp_val(S::ZERO, S::ONE);
        let complement = (S::ONE - duty - self.dead_time).clamp_val(S::ZERO, S::ONE);

        if effective >= S::ZERO {
            // Forward: HA active, LB active
            HBridgeOutput {
                ha: duty_with_dt,
                la: S::ZERO,
                hb: S::ZERO,
                lb: complement,
                effective,
            }
        } else {
            // Reverse: HB active, LA active
            HBridgeOutput {
                ha: S::ZERO,
                la: complement,
                hb: duty_with_dt,
                lb: S::ZERO,
                effective,
            }
        }
    }

    /// Brake (both low sides active — regenerative braking).
    pub fn brake(&self) -> HBridgeOutput<S> {
        HBridgeOutput {
            ha: S::ZERO,
            la: S::ONE,
            hb: S::ZERO,
            lb: S::ONE,
            effective: S::ZERO,
        }
    }

    /// Coast (all switches off — motor freewheels).
    pub fn coast(&self) -> HBridgeOutput<S> {
        HBridgeOutput {
            ha: S::ZERO,
            la: S::ZERO,
            hb: S::ZERO,
            lb: S::ZERO,
            effective: S::ZERO,
        }
    }

    /// Last applied command.
    pub fn command(&self) -> S {
        self.command
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_command_activates_correct_switches() {
        let mut hb = HBridge::new(0.02_f64, 0.02);
        let out = hb.compute(0.5);
        assert!(out.ha > 0.0, "HA should be active for +cmd");
        assert_eq!(out.hb, 0.0, "HB should be off for +cmd");
        assert_eq!(out.la, 0.0, "LA should be off for +cmd");
        assert!(out.lb > 0.0, "LB should be active for +cmd");
    }

    #[test]
    fn reverse_command_activates_correct_switches() {
        let mut hb = HBridge::new(0.02_f64, 0.02);
        let out = hb.compute(-0.5);
        assert_eq!(out.ha, 0.0, "HA should be off for -cmd");
        assert!(out.hb > 0.0, "HB should be active for -cmd");
        assert!(out.la > 0.0, "LA should be active for -cmd");
        assert_eq!(out.lb, 0.0, "LB should be off for -cmd");
    }

    #[test]
    fn dead_band_clamps_small_commands() {
        let mut hb = HBridge::new(0.0_f64, 0.05);
        let out = hb.compute(0.03); // below dead_band = 0.05
        assert_eq!(out.effective, 0.0, "Should be clamped by dead_band");
    }

    #[test]
    fn duty_clamps_to_one() {
        let mut hb = HBridge::new(0.02_f64, 0.02);
        let out = hb.compute(1.0);
        assert!(out.ha <= 1.0, "HA duty should not exceed 1");
        assert!(out.ha >= 0.0, "HA duty should be non-negative");
    }

    #[test]
    fn brake_mode_grounds_both_sides() {
        let hb = HBridge::new(0.02_f64, 0.02);
        let out = hb.brake();
        assert_eq!(out.ha, 0.0);
        assert_eq!(out.la, 1.0);
        assert_eq!(out.hb, 0.0);
        assert_eq!(out.lb, 1.0);
    }
}
