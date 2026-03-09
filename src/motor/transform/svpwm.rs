use crate::core::scalar::ControlScalar;
use crate::motor::transform::clarke::AlphaBeta;

/// Space Vector PWM duty cycles for three-phase inverter.
#[derive(Debug, Clone, Copy)]
pub struct SvpwmDuty<S: ControlScalar> {
    pub ta: S,
    pub tb: S,
    pub tc: S,
}

/// Space Vector PWM (SVPWM) modulator.
///
/// Converts αβ voltage reference to three-phase duty cycles.
/// Input: Vα, Vβ normalized to DC bus voltage (range -1..1).
/// Output: duty cycles 0..1 for each phase.
///
/// Uses the standard sector-based algorithm with min-max injection
/// (equivalent to saddle-wave / third-harmonic injection).
pub fn svpwm<S: ControlScalar>(ab: &AlphaBeta<S>, vdc: S) -> SvpwmDuty<S> {
    if vdc <= S::ZERO {
        return SvpwmDuty {
            ta: S::HALF,
            tb: S::HALF,
            tc: S::HALF,
        };
    }

    let sqrt3 = S::from_f64(1.7320508075688772);
    let inv_vdc = S::ONE / vdc;

    // Phase voltages (not yet normalized)
    let sqrt3_over2 = sqrt3 * S::HALF;
    let va = ab.alpha;
    let vb = -ab.alpha * S::HALF + sqrt3_over2 * ab.beta;
    let vc = -ab.alpha * S::HALF - sqrt3_over2 * ab.beta;

    // Min-max injection (saddle wave) to maximize linear range
    let v_min = va.min(vb).min(vc);
    let v_max = va.max(vb).max(vc);
    let v_offset = -(v_min + v_max) * S::HALF;

    // Duty cycles: centered around 0.5
    let da = S::HALF + (va + v_offset) * inv_vdc * S::HALF;
    let db = S::HALF + (vb + v_offset) * inv_vdc * S::HALF;
    let dc = S::HALF + (vc + v_offset) * inv_vdc * S::HALF;

    // Clamp to [0,1]
    SvpwmDuty {
        ta: da.clamp_val(S::ZERO, S::ONE),
        tb: db.clamp_val(S::ZERO, S::ONE),
        tc: dc.clamp_val(S::ZERO, S::ONE),
    }
}

/// Simple sinusoidal PWM (SPWM) for comparison.
pub fn spwm<S: ControlScalar>(ab: &AlphaBeta<S>, vdc: S) -> SvpwmDuty<S> {
    if vdc <= S::ZERO {
        return SvpwmDuty {
            ta: S::HALF,
            tb: S::HALF,
            tc: S::HALF,
        };
    }
    let sqrt3_over2 = S::from_f64(1.7320508075688772 / 2.0);
    let inv_vdc = S::ONE / vdc;

    let va = ab.alpha;
    let vb = -ab.alpha * S::HALF + sqrt3_over2 * ab.beta;
    let vc = -ab.alpha * S::HALF - sqrt3_over2 * ab.beta;

    let da = (S::HALF + va * inv_vdc * S::HALF).clamp_val(S::ZERO, S::ONE);
    let db = (S::HALF + vb * inv_vdc * S::HALF).clamp_val(S::ZERO, S::ONE);
    let dc = (S::HALF + vc * inv_vdc * S::HALF).clamp_val(S::ZERO, S::ONE);

    SvpwmDuty {
        ta: da,
        tb: db,
        tc: dc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_voltage_gives_half_duty() {
        let ab = AlphaBeta {
            alpha: 0.0_f64,
            beta: 0.0,
            zero: 0.0,
        };
        let duty = svpwm(&ab, 24.0);
        assert!((duty.ta - 0.5).abs() < 1e-6, "ta={}", duty.ta);
        assert!((duty.tb - 0.5).abs() < 1e-6, "tb={}", duty.tb);
        assert!((duty.tc - 0.5).abs() < 1e-6, "tc={}", duty.tc);
    }

    #[test]
    fn duty_in_range() {
        for angle_deg in (0..360).step_by(15) {
            let theta = (angle_deg as f64) * core::f64::consts::PI / 180.0;
            let ab = AlphaBeta {
                alpha: theta.cos(),
                beta: theta.sin(),
                zero: 0.0,
            };
            let duty = svpwm(&ab, 24.0);
            assert!(
                duty.ta >= 0.0 && duty.ta <= 1.0,
                "ta={} at {}°",
                duty.ta,
                angle_deg
            );
            assert!(
                duty.tb >= 0.0 && duty.tb <= 1.0,
                "tb={} at {}°",
                duty.tb,
                angle_deg
            );
            assert!(
                duty.tc >= 0.0 && duty.tc <= 1.0,
                "tc={} at {}°",
                duty.tc,
                angle_deg
            );
        }
    }

    #[test]
    fn svpwm_symmetry() {
        // A-axis aligned: ta > tb ≈ tc
        let ab = AlphaBeta {
            alpha: 12.0_f64,
            beta: 0.0,
            zero: 0.0,
        };
        let duty = svpwm(&ab, 24.0);
        assert!(duty.ta > duty.tb, "ta={}, tb={}", duty.ta, duty.tb);
        assert!(
            (duty.tb - duty.tc).abs() < 1e-10,
            "tb={}, tc={}",
            duty.tb,
            duty.tc
        );
    }

    #[test]
    fn zero_vdc_gives_half_duty() {
        let ab = AlphaBeta {
            alpha: 1.0_f64,
            beta: 0.0,
            zero: 0.0,
        };
        let duty = svpwm(&ab, 0.0);
        assert_eq!(duty.ta, 0.5);
        assert_eq!(duty.tb, 0.5);
        assert_eq!(duty.tc, 0.5);
    }
}
