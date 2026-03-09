//! Synchronous Reluctance Motor (SynRM) control.
//!
//! A SynRM produces torque purely from saliency (Ld ≠ Lq); there is no
//! permanent magnet. The electromagnetic torque is:
//!
//!   τ = (3/2) · p · (Ld - Lq) · id · iq
//!
//! MTPA (Maximum Torque Per Ampere) for a SynRM operates at β = 45°,
//! i.e., id = iq = Is/√2, where Is is the current amplitude.
#![allow(clippy::excessive_precision)]

use crate::core::scalar::ControlScalar;

/// SynRM controller with MTPA optimization.
///
/// Provides current reference generation, torque estimation, and
/// flux linkage computation for field-oriented control of a SynRM.
#[derive(Debug, Clone, Copy)]
pub struct SynRmController<S: ControlScalar> {
    /// d-axis inductance (H).
    pub ld: S,
    /// q-axis inductance (H).
    pub lq: S,
    /// Number of pole pairs.
    pub poles: u8,
    /// Maximum current amplitude (A).
    pub i_max: S,
}

impl<S: ControlScalar> SynRmController<S> {
    /// Create a new SynRM controller.
    ///
    /// # Arguments
    /// * `ld` - d-axis inductance (H). Must be greater than `lq` for positive
    ///   reluctance torque.
    /// * `lq` - q-axis inductance (H).
    /// * `poles` - Number of pole pairs.
    /// * `i_max` - Maximum current amplitude (A).
    pub fn new(ld: S, lq: S, poles: u8, i_max: S) -> Self {
        Self {
            ld,
            lq,
            poles,
            i_max,
        }
    }

    /// MTPA current references for SynRM.
    ///
    /// For a SynRM (no PM flux), MTPA operates at β = 45°:
    ///   τ = (3/2)·p·(Ld-Lq)·id·iq = (3/2)·p·(Ld-Lq)·(Is/√2)²
    ///
    /// Given torque_ref, solve for Is:
    ///   Is² = 2·τ_ref / (3/2·p·(Ld-Lq))
    ///   Is  = √(Is²)
    ///   id = iq = Is/√2
    ///
    /// The sign of iq follows sign of torque_ref; id is always positive
    /// (field-producing direction).
    ///
    /// # Arguments
    /// * `torque_ref` - Desired torque (N·m).
    ///
    /// # Returns
    /// `(id_ref, iq_ref)` in amperes, clamped to `i_max`.
    pub fn mtpa_current_refs(&self, torque_ref: S) -> (S, S) {
        let delta_l = self.ld - self.lq;
        let p = S::from_f64(self.poles as f64);
        let three_halves = S::from_f64(1.5);
        let denom = three_halves * p * delta_l;

        if denom.abs() < S::EPSILON {
            return (S::ZERO, S::ZERO);
        }

        let torque_abs = torque_ref.abs();
        // Is² = 2·|τ| / (3/2·p·ΔL)  →  each axis = Is/√2
        // id² = iq² = Is²/2 = |τ| / (3/2·p·ΔL)
        let i_sq = torque_abs / denom;
        let i_axis = i_sq.max(S::ZERO).sqrt();

        // Clamp to i_max/√2 (so total Is = √(id²+iq²) ≤ i_max)
        let inv_sqrt2 = S::from_f64(1.0 / 2.0_f64.sqrt());
        let i_axis_max = self.i_max * inv_sqrt2;
        let i_axis_clamped = i_axis.min(i_axis_max);

        // id is always positive; iq follows torque sign
        let id = i_axis_clamped;
        let iq = i_axis_clamped * torque_ref.signum();

        (id, iq)
    }

    /// Compute electromagnetic (reluctance) torque.
    ///
    /// τ = (3/2) · p · (Ld - Lq) · id · iq
    pub fn reluctance_torque(&self, id: S, iq: S) -> S {
        let p = S::from_f64(self.poles as f64);
        let three_halves = S::from_f64(1.5);
        three_halves * p * (self.ld - self.lq) * id * iq
    }

    /// Compute dq flux linkages.
    ///
    /// ψd = Ld · id
    /// ψq = Lq · iq
    ///
    /// # Returns
    /// `(psi_d, psi_q)` in Wb (volt·seconds).
    pub fn flux_references(&self, id: S, iq: S) -> (S, S) {
        (self.ld * id, self.lq * iq)
    }

    /// MTPA angle for pure SynRM: always 45° (π/4 rad).
    ///
    /// The optimal current angle β = arctan(id/iq) = 45° for SynRM since
    /// id = iq at MTPA.
    pub fn mtpa_angle(&self) -> S {
        S::PI / S::from_f64(4.0)
    }

    /// Clamp current vector to maximum current amplitude.
    ///
    /// Scales (id, iq) uniformly if |Is| = √(id² + iq²) > i_max.
    pub fn clamp_current(&self, id: S, iq: S) -> (S, S) {
        let mag = (id * id + iq * iq).sqrt();
        if mag > self.i_max && mag > S::EPSILON {
            let scale = self.i_max / mag;
            (id * scale, iq * scale)
        } else {
            (id, iq)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reluctance_torque_positive() {
        // Ld=30mH, Lq=10mH, p=2, id=iq=5A
        // τ = 1.5*2*(0.03-0.01)*5*5 = 1.5*2*0.02*25 = 1.5 N·m
        let ctrl = SynRmController::<f32>::new(0.03, 0.01, 2, 20.0);
        let tau = ctrl.reluctance_torque(5.0, 5.0);
        assert!((tau - 1.5_f32).abs() < 1e-4, "tau={tau}");
    }

    #[test]
    fn test_reluctance_torque_zero_at_zero_current() {
        let ctrl = SynRmController::<f32>::new(0.03, 0.01, 2, 20.0);
        let tau = ctrl.reluctance_torque(0.0, 0.0);
        assert_eq!(tau, 0.0_f32);
    }

    #[test]
    fn test_mtpa_angle_is_45_degrees() {
        let ctrl = SynRmController::<f32>::new(0.03, 0.01, 2, 20.0);
        let angle = ctrl.mtpa_angle();
        let expected = core::f32::consts::PI / 4.0;
        assert!((angle - expected).abs() < 1e-6, "angle={angle}");
    }

    #[test]
    fn test_mtpa_current_refs_torque_consistency() {
        // Generate current refs for given torque, then verify actual torque ≈ ref
        let ctrl = SynRmController::<f32>::new(0.03, 0.01, 2, 20.0);
        let torque_ref = 1.0_f32;
        let (id, iq) = ctrl.mtpa_current_refs(torque_ref);
        let tau = ctrl.reluctance_torque(id, iq);
        assert!(
            (tau - torque_ref).abs() < 0.01,
            "tau={tau}, torque_ref={torque_ref}"
        );
    }

    #[test]
    fn test_clamp_current_within_limit() {
        let ctrl = SynRmController::<f32>::new(0.03, 0.01, 2, 10.0);
        // Overcurrent: id=iq=10 → |Is|=14.14 > 10
        let (id_c, iq_c) = ctrl.clamp_current(10.0, 10.0);
        let mag = (id_c * id_c + iq_c * iq_c).sqrt();
        assert!((mag - 10.0_f32).abs() < 1e-4, "clamped magnitude={mag}");
        // Angle should be preserved (45°)
        assert!(
            (id_c - iq_c).abs() < 1e-4,
            "angle not preserved: id={id_c}, iq={iq_c}"
        );
    }

    #[test]
    fn test_flux_references() {
        let ctrl = SynRmController::<f32>::new(0.03, 0.01, 2, 20.0);
        let (psi_d, psi_q) = ctrl.flux_references(5.0, 3.0);
        assert!((psi_d - 0.15_f32).abs() < 1e-5, "psi_d={psi_d}");
        assert!((psi_q - 0.03_f32).abs() < 1e-5, "psi_q={psi_q}");
    }
}
