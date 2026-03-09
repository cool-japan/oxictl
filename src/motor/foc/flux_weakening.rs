use crate::core::scalar::ControlScalar;

/// Flux Weakening controller for PMSM drives.
///
/// Above base speed, the motor back-EMF exceeds the supply voltage.
/// Flux weakening reduces the d-axis current (Id) to negative values,
/// which reduces the effective flux and allows operation at higher speeds.
///
/// Strategy:
///   - Below base speed: Id_ref = 0 (maximum torque per amp)
///   - Above base speed: Id_ref = -(Vs_max/ωe - ψf) / Ld  (linearized)
///     where Vs_max is the maximum voltage circle radius
///
/// The controller limits the voltage vector magnitude to Vs_max and
/// computes the required Id to stay on the voltage limit circle.
#[derive(Debug, Clone, Copy)]
pub struct FluxWeakening<S: ControlScalar> {
    /// Stator inductance d-axis (H).
    pub l_d: S,
    /// Permanent magnet flux linkage (Wb).
    pub psi_f: S,
    /// Maximum voltage vector magnitude (V).
    pub v_s_max: S,
    /// Base electrical speed above which flux weakening activates (rad/s).
    pub omega_base: S,
    /// Minimum Id reference (most negative, limits flux weakening depth).
    pub id_min: S,
    /// Current Id reference output.
    id_ref: S,
}

impl<S: ControlScalar> FluxWeakening<S> {
    /// Create a flux weakening controller.
    ///
    /// - `l_d`: d-axis inductance (H)
    /// - `psi_f`: permanent magnet flux linkage (Wb)
    /// - `v_s_max`: available voltage amplitude (Vdc / sqrt(3) for SVM)
    /// - `omega_base`: base electrical speed where flux weakening starts
    /// - `id_min`: minimum Id (e.g. -I_rated to limit demagnetization risk)
    pub fn new(l_d: S, psi_f: S, v_s_max: S, omega_base: S, id_min: S) -> Self {
        Self {
            l_d,
            psi_f,
            v_s_max,
            omega_base,
            id_min,
            id_ref: S::ZERO,
        }
    }

    /// Update flux weakening command.
    ///
    /// - `omega_e`: electrical angular speed (rad/s), sign indicates direction
    ///
    /// Returns the d-axis current reference (Id_ref ≤ 0 in flux weakening).
    pub fn update(&mut self, omega_e: S) -> S {
        let omega_abs = omega_e.abs();

        if omega_abs <= self.omega_base || omega_abs < S::EPSILON {
            // Below base speed: no flux weakening needed
            self.id_ref = S::ZERO;
            return S::ZERO;
        }

        // Voltage limit: |Vs| = ωe * sqrt((Ld*Id + ψf)² + (Lq*Iq)²) ≤ Vs_max
        // Simplified (ignoring Lq*Iq term, conservative): Ld*|Id| + ψf = Vs_max/ωe
        // Id_ref = (Vs_max/ωe - ψf) / Ld  (negative since Vs_max/ωe < ψf above base speed)
        let flux_limit = self.v_s_max / omega_abs;
        let id = (flux_limit - self.psi_f) / self.l_d;

        // Id must be negative for flux weakening, and limited
        self.id_ref = id.clamp_val(self.id_min, S::ZERO);
        self.id_ref
    }

    pub fn id_ref(&self) -> S {
        self.id_ref
    }

    pub fn reset(&mut self) {
        self.id_ref = S::ZERO;
    }

    /// Returns true if flux weakening is currently active.
    pub fn is_active(&self) -> bool {
        self.id_ref < S::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_fw() -> FluxWeakening<f64> {
        // Consistent parameters: ω_base = Vs_max/ψf → 200 = 10/0.05
        // So Vs_max=10V, ψf=0.05Wb → base speed = 200 rad/s
        FluxWeakening::new(0.001, 0.05, 10.0, 200.0, -20.0)
    }

    #[test]
    fn below_base_speed_no_fw() {
        let mut fw = build_fw();
        let id = fw.update(100.0); // below 200 rad/s
        assert_eq!(id, 0.0);
        assert!(!fw.is_active());
    }

    #[test]
    fn above_base_speed_activates() {
        let mut fw = build_fw();
        let id = fw.update(400.0);
        assert!(id < 0.0, "id={:.4}", id);
        assert!(fw.is_active());
    }

    #[test]
    fn clamps_to_id_min() {
        let mut fw = build_fw();
        let id = fw.update(10000.0); // extreme speed
        assert!(id >= -20.0);
    }

    #[test]
    fn negative_speed_works() {
        let mut fw = build_fw();
        let id_pos = fw.update(400.0);
        let id_neg = fw.update(-400.0);
        assert!((id_pos - id_neg).abs() < 1e-10, "Should be symmetric");
    }

    #[test]
    fn reset_clears_state() {
        let mut fw = build_fw();
        fw.update(500.0);
        fw.reset();
        assert_eq!(fw.id_ref(), 0.0);
    }
}
