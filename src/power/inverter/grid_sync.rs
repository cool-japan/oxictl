use crate::core::scalar::ControlScalar;

/// Grid-tied Voltage Source Inverter (VSI) current controller.
///
/// Implements synchronous reference frame (dq) current control for
/// single or three-phase grid-connected inverters.
///
/// Control structure:
///   1. PLL provides grid angle θ (external, e.g. from `power::pll::Pll`)
///   2. Park transform: i_abc → i_dq (external, e.g. from `motor::transform::park`)
///   3. PI current control in dq frame with cross-coupling decoupling
///   4. Inverse Park: v_dq_ref → v_abc_ref (external)
///
/// Sign convention:
///   - i_d aligned with grid voltage V_d: controls active power P = (3/2)*V_d*i_d
///   - i_q: controls reactive power Q = -(3/2)*V_d*i_q
///
/// Cross-coupling compensation:
///   V_d_ff = -ω·L·i_q + V_d_grid
///   V_q_ff =  ω·L·i_d + V_q_grid (usually small)
#[derive(Debug, Clone, Copy)]
pub struct GridCurrentController<S: ControlScalar> {
    /// d-axis proportional gain.
    pub kp_d: S,
    /// d-axis integral gain.
    pub ki_d: S,
    /// q-axis proportional gain.
    pub kp_q: S,
    /// q-axis integral gain.
    pub ki_q: S,
    /// Grid-side inductance (H) for cross-coupling compensation.
    pub l_filter: S,
    /// Output voltage limits (V).
    pub v_max: S,
    /// d-axis integrator.
    int_d: S,
    /// q-axis integrator.
    int_q: S,
}

impl<S: ControlScalar> GridCurrentController<S> {
    pub fn new(kp: S, ki: S, l_filter: S, v_max: S) -> Self {
        Self {
            kp_d: kp,
            ki_d: ki,
            kp_q: kp,
            ki_q: ki,
            l_filter,
            v_max,
            int_d: S::ZERO,
            int_q: S::ZERO,
        }
    }

    /// Compute dq voltage references for grid current control.
    ///
    /// - `id_ref`, `iq_ref`: current references in dq frame (A)
    /// - `id`, `iq`: measured currents (A)
    /// - `v_d`, `v_q`: measured grid voltages in dq frame (V)
    /// - `omega`: grid angular frequency (rad/s)
    /// - `dt`: time step (s)
    ///
    /// Returns (v_d_ref, v_q_ref) — voltage references for the inverter.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        id_ref: S,
        iq_ref: S,
        id: S,
        iq: S,
        v_d: S,
        v_q: S,
        omega: S,
        dt: S,
    ) -> (S, S) {
        let e_d = id_ref - id;
        let e_q = iq_ref - iq;

        self.int_d += e_d * dt;
        self.int_q += e_q * dt;

        // PI output
        let pi_d = self.kp_d * e_d + self.ki_d * self.int_d;
        let pi_q = self.kp_q * e_q + self.ki_q * self.int_q;

        // Cross-coupling decoupling + voltage feedforward
        let v_d_ref = pi_d - omega * self.l_filter * iq + v_d;
        let v_q_ref = pi_q + omega * self.l_filter * id + v_q;

        // Limit output magnitude
        let v_mag = (v_d_ref * v_d_ref + v_q_ref * v_q_ref).sqrt();
        if v_mag > self.v_max {
            let scale = self.v_max / v_mag;
            // Anti-windup: undo last integral step
            self.int_d -= e_d * dt;
            self.int_q -= e_q * dt;
            return (v_d_ref * scale, v_q_ref * scale);
        }

        (v_d_ref, v_q_ref)
    }

    /// Active power reference → d-axis current reference.
    ///
    /// P_ref = (3/2) * V_d * i_d  →  i_d_ref = P_ref / ((3/2)*V_d)
    pub fn active_power_to_id(&self, p_ref: S, v_d: S) -> S {
        let three_halves = S::from_f64(1.5);
        if v_d.abs() > S::from_f64(1.0) {
            p_ref / (three_halves * v_d)
        } else {
            S::ZERO
        }
    }

    /// Reactive power reference → q-axis current reference.
    ///
    /// Q_ref = -(3/2) * V_d * i_q  →  i_q_ref = -Q_ref / ((3/2)*V_d)
    pub fn reactive_power_to_iq(&self, q_ref: S, v_d: S) -> S {
        let three_halves = S::from_f64(1.5);
        if v_d.abs() > S::from_f64(1.0) {
            -q_ref / (three_halves * v_d)
        } else {
            S::ZERO
        }
    }

    pub fn reset(&mut self) {
        self.int_d = S::ZERO;
        self.int_q = S::ZERO;
    }
}

/// Anti-islanding monitor (passive detection via frequency deviation).
///
/// Detects loss of grid connection by monitoring estimated frequency drift.
/// When the grid is disconnected, the local frequency drifts away from nominal.
#[derive(Debug, Clone, Copy)]
pub struct IslandingDetector<S: ControlScalar> {
    /// Nominal grid frequency (rad/s).
    pub omega_nominal: S,
    /// Maximum allowed frequency deviation before trip (rad/s).
    pub omega_tolerance: S,
    /// Voltage under-threshold factor (fraction of nominal, e.g. 0.85).
    pub v_under_threshold: S,
    /// Voltage over-threshold factor (e.g. 1.10).
    pub v_over_threshold: S,
    /// Trip flag.
    islanded: bool,
}

impl<S: ControlScalar> IslandingDetector<S> {
    pub fn new(omega_nominal: S, omega_tolerance: S) -> Self {
        Self {
            omega_nominal,
            omega_tolerance,
            v_under_threshold: S::from_f64(0.85),
            v_over_threshold: S::from_f64(1.10),
            islanded: false,
        }
    }

    /// Check for islanding.
    ///
    /// - `omega_est`: PLL-estimated frequency
    /// - `v_mag`: measured grid voltage magnitude
    /// - `v_nominal`: expected nominal voltage magnitude
    pub fn check(&mut self, omega_est: S, v_mag: S, v_nominal: S) -> bool {
        let freq_fault = (omega_est - self.omega_nominal).abs() > self.omega_tolerance;
        let v_ratio = if v_nominal > S::ZERO {
            v_mag / v_nominal
        } else {
            S::ZERO
        };
        let volt_fault = v_ratio < self.v_under_threshold || v_ratio > self.v_over_threshold;

        if freq_fault || volt_fault {
            self.islanded = true;
        }

        self.islanded
    }

    pub fn is_islanded(&self) -> bool {
        self.islanded
    }

    pub fn reset(&mut self) {
        self.islanded = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_controller_drives_error_to_zero() {
        let mut ctrl = GridCurrentController::new(10.0_f64, 200.0, 5e-3, 400.0);
        let omega = 2.0 * core::f64::consts::PI * 50.0;
        let v_d = 230.0_f64 * 1.414; // grid voltage (peak)
        let dt = 1e-4_f64;

        let id_ref = 10.0_f64; // 10A active current
        let iq_ref = 0.0_f64; // unity power factor

        let mut id = 0.0_f64;
        let mut iq = 0.0_f64;

        for _ in 0..5000 {
            let (vd, vq) = ctrl.update(id_ref, iq_ref, id, iq, v_d, 0.0, omega, dt);
            // Simple RL load: L*di/dt = V - R*i (R=1Ω)
            id += (vd - id) * dt / 5e-3;
            iq += (vq - iq) * dt / 5e-3;
        }

        assert!((id - id_ref).abs() < 1.0, "id={:.2}, ref={:.2}", id, id_ref);
    }

    #[test]
    fn power_to_current_conversion() {
        let ctrl = GridCurrentController::new(10.0_f64, 200.0, 5e-3, 400.0);
        let v_d = 325.0_f64; // ~230V peak
        let p_ref = 1000.0_f64; // 1kW

        let id = ctrl.active_power_to_id(p_ref, v_d);
        let p_actual = 1.5 * v_d * id;
        assert!(
            (p_actual - p_ref).abs() < 0.1,
            "P={:.1}W (ref={:.1}W)",
            p_actual,
            p_ref
        );
    }

    #[test]
    fn islanding_detects_frequency_drift() {
        let omega_nom = 2.0 * core::f64::consts::PI * 50.0_f64;
        let mut det = IslandingDetector::new(omega_nom, 2.0);
        // Normal operation
        assert!(!det.check(omega_nom + 0.5, 230.0, 230.0));
        // Frequency drift > 2 rad/s
        assert!(det.check(omega_nom + 3.0, 230.0, 230.0));
        assert!(det.is_islanded());
    }

    #[test]
    fn islanding_detects_undervoltage() {
        let omega_nom = 2.0 * core::f64::consts::PI * 50.0_f64;
        let mut det = IslandingDetector::new(omega_nom, 5.0);
        // Voltage drops to 80% (< 85% threshold)
        assert!(det.check(omega_nom, 184.0, 230.0));
        assert!(det.is_islanded());
    }

    #[test]
    fn no_islanding_in_normal_operation() {
        let omega_nom = 2.0 * core::f64::consts::PI * 50.0_f64;
        let mut det = IslandingDetector::new(omega_nom, 5.0);
        assert!(!det.check(omega_nom + 0.1, 228.0, 230.0));
        assert!(!det.is_islanded());
    }
}
