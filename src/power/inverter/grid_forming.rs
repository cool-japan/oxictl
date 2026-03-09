//! Grid-forming inverter: Virtual Synchronous Machine (VSM) mode.
//!
//! A grid-forming inverter establishes voltage and frequency references for
//! islanded or weak-grid operation.  Two control strategies are provided:
//!
//! 1. **Droop control** – static frequency and voltage droops from P/Q error.
//! 2. **VSM (Virtual Synchronous Machine)** – swing-equation emulation giving
//!    inertia-like dynamic response.
use crate::core::scalar::ControlScalar;

/// Grid-forming inverter controller using VSM/droop control.
#[derive(Debug, Clone, Copy)]
pub struct GridFormingInverter<S: ControlScalar> {
    /// Nominal angular frequency (rad/s), e.g. 2π·50.
    pub omega_n: S,
    /// Nominal AC voltage magnitude (V).
    pub v_n: S,
    /// Virtual inertia constant H (seconds).  Larger → slower frequency response.
    pub inertia: S,
    /// Frequency droop coefficient Dp (rad/s per W).
    pub d_p: S,
    /// Voltage droop coefficient Dq (V per VAR).
    pub d_q: S,
    /// Current angular frequency (rad/s).
    pub omega: S,
    /// Current voltage magnitude reference (V).
    pub v_ref: S,
    /// Measured active power output (W).
    pub p_out: S,
    /// Measured reactive power output (VAR).
    pub q_out: S,
}

impl<S: ControlScalar> GridFormingInverter<S> {
    /// Create a new grid-forming inverter controller.
    pub fn new(omega_n: S, v_n: S, inertia: S, d_p: S, d_q: S) -> Self {
        Self {
            omega_n,
            v_n,
            inertia,
            d_p,
            d_q,
            omega: omega_n,
            v_ref: v_n,
            p_out: S::ZERO,
            q_out: S::ZERO,
        }
    }

    /// Apply static droop control from measured active and reactive power.
    ///
    /// Updates `omega` and `v_ref` according to:
    /// ```text
    ///   ω  = ω_n − Dp·(P − P_ref)
    ///   Vref = Vn − Dq·(Q − Q_ref)
    /// ```
    ///
    /// Returns `(omega, v_ref)`.
    pub fn droop_control(&mut self, p: S, q: S, p_ref: S, q_ref: S) -> (S, S) {
        self.p_out = p;
        self.q_out = q;
        self.omega = self.omega_n - self.d_p * (p - p_ref);
        self.v_ref = self.v_n - self.d_q * (q - q_ref);
        (self.omega, self.v_ref)
    }

    /// Integrate the VSM swing equation for one time step `dt`.
    ///
    /// Swing equation (per-unit form):
    /// ```text
    ///   dω/dt = (P_ref − P_meas − Dp·Δω) / (2·H)
    /// ```
    /// where `Δω = ω − ω_n`.
    pub fn vsm_step(&mut self, p_ref: S, p_meas: S, dt: S) {
        self.p_out = p_meas;
        let delta_omega = self.omega - self.omega_n;
        let dw = (p_ref - p_meas - self.d_p * delta_omega) / (S::TWO * self.inertia);
        self.omega += dw * dt;
    }

    /// Compute voltage reference from reactive power droop.
    ///
    /// `V_ref = V_n − Dq·(Q_meas − Q_ref)`
    pub fn voltage_droop(&self, q_meas: S, q_ref: S) -> S {
        self.v_n - self.d_q * (q_meas - q_ref)
    }

    /// Current output frequency in Hz.
    pub fn frequency_hz(&self) -> S {
        self.omega / (S::TWO * S::PI)
    }

    /// Whether the inverter frequency is within `tolerance` rad/s of the grid.
    pub fn is_synchronized(&self, omega_grid: S, tolerance: S) -> bool {
        (self.omega - omega_grid).abs() <= tolerance
    }

    /// Reset to nominal operating point.
    pub fn reset(&mut self) {
        self.omega = self.omega_n;
        self.v_ref = self.v_n;
        self.p_out = S::ZERO;
        self.q_out = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn droop_reduces_frequency_on_overload() {
        let omega_n = 2.0 * PI * 50.0_f64;
        let mut inv = GridFormingInverter::new(omega_n, 230.0, 5.0, 1e-3, 1e-4);
        // Generating 1000 W above reference → frequency should drop.
        let (omega, _) = inv.droop_control(2000.0, 0.0, 1000.0, 0.0);
        assert!(omega < omega_n, "omega should drop: {}", omega);
    }

    #[test]
    fn vsm_step_integrates_swing_equation() {
        let omega_n = 2.0 * PI * 50.0_f64;
        let mut inv = GridFormingInverter::new(omega_n, 230.0, 5.0, 1.0, 0.1);
        // Power surplus → frequency should increase.
        inv.vsm_step(1500.0, 1000.0, 0.01);
        assert!(inv.omega > omega_n, "omega should increase: {}", inv.omega);
    }

    #[test]
    fn voltage_droop_drops_on_inductive_load() {
        let omega_n = 2.0 * PI * 50.0_f64;
        let inv = GridFormingInverter::new(omega_n, 230.0, 5.0, 1e-3, 1e-3);
        // Absorbing 500 VAR above reference → voltage reference drops.
        let v = inv.voltage_droop(500.0, 0.0);
        assert!(v < 230.0, "v_ref should drop: {}", v);
    }

    #[test]
    fn synchronization_check() {
        let omega_n = 2.0 * PI * 50.0_f64;
        let inv = GridFormingInverter::new(omega_n, 230.0, 5.0, 1e-3, 1e-3);
        assert!(inv.is_synchronized(omega_n, 0.1));
        assert!(!inv.is_synchronized(omega_n + 1.0, 0.1));
    }
}
