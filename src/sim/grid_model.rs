//! Simplified AC grid model with frequency and voltage droop response.
//!
//! Models a single-phase equivalent grid that responds to active/reactive
//! power injections with frequency and voltage droop.
#![cfg(feature = "std")]

use crate::core::scalar::ControlScalar;

/// AC grid model parameters.
#[derive(Debug, Clone, Copy)]
pub struct GridParams<S: ControlScalar> {
    /// Nominal voltage (V rms).
    pub v_nom: S,
    /// Nominal frequency (Hz).
    pub f_nom: S,
    /// Grid impedance magnitude (Ω).
    pub z_grid: S,
    /// Frequency droop coefficient (Hz/W).
    pub kf: S,
    /// Voltage droop coefficient (V/VAr).
    pub kv: S,
}

impl<S: ControlScalar> GridParams<S> {
    pub fn new(v_nom: S, f_nom: S, z_grid: S, kf: S, kv: S) -> Self {
        Self {
            v_nom,
            f_nom,
            z_grid,
            kf,
            kv,
        }
    }
}

/// Simplified single-phase equivalent AC grid model.
///
/// Frequency and voltage are updated via first-order low-pass droop dynamics:
///   f_grid += dt/tau * (f_droop - f_grid)
///   V_pcc  += dt/tau * (V_droop - V_pcc)
/// where tau is an internal settling time constant.
pub struct GridModel<S: ControlScalar> {
    pub params: GridParams<S>,
    /// Point of common coupling voltage (V rms).
    pub v_pcc: S,
    /// Current grid frequency (Hz).
    pub f_grid: S,
    /// Active power injection (W).
    pub p_inj: S,
    /// Reactive power injection (VAr).
    pub q_inj: S,
}

impl<S: ControlScalar> GridModel<S> {
    /// Voltage stability band: ±10 % of nominal.
    const V_BAND: f64 = 0.10;
    /// Frequency stability band: ±0.5 Hz from nominal.
    const F_BAND: f64 = 0.5;
    /// First-order settling time constant (s).
    const TAU: f64 = 0.1;

    pub fn new(params: GridParams<S>) -> Self {
        let v_nom = params.v_nom;
        let f_nom = params.f_nom;
        Self {
            params,
            v_pcc: v_nom,
            f_grid: f_nom,
            p_inj: S::ZERO,
            q_inj: S::ZERO,
        }
    }

    /// Voltage droop: V = V_nom - kv * Q.
    fn voltage_droop(&self) -> S {
        self.params.v_nom - self.params.kv * self.q_inj
    }

    /// Frequency droop: f = f_nom - kf * P.
    fn frequency_droop(&self) -> S {
        self.params.f_nom - self.params.kf * self.p_inj
    }

    /// Update grid state with new power injection.
    ///
    /// Returns (V_pcc, f_grid) after settling.
    pub fn step(&mut self, p_inj: S, q_inj: S, dt: S) -> (S, S) {
        self.p_inj = p_inj;
        self.q_inj = q_inj;

        let tau = S::from_f64(Self::TAU);
        let alpha = (dt / tau).clamp_val(S::ZERO, S::ONE);

        // First-order low-pass towards droop set-point
        let v_target = self.voltage_droop();
        let f_target = self.frequency_droop();

        self.v_pcc += alpha * (v_target - self.v_pcc);
        self.f_grid += alpha * (f_target - self.f_grid);

        (self.v_pcc, self.f_grid)
    }

    /// Returns true if voltage and frequency are within stability limits.
    pub fn is_stable(&self) -> bool {
        let v_nom = self.params.v_nom.to_f64();
        let f_nom = self.params.f_nom.to_f64();
        let v_err = (self.v_pcc.to_f64() - v_nom).abs() / v_nom;
        let f_err = (self.f_grid.to_f64() - f_nom).abs();
        v_err <= Self::V_BAND && f_err <= Self::F_BAND
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid() -> GridModel<f64> {
        let params = GridParams::new(
            230.0, // V_nom
            50.0,  // f_nom
            0.1,   // z_grid
            1e-6,  // kf (Hz/W)
            1e-4,  // kv (V/VAr)
        );
        GridModel::new(params)
    }

    #[test]
    fn test_grid_stable_at_rest() {
        let grid = make_grid();
        assert!(grid.is_stable());
        assert!((grid.v_pcc - 230.0).abs() < 1e-9);
        assert!((grid.f_grid - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_frequency_droops_with_active_power() {
        let mut grid = make_grid();
        // Inject 100 kW
        for _ in 0..200 {
            grid.step(100_000.0, 0.0, 0.01);
        }
        // Frequency should be below nominal
        assert!(grid.f_grid < 50.0, "f={}", grid.f_grid);
        let expected_f = 50.0 - 1e-6 * 100_000.0;
        assert!((grid.f_grid - expected_f).abs() < 0.01, "f={}", grid.f_grid);
    }

    #[test]
    fn test_voltage_droops_with_reactive_power() {
        let mut grid = make_grid();
        // Inject 50 kVAr
        for _ in 0..200 {
            grid.step(0.0, 50_000.0, 0.01);
        }
        assert!(grid.v_pcc < 230.0, "v={}", grid.v_pcc);
        let expected_v = 230.0 - 1e-4 * 50_000.0;
        assert!((grid.v_pcc - expected_v).abs() < 0.5, "v={}", grid.v_pcc);
    }

    #[test]
    fn test_stability_flag_large_injection() {
        let mut grid = make_grid();
        // 50 MW active power → huge frequency drop
        for _ in 0..500 {
            grid.step(50_000_000.0, 0.0, 0.01);
        }
        assert!(!grid.is_stable(), "f={}", grid.f_grid);
    }
}
