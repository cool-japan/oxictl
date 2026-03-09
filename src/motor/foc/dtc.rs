//! Direct Torque Control (DTC) for induction/PMSM motors.
//!
//! Implements hysteresis-based torque and flux control with a
//! voltage vector switching table indexed by flux sector.
#![allow(clippy::excessive_precision)]

use crate::core::scalar::ControlScalar;

/// Hysteresis band for torque control.
#[derive(Debug, Clone, Copy)]
pub struct TorqueBand<S: ControlScalar> {
    pub upper: S,
    pub lower: S,
}

impl<S: ControlScalar> TorqueBand<S> {
    pub fn new(hysteresis: S) -> Self {
        Self {
            upper: hysteresis,
            lower: -hysteresis,
        }
    }

    /// Returns +1 if above upper, -1 if below lower, 0 otherwise.
    pub fn compare(&self, error: S) -> i8 {
        if error > self.upper {
            1
        } else if error < self.lower {
            -1
        } else {
            0
        }
    }
}

/// Hysteresis band for flux control.
#[derive(Debug, Clone, Copy)]
pub struct FluxBand<S: ControlScalar> {
    pub upper: S,
    pub lower: S,
}

impl<S: ControlScalar> FluxBand<S> {
    pub fn new(hysteresis: S) -> Self {
        Self {
            upper: hysteresis,
            lower: -hysteresis,
        }
    }

    /// Returns 1 if flux error is positive (increase flux), 0 otherwise.
    pub fn compare(&self, error: S) -> i8 {
        if error > self.upper {
            1
        } else if error < self.lower {
            0
        } else {
            1 // hold last state; default increase
        }
    }
}

/// Voltage vector switching table for DTC.
///
/// Indexed by [sector (0..5)][dtau+1 (0=dec,1=zero,2=inc)][dpsi (0=dec,1=inc)].
/// Voltage vectors: 0=V0, 1=V1..6=V6, 7=V7 (zero).
///
/// Classic DTC switching table (Takahashi):
/// - Sectors 1-6 (stored as 0-5)
/// - dtau: -1 → decrease torque, 0 → zero vector, +1 → increase torque
/// - dpsi:  0 → decrease flux,   1 → increase flux
const SWITCHING_TABLE: [[[u8; 2]; 3]; 6] = [
    // Sector 1 (0°..60°)
    [
        [5, 3], // dtau=-1: dpsi=0 → V5, dpsi=1 → V3
        [7, 0], // dtau= 0: zero vectors
        [2, 6], // dtau=+1: dpsi=0 → V2, dpsi=1 → V6
    ],
    // Sector 2 (60°..120°)
    [[6, 4], [0, 7], [3, 1]],
    // Sector 3 (120°..180°)
    [[1, 5], [7, 0], [4, 2]],
    // Sector 4 (180°..240°)
    [[2, 6], [0, 7], [5, 3]],
    // Sector 5 (240°..300°)
    [[3, 1], [7, 0], [6, 4]],
    // Sector 6 (300°..360°)
    [[4, 2], [0, 7], [1, 5]],
];

/// Select voltage vector from DTC switching table.
///
/// # Arguments
/// * `flux_sector` - Flux sector (1..=6), clamped internally.
/// * `dtau` - Torque comparator output: -1, 0, or +1.
/// * `dpsi` - Flux comparator output: 0 or 1.
///
/// # Returns
/// Voltage vector index 0-7 (0 and 7 are zero vectors, 1-6 are active).
pub fn select_voltage_vector(flux_sector: usize, dtau: i8, dpsi: i8) -> u8 {
    // Clamp sector to 1..=6 then map to 0..5
    let sector = flux_sector.saturating_sub(1).min(5);

    // dtau index: -1 → 0, 0 → 1, +1 → 2
    let tau_idx = match dtau {
        i8::MIN..=-1 => 0,
        0 => 1,
        _ => 2,
    };

    // dpsi index: 0 or 1
    let psi_idx = if dpsi <= 0 { 0usize } else { 1usize };

    SWITCHING_TABLE[sector][tau_idx][psi_idx]
}

/// Flux estimator using the voltage model in the αβ stationary frame.
///
/// Integrates: ψ = ∫(v - Rs·i) dt
/// A low-gain drift correction is applied.
#[derive(Debug, Clone, Copy)]
pub struct FluxEstimator<S: ControlScalar> {
    /// Estimated α-axis flux linkage (Wb).
    pub psi_alpha: S,
    /// Estimated β-axis flux linkage (Wb).
    pub psi_beta: S,
    /// Low-pass gain for drift correction (0..1, typically small).
    pub drift_reset_gain: S,
}

impl<S: ControlScalar> FluxEstimator<S> {
    pub fn new(drift_reset_gain: S) -> Self {
        Self {
            psi_alpha: S::ZERO,
            psi_beta: S::ZERO,
            drift_reset_gain,
        }
    }

    /// Integrate stator voltage minus resistive drop to update flux estimate.
    ///
    /// Uses forward Euler with a soft drift correction:
    /// `ψ(k+1) = ψ(k) + (v - Rs·i - g·ψ) * dt`
    pub fn update(&mut self, v_alpha: S, v_beta: S, i_alpha: S, i_beta: S, r_s: S, dt: S) {
        let e_alpha = v_alpha - r_s * i_alpha - self.drift_reset_gain * self.psi_alpha;
        let e_beta = v_beta - r_s * i_beta - self.drift_reset_gain * self.psi_beta;
        self.psi_alpha += e_alpha * dt;
        self.psi_beta += e_beta * dt;
    }

    /// Flux magnitude |ψ| = √(ψα² + ψβ²).
    pub fn flux_magnitude(&self) -> S {
        (self.psi_alpha * self.psi_alpha + self.psi_beta * self.psi_beta).sqrt()
    }

    /// Flux angle θ = atan2(ψβ, ψα) in range (-π, π], shifted to [0, 2π).
    pub fn flux_angle(&self) -> S {
        let angle = self.psi_beta.atan2(self.psi_alpha);
        if angle < S::ZERO {
            angle + S::PI * S::TWO
        } else {
            angle
        }
    }

    /// Flux sector (1..=6) based on flux angle.
    ///
    /// Each sector spans 60° (π/3 rad). Computation is done entirely in f64
    /// with a tiny epsilon nudge to handle IEEE754 rounding at exact boundaries
    /// (e.g. atan2(0,-1) = π, but π / (π/3) = 2.999… in floating-point).
    pub fn flux_sector(&self) -> usize {
        let psi_a = self.psi_alpha.to_f64();
        let psi_b = self.psi_beta.to_f64();
        let angle = psi_b.atan2(psi_a);
        let angle = if angle < 0.0 {
            angle + 2.0 * core::f64::consts::PI
        } else {
            angle
        };
        let sector_width = core::f64::consts::FRAC_PI_3;
        let sector = (angle / sector_width + 1e-12).floor() as usize;
        (sector % 6) + 1
    }
}

/// DTC controller combining flux estimator, hysteresis comparators, and switching table.
#[derive(Debug, Clone, Copy)]
pub struct DtcController<S: ControlScalar> {
    /// Flux estimator.
    pub flux_est: FluxEstimator<S>,
    /// Torque hysteresis band.
    pub torque_band: TorqueBand<S>,
    /// Flux hysteresis band.
    pub flux_band: FluxBand<S>,
    /// Last estimated electromagnetic torque (N·m).
    pub torque_est: S,
    /// Number of pole pairs.
    pub poles: u8,
}

impl<S: ControlScalar> DtcController<S> {
    /// Create new DTC controller.
    ///
    /// # Arguments
    /// * `r_s` - Stator resistance (Ω) — used in flux estimator via update call.
    /// * `torque_hysteresis` - Half-width of torque hysteresis band (N·m).
    /// * `flux_hysteresis` - Half-width of flux hysteresis band (Wb).
    /// * `flux_ref` - Not stored; flux reference is passed per-update.
    /// * `poles` - Number of pole pairs.
    pub fn new(_r_s: S, torque_hysteresis: S, flux_hysteresis: S, _flux_ref: S, poles: u8) -> Self {
        Self {
            flux_est: FluxEstimator::new(S::from_f64(0.01)),
            torque_band: TorqueBand::new(torque_hysteresis),
            flux_band: FluxBand::new(flux_hysteresis),
            torque_est: S::ZERO,
            poles,
        }
    }

    /// Run one DTC step.
    ///
    /// Computes estimated torque from flux and current, applies hysteresis
    /// comparators, and looks up the appropriate voltage vector.
    ///
    /// # Arguments
    /// * `v_alpha`, `v_beta` - Applied αβ voltages from previous step (V).
    /// * `i_alpha`, `i_beta` - Measured αβ currents (A).
    /// * `torque_ref` - Torque reference (N·m).
    /// * `flux_ref` - Flux reference (Wb).
    /// * `r_s` - Stator resistance (Ω).
    /// * `dt` - Time step (s).
    ///
    /// # Returns
    /// Voltage vector index (0-7).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        v_alpha: S,
        v_beta: S,
        i_alpha: S,
        i_beta: S,
        torque_ref: S,
        flux_ref: S,
        r_s: S,
        dt: S,
    ) -> u8 {
        // Update flux estimate
        self.flux_est
            .update(v_alpha, v_beta, i_alpha, i_beta, r_s, dt);

        // Estimate electromagnetic torque:
        // τ = (3/2) * p * (ψα·iβ - ψβ·iα)
        let p = S::from_f64(self.poles as f64);
        let three_halves = S::from_f64(1.5);
        self.torque_est = three_halves
            * p
            * (self.flux_est.psi_alpha * i_beta - self.flux_est.psi_beta * i_alpha);

        // Torque error → hysteresis comparator
        let torque_error = torque_ref - self.torque_est;
        let dtau = self.torque_band.compare(torque_error);

        // Flux error → hysteresis comparator
        let flux_mag = self.flux_est.flux_magnitude();
        let flux_error = flux_ref - flux_mag;
        let dpsi = self.flux_band.compare(flux_error);

        // Flux sector
        let sector = self.flux_est.flux_sector();

        select_voltage_vector(sector, dtau, dpsi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switching_table_active_vectors() {
        // For sector 1, dtau=+1, dpsi=1 → V6
        let v = select_voltage_vector(1, 1, 1);
        assert_eq!(v, 6);

        // For sector 1, dtau=-1, dpsi=0 → V5
        let v = select_voltage_vector(1, -1, 0);
        assert_eq!(v, 5);

        // For sector 3, dtau=+1, dpsi=0 → V4
        let v = select_voltage_vector(3, 1, 0);
        assert_eq!(v, 4);
    }

    #[test]
    fn test_switching_table_zero_vectors() {
        // dtau=0 → always zero vector (0 or 7)
        for sector in 1..=6 {
            let v0 = select_voltage_vector(sector, 0, 0);
            let v1 = select_voltage_vector(sector, 0, 1);
            assert!(
                v0 == 0 || v0 == 7,
                "sector {sector}: dtau=0 dpsi=0 should be zero vector, got {v0}"
            );
            assert!(
                v1 == 0 || v1 == 7,
                "sector {sector}: dtau=0 dpsi=1 should be zero vector, got {v1}"
            );
        }
    }

    #[test]
    fn test_flux_estimator_integration() {
        let mut est = FluxEstimator::<f32>::new(0.0);
        // Apply constant αβ voltage for 10 steps with zero current
        for _ in 0..10 {
            est.update(10.0, 5.0, 0.0, 0.0, 0.1, 0.001);
        }
        // After 10 steps at 0.001s: ψα ≈ 0.1, ψβ ≈ 0.05
        assert!((est.psi_alpha - 0.1_f32).abs() < 1e-5);
        assert!((est.psi_beta - 0.05_f32).abs() < 1e-5);
    }

    #[test]
    fn test_flux_sector_boundaries() {
        let mut est = FluxEstimator::<f32>::new(0.0);
        // Force known flux values: ψα=1, ψβ=0 → angle=0 → sector 1
        est.psi_alpha = 1.0;
        est.psi_beta = 0.0;
        assert_eq!(est.flux_sector(), 1);

        // ψα=0, ψβ=1 → angle=π/2 → sector 2 (60°..120°)
        est.psi_alpha = 0.0;
        est.psi_beta = 1.0;
        assert_eq!(est.flux_sector(), 2);

        // ψα=-1, ψβ=0 → angle=π → sector 4 (180°..240°)
        est.psi_alpha = -1.0;
        est.psi_beta = 0.0;
        assert_eq!(est.flux_sector(), 4);
    }

    #[test]
    fn test_dtc_controller_returns_valid_vector() {
        let mut dtc = DtcController::<f32>::new(0.1, 0.5, 0.05, 0.5, 2);

        // Run several steps to build up flux estimate
        for _ in 0..50 {
            let v = dtc.update(10.0, 5.0, 1.0, 0.5, 1.0, 0.5, 0.1, 0.001);
            assert!(v <= 7, "Voltage vector index must be 0..=7, got {v}");
        }
    }
}
