//! Direct Thrust/Force Control (DTC variant) for linear motor drives.
//!
//! Implements a hysteresis-based direct thrust control scheme analogous to
//! classical DTC for rotary motors but adapted for linear permanent magnet
//! synchronous motors (LPMSMs). The primary controlled quantity is thrust
//! (force) rather than torque, and flux linkage is estimated from stator
//! voltages and currents in the αβ plane (where α corresponds to the direction
//! of linear motion and β is orthogonal).
//!
//! # Control Structure
//!
//! 1. **Flux estimation** — Voltage-model integrator with drift correction:
//!    `ψ = ∫(v − Rs·i) dt`
//!
//! 2. **Thrust estimation** — Cross-product of flux and current:
//!    `F = (3/2) · (π/τp) · (ψα·iβ − ψβ·iα)`
//!    where τp is the pole pitch of the linear motor.
//!
//! 3. **Hysteresis comparators** — Separate bands for flux magnitude and thrust.
//!
//! 4. **Switching logic** — Three-phase voltage vector selected from a 6-sector
//!    switching table identical in structure to rotary DTC.
//!
//! # Force Ripple Minimisation
//!
//! A deadband is applied around the force reference to suppress chattering:
//! when the force error is within the deadband, the zero voltage vector is
//! selected regardless of other comparator outputs.

use crate::core::scalar::ControlScalar;

/// Error returned by Direct Thrust Control construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtcLinearError {
    /// Pole pitch must be strictly positive.
    InvalidPolePitch,
    /// Stator resistance must be non-negative.
    InvalidResistance,
    /// Hysteresis band must be positive.
    InvalidHysteresis,
}

impl core::fmt::Display for DtcLinearError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidPolePitch => write!(f, "pole pitch must be > 0"),
            Self::InvalidResistance => write!(f, "stator resistance must be >= 0"),
            Self::InvalidHysteresis => write!(f, "hysteresis band must be > 0"),
        }
    }
}

/// Hysteresis comparator that returns a signed discrete output.
#[derive(Debug, Clone, Copy)]
struct HysteresisComparator<S: ControlScalar> {
    /// Upper threshold: if error > upper → output +1.
    upper: S,
    /// Lower threshold: if error < lower → output -1.
    lower: S,
    /// Last output (memory for hysteresis).
    last: i8,
}

impl<S: ControlScalar> HysteresisComparator<S> {
    fn new(band: S) -> Self {
        Self {
            upper: band,
            lower: -band,
            last: 0,
        }
    }

    /// Compare error against hysteresis thresholds.
    ///
    /// Returns:
    /// * `+1` if error exceeded upper threshold
    /// * `-1` if error fell below lower threshold
    /// * previous value otherwise (memory behaviour)
    fn compare(&mut self, error: S) -> i8 {
        if error > self.upper {
            self.last = 1;
        } else if error < self.lower {
            self.last = -1;
        }
        // Otherwise retain last: true hysteresis behaviour
        self.last
    }
}

/// Flux-magnitude comparator returning 0 (decrease) or 1 (increase).
#[derive(Debug, Clone, Copy)]
struct FluxHysteresis<S: ControlScalar> {
    upper: S,
    lower: S,
    last: i8,
}

impl<S: ControlScalar> FluxHysteresis<S> {
    fn new(band: S) -> Self {
        Self {
            upper: band,
            lower: -band,
            last: 1,
        }
    }

    fn compare(&mut self, error: S) -> i8 {
        if error > self.upper {
            self.last = 1;
        } else if error < self.lower {
            self.last = 0;
        }
        self.last
    }
}

/// Voltage model flux estimator for the αβ stationary frame.
///
/// Integrates: ψ = ∫(v − Rs·i) dt
/// A low-gain drift correction term prevents integrator wind-up.
#[derive(Debug, Clone, Copy)]
pub struct LinearFluxEstimator<S: ControlScalar> {
    /// Estimated flux α-component (Wb/m for linear motor).
    pub psi_alpha: S,
    /// Estimated flux β-component (Wb/m for linear motor).
    pub psi_beta: S,
    /// Drift correction gain (small positive value, e.g. 0.01).
    drift_gain: S,
}

impl<S: ControlScalar> LinearFluxEstimator<S> {
    /// Construct with a given drift-correction gain.
    pub fn new(drift_gain: S) -> Self {
        Self {
            psi_alpha: S::ZERO,
            psi_beta: S::ZERO,
            drift_gain,
        }
    }

    /// Integrate for one timestep.
    ///
    /// # Arguments
    /// * `v_alpha`, `v_beta` - Applied αβ voltages (V).
    /// * `i_alpha`, `i_beta` - Measured αβ currents (A).
    /// * `r_s` - Stator resistance (Ω).
    /// * `dt` - Timestep (s).
    pub fn update(&mut self, v_alpha: S, v_beta: S, i_alpha: S, i_beta: S, r_s: S, dt: S) {
        let e_alpha = v_alpha - r_s * i_alpha - self.drift_gain * self.psi_alpha;
        let e_beta = v_beta - r_s * i_beta - self.drift_gain * self.psi_beta;
        self.psi_alpha += e_alpha * dt;
        self.psi_beta += e_beta * dt;
    }

    /// Flux magnitude |ψ| = √(ψα² + ψβ²).
    pub fn magnitude(&self) -> S {
        (self.psi_alpha * self.psi_alpha + self.psi_beta * self.psi_beta).sqrt()
    }

    /// Flux angle in [0, 2π).
    pub fn angle(&self) -> S {
        let angle = self.psi_beta.atan2(self.psi_alpha);
        if angle < S::ZERO {
            angle + S::PI * S::TWO
        } else {
            angle
        }
    }

    /// Flux sector (1..=6).
    pub fn sector(&self) -> usize {
        let angle = self.angle();
        let normalized = angle / (S::PI * S::TWO);
        let sector_f = normalized * S::from_f64(6.0);
        (sector_f.to_f64() as usize % 6) + 1
    }
}

/// DTC switching table (same topology as rotary DTC, Takahashi).
/// Indexed by [sector 0..5][thrust_idx 0..2][flux_idx 0..1].
const SWITCHING_TABLE_LINEAR: [[[u8; 2]; 3]; 6] = [
    // Sector 1 (0°..60°)
    [[5, 3], [7, 0], [2, 6]],
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

/// Select voltage vector from switching table.
fn select_vector_linear(sector: usize, df: i8, dpsi: i8) -> u8 {
    let s = sector.saturating_sub(1).min(5);
    let f_idx = match df {
        i8::MIN..=-1 => 0,
        0 => 1,
        _ => 2,
    };
    let psi_idx = if dpsi <= 0 { 0usize } else { 1usize };
    SWITCHING_TABLE_LINEAR[s][f_idx][psi_idx]
}

/// Direct Thrust Control state snapshot.
#[derive(Debug, Clone, Copy)]
pub struct DtcLinearState<S: ControlScalar> {
    /// Estimated thrust/force (N).
    pub thrust_est: S,
    /// Flux magnitude (Wb/m).
    pub flux_magnitude: S,
    /// α-component of flux (Wb/m).
    pub psi_alpha: S,
    /// β-component of flux (Wb/m).
    pub psi_beta: S,
    /// Selected voltage vector index (0..=7).
    pub voltage_vector: u8,
}

/// Direct Thrust/Force Control for linear permanent magnet synchronous motors.
///
/// # Generic Parameter
/// `S` — scalar type implementing [`ControlScalar`], typically `f32` or `f64`.
#[derive(Debug, Clone)]
pub struct DirectThrustController<S: ControlScalar> {
    /// Flux estimator.
    flux_est: LinearFluxEstimator<S>,
    /// Thrust hysteresis comparator.
    thrust_comparator: HysteresisComparator<S>,
    /// Flux hysteresis comparator.
    flux_comparator: FluxHysteresis<S>,
    /// Stator resistance (Ω).
    r_s: S,
    /// Pole pitch τp (m) — distance from N to N pole on primary.
    pole_pitch: S,
    /// Deadband half-width around force reference to suppress ripple (N).
    force_deadband: S,
    /// Last estimated thrust (N).
    thrust_est: S,
    /// Last selected voltage vector.
    voltage_vector: u8,
}

impl<S: ControlScalar> DirectThrustController<S> {
    /// Construct a new Direct Thrust Controller.
    ///
    /// # Arguments
    /// * `r_s` - Stator resistance (Ω).
    /// * `pole_pitch` - Pole pitch τp (m). Must be > 0.
    /// * `thrust_hysteresis` - Thrust hysteresis half-band (N).
    /// * `flux_hysteresis` - Flux hysteresis half-band (Wb/m).
    /// * `force_deadband` - Half-width of deadband around force reference (N).
    ///   Set to 0 to disable deadband.
    /// * `drift_gain` - Flux estimator drift correction (small, e.g. 0.005–0.02).
    ///
    /// # Errors
    /// Returns `DtcLinearError` if any parameter is physically invalid.
    pub fn new(
        r_s: S,
        pole_pitch: S,
        thrust_hysteresis: S,
        flux_hysteresis: S,
        force_deadband: S,
        drift_gain: S,
    ) -> Result<Self, DtcLinearError> {
        if pole_pitch <= S::ZERO {
            return Err(DtcLinearError::InvalidPolePitch);
        }
        if r_s < S::ZERO {
            return Err(DtcLinearError::InvalidResistance);
        }
        if thrust_hysteresis <= S::ZERO || flux_hysteresis <= S::ZERO {
            return Err(DtcLinearError::InvalidHysteresis);
        }

        Ok(Self {
            flux_est: LinearFluxEstimator::new(drift_gain),
            thrust_comparator: HysteresisComparator::new(thrust_hysteresis),
            flux_comparator: FluxHysteresis::new(flux_hysteresis),
            r_s,
            pole_pitch,
            force_deadband,
            thrust_est: S::ZERO,
            voltage_vector: 0,
        })
    }

    /// Run one control step.
    ///
    /// # Arguments
    /// * `v_alpha`, `v_beta` - Applied αβ voltages from previous switching state (V).
    /// * `i_alpha`, `i_beta` - Measured αβ currents (A).
    /// * `force_ref` - Thrust reference (N). Positive = forward direction.
    /// * `flux_ref` - Flux magnitude reference (Wb/m).
    /// * `dt` - Control timestep (s).
    ///
    /// # Returns
    /// Voltage vector index 0–7 (0 and 7 are zero vectors; 1–6 are active).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        v_alpha: S,
        v_beta: S,
        i_alpha: S,
        i_beta: S,
        force_ref: S,
        flux_ref: S,
        dt: S,
    ) -> u8 {
        // Update flux estimate
        self.flux_est
            .update(v_alpha, v_beta, i_alpha, i_beta, self.r_s, dt);

        // Estimate thrust:
        // F = (3/2) · (π/τp) · (ψα·iβ − ψβ·iα)
        let pi_over_tau = S::PI / self.pole_pitch;
        let three_half = S::from_f64(1.5);
        self.thrust_est = three_half
            * pi_over_tau
            * (self.flux_est.psi_alpha * i_beta - self.flux_est.psi_beta * i_alpha);

        // Flux magnitude and sector
        let flux_mag = self.flux_est.magnitude();
        let sector = self.flux_est.sector();

        // Flux error → comparator
        let flux_err = flux_ref - flux_mag;
        let dpsi = self.flux_comparator.compare(flux_err);

        // Thrust error with deadband
        let force_err = force_ref - self.thrust_est;
        let force_err_abs = if force_err < S::ZERO {
            -force_err
        } else {
            force_err
        };

        let vector = if force_err_abs <= self.force_deadband {
            // Within deadband: zero vector to minimise ripple
            7u8
        } else {
            let df = self.thrust_comparator.compare(force_err);
            select_vector_linear(sector, df, dpsi)
        };

        self.voltage_vector = vector;
        vector
    }

    /// Current controller state snapshot.
    pub fn state(&self) -> DtcLinearState<S> {
        DtcLinearState {
            thrust_est: self.thrust_est,
            flux_magnitude: self.flux_est.magnitude(),
            psi_alpha: self.flux_est.psi_alpha,
            psi_beta: self.flux_est.psi_beta,
            voltage_vector: self.voltage_vector,
        }
    }

    /// Estimated thrust/force (N).
    pub fn thrust_estimate(&self) -> S {
        self.thrust_est
    }

    /// Reset internal state (flux integrator, hysteresis, estimates).
    pub fn reset(&mut self) {
        self.flux_est = LinearFluxEstimator::new(self.flux_est.drift_gain);
        self.thrust_comparator.last = 0;
        self.flux_comparator.last = 1;
        self.thrust_est = S::ZERO;
        self.voltage_vector = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_controller() -> DirectThrustController<f64> {
        DirectThrustController::new(
            0.5,  // r_s
            0.06, // pole_pitch (60mm)
            5.0,  // thrust hysteresis (N)
            0.02, // flux hysteresis (Wb/m)
            1.0,  // force deadband (N)
            0.01, // drift gain
        )
        .expect("valid params")
    }

    #[test]
    fn construction_succeeds_with_valid_params() {
        let ctrl = make_controller();
        assert_eq!(ctrl.thrust_estimate(), 0.0);
    }

    #[test]
    fn invalid_pole_pitch_returns_error() {
        let result = DirectThrustController::<f64>::new(0.5, 0.0, 5.0, 0.02, 1.0, 0.01);
        assert_eq!(result.unwrap_err(), DtcLinearError::InvalidPolePitch);
    }

    #[test]
    fn invalid_hysteresis_returns_error() {
        let result = DirectThrustController::<f64>::new(0.5, 0.06, 0.0, 0.02, 1.0, 0.01);
        assert_eq!(result.unwrap_err(), DtcLinearError::InvalidHysteresis);
    }

    #[test]
    fn output_vector_is_in_valid_range() {
        let mut ctrl = make_controller();
        for step in 0..100 {
            let t = step as f64 * 1e-4;
            let v = 100.0 * libm::sin(2.0 * core::f64::consts::PI * 50.0 * t);
            let vec = ctrl.update(v, v * 0.5, 2.0 * libm::cos(100.0 * t), 1.0, 50.0, 0.5, 1e-4);
            assert!(vec <= 7, "vector {} out of range", vec);
        }
    }

    #[test]
    fn deadband_produces_zero_vector() {
        let mut ctrl = DirectThrustController::<f64>::new(
            0.1, 0.06, 5.0, 0.02, 1000.0, // huge deadband → always zero vector
            0.01,
        )
        .expect("valid");
        // With a huge deadband, the output should always be a zero vector (0 or 7)
        for _ in 0..20 {
            let v = ctrl.update(1.0, 0.0, 0.1, 0.0, 0.0, 0.5, 1e-4);
            assert!(v == 0 || v == 7, "expected zero vector, got {}", v);
        }
    }

    #[test]
    fn flux_estimator_integrates_correctly() {
        let mut est = LinearFluxEstimator::<f64>::new(0.0);
        // v=10V, i=0, r_s=0, dt=0.001s × 10 steps → ψα ≈ 0.1
        for _ in 0..10 {
            est.update(10.0, 0.0, 0.0, 0.0, 0.0, 1e-3);
        }
        assert!((est.psi_alpha - 0.1).abs() < 1e-9, "ψα = {}", est.psi_alpha);
    }

    #[test]
    fn reset_clears_flux_estimate() {
        let mut ctrl = make_controller();
        for _ in 0..50 {
            ctrl.update(50.0, 25.0, 2.0, 1.0, 20.0, 0.5, 1e-4);
        }
        ctrl.reset();
        let state = ctrl.state();
        assert_eq!(state.psi_alpha, 0.0);
        assert_eq!(state.psi_beta, 0.0);
        assert_eq!(state.thrust_est, 0.0);
    }
}
