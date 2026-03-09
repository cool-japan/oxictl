//! Switched Reluctance Motor (SRM) nonlinear torque and flux model.
//!
//! Models the highly nonlinear magnetic characteristics of an SRM phase, where
//! both inductance and flux linkage depend strongly on rotor position and current.
//!
//! # Inductance Profile
//!
//! The per-phase inductance is approximated as a position-dependent function:
//! ```text
//! L(θ, i) = L_min + (L_max − L_min) · f(θ) · g(i)
//! ```
//! where:
//! - `f(θ)` is a piecewise sinusoidal function of rotor angle normalised to [0,1].
//! - `g(i) = 1 / (1 + k_sat · i²)` is a saturation correction.
//!
//! # Torque Production
//!
//! Instantaneous torque per phase via co-energy:
//! ```text
//! T_phase = 0.5 · i² · dL/dθ
//! ```
//!
//! # Commutation Logic
//!
//! Phases are switched on/off according to the optimal turn-on (θ_on) and
//! turn-off (θ_off) angles. The commutation window is computed from motor
//! geometry (number of stator and rotor poles).
//!
//! # Torque Ripple
//!
//! The ripple metric is defined as:
//! ```text
//! ripple = (T_max − T_min) / T_avg
//! ```
//! over one commutation period. It is tracked as a running estimate using an
//! exponential moving average.

use crate::core::scalar::ControlScalar;

/// Error type for SRM model construction and queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrmError {
    /// Number of stator poles must be even and ≥ 4.
    InvalidStatorPoles,
    /// Number of rotor poles must be ≥ 2.
    InvalidRotorPoles,
    /// Inductance values must satisfy L_min < L_max, both > 0.
    InvalidInductance,
    /// Phase index out of range.
    InvalidPhaseIndex,
    /// Saturation coefficient must be non-negative.
    InvalidSaturation,
}

impl core::fmt::Display for SrmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidStatorPoles => write!(f, "stator poles must be even and >= 4"),
            Self::InvalidRotorPoles => write!(f, "rotor poles must be >= 2"),
            Self::InvalidInductance => write!(f, "L_min < L_max, both > 0 required"),
            Self::InvalidPhaseIndex => write!(f, "phase index out of range"),
            Self::InvalidSaturation => write!(f, "saturation coefficient must be >= 0"),
        }
    }
}

/// Parameters for a single SRM phase inductance profile.
#[derive(Debug, Clone, Copy)]
pub struct SrmPhaseParams<S: ControlScalar> {
    /// Minimum (unaligned) inductance (H).
    pub l_min: S,
    /// Maximum (aligned) inductance (H).
    pub l_max: S,
    /// Saturation coefficient k_sat (A⁻²). Larger → more saturation.
    pub k_sat: S,
    /// Phase resistance (Ω).
    pub r_phase: S,
    /// Rated phase current (A) — used for normalisation.
    pub i_rated: S,
}

/// Commutation angle set for one phase.
#[derive(Debug, Clone, Copy)]
pub struct CommutationAngles<S: ControlScalar> {
    /// Turn-on angle θ_on (rad, relative to aligned position).
    pub theta_on: S,
    /// Turn-off angle θ_off (rad, relative to aligned position).
    pub theta_off: S,
}

/// Phase state: current and voltage.
#[derive(Debug, Clone, Copy)]
pub struct SrmPhaseState<S: ControlScalar> {
    /// Phase current (A).
    pub current: S,
    /// Applied phase voltage (V). Positive or negative DC bus.
    pub voltage: S,
    /// Whether this phase is active (switched on).
    pub active: bool,
}

impl<S: ControlScalar> Default for SrmPhaseState<S> {
    fn default() -> Self {
        Self {
            current: S::ZERO,
            voltage: S::ZERO,
            active: false,
        }
    }
}

/// Torque ripple estimator using exponential moving average.
#[derive(Debug, Clone, Copy)]
struct RippleEstimator<S: ControlScalar> {
    /// Smoothing coefficient α ∈ (0, 1].
    alpha: S,
    /// Minimum torque in current window.
    t_min: S,
    /// Maximum torque in current window.
    t_max: S,
    /// Running average torque.
    t_avg: S,
    /// Current ripple estimate.
    ripple: S,
    /// Sample count since reset.
    count: u32,
}

impl<S: ControlScalar> RippleEstimator<S> {
    fn new(alpha: S) -> Self {
        Self {
            alpha,
            t_min: S::from_f64(f64::MAX / 2.0),
            t_max: S::from_f64(f64::MIN / 2.0),
            t_avg: S::ZERO,
            ripple: S::ZERO,
            count: 0,
        }
    }

    fn update(&mut self, torque: S) {
        if torque < self.t_min {
            self.t_min = torque;
        }
        if torque > self.t_max {
            self.t_max = torque;
        }

        // Exponential moving average of torque
        self.t_avg = self.alpha * torque + (S::ONE - self.alpha) * self.t_avg;
        self.count += 1;

        // Update ripple estimate periodically
        if self.count % 32 == 0 && self.t_avg.abs() > S::from_f64(1e-9) {
            let raw_ripple = (self.t_max - self.t_min) / self.t_avg.abs();
            self.ripple = self.alpha * raw_ripple + (S::ONE - self.alpha) * self.ripple;
            // Reset window
            self.t_min = torque;
            self.t_max = torque;
        }
    }
}

/// Switched Reluctance Motor model (N-phase, up to 8 phases via heapless storage).
///
/// Supported configurations: 6/4, 8/6, 12/8 SRM (stator/rotor poles).
///
/// # Generic Parameters
/// * `S` — scalar type (f32 or f64).
/// * `N` — number of phases (const generic; max 8 for embedded applications).
#[derive(Debug, Clone)]
pub struct SrmModel<S: ControlScalar, const N: usize> {
    /// Per-phase parameters.
    phase_params: [SrmPhaseParams<S>; N],
    /// Number of stator poles.
    n_stator_poles: u32,
    /// Number of rotor poles.
    n_rotor_poles: u32,
    /// Number of phases (= N).
    n_phases: usize,
    /// Commutation angles per phase.
    commutation: [CommutationAngles<S>; N],
    /// Phase states (current, voltage, active flag).
    phase_states: [SrmPhaseState<S>; N],
    /// Rotor position (rad, mechanical).
    theta_mech: S,
    /// Mechanical angular velocity (rad/s).
    omega_mech: S,
    /// Moment of inertia (kg·m²).
    inertia: S,
    /// Viscous friction (N·m·s/rad).
    b_friction: S,
    /// Total electromagnetic torque estimate (N·m).
    torque_total: S,
    /// Torque ripple estimator.
    ripple_est: RippleEstimator<S>,
    /// Rotor pole pitch (rad) = 2π / n_rotor_poles.
    rotor_pole_pitch: S,
}

impl<S: ControlScalar, const N: usize> SrmModel<S, N> {
    /// Construct a new SRM model.
    ///
    /// # Arguments
    /// * `n_stator_poles` - Number of stator poles (must be even, ≥ 4).
    /// * `n_rotor_poles` - Number of rotor poles (≥ 2).
    /// * `phase_params` - Per-phase magnetic and electrical parameters.
    /// * `commutation` - Turn-on/off angles per phase.
    /// * `inertia` - Moment of inertia (kg·m²).
    /// * `b_friction` - Viscous friction coefficient (N·m·s/rad).
    ///
    /// # Errors
    /// Returns `SrmError` if any motor parameter is physically inconsistent.
    pub fn new(
        n_stator_poles: u32,
        n_rotor_poles: u32,
        phase_params: [SrmPhaseParams<S>; N],
        commutation: [CommutationAngles<S>; N],
        inertia: S,
        b_friction: S,
    ) -> Result<Self, SrmError> {
        if n_stator_poles < 4 || n_stator_poles % 2 != 0 {
            return Err(SrmError::InvalidStatorPoles);
        }
        if n_rotor_poles < 2 {
            return Err(SrmError::InvalidRotorPoles);
        }
        for p in &phase_params {
            if p.l_min <= S::ZERO || p.l_max <= p.l_min {
                return Err(SrmError::InvalidInductance);
            }
            if p.k_sat < S::ZERO {
                return Err(SrmError::InvalidSaturation);
            }
        }

        let n_phases = N;
        let rotor_pole_pitch = S::PI * S::TWO / S::from_f64(n_rotor_poles as f64);
        let phase_states = core::array::from_fn(|_| SrmPhaseState::default());

        Ok(Self {
            phase_params,
            n_stator_poles,
            n_rotor_poles,
            n_phases,
            commutation,
            phase_states,
            theta_mech: S::ZERO,
            omega_mech: S::ZERO,
            inertia,
            b_friction,
            torque_total: S::ZERO,
            ripple_est: RippleEstimator::new(S::from_f64(0.05)),
            rotor_pole_pitch,
        })
    }

    /// Compute phase inductance L(θ, i) for a given phase.
    ///
    /// Uses a piecewise sinusoidal position profile with a saturation correction.
    ///
    /// # Arguments
    /// * `phase` - Phase index (0..N).
    /// * `theta` - Rotor mechanical angle (rad).
    /// * `current` - Phase current (A).
    ///
    /// # Errors
    /// Returns `SrmError::InvalidPhaseIndex` if phase ≥ N.
    pub fn inductance(&self, phase: usize, theta: S, current: S) -> Result<S, SrmError> {
        if phase >= self.n_phases {
            return Err(SrmError::InvalidPhaseIndex);
        }
        let params = &self.phase_params[phase];
        let l_min = params.l_min;
        let l_max = params.l_max;
        let k_sat = params.k_sat;

        // Phase offset: phases are evenly distributed over one rotor pole pitch.
        let phase_offset =
            self.rotor_pole_pitch * S::from_f64(phase as f64) / S::from_f64(self.n_phases as f64);

        // Normalise angle to [0, 1] over one rotor pole pitch (θ ↦ φ ∈ [0,1]).
        let theta_rel = theta - phase_offset;
        // Wrap to [0, rotor_pole_pitch)
        let wrapped = self.wrap_angle(theta_rel, self.rotor_pole_pitch);
        let phi = wrapped / self.rotor_pole_pitch; // 0..1

        // Position profile f(φ): sinusoidal, 1.0 at alignment (φ=0.5), 0.0 at edges.
        // f(φ) = 0.5 · (1 − cos(2π·φ))  — smoothly varies from 0 to 1 and back.
        let two_pi_phi = S::TWO * S::PI * phi;
        let f_theta = S::HALF * (S::ONE - two_pi_phi.cos());

        // Saturation correction g(i) = 1 / (1 + k_sat·i²)
        let i_sq = current * current;
        let g_current = S::ONE / (S::ONE + k_sat * i_sq);

        let l = l_min + (l_max - l_min) * f_theta * g_current;
        Ok(l)
    }

    /// Compute dL/dθ (inductance slope) via symmetric finite difference.
    ///
    /// # Arguments
    /// * `phase` - Phase index (0..N).
    /// * `theta` - Rotor angle (rad).
    /// * `current` - Phase current (A).
    fn inductance_gradient(&self, phase: usize, theta: S, current: S) -> S {
        let h = S::from_f64(1e-4); // rad
        let l_plus = self
            .inductance(phase, theta + h, current)
            .unwrap_or(S::ZERO);
        let l_minus = self
            .inductance(phase, theta - h, current)
            .unwrap_or(S::ZERO);
        (l_plus - l_minus) / (S::TWO * h)
    }

    /// Torque produced by a single phase.
    ///
    /// `T = 0.5 · i² · dL/dθ`
    ///
    /// # Arguments
    /// * `phase` - Phase index.
    /// * `theta` - Rotor angle (rad).
    /// * `current` - Phase current (A).
    pub fn phase_torque(&self, phase: usize, theta: S, current: S) -> Result<S, SrmError> {
        if phase >= self.n_phases {
            return Err(SrmError::InvalidPhaseIndex);
        }
        let dl_dtheta = self.inductance_gradient(phase, theta, current);
        Ok(S::HALF * current * current * dl_dtheta)
    }

    /// Determine whether a phase should be active at the current rotor position.
    ///
    /// The commutation window is defined by `[θ_on, θ_off)` within one rotor pole pitch.
    ///
    /// # Arguments
    /// * `phase` - Phase index.
    /// * `theta` - Rotor angle (rad).
    fn is_phase_active(&self, phase: usize, theta: S) -> bool {
        if phase >= self.n_phases {
            return false;
        }
        let ca = &self.commutation[phase];
        let phase_offset =
            self.rotor_pole_pitch * S::from_f64(phase as f64) / S::from_f64(self.n_phases as f64);
        let theta_rel = theta - phase_offset;
        let wrapped = self.wrap_angle(theta_rel, self.rotor_pole_pitch);

        // Phase active if wrapped ∈ [θ_on, θ_off)
        wrapped >= ca.theta_on && wrapped < ca.theta_off
    }

    /// Advance the motor model by one timestep.
    ///
    /// # Arguments
    /// * `voltages` - Applied voltage per phase (V). Length must equal N.
    ///   For inactive phases, voltage is ignored (zero is applied).
    /// * `tau_load` - External load torque (N·m).
    /// * `dt` - Timestep (s). Must be > 0.
    pub fn step(&mut self, voltages: &[S; N], tau_load: S, dt: S) {
        // --- Phase currents (RL circuit with position-dependent L) ---
        let mut torque_sum = S::ZERO;
        let theta = self.theta_mech;
        let omega = self.omega_mech;

        #[allow(clippy::needless_range_loop)]
        for ph in 0..self.n_phases {
            let active = self.is_phase_active(ph, theta);
            self.phase_states[ph].active = active;

            let v = if active { voltages[ph] } else { S::ZERO };
            self.phase_states[ph].voltage = v;

            let i = self.phase_states[ph].current;
            let r = self.phase_params[ph].r_phase;
            let l = self
                .inductance(ph, theta, i)
                .unwrap_or(self.phase_params[ph].l_min);

            // Back-EMF term: e = i · ω · dL/dθ
            let dl = self.inductance_gradient(ph, theta, i);
            let back_emf = i * omega * dl;

            // di/dt = (v − R·i − e) / L
            let l_safe = if l > S::from_f64(1e-12) {
                l
            } else {
                S::from_f64(1e-12)
            };
            let di_dt = (v - r * i - back_emf) / l_safe;
            self.phase_states[ph].current += di_dt * dt;

            // Clamp current to non-negative (SRM phases are unidirectional)
            if self.phase_states[ph].current < S::ZERO {
                self.phase_states[ph].current = S::ZERO;
            }

            // Phase torque contribution
            let t_ph = self
                .phase_torque(ph, theta, self.phase_states[ph].current)
                .unwrap_or(S::ZERO);
            torque_sum += t_ph;
        }

        self.torque_total = torque_sum;

        // --- Mechanical dynamics ---
        // dω/dt = (T_em − B·ω − T_load) / J
        let j_safe = if self.inertia > S::from_f64(1e-15) {
            self.inertia
        } else {
            S::from_f64(1e-15)
        };
        let domega = (torque_sum - self.b_friction * omega - tau_load) / j_safe;
        self.omega_mech += domega * dt;
        self.theta_mech += self.omega_mech * dt;

        // Keep θ in [0, 2π)
        self.theta_mech = self.wrap_angle(self.theta_mech, S::PI * S::TWO);

        // Update ripple estimator
        self.ripple_est.update(torque_sum);
    }

    /// Total electromagnetic torque (N·m) — sum over all phases.
    pub fn torque_total(&self) -> S {
        self.torque_total
    }

    /// Rotor mechanical speed (rad/s).
    pub fn omega_mech(&self) -> S {
        self.omega_mech
    }

    /// Rotor mechanical angle (rad).
    pub fn theta_mech(&self) -> S {
        self.theta_mech
    }

    /// Current in a specific phase (A).
    ///
    /// Returns 0 if phase index is out of range.
    pub fn phase_current(&self, phase: usize) -> S {
        if phase < self.n_phases {
            self.phase_states[phase].current
        } else {
            S::ZERO
        }
    }

    /// Whether a specific phase is currently active.
    ///
    /// Returns false if phase index is out of range.
    pub fn phase_active(&self, phase: usize) -> bool {
        if phase < self.n_phases {
            self.phase_states[phase].active
        } else {
            false
        }
    }

    /// Torque ripple estimate (dimensionless ratio, 0 = no ripple).
    pub fn torque_ripple(&self) -> S {
        self.ripple_est.ripple
    }

    /// Number of stator poles.
    pub fn n_stator_poles(&self) -> u32 {
        self.n_stator_poles
    }

    /// Number of rotor poles.
    pub fn n_rotor_poles(&self) -> u32 {
        self.n_rotor_poles
    }

    /// Reset all dynamic states (currents, position, speed).
    pub fn reset(&mut self) {
        for ph in 0..self.n_phases {
            self.phase_states[ph] = SrmPhaseState::default();
        }
        self.theta_mech = S::ZERO;
        self.omega_mech = S::ZERO;
        self.torque_total = S::ZERO;
        self.ripple_est = RippleEstimator::new(S::from_f64(0.05));
    }

    /// Wrap angle to [0, period).
    fn wrap_angle(&self, angle: S, period: S) -> S {
        if period <= S::ZERO {
            return S::ZERO;
        }
        // Use modulo via subtraction loop (no-std compatible)
        let mut a = angle;
        while a >= period {
            a -= period;
        }
        while a < S::ZERO {
            a += period;
        }
        a
    }
}

/// Convenience constructor for a standard 6/4 three-phase SRM.
///
/// Default parameters are representative of a small (≈100W) laboratory SRM.
pub fn srm_6_4_default<S: ControlScalar>() -> Result<SrmModel<S, 3>, SrmError> {
    let params = core::array::from_fn(|_| SrmPhaseParams {
        l_min: S::from_f64(5e-3),
        l_max: S::from_f64(30e-3),
        k_sat: S::from_f64(0.05),
        r_phase: S::from_f64(1.2),
        i_rated: S::from_f64(5.0),
    });

    // For a 6/4 SRM, rotor pole pitch = 2π/4 = π/2 rad
    // Commutation: turn-on slightly before aligned, turn-off at aligned
    let rpp = core::f64::consts::PI / 2.0; // rotor pole pitch
    let commutation = core::array::from_fn(|_| CommutationAngles {
        theta_on: S::from_f64(rpp * 0.1),
        theta_off: S::from_f64(rpp * 0.6),
    });

    SrmModel::new(
        6, // stator poles
        4, // rotor poles
        params,
        commutation,
        S::from_f64(5e-5), // inertia
        S::from_f64(1e-4), // friction
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srm_64_construction_succeeds() {
        let srm = srm_6_4_default::<f64>();
        assert!(srm.is_ok());
    }

    #[test]
    fn invalid_stator_poles_returns_error() {
        let params: [SrmPhaseParams<f64>; 3] = core::array::from_fn(|_| SrmPhaseParams {
            l_min: 5e-3,
            l_max: 30e-3,
            k_sat: 0.05,
            r_phase: 1.2,
            i_rated: 5.0,
        });
        let comm: [CommutationAngles<f64>; 3] = core::array::from_fn(|_| CommutationAngles {
            theta_on: 0.1,
            theta_off: 0.9,
        });
        let result: Result<SrmModel<f64, 3>, _> = SrmModel::new(3, 4, params, comm, 5e-5, 1e-4); // 3 is odd, invalid
        assert_eq!(result.unwrap_err(), SrmError::InvalidStatorPoles);
    }

    #[test]
    fn invalid_inductance_returns_error() {
        let mut params: [SrmPhaseParams<f64>; 3] = core::array::from_fn(|_| SrmPhaseParams {
            l_min: 5e-3,
            l_max: 30e-3,
            k_sat: 0.05,
            r_phase: 1.2,
            i_rated: 5.0,
        });
        params[0].l_min = 40e-3; // l_min > l_max → invalid
        let comm: [CommutationAngles<f64>; 3] = core::array::from_fn(|_| CommutationAngles {
            theta_on: 0.1,
            theta_off: 0.9,
        });
        let result: Result<SrmModel<f64, 3>, _> = SrmModel::new(6, 4, params, comm, 5e-5, 1e-4);
        assert_eq!(result.unwrap_err(), SrmError::InvalidInductance);
    }

    #[test]
    fn inductance_in_bounds() {
        let srm = srm_6_4_default::<f64>().expect("construction ok");
        for i_ph in 0..3 {
            for angle_step in 0..100 {
                let theta = angle_step as f64 * core::f64::consts::PI / 50.0;
                let l = srm.inductance(i_ph, theta, 2.0).expect("valid phase");
                let lmin = srm.phase_params[i_ph].l_min;
                let lmax = srm.phase_params[i_ph].l_max;
                assert!(l >= lmin * 0.99, "L={} < L_min={}", l, lmin);
                assert!(l <= lmax * 1.01, "L={} > L_max={}", l, lmax);
            }
        }
    }

    #[test]
    fn step_accelerates_motor_with_applied_voltage() {
        let mut srm = srm_6_4_default::<f64>().expect("ok");
        let voltages = [100.0_f64; 3];
        for _ in 0..1000 {
            srm.step(&voltages, 0.0, 1e-5);
        }
        assert!(
            srm.omega_mech() > 0.0,
            "motor should accelerate, got ω={}",
            srm.omega_mech()
        );
    }

    #[test]
    fn phase_current_non_negative() {
        let mut srm = srm_6_4_default::<f64>().expect("ok");
        let voltages = [50.0_f64; 3];
        for _ in 0..500 {
            srm.step(&voltages, 0.0, 1e-5);
            for ph in 0..3 {
                assert!(
                    srm.phase_current(ph) >= 0.0,
                    "phase {} current negative: {}",
                    ph,
                    srm.phase_current(ph)
                );
            }
        }
    }

    #[test]
    fn reset_restores_zero_state() {
        let mut srm = srm_6_4_default::<f64>().expect("ok");
        let voltages = [100.0_f64; 3];
        for _ in 0..500 {
            srm.step(&voltages, 0.0, 1e-5);
        }
        srm.reset();
        assert_eq!(srm.omega_mech(), 0.0);
        assert_eq!(srm.theta_mech(), 0.0);
        for ph in 0..3 {
            assert_eq!(srm.phase_current(ph), 0.0);
        }
    }

    #[test]
    fn out_of_range_phase_returns_error() {
        let srm = srm_6_4_default::<f64>().expect("ok");
        let result = srm.inductance(10, 0.0, 1.0);
        assert_eq!(result.unwrap_err(), SrmError::InvalidPhaseIndex);
    }
}
