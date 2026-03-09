use crate::core::scalar::ControlScalar;

/// Errors returned by energy storage models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnergyStorageError {
    /// A parameter was out of its valid range.
    InvalidParameter,
    /// Supercapacitor voltage exceeded its maximum rating.
    OverVoltage,
    /// Supercapacitor voltage went below zero.
    UnderVoltage,
    /// Battery state of charge exhausted (over-discharged).
    OverDischarge,
}

impl core::fmt::Display for EnergyStorageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EnergyStorageError::InvalidParameter => {
                write!(f, "EnergyStorageError: invalid parameter")
            }
            EnergyStorageError::OverVoltage => {
                write!(f, "EnergyStorageError: over-voltage on supercapacitor")
            }
            EnergyStorageError::UnderVoltage => {
                write!(f, "EnergyStorageError: under-voltage on supercapacitor")
            }
            EnergyStorageError::OverDischarge => {
                write!(f, "EnergyStorageError: battery over-discharged")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Supercapacitor
// ---------------------------------------------------------------------------

/// Supercapacitor (ultracapacitor) simple RC model.
///
/// ## Model
///
/// ```text
///   I_sc ──┬── R_sc ──┬──  terminal
///          │          │
///         C_sc       V_sc
///          │          │
///   GND ───┴──────────┘
/// ```
///
/// **State**: internal capacitor voltage `V_sc` \[V\].
///
/// **Dynamics** (Euler integration):
/// ```text
/// V_sc_dot = I_sc / C_sc − V_sc / (R_sc · C_sc)
/// ```
///
/// **Terminal voltage**:
/// ```text
/// V_term = V_sc − I_sc · R_sc     (ESR drop)
/// ```
///
/// Positive current `I_sc` means **charging** (current flows into the SC).
///
/// ## Energy
/// ```text
/// E = 0.5 · C_sc · V_sc²     [J]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Supercapacitor<S: ControlScalar> {
    /// Internal (capacitor) voltage \[V\].
    v: S,
    /// Capacitance \[F\].
    c_sc: S,
    /// Equivalent series resistance (ESR) \[Ω\].
    r_sc: S,
    /// Maximum allowable voltage \[V\].
    v_max: S,
    /// Integration time step \[s\].
    dt: S,
}

impl<S: ControlScalar> Supercapacitor<S> {
    /// Create a new supercapacitor model.
    ///
    /// # Parameters
    /// - `c_sc`   — Capacitance \[F\], must be > 0
    /// - `r_sc`   — ESR \[Ω\], must be ≥ 0
    /// - `v_max`  — Maximum voltage rating \[V\], must be > 0
    /// - `v_init` — Initial voltage \[V\], must be in \[0, v_max\]
    /// - `dt`     — Integration step \[s\], must be > 0
    ///
    /// # Errors
    /// [`EnergyStorageError::InvalidParameter`] if any constraint fails.
    pub fn new(c_sc: S, r_sc: S, v_max: S, v_init: S, dt: S) -> Result<Self, EnergyStorageError> {
        if c_sc <= S::ZERO || r_sc < S::ZERO || v_max <= S::ZERO || dt <= S::ZERO {
            return Err(EnergyStorageError::InvalidParameter);
        }
        if v_init < S::ZERO || v_init > v_max {
            return Err(EnergyStorageError::InvalidParameter);
        }
        Ok(Self {
            v: v_init,
            c_sc,
            r_sc,
            v_max,
            dt,
        })
    }

    /// Advance the supercapacitor by one time step.
    ///
    /// # Parameters
    /// - `i_sc` — Current into the supercapacitor \[A\].
    ///   Positive = charging, negative = discharging.
    ///
    /// # Returns
    /// Terminal voltage \[V\] on success.
    ///
    /// # Errors
    /// - [`EnergyStorageError::OverVoltage`] if the internal voltage exceeds `v_max`.
    pub fn step(&mut self, i_sc: S) -> Result<S, EnergyStorageError> {
        // dV/dt = I/C - V/(R*C)   [second term absent when R ≈ 0]
        let v_dot = if self.r_sc > S::EPSILON {
            i_sc / self.c_sc - self.v / (self.r_sc * self.c_sc)
        } else {
            i_sc / self.c_sc
        };

        self.v += v_dot * self.dt;

        // Clamp at zero (capacitor cannot have negative voltage in this model)
        if self.v < S::ZERO {
            self.v = S::ZERO;
        }

        if self.v > self.v_max {
            return Err(EnergyStorageError::OverVoltage);
        }

        // Terminal voltage accounts for ESR drop
        let v_term = self.v - i_sc * self.r_sc;
        Ok(v_term)
    }

    /// Internal capacitor voltage \[V\].
    #[inline]
    pub fn voltage(&self) -> S {
        self.v
    }

    /// State of charge = V_sc / V_max, clamped to \[0, 1\].
    #[inline]
    pub fn soc(&self) -> S {
        (self.v / self.v_max).clamp_val(S::ZERO, S::ONE)
    }

    /// Stored energy \[J\] = 0.5 · C · V².
    #[inline]
    pub fn energy(&self) -> S {
        S::HALF * self.c_sc * self.v * self.v
    }

    /// Maximum voltage rating \[V\].
    #[inline]
    pub fn v_max(&self) -> S {
        self.v_max
    }

    /// Capacitance \[F\].
    #[inline]
    pub fn capacitance(&self) -> S {
        self.c_sc
    }
}

// ---------------------------------------------------------------------------
// Simplified first-order battery (private, used only by HybridEnergyStorage)
// ---------------------------------------------------------------------------

/// First-order simplified battery model.
///
/// Models SOC via Coulomb counting and terminal voltage via a static IR model.
/// No dynamic RC branch (use [`crate::sim::battery::TheveninBattery`] for that).
///
/// **Dynamics**:
/// ```text
/// soc_dot = −I / (3600 · Q_nom)
/// V_term  = V_nom − R_bat · I
/// ```
///
/// Positive current = discharge.
#[derive(Debug, Clone, Copy)]
struct SimpleBattery<S: ControlScalar> {
    soc: S,
    q_nom: S, // nominal capacity [Ah]
    r_bat: S, // internal resistance [Ω]
    v_nom: S, // nominal voltage [V]
    dt: S,
}

impl<S: ControlScalar> SimpleBattery<S> {
    fn new(q_nom: S, r_bat: S, v_nom: S, soc_init: S, dt: S) -> Result<Self, EnergyStorageError> {
        if q_nom <= S::ZERO || r_bat < S::ZERO || v_nom <= S::ZERO || dt <= S::ZERO {
            return Err(EnergyStorageError::InvalidParameter);
        }
        if soc_init < S::ZERO || soc_init > S::ONE {
            return Err(EnergyStorageError::InvalidParameter);
        }
        Ok(Self {
            soc: soc_init,
            q_nom,
            r_bat,
            v_nom,
            dt,
        })
    }

    /// Step the simplified battery.
    ///
    /// SOC is clamped to \[0, 1\]; no error is returned for SOC limits.
    fn step(&mut self, current: S) -> S {
        let soc_dot = -current / (S::from_f64(3600.0) * self.q_nom);
        self.soc = (self.soc + soc_dot * self.dt).clamp_val(S::ZERO, S::ONE);
        self.v_nom - self.r_bat * current
    }

    #[inline]
    fn soc(&self) -> S {
        self.soc
    }
}

// ---------------------------------------------------------------------------
// Hybrid Energy Storage System
// ---------------------------------------------------------------------------

/// Hybrid Energy Storage System (HESS): battery + supercapacitor.
///
/// ## Power split strategy
///
/// A rule-based split allocates current between the supercapacitor (SC) and
/// battery based on the SC state of charge:
///
/// | SC SoC         | Split ratio α (SC fraction) |
/// |----------------|-----------------------------|
/// | SoC > 0.8      | `α_base × 1.5` (SC-heavy)  |
/// | SoC < 0.2      | `α_base × 0.5` (bat-heavy)  |
/// | otherwise      | `α_base`                    |
///
/// `α` is always clamped to \[0, 1\].
///
/// ```text
/// I_sc  = I_total × α
/// I_bat = I_total × (1 − α)
/// ```
///
/// Positive total current = discharge (power delivered to the load).
///
/// ## Architecture
///
/// ```text
/// Load ←── I_total ──┬── I_bat ──[ SimpleBattery ]
///                    │
///                    └── I_sc  ──[ Supercapacitor ]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HybridEnergyStorage<S: ControlScalar> {
    battery: SimpleBattery<S>,
    sc: Supercapacitor<S>,
    /// Base fraction of total current sent to the supercapacitor \[0, 1\].
    alpha_base: S,
}

impl<S: ControlScalar> HybridEnergyStorage<S> {
    /// Create a new hybrid energy storage system.
    ///
    /// # Parameters
    /// - `q_nom`       — Battery nominal capacity \[Ah\]
    /// - `r_bat`       — Battery internal resistance \[Ω\]
    /// - `v_nom`       — Battery nominal voltage \[V\]
    /// - `bat_soc_init`— Battery initial SOC \[0, 1\]
    /// - `c_sc`        — SC capacitance \[F\]
    /// - `r_sc`        — SC ESR \[Ω\]
    /// - `v_sc_max`    — SC maximum voltage \[V\]
    /// - `v_sc_init`   — SC initial voltage \[V\]
    /// - `alpha_base`  — Base SC power split ratio \[0, 1\]
    /// - `dt`          — Integration step \[s\]
    ///
    /// # Errors
    /// [`EnergyStorageError::InvalidParameter`] if any constraint is violated.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_nom: S,
        r_bat: S,
        v_nom: S,
        bat_soc_init: S,
        c_sc: S,
        r_sc: S,
        v_sc_max: S,
        v_sc_init: S,
        alpha_base: S,
        dt: S,
    ) -> Result<Self, EnergyStorageError> {
        if alpha_base < S::ZERO || alpha_base > S::ONE || dt <= S::ZERO {
            return Err(EnergyStorageError::InvalidParameter);
        }
        let battery = SimpleBattery::new(q_nom, r_bat, v_nom, bat_soc_init, dt)?;
        let sc = Supercapacitor::new(c_sc, r_sc, v_sc_max, v_sc_init, dt)?;
        Ok(Self {
            battery,
            sc,
            alpha_base,
        })
    }

    /// Convenience constructor: small EV-scale hybrid storage.
    ///
    /// - Battery: 50 Ah, 0.05 Ω, 400 V nominal, SOC = 0.8
    /// - SC: 100 F, 0.01 Ω ESR, 48 V max, initial 40 V
    /// - α_base = 0.3, dt = 1 ms
    pub fn default_hybrid() -> Result<Self, EnergyStorageError> {
        Self::new(
            S::from_f64(50.0),
            S::from_f64(0.05),
            S::from_f64(400.0),
            S::from_f64(0.8),
            S::from_f64(100.0),
            S::from_f64(0.01),
            S::from_f64(48.0),
            S::from_f64(40.0),
            S::from_f64(0.3),
            S::from_f64(1e-3),
        )
    }

    /// Advance the hybrid system by one step.
    ///
    /// # Parameters
    /// - `i_total` — Total discharge current \[A\].
    ///
    /// # Returns
    /// `(v_bat, v_sc)` — Battery and SC terminal voltages \[V\].
    ///
    /// # Errors
    /// Propagates SC voltage errors (e.g., [`EnergyStorageError::OverVoltage`]).
    pub fn step(&mut self, i_total: S) -> Result<(S, S), EnergyStorageError> {
        let (i_sc, i_bat) = self.split_current(i_total);
        let v_bat = self.battery.step(i_bat);
        let v_sc = self.sc.step(i_sc)?;
        Ok((v_bat, v_sc))
    }

    /// Compute the current split between SC and battery.
    ///
    /// Returns `(i_sc, i_bat)` such that `i_sc + i_bat = i_total`.
    pub fn split_current(&self, i_total: S) -> (S, S) {
        let sc_soc = self.sc.soc();
        let alpha = if sc_soc > S::from_f64(0.8) {
            self.alpha_base * S::from_f64(1.5)
        } else if sc_soc < S::from_f64(0.2) {
            self.alpha_base * S::from_f64(0.5)
        } else {
            self.alpha_base
        };
        let alpha_clamped = alpha.clamp_val(S::ZERO, S::ONE);
        let i_sc = i_total * alpha_clamped;
        let i_bat = i_total * (S::ONE - alpha_clamped);
        (i_sc, i_bat)
    }

    /// Battery state of charge \[0, 1\].
    #[inline]
    pub fn battery_soc(&self) -> S {
        self.battery.soc()
    }

    /// Supercapacitor state of charge \[0, 1\].
    #[inline]
    pub fn sc_soc(&self) -> S {
        self.sc.soc()
    }

    /// Supercapacitor stored energy \[J\].
    #[inline]
    pub fn sc_energy(&self) -> S {
        self.sc.energy()
    }

    /// Supercapacitor internal voltage \[V\].
    #[inline]
    pub fn sc_voltage(&self) -> S {
        self.sc.voltage()
    }

    /// Base SC split ratio α_base.
    #[inline]
    pub fn alpha_base(&self) -> S {
        self.alpha_base
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Supercapacitor tests
    // ------------------------------------------------------------------

    #[test]
    fn sc_charges_and_discharges() {
        // C=1F, R=0 (ideal SC), v_max=20V, v_init=10V, dt=0.1s
        let mut sc =
            Supercapacitor::<f64>::new(1.0, 0.0, 20.0, 10.0, 0.1).expect("SC construction");

        // Charge: positive current increases voltage
        for _ in 0..5 {
            sc.step(5.0).expect("charge step");
        }
        let v_after_charge = sc.voltage();
        assert!(
            v_after_charge > 10.0,
            "Voltage should increase during charging, got {v_after_charge}"
        );

        // Discharge: negative current decreases voltage
        let v_before_discharge = sc.voltage();
        for _ in 0..5 {
            sc.step(-5.0).expect("discharge step");
        }
        let v_after_discharge = sc.voltage();
        assert!(
            v_after_discharge < v_before_discharge,
            "Voltage should decrease during discharging: {v_after_discharge} < {v_before_discharge}"
        );
    }

    #[test]
    fn sc_energy_correct() {
        // C=1F, V=10V → E = 0.5 * 1 * 100 = 50 J
        let sc = Supercapacitor::<f64>::new(1.0, 0.0, 20.0, 10.0, 0.1).expect("SC construction");
        let energy = sc.energy();
        assert!(
            (energy - 50.0).abs() < 1e-10,
            "Energy should be 50 J, got {energy}"
        );
    }

    #[test]
    fn sc_over_voltage_detected() {
        // C=1F, v_max=12V, v_init=11.9V, dt=0.1s, R=0 (ideal SC)
        // Each step with 5A: dV = 5/1 * 0.1 = 0.5V → 12.4V > 12V → OverVoltage
        let mut sc =
            Supercapacitor::<f64>::new(1.0, 0.0, 12.0, 11.9, 0.1).expect("SC construction");

        // Each step with 5A charges: dV = 5/1 * 0.1 = 0.5V → 12.4V > 12V → error
        let result = sc.step(5.0);
        assert!(
            matches!(result, Err(EnergyStorageError::OverVoltage)),
            "Expected OverVoltage, got {result:?}"
        );
    }

    #[test]
    fn sc_soc_in_range() {
        let sc = Supercapacitor::<f64>::new(100.0, 0.01, 48.0, 40.0, 1e-3).expect("SC");
        let soc = sc.soc();
        assert!((0.0..=1.0).contains(&soc), "SoC {soc} must be in [0, 1]");
        // 40/48 ≈ 0.833
        assert!(
            (soc - 40.0 / 48.0).abs() < 1e-10,
            "Expected SoC ≈ {:.4}, got {soc:.4}",
            40.0 / 48.0
        );
    }

    // ------------------------------------------------------------------
    // Hybrid Energy Storage tests
    // ------------------------------------------------------------------

    #[test]
    fn hybrid_step_returns_two_voltages() {
        let mut hess = HybridEnergyStorage::<f64>::default_hybrid().expect("default_hybrid");
        let result = hess.step(10.0);
        assert!(result.is_ok(), "Hybrid step should succeed, got {result:?}");
        let (v_bat, v_sc) = result.unwrap();
        assert!(
            v_bat > 0.0,
            "Battery voltage should be positive, got {v_bat}"
        );
        assert!(v_sc > 0.0, "SC voltage should be positive, got {v_sc}");
    }

    #[test]
    fn split_ratio_shifts_with_sc_soc() {
        // alpha_base = 0.4
        // High SC SoC (> 0.8): alpha = 0.4 * 1.5 = 0.6
        // Low SC SoC (< 0.2):  alpha = 0.4 * 0.5 = 0.2

        // Create HESS with SC at high SoC (v_init = v_max → SoC = 1.0 > 0.8)
        let hess_high = HybridEnergyStorage::<f64>::new(
            50.0, 0.05, 400.0, 0.8, 100.0, 0.01, 48.0,
            48.0, // v_sc_init = v_max → SoC = 1.0 > 0.8
            0.4,  // alpha_base
            1e-3,
        )
        .expect("high-SoC HESS");

        let i_total = 100.0_f64;
        let (i_sc_high, i_bat_high) = hess_high.split_current(i_total);
        // alpha = 0.4 * 1.5 = 0.6 → i_sc = 60, i_bat = 40
        assert!(
            (i_sc_high - 60.0).abs() < 1e-10,
            "i_sc at high SoC should be 60, got {i_sc_high}"
        );
        assert!(
            (i_bat_high - 40.0).abs() < 1e-10,
            "i_bat at high SoC should be 40, got {i_bat_high}"
        );
        // Verify sum
        assert!(
            (i_sc_high + i_bat_high - i_total).abs() < 1e-10,
            "Currents must sum to i_total"
        );

        // Create HESS with SC at low SoC (v_init ≈ 0 → SoC = 0 < 0.2)
        let hess_low = HybridEnergyStorage::<f64>::new(
            50.0, 0.05, 400.0, 0.8, 100.0, 0.01, 48.0,
            1.0, // v_sc_init very low → SoC ≈ 1/48 ≈ 0.02 < 0.2
            0.4, // alpha_base
            1e-3,
        )
        .expect("low-SoC HESS");

        let (i_sc_low, i_bat_low) = hess_low.split_current(i_total);
        // alpha = 0.4 * 0.5 = 0.2 → i_sc = 20, i_bat = 80
        assert!(
            (i_sc_low - 20.0).abs() < 1e-10,
            "i_sc at low SoC should be 20, got {i_sc_low}"
        );
        assert!(
            (i_bat_low - 80.0).abs() < 1e-10,
            "i_bat at low SoC should be 80, got {i_bat_low}"
        );
        assert!(
            (i_sc_low + i_bat_low - i_total).abs() < 1e-10,
            "Currents must sum to i_total"
        );
    }

    #[test]
    fn hybrid_battery_soc_decreases_on_discharge() {
        let mut hess = HybridEnergyStorage::<f64>::default_hybrid().expect("default_hybrid");
        let initial_bat_soc = hess.battery_soc();
        for _ in 0..100 {
            hess.step(50.0).expect("hybrid step");
        }
        assert!(
            hess.battery_soc() < initial_bat_soc,
            "Battery SoC should decrease during discharge: {} < {}",
            hess.battery_soc(),
            initial_bat_soc
        );
    }

    #[test]
    fn invalid_params_rejected() {
        // alpha_base > 1
        let res = HybridEnergyStorage::<f64>::new(
            50.0, 0.05, 400.0, 0.8, 100.0, 0.01, 48.0, 40.0, 1.5, 1e-3,
        );
        assert!(
            matches!(res, Err(EnergyStorageError::InvalidParameter)),
            "Expected InvalidParameter for alpha_base > 1"
        );

        // dt = 0
        let res = HybridEnergyStorage::<f64>::new(
            50.0, 0.05, 400.0, 0.8, 100.0, 0.01, 48.0, 40.0, 0.3, 0.0,
        );
        assert!(
            matches!(res, Err(EnergyStorageError::InvalidParameter)),
            "Expected InvalidParameter for dt = 0"
        );

        // SC: c_sc = 0
        let res = Supercapacitor::<f64>::new(0.0, 0.01, 48.0, 40.0, 1e-3);
        assert!(
            matches!(res, Err(EnergyStorageError::InvalidParameter)),
            "Expected InvalidParameter for c_sc = 0"
        );
    }

    #[test]
    fn sc_zero_current_stable() {
        // With zero current, SC voltage should remain constant
        let mut sc = Supercapacitor::<f64>::new(10.0, 0.1, 50.0, 25.0, 1e-3).expect("SC");
        let v_init = sc.voltage();
        // With R > 0 and zero current: dV = -V/(R*C) → voltage decays
        // After short time it should just be close to initial (small decay)
        for _ in 0..100 {
            sc.step(0.0).expect("zero current step");
        }
        // Decay: τ = R*C = 0.1*10 = 1s, after 0.1s ≈ 90% of initial
        let v_after = sc.voltage();
        assert!(
            v_after < v_init && v_after > v_init * 0.5,
            "SC voltage should decay slowly with zero current: {v_after} vs init {v_init}"
        );
    }
}
