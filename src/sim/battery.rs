use crate::core::scalar::ControlScalar;

/// Errors returned by the Thevenin battery model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatteryError {
    /// A parameter was out of its valid range.
    InvalidParameter,
    /// SOC exceeded 1.0 (battery overcharged).
    Overcharged,
    /// SOC fell below 0.0 (battery over-discharged).
    Overdischarged,
}

impl core::fmt::Display for BatteryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BatteryError::InvalidParameter => write!(f, "BatteryError: invalid parameter"),
            BatteryError::Overcharged => write!(f, "BatteryError: overcharged (SOC > 1)"),
            BatteryError::Overdischarged => write!(f, "BatteryError: overdischarged (SOC < 0)"),
        }
    }
}

/// Thevenin Equivalent Battery Model (RC circuit model).
///
/// ## Circuit topology
///
/// ```text
///   +--R0--+--R1--+
///   |      |      |
///  V_oc   ---    C1
///   |     ---     |
///   +------+------+
///                 |
///              V_term
/// ```
///
/// **State vector**: `[soc, v_rc]`
/// - `soc`  — State of Charge ∈ \[0, 1\]
/// - `v_rc` — Voltage across the RC branch (polarization voltage) \[V\]
///
/// **Dynamics** (Euler integration):
/// - `soc_dot  = -I / (3600 · Q_nom)`       (Coulomb counting, Q_nom in Ah)
/// - `v_rc_dot = -v_rc / (R1·C1) + I / C1`
///
/// **Terminal voltage**:
/// - `V_oc(soc) = V_min + (V_max − V_min) · soc`  (linear OCV approximation)
/// - `V_term = V_oc(soc) − R0·I − v_rc`
///
/// Positive current `I` means **discharging** (current flows out of the battery).
#[derive(Debug, Clone, Copy)]
pub struct TheveninBattery<S: ControlScalar> {
    /// State: \[soc, v_rc\]
    state: [S; 2],
    /// Nominal capacity \[Ah\].
    q_nom: S,
    /// Series (internal) resistance \[Ω\].
    r0: S,
    /// RC-branch resistance \[Ω\].
    r1: S,
    /// RC-branch capacitance \[F\].
    c1: S,
    /// Open-circuit voltage at SOC = 0 \[V\].
    v_min: S,
    /// Open-circuit voltage at SOC = 1 \[V\].
    v_max: S,
    /// Euler integration step \[s\].
    dt: S,
    /// Cumulative absolute charge throughput \[Ah\] across all step() calls.
    cumulative_throughput_ah: S,
    /// Total throughput at end-of-life \[Ah\]; default = 500 × q_nom.
    lifetime_throughput_ah: S,
}

impl<S: ControlScalar> TheveninBattery<S> {
    /// Create a new Thevenin battery model.
    ///
    /// # Parameters
    /// - `q_nom`    — Nominal capacity \[Ah\], must be > 0
    /// - `r0`       — Series resistance \[Ω\], must be > 0
    /// - `r1`       — RC-branch resistance \[Ω\], must be > 0
    /// - `c1`       — RC-branch capacitance \[F\], must be > 0
    /// - `v_min`    — OCV at SOC = 0 \[V\]
    /// - `v_max`    — OCV at SOC = 1 \[V\], must be > v_min
    /// - `soc_init` — Initial state of charge \[0, 1\]
    /// - `dt`       — Integration time step \[s\], must be > 0
    ///
    /// # Errors
    /// Returns [`BatteryError::InvalidParameter`] if any constraint is violated.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_nom: S,
        r0: S,
        r1: S,
        c1: S,
        v_min: S,
        v_max: S,
        soc_init: S,
        dt: S,
    ) -> Result<Self, BatteryError> {
        if q_nom <= S::ZERO || r0 <= S::ZERO || r1 <= S::ZERO || c1 <= S::ZERO || dt <= S::ZERO {
            return Err(BatteryError::InvalidParameter);
        }
        if v_min >= v_max {
            return Err(BatteryError::InvalidParameter);
        }
        if soc_init < S::ZERO || soc_init > S::ONE {
            return Err(BatteryError::InvalidParameter);
        }
        Ok(Self {
            state: [soc_init, S::ZERO],
            q_nom,
            r0,
            r1,
            c1,
            v_min,
            v_max,
            dt,
            cumulative_throughput_ah: S::ZERO,
            lifetime_throughput_ah: q_nom * S::from_f64(500.0),
        })
    }

    /// Convenience constructor: typical 18650 Li-ion cell.
    ///
    /// - Q_nom = 2.5 Ah, R0 = 0.05 Ω, R1 = 0.02 Ω, C1 = 2000 F
    /// - V_min = 3.0 V, V_max = 4.2 V, SOC_init = 1.0, dt = 1 ms
    pub fn default_18650() -> Result<Self, BatteryError> {
        Self::new(
            S::from_f64(2.5),
            S::from_f64(0.05),
            S::from_f64(0.02),
            S::from_f64(2000.0),
            S::from_f64(3.0),
            S::from_f64(4.2),
            S::ONE,
            S::from_f64(1e-3),
        )
    }

    /// Override the lifetime throughput used for SOH calculation \[Ah\].
    ///
    /// Default is 500 × q_nom (≈500 full equivalent cycles for typical Li-ion).
    ///
    /// # Errors
    /// Returns [`BatteryError::InvalidParameter`] if `ah` is not positive.
    pub fn with_lifetime_throughput(mut self, ah: S) -> Result<Self, BatteryError> {
        if ah <= S::ZERO {
            return Err(BatteryError::InvalidParameter);
        }
        self.lifetime_throughput_ah = ah;
        Ok(self)
    }

    /// Advance the simulation by one time step.
    ///
    /// # Parameters
    /// - `current` — Applied current \[A\].
    ///   Positive = **discharge** (conventional positive current out of the terminal).
    ///
    /// # Returns
    /// Terminal voltage `V_term` \[V\] on success.
    ///
    /// # Errors
    /// - [`BatteryError::Overcharged`]   — if SOC exceeds 1.0 after the step.
    /// - [`BatteryError::Overdischarged`] — if SOC falls below 0.0 after the step.
    pub fn step(&mut self, current: S) -> Result<S, BatteryError> {
        let soc = self.state[0];
        let v_rc = self.state[1];

        // Open-circuit voltage (linear OCV model)
        let v_oc = self.v_min + (self.v_max - self.v_min) * soc;

        // Terminal voltage (computed before state update for zero-order hold consistency)
        let v_term = v_oc - self.r0 * current - v_rc;

        // State derivatives
        let soc_dot = -current / (S::from_f64(3600.0) * self.q_nom);

        // Guard against degenerate R1*C1 product
        let tau_rc = self.r1 * self.c1;
        let v_rc_dot = if tau_rc.abs() > S::EPSILON {
            -v_rc / tau_rc + current / self.c1
        } else {
            current / self.c1
        };

        // Euler integration
        self.state[0] = soc + soc_dot * self.dt;
        self.state[1] = v_rc + v_rc_dot * self.dt;

        // Track cumulative charge throughput for SOH model (|I| * dt / 3600 = Ah).
        self.cumulative_throughput_ah += current.abs() * self.dt / S::from_f64(3600.0);

        // Bounds check on SOC
        if self.state[0] > S::ONE {
            return Err(BatteryError::Overcharged);
        }
        if self.state[0] < S::ZERO {
            return Err(BatteryError::Overdischarged);
        }

        Ok(v_term)
    }

    /// Current state of charge \[0, 1\].
    #[inline]
    pub fn soc(&self) -> S {
        self.state[0]
    }

    /// RC-branch polarization voltage \[V\].
    #[inline]
    pub fn v_rc(&self) -> S {
        self.state[1]
    }

    /// Open-circuit voltage at the current SOC \[V\].
    #[inline]
    pub fn open_circuit_voltage(&self) -> S {
        self.v_min + (self.v_max - self.v_min) * self.state[0]
    }

    /// Terminal voltage at zero current load \[V\].
    ///
    /// Equals `V_oc(soc) − v_rc` (no R0 drop when I = 0).
    #[inline]
    pub fn terminal_voltage(&self) -> S {
        self.open_circuit_voltage() - self.state[1]
    }

    /// State of health based on cumulative charge throughput.
    ///
    /// Uses a linear capacity-fade model: SOH = 1 − (throughput / lifetime_throughput).
    /// Returns 1.0 for a new battery, 0.0 after `lifetime_throughput_ah` of usage.
    /// Clamped to \[0, 1\].
    #[inline]
    pub fn state_of_health(&self) -> S {
        let frac = self.cumulative_throughput_ah / self.lifetime_throughput_ah;
        let soh = S::ONE - frac;
        if soh < S::ZERO {
            S::ZERO
        } else if soh > S::ONE {
            S::ONE
        } else {
            soh
        }
    }

    /// Cumulative absolute charge throughput \[Ah\].
    #[inline]
    pub fn cumulative_throughput_ah(&self) -> S {
        self.cumulative_throughput_ah
    }

    /// Reset the battery to a given initial SOC.
    ///
    /// Clears the RC polarization voltage.
    ///
    /// # Errors
    /// [`BatteryError::InvalidParameter`] if `soc_init` is not in \[0, 1\].
    pub fn reset(&mut self, soc_init: S) -> Result<(), BatteryError> {
        if soc_init < S::ZERO || soc_init > S::ONE {
            return Err(BatteryError::InvalidParameter);
        }
        self.state = [soc_init, S::ZERO];
        Ok(())
    }

    /// Nominal capacity \[Ah\].
    #[inline]
    pub fn q_nom(&self) -> S {
        self.q_nom
    }

    /// Series internal resistance \[Ω\].
    #[inline]
    pub fn r0(&self) -> S {
        self.r0
    }

    /// Estimated remaining energy \[J\] based on linear OCV model.
    ///
    /// `E ≈ Q_nom · 3600 · (V_min + (V_max − V_min) · soc / 2) · soc`
    pub fn remaining_energy_joules(&self) -> S {
        let soc = self.state[0];
        let v_avg = self.v_min + (self.v_max - self.v_min) * soc * S::HALF;
        self.q_nom * S::from_f64(3600.0) * v_avg * soc
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a standard test battery (SOC=1, dt=1ms).
    fn make_battery(soc_init: f64) -> TheveninBattery<f64> {
        TheveninBattery::new(2.5, 0.05, 0.02, 2000.0, 3.0, 4.2, soc_init, 1e-3).unwrap()
    }

    #[test]
    fn zero_current_constant_soc() {
        let mut bat = make_battery(0.8);
        let initial_soc = bat.soc();
        for _ in 0..100 {
            let v = bat.step(0.0).expect("step should not fail at zero current");
            // With zero current: soc_dot = 0, v_rc_dot = 0 → SOC is constant
            assert!(
                (bat.soc() - initial_soc).abs() < 1e-12,
                "SOC changed: {} vs {}",
                bat.soc(),
                initial_soc
            );
            // Terminal voltage = OCV (no IR drop)
            let v_oc = bat.open_circuit_voltage();
            assert!(
                (v - v_oc).abs() < 1e-10,
                "V_term={v} should equal V_oc={v_oc} at zero current"
            );
        }
    }

    #[test]
    fn discharge_reduces_soc() {
        let mut bat = make_battery(1.0);
        // Discharge at 1 A for 360 steps × 1 ms = 0.36 s
        for _ in 0..360 {
            bat.step(1.0).expect("discharge step failed");
        }
        assert!(
            bat.soc() < 1.0,
            "SOC should decrease during discharge, got {}",
            bat.soc()
        );
    }

    #[test]
    fn rc_dynamics_decay_toward_zero() {
        // Charge the RC branch by discharging for a while, then apply zero current.
        let mut bat = make_battery(0.9);

        // Build up v_rc by discharging
        for _ in 0..200 {
            bat.step(5.0).expect("discharge step");
        }
        let v_rc_charged = bat.v_rc().abs();
        assert!(
            v_rc_charged > 1e-6,
            "v_rc should be non-zero after discharge"
        );

        // Now let it decay with zero current
        // Time constant τ = R1*C1 = 0.02 * 2000 = 40 s → need many steps
        // After 5τ ≈ 200 s = 200_000 steps of 1ms to reach ~1% of initial
        // Test for 40 s (1 time constant) → decay to ~37%
        for _ in 0..40_000 {
            bat.step(0.0).expect("idle step");
        }
        let v_rc_after = bat.v_rc().abs();
        // Should have decayed significantly (less than 50% of initial charged value)
        assert!(
            v_rc_after < v_rc_charged * 0.5,
            "v_rc={v_rc_after:.6} should be < 50% of {v_rc_charged:.6} after 1 time constant"
        );
    }

    #[test]
    fn terminal_voltage_drops_with_load() {
        let mut bat_idle = make_battery(0.8);
        let mut bat_load = make_battery(0.8);

        let v_idle = bat_idle.step(0.0).expect("idle step");
        let v_load = bat_load.step(2.0).expect("load step");

        assert!(
            v_load < v_idle,
            "Terminal voltage under load ({v_load:.4}) should be less than at no load ({v_idle:.4})"
        );
    }

    #[test]
    fn over_discharge_detected() {
        // Use tiny capacity so it drains quickly
        let mut bat = TheveninBattery::<f64>::new(
            0.001, // 1 mAh
            0.05, 0.02, 2000.0, 3.0, 4.2, 0.01, // Start at 1% SOC
            0.1,  // dt = 100 ms
        )
        .expect("battery construction");

        // Apply large current → SOC should hit 0 quickly
        let mut got_error = false;
        for _ in 0..10_000 {
            match bat.step(100.0) {
                Err(BatteryError::Overdischarged) => {
                    got_error = true;
                    break;
                }
                Err(e) => panic!("Unexpected error: {e:?}"),
                Ok(_) => {}
            }
        }
        assert!(got_error, "Expected Overdischarged error but none occurred");
    }

    #[test]
    fn invalid_params_rejected() {
        // r0 = 0 → InvalidParameter
        let res = TheveninBattery::<f64>::new(2.5, 0.0, 0.02, 2000.0, 3.0, 4.2, 1.0, 1e-3);
        assert!(
            matches!(res, Err(BatteryError::InvalidParameter)),
            "Expected InvalidParameter for r0=0"
        );

        // q_nom negative → InvalidParameter
        let res = TheveninBattery::<f64>::new(-1.0, 0.05, 0.02, 2000.0, 3.0, 4.2, 1.0, 1e-3);
        assert!(
            matches!(res, Err(BatteryError::InvalidParameter)),
            "Expected InvalidParameter for q_nom<0"
        );

        // v_min >= v_max → InvalidParameter
        let res = TheveninBattery::<f64>::new(2.5, 0.05, 0.02, 2000.0, 4.2, 3.0, 1.0, 1e-3);
        assert!(
            matches!(res, Err(BatteryError::InvalidParameter)),
            "Expected InvalidParameter for v_min >= v_max"
        );

        // soc_init > 1 → InvalidParameter
        let res = TheveninBattery::<f64>::new(2.5, 0.05, 0.02, 2000.0, 3.0, 4.2, 1.5, 1e-3);
        assert!(
            matches!(res, Err(BatteryError::InvalidParameter)),
            "Expected InvalidParameter for soc_init > 1"
        );

        // dt = 0 → InvalidParameter
        let res = TheveninBattery::<f64>::new(2.5, 0.05, 0.02, 2000.0, 3.0, 4.2, 1.0, 0.0);
        assert!(
            matches!(res, Err(BatteryError::InvalidParameter)),
            "Expected InvalidParameter for dt=0"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut bat = make_battery(1.0);
        // Discharge partway
        for _ in 0..1000 {
            if bat.step(1.0).is_err() {
                break;
            }
        }
        let soc_after_discharge = bat.soc();
        assert!(soc_after_discharge < 1.0);

        // Reset to 0.5
        bat.reset(0.5).expect("reset should succeed");
        assert!(
            (bat.soc() - 0.5).abs() < 1e-12,
            "SOC after reset should be 0.5, got {}",
            bat.soc()
        );
        assert!(
            bat.v_rc().abs() < 1e-12,
            "v_rc should be 0 after reset, got {}",
            bat.v_rc()
        );
    }

    #[test]
    fn default_18650_constructs() {
        let bat = TheveninBattery::<f64>::default_18650().expect("default_18650 should succeed");
        assert!((bat.soc() - 1.0).abs() < 1e-12);
        assert_eq!(bat.state_of_health(), 1.0);
    }

    #[test]
    fn ocv_linear_interpolation() {
        // At SOC=0 → V_oc = V_min; at SOC=1 → V_oc = V_max; at SOC=0.5 → midpoint
        let bat = TheveninBattery::<f64>::new(2.5, 0.05, 0.02, 2000.0, 3.0, 4.2, 0.5, 1e-3)
            .expect("construct");
        let v_oc = bat.open_circuit_voltage();
        assert!(
            (v_oc - 3.6).abs() < 1e-10,
            "OCV at SOC=0.5 should be 3.6 V, got {v_oc}"
        );
    }

    #[test]
    fn soh_starts_at_one_for_new_battery() {
        let bat = make_battery(1.0);
        assert_eq!(
            bat.state_of_health(),
            1.0_f64,
            "Fresh battery SOH should be exactly 1.0"
        );
    }

    #[test]
    fn soh_decreases_with_throughput() {
        // Run 100 steps and record SOH
        let mut bat_100 = make_battery(1.0);
        for _ in 0..100 {
            bat_100.step(1.0).expect("step failed");
        }
        let soh_at_100 = bat_100.state_of_health();

        // Run 1000 steps (same battery fresh)
        let mut bat_1000 = make_battery(1.0);
        for _ in 0..1000 {
            bat_1000.step(1.0).expect("step failed");
        }
        let soh_at_1000 = bat_1000.state_of_health();

        assert!(
            soh_at_1000 < 1.0_f64,
            "SOH should be less than 1.0 after 1000 discharge steps, got {soh_at_1000}"
        );
        assert!(
            soh_at_100 > soh_at_1000,
            "SOH after 100 steps ({soh_at_100}) should be greater than after 1000 steps ({soh_at_1000})"
        );
    }

    #[test]
    fn soh_clamped_to_zero_after_lifetime() {
        // Tiny lifetime (0.001 Ah) so it is exhausted quickly
        let mut bat = TheveninBattery::<f64>::new(2.5, 0.05, 0.02, 2000.0, 3.0, 4.2, 1.0, 1e-3)
            .expect("construct")
            .with_lifetime_throughput(0.001)
            .expect("set lifetime throughput");

        // Step with high current until lifetime is exceeded (or overdischarged)
        for _ in 0..100_000 {
            match bat.step(10.0) {
                Ok(_) => {}
                Err(BatteryError::Overdischarged) => break,
                Err(e) => panic!("Unexpected error: {e:?}"),
            }
        }

        let soh = bat.state_of_health();
        assert_eq!(
            soh, 0.0_f64,
            "SOH should be clamped to 0.0 after exceeding lifetime throughput, got {soh}"
        );
    }
}
