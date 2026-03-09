use crate::core::scalar::ControlScalar;

/// Errors returned by the PEMFC fuel cell model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FuelCellError {
    /// A parameter was out of its valid range.
    InvalidParameter,
    /// Requested current density exceeds the limiting current density.
    OverCurrentLimit,
    /// Computed terminal voltage is negative (model operating out of range).
    NegativeVoltage,
}

impl core::fmt::Display for FuelCellError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FuelCellError::InvalidParameter => write!(f, "FuelCellError: invalid parameter"),
            FuelCellError::OverCurrentLimit => {
                write!(f, "FuelCellError: current exceeds limiting current")
            }
            FuelCellError::NegativeVoltage => {
                write!(f, "FuelCellError: computed voltage is negative")
            }
        }
    }
}

/// PEMFC (Polymer Electrolyte Membrane Fuel Cell) quasi-static model.
///
/// ## Polarization curve
///
/// The terminal voltage is determined by the Nernst potential minus three
/// overpotential losses:
///
/// ```text
/// V = E_nernst − V_act − V_ohm − V_conc
/// ```
///
/// where:
/// - `V_act  = A · ln(i / i0)`             activation overpotential \[V\]
/// - `V_ohm  = i · R_ohm`                  ohmic (resistive) losses \[V\]
/// - `V_conc = −B · ln(1 − i / i_lim)`    concentration (mass-transport) losses \[V\]
///
/// Here `i` is the **current density** \[A/cm²\] = total current / cell area.
///
/// ## Sign convention
/// All overpotentials are positive quantities that subtract from `E_nernst`.
/// - `V_act` is positive for `i > i0` (since A > 0 and ln(i/i0) > 0).
/// - `V_conc` is positive because `ln(1 − i/i_lim) < 0` for `i ∈ (0, i_lim)`,
///   and the formula negates it: `−B · (negative) = positive`.
///
/// ## Note on Nernst potential
/// `E_nernst` is treated as a precomputed scalar that encodes temperature and
/// partial-pressure conditions. A typical value for standard PEMFC conditions
/// is approximately 1.0–1.23 V per cell.
#[derive(Debug, Clone, Copy)]
pub struct PemFuelCell<S: ControlScalar> {
    /// Tafel slope \[V\] (must be > 0).
    a: S,
    /// Exchange current density \[A/cm²\] (must be > 0, typically very small).
    i0: S,
    /// Ohmic resistance \[Ω\] (area-specific or lumped).
    r_ohm: S,
    /// Mass-transport coefficient \[V\] (must be > 0).
    b: S,
    /// Limiting current density \[A/cm²\] (must be > 0).
    i_lim: S,
    /// Nernst (open-circuit) potential \[V\] (must be > 0).
    e_nernst: S,
    /// Active cell area \[cm²\] (must be > 0).
    area: S,
    /// Last set total current \[A\].
    current: S,
    /// Last computed terminal voltage \[V\].
    voltage: S,
}

impl<S: ControlScalar> PemFuelCell<S> {
    /// Create a new PEMFC model.
    ///
    /// # Parameters
    /// - `a`         — Tafel slope \[V\], must be > 0
    /// - `i0`        — Exchange current density \[A/cm²\], must be > 0
    /// - `r_ohm`     — Ohmic resistance \[Ω\], must be ≥ 0
    /// - `b`         — Mass-transport coefficient \[V\], must be > 0
    /// - `i_lim`     — Limiting current density \[A/cm²\], must be > 0
    /// - `e_nernst`  — Nernst potential \[V\], must be > 0
    /// - `area`      — Active cell area \[cm²\], must be > 0
    ///
    /// # Errors
    /// Returns [`FuelCellError::InvalidParameter`] if any constraint is violated.
    pub fn new(
        a: S,
        i0: S,
        r_ohm: S,
        b: S,
        i_lim: S,
        e_nernst: S,
        area: S,
    ) -> Result<Self, FuelCellError> {
        if a <= S::ZERO
            || i0 <= S::ZERO
            || r_ohm < S::ZERO
            || b <= S::ZERO
            || i_lim <= S::ZERO
            || e_nernst <= S::ZERO
            || area <= S::ZERO
        {
            return Err(FuelCellError::InvalidParameter);
        }
        Ok(Self {
            a,
            i0,
            r_ohm,
            b,
            i_lim,
            e_nernst,
            area,
            current: S::ZERO,
            voltage: e_nernst,
        })
    }

    /// Convenience constructor: typical 100 cm² PEMFC stack cell.
    ///
    /// - A = 0.06 V, i0 = 1×10⁻⁴ A/cm², R_ohm = 0.1 Ω
    /// - B = 0.05 V, i_lim = 1.5 A/cm², E_nernst = 1.0 V
    /// - Area = 100 cm²
    pub fn standard_pemfc() -> Result<Self, FuelCellError> {
        Self::new(
            S::from_f64(0.06),
            S::from_f64(1e-4),
            S::from_f64(0.1),
            S::from_f64(0.05),
            S::from_f64(1.5),
            S::from_f64(1.0),
            S::from_f64(100.0),
        )
    }

    /// Set the operating current and compute the terminal voltage.
    ///
    /// # Parameters
    /// - `i_total` — Total current drawn from the cell \[A\].
    ///
    /// # Returns
    /// Terminal voltage \[V\] on success.
    ///
    /// # Errors
    /// - [`FuelCellError::OverCurrentLimit`] — current density ≥ `i_lim`.
    /// - [`FuelCellError::NegativeVoltage`]  — polarization losses exceed `E_nernst`.
    pub fn set_current(&mut self, i_total: S) -> Result<S, FuelCellError> {
        let i_density = i_total / self.area;

        if i_density >= self.i_lim {
            return Err(FuelCellError::OverCurrentLimit);
        }

        // Clamp to i0 to avoid ln(0) or ln(negative) at near-zero current
        let i_eff = if i_density < self.i0 {
            self.i0
        } else {
            i_density
        };

        // Activation overpotential: A * ln(i / i0)
        // ControlScalar: Float (num_traits) provides .ln()
        let v_act = self.a * (i_eff / self.i0).ln();

        // Ohmic overpotential: i * R_ohm
        let v_ohm = i_eff * self.r_ohm;

        // Concentration overpotential: -B * ln(1 - i/i_lim)
        // Since i_eff < i_lim, (1 - i_eff/i_lim) ∈ (0, 1] → ln is ≤ 0 → negating gives ≥ 0
        let ratio = S::ONE - i_eff / self.i_lim;
        // Guard against ratio ≤ 0 (numerical edge case)
        let v_conc = if ratio > S::EPSILON {
            -self.b * ratio.ln()
        } else {
            // Concentration loss is very large; cap at a large but finite value
            self.b * S::from_f64(20.0)
        };

        let voltage = self.e_nernst - v_act - v_ohm - v_conc;

        if voltage < S::ZERO {
            return Err(FuelCellError::NegativeVoltage);
        }

        self.current = i_total;
        self.voltage = voltage;
        Ok(voltage)
    }

    /// Last computed terminal voltage \[V\].
    #[inline]
    pub fn voltage(&self) -> S {
        self.voltage
    }

    /// Last set total current \[A\].
    #[inline]
    pub fn current(&self) -> S {
        self.current
    }

    /// Electrical power output \[W\] = I · V.
    #[inline]
    pub fn power(&self) -> S {
        self.current * self.voltage
    }

    /// Thermodynamic efficiency = V / E_nernst.
    ///
    /// Returns 0 if `E_nernst` is negligibly small (degenerate case).
    pub fn efficiency(&self) -> S {
        if self.e_nernst.abs() < S::EPSILON {
            S::ZERO
        } else {
            self.voltage / self.e_nernst
        }
    }

    /// Nernst (open-circuit) potential \[V\].
    #[inline]
    pub fn e_nernst(&self) -> S {
        self.e_nernst
    }

    /// Limiting current density \[A/cm²\].
    #[inline]
    pub fn i_lim(&self) -> S {
        self.i_lim
    }

    /// Active cell area \[cm²\].
    #[inline]
    pub fn area(&self) -> S {
        self.area
    }

    /// Maximum total current before concentration losses become unbounded \[A\].
    ///
    /// Returns `i_lim × area`.
    #[inline]
    pub fn max_current(&self) -> S {
        self.i_lim * self.area
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fc() -> PemFuelCell<f64> {
        PemFuelCell::standard_pemfc().expect("standard_pemfc construction")
    }

    #[test]
    fn open_circuit_near_nernst() {
        let mut fc = make_fc();
        // At i_density = i0, V_act = A*ln(1) = 0
        // V_ohm = i0 * R_ohm ≈ 1e-4 * 0.1 = 1e-5 (tiny)
        // V_conc = -B * ln(1 - 1e-4/1.5) ≈ tiny
        // → voltage ≈ e_nernst
        let i_total = 1e-4 * 100.0; // i_density = i0
        let v = fc.set_current(i_total).expect("set_current at i0");
        assert!(
            (v - fc.e_nernst()).abs() < 0.01,
            "At i=i0, voltage {v:.4} should be near e_nernst {:.4}",
            fc.e_nernst()
        );
    }

    #[test]
    fn voltage_decreases_with_current() {
        let mut fc = make_fc();
        let v_low = fc.set_current(100.0).expect("low current"); // 1 A/cm²
        let v_high = fc.set_current(50.0).expect("high current"); // 0.5 A/cm²
                                                                  // NOTE: 100 A total / 100 cm² = 1 A/cm², 50/100 = 0.5 A/cm²
                                                                  // Higher current → more losses → lower voltage
                                                                  // v_low used 100 A, v_high used 50 A
                                                                  // Re-read: we need v at 100A < v at 50A
        let mut fc2 = make_fc();
        let v_50a = fc2.set_current(50.0).expect("50A");
        let mut fc3 = make_fc();
        let v_100a = fc3.set_current(100.0).expect("100A");
        assert!(
            v_100a < v_50a,
            "Voltage at 100A ({v_100a:.4}) should be less than at 50A ({v_50a:.4})"
        );
        let _ = v_low;
        let _ = v_high;
    }

    #[test]
    fn over_limit_rejected() {
        let mut fc = make_fc();
        // i_lim = 1.5 A/cm², area = 100 cm² → max_current = 150 A
        let res = fc.set_current(151.0);
        assert!(
            matches!(res, Err(FuelCellError::OverCurrentLimit)),
            "Expected OverCurrentLimit, got {res:?}"
        );
    }

    #[test]
    fn power_equals_i_times_v() {
        let mut fc = make_fc();
        let v = fc.set_current(50.0).expect("set_current");
        let power = fc.power();
        let expected = fc.current() * v;
        assert!(
            (power - expected).abs() < 1e-10,
            "power={power} should equal i*v={expected}"
        );
    }

    #[test]
    fn efficiency_less_than_one() {
        let mut fc = make_fc();
        fc.set_current(50.0).expect("set_current");
        let eff = fc.efficiency();
        assert!(
            eff < 1.0 && eff > 0.0,
            "Efficiency {eff} should be in (0, 1)"
        );
    }

    #[test]
    fn invalid_params_rejected() {
        // a = 0
        let res = PemFuelCell::<f64>::new(0.0, 1e-4, 0.1, 0.05, 1.5, 1.0, 100.0);
        assert!(matches!(res, Err(FuelCellError::InvalidParameter)));

        // i_lim = 0
        let res = PemFuelCell::<f64>::new(0.06, 1e-4, 0.1, 0.05, 0.0, 1.0, 100.0);
        assert!(matches!(res, Err(FuelCellError::InvalidParameter)));

        // area negative
        let res = PemFuelCell::<f64>::new(0.06, 1e-4, 0.1, 0.05, 1.5, 1.0, -5.0);
        assert!(matches!(res, Err(FuelCellError::InvalidParameter)));

        // e_nernst = 0
        let res = PemFuelCell::<f64>::new(0.06, 1e-4, 0.1, 0.05, 1.5, 0.0, 100.0);
        assert!(matches!(res, Err(FuelCellError::InvalidParameter)));

        // r_ohm negative
        let res = PemFuelCell::<f64>::new(0.06, 1e-4, -0.1, 0.05, 1.5, 1.0, 100.0);
        assert!(matches!(res, Err(FuelCellError::InvalidParameter)));
    }

    #[test]
    fn max_current_equals_i_lim_times_area() {
        let fc = make_fc();
        let expected = fc.i_lim() * fc.area();
        assert!(
            (fc.max_current() - expected).abs() < 1e-10,
            "max_current={} expected={}",
            fc.max_current(),
            expected
        );
    }

    #[test]
    fn zero_current_gives_nernst_voltage() {
        let mut fc = make_fc();
        // i_total = 0 → clamp to i0, v_act = 0, v_ohm tiny, v_conc tiny
        let v = fc.set_current(0.0).expect("zero current");
        // Should be very close to e_nernst
        assert!(
            (v - fc.e_nernst()).abs() < 0.05,
            "At zero current, voltage {v:.4} should be within 0.05 V of e_nernst {:.4}",
            fc.e_nernst()
        );
    }
}
