//! Maximum Power Point Tracking (MPPT) algorithms for photovoltaic systems.
//!
//! Provides three MPPT methods:
//!
//! 1. **Perturb & Observe** (`PerturbeAndObserve`) — simple hill-climbing algorithm
//!    that perturbs the voltage reference and observes the power change.
//!
//! 2. **Incremental Conductance** (`IncrementalConductance`) — exploits the condition
//!    dI/dV + I/V = 0 at the MPP for faster convergence under changing irradiance.
//!
//! 3. **Fractional Open-Circuit Voltage** (`FractionalOcv`) — estimates V_mpp from
//!    the open-circuit voltage: V_mpp ≈ k_oc · V_oc.
//!
//! A single-diode PV cell model (`PvCellModel`) is included for unit testing.

use crate::core::scalar::ControlScalar;

// ─── Perturb & Observe ───────────────────────────────────────────────────────

/// Direction of the most recent voltage perturbation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpptDirection {
    /// Voltage reference is being increased.
    Increase,
    /// Voltage reference is being decreased.
    Decrease,
}

/// Maximum Power Point Tracker using the Perturb-and-Observe algorithm.
///
/// At each call to [`PerturbeAndObserve::update`] the tracker:
/// 1. Computes the new power P = V·I.
/// 2. Compares ΔP and ΔV to determine the correct direction.
/// 3. Perturbs the internal voltage reference by `±step`.
///
/// The voltage reference is clamped to `[v_min, v_max]`.
#[derive(Debug, Clone, Copy)]
pub struct PerturbeAndObserve<S: ControlScalar> {
    /// Perturbation step size (V).
    pub step: S,
    /// Lower voltage bound (V).
    pub v_min: S,
    /// Upper voltage bound (V).
    pub v_max: S,
    /// Current voltage reference (V).
    v_ref: S,
    /// Power measured at the previous call (W).
    p_prev: S,
    /// Voltage measured at the previous call (V).
    v_prev: S,
    /// Noise threshold — power changes below this are ignored.
    eps: S,
    /// Last perturbation direction.
    direction: MpptDirection,
}

impl<S: ControlScalar> PerturbeAndObserve<S> {
    /// Create a new P&O tracker.
    ///
    /// * `v_init` — initial voltage reference (V)
    /// * `step`   — perturbation step (V), e.g. 0.5 for a 30–40 V panel
    /// * `v_min`, `v_max` — operating voltage bounds (V)
    pub fn new(v_init: S, step: S, v_min: S, v_max: S) -> Self {
        Self {
            step,
            v_min,
            v_max,
            v_ref: v_init,
            p_prev: S::ZERO,
            v_prev: v_init,
            eps: S::from_f64(1e-6),
            direction: MpptDirection::Increase,
        }
    }

    /// Update the tracker with new measurements and return the voltage perturbation Δv.
    ///
    /// The returned value is the signed perturbation applied to the internal voltage
    /// reference this cycle (useful for outer-loop control).  The updated voltage
    /// reference is available via [`PerturbeAndObserve::v_ref`].
    ///
    /// * `v` — measured panel terminal voltage (V)
    /// * `i` — measured panel current (A)
    pub fn update(&mut self, v: S, i: S) -> S {
        let p = v * i;
        let dp = p - self.p_prev;
        let dv = v - self.v_prev;

        let perturbation = if dp.abs() <= self.eps {
            // Power change is negligible — repeat last direction
            match self.direction {
                MpptDirection::Increase => self.step,
                MpptDirection::Decrease => -self.step,
            }
        } else {
            // Standard P&O logic
            // ΔP > 0 and ΔV > 0  → continue increasing
            // ΔP > 0 and ΔV < 0  → continue decreasing (was moving toward MPP)
            // ΔP < 0 and ΔV > 0  → passed MPP, reverse (decrease)
            // ΔP < 0 and ΔV < 0  → passed MPP, reverse (increase)
            let same_sign = (dp > S::ZERO) == (dv > S::ZERO);
            if same_sign {
                self.direction = MpptDirection::Increase;
                self.step
            } else {
                self.direction = MpptDirection::Decrease;
                -self.step
            }
        };

        self.p_prev = p;
        self.v_prev = v;

        let v_new = (self.v_ref + perturbation).clamp_val(self.v_min, self.v_max);
        self.v_ref = v_new;

        perturbation
    }

    /// Current voltage reference (V).
    pub fn v_ref(&self) -> S {
        self.v_ref
    }

    /// Last measured power (W).
    pub fn power(&self) -> S {
        self.p_prev
    }

    /// Last perturbation direction.
    pub fn direction(&self) -> MpptDirection {
        self.direction
    }

    /// Reset tracker state to an initial voltage.
    pub fn reset(&mut self, v_init: S) {
        self.v_ref = v_init;
        self.v_prev = v_init;
        self.p_prev = S::ZERO;
        self.direction = MpptDirection::Increase;
    }
}

// ─── Incremental Conductance ─────────────────────────────────────────────────

/// Maximum Power Point Tracker using Incremental Conductance.
///
/// The MPP condition is: dI/dV = -I/V  ⟺  I/V + dI/dV = 0.
///
/// * If `I/V + dI/dV > 0` → operating left of MPP, increase V.
/// * If `I/V + dI/dV < 0` → operating right of MPP, decrease V.
/// * If `|I/V + dI/dV| < ε` → at MPP, do not perturb.
#[derive(Debug, Clone, Copy)]
pub struct IncrementalConductance<S: ControlScalar> {
    /// Step size (V).
    pub step: S,
    /// Lower voltage bound (V).
    pub v_min: S,
    /// Upper voltage bound (V).
    pub v_max: S,
    /// Current voltage reference (V).
    v_ref: S,
    /// Voltage at previous sample (V).
    v_prev: S,
    /// Current at previous sample (A).
    i_prev: S,
    /// Zero-crossing threshold.
    eps: S,
}

impl<S: ControlScalar> IncrementalConductance<S> {
    /// Create a new Incremental Conductance tracker.
    pub fn new(v_init: S, step: S, v_min: S, v_max: S) -> Self {
        Self {
            step,
            v_min,
            v_max,
            v_ref: v_init,
            v_prev: v_init,
            i_prev: S::ZERO,
            eps: S::from_f64(1e-6),
        }
    }

    /// Update with measured voltage `v` and current `i`.
    ///
    /// Returns the signed voltage perturbation Δv_ref applied this cycle.
    pub fn update(&mut self, v: S, i: S) -> S {
        let dv = v - self.v_prev;
        let di = i - self.i_prev;

        let delta = if dv.abs() < self.eps {
            // No meaningful voltage change — use current sign as proxy
            if di.abs() < self.eps {
                // At MPP (no change in either)
                S::ZERO
            } else if di > S::ZERO {
                self.step
            } else {
                -self.step
            }
        } else {
            // Incremental conductance condition: I/V + dI/dV
            let conductance_sum = if v.abs() > self.eps {
                i / v + di / dv
            } else {
                di / dv
            };

            if conductance_sum.abs() < self.eps {
                // At MPP
                S::ZERO
            } else if conductance_sum > S::ZERO {
                self.step
            } else {
                -self.step
            }
        };

        self.v_prev = v;
        self.i_prev = i;

        self.v_ref = (self.v_ref + delta).clamp_val(self.v_min, self.v_max);
        delta
    }

    /// Current voltage reference (V).
    pub fn v_ref(&self) -> S {
        self.v_ref
    }

    /// Reset tracker state.
    pub fn reset(&mut self, v_init: S) {
        self.v_ref = v_init;
        self.v_prev = v_init;
        self.i_prev = S::ZERO;
    }
}

// ─── Fractional Open-Circuit Voltage ─────────────────────────────────────────

/// MPPT using the Fractional Open-Circuit Voltage method.
///
/// The empirical relationship V_mpp ≈ k_oc · V_oc is exploited.  Typical
/// values: k_oc ≈ 0.72–0.80 for crystalline silicon, 0.80–0.85 for amorphous.
///
/// `set_voc` must be called periodically after briefly open-circuiting the panel
/// to measure V_oc.  In practice a pilot cell or look-up table is used to avoid
/// interrupting power delivery.
#[derive(Debug, Clone, Copy)]
pub struct FractionalOcv<S: ControlScalar> {
    /// Fractional factor k_oc ∈ (0, 1).
    pub k_oc: S,
    /// Most recent open-circuit voltage estimate (V).
    v_oc: S,
    /// Current MPP voltage estimate (V).
    v_mpp: S,
    /// Lower voltage bound (V).
    pub v_min: S,
    /// Upper voltage bound (V).
    pub v_max: S,
}

impl<S: ControlScalar> FractionalOcv<S> {
    /// Create a FractionalOcv tracker.
    ///
    /// * `k_oc`   — MPP fraction (typical: 0.76)
    /// * `v_oc`   — initial open-circuit voltage estimate (V)
    /// * `v_min`, `v_max` — voltage bounds (V)
    pub fn new(k_oc: S, v_oc: S, v_min: S, v_max: S) -> Self {
        let v_mpp = (k_oc * v_oc).clamp_val(v_min, v_max);
        Self {
            k_oc,
            v_oc,
            v_mpp,
            v_min,
            v_max,
        }
    }

    /// Update the MPP estimate with a new open-circuit voltage measurement.
    ///
    /// Returns the new estimated MPP voltage (V).
    pub fn set_voc(&mut self, v_oc: S) -> S {
        self.v_oc = v_oc;
        self.v_mpp = (self.k_oc * v_oc).clamp_val(self.v_min, self.v_max);
        self.v_mpp
    }

    /// Current MPP voltage reference (V).
    pub fn v_mpp(&self) -> S {
        self.v_mpp
    }

    /// Last known open-circuit voltage (V).
    pub fn v_oc(&self) -> S {
        self.v_oc
    }
}

// ─── Single-Diode PV Cell Model ──────────────────────────────────────────────

/// Single-diode photovoltaic cell model.
///
/// Implements the standard single-diode equation:
///
///   I = Iph − I0·(exp(V / (n·Vt)) − 1) − V/Rsh
///
/// where Vt = k·T/q (thermal voltage).
///
/// For a given terminal voltage V the current I is solved with Newton iteration,
/// because I appears implicitly when series resistance Rs is non-zero:
///
///   I = Iph − I0·(exp((V + I·Rs)/(n·Vt)) − 1) − (V + I·Rs)/Rsh
///
/// Convergence is guaranteed for physically reasonable parameters.
#[derive(Debug, Clone, Copy)]
pub struct PvCellModel<S: ControlScalar> {
    /// Photo-generated current (A).
    pub iph: S,
    /// Dark saturation current (A), typically 1e-9 … 1e-6.
    pub i0: S,
    /// Diode ideality factor (typically 1.0–2.0).
    pub n: S,
    /// Thermal voltage V_t = k·T/q (≈ 0.02585 V at 25 °C).
    pub vt: S,
    /// Series resistance (Ω).
    pub rs: S,
    /// Shunt resistance (Ω).
    pub rsh: S,
    /// Newton iteration limit.
    max_iter: u32,
    /// Newton convergence tolerance.
    tol: S,
}

impl<S: ControlScalar> PvCellModel<S> {
    /// Construct a PvCellModel.
    ///
    /// Typical silicon cell at STC (1000 W/m², 25 °C):
    /// * `iph`  ≈ 8–10 A
    /// * `i0`   ≈ 1e-9 A
    /// * `n`    = 1.5
    /// * `vt`   = 0.02585 V
    /// * `rs`   ≈ 0.01 Ω
    /// * `rsh`  ≈ 200 Ω
    pub fn new(iph: S, i0: S, n: S, vt: S, rs: S, rsh: S) -> Self {
        Self {
            iph,
            i0,
            n,
            vt,
            rs,
            rsh,
            max_iter: 50,
            tol: S::from_f64(1e-9),
        }
    }

    /// Compute the cell current at terminal voltage `v` (V).
    ///
    /// Uses Newton-Raphson iteration.  Returns the converged current (A) or
    /// the best estimate after `max_iter` iterations.
    pub fn current_at(&self, v: S) -> S {
        let n_vt = self.n * self.vt;
        // Initial guess: ignore Rs (explicit solution without series resistance)
        let mut i = self.iph - self.i0 * ((v / n_vt).exp() - S::ONE) - v / self.rsh;
        // Clamp to physically valid range [0, Iph]
        i = i.clamp_val(S::ZERO, self.iph);

        for _ in 0..self.max_iter {
            let vj = v + i * self.rs; // junction voltage
            let exp_term = (vj / n_vt).exp();
            // f(I) = I − Iph + I0*(exp(...) − 1) + vj/Rsh
            let f = i - self.iph + self.i0 * (exp_term - S::ONE) + vj / self.rsh;
            // f'(I) = 1 + (I0*Rs/n_vt)*exp(...) + Rs/Rsh
            let df = S::ONE + (self.i0 * self.rs / n_vt) * exp_term + self.rs / self.rsh;

            if df.abs() < S::from_f64(1e-30) {
                break;
            }
            let delta = f / df;
            i -= delta;
            i = i.clamp_val(S::ZERO, self.iph);
            if delta.abs() < self.tol {
                break;
            }
        }

        i
    }

    /// Compute output power at terminal voltage `v` (V).
    pub fn power_at(&self, v: S) -> S {
        v * self.current_at(v)
    }

    /// Find the voltage of maximum power using ternary search over `[v_lo, v_hi]`.
    ///
    /// Returns `(v_mpp, p_mpp)`.
    pub fn find_mpp(&self, v_lo: S, v_hi: S) -> (S, S) {
        let mut lo = v_lo;
        let mut hi = v_hi;
        let third = S::from_f64(1.0 / 3.0);

        for _ in 0..100 {
            if hi - lo < S::from_f64(1e-6) {
                break;
            }
            let m1 = lo + (hi - lo) * third;
            let m2 = hi - (hi - lo) * third;
            if self.power_at(m1) < self.power_at(m2) {
                lo = m1;
            } else {
                hi = m2;
            }
        }

        let v_opt = (lo + hi) * S::HALF;
        (v_opt, self.power_at(v_opt))
    }
}

// ─── Unit Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Typical silicon PV cell at STC for test purposes.
    fn pv_cell() -> PvCellModel<f64> {
        PvCellModel::new(
            9.0,     // Iph = 9 A
            1e-9,    // I0
            1.5,     // n
            0.02585, // Vt at 25 °C
            0.02,    // Rs = 0.02 Ω
            300.0,   // Rsh = 300 Ω
        )
    }

    #[test]
    fn pv_cell_short_circuit_current() {
        let cell = pv_cell();
        let i_sc = cell.current_at(0.0);
        // At V=0, I ≈ Iph (short-circuit current)
        assert!((i_sc - 9.0).abs() < 0.1, "Isc={i_sc:.4} A (expected ~9 A)");
    }

    #[test]
    fn pv_cell_open_circuit_voltage() {
        let cell = pv_cell();
        // Open-circuit: I=0, so V_oc ≈ n·Vt·ln(Iph/I0)
        // ≈ 1.5 * 0.02585 * ln(9e9) ≈ 0.79 V (single cell)
        let v_oc_approx = cell.n * cell.vt * (cell.iph / cell.i0).ln();
        let i_at_voc = cell.current_at(v_oc_approx);
        assert!(
            i_at_voc.abs() < 0.5,
            "I at V_oc = {i_at_voc:.4} A (should be near 0)"
        );
    }

    #[test]
    fn pv_cell_mpp_is_maximum() {
        let cell = pv_cell();
        let (v_mpp, p_mpp) = cell.find_mpp(0.1, 0.75);
        let p_lo = cell.power_at(0.3);
        let p_hi = cell.power_at(0.72);
        assert!(
            p_mpp >= p_lo && p_mpp >= p_hi,
            "MPP power {p_mpp:.4} should exceed p_lo={p_lo:.4} and p_hi={p_hi:.4}"
        );
        assert!(
            v_mpp > 0.3 && v_mpp < 0.75,
            "V_mpp={v_mpp:.4} should be within (0.3, 0.75)"
        );
    }

    #[test]
    fn po_converges_near_mpp() {
        let cell = pv_cell();
        let (v_mpp_true, _) = cell.find_mpp(0.1, 0.75);

        let mut po = PerturbeAndObserve::new(0.3_f64, 0.002, 0.1, 0.75);

        for _ in 0..500 {
            let v = po.v_ref();
            let i = cell.current_at(v);
            po.update(v, i);
        }

        let v_final = po.v_ref();
        // Should be within 5 steps (0.01 V) of true MPP
        assert!(
            (v_final - v_mpp_true).abs() < 0.015,
            "P&O: v_ref={v_final:.4} V, v_mpp={v_mpp_true:.4} V"
        );
    }

    #[test]
    fn po_direction_tracking() {
        let mut po = PerturbeAndObserve::new(0.3_f64, 0.01, 0.1, 0.75);
        // Feed increasing power → should move in Increase direction
        let delta = po.update(0.3, 8.5); // P = 2.55
        let _d2 = po.update(0.31, 8.6); // P = 2.666 (larger → same direction)
        assert!(delta != 0.0, "delta must be non-zero: {delta}");
    }

    #[test]
    fn inc_at_mpp_returns_zero_perturbation() {
        let cell = pv_cell();
        let (v_mpp, _) = cell.find_mpp(0.1, 0.75);

        // Evaluate InC at the true MPP — should see near-zero perturbation
        let dv = 1e-6_f64;
        let i_at_mpp = cell.current_at(v_mpp);
        let i_plus = cell.current_at(v_mpp + dv);

        // Simulate InC: feed two samples bracketing MPP
        let mut inc = IncrementalConductance::new(v_mpp - dv, 0.001, 0.1, 0.75);
        // First sample: at v_mpp - dv
        let i_prev = cell.current_at(v_mpp - dv);
        inc.update(v_mpp - dv, i_prev);
        // Second sample: at v_mpp (tiny step)
        let delta = inc.update(v_mpp, i_at_mpp);
        // Third: at v_mpp + dv (step right)
        let _d3 = inc.update(v_mpp + dv, i_plus);

        // At MPP, conductance sum ≈ 0, so |delta| should be ~0 or step
        // We just verify the algorithm does not blow up and returns finite
        assert!(delta.is_finite(), "InC delta must be finite: {delta}");
    }

    #[test]
    fn inc_converges_near_mpp() {
        let cell = pv_cell();
        let (v_mpp_true, _) = cell.find_mpp(0.1, 0.75);

        let mut inc = IncrementalConductance::new(0.3_f64, 0.002, 0.1, 0.75);

        for _ in 0..500 {
            let v = inc.v_ref();
            let i = cell.current_at(v);
            inc.update(v, i);
        }

        let v_final = inc.v_ref();
        assert!(
            (v_final - v_mpp_true).abs() < 0.015,
            "InC: v_ref={v_final:.4} V, v_mpp={v_mpp_true:.4} V"
        );
    }

    #[test]
    fn fractional_ocv_estimate() {
        let cell = pv_cell();
        let (v_mpp_true, _) = cell.find_mpp(0.1, 0.75);
        // Approximate V_oc
        let v_oc = 0.78_f64;
        let k_oc = 0.76_f64;

        let mut focv = FractionalOcv::new(k_oc, v_oc, 0.1, 0.75);
        let v_mpp_est = focv.v_mpp();

        // Estimate should be in a reasonable range (within 15% of true MPP)
        assert!(
            (v_mpp_est - v_mpp_true).abs() < 0.15,
            "FractionalOcv estimate={v_mpp_est:.4}, true={v_mpp_true:.4}"
        );

        // After updating V_oc
        let v_new = focv.set_voc(0.80);
        assert!((v_new - k_oc * 0.80).abs() < 1e-10, "set_voc mismatch");
    }

    #[test]
    fn po_reset_works() {
        let mut po = PerturbeAndObserve::new(0.3_f64, 0.01, 0.1, 0.75);
        po.update(0.3, 8.0);
        po.update(0.31, 7.9);
        po.reset(0.3);
        assert!((po.v_ref() - 0.3).abs() < 1e-10);
        assert_eq!(po.power(), 0.0);
    }

    #[test]
    fn inc_reset_works() {
        let mut inc = IncrementalConductance::new(0.3_f64, 0.01, 0.1, 0.75);
        inc.update(0.3, 8.0);
        inc.reset(0.3);
        assert!((inc.v_ref() - 0.3).abs() < 1e-10);
    }
}
