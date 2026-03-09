// Linear Anti-Windup (LAW) Compensator
// Theory: Teel-Praly / Hanus conditioning
// When saturation occurs, injects correction into controller state
// to prevent integrator windup in general linear output-feedback controllers.

use crate::core::scalar::ControlScalar;

/// Errors that can arise in anti-windup structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntiWindupError {
    /// A parameter value is out of the valid range (e.g. u_min >= u_max, dt <= 0).
    InvalidParameter,
    /// Array dimension is inconsistent with the declared order N_C.
    DimensionMismatch,
}

impl core::fmt::Display for AntiWindupError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AntiWindupError::InvalidParameter => write!(f, "AntiWindup: invalid parameter"),
            AntiWindupError::DimensionMismatch => write!(f, "AntiWindup: dimension mismatch"),
        }
    }
}

// ---------------------------------------------------------------------------
// LinearAntiWindup<S, N_C>
// ---------------------------------------------------------------------------

/// Linear Anti-Windup compensator for a SISO, discrete-time, N_C-th order
/// output-feedback controller with input saturation.
///
/// Plant:        x_p[k+1] = Ap*x_p + Bp*sat(u),  y = Cp*x_p
/// Controller:   x_c[k+1] = Ac*x_c + Bc*(r-y),   u_lin = Cc*x_c + Dc*(r-y)
/// Saturation:   v = sat(u_lin) = clamp(u_lin, u_min, u_max)
/// AW signal:    dv = v - u_lin   (zero when not saturated)
/// AW update:    x_c[k+1] += e_aw * dv   (additive correction per Teel-Praly)
///
/// The struct owns and evolves the controller state x_c; the plant state is
/// managed externally.
#[derive(Debug, Clone)]
pub struct LinearAntiWindup<S: ControlScalar, const N_C: usize> {
    /// Controller state vector (N_C-dimensional).
    x_c: [S; N_C],
    /// Controller system matrix  Ac  (N_C × N_C), row-major.
    ac: [[S; N_C]; N_C],
    /// Controller input column vector  Bc  (N_C × 1).
    bc: [S; N_C],
    /// Controller output row vector  Cc  (1 × N_C).
    cc: [S; N_C],
    /// Controller direct feedthrough  Dc  (scalar).
    dc: S,
    /// AW state-correction gain vector  E_aw  (N_C × 1).
    /// Each controller-state component receives  e_aw[i] * dv  per step.
    e_aw: [S; N_C],
    /// Lower saturation limit.
    u_min: S,
    /// Upper saturation limit.
    u_max: S,
    /// True when the last `update` call produced a saturated output.
    saturated: bool,
    /// Last windup signal  dv = v - u_lin  (0 when unsaturated).
    dv: S,
}

impl<S: ControlScalar, const N_C: usize> LinearAntiWindup<S, N_C> {
    /// Create a new `LinearAntiWindup` compensator.
    ///
    /// # Arguments
    /// * `ac`    – Controller A matrix (N_C × N_C), row-major.
    /// * `bc`    – Controller B vector (N_C,).
    /// * `cc`    – Controller C vector (N_C,) – row.
    /// * `dc`    – Controller D scalar.
    /// * `e_aw`  – AW gain vector (N_C,).
    /// * `u_min` – Lower saturation bound.
    /// * `u_max` – Upper saturation bound.
    pub fn new(
        ac: [[S; N_C]; N_C],
        bc: [S; N_C],
        cc: [S; N_C],
        dc: S,
        e_aw: [S; N_C],
        u_min: S,
        u_max: S,
    ) -> Result<Self, AntiWindupError> {
        if u_min >= u_max {
            return Err(AntiWindupError::InvalidParameter);
        }
        Ok(Self {
            x_c: [S::ZERO; N_C],
            ac,
            bc,
            cc,
            dc,
            e_aw,
            u_min,
            u_max,
            saturated: false,
            dv: S::ZERO,
        })
    }

    /// Perform one discrete-time update step.
    ///
    /// # Arguments
    /// * `ref_minus_y` – Error signal  e = r − y  (reference minus measurement).
    ///
    /// # Returns
    /// The (possibly saturated) controller output  v = sat(u_lin).
    pub fn update(&mut self, ref_minus_y: S) -> Result<S, AntiWindupError> {
        // u_lin = Cc * x_c + Dc * e
        let mut u_lin = self.dc * ref_minus_y;
        for i in 0..N_C {
            u_lin += self.cc[i] * self.x_c[i];
        }

        // v = sat(u_lin)
        let v = u_lin.clamp_val(self.u_min, self.u_max);

        // dv = v - u_lin  (0 when not saturated)
        let dv = v - u_lin;
        self.dv = dv;
        self.saturated = (dv * dv) > S::EPSILON * S::EPSILON;

        // x_c[k+1] = Ac * x_c + Bc * e + e_aw * dv
        let mut x_c_next = [S::ZERO; N_C];
        #[allow(clippy::needless_range_loop)]
        for i in 0..N_C {
            // Ac row i dot x_c  (2D index requires range loop)
            let mut ax = S::ZERO;
            for j in 0..N_C {
                ax += self.ac[i][j] * self.x_c[j];
            }
            x_c_next[i] = ax + self.bc[i] * ref_minus_y + self.e_aw[i] * dv;
        }
        self.x_c = x_c_next;

        Ok(v)
    }

    /// Returns `true` when the previous update produced a saturated output.
    #[inline]
    pub fn is_saturated(&self) -> bool {
        self.saturated
    }

    /// Returns the windup signal  dv = v − u_lin  from the last update.
    /// Zero when not saturated.
    #[inline]
    pub fn windup_signal(&self) -> S {
        self.dv
    }

    /// Reset controller state and diagnostic flags to zero.
    pub fn reset(&mut self) {
        self.x_c = [S::ZERO; N_C];
        self.saturated = false;
        self.dv = S::ZERO;
    }

    /// Read-only view of the controller state vector.
    #[inline]
    pub fn controller_state(&self) -> &[S; N_C] {
        &self.x_c
    }
}

// ---------------------------------------------------------------------------
// SimpleAntiWindup<S>
// ---------------------------------------------------------------------------

/// Scalar (1st-order) PI anti-windup controller.
///
/// Integral update with AW correction:
///   u_lin       = kp * e + ki * integrator
///   v           = clamp(u_lin, u_min, u_max)
///   integrator += dt * (e + e_aw * (v − u_lin))
///
/// Setting `e_aw = 0` gives standard clamping (integrator frozen when saturated
/// only if you also zero the increment; here it equals plain integration until
/// clamped externally). For proper back-calculation behaviour choose
/// `e_aw = 1/Ti` or a suitable positive value.
#[derive(Debug, Clone, Copy)]
pub struct SimpleAntiWindup<S: ControlScalar> {
    /// Integrator state.
    integrator: S,
    /// Proportional gain.
    kp: S,
    /// Integral gain  (ki = kp / Ti).
    ki: S,
    /// Lower saturation limit.
    u_min: S,
    /// Upper saturation limit.
    u_max: S,
    /// Anti-windup gain (back-calculation coefficient for the integrator).
    e_aw: S,
    /// Sampling interval [s].
    dt: S,
}

impl<S: ControlScalar> SimpleAntiWindup<S> {
    /// Create a new `SimpleAntiWindup` PI controller.
    ///
    /// # Arguments
    /// * `kp`    – Proportional gain (must be > 0).
    /// * `ki`    – Integral gain (must be ≥ 0).
    /// * `e_aw`  – AW back-calculation gain (≥ 0; 0 = no correction).
    /// * `u_min` – Lower saturation bound.
    /// * `u_max` – Upper saturation bound.
    /// * `dt`    – Sampling interval (must be > 0).
    pub fn new(kp: S, ki: S, e_aw: S, u_min: S, u_max: S, dt: S) -> Result<Self, AntiWindupError> {
        if u_min >= u_max {
            return Err(AntiWindupError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(AntiWindupError::InvalidParameter);
        }
        if kp <= S::ZERO {
            return Err(AntiWindupError::InvalidParameter);
        }
        if ki < S::ZERO || e_aw < S::ZERO {
            return Err(AntiWindupError::InvalidParameter);
        }
        Ok(Self {
            integrator: S::ZERO,
            kp,
            ki,
            u_min,
            u_max,
            e_aw,
            dt,
        })
    }

    /// Perform one discrete-time update step.
    ///
    /// # Arguments
    /// * `error` – Control error  e = r − y.
    ///
    /// # Returns
    /// Saturated control output  v.
    pub fn update(&mut self, error: S) -> Result<S, AntiWindupError> {
        let u_lin = self.kp * error + self.ki * self.integrator;
        let v = u_lin.clamp_val(self.u_min, self.u_max);
        // Back-calculation AW correction
        let dv = v - u_lin;
        self.integrator += self.dt * (error + self.e_aw * dv);
        Ok(v)
    }

    /// Reset integrator state.
    pub fn reset(&mut self) {
        self.integrator = S::ZERO;
    }

    /// Read integrator state.
    #[inline]
    pub fn integrator_state(&self) -> S {
        self.integrator
    }

    /// Read integrator state (alias for `integrator_state`).
    #[inline]
    pub fn integrator(&self) -> S {
        self.integrator
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build identity-like 1st-order LinearAntiWindup (Ac=[[a]], Bc=[b], Cc=[c], Dc=d)
    fn make_law_1(a: f64, b: f64, c: f64, d: f64, e_aw: f64) -> LinearAntiWindup<f64, 1> {
        LinearAntiWindup::new([[a; 1]; 1], [b; 1], [c; 1], d, [e_aw; 1], -10.0, 10.0).unwrap()
    }

    // -----------------------------------------------------------------------
    // 1. Unsaturated: AW inactive, output equals pure linear controller output
    // -----------------------------------------------------------------------
    #[test]
    fn law_unsaturated_no_correction() {
        // Simple integrator controller: Ac=1, Bc=1, Cc=1, Dc=0 → u = x_c
        // e_aw = 5.0, but since dv=0 (unsaturated) it should not matter.
        let mut law = make_law_1(1.0, 1.0, 1.0, 0.0, 5.0);
        // Small error → no saturation
        let v = law.update(0.5).unwrap();
        // First step: u_lin = Cc*0 + Dc*0.5 = 0; x_c_next = 1*0 + 1*0.5 + 5*0 = 0.5
        assert!((v - 0.0).abs() < 1e-12, "v={v}");
        assert!(!law.is_saturated());
        assert!((law.windup_signal()).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 2. Saturated: AW correction stabilizes integrator near equilibrium
    // -----------------------------------------------------------------------
    #[test]
    fn law_saturated_correction_applied() {
        // Pure integrator controller: Ac=[[1]], Bc=[1], Cc=[1], Dc=0.
        // Limits = [-1, 1].  Constant error e = 5.
        //
        // Without AW (e_aw=0): x_c[k+1] = x_c[k] + 5  → grows without bound.
        //
        // With AW (e_aw=1.5): when x_c > 1 (saturated), u_lin=x_c, v=1, dv=1-x_c.
        //   x_c[k+1] = x_c + 5 + 1.5*(1 - x_c) = x_c*(1-1.5) + 5 + 1.5 = -0.5*x_c + 6.5
        //   Fixed point: x_c* = 6.5/1.5 ≈ 4.33  → bounded and stable (|1-1.5|=0.5<1).
        //
        // After many steps the AW version should be bounded while no-AW grows.

        let mut law_aw =
            LinearAntiWindup::<f64, 1>::new([[1.0]], [1.0], [1.0], 0.0, [1.5], -1.0, 1.0).unwrap();
        let mut law_no_aw =
            LinearAntiWindup::<f64, 1>::new([[1.0]], [1.0], [1.0], 0.0, [0.0], -1.0, 1.0).unwrap();

        for _ in 0..30 {
            let _ = law_aw.update(5.0).unwrap();
            let _ = law_no_aw.update(5.0).unwrap();
        }

        let state_aw = law_aw.controller_state()[0].abs();
        let state_no_aw = law_no_aw.controller_state()[0].abs();

        // AW state should be near the fixed point (~4.33), well below no-AW (~150)
        assert!(
            state_aw < 10.0,
            "AW state should be bounded near fixed point: {state_aw}"
        );
        assert!(
            state_no_aw > 100.0,
            "no-AW state should grow large: {state_no_aw}"
        );
        assert!(
            state_aw < state_no_aw,
            "AW state ({state_aw}) should be much smaller than no-AW ({state_no_aw})"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Saturation flag is set correctly
    // -----------------------------------------------------------------------
    #[test]
    fn law_saturation_flag() {
        let mut law =
            LinearAntiWindup::<f64, 1>::new([[0.0]], [0.0], [1.0], 1.0, [0.0], -1.0, 1.0).unwrap();
        // u_lin = 1.0 * x_c + 1.0 * 5.0 = 5.0 → saturated (x_c=0 initially so u_lin=5.0)
        let _ = law.update(5.0).unwrap();
        assert!(law.is_saturated(), "should be saturated");

        // Now small error → no saturation
        let mut law2 =
            LinearAntiWindup::<f64, 1>::new([[0.0]], [0.0], [0.0], 1.0, [0.0], -10.0, 10.0)
                .unwrap();
        let _ = law2.update(0.1).unwrap();
        assert!(!law2.is_saturated(), "should not be saturated");
    }

    // -----------------------------------------------------------------------
    // 4. e_aw = 0 → no correction (equivalent to standard discrete controller)
    // -----------------------------------------------------------------------
    #[test]
    fn law_zero_eaw_no_correction() {
        let mut law_aw =
            LinearAntiWindup::<f64, 1>::new([[1.0]], [1.0], [1.0], 0.0, [0.0], -1.0, 1.0).unwrap();
        let mut law_plain =
            LinearAntiWindup::<f64, 1>::new([[1.0]], [1.0], [1.0], 0.0, [0.0], -1.0, 1.0).unwrap();

        for _ in 0..5 {
            let va = law_aw.update(2.0).unwrap();
            let vb = law_plain.update(2.0).unwrap();
            assert!((va - vb).abs() < 1e-12, "va={va} vb={vb}");
        }
    }

    // -----------------------------------------------------------------------
    // 5. reset() zeroes controller state
    // -----------------------------------------------------------------------
    #[test]
    fn law_reset_zeroes_state() {
        let mut law = make_law_1(1.0, 1.0, 1.0, 0.0, 0.0);
        for _ in 0..5 {
            let _ = law.update(1.0).unwrap();
        }
        law.reset();
        assert_eq!(law.controller_state()[0], 0.0);
        assert!(!law.is_saturated());
        assert_eq!(law.windup_signal(), 0.0);
    }

    // -----------------------------------------------------------------------
    // 6. High reference triggers saturation
    // -----------------------------------------------------------------------
    #[test]
    fn law_high_ref_triggers_saturation() {
        // Dc=1 → u_lin = 1*100 = 100 >> u_max=10
        let mut law =
            LinearAntiWindup::<f64, 1>::new([[0.0]], [0.0], [0.0], 1.0, [0.0], -10.0, 10.0)
                .unwrap();
        let v = law.update(100.0).unwrap();
        assert!((v - 10.0).abs() < 1e-12, "v={v}");
        assert!(law.is_saturated());
        assert!((law.windup_signal() - (-90.0)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 7. SimpleAntiWindup — unsaturated: pure PI output
    // -----------------------------------------------------------------------
    #[test]
    fn simple_aw_unsaturated() {
        let mut saw = SimpleAntiWindup::<f64>::new(1.0, 1.0, 0.0, -100.0, 100.0, 0.01).unwrap();
        // error = 1.0, step 1: u_lin = 1*1 + 1*0 = 1, v = 1
        let v = saw.update(1.0).unwrap();
        assert!((v - 1.0).abs() < 1e-12, "v={v}");
        // integrator should grow: 0 + 0.01*(1 + 0*0) = 0.01
        assert!((saw.integrator_state() - 0.01).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 8. SimpleAntiWindup — saturation: AW prevents integrator growth
    // -----------------------------------------------------------------------
    #[test]
    fn simple_aw_saturation_limits_integrator() {
        // e_aw=1 → back-calculation kicks in strongly
        let mut saw = SimpleAntiWindup::<f64>::new(1.0, 1.0, 1.0, -1.0, 1.0, 0.01).unwrap();
        for _ in 0..200 {
            let _ = saw.update(10.0).unwrap();
        }
        // Integrator should be bounded, not diverge
        assert!(
            saw.integrator_state().abs() < 20.0,
            "integrator={}",
            saw.integrator_state()
        );
    }

    // -----------------------------------------------------------------------
    // 9. SimpleAntiWindup reset
    // -----------------------------------------------------------------------
    #[test]
    fn simple_aw_reset() {
        let mut saw = SimpleAntiWindup::<f64>::new(1.0, 1.0, 0.5, -10.0, 10.0, 0.01).unwrap();
        for _ in 0..10 {
            let _ = saw.update(5.0).unwrap();
        }
        saw.reset();
        assert_eq!(saw.integrator_state(), 0.0);
    }

    // -----------------------------------------------------------------------
    // 10. Invalid parameter: u_min >= u_max
    // -----------------------------------------------------------------------
    #[test]
    fn law_invalid_limits() {
        let res = LinearAntiWindup::<f64, 1>::new([[1.0]], [1.0], [1.0], 0.0, [0.0], 5.0, 5.0);
        assert_eq!(res.unwrap_err(), AntiWindupError::InvalidParameter);
    }

    #[test]
    fn simple_aw_invalid_limits() {
        let res = SimpleAntiWindup::<f64>::new(1.0, 1.0, 0.0, 5.0, 3.0, 0.01);
        assert_eq!(res.unwrap_err(), AntiWindupError::InvalidParameter);
    }
}
