use crate::core::scalar::ControlScalar;

/// Average model of a non-inverting synchronous buck-boost DC-DC converter.
///
/// Uses 4-switch (H-bridge) topology operating in:
///   - Buck mode  when V_out < V_in (d2=0, d1=d)
///   - Boost mode when V_out > V_in (d1=1, d2=d)
///   - Buck-boost transition managed by mode selection
///
/// Simplified single-inductor averaged model (both modes):
///   L · dᵢL/dt  = V_eff_in − V_C         where V_eff_in = d1*V_in, V_eff_out = d2*V_C
///   C · dV_C/dt = i_L · d_factor − V_C/R  where d_factor = 1 in buck, (1−d2) in boost
///
/// For practical use, this model switches between buck and boost modes automatically
/// based on V_ref vs V_in.
#[derive(Debug, Clone, Copy)]
pub struct BuckBoostConverter<S: ControlScalar> {
    /// Inductance (H).
    pub l: S,
    /// Output capacitance (F).
    pub c: S,
    /// Load resistance (Ω).
    pub r_load: S,
    /// Input voltage (V).
    pub v_in: S,
    /// Inductor current (A).
    i_l: S,
    /// Output voltage (V).
    v_c: S,
}

/// Operating mode selected automatically by the controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BuckBoostMode {
    Buck,
    Boost,
}

impl<S: ControlScalar> BuckBoostConverter<S> {
    pub fn new(l: S, c: S, r_load: S, v_in: S) -> Self {
        Self {
            l,
            c,
            r_load,
            v_in,
            i_l: S::ZERO,
            v_c: S::ZERO,
        }
    }

    /// Step with explicit duty cycle `d` and mode.
    ///
    /// - Buck mode:  di/dt = (d*V_in − V_C)/L,  dV_C/dt = (i_L − V_C/R)/C
    /// - Boost mode: di/dt = (V_in − (1−d)*V_C)/L, dV_C/dt = ((1−d)*i_L − V_C/R)/C
    pub fn step_with_mode(&mut self, d: S, mode: BuckBoostMode, dt: S) {
        let d = d.clamp_val(S::ZERO, S::ONE);
        let (di_l, dv_c) = match mode {
            BuckBoostMode::Buck => {
                let di = (d * self.v_in - self.v_c) / self.l;
                let dv = (self.i_l - self.v_c / self.r_load) / self.c;
                (di, dv)
            }
            BuckBoostMode::Boost => {
                let one_minus_d = S::ONE - d;
                let di = (self.v_in - one_minus_d * self.v_c) / self.l;
                let dv = (one_minus_d * self.i_l - self.v_c / self.r_load) / self.c;
                (di, dv)
            }
        };
        self.i_l += di_l * dt;
        self.v_c += dv_c * dt;
        if self.i_l < S::ZERO {
            self.i_l = S::ZERO;
        }
    }

    pub fn v_out(&self) -> S {
        self.v_c
    }
    pub fn i_l(&self) -> S {
        self.i_l
    }

    pub fn reset(&mut self) {
        self.i_l = S::ZERO;
        self.v_c = S::ZERO;
    }
}

/// PI voltage controller for buck-boost converter.
///
/// Automatically selects buck or boost mode based on V_ref vs V_in.
#[derive(Debug, Clone, Copy)]
pub struct BuckBoostController<S: ControlScalar> {
    pub kp: S,
    pub ki: S,
    pub d_min: S,
    pub d_max: S,
    /// Threshold ratio (V_ref/V_in) for mode switching. Default 0.9.
    pub mode_threshold: S,
    integral: S,
}

impl<S: ControlScalar> BuckBoostController<S> {
    pub fn new(kp: S, ki: S, d_min: S, d_max: S) -> Self {
        Self {
            kp,
            ki,
            d_min,
            d_max,
            mode_threshold: S::from_f64(0.9),
            integral: S::ZERO,
        }
    }

    /// Compute duty cycle and mode for voltage regulation.
    pub fn update(&mut self, v_ref: S, v_out: S, v_in: S, dt: S) -> (S, BuckBoostMode) {
        let mode = if v_ref < v_in * self.mode_threshold {
            BuckBoostMode::Buck
        } else {
            BuckBoostMode::Boost
        };

        let error = v_ref - v_out;
        self.integral += error * dt;
        let d = self.kp * error + self.ki * self.integral;
        let d_clamped = d.clamp_val(self.d_min, self.d_max);

        if d_clamped != d {
            self.integral -= error * dt;
        }

        (d_clamped, mode)
    }

    pub fn reset(&mut self) {
        self.integral = S::ZERO;
    }
}

#[cfg(test)]
impl<S: ControlScalar> BuckBoostController<S> {
    fn update_mode(&self, v_ref: S, v_in: S) -> BuckBoostMode {
        if v_ref < v_in * self.mode_threshold {
            BuckBoostMode::Buck
        } else {
            BuckBoostMode::Boost
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buck_mode_steps_down() {
        // 24V → 12V (buck)
        let mut conv = BuckBoostConverter::new(1e-3_f64, 100e-6, 10.0, 24.0);
        for _ in 0..200_000 {
            conv.step_with_mode(0.5, BuckBoostMode::Buck, 1e-6);
        }
        let v = conv.v_out();
        assert!(v > 9.0 && v < 15.0, "Buck: v={:.2}V (expect ~12V)", v);
    }

    #[test]
    fn boost_mode_steps_up() {
        // 12V → 24V (boost)
        let mut conv = BuckBoostConverter::new(1e-3_f64, 100e-6, 20.0, 12.0);
        for _ in 0..200_000 {
            conv.step_with_mode(0.5, BuckBoostMode::Boost, 1e-6);
        }
        let v = conv.v_out();
        assert!(v > 18.0 && v < 30.0, "Boost: v={:.2}V (expect ~24V)", v);
    }

    #[test]
    fn mode_selector_chooses_correctly() {
        let ctrl = BuckBoostController::new(0.01_f64, 5.0, 0.0, 0.95);
        // V_ref=12, V_in=24: 12 < 24*0.9=21.6 → Buck
        assert_eq!(ctrl.update_mode(12.0, 24.0), BuckBoostMode::Buck);
        // V_ref=24, V_in=12: 24 < 12*0.9=10.8? No → Boost
        assert_eq!(ctrl.update_mode(24.0, 12.0), BuckBoostMode::Boost);
    }

    #[test]
    fn closed_loop_buck_regulation() {
        let mut conv = BuckBoostConverter::new(1e-3_f64, 100e-6, 20.0, 24.0);
        let mut ctrl = BuckBoostController::new(0.1_f64, 100.0, 0.0, 1.0);
        let v_ref = 12.0_f64;

        for _ in 0..200_000 {
            let (d, mode) = ctrl.update(v_ref, conv.v_out(), conv.v_in, 1e-5);
            conv.step_with_mode(d, mode, 1e-5);
        }
        assert!(
            (conv.v_out() - v_ref).abs() < 2.0,
            "v_out={:.2}V, ref={:.2}V",
            conv.v_out(),
            v_ref
        );
    }
}
