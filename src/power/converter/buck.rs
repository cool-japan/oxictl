use crate::core::scalar::ControlScalar;

/// Average model of a synchronous buck DC-DC converter.
///
/// Continuous conduction mode (CCM) averaged equations:
///   L · dᵢL/dt = d·V_in − V_C
///   C · dV_C/dt = iL − V_C/R_load
///
/// State: [iL (inductor current A), V_C (output voltage V)]
/// Input: d ∈ [0, 1] (duty cycle)
///
/// Ideal: V_out = d · V_in
#[derive(Debug, Clone, Copy)]
pub struct BuckConverter<S: ControlScalar> {
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

impl<S: ControlScalar> BuckConverter<S> {
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

    /// Step the average model by `dt` seconds.
    pub fn step(&mut self, d: S, dt: S) {
        let d_clamped = d.clamp_val(S::ZERO, S::ONE);

        // di_L/dt = (d*V_in - V_C) / L
        let di_l = (d_clamped * self.v_in - self.v_c) / self.l;
        // dV_C/dt = (i_L - V_C/R) / C
        let dv_c = (self.i_l - self.v_c / self.r_load) / self.c;

        self.i_l += di_l * dt;
        self.v_c += dv_c * dt;

        if self.i_l < S::ZERO {
            self.i_l = S::ZERO;
        }
    }

    /// Output voltage (V).
    pub fn v_out(&self) -> S {
        self.v_c
    }

    /// Inductor current (A).
    pub fn i_l(&self) -> S {
        self.i_l
    }

    /// Nominal duty cycle for target output voltage.
    pub fn duty_for_voltage(&self, v_out_ref: S) -> S {
        if self.v_in > S::ZERO {
            (v_out_ref / self.v_in).clamp_val(S::ZERO, S::ONE)
        } else {
            S::ZERO
        }
    }

    pub fn reset(&mut self) {
        self.i_l = S::ZERO;
        self.v_c = S::ZERO;
    }
}

/// PI voltage controller for buck converter.
#[derive(Debug, Clone, Copy)]
pub struct BuckVoltageController<S: ControlScalar> {
    pub kp: S,
    pub ki: S,
    pub d_min: S,
    pub d_max: S,
    integral: S,
}

impl<S: ControlScalar> BuckVoltageController<S> {
    pub fn new(kp: S, ki: S, d_min: S, d_max: S) -> Self {
        Self {
            kp,
            ki,
            d_min,
            d_max,
            integral: S::ZERO,
        }
    }

    pub fn update(&mut self, v_ref: S, v_out: S, dt: S) -> S {
        let error = v_ref - v_out;
        self.integral += error * dt;
        let d = self.kp * error + self.ki * self.integral;
        let d_clamped = d.clamp_val(self.d_min, self.d_max);

        if d_clamped != d {
            self.integral -= error * dt;
        }

        d_clamped
    }

    pub fn reset(&mut self) {
        self.integral = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buck_open_loop_steady_state() {
        // 24V in, d=0.5 → ideal 12V out
        let mut buck = BuckConverter::new(1e-3_f64, 100e-6, 10.0, 24.0);
        let dt = 1e-6_f64;
        for _ in 0..200_000 {
            buck.step(0.5, dt);
        }
        let v_out = buck.v_out();
        assert!(
            v_out > 9.0 && v_out < 15.0,
            "v_out={:.2}V (expected ~12V)",
            v_out
        );
    }

    #[test]
    fn buck_closed_loop_regulation() {
        let mut buck = BuckConverter::new(1e-3_f64, 100e-6, 20.0, 24.0);
        let mut ctrl = BuckVoltageController::new(0.1_f64, 100.0, 0.0, 1.0);
        let dt = 1e-5_f64;
        let v_ref = 12.0_f64;

        for _ in 0..200_000 {
            let d = ctrl.update(v_ref, buck.v_out(), dt);
            buck.step(d, dt);
        }

        let v_out = buck.v_out();
        assert!(
            (v_out - v_ref).abs() < 2.0,
            "v_out={:.2}V, ref={:.2}V",
            v_out,
            v_ref
        );
    }

    #[test]
    fn duty_for_voltage_correct() {
        let buck = BuckConverter::new(1e-3_f64, 100e-6, 10.0, 24.0);
        let d = buck.duty_for_voltage(12.0_f64);
        assert!((d - 0.5).abs() < 0.01, "d={:.3}", d);
    }

    #[test]
    fn unity_duty_reaches_vin() {
        let mut buck = BuckConverter::new(1e-4_f64, 100e-6, 5.0, 12.0);
        for _ in 0..100_000 {
            buck.step(1.0, 1e-6);
        }
        // With d=1.0, output should approach V_in = 12V
        assert!(buck.v_out() > 8.0, "v_out={:.2}", buck.v_out());
    }
}
