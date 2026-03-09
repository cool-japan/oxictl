use crate::core::scalar::ControlScalar;

/// Average model of a boost DC-DC converter.
///
/// Continuous conduction mode (CCM) averaged equations:
///   L · dᵢL/dt = V_in − (1−d)·V_C
///   C · dV_C/dt = (1−d)·iL − V_C/R_load
///
/// State: [iL (inductor current A), V_C (output voltage V)]
/// Input: d ∈ [0, 1] (duty cycle, 0 = diode on, 1 = switch fully on)
///
/// Ideal boost conversion: V_out = V_in / (1 − d)
/// (d must stay below 1 − V_in/V_out_max for continuous operation)
#[derive(Debug, Clone, Copy)]
pub struct BoostConverter<S: ControlScalar> {
    /// Inductance (H).
    pub l: S,
    /// Output capacitance (F).
    pub c: S,
    /// Load resistance (Ω).
    pub r_load: S,
    /// Input voltage (V).
    pub v_in: S,
    /// Inductor current state (A).
    i_l: S,
    /// Capacitor/output voltage state (V).
    v_c: S,
}

impl<S: ControlScalar> BoostConverter<S> {
    /// Create a boost converter.
    ///
    /// Initial state is zero (cold start).
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

    /// Step the average model by `dt` seconds with duty cycle `d` ∈ [0, 1].
    ///
    /// Uses Euler integration of the averaged state equations.
    pub fn step(&mut self, d: S, dt: S) {
        let d_clamped = d.clamp_val(S::ZERO, S::ONE);
        let one_minus_d = S::ONE - d_clamped;

        // di_L/dt = (V_in - (1-d)*V_C) / L
        let di_l = (self.v_in - one_minus_d * self.v_c) / self.l;
        // dV_C/dt = ((1-d)*i_L - V_C/R) / C
        let dv_c = (one_minus_d * self.i_l - self.v_c / self.r_load) / self.c;

        self.i_l += di_l * dt;
        self.v_c += dv_c * dt;

        // Clamp inductor current to non-negative (CCM assumption: no reverse current)
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
    ///
    /// d = 1 − V_in / V_out_ref
    pub fn duty_for_voltage(&self, v_out_ref: S) -> S {
        if v_out_ref > self.v_in {
            let d = S::ONE - self.v_in / v_out_ref;
            d.clamp_val(S::ZERO, S::from_f64(0.95))
        } else {
            S::ZERO
        }
    }

    pub fn reset(&mut self) {
        self.i_l = S::ZERO;
        self.v_c = S::ZERO;
    }
}

/// Voltage-mode PI controller for boost converter.
///
/// Outer voltage loop: error = V_ref − V_out → duty cycle d
/// Includes anti-windup clamping and duty cycle limiting.
#[derive(Debug, Clone, Copy)]
pub struct BoostVoltageController<S: ControlScalar> {
    /// Proportional gain.
    pub kp: S,
    /// Integral gain.
    pub ki: S,
    /// Minimum duty cycle.
    pub d_min: S,
    /// Maximum duty cycle.
    pub d_max: S,
    /// Integral state.
    integral: S,
}

impl<S: ControlScalar> BoostVoltageController<S> {
    pub fn new(kp: S, ki: S, d_min: S, d_max: S) -> Self {
        Self {
            kp,
            ki,
            d_min,
            d_max,
            integral: S::ZERO,
        }
    }

    /// Compute duty cycle for voltage regulation.
    ///
    /// - `v_ref`: target output voltage
    /// - `v_out`: measured output voltage
    /// - `dt`: time step
    pub fn update(&mut self, v_ref: S, v_out: S, dt: S) -> S {
        let error = v_ref - v_out;
        self.integral += error * dt;
        let d = self.kp * error + self.ki * self.integral;
        let d_clamped = d.clamp_val(self.d_min, self.d_max);

        // Anti-windup: clamp integral if output is saturated
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
    fn boost_open_loop_steady_state() {
        // 12V in, d=0.5 → ideal 24V out
        let mut boost = BoostConverter::new(1e-3_f64, 100e-6, 10.0, 12.0);
        let dt = 1e-6_f64;
        let d = 0.5_f64;

        for _ in 0..200_000 {
            boost.step(d, dt);
        }

        // After 0.2s (200000 steps), should be near 24V
        let v_out = boost.v_out();
        assert!(
            v_out > 18.0 && v_out < 30.0,
            "v_out={:.2}V (expected ~24V)",
            v_out
        );
    }

    #[test]
    fn boost_closed_loop_regulation() {
        // Use conservative gains (kp small enough that kp*v_ref < d_max avoids initial saturation)
        // With kp=0.01 and v_ref=24: kp*v_ref = 0.24 < 0.9 ✓
        let mut boost = BoostConverter::new(1e-3_f64, 100e-6, 20.0, 12.0);
        let mut ctrl = BoostVoltageController::new(0.01_f64, 5.0, 0.0, 0.9);
        let dt = 1e-5_f64;
        let v_ref = 24.0_f64;

        for _ in 0..500_000 {
            let d = ctrl.update(v_ref, boost.v_out(), dt);
            boost.step(d, dt);
        }

        let v_out = boost.v_out();
        // Should significantly boost above input voltage (12V → target 24V)
        assert!(
            v_out > 20.0,
            "v_out={:.2}V should exceed 20V (ref={:.2}V)",
            v_out,
            v_ref
        );
    }

    #[test]
    fn duty_for_voltage_correct() {
        let boost = BoostConverter::new(1e-3_f64, 100e-6, 10.0, 12.0);
        let d = boost.duty_for_voltage(24.0_f64);
        assert!((d - 0.5).abs() < 0.01, "d={:.3}", d);
    }

    #[test]
    fn zero_duty_no_boost() {
        let mut boost = BoostConverter::new(1e-4_f64, 100e-6, 10.0, 5.0);
        for _ in 0..10000 {
            boost.step(0.0, 1e-6);
        }
        // With d=0, output should stay near 0 (no energy transfer)
        assert!(boost.v_out() < 5.0, "v_out={:.2}", boost.v_out());
    }
}
