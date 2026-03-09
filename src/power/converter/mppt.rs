use crate::core::scalar::ControlScalar;

/// Maximum Power Point Tracking: Perturb and Observe (P&O).
///
/// Algorithm:
///   1. Measure V, I → compute P = V·I
///   2. Compare with previous P and V
///   3. If ΔP > 0 and ΔV > 0 → keep increasing V_ref
///      If ΔP > 0 and ΔV < 0 → keep decreasing V_ref
///      If ΔP < 0 and ΔV > 0 → reverse: decrease V_ref
///      If ΔP < 0 and ΔV < 0 → reverse: increase V_ref
///
/// Suitable for slowly varying irradiance. Simple and widely deployed.
#[derive(Debug, Clone, Copy)]
pub struct MpptPerturb<S: ControlScalar> {
    /// Voltage perturbation step size (V).
    pub step: S,
    /// Minimum voltage reference (V).
    pub v_min: S,
    /// Maximum voltage reference (V).
    pub v_max: S,
    /// Current voltage reference (V).
    v_ref: S,
    /// Previous power measurement (W).
    p_prev: S,
    /// Previous voltage measurement (V).
    v_prev: S,
    /// Small threshold to ignore noise.
    eps: S,
}

impl<S: ControlScalar> MpptPerturb<S> {
    /// Create P&O MPPT.
    ///
    /// - `v_init`: initial voltage reference
    /// - `step`: perturbation step size (V), e.g. 0.5 for a 100V panel
    /// - `v_min`, `v_max`: operating range limits
    pub fn new(v_init: S, step: S, v_min: S, v_max: S) -> Self {
        Self {
            step,
            v_min,
            v_max,
            v_ref: v_init,
            p_prev: S::ZERO,
            v_prev: v_init,
            eps: S::from_f64(1e-6),
        }
    }

    /// Update MPPT and return new voltage reference.
    ///
    /// - `v`: measured panel voltage (V)
    /// - `i`: measured panel current (A)
    pub fn update(&mut self, v: S, i: S) -> S {
        let p = v * i;
        let dp = p - self.p_prev;
        let dv = v - self.v_prev;

        if dp.abs() > self.eps {
            // Determine perturbation direction
            let same_direction = (dp > S::ZERO && dv > S::ZERO) || (dp < S::ZERO && dv < S::ZERO);
            if same_direction {
                self.v_ref += self.step;
            } else {
                self.v_ref -= self.step;
            }
            self.v_ref = self.v_ref.clamp_val(self.v_min, self.v_max);
        }

        self.p_prev = p;
        self.v_prev = v;

        self.v_ref
    }

    /// Current voltage reference.
    pub fn v_ref(&self) -> S {
        self.v_ref
    }

    /// Last measured power.
    pub fn power(&self) -> S {
        self.p_prev
    }

    pub fn reset(&mut self, v_init: S) {
        self.v_ref = v_init;
        self.p_prev = S::ZERO;
        self.v_prev = v_init;
    }
}

/// MPPT: Incremental Conductance (InC).
///
/// At the MPP: dI/dV = -I/V  ⟺  I/V + dI/dV = 0
///
/// More accurate than P&O under rapidly changing conditions.
/// Requires current and voltage derivatives.
#[derive(Debug, Clone, Copy)]
pub struct MpptInc<S: ControlScalar> {
    /// Voltage step size (V).
    pub step: S,
    pub v_min: S,
    pub v_max: S,
    v_ref: S,
    v_prev: S,
    i_prev: S,
    eps: S,
}

impl<S: ControlScalar> MpptInc<S> {
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

    /// Update with measured V and I.
    pub fn update(&mut self, v: S, i: S) -> S {
        let dv = v - self.v_prev;
        let di = i - self.i_prev;

        if dv.abs() < self.eps {
            // No voltage change: check current
            if di.abs() > self.eps {
                if di > S::ZERO {
                    self.v_ref += self.step;
                } else {
                    self.v_ref -= self.step;
                }
            }
            // di ≈ 0 → at MPP, no change
        } else {
            // Compute incremental conductance condition
            // If (I/V + dI/dV) > 0 → left of MPP, increase V
            // If (I/V + dI/dV) < 0 → right of MPP, decrease V
            let cond = if v.abs() > self.eps {
                i / v + di / dv
            } else {
                di / dv
            };

            if cond.abs() < self.eps {
                // At MPP
            } else if cond > S::ZERO {
                self.v_ref += self.step;
            } else {
                self.v_ref -= self.step;
            }
        }

        self.v_ref = self.v_ref.clamp_val(self.v_min, self.v_max);
        self.v_prev = v;
        self.i_prev = i;
        self.v_ref
    }

    pub fn v_ref(&self) -> S {
        self.v_ref
    }

    pub fn reset(&mut self, v_init: S) {
        self.v_ref = v_init;
        self.v_prev = v_init;
        self.i_prev = S::ZERO;
    }
}

/// Simple solar panel model for testing MPPT.
///
/// I = I_sc - I_0 * (exp((V + I*R_s)/(n*V_T)) - 1)
/// Simplified single-diode approximation.
#[derive(Debug, Clone, Copy)]
pub struct SolarPanel<S: ControlScalar> {
    /// Short-circuit current (A).
    pub i_sc: S,
    /// Open-circuit voltage (V).
    pub v_oc: S,
    /// Voltage at MPP (V).
    pub v_mpp: S,
    /// Current at MPP (A).
    pub i_mpp: S,
}

impl<S: ControlScalar> SolarPanel<S> {
    /// Approximate I-V curve using piecewise linear model near MPP.
    pub fn current(&self, v: S) -> S {
        // Simplified linear approximation: I = I_sc * (1 - v/v_oc)^n ≈ I_sc - slope*v
        // Use two-segment: flat from 0..v_mpp, then steep from v_mpp..v_oc
        let v_clamped = v.clamp_val(S::ZERO, self.v_oc);
        if v_clamped <= self.v_mpp {
            let slope = (self.i_sc - self.i_mpp) / self.v_mpp;
            self.i_sc - slope * v_clamped
        } else {
            let slope = self.i_mpp / (self.v_oc - self.v_mpp);
            self.i_mpp - slope * (v_clamped - self.v_mpp)
        }
    }

    /// Power at voltage V.
    pub fn power(&self, v: S) -> S {
        v * self.current(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical_panel() -> SolarPanel<f64> {
        SolarPanel {
            i_sc: 10.0,
            v_oc: 40.0,
            v_mpp: 33.0,
            i_mpp: 9.3,
        }
    }

    #[test]
    fn po_converges_to_mpp() {
        let panel = typical_panel();
        let mut mppt = MpptPerturb::new(20.0_f64, 0.5, 5.0, 39.0);

        for _ in 0..200 {
            let v = mppt.v_ref();
            let i = panel.current(v);
            mppt.update(v, i);
        }

        let v_final = mppt.v_ref();
        // Should converge near V_mpp = 33V (within 2V = 4 steps)
        assert!(
            (v_final - 33.0).abs() < 2.0,
            "P&O: v_ref={:.2}V (expect ~33V)",
            v_final
        );
    }

    #[test]
    fn inc_converges_to_mpp() {
        let panel = typical_panel();
        let mut mppt = MpptInc::new(20.0_f64, 0.5, 5.0, 39.0);

        for _ in 0..300 {
            let v = mppt.v_ref();
            let i = panel.current(v);
            mppt.update(v, i);
        }

        let v_final = mppt.v_ref();
        assert!(
            (v_final - 33.0).abs() < 2.0,
            "InC: v_ref={:.2}V (expect ~33V)",
            v_final
        );
    }

    #[test]
    fn panel_mpp_is_maximum() {
        let panel = typical_panel();
        let p_mpp = panel.power(33.0_f64);
        let p_low = panel.power(20.0_f64);
        let p_high = panel.power(38.0_f64);
        assert!(p_mpp > p_low, "MPP should give more power than low V");
        assert!(p_mpp > p_high, "MPP should give more power than high V");
    }

    #[test]
    fn mppt_resets_cleanly() {
        let mut mppt = MpptPerturb::new(20.0_f64, 0.5, 5.0, 39.0);
        mppt.update(25.0, 8.0);
        mppt.reset(20.0);
        assert!((mppt.v_ref() - 20.0).abs() < 1e-10);
    }
}
