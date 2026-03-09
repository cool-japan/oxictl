use crate::core::scalar::ControlScalar;

/// Results from standard motor parameter identification tests.
#[derive(Debug, Clone, Copy)]
pub struct DcTestResult<S: ControlScalar> {
    /// Applied DC voltage (V).
    pub v_dc: S,
    /// Measured DC current (A).
    pub i_dc: S,
}

/// Results from AC no-load test (motor spinning freely at rated frequency).
#[derive(Debug, Clone, Copy)]
pub struct NoLoadTestResult<S: ControlScalar> {
    /// Applied RMS line voltage (V).
    pub v_rms: S,
    /// Measured no-load RMS current (A).
    pub i_rms: S,
    /// Measured no-load power (W).
    pub p_nl: S,
    /// Test frequency (Hz).
    pub freq: S,
}

/// Results from locked-rotor (short-circuit) test.
#[derive(Debug, Clone, Copy)]
pub struct LockedRotorTestResult<S: ControlScalar> {
    /// Applied RMS voltage at rated frequency (V).
    pub v_rms: S,
    /// Measured short-circuit RMS current (A).
    pub i_rms: S,
    /// Measured short-circuit power (W).
    pub p_sc: S,
    /// Test frequency (Hz).
    pub freq: S,
}

/// Identified parameters for an induction motor.
#[derive(Debug, Clone, Copy)]
pub struct InductionMotorParams<S: ControlScalar> {
    /// Stator resistance (Ω).
    pub rs: S,
    /// Rotor resistance referred to stator (Ω).
    pub rr: S,
    /// Stator leakage inductance (H).
    pub lls: S,
    /// Rotor leakage inductance referred to stator (H) — assume equal to Lls.
    pub llr: S,
    /// Magnetizing inductance (H).
    pub lm: S,
}

/// Identified parameters for a PMSM.
#[derive(Debug, Clone, Copy)]
pub struct PmsmParams<S: ControlScalar> {
    /// Phase resistance (Ω).
    pub rs: S,
    /// d-axis inductance (H).
    pub ld: S,
    /// q-axis inductance (H).
    pub lq: S,
    /// Back-EMF constant (V·s/rad electrical).
    pub ke: S,
}

impl<S: ControlScalar> PmsmParams<S> {
    /// Identify Rs from DC standstill test.
    ///
    /// Apply DC voltage across two phases (series), measure current.
    /// Rs = (V_dc/2) / I_dc  (two stator resistances in series, voltage across two)
    pub fn identify_rs(test: &DcTestResult<S>) -> S {
        test.v_dc / (S::TWO * test.i_dc)
    }

    /// Estimate phase inductance from AC test at known frequency.
    ///
    /// Z_total = V_rms / I_rms
    /// Z_r = Rs (known)
    /// X_L = sqrt(Z_total² − Rs²)
    /// L = X_L / (2π·f)
    pub fn identify_ls(v_rms: S, i_rms: S, rs: S, freq: S) -> S {
        let z = v_rms / i_rms;
        let x_sq = z * z - rs * rs;
        if x_sq <= S::ZERO {
            return S::ZERO;
        }
        let x_l = x_sq.sqrt();
        x_l / (S::TWO * S::PI * freq)
    }

    /// Identify back-EMF constant from back-EMF voltage at known speed.
    ///
    /// Ke = V_bemf_rms / ω_e   (where ω_e = Pp * ω_mech)
    pub fn identify_ke(v_bemf_rms: S, omega_e: S) -> S {
        if omega_e.abs() > S::from_f64(0.1) {
            v_bemf_rms / omega_e
        } else {
            S::ZERO
        }
    }
}

impl<S: ControlScalar> InductionMotorParams<S> {
    /// Identify all induction motor parameters from three standard tests.
    ///
    /// Method:
    ///   1. DC test → Rs
    ///   2. No-load test → Lm
    ///   3. Locked-rotor test → Rr, Lls (= Llr assumed)
    pub fn identify(
        dc: &DcTestResult<S>,
        no_load: &NoLoadTestResult<S>,
        locked_rotor: &LockedRotorTestResult<S>,
    ) -> Self {
        let two_pi = S::TWO * S::PI;

        // Rs from DC test
        let rs = dc.v_dc / (S::TWO * dc.i_dc);

        // No-load: estimate Lm from magnetizing reactance
        // X_m ≈ V_nl / I_nl (ignore Rs for no-load, current mostly reactive)
        let omega_nl = two_pi * no_load.freq;
        let z_nl = no_load.v_rms / no_load.i_rms;
        let x_m = (z_nl * z_nl - rs * rs)
            .clamp_val(S::ZERO, S::from_f64(f64::MAX / 2.0))
            .sqrt();
        let lm = if omega_nl > S::ZERO {
            x_m / omega_nl
        } else {
            S::ZERO
        };

        // Locked-rotor: total Rsc = P_sc / I_sc² ≈ Rs + Rr
        let rsc = locked_rotor.p_sc / (locked_rotor.i_rms * locked_rotor.i_rms);
        let rr = (rsc - rs).clamp_val(S::ZERO, S::from_f64(f64::MAX / 2.0));

        // Leakage inductance: Z_sc = V_sc / I_sc
        let omega_lr = two_pi * locked_rotor.freq;
        let z_sc = locked_rotor.v_rms / locked_rotor.i_rms;
        let x_sc = (z_sc * z_sc - rsc * rsc)
            .clamp_val(S::ZERO, S::from_f64(f64::MAX / 2.0))
            .sqrt();
        // Total leakage X_sc = ω*(Lls + Llr) → split equally
        let lls = if omega_lr > S::ZERO {
            x_sc / (S::TWO * omega_lr)
        } else {
            S::ZERO
        };

        Self {
            rs,
            rr,
            lls,
            llr: lls,
            lm,
        }
    }

    /// Total stator inductance Ls = Lls + Lm.
    pub fn ls(&self) -> S {
        self.lls + self.lm
    }

    /// Total rotor inductance Lr = Llr + Lm.
    pub fn lr(&self) -> S {
        self.llr + self.lm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_test_identifies_rs() {
        let test = DcTestResult {
            v_dc: 10.0_f64,
            i_dc: 2.5,
        };
        let rs = PmsmParams::identify_rs(&test);
        // V_dc = 2 * I_dc * Rs → Rs = V/(2I) = 10/(2*2.5) = 2.0
        assert!((rs - 2.0).abs() < 1e-10, "rs={rs:.4}");
    }

    #[test]
    fn identify_inductance_from_ac_test() {
        // Z = 10 Ω, Rs = 3 Ω → XL = sqrt(100-9) = sqrt(91) ≈ 9.54
        // f = 50 Hz → L = 9.54 / (2π*50) ≈ 0.0304 H
        let ls = PmsmParams::identify_ls(10.0_f64, 1.0, 3.0, 50.0);
        let expected = (91.0_f64.sqrt()) / (2.0 * core::f64::consts::PI * 50.0);
        assert!(
            (ls - expected).abs() < 1e-6,
            "ls={ls:.6}, expected={expected:.6}"
        );
    }

    #[test]
    fn induction_motor_identification() {
        let dc = DcTestResult {
            v_dc: 20.0_f64,
            i_dc: 2.0,
        };
        let no_load = NoLoadTestResult {
            v_rms: 220.0,
            i_rms: 3.0,
            p_nl: 200.0,
            freq: 50.0,
        };
        let locked = LockedRotorTestResult {
            v_rms: 50.0,
            i_rms: 5.0,
            p_sc: 100.0,
            freq: 50.0,
        };

        let params = InductionMotorParams::identify(&dc, &no_load, &locked);

        // Rs = 20/(2*2) = 5 Ω
        assert!((params.rs - 5.0).abs() < 1e-10, "rs={:.4}", params.rs);
        // Rr should be positive
        assert!(params.rr >= 0.0, "rr={:.4}", params.rr);
        // Lm should be positive
        assert!(params.lm > 0.0, "lm={:.4}", params.lm);
    }

    #[test]
    fn ke_identification() {
        let ke = PmsmParams::identify_ke(100.0_f64, 1000.0);
        assert!((ke - 0.1).abs() < 1e-10, "ke={ke:.6}");
    }
}
