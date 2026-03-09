use crate::core::scalar::ControlScalar;

/// SIL (Safety Integrity Level) classification per IEC 61508.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SilLevel {
    /// No SIL requirement.
    None,
    /// SIL 1: PFD ∈ [10⁻², 10⁻¹)
    Sil1,
    /// SIL 2: PFD ∈ [10⁻³, 10⁻²)
    Sil2,
    /// SIL 3: PFD ∈ [10⁻⁴, 10⁻³)
    Sil3,
    /// SIL 4: PFD ∈ [10⁻⁵, 10⁻⁴)
    Sil4,
}

/// Diagnostic coverage (DC) tier per IEC 61508 / IEC 62061.
///
/// DC = λ_DD / (λ_DD + λ_DU), where:
///   λ_DD = detected dangerous failure rate
///   λ_DU = undetected dangerous failure rate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticCoverage {
    /// DC < 60%
    Low,
    /// 60% ≤ DC < 90%
    Medium,
    /// 90% ≤ DC < 99%
    High,
    /// DC ≥ 99%
    VeryHigh,
}

impl DiagnosticCoverage {
    /// Typical DC fraction (midpoint of range).
    pub fn fraction(self) -> f64 {
        match self {
            Self::Low => 0.30,
            Self::Medium => 0.75,
            Self::High => 0.945,
            Self::VeryHigh => 0.995,
        }
    }

    /// Classify a computed DC fraction.
    pub fn classify(dc: f64) -> Self {
        if dc >= 0.99 {
            Self::VeryHigh
        } else if dc >= 0.90 {
            Self::High
        } else if dc >= 0.60 {
            Self::Medium
        } else {
            Self::Low
        }
    }
}

/// PFH/PFD calculator for a safety function.
///
/// Models a single-channel subsystem (HFT=0 architecture):
///   PFD = λ_DU * T_proof / 2
///   PFH = λ_DU
///
/// Where:
///   λ_DU = (1 - DC) * λ_D   (undetected dangerous failure rate)
///   λ_D  = dangerous failure rate (fraction of λ_total)
///   DC   = diagnostic coverage fraction
#[derive(Debug, Clone, Copy)]
pub struct SafetyFunctionCoverage<S: ControlScalar> {
    /// Total failure rate (failures/hour).
    pub lambda_total: S,
    /// Fraction of failures that are dangerous.
    pub dangerous_fraction: S,
    /// Diagnostic coverage (0..1).
    pub dc: S,
    /// Proof-test interval (hours).
    pub t_proof: S,
}

impl<S: ControlScalar> SafetyFunctionCoverage<S> {
    pub fn new(lambda_total: S, dangerous_fraction: S, dc: S, t_proof: S) -> Self {
        Self {
            lambda_total,
            dangerous_fraction,
            dc,
            t_proof,
        }
    }

    /// Dangerous failure rate λ_D (failures/hour).
    pub fn lambda_dangerous(&self) -> S {
        self.lambda_total * self.dangerous_fraction
    }

    /// Undetected dangerous failure rate λ_DU (failures/hour).
    pub fn lambda_du(&self) -> S {
        self.lambda_dangerous() * (S::ONE - self.dc)
    }

    /// PFH (Probability of dangerous Failure per Hour) for continuous mode.
    pub fn pfh(&self) -> S {
        self.lambda_du()
    }

    /// PFD_avg (Average Probability of Failure on Demand) for low-demand mode.
    pub fn pfd_avg(&self) -> S {
        self.lambda_du() * self.t_proof / S::TWO
    }

    /// SIL classification based on PFD for low-demand mode.
    pub fn sil_low_demand(&self) -> SilLevel {
        let pfd = self.pfd_avg();
        // Convert to f64 for comparison
        let pfd_f = pfd.abs() * S::from_f64(1.0); // keep as S
                                                  // Use rough breakpoints: SIL4 < 1e-5, SIL3 < 1e-4, SIL2 < 1e-3, SIL1 < 1e-2
        if pfd_f < S::from_f64(1e-5) {
            SilLevel::Sil4
        } else if pfd_f < S::from_f64(1e-4) {
            SilLevel::Sil3
        } else if pfd_f < S::from_f64(1e-3) {
            SilLevel::Sil2
        } else if pfd_f < S::from_f64(1e-2) {
            SilLevel::Sil1
        } else {
            SilLevel::None
        }
    }

    /// SIL classification based on PFH for high-demand/continuous mode.
    pub fn sil_continuous(&self) -> SilLevel {
        let pfh = self.pfh();
        if pfh < S::from_f64(1e-8) {
            SilLevel::Sil4
        } else if pfh < S::from_f64(1e-7) {
            SilLevel::Sil3
        } else if pfh < S::from_f64(1e-6) {
            SilLevel::Sil2
        } else if pfh < S::from_f64(1e-5) {
            SilLevel::Sil1
        } else {
            SilLevel::None
        }
    }
}

/// Multi-channel (redundant) subsystem PFD calculator.
///
/// For MooN architecture with N channels and M-out-of-N vote:
///   - 1oo1: PFD = PFD_single
///   - 1oo2: PFD ≈ PFD²  (for independent channels)
///   - 2oo2: PFD ≈ 2*PFD  (both must fail)
#[derive(Debug, Clone, Copy)]
pub struct RedundancyCoverage<S: ControlScalar> {
    pub channel_pfd: S,
    pub n_channels: u32,
    pub m_of_n: u32,
}

impl<S: ControlScalar> RedundancyCoverage<S> {
    pub fn new(channel_pfd: S, n_channels: u32, m_of_n: u32) -> Self {
        Self {
            channel_pfd,
            n_channels,
            m_of_n,
        }
    }

    /// Compute system PFD for 1oo1, 1oo2 (parallel), 2oo2 (series) architectures.
    pub fn system_pfd(&self) -> S {
        match (self.m_of_n, self.n_channels) {
            (1, 1) => self.channel_pfd,
            (1, 2) => self.channel_pfd * self.channel_pfd, // 1oo2: both fail = parallel
            (2, 2) => S::TWO * self.channel_pfd,           // 2oo2: either fails = series
            (1, n) => {
                // 1ooN: all N channels must fail
                let mut p = S::ONE;
                for _ in 0..n {
                    p *= self.channel_pfd;
                }
                p
            }
            _ => self.channel_pfd, // fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sil2_classification() {
        // λ=1e-6/h, 50% dangerous, DC=90%, Tproof=8760h
        let sf = SafetyFunctionCoverage::new(1e-6_f64, 0.5, 0.90, 8760.0);
        let pfd = sf.pfd_avg();
        // λ_DU = 1e-6 * 0.5 * 0.1 = 5e-8/h
        // PFD = 5e-8 * 8760 / 2 = 2.19e-4
        assert!((pfd - 2.19e-4).abs() < 1e-5, "PFD={pfd:.2e}");
        assert_eq!(sf.sil_low_demand(), SilLevel::Sil2);
    }

    #[test]
    fn pfh_matches_lambda_du() {
        let sf = SafetyFunctionCoverage::new(1e-5_f64, 0.8, 0.95, 1000.0);
        let lambda_du = 1e-5 * 0.8 * 0.05;
        assert!((sf.pfh() - lambda_du).abs() < 1e-12);
    }

    #[test]
    fn dc_classification() {
        assert_eq!(
            DiagnosticCoverage::classify(0.99),
            DiagnosticCoverage::VeryHigh
        );
        assert_eq!(DiagnosticCoverage::classify(0.95), DiagnosticCoverage::High);
        assert_eq!(
            DiagnosticCoverage::classify(0.75),
            DiagnosticCoverage::Medium
        );
        assert_eq!(DiagnosticCoverage::classify(0.30), DiagnosticCoverage::Low);
    }

    #[test]
    fn redundancy_1oo2_lower_pfd() {
        let single_pfd = 1e-3_f64;
        let r1oo2 = RedundancyCoverage::new(single_pfd, 2, 1);
        let pfd_sys = r1oo2.system_pfd();
        assert!(pfd_sys < single_pfd, "1oo2 should have lower PFD");
        assert!((pfd_sys - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn redundancy_2oo2_higher_pfd() {
        let single_pfd = 1e-3_f64;
        let r2oo2 = RedundancyCoverage::new(single_pfd, 2, 2);
        let pfd_sys = r2oo2.system_pfd();
        assert!(pfd_sys > single_pfd, "2oo2 should have higher PFD");
    }

    #[test]
    fn sil_continuous_mode() {
        // λ_DU = 5e-8 → SIL 3 (< 1e-7)
        let sf = SafetyFunctionCoverage::new(1e-6_f64, 0.5, 0.90, 8760.0);
        assert_eq!(sf.sil_continuous(), SilLevel::Sil3);
    }
}
