//! Reliability models: exponential, Weibull, bathtub; SIL classification.
#![allow(dead_code)]

use crate::core::scalar::ControlScalar;

/// SIL level per IEC 62061 Table 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SilLevel {
    None = 0,
    Sil1 = 1,
    Sil2 = 2,
    Sil3 = 3,
    Sil4 = 4,
}

impl SilLevel {
    /// Classify SIL from PFH (probability of dangerous failure per hour).
    ///
    /// IEC 62061 PFH ranges:
    ///   SIL4: PFH < 1e-8
    ///   SIL3: 1e-8 ≤ PFH < 1e-7
    ///   SIL2: 1e-7 ≤ PFH < 1e-6
    ///   SIL1: 1e-6 ≤ PFH < 1e-5
    ///   None: PFH ≥ 1e-5
    pub fn from_pfh<S: ControlScalar>(pfh: S) -> Self {
        let v = pfh.to_f64();
        if v < 1e-8 {
            SilLevel::Sil4
        } else if v < 1e-7 {
            SilLevel::Sil3
        } else if v < 1e-6 {
            SilLevel::Sil2
        } else if v < 1e-5 {
            SilLevel::Sil1
        } else {
            SilLevel::None
        }
    }

    /// Upper PFH threshold for this SIL level.
    pub fn pfh_threshold<S: ControlScalar>(self) -> S {
        match self {
            SilLevel::Sil4 => S::from_f64(1e-8),
            SilLevel::Sil3 => S::from_f64(1e-7),
            SilLevel::Sil2 => S::from_f64(1e-6),
            SilLevel::Sil1 => S::from_f64(1e-5),
            SilLevel::None => S::from_f64(1.0),
        }
    }
}

/// Failure rate model variant.
#[derive(Debug, Clone, Copy)]
pub enum FailureModel {
    /// Constant failure rate (exponential distribution).
    Exponential,
    /// Weibull distribution with shape parameter β.
    Weibull { beta: f64 },
    /// Bathtub curve: early-life (β_early < 1) and wear-out (β_wear > 1) phases.
    Bathtub { beta_early: f64, beta_wear: f64 },
}

/// Reliability model for a component.
#[derive(Debug, Clone, Copy)]
pub struct ReliabilityModel<S: ControlScalar> {
    /// Failure rate λ (failures per hour).
    pub lambda: S,
    /// Failure distribution model.
    pub model: FailureModel,
}

impl<S: ControlScalar> ReliabilityModel<S> {
    /// Exponential (constant hazard rate) model.
    pub fn exponential(lambda: S) -> Self {
        Self {
            lambda,
            model: FailureModel::Exponential,
        }
    }

    /// Weibull model with shape parameter β.
    pub fn weibull(lambda: S, beta: f64) -> Self {
        Self {
            lambda,
            model: FailureModel::Weibull { beta },
        }
    }

    /// Bathtub model.
    pub fn bathtub(lambda: S, beta_early: f64, beta_wear: f64) -> Self {
        Self {
            lambda,
            model: FailureModel::Bathtub {
                beta_early,
                beta_wear,
            },
        }
    }

    /// Mean Time Between Failures = 1/λ (valid for exponential model).
    pub fn mtbf(&self) -> S {
        if self.lambda <= S::ZERO {
            return S::from_f64(f64::MAX);
        }
        S::ONE / self.lambda
    }

    /// Steady-state availability given repair rate μ: A = μ / (λ + μ).
    pub fn availability(&self, repair_rate: S) -> S {
        let denom = self.lambda + repair_rate;
        if denom <= S::ZERO {
            return S::ONE;
        }
        repair_rate / denom
    }

    /// Average Probability of Failure on Demand for proof-test interval T1 (hours).
    /// PFD_avg = λ * T1 / 2  (low demand mode approximation, IEC 61508).
    pub fn pfd_avg(&self, t1: S) -> S {
        self.lambda * t1 * S::HALF
    }

    /// Probability of Dangerous Failure per Hour (continuous / high demand mode).
    /// PFH = λ (exponential approximation).
    pub fn pfh(&self) -> S {
        self.lambda
    }

    /// Classify SIL level based on PFH.
    pub fn sil_level(&self) -> SilLevel {
        SilLevel::from_pfh(self.pfh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sil_classification_from_pfh() {
        assert_eq!(SilLevel::from_pfh(5e-9_f64), SilLevel::Sil4);
        assert_eq!(SilLevel::from_pfh(5e-8_f64), SilLevel::Sil3);
        assert_eq!(SilLevel::from_pfh(5e-7_f64), SilLevel::Sil2);
        assert_eq!(SilLevel::from_pfh(5e-6_f64), SilLevel::Sil1);
        assert_eq!(SilLevel::from_pfh(1e-4_f64), SilLevel::None);
    }

    #[test]
    fn mtbf_and_availability() {
        let model = ReliabilityModel::exponential(1e-4_f64);
        // MTBF = 1 / 1e-4 = 10_000 hours
        assert!((model.mtbf() - 10_000.0).abs() < 1.0);

        // With repair rate μ = 1e-3: A = 1e-3 / (1e-4 + 1e-3) ≈ 0.9091
        let avail = model.availability(1e-3_f64);
        assert!((avail - 0.9090909).abs() < 1e-5, "avail={avail}");
    }

    #[test]
    fn pfd_avg_and_sil_level() {
        // λ = 1e-6/h, T1 = 8760 h (annual proof test)
        let model = ReliabilityModel::exponential(1e-6_f64);
        let pfd = model.pfd_avg(8760.0_f64);
        // PFD_avg = 1e-6 * 8760 / 2 = 4.38e-3
        assert!((pfd - 4.38e-3).abs() < 1e-6, "pfd={pfd}");

        // SIL: PFH = 1e-6 → SIL1
        assert_eq!(model.sil_level(), SilLevel::Sil1);
    }

    #[test]
    fn weibull_model_stored() {
        // λ = 5e-6 → PFH = 5e-6 ∈ [1e-6, 1e-5) → SIL1.
        let model = ReliabilityModel::weibull(5e-6_f64, 2.5_f64);
        assert!(
            matches!(model.model, FailureModel::Weibull { beta } if (beta - 2.5).abs() < 1e-10)
        );
        assert_eq!(model.sil_level(), SilLevel::Sil1);
    }
}
