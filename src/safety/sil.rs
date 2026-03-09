//! Safety Integrity Level (SIL) classification per IEC 61508.
//!
//! Provides SIL classification from PFD (low-demand mode, Table 2) and
//! PFH (continuous/high-demand mode, Table 3) as defined in IEC 61508-1.
//!
//! PFD: Probability of Failure on Demand (dimensionless).
//! PFH: Probability of dangerous Failure per Hour (h⁻¹).

#![allow(dead_code)]

/// Probability of Failure on Demand range (dimensionless).
///
/// Represents the IEC 61508 Table 2 PFD interval [min, max) for a SIL level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PfdRange {
    /// Lower bound (inclusive) of the PFD interval.
    pub min: f64,
    /// Upper bound (exclusive) of the PFD interval.
    pub max: f64,
}

impl PfdRange {
    /// Returns true if `pfd` falls within this range [min, max).
    #[inline]
    pub fn contains(self, pfd: f64) -> bool {
        pfd >= self.min && pfd < self.max
    }
}

/// Probability of dangerous Failure per Hour range (h⁻¹).
///
/// Represents the IEC 61508 Table 3 PFH interval [min, max) for a SIL level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PfhRange {
    /// Lower bound (inclusive) of the PFH interval.
    pub min: f64,
    /// Upper bound (exclusive) of the PFH interval.
    pub max: f64,
}

impl PfhRange {
    /// Returns true if `pfh` falls within this range [min, max).
    #[inline]
    pub fn contains(self, pfh: f64) -> bool {
        pfh >= self.min && pfh < self.max
    }
}

/// Safety Integrity Level per IEC 61508.
///
/// `None` indicates no SIL requirement is met (PFD ≥ 10⁻¹ or PFH ≥ 10⁻⁵).
/// Levels 1–4 correspond to increasing integrity requirements.
///
/// Low-demand mode uses PFD (Table 2); continuous/high-demand mode uses PFH (Table 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SilLevel {
    /// No SIL requirement satisfied.
    None,
    /// SIL 1 — low integrity.
    Sil1,
    /// SIL 2 — moderate integrity.
    Sil2,
    /// SIL 3 — high integrity.
    Sil3,
    /// SIL 4 — very high integrity.
    Sil4,
}

impl SilLevel {
    /// IEC 61508 Table 2 — PFD range for low-demand mode.
    ///
    /// | SIL | PFD interval        |
    /// |-----|---------------------|
    /// |  4  | [10⁻⁵, 10⁻⁴)       |
    /// |  3  | [10⁻⁴, 10⁻³)       |
    /// |  2  | [10⁻³, 10⁻²)       |
    /// |  1  | [10⁻², 10⁻¹)       |
    /// | None| [10⁻¹, ∞)           |
    pub fn pfd_range(self) -> PfdRange {
        match self {
            SilLevel::None => PfdRange {
                min: 1e-1,
                max: f64::INFINITY,
            },
            SilLevel::Sil1 => PfdRange {
                min: 1e-2,
                max: 1e-1,
            },
            SilLevel::Sil2 => PfdRange {
                min: 1e-3,
                max: 1e-2,
            },
            SilLevel::Sil3 => PfdRange {
                min: 1e-4,
                max: 1e-3,
            },
            SilLevel::Sil4 => PfdRange {
                min: 1e-5,
                max: 1e-4,
            },
        }
    }

    /// IEC 61508 Table 3 — PFH range for continuous / high-demand mode.
    ///
    /// | SIL | PFH interval        |
    /// |-----|---------------------|
    /// |  4  | [10⁻⁹, 10⁻⁸)       |
    /// |  3  | [10⁻⁸, 10⁻⁷)       |
    /// |  2  | [10⁻⁷, 10⁻⁶)       |
    /// |  1  | [10⁻⁶, 10⁻⁵)       |
    /// | None| [10⁻⁵, ∞)           |
    pub fn pfh_range(self) -> PfhRange {
        match self {
            SilLevel::None => PfhRange {
                min: 1e-5,
                max: f64::INFINITY,
            },
            SilLevel::Sil1 => PfhRange {
                min: 1e-6,
                max: 1e-5,
            },
            SilLevel::Sil2 => PfhRange {
                min: 1e-7,
                max: 1e-6,
            },
            SilLevel::Sil3 => PfhRange {
                min: 1e-8,
                max: 1e-7,
            },
            SilLevel::Sil4 => PfhRange {
                min: 1e-9,
                max: 1e-8,
            },
        }
    }

    /// Classify a measured PFD value into its IEC 61508 SIL level (low-demand mode).
    ///
    /// Returns [`SilLevel::None`] if PFD ≥ 10⁻¹ or if `pfd` is NaN / negative.
    pub fn classify_from_pfd(pfd: f64) -> SilLevel {
        if !pfd.is_finite() || pfd < 0.0 {
            return SilLevel::None;
        }
        if pfd < 1e-4 {
            SilLevel::Sil4
        } else if pfd < 1e-3 {
            SilLevel::Sil3
        } else if pfd < 1e-2 {
            SilLevel::Sil2
        } else if pfd < 1e-1 {
            SilLevel::Sil1
        } else {
            SilLevel::None
        }
    }

    /// Classify a measured PFH value into its IEC 61508 SIL level (continuous mode).
    ///
    /// Returns [`SilLevel::None`] if PFH ≥ 10⁻⁵ or if `pfh` is NaN / negative.
    pub fn classify_from_pfh(pfh: f64) -> SilLevel {
        if !pfh.is_finite() || pfh < 0.0 {
            return SilLevel::None;
        }
        if pfh < 1e-8 {
            SilLevel::Sil4
        } else if pfh < 1e-7 {
            SilLevel::Sil3
        } else if pfh < 1e-6 {
            SilLevel::Sil2
        } else if pfh < 1e-5 {
            SilLevel::Sil1
        } else {
            SilLevel::None
        }
    }

    /// Returns the numeric SIL level (0 = None, 1–4).
    pub fn level(self) -> u8 {
        match self {
            SilLevel::None => 0,
            SilLevel::Sil1 => 1,
            SilLevel::Sil2 => 2,
            SilLevel::Sil3 => 3,
            SilLevel::Sil4 => 4,
        }
    }

    /// Returns a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            SilLevel::None => "None",
            SilLevel::Sil1 => "SIL 1",
            SilLevel::Sil2 => "SIL 2",
            SilLevel::Sil3 => "SIL 3",
            SilLevel::Sil4 => "SIL 4",
        }
    }
}

/// Errors arising from SIL requirement evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilError {
    /// The achieved SIL is lower than the required SIL.
    RequirementNotMet {
        required: SilLevel,
        achieved: SilLevel,
    },
    /// PFD or PFH value is not a finite positive number.
    InvalidMetric,
}

impl core::fmt::Display for SilError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SilError::RequirementNotMet { required, achieved } => write!(
                f,
                "SIL requirement not met: required {}, achieved {}",
                required.label(),
                achieved.label(),
            ),
            SilError::InvalidMetric => write!(f, "SIL metric is not a finite positive value"),
        }
    }
}

/// A pair of required and achieved SIL levels with a satisfaction check.
///
/// Used to record whether a safety function meets its design-time SIL target.
///
/// # Example
/// ```
/// use oxictl::safety::sil::{SafetyRequirement, SilLevel};
///
/// let req = SafetyRequirement::new(SilLevel::Sil2, SilLevel::Sil3);
/// assert!(req.is_satisfied());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetyRequirement {
    /// The minimum SIL level required by the safety function specification.
    pub required_sil: SilLevel,
    /// The SIL level actually achieved by the implementation, as computed
    /// from PFD/PFH measurements or architectural analysis.
    pub achieved_sil: SilLevel,
}

impl SafetyRequirement {
    /// Construct a new requirement/achievement pair.
    pub fn new(required_sil: SilLevel, achieved_sil: SilLevel) -> Self {
        Self {
            required_sil,
            achieved_sil,
        }
    }

    /// Build from a required SIL and a measured PFD (low-demand mode).
    ///
    /// Classifies `achieved_pfd` and checks it against `required_sil`.
    /// Returns `Err(SilError::InvalidMetric)` if `achieved_pfd` is not finite
    /// and positive.
    pub fn from_pfd(required_sil: SilLevel, achieved_pfd: f64) -> Result<Self, SilError> {
        if !achieved_pfd.is_finite() || achieved_pfd < 0.0 {
            return Err(SilError::InvalidMetric);
        }
        let achieved_sil = SilLevel::classify_from_pfd(achieved_pfd);
        Ok(Self::new(required_sil, achieved_sil))
    }

    /// Build from a required SIL and a measured PFH (continuous mode).
    ///
    /// Returns `Err(SilError::InvalidMetric)` if `achieved_pfh` is not finite
    /// and positive.
    pub fn from_pfh(required_sil: SilLevel, achieved_pfh: f64) -> Result<Self, SilError> {
        if !achieved_pfh.is_finite() || achieved_pfh < 0.0 {
            return Err(SilError::InvalidMetric);
        }
        let achieved_sil = SilLevel::classify_from_pfh(achieved_pfh);
        Ok(Self::new(required_sil, achieved_sil))
    }

    /// Returns `true` if the achieved SIL meets or exceeds the required SIL.
    pub fn is_satisfied(&self) -> bool {
        self.achieved_sil >= self.required_sil
    }

    /// Verify the requirement, returning an error if not satisfied.
    pub fn verify(&self) -> Result<(), SilError> {
        if self.is_satisfied() {
            Ok(())
        } else {
            Err(SilError::RequirementNotMet {
                required: self.required_sil,
                achieved: self.achieved_sil,
            })
        }
    }

    /// How many SIL levels of margin the achieved level provides over the requirement.
    ///
    /// Returns 0 if the requirement is exactly met, a positive value if
    /// the achieved level exceeds it, or a negative value if it falls short.
    pub fn margin_levels(&self) -> i8 {
        self.achieved_sil.level() as i8 - self.required_sil.level() as i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PFD range containment ────────────────────────────────────────────────

    #[test]
    fn pfd_ranges_contain_midpoints() {
        // SIL 1: [1e-2, 1e-1)
        assert!(SilLevel::Sil1.pfd_range().contains(5e-2));
        // SIL 2: [1e-3, 1e-2)
        assert!(SilLevel::Sil2.pfd_range().contains(5e-3));
        // SIL 3: [1e-4, 1e-3)
        assert!(SilLevel::Sil3.pfd_range().contains(5e-4));
        // SIL 4: [1e-5, 1e-4)
        assert!(SilLevel::Sil4.pfd_range().contains(5e-5));
        // None: [1e-1, ∞)
        assert!(SilLevel::None.pfd_range().contains(0.5));
    }

    #[test]
    fn pfd_range_boundaries_are_exclusive_upper() {
        // Upper bound is exclusive: 1e-1 (= 0.1) is NOT in SIL 1 range [1e-2, 1e-1)
        // It belongs to the None range [1e-1, ∞)
        assert!(!SilLevel::Sil1.pfd_range().contains(1e-1));
        assert!(SilLevel::None.pfd_range().contains(1e-1));
        // Similarly 1e-2 is not in SIL 2 range [1e-3, 1e-2); it is the lower bound of SIL 1
        assert!(!SilLevel::Sil2.pfd_range().contains(1e-2));
        assert!(SilLevel::Sil1.pfd_range().contains(1e-2));
    }

    #[test]
    fn pfd_range_lower_boundary_is_inclusive() {
        // 1e-2 is the lower bound of SIL 1 → inclusive
        assert!(SilLevel::Sil1.pfd_range().contains(1e-2));
        // 1e-3 is the lower bound of SIL 2 → inclusive
        assert!(SilLevel::Sil2.pfd_range().contains(1e-3));
    }

    // ── PFH range containment ────────────────────────────────────────────────

    #[test]
    fn pfh_ranges_contain_midpoints() {
        assert!(SilLevel::Sil1.pfh_range().contains(5e-6));
        assert!(SilLevel::Sil2.pfh_range().contains(5e-7));
        assert!(SilLevel::Sil3.pfh_range().contains(5e-8));
        assert!(SilLevel::Sil4.pfh_range().contains(5e-9));
        assert!(SilLevel::None.pfh_range().contains(1e-4));
    }

    // ── classify_from_pfd ───────────────────────────────────────────────────

    #[test]
    fn classify_pfd_roundtrip() {
        let cases = [
            (5e-5, SilLevel::Sil4),
            (5e-4, SilLevel::Sil3),
            (5e-3, SilLevel::Sil2),
            (5e-2, SilLevel::Sil1),
            (0.5, SilLevel::None),
        ];
        for (pfd, expected) in cases {
            let got = SilLevel::classify_from_pfd(pfd);
            assert_eq!(got, expected, "PFD={pfd:.1e}");
            // The classified level's range must contain the input PFD
            assert!(
                got.pfd_range().contains(pfd) || got == SilLevel::None,
                "Range check failed for pfd={pfd:.1e}"
            );
        }
    }

    #[test]
    fn classify_pfd_boundary_values() {
        // PFD exactly at lower bound of SIL 2 → classified as SIL 2
        assert_eq!(SilLevel::classify_from_pfd(1e-3), SilLevel::Sil2);
        // PFD exactly at lower bound of SIL 1 → classified as SIL 1
        assert_eq!(SilLevel::classify_from_pfd(1e-2), SilLevel::Sil1);
    }

    #[test]
    fn classify_pfd_invalid_inputs() {
        assert_eq!(SilLevel::classify_from_pfd(f64::NAN), SilLevel::None);
        assert_eq!(SilLevel::classify_from_pfd(-1.0), SilLevel::None);
        assert_eq!(SilLevel::classify_from_pfd(f64::INFINITY), SilLevel::None);
    }

    // ── classify_from_pfh ───────────────────────────────────────────────────

    #[test]
    fn classify_pfh_roundtrip() {
        let cases = [
            (5e-9, SilLevel::Sil4),
            (5e-8, SilLevel::Sil3),
            (5e-7, SilLevel::Sil2),
            (5e-6, SilLevel::Sil1),
            (1e-4, SilLevel::None),
        ];
        for (pfh, expected) in cases {
            let got = SilLevel::classify_from_pfh(pfh);
            assert_eq!(got, expected, "PFH={pfh:.1e}");
        }
    }

    #[test]
    fn classify_pfh_invalid_inputs() {
        assert_eq!(SilLevel::classify_from_pfh(f64::NAN), SilLevel::None);
        assert_eq!(SilLevel::classify_from_pfh(-0.1), SilLevel::None);
    }

    // ── SilLevel ordering ───────────────────────────────────────────────────

    #[test]
    fn sil_ordering() {
        assert!(SilLevel::None < SilLevel::Sil1);
        assert!(SilLevel::Sil1 < SilLevel::Sil2);
        assert!(SilLevel::Sil2 < SilLevel::Sil3);
        assert!(SilLevel::Sil3 < SilLevel::Sil4);
    }

    // ── SafetyRequirement ───────────────────────────────────────────────────

    #[test]
    fn requirement_satisfied_when_achieved_meets_required() {
        let req = SafetyRequirement::new(SilLevel::Sil2, SilLevel::Sil2);
        assert!(req.is_satisfied());
        assert!(req.verify().is_ok());
    }

    #[test]
    fn requirement_satisfied_when_achieved_exceeds_required() {
        let req = SafetyRequirement::new(SilLevel::Sil2, SilLevel::Sil3);
        assert!(req.is_satisfied());
        assert_eq!(req.margin_levels(), 1);
    }

    #[test]
    fn requirement_not_satisfied_when_achieved_below_required() {
        let req = SafetyRequirement::new(SilLevel::Sil3, SilLevel::Sil2);
        assert!(!req.is_satisfied());
        assert_eq!(req.margin_levels(), -1);
        assert!(matches!(
            req.verify(),
            Err(SilError::RequirementNotMet { .. })
        ));
    }

    #[test]
    fn requirement_from_pfd_valid() {
        // PFD = 5e-4 → SIL 3; required SIL 2 → satisfied
        let req = SafetyRequirement::from_pfd(SilLevel::Sil2, 5e-4).unwrap();
        assert_eq!(req.achieved_sil, SilLevel::Sil3);
        assert!(req.is_satisfied());
    }

    #[test]
    fn requirement_from_pfd_invalid_metric() {
        let result = SafetyRequirement::from_pfd(SilLevel::Sil2, f64::NAN);
        assert_eq!(result, Err(SilError::InvalidMetric));
    }

    #[test]
    fn requirement_from_pfh_valid() {
        // PFH = 5e-8 → SIL 3; required SIL 3 → exactly satisfied
        let req = SafetyRequirement::from_pfh(SilLevel::Sil3, 5e-8).unwrap();
        assert_eq!(req.achieved_sil, SilLevel::Sil3);
        assert!(req.is_satisfied());
    }

    #[test]
    fn requirement_from_pfh_invalid_metric() {
        let result = SafetyRequirement::from_pfh(SilLevel::Sil1, -1.0);
        assert_eq!(result, Err(SilError::InvalidMetric));
    }
}
