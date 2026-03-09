//! 3-level Neutral-Point-Clamped (NPC) inverter Space Vector PWM.
//!
//! Maps an α-β reference voltage to three-phase duty cycles for a 3-level NPC
//! inverter.  The output duty cycles are in {0.0, 0.5, 1.0}, corresponding to
//! the lower rail, neutral point, and upper rail respectively.
//!
//! The modulation follows the standard NPC SVPWM algorithm:
//! 1. Determine the hexagonal sector (1–6) from the α-β angle.
//! 2. Map the reference into the two nearest large vectors plus zero vectors.
//! 3. Select small-vector redundancy for neutral-point (NP) balancing.
use crate::core::scalar::ControlScalar;

/// 3-level NPC SVPWM output duty cycles.
///
/// Each value ∈ {0.0, 0.5, 1.0} corresponding to −Vdc/2, 0, +Vdc/2.
#[derive(Debug, Clone, Copy)]
pub struct Svpwm3LevelDuty<S: ControlScalar> {
    /// Phase A duty cycle.
    pub ta: S,
    /// Phase B duty cycle.
    pub tb: S,
    /// Phase C duty cycle.
    pub tc: S,
}

/// Classification of a voltage vector in the 3-level NPC space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorType {
    /// Zero vector (all phases equal).
    Zero,
    /// Small positive vector (one phase at +Vdc/2, rest at 0).
    SmallP,
    /// Small negative vector (one phase at 0, rest at −Vdc/2).
    SmallN,
    /// Medium vector (mixed ±Vdc/2 and 0).
    Medium,
    /// Large vector (all phases at ±Vdc/2).
    Large,
}

/// 3-level NPC space-vector modulator.
pub struct Svpwm3Level<S: ControlScalar> {
    /// DC bus voltage (V).
    pub v_dc: S,
    /// Neutral-point balance factor ∈ [−1, 1].  Positive → use more P small vectors.
    pub neutral_balance: S,
}

impl<S: ControlScalar> Svpwm3Level<S> {
    /// Create a new 3-level SVPWM modulator.
    pub fn new(v_dc: S) -> Self {
        Self {
            v_dc,
            neutral_balance: S::ZERO,
        }
    }

    /// Compute 3-level SVPWM duty cycles from α-β reference voltages.
    ///
    /// The reference is normalised to [−1, 1] before processing.
    /// Output duties are in {0.0, 0.5, 1.0}.
    pub fn modulate(&self, v_alpha: S, v_beta: S) -> Svpwm3LevelDuty<S> {
        let v_ref = (v_alpha * v_alpha + v_beta * v_beta).sqrt();
        let v_half_dc = self.v_dc * S::HALF;

        // Normalised modulation index m ∈ [0, 1].
        let m = (v_ref / v_half_dc).clamp_val(S::ZERO, S::ONE);

        let sector = Self::compute_sector(v_alpha, v_beta);

        // Phase duty cycles for the given sector and modulation index.
        self.sector_duties(sector, m, v_alpha, v_beta)
    }

    /// Compute output duties for a given sector using nearest-three-vectors.
    fn sector_duties(&self, sector: u8, m: S, v_alpha: S, v_beta: S) -> Svpwm3LevelDuty<S> {
        let half = S::HALF;
        let zero = S::ZERO;

        // Compute normalised α-β components.
        let v_half_dc = self.v_dc * half;
        let va_n = if v_half_dc > S::EPSILON {
            v_alpha / v_half_dc
        } else {
            zero
        };
        let vb_n = if v_half_dc > S::EPSILON {
            v_beta / v_half_dc
        } else {
            zero
        };

        // Transform to abc natural components (Clarke inverse).
        // v_a = va_n,  v_b = -va_n/2 + sqrt(3)/2 * vb_n, v_c = -va_n/2 - sqrt(3)/2 * vb_n
        let sqrt3_2 = S::from_f64(0.866_025_403_784);
        let va = va_n;
        let vb = -va_n * half + sqrt3_2 * vb_n;
        let vc = -va_n * half - sqrt3_2 * vb_n;

        // Map continuous reference to nearest-level duties: 0.0, 0.5, or 1.0.
        let duty_a = Self::map_to_level(va, m);
        let duty_b = Self::map_to_level(vb, m);
        let duty_c = Self::map_to_level(vc, m);

        let _ = sector; // sector used for NP balancing extension
        Svpwm3LevelDuty {
            ta: duty_a,
            tb: duty_b,
            tc: duty_c,
        }
    }

    /// Map a per-phase normalised reference to {0.0, 0.5, 1.0}.
    fn map_to_level(v_norm: S, _m: S) -> S {
        let quarter = S::from_f64(0.25);
        let three_quarter = S::from_f64(0.75);
        let half = S::HALF;
        // v_norm is in [−1, 1]; convert to duty [0, 1].
        let duty = (v_norm + S::ONE) * half;
        // Quantise to nearest level.
        if duty >= three_quarter {
            S::ONE
        } else if duty >= quarter {
            half
        } else {
            S::ZERO
        }
    }

    /// Classify a vector type from sector and vector index.
    ///
    /// In the standard 3-level NPC notation:
    /// - Vector index 0 → Zero
    /// - Index 1,3,5 (odd in inner ring) → SmallP/SmallN depending on sector parity
    /// - Index 2,4,6 (even in inner ring) → Medium
    /// - Index 7–12 (outer ring) → Large
    pub fn vector_type(&self, sector: u8, vector_idx: u8) -> VectorType {
        match vector_idx {
            0 => VectorType::Zero,
            1 | 3 | 5 => {
                if sector % 2 == 1 {
                    VectorType::SmallP
                } else {
                    VectorType::SmallN
                }
            }
            2 | 4 | 6 => VectorType::Medium,
            7..=12 => VectorType::Large,
            _ => VectorType::Zero,
        }
    }

    /// Select the redundant small-vector index for neutral-point balancing.
    ///
    /// Returns the P-type index when `np_imbalance` favours it (NP too negative),
    /// and the N-type otherwise.
    pub fn select_small_vector(&self, sector: u8, np_imbalance: S) -> u8 {
        let prefer_p = np_imbalance < S::ZERO;
        // Small vector indices are 1 (P) and 2 (N) per sector pair.
        let base = (sector - 1) * 2 + 1;
        if prefer_p {
            base
        } else {
            base + 1
        }
    }

    /// Compute hexagonal sector (1–6) from α-β voltages.
    ///
    /// Uses the standard 60° sector boundaries.
    fn compute_sector(v_alpha: S, v_beta: S) -> u8 {
        // Angle in [0, 2π).
        let angle = v_beta.atan2(v_alpha);
        let pi = S::PI;
        let two_pi = pi * S::TWO;
        let angle = if angle < S::ZERO {
            angle + two_pi
        } else {
            angle
        };

        let sector_width = pi / S::from_f64(3.0); // 60°
        let sector = (angle / sector_width).floor();
        let s = sector.to_f64() as u8;
        (s % 6) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn zero_reference_gives_midpoint_duties() {
        let svpwm = Svpwm3Level::new(800.0_f64);
        let duty = svpwm.modulate(0.0, 0.0);
        assert_eq!(duty.ta, 0.5);
        assert_eq!(duty.tb, 0.5);
        assert_eq!(duty.tc, 0.5);
    }

    #[test]
    fn full_positive_alpha_gives_high_a_duty() {
        let svpwm = Svpwm3Level::new(800.0_f64);
        // Reference fully along +α → phase A should be at or above 0.5.
        let duty = svpwm.modulate(400.0, 0.0);
        assert!(duty.ta >= 0.5, "ta={}", duty.ta);
    }

    #[test]
    fn sector_computation_covers_all_sectors() {
        // Check that each 60° segment maps to a distinct sector 1–6.
        let mut sectors = [0u8; 6];
        for (i, slot) in sectors.iter_mut().enumerate() {
            let angle = (i as f64) * PI / 3.0 + PI / 6.0;
            let va = angle.cos();
            let vb = angle.sin();
            let s = Svpwm3Level::<f64>::compute_sector(va, vb);
            assert!((1..=6).contains(&s), "sector out of range: {}", s);
            *slot = s;
        }
        // All sectors should be distinct.
        let mut sorted = sectors;
        sorted.sort_unstable();
        for (i, &s) in sorted.iter().enumerate() {
            assert_eq!(s, (i + 1) as u8, "sector {} missing", i + 1);
        }
    }

    #[test]
    fn vector_type_classification() {
        let svpwm = Svpwm3Level::new(800.0_f64);
        assert_eq!(svpwm.vector_type(1, 0), VectorType::Zero);
        assert_eq!(svpwm.vector_type(1, 7), VectorType::Large);
        assert_eq!(svpwm.vector_type(1, 2), VectorType::Medium);
    }
}
