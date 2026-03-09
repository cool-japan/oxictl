//! Maximum Torque Per Ampere (MTPA) optimization for PMSM drives.
//!
//! Computes the optimal id/iq reference pair that produces a requested torque
//! command with minimum stator current magnitude. This minimizes copper losses
//! and enables higher efficiency, especially important in salient PMSMs where
//! reluctance torque contributes to total electromagnetic torque.
//!
//! # Motor Torque Equation
//!
//! For a salient PMSM with p pole pairs:
//! ```text
//! Te = (3/2) · p · [λ_pm · iq + (Ld - Lq) · id · iq]
//!    = (3/2) · p · iq · [λ_pm + (Ld - Lq) · id]
//! ```
//!
//! The stator current magnitude: Is² = id² + iq²
//!
//! # MTPA Condition
//!
//! Minimizing Is² subject to fixed Te gives the MTPA condition:
//! ```text
//! id_mtpa = λ_pm / (2·(Lq - Ld)) − √[(λ_pm / (2·(Lq - Ld)))² + iq²]
//! ```
//! For non-salient motors (Ld = Lq) the reluctance term vanishes → id_mtpa = 0.
//!
//! # Implementation Strategy
//!
//! A lookup table is precomputed over a range of torque commands at construction.
//! Per-cycle queries interpolate linearly between table entries, avoiding
//! trigonometric computation in the real-time loop.

use crate::core::scalar::ControlScalar;

/// Error type for MTPA construction and query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpaError {
    /// Table size must be at least 2.
    TableTooSmall,
    /// Maximum torque must be strictly positive.
    InvalidMaxTorque,
    /// Motor parameters are physically inconsistent.
    InvalidMotorParams,
}

impl core::fmt::Display for MtpaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TableTooSmall => write!(f, "MTPA table size must be >= 2"),
            Self::InvalidMaxTorque => write!(f, "max torque must be > 0"),
            Self::InvalidMotorParams => write!(f, "invalid motor parameters for MTPA"),
        }
    }
}

/// Single MTPA operating point.
#[derive(Debug, Clone, Copy)]
pub struct MtpaPoint<S: ControlScalar> {
    /// Torque at this operating point (N·m).
    pub torque: S,
    /// Optimal d-axis current reference (A). ≤ 0 for motoring.
    pub id_ref: S,
    /// Optimal q-axis current reference (A). Sign matches torque sign.
    pub iq_ref: S,
    /// Stator current magnitude (A).
    pub is_magnitude: S,
}

/// Motor parameters required for MTPA calculation.
#[derive(Debug, Clone, Copy)]
pub struct MtpaMotorParams<S: ControlScalar> {
    /// Number of pole pairs.
    pub pole_pairs: u32,
    /// d-axis inductance Ld (H).
    pub ld: S,
    /// q-axis inductance Lq (H).
    pub lq: S,
    /// Permanent magnet flux linkage λ_pm (Wb).
    pub lambda_pm: S,
    /// Maximum stator current magnitude (A). Used to bound the table.
    pub i_s_max: S,
}

/// Fixed-size MTPA lookup table using `heapless::Vec`.
///
/// The table is indexed from minimum to maximum positive torque; negative
/// torque is handled by sign symmetry.
///
/// `N` is the maximum number of table entries (const generic capacity for
/// `heapless::Vec`).
#[derive(Debug, Clone)]
pub struct MtpaTable<S: ControlScalar, const N: usize> {
    /// Precomputed operating points (positive torque half only).
    table: heapless::Vec<MtpaPoint<S>, N>,
    /// Motor parameters stored for reference.
    params: MtpaMotorParams<S>,
    /// Whether the motor is salient (|Ld - Lq| > threshold).
    is_salient: bool,
}

impl<S: ControlScalar, const N: usize> MtpaTable<S, N> {
    /// Construct and precompute the MTPA lookup table.
    ///
    /// # Arguments
    /// * `params` - Motor parameters including maximum stator current.
    /// * `n_points` - Number of table entries (2 ≤ n_points ≤ N).
    ///
    /// # Errors
    /// Returns `MtpaError` if parameters are invalid or n_points < 2.
    pub fn new(params: MtpaMotorParams<S>, n_points: usize) -> Result<Self, MtpaError> {
        if n_points < 2 {
            return Err(MtpaError::TableTooSmall);
        }
        if n_points > N {
            return Err(MtpaError::TableTooSmall);
        }
        if params.lambda_pm <= S::ZERO || params.i_s_max <= S::ZERO {
            return Err(MtpaError::InvalidMotorParams);
        }
        if params.ld <= S::ZERO || params.lq <= S::ZERO {
            return Err(MtpaError::InvalidMotorParams);
        }

        let saliency_ratio = params.lq - params.ld;
        let saliency_threshold = S::from_f64(1e-6);
        let is_salient = if saliency_ratio < S::ZERO {
            -saliency_ratio > saliency_threshold
        } else {
            saliency_ratio > saliency_threshold
        };

        // Maximum torque attainable at i_s_max
        // For a quick bound: Te_max_approx = (3/2)·p·λ_pm·I_s_max
        let three_half = S::from_f64(1.5);
        let p = S::from_f64(params.pole_pairs as f64);
        let te_max = three_half * p * params.lambda_pm * params.i_s_max;

        if te_max <= S::ZERO {
            return Err(MtpaError::InvalidMaxTorque);
        }

        let mut table: heapless::Vec<MtpaPoint<S>, N> = heapless::Vec::new();

        // Build table: torque from 0 to te_max in n_points steps
        let n_f = S::from_f64((n_points - 1) as f64);
        for k in 0..n_points {
            let frac = S::from_f64(k as f64) / n_f;
            let torque = frac * te_max;

            let (id_ref, iq_ref) = compute_mtpa_point(&params, torque, is_salient);
            let is_mag = (id_ref * id_ref + iq_ref * iq_ref).sqrt();

            let point = MtpaPoint {
                torque,
                id_ref,
                iq_ref,
                is_magnitude: is_mag,
            };

            // heapless::Vec::push returns Err if full; we checked n_points <= N above.
            table.push(point).ok();
        }

        Ok(Self {
            table,
            params,
            is_salient,
        })
    }

    /// Query id/iq references for a given torque command via linear interpolation.
    ///
    /// Negative torque is handled by sign symmetry: the table covers only
    /// positive torque; for negative torque_cmd, the signs of id_ref and iq_ref
    /// are flipped (id stays negative, iq flips).
    ///
    /// # Arguments
    /// * `torque_cmd` - Requested electromagnetic torque (N·m).
    ///
    /// # Returns
    /// `(id_ref, iq_ref)` — optimal current references in amperes.
    pub fn query(&self, torque_cmd: S) -> (S, S) {
        if self.table.is_empty() {
            return (S::ZERO, S::ZERO);
        }

        let sign = if torque_cmd < S::ZERO {
            -S::ONE
        } else {
            S::ONE
        };
        let t_abs = if torque_cmd < S::ZERO {
            -torque_cmd
        } else {
            torque_cmd
        };

        // Clamp to table range
        let t_max = self.table[self.table.len() - 1].torque;
        let t_clamped = if t_abs > t_max { t_max } else { t_abs };

        // Binary search for bracket [lo, hi]
        let (lo_idx, hi_idx) = self.find_bracket(t_clamped);

        let lo = &self.table[lo_idx];
        let hi = &self.table[hi_idx];

        let (id_ref, iq_ref) = if lo_idx == hi_idx {
            (lo.id_ref, lo.iq_ref)
        } else {
            let dt = hi.torque - lo.torque;
            let alpha = if dt > S::ZERO {
                (t_clamped - lo.torque) / dt
            } else {
                S::ZERO
            };
            let id = lo.id_ref + alpha * (hi.id_ref - lo.id_ref);
            let iq = lo.iq_ref + alpha * (hi.iq_ref - lo.iq_ref);
            (id, iq)
        };

        // Apply sign: id stays negative (demagnetizing), iq tracks torque sign
        (id_ref, sign * iq_ref)
    }

    /// Whether the table was built for a salient motor.
    pub fn is_salient(&self) -> bool {
        self.is_salient
    }

    /// Number of table entries.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Motor parameters used to build this table.
    pub fn params(&self) -> &MtpaMotorParams<S> {
        &self.params
    }

    /// Retrieve a precomputed operating point by index.
    pub fn point(&self, index: usize) -> Option<&MtpaPoint<S>> {
        self.table.get(index)
    }

    /// Find the index pair [lo, hi] bracketing `torque` via linear scan.
    /// For small N this is faster than binary search due to cache locality.
    fn find_bracket(&self, torque: S) -> (usize, usize) {
        let n = self.table.len();
        if n == 0 {
            return (0, 0);
        }
        if n == 1 {
            return (0, 0);
        }
        // Linear scan from the left
        for i in 0..n - 1 {
            if torque <= self.table[i + 1].torque {
                return (i, i + 1);
            }
        }
        (n - 1, n - 1)
    }
}

/// Compute the MTPA id/iq pair for a specific torque command.
///
/// For salient motors: solves the MTPA optimality condition analytically.
/// For non-salient (Ld = Lq): id = 0, iq derived from torque equation.
///
/// Returns `(id_ref, iq_ref)` for positive torque.
fn compute_mtpa_point<S: ControlScalar>(
    params: &MtpaMotorParams<S>,
    torque: S,
    is_salient: bool,
) -> (S, S) {
    let three_half = S::from_f64(1.5);
    let p = S::from_f64(params.pole_pairs as f64);
    let lam = params.lambda_pm;
    let ld = params.ld;
    let lq = params.lq;

    if !is_salient || torque <= S::ZERO {
        // Non-salient: id = 0, iq = Te / (1.5·p·λ_pm)
        let denom = three_half * p * lam;
        let iq = if denom > S::from_f64(1e-12) {
            torque / denom
        } else {
            S::ZERO
        };
        return (S::ZERO, iq);
    }

    // Salient: MTPA condition derived from Lagrange optimisation.
    // id_mtpa = λ_pm / (2·(Lq − Ld)) − √[(λ_pm / (2·(Lq − Ld)))² + iq²]
    // Coupled: Te = 1.5·p·[λ_pm·iq + (Ld − Lq)·id·iq]
    //
    // We iterate: start with id=0, find iq from torque, recompute id, repeat.
    // Typically converges in 3–5 iterations.

    let delta_l = lq - ld; // > 0 for interior PMSM (Ld < Lq)
    let two_delta_l = delta_l * S::TWO;
    let lam_over_2dl = if two_delta_l.abs() > S::from_f64(1e-12) {
        lam / two_delta_l
    } else {
        S::ZERO
    };

    // Initial iq estimate (no-id approximation)
    let denom0 = three_half * p * lam;
    let mut iq = if denom0 > S::from_f64(1e-12) {
        torque / denom0
    } else {
        S::ZERO
    };
    let mut id = S::ZERO;

    for _ in 0..8 {
        // MTPA id from current iq
        let iq_sq = iq * iq;
        let discriminant = lam_over_2dl * lam_over_2dl + iq_sq;
        id = if discriminant >= S::ZERO {
            lam_over_2dl - discriminant.sqrt()
        } else {
            S::ZERO
        };

        // Update iq from torque equation
        // Te = 1.5·p·iq·(λ_pm + (Ld − Lq)·id)
        let reluctance_term = (ld - lq) * id; // negative for interior PMSM
        let effective_lam = lam + reluctance_term;
        let denom = three_half * p * effective_lam;
        iq = if denom.abs() > S::from_f64(1e-12) {
            torque / denom
        } else {
            iq // keep previous if degenerate
        };
    }

    (id, iq)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_salient_params() -> MtpaMotorParams<f64> {
        MtpaMotorParams {
            pole_pairs: 4,
            ld: 3.0e-4,
            lq: 3.0e-4, // identical → non-salient
            lambda_pm: 0.05,
            i_s_max: 10.0,
        }
    }

    fn salient_params() -> MtpaMotorParams<f64> {
        MtpaMotorParams {
            pole_pairs: 4,
            ld: 2.5e-4,
            lq: 4.0e-4, // Lq > Ld → interior PMSM
            lambda_pm: 0.05,
            i_s_max: 10.0,
        }
    }

    #[test]
    fn non_salient_id_is_zero() {
        let params = non_salient_params();
        let table = MtpaTable::<f64, 64>::new(params, 32).expect("table creation failed");
        assert!(!table.is_salient());
        // For non-salient, all id references should be ~0
        for i in 0..table.len() {
            let pt = table.point(i).unwrap();
            assert!(
                pt.id_ref.abs() < 1e-9,
                "id_ref should be 0 for non-salient, got {}",
                pt.id_ref
            );
        }
    }

    #[test]
    fn query_zero_torque_gives_zero_current() {
        let params = non_salient_params();
        let table = MtpaTable::<f64, 64>::new(params, 32).expect("table creation failed");
        let (id, iq) = table.query(0.0);
        assert!(id.abs() < 1e-9);
        assert!(iq.abs() < 1e-9);
    }

    #[test]
    fn negative_torque_flips_iq_sign() {
        let params = non_salient_params();
        let table = MtpaTable::<f64, 64>::new(params, 32).expect("table creation failed");
        let (_, iq_pos) = table.query(1.0);
        let (_, iq_neg) = table.query(-1.0);
        assert!(iq_pos > 0.0);
        assert!(iq_neg < 0.0);
        assert!((iq_pos + iq_neg).abs() < 1e-9);
    }

    #[test]
    fn salient_id_is_negative() {
        let params = salient_params();
        let table = MtpaTable::<f64, 64>::new(params, 32).expect("table creation failed");
        assert!(table.is_salient());
        // For salient positive-torque points beyond zero, id_ref should be <= 0
        for i in 1..table.len() {
            let pt = table.point(i).unwrap();
            assert!(
                pt.id_ref <= 0.0,
                "id_ref={} should be ≤ 0 for salient motor (demagnetising)",
                pt.id_ref
            );
        }
    }

    #[test]
    fn table_too_small_returns_error() {
        let params = non_salient_params();
        let result = MtpaTable::<f64, 64>::new(params, 1);
        assert_eq!(result.unwrap_err(), MtpaError::TableTooSmall);
    }

    #[test]
    fn invalid_lambda_returns_error() {
        let mut params = non_salient_params();
        params.lambda_pm = 0.0;
        let result = MtpaTable::<f64, 64>::new(params, 32);
        assert!(result.is_err());
    }

    #[test]
    fn torque_monotone_with_iq_non_salient() {
        let params = non_salient_params();
        let table = MtpaTable::<f64, 64>::new(params, 32).expect("table creation failed");
        // iq should increase monotonically with torque for non-salient
        for i in 1..table.len() {
            let prev = table.point(i - 1).unwrap();
            let curr = table.point(i).unwrap();
            assert!(
                curr.iq_ref >= prev.iq_ref - 1e-9,
                "iq should be non-decreasing: prev={} curr={}",
                prev.iq_ref,
                curr.iq_ref
            );
        }
    }
}
