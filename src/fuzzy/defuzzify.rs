//! Defuzzification methods for Mamdani fuzzy inference.
//!
//! All methods accept a slice of `(x, mu)` samples representing the aggregated
//! output membership function. The samples must be ordered by `x`.

use crate::core::scalar::ControlScalar;
use crate::fuzzy::FuzzyError;

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Validate that the sample slice is non-empty and that all `mu` values are
/// finite. Returns `FuzzyError::InsufficientSamples` if empty.
fn validate_samples<S: ControlScalar>(samples: &[(S, S)]) -> Result<(), FuzzyError> {
    if samples.is_empty() {
        return Err(FuzzyError::InsufficientSamples);
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// centroid_of_gravity (CoG)
// ────────────────────────────────────────────────────────────────────────────

/// Centroid-of-gravity (CoG) defuzzification via trapezoidal integration.
///
/// `mf_samples` is a slice of `(x, mu)` pairs ordered by `x`.
///
/// Returns `FuzzyError::InsufficientSamples` for empty or single-element
/// input (trapezoidal integration requires at least two points), and
/// `FuzzyError::DivisionByZero` when the total area is zero (all `mu = 0`).
pub fn centroid_of_gravity<S: ControlScalar>(mf_samples: &[(S, S)]) -> Result<S, FuzzyError> {
    validate_samples(mf_samples)?;
    if mf_samples.len() < 2 {
        return Err(FuzzyError::InsufficientSamples);
    }

    let mut numerator = S::ZERO;
    let mut denominator = S::ZERO;

    for window in mf_samples.windows(2) {
        let (x0, mu0) = window[0];
        let (x1, mu1) = window[1];
        let dx = x1 - x0;

        // Trapezoidal: area = dx * (mu0 + mu1) / 2
        let area = dx * (mu0 + mu1) * S::HALF;
        // Centroid of trapezoid strip: x_c = (x0*(2*mu0+mu1) + x1*(mu0+2*mu1)) / (3*(mu0+mu1))
        // But simpler: numerator contribution = area * x_centroid_of_strip
        // x_centroid = (x0 + x1)/2 for uniform, but trapezoidal centroid is:
        // (x0*(2*mu0+mu1) + x1*(mu0+2*mu1)) / (3*(mu0+mu1))
        let sum_mu = mu0 + mu1;
        let x_strip_centroid = if sum_mu <= S::ZERO {
            (x0 + x1) * S::HALF
        } else {
            let two = S::TWO;
            let three = S::from_f64(3.0);
            (x0 * (two * mu0 + mu1) + x1 * (mu0 + two * mu1)) / (three * sum_mu)
        };

        numerator += area * x_strip_centroid;
        denominator += area;
    }

    if denominator <= S::ZERO {
        return Err(FuzzyError::DivisionByZero);
    }
    Ok(numerator / denominator)
}

// ────────────────────────────────────────────────────────────────────────────
// mean_of_maxima (MoM)
// ────────────────────────────────────────────────────────────────────────────

/// Mean-of-maxima (MoM) defuzzification: average of all `x` values where `mu`
/// reaches its global maximum.
///
/// Returns `FuzzyError::InsufficientSamples` if the slice is empty or
/// `FuzzyError::DivisionByZero` if no sample has positive membership.
pub fn mean_of_maxima<S: ControlScalar>(mf_samples: &[(S, S)]) -> Result<S, FuzzyError> {
    validate_samples(mf_samples)?;

    // Find global maximum mu
    let max_mu =
        mf_samples
            .iter()
            .map(|&(_, mu)| mu)
            .fold(
                S::from_f64(f64::NEG_INFINITY),
                |a, b| if b > a { b } else { a },
            );

    if max_mu <= S::ZERO {
        return Err(FuzzyError::DivisionByZero);
    }

    // Collect all x values whose mu equals (or is within epsilon of) max_mu
    let threshold = max_mu - S::EPSILON * S::from_f64(1e6);
    let mut sum_x = S::ZERO;
    let mut count = S::ZERO;
    for &(x, mu) in mf_samples {
        if mu >= threshold {
            sum_x += x;
            count += S::ONE;
        }
    }

    if count <= S::ZERO {
        return Err(FuzzyError::DivisionByZero);
    }
    Ok(sum_x / count)
}

// ────────────────────────────────────────────────────────────────────────────
// bisector_of_area (BoA)
// ────────────────────────────────────────────────────────────────────────────

/// Bisector-of-area (BoA) defuzzification: the `x` value that divides the
/// area under the MF into two equal halves.
///
/// Uses trapezoidal integration and linear interpolation to find the exact
/// bisection point.
pub fn bisector_of_area<S: ControlScalar>(mf_samples: &[(S, S)]) -> Result<S, FuzzyError> {
    validate_samples(mf_samples)?;
    if mf_samples.len() < 2 {
        return Err(FuzzyError::InsufficientSamples);
    }

    // Compute total area via trapezoidal integration
    let mut total_area = S::ZERO;
    for window in mf_samples.windows(2) {
        let (x0, mu0) = window[0];
        let (x1, mu1) = window[1];
        let dx = x1 - x0;
        total_area += dx * (mu0 + mu1) * S::HALF;
    }

    if total_area <= S::ZERO {
        return Err(FuzzyError::DivisionByZero);
    }

    let half_area = total_area * S::HALF;

    // Walk segments accumulating area until we exceed half
    let mut cumulative = S::ZERO;
    for window in mf_samples.windows(2) {
        let (x0, mu0) = window[0];
        let (x1, mu1) = window[1];
        let dx = x1 - x0;
        let seg_area = dx * (mu0 + mu1) * S::HALF;

        if cumulative + seg_area >= half_area {
            // Bisector lies within this segment; solve for x via linear interp.
            let remaining = half_area - cumulative;
            // Solve: remaining = (mu0 + mu_t)/2 * t where mu_t = mu0 + (mu1-mu0)/dx * t
            // => (2*mu0 + (mu1-mu0)/dx * t) * t / 2 = remaining
            // => mu0*t + (mu1-mu0)/(2*dx)*t^2 = remaining
            let a_coeff = (mu1 - mu0) / (S::TWO * dx);
            let b_coeff = mu0;
            // Quadratic: a*t^2 + b*t - remaining = 0
            // t = (-b + sqrt(b^2 + 4*a*remaining)) / (2*a) if a != 0
            let t = if a_coeff.abs() < S::EPSILON * S::from_f64(1e6) {
                // Degenerate (mu0 ≈ mu1): uniform trapezoid → t = remaining / mu0
                if b_coeff <= S::ZERO {
                    dx * S::HALF // fallback midpoint
                } else {
                    remaining / b_coeff
                }
            } else {
                let discriminant = b_coeff * b_coeff + S::from_f64(4.0) * a_coeff * remaining;
                if discriminant < S::ZERO {
                    S::ZERO
                } else {
                    let two_a = S::TWO * a_coeff;
                    (-b_coeff + discriminant.sqrt()) / two_a
                }
            };
            // Clamp t to [0, dx]
            let t_clamped = t.clamp_val(S::ZERO, dx);
            return Ok(x0 + t_clamped);
        }
        cumulative += seg_area;
    }

    // Fallback: return last x value
    Ok(mf_samples.last().map(|&(x, _)| x).unwrap_or(S::ZERO))
}

// ────────────────────────────────────────────────────────────────────────────
// largest_of_maxima / smallest_of_maxima
// ────────────────────────────────────────────────────────────────────────────

/// Largest-of-maxima (LoM): returns the largest `x` where `mu` is maximum.
///
/// Returns `S::ZERO` if the slice is empty.
pub fn largest_of_maxima<S: ControlScalar>(mf_samples: &[(S, S)]) -> S {
    if mf_samples.is_empty() {
        return S::ZERO;
    }
    let max_mu =
        mf_samples
            .iter()
            .map(|&(_, mu)| mu)
            .fold(
                S::from_f64(f64::NEG_INFINITY),
                |a, b| if b > a { b } else { a },
            );

    let threshold = max_mu - S::EPSILON * S::from_f64(1e6);
    let mut best_x = mf_samples[0].0;
    for &(x, mu) in mf_samples {
        if mu >= threshold && x > best_x {
            best_x = x;
        }
    }
    best_x
}

/// Smallest-of-maxima (SoM): returns the smallest `x` where `mu` is maximum.
///
/// Returns `S::ZERO` if the slice is empty.
pub fn smallest_of_maxima<S: ControlScalar>(mf_samples: &[(S, S)]) -> S {
    if mf_samples.is_empty() {
        return S::ZERO;
    }
    let max_mu =
        mf_samples
            .iter()
            .map(|&(_, mu)| mu)
            .fold(
                S::from_f64(f64::NEG_INFINITY),
                |a, b| if b > a { b } else { a },
            );

    let threshold = max_mu - S::EPSILON * S::from_f64(1e6);
    let mut best_x = mf_samples
        .iter()
        .find(|&&(_, mu)| mu >= threshold)
        .map(|&(x, _)| x)
        .unwrap_or(S::ZERO);

    for &(x, mu) in mf_samples {
        if mu >= threshold && x < best_x {
            best_x = x;
        }
    }
    best_x
}

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a symmetric triangular MF over [0, 10] centered at 5.
    fn symmetric_triangle_samples(n: usize) -> Vec<(f64, f64)> {
        (0..=n)
            .map(|i| {
                let x = (i as f64) * 10.0 / (n as f64);
                let mu = if x <= 5.0 { x / 5.0 } else { (10.0 - x) / 5.0 };
                (x, mu)
            })
            .collect()
    }

    #[test]
    fn centroid_symmetric_triangle_is_center() {
        let samples = symmetric_triangle_samples(1000);
        let cog = centroid_of_gravity(&samples).unwrap();
        assert!(
            (cog - 5.0).abs() < 0.01,
            "CoG of symmetric triangle should be ~5.0, got {cog}"
        );
    }

    #[test]
    fn centroid_empty_returns_error() {
        let samples: Vec<(f64, f64)> = vec![];
        assert!(centroid_of_gravity(&samples).is_err());
    }

    #[test]
    fn centroid_all_zero_returns_division_by_zero() {
        let samples: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        assert!(matches!(
            centroid_of_gravity(&samples),
            Err(FuzzyError::DivisionByZero)
        ));
    }

    #[test]
    fn mean_of_maxima_flat_top() {
        // Trapezoid: max at [4, 6]
        let samples: Vec<(f64, f64)> = vec![(0.0, 0.0), (4.0, 1.0), (6.0, 1.0), (10.0, 0.0)];
        let mom = mean_of_maxima(&samples).unwrap();
        assert!((mom - 5.0).abs() < 0.01, "MoM should be 5.0, got {mom}");
    }

    #[test]
    fn mean_of_maxima_empty_returns_error() {
        let samples: Vec<(f64, f64)> = vec![];
        assert!(mean_of_maxima(&samples).is_err());
    }

    #[test]
    fn bisector_symmetric_triangle_is_center() {
        let samples = symmetric_triangle_samples(1000);
        let boa = bisector_of_area(&samples).unwrap();
        assert!(
            (boa - 5.0).abs() < 0.1,
            "BoA of symmetric triangle should be ~5.0, got {boa}"
        );
    }

    #[test]
    fn largest_of_maxima_flat_top() {
        let samples: Vec<(f64, f64)> = vec![(0.0, 0.0), (3.0, 1.0), (7.0, 1.0), (10.0, 0.0)];
        let lom = largest_of_maxima(&samples);
        assert!((lom - 7.0).abs() < 1e-9, "LoM should be 7.0, got {lom}");
    }

    #[test]
    fn smallest_of_maxima_flat_top() {
        let samples: Vec<(f64, f64)> = vec![(0.0, 0.0), (3.0, 1.0), (7.0, 1.0), (10.0, 0.0)];
        let som = smallest_of_maxima(&samples);
        assert!((som - 3.0).abs() < 1e-9, "SoM should be 3.0, got {som}");
    }

    #[test]
    fn largest_of_maxima_empty() {
        let samples: Vec<(f64, f64)> = vec![];
        assert_eq!(largest_of_maxima::<f64>(&samples), 0.0);
    }

    #[test]
    fn smallest_of_maxima_empty() {
        let samples: Vec<(f64, f64)> = vec![];
        assert_eq!(smallest_of_maxima::<f64>(&samples), 0.0);
    }
}
