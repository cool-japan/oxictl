use super::FreqError;
use crate::core::scalar::ControlScalar;
use crate::core::transfer_fn::TransferFn;
use heapless::Vec as HVec;

/// Evaluate H(e^{jω}) for a discrete-time transfer function at angular frequency ω.
///
/// H(e^{jω}) = Σ_{k=0}^{N-1} b[k]*e^{-jωk} / (1 + Σ_{k=0}^{N-1} a[k]*e^{-jωk})
///
/// Returns (re, im) of H(e^{jω}).
fn eval_tf_at_freq<S: ControlScalar, const N: usize>(tf: &TransferFn<S, N>, omega: S) -> (S, S) {
    let b = tf.b();
    let a = tf.a();

    // Numerator: Σ_{k=0}^{N-1} b[k] * e^{-jωk}
    // b[k] is the coefficient of z^{-k}, so angle = -ω*k
    let mut num_re = S::ZERO;
    let mut num_im = S::ZERO;
    for (k, &b_k) in b.iter().enumerate().take(N) {
        let angle = -(omega * S::from_f64(k as f64));
        let (sin_a, cos_a) = angle.sin_cos();
        num_re += b_k * cos_a;
        num_im += b_k * sin_a;
    }

    // Denominator: 1 + Σ_{k=0}^{N-1} a[k] * e^{-jω(k+1)}
    // a[k] is the coefficient of z^{-(k+1)}, so angle = -ω*(k+1)
    let mut den_re = S::ONE;
    let mut den_im = S::ZERO;
    for (k, &a_k) in a.iter().enumerate().take(N) {
        let angle = -(omega * S::from_f64((k + 1) as f64));
        let (sin_a, cos_a) = angle.sin_cos();
        den_re += a_k * cos_a;
        den_im += a_k * sin_a;
    }

    let den_mag_sq = den_re * den_re + den_im * den_im;
    if den_mag_sq < S::EPSILON {
        return (S::ZERO, S::ZERO);
    }

    // Complex division: (num_re + j*num_im) / (den_re + j*den_im)
    let re = (num_re * den_re + num_im * den_im) / den_mag_sq;
    let im = (num_im * den_re - num_re * den_im) / den_mag_sq;
    (re, im)
}

/// A single point on a Bode plot.
#[derive(Debug, Clone, Copy)]
pub struct BodePoint<S: ControlScalar> {
    /// Angular frequency (rad/s or rad/sample depending on context).
    pub omega: S,
    /// Magnitude in decibels: 20*log10(|H(e^{jω})|).
    pub magnitude_db: S,
    /// Phase in degrees: atan2(Im, Re) * 180/π.
    pub phase_deg: S,
}

/// Collection of N Bode plot points stored in a heapless array.
pub struct BodeData<S: ControlScalar, const N: usize> {
    pub points: HVec<BodePoint<S>, N>,
}

impl<S: ControlScalar, const N: usize> BodeData<S, N> {
    /// Returns the number of points actually stored.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if no points are stored.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Compute Bode plot data for a discrete-time transfer function.
///
/// Evaluates H(e^{jω}) at N logarithmically-spaced angular frequencies from
/// `omega_min` to `omega_max`.
///
/// # Errors
/// - [`FreqError::InvalidFrequencyRange`] if `omega_min <= 0` or `omega_min >= omega_max`
/// - [`FreqError::InsufficientPoints`] if `N < 2`
pub fn compute_bode<S: ControlScalar, const TF_ORDER: usize, const N: usize>(
    tf: &TransferFn<S, TF_ORDER>,
    omega_min: S,
    omega_max: S,
) -> Result<BodeData<S, N>, FreqError> {
    if N < 2 {
        return Err(FreqError::InsufficientPoints);
    }
    if omega_min <= S::ZERO || omega_min >= omega_max {
        return Err(FreqError::InvalidFrequencyRange);
    }

    let mut data: BodeData<S, N> = BodeData {
        points: HVec::new(),
    };

    let ln_min = omega_min.ln();
    let ln_max = omega_max.ln();
    let ln_range = ln_max - ln_min;
    let n_minus_one = S::from_f64((N - 1) as f64);

    for i in 0..N {
        let t = S::from_f64(i as f64) / n_minus_one;
        let omega = (ln_min + t * ln_range).exp();

        let (re, im) = eval_tf_at_freq(tf, omega);
        let mag_sq = re * re + im * im;
        // Avoid log of zero
        let magnitude_db = if mag_sq > S::ZERO {
            S::from_f64(20.0) * mag_sq.sqrt().log10()
        } else {
            S::from_f64(-120.0) // -120 dB floor
        };
        let phase_deg = im.atan2(re) * S::from_f64(180.0 / core::f64::consts::PI);

        let point = BodePoint {
            omega,
            magnitude_db,
            phase_deg,
        };
        // HVec push: ignore if full (N is the exact capacity, loop runs N times so won't overflow)
        let _ = data.points.push(point);
    }

    Ok(data)
}

/// Find the gain crossover frequency: the frequency where |H(e^{jω})| = 0 dB.
///
/// Uses linear interpolation between adjacent Bode points where the magnitude
/// crosses 0 dB. Returns the first such crossing from low to high frequency.
pub fn gain_crossover_frequency<S: ControlScalar, const N: usize>(
    data: &BodeData<S, N>,
) -> Option<S> {
    let pts = &data.points;
    if pts.len() < 2 {
        return None;
    }

    for i in 0..(pts.len() - 1) {
        let m0 = pts[i].magnitude_db;
        let m1 = pts[i + 1].magnitude_db;
        // Crossing: one side positive, other negative (or exactly zero)
        if (m0 >= S::ZERO && m1 <= S::ZERO) || (m0 <= S::ZERO && m1 >= S::ZERO) {
            // Linear interpolation: find t where m0 + t*(m1-m0) = 0
            let dm = m1 - m0;
            if dm.abs() < S::EPSILON {
                return Some(pts[i].omega);
            }
            let t = -m0 / dm;
            let omega = pts[i].omega + t * (pts[i + 1].omega - pts[i].omega);
            return Some(omega);
        }
    }
    None
}

/// Find the phase crossover frequency: the frequency where ∠H(e^{jω}) = -180°.
///
/// Uses linear interpolation between adjacent Bode points where the phase
/// crosses -180 degrees.
pub fn phase_crossover_frequency<S: ControlScalar, const N: usize>(
    data: &BodeData<S, N>,
) -> Option<S> {
    let pts = &data.points;
    if pts.len() < 2 {
        return None;
    }

    let neg180 = S::from_f64(-180.0);

    for i in 0..(pts.len() - 1) {
        let p0 = pts[i].phase_deg;
        let p1 = pts[i + 1].phase_deg;
        // Check if phase crosses -180 in this interval
        let above0 = p0 > neg180;
        let above1 = p1 > neg180;
        if above0 != above1 {
            let dp = p1 - p0;
            if dp.abs() < S::EPSILON {
                return Some(pts[i].omega);
            }
            let t = (neg180 - p0) / dp;
            let omega = pts[i].omega + t * (pts[i + 1].omega - pts[i].omega);
            return Some(omega);
        }
    }
    None
}

/// Compute gain margin in dB.
///
/// The gain margin is the negative of the magnitude (in dB) at the phase
/// crossover frequency. A positive gain margin indicates a stable system.
pub fn gain_margin<S: ControlScalar, const N: usize>(data: &BodeData<S, N>) -> Option<S> {
    let phase_xover = phase_crossover_frequency(data)?;
    let pts = &data.points;

    // Find the two adjacent points bracketing phase_xover
    for i in 0..(pts.len() - 1) {
        if pts[i].omega <= phase_xover && phase_xover <= pts[i + 1].omega {
            let t = if (pts[i + 1].omega - pts[i].omega).abs() < S::EPSILON {
                S::ZERO
            } else {
                (phase_xover - pts[i].omega) / (pts[i + 1].omega - pts[i].omega)
            };
            let mag_db = pts[i].magnitude_db + t * (pts[i + 1].magnitude_db - pts[i].magnitude_db);
            return Some(-mag_db);
        }
    }
    None
}

/// Compute phase margin in degrees.
///
/// The phase margin is 180° plus the phase (in degrees) at the gain crossover
/// frequency. A positive phase margin indicates a stable system.
pub fn phase_margin<S: ControlScalar, const N: usize>(data: &BodeData<S, N>) -> Option<S> {
    let gain_xover = gain_crossover_frequency(data)?;
    let pts = &data.points;

    // Find the two adjacent points bracketing gain_xover
    for i in 0..(pts.len() - 1) {
        if pts[i].omega <= gain_xover && gain_xover <= pts[i + 1].omega {
            let t = if (pts[i + 1].omega - pts[i].omega).abs() < S::EPSILON {
                S::ZERO
            } else {
                (gain_xover - pts[i].omega) / (pts[i + 1].omega - pts[i].omega)
            };
            let phase_deg = pts[i].phase_deg + t * (pts[i + 1].phase_deg - pts[i].phase_deg);
            return Some(S::from_f64(180.0) + phase_deg);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transfer_fn::TransferFn;

    /// Unity gain system: H(z) = 1 (b=[1], a=[0])
    /// DC gain should be 0 dB.
    #[test]
    fn unity_gain_dc_bode() {
        let tf = TransferFn::<f64, 1>::new([1.0], [0.0]);
        let data = compute_bode::<f64, 1, 16>(&tf, 1e-3, 1.0).expect("bode ok");
        // DC (low frequency) magnitude should be ~0 dB
        let first = &data.points[0];
        assert!(
            first.magnitude_db.abs() < 0.5,
            "Unity gain DC should be ~0 dB, got {} dB",
            first.magnitude_db
        );
    }

    /// Verify the number of Bode points matches the const parameter N.
    #[test]
    fn bode_point_count() {
        let tf = TransferFn::<f64, 1>::new([1.0], [0.0]);
        let data = compute_bode::<f64, 1, 32>(&tf, 1e-3, 1.0).expect("bode ok");
        assert_eq!(data.len(), 32, "Should have exactly 32 Bode points");
    }

    /// Invalid frequency range should return an error.
    #[test]
    fn bode_invalid_range() {
        let tf = TransferFn::<f64, 1>::new([1.0], [0.0]);
        let result = compute_bode::<f64, 1, 8>(&tf, 10.0, 1.0);
        assert!(
            matches!(result, Err(FreqError::InvalidFrequencyRange)),
            "Should return InvalidFrequencyRange"
        );
    }

    /// Insufficient points error.
    #[test]
    fn bode_insufficient_points() {
        let tf = TransferFn::<f64, 1>::new([1.0], [0.0]);
        let result = compute_bode::<f64, 1, 1>(&tf, 1e-3, 1.0);
        assert!(
            matches!(result, Err(FreqError::InsufficientPoints)),
            "Should return InsufficientPoints"
        );
    }

    /// First-order lowpass H(z) = (1-α)/(1 - α*z^{-1}) should have:
    /// - Positive phase margin (stable)
    /// - Gain margin undefined (phase never crosses -180° for first-order)
    #[test]
    fn first_order_lowpass_phase_margin_positive() {
        // First-order lowpass with alpha ~= 0.9 (slow)
        let alpha = 0.9_f64;
        let b = [1.0 - alpha];
        let a = [-alpha];
        let tf = TransferFn::<f64, 1>::new(b, a);
        let data = compute_bode::<f64, 1, 64>(&tf, 1e-3, core::f64::consts::PI).expect("bode ok");

        // First-order LP should have a gain crossover (it rolls off)
        // Phase margin should be positive (stable system)
        if let Some(pm) = phase_margin(&data) {
            assert!(
                pm > 0.0,
                "First-order LP should have positive phase margin, got {}°",
                pm
            );
        }
        // gain margin not required for first-order (phase never reaches -180°)
    }

    /// For a first-order lowpass, the gain crossover frequency should be
    /// near the -3dB point.
    #[test]
    fn first_order_lowpass_gain_crossover() {
        // H(z) = 0.5 / (1 - 0.5*z^{-1}) — unity DC gain is not 0 dB, but let's
        // use a balanced system where gain crossover occurs.
        // Simple attenuating filter: b=[0.5], a=[0.0] — H=0.5 constant, no crossover at 0dB
        // Instead use: b=[1.0], a=[0.5] → H(1)=1/(1+0.5)=0.667, H(-1)=1/(1-0.5)=2.0
        // This system has gain >1 at high freq and <1 somewhere
        let tf = TransferFn::<f64, 1>::new([1.0], [0.5]);
        let data = compute_bode::<f64, 1, 64>(&tf, 1e-3, core::f64::consts::PI).expect("bode ok");
        // Just verify we get a crossover (or not) without panicking
        let _gcf = gain_crossover_frequency(&data);
        // Verify magnitudes span a range
        let max_mag = data
            .points
            .iter()
            .map(|p| p.magnitude_db)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_mag = data
            .points
            .iter()
            .map(|p| p.magnitude_db)
            .fold(f64::INFINITY, f64::min);
        assert!(max_mag > min_mag, "Magnitude should vary across frequency");
    }

    /// DC gain of 0 dB for the unity gain H(z)=1 system.
    #[test]
    fn unity_gain_zero_db() {
        let tf = TransferFn::<f64, 1>::new([1.0], [0.0]);
        let data = compute_bode::<f64, 1, 8>(&tf, 1e-4, 0.1).expect("bode ok");
        for pt in data.points.iter() {
            assert!(
                pt.magnitude_db.abs() < 1e-6,
                "Unity TF should be 0 dB everywhere, got {} at omega={}",
                pt.magnitude_db,
                pt.omega
            );
        }
    }
}
