use super::FreqError;
use crate::core::scalar::ControlScalar;
use crate::core::transfer_fn::TransferFn;
use heapless::Vec as HVec;

/// Evaluate H(e^{jω}) for a discrete-time TF, returning (re, im).
///
/// H(e^{jω}) = Σ_{k=0}^{N-1} b[k]*e^{-jωk} / (1 + Σ_{k=0}^{N-1} a[k]*e^{-jωk})
fn eval_tf_at_freq<S: ControlScalar, const N: usize>(tf: &TransferFn<S, N>, omega: S) -> (S, S) {
    let b = tf.b();
    let a = tf.a();

    // Numerator: Σ_{k=0}^{N-1} b[k] * e^{-jωk}  (b[k] is coeff of z^{-k})
    let mut num_re = S::ZERO;
    let mut num_im = S::ZERO;
    for (k, &b_k) in b.iter().enumerate().take(N) {
        let angle = -(omega * S::from_f64(k as f64));
        let (sin_a, cos_a) = angle.sin_cos();
        num_re += b_k * cos_a;
        num_im += b_k * sin_a;
    }

    // Denominator: 1 + Σ_{k=0}^{N-1} a[k] * e^{-jω(k+1)}  (a[k] is coeff of z^{-(k+1)})
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

    let re = (num_re * den_re + num_im * den_im) / den_mag_sq;
    let im = (num_im * den_re - num_re * den_im) / den_mag_sq;
    (re, im)
}

/// A single point on the Nyquist curve.
#[derive(Debug, Clone, Copy)]
pub struct NyquistPoint<S: ControlScalar> {
    /// Real part of H(e^{jω}).
    pub re: S,
    /// Imaginary part of H(e^{jω}).
    pub im: S,
    /// Angular frequency at which this point is evaluated.
    pub omega: S,
}

/// Collection of N Nyquist curve points stored in a heapless array.
pub struct NyquistData<S: ControlScalar, const N: usize> {
    pub points: HVec<NyquistPoint<S>, N>,
}

impl<S: ControlScalar, const N: usize> NyquistData<S, N> {
    /// Returns the number of points stored.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Compute the Nyquist curve for a discrete-time transfer function.
///
/// Evaluates H(e^{jω}) at N uniformly-spaced frequencies from 0 to `omega_max`
/// (typically π/T for normalized discrete-time, or π for ω in [0, π]).
///
/// # Errors
/// - [`FreqError::InvalidFrequencyRange`] if `omega_max <= 0`
/// - [`FreqError::InsufficientPoints`] if `N < 2`
pub fn compute_nyquist<S: ControlScalar, const TF_ORDER: usize, const N: usize>(
    tf: &TransferFn<S, TF_ORDER>,
    omega_max: S,
) -> Result<NyquistData<S, N>, FreqError> {
    if N < 2 {
        return Err(FreqError::InsufficientPoints);
    }
    if omega_max <= S::ZERO {
        return Err(FreqError::InvalidFrequencyRange);
    }

    let mut data: NyquistData<S, N> = NyquistData {
        points: HVec::new(),
    };

    let n_minus_one = S::from_f64((N - 1) as f64);
    for i in 0..N {
        let t = S::from_f64(i as f64) / n_minus_one;
        let omega = t * omega_max;
        let (re, im) = eval_tf_at_freq(tf, omega);
        let _ = data.points.push(NyquistPoint { re, im, omega });
    }

    Ok(data)
}

/// Compute the winding number of the Nyquist curve around the critical point (-1, 0).
///
/// Uses the signed angle change method: sum the signed angles subtended by each
/// successive pair of curve points as seen from (-1, 0). Divide by 2π to get
/// the winding number.
///
/// A positive count means counter-clockwise encirclements; a negative count means
/// clockwise encirclements (which indicate closed-loop instability when the number
/// of open-loop RHP poles is zero).
pub fn encirclement_count<S: ControlScalar, const N: usize>(data: &NyquistData<S, N>) -> i32 {
    let pts = &data.points;
    if pts.len() < 2 {
        return 0;
    }

    let mut total_angle = S::ZERO;
    let critical_re = S::from_f64(-1.0);
    let critical_im = S::ZERO;

    for i in 0..(pts.len() - 1) {
        // Vector from critical point to pts[i]
        let v0_re = pts[i].re - critical_re;
        let v0_im = pts[i].im - critical_im;
        // Vector from critical point to pts[i+1]
        let v1_re = pts[i + 1].re - critical_re;
        let v1_im = pts[i + 1].im - critical_im;

        // Signed angle from v0 to v1: atan2(cross, dot)
        // cross = v0_re * v1_im - v0_im * v1_re
        // dot   = v0_re * v1_re + v0_im * v1_im
        let cross = v0_re * v1_im - v0_im * v1_re;
        let dot = v0_re * v1_re + v0_im * v1_im;
        let angle = cross.atan2(dot);
        total_angle += angle;
    }

    let two_pi = S::from_f64(2.0 * core::f64::consts::PI);
    let winding = total_angle / two_pi;
    // Round to nearest integer
    let winding_f64 = winding.to_f64();
    if winding_f64 >= 0.0 {
        (winding_f64 + 0.5) as i32
    } else {
        (winding_f64 - 0.5) as i32
    }
}

/// Nyquist stability criterion for a unity-feedback loop.
///
/// Assumes the open-loop transfer function `tf` has no right-half-plane poles
/// (i.e., all open-loop poles are inside the unit circle for discrete-time).
/// Under this assumption, the closed-loop system is stable if and only if the
/// Nyquist curve does not encircle (-1, 0).
///
/// Returns `true` if stable (zero clockwise encirclements), `false` otherwise.
pub fn is_stable_nyquist<S: ControlScalar, const TF_ORDER: usize>(
    tf: &TransferFn<S, TF_ORDER>,
    n_points: usize,
) -> bool {
    // We can't use a const generic for n_points directly, so we use a fixed
    // large capacity and fill up to n_points.
    const MAX_PTS: usize = 512;
    let n_use = if n_points > MAX_PTS {
        MAX_PTS
    } else {
        n_points
    };

    let omega_max = S::PI; // Nyquist frequency for normalized discrete-time
                           // Build Nyquist data manually to avoid generic const issues
    let mut pts: HVec<NyquistPoint<S>, MAX_PTS> = HVec::new();

    if n_use < 2 {
        return true; // Cannot determine, assume stable
    }

    let n_minus_one = S::from_f64((n_use - 1) as f64);
    for i in 0..n_use {
        let t = S::from_f64(i as f64) / n_minus_one;
        let omega = t * omega_max;
        let (re, im) = eval_tf_at_freq(tf, omega);
        let _ = pts.push(NyquistPoint { re, im, omega });
    }

    // Compute winding number inline
    let mut total_angle = S::ZERO;
    let critical_re = S::from_f64(-1.0);

    for i in 0..(pts.len() - 1) {
        let v0_re = pts[i].re - critical_re;
        let v0_im = pts[i].im;
        let v1_re = pts[i + 1].re - critical_re;
        let v1_im = pts[i + 1].im;

        let cross = v0_re * v1_im - v0_im * v1_re;
        let dot = v0_re * v1_re + v0_im * v1_im;
        total_angle += cross.atan2(dot);
    }

    let two_pi = S::from_f64(2.0 * core::f64::consts::PI);
    let winding_f64 = (total_angle / two_pi).to_f64();
    let winding = if winding_f64 >= 0.0 {
        (winding_f64 + 0.5) as i32
    } else {
        (winding_f64 - 0.5) as i32
    };

    // Stable if no clockwise (negative) encirclements
    winding >= 0
}

/// Compute the minimum distance from the Nyquist curve to the critical point (-1, 0).
///
/// This is `min |H(e^{jω}) + 1|` over all evaluated frequencies. A larger distance
/// indicates more robust stability.
pub fn distance_to_critical<S: ControlScalar, const N: usize>(data: &NyquistData<S, N>) -> S {
    let critical_re = S::from_f64(-1.0);
    let mut min_dist = S::from_f64(f64::MAX);

    for pt in data.points.iter() {
        let dr = pt.re - critical_re;
        let di = pt.im;
        let dist = (dr * dr + di * di).sqrt();
        if dist < min_dist {
            min_dist = dist;
        }
    }

    if min_dist > S::from_f64(f64::MAX / 2.0) {
        S::ZERO
    } else {
        min_dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transfer_fn::TransferFn;

    /// Unity gain system H(z)=1: Nyquist curve is the point (1, 0), no encirclements.
    #[test]
    fn unity_gain_nyquist_no_encirclement() {
        let tf = TransferFn::<f64, 1>::new([1.0], [0.0]);
        let data = compute_nyquist::<f64, 1, 32>(&tf, core::f64::consts::PI).expect("nyquist ok");
        let enc = encirclement_count(&data);
        assert_eq!(enc, 0, "Unity gain has no encirclements of -1");
    }

    /// Attenuating system H(z)=0.5: still no encirclements.
    #[test]
    fn attenuating_nyquist_no_encirclement() {
        let tf = TransferFn::<f64, 1>::new([0.5], [0.0]);
        let data = compute_nyquist::<f64, 1, 64>(&tf, core::f64::consts::PI).expect("nyquist ok");
        let enc = encirclement_count(&data);
        assert_eq!(enc, 0, "Attenuating system should have no encirclements");
    }

    /// Verify point count matches const generic N.
    #[test]
    fn nyquist_point_count() {
        let tf = TransferFn::<f64, 1>::new([1.0], [0.0]);
        let data = compute_nyquist::<f64, 1, 32>(&tf, core::f64::consts::PI).expect("nyquist ok");
        assert_eq!(data.len(), 32, "Should have 32 Nyquist points");
    }

    /// Invalid omega_max should return error.
    #[test]
    fn nyquist_invalid_omega() {
        let tf = TransferFn::<f64, 1>::new([1.0], [0.0]);
        let result = compute_nyquist::<f64, 1, 8>(&tf, -1.0);
        assert!(
            matches!(result, Err(FreqError::InvalidFrequencyRange)),
            "Negative omega_max should error"
        );
    }

    /// First-order stable lowpass should be stable per Nyquist criterion.
    #[test]
    fn first_order_stable_nyquist() {
        let alpha = 0.5_f64;
        let tf = TransferFn::<f64, 1>::new([1.0 - alpha], [-alpha]);
        let stable = is_stable_nyquist::<f64, 1>(&tf, 128);
        assert!(stable, "First-order LP with α=0.5 should be stable");
    }

    /// Distance to critical point for unity gain system is 2.0 (point is at (1,0)).
    #[test]
    fn unity_distance_to_critical() {
        let tf = TransferFn::<f64, 1>::new([1.0], [0.0]);
        let data = compute_nyquist::<f64, 1, 64>(&tf, core::f64::consts::PI).expect("nyquist ok");
        let dist = distance_to_critical(&data);
        // At omega=0: H=1+0j, distance from -1 is 2
        // At omega=pi: H depends on coefficients
        assert!(dist > 0.0, "Distance should be positive");
        // For H(z)=1 the real part is 1 for all omega (b=[1], a=[0])
        // Actually: H(e^{jω}) = 1 for all ω, so distance = |1+1| = 2
        assert!(
            (dist - 2.0).abs() < 0.1,
            "Unity gain distance to -1 should be ~2, got {}",
            dist
        );
    }

    /// Test that Nyquist data starts near DC value at omega=0.
    #[test]
    fn nyquist_dc_value() {
        // H(z) at z=1 (omega=0): H = b[0]/(1+a[0])
        // For b=[0.5], a=[0.0]: H(1) = 0.5
        let tf = TransferFn::<f64, 1>::new([0.5], [0.0]);
        let data = compute_nyquist::<f64, 1, 16>(&tf, core::f64::consts::PI).expect("nyquist ok");
        let first = &data.points[0];
        assert!(
            (first.re - 0.5).abs() < 1e-9,
            "DC real part should be 0.5, got {}",
            first.re
        );
        assert!(
            first.im.abs() < 1e-9,
            "DC imag part should be 0, got {}",
            first.im
        );
    }
}
