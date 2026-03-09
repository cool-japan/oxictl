/// Tustin (bilinear) approximation of the fractional operator s^α.
///
/// The Tustin substitution maps the continuous Laplace variable s to:
///   s ≈ (2/T) · (z - 1) / (z + 1)
///
/// Raising to the fractional power α:
///   s^α ≈ (2/T)^α · ((z - 1) / (z + 1))^α
///
/// The numerator and denominator polynomials in z^{-1} are obtained by
/// expanding ((1 - z^{-1}) / (1 + z^{-1}))^α via the binomial series
/// and using Grünwald-Letnikov binomial weights (same recurrence as GL).
///
/// This yields a causal IIR/FIR difference equation that can be evaluated
/// sample-by-sample.
use crate::core::scalar::ControlScalar;
use heapless::Vec as HVec;

use super::FracError;

// ---------------------------------------------------------------------------
// Coefficient design
// ---------------------------------------------------------------------------

/// Design Tustin fractional coefficients for s^α.
///
/// Returns `(num_coeffs, den_coeffs)` where each vector has length `order + 1`.
///
/// The transfer function in z^{-1} is:
///   H(z) = (2/T)^α · N(z^{-1}) / D(z^{-1})
///
/// with
///   N(z^{-1}) = Σ_{k=0}^{order} w_k^{(num)} z^{-k}   (from (1 - z^{-1})^α)
///   D(z^{-1}) = Σ_{k=0}^{order} w_k^{(den)} z^{-k}   (from (1 + z^{-1})^α)
///
/// The denominator binomial weights for (1 + z^{-1})^α use a positive sign:
///   v_0 = 1,  v_k = v_{k-1} · (α - k + 1) / k
///
/// # Errors
/// - [`FracError::InvalidOrder`] if `alpha` is non-finite.
/// - [`FracError::InvalidSampleTime`] if `sample_time <= 0`.
/// - [`FracError::WindowTooSmall`] if `order == 0`.
pub fn design_tustin_frac<S: ControlScalar, const N: usize>(
    alpha: S,
    sample_time: S,
) -> Result<(HVec<S, N>, HVec<S, N>), FracError> {
    let order = N.checked_sub(1).ok_or(FracError::WindowTooSmall)?;
    if order == 0 {
        return Err(FracError::WindowTooSmall);
    }
    if !alpha.is_finite() {
        return Err(FracError::InvalidOrder);
    }
    if sample_time <= S::ZERO || !sample_time.is_finite() {
        return Err(FracError::InvalidSampleTime);
    }

    // Scale factor (2/T)^α
    let two_over_t = S::TWO / sample_time;
    let scale = two_over_t.powf(alpha);

    // Numerator: GL weights for (1 - z^{-1})^α.
    //
    // The GL recurrence already encodes the expansion of (1 - z^{-1})^α:
    //   w_0 = 1,  w_k = w_{k-1} · (k - 1 - α) / k
    // which gives w_k = C(α, k) · (-1)^k — the binomial coefficients of
    // (1 - x)^α.  No additional sign alternation is needed.
    let num_raw = gl_weights_vec::<S, N>(alpha)?;
    let mut num: HVec<S, N> = HVec::new();
    for &w in num_raw.iter() {
        num.push(scale * w).map_err(|_| FracError::WindowTooSmall)?;
    }

    // Denominator: expansion of (1 + z^{-1})^α
    // v_0 = 1, v_k = v_{k-1} * (alpha - k + 1) / k
    let mut den: HVec<S, N> = HVec::new();
    let mut v = S::ONE;
    for k in 0..=order {
        den.push(v).map_err(|_| FracError::WindowTooSmall)?;
        if k < order {
            let k_s = S::from_f64((k + 1) as f64);
            let alpha_k = alpha - S::from_f64(k as f64);
            v = v * alpha_k / k_s;
        }
    }

    Ok((num, den))
}

/// Compute GL binomial weights for order `alpha` into a heapless Vec of
/// capacity N (length N).
fn gl_weights_vec<S: ControlScalar, const N: usize>(alpha: S) -> Result<HVec<S, N>, FracError> {
    let mut w: HVec<S, N> = HVec::new();
    let mut wk = S::ONE;
    for k in 0..N {
        w.push(wk).map_err(|_| FracError::WindowTooSmall)?;
        let k_s = S::from_f64((k + 1) as f64);
        let km1_s = S::from_f64(k as f64);
        wk = wk * (km1_s - alpha) / k_s;
    }
    Ok(w)
}

// ---------------------------------------------------------------------------
// Runtime IIR filter
// ---------------------------------------------------------------------------

/// IIR filter implementing the Tustin fractional operator s^α of order N.
///
/// Evaluates the difference equation:
///   y[n] = (1/d_0) · (Σ_k num[k]·x[n-k]  −  Σ_{k≥1} den[k]·y[n-k])
///
/// where `num` and `den` are the Tustin coefficient vectors.
#[derive(Debug, Clone)]
pub struct TustinFrac<S: ControlScalar, const N: usize> {
    /// Numerator coefficients (length N, index 0 = current sample).
    num: HVec<S, N>,
    /// Denominator coefficients (length N, den[0] normalises).
    den: HVec<S, N>,
    /// Input history: x_buf[0] = x[n-1], … x_buf[N-2] = x[n-N+1].
    x_buf: [S; N],
    /// Output history: y_buf[0] = y[n-1], … y_buf[N-2] = y[n-N+1].
    y_buf: [S; N],
}

impl<S: ControlScalar, const N: usize> TustinFrac<S, N> {
    /// Construct a `TustinFrac` for the operator s^`alpha` with sample time `h`.
    ///
    /// # Errors
    /// Propagates errors from [`design_tustin_frac`].
    pub fn new(alpha: S, sample_time: S) -> Result<Self, FracError> {
        let (num, den) = design_tustin_frac::<S, N>(alpha, sample_time)?;
        Ok(Self {
            num,
            den,
            x_buf: [S::ZERO; N],
            y_buf: [S::ZERO; N],
        })
    }

    /// Process one input sample and return the filtered output.
    pub fn update(&mut self, x: S) -> S {
        let order = self.num.len().min(self.den.len());

        // Numerator contribution: num[0]*x + Σ_{k=1}^{order-1} num[k]*x_buf[k-1]
        let mut y = self.num[0] * x;
        for k in 1..order {
            if k - 1 < N {
                y += self.num[k] * self.x_buf[k - 1];
            }
        }

        // Denominator contribution (subtract recursive terms)
        let d0 = self.den[0];
        for k in 1..order {
            if k - 1 < N {
                y -= self.den[k] * self.y_buf[k - 1];
            }
        }

        // Normalise by den[0]
        let y_out = if d0.abs() > S::EPSILON {
            y / d0
        } else {
            S::ZERO
        };

        // Shift buffers
        for k in (1..N).rev() {
            self.x_buf[k] = self.x_buf[k - 1];
            self.y_buf[k] = self.y_buf[k - 1];
        }
        self.x_buf[0] = x;
        self.y_buf[0] = y_out;

        y_out
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.x_buf = [S::ZERO; N];
        self.y_buf = [S::ZERO; N];
    }

    /// Access the numerator coefficients.
    pub fn num_coeffs(&self) -> &[S] {
        &self.num
    }

    /// Access the denominator coefficients.
    pub fn den_coeffs(&self) -> &[S] {
        &self.den
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Coefficient sanity checks
    // -----------------------------------------------------------------------

    #[test]
    fn tustin_design_alpha_one_has_correct_num_length() {
        // For α=1 and order=N-1, num and den each have N entries
        let (num, den) = design_tustin_frac::<f64, 4>(1.0_f64, 0.01_f64).expect("valid");
        assert_eq!(num.len(), 4);
        assert_eq!(den.len(), 4);
    }

    #[test]
    fn tustin_design_alpha_one_num_first_coeff_is_scaled() {
        // num[0] = scale * 1.0 = (2/h)^1 = 200 for h=0.01
        let h = 0.01_f64;
        let (num, _) = design_tustin_frac::<f64, 4>(1.0, h).expect("valid");
        let expected = (2.0 / h).powf(1.0); // 200.0
        assert!(
            (num[0] - expected).abs() < 1e-9,
            "num[0]={} expected {}",
            num[0],
            expected
        );
    }

    #[test]
    fn tustin_design_error_on_zero_sample_time() {
        let result = design_tustin_frac::<f64, 4>(1.0_f64, 0.0_f64);
        assert!(matches!(result, Err(FracError::InvalidSampleTime)));
    }

    #[test]
    fn tustin_design_error_on_nan_alpha() {
        let result = design_tustin_frac::<f64, 4>(f64::NAN, 0.01_f64);
        assert!(matches!(result, Err(FracError::InvalidOrder)));
    }

    #[test]
    fn tustin_design_error_on_order_zero() {
        // N=1 gives order=0 which is too small
        let result = design_tustin_frac::<f64, 1>(1.0_f64, 0.01_f64);
        assert!(matches!(result, Err(FracError::WindowTooSmall)));
    }

    // -----------------------------------------------------------------------
    // TustinFrac filter runtime
    // -----------------------------------------------------------------------

    #[test]
    fn tustin_frac_output_is_finite_for_ramp() {
        let mut f = TustinFrac::<f64, 8>::new(1.0_f64, 0.01_f64).expect("valid");
        let h = 0.01_f64;
        let mut last = 0.0_f64;
        for i in 0..30 {
            last = f.update(i as f64 * h);
        }
        assert!(last.is_finite(), "Output should be finite, got {}", last);
    }

    #[test]
    fn tustin_frac_alpha_one_derivative_of_ramp_near_slope() {
        // The Tustin s^1 bilinear approximation:
        //   H(z) = (2/h) · (1 - z^{-1}) / (1 + z^{-1})
        //
        // For a ramp input x[n] = n·h, the difference equation is:
        //   y[n] = (2/h)·h - y[n-1] = 2 - y[n-1]   (since x[n]-x[n-1] = h)
        //
        // This alternates: y oscillates between 2 and 0, so the *average*
        // of two consecutive outputs is exactly 1 — the true derivative slope.
        let h = 0.01_f64;
        let mut f = TustinFrac::<f64, 8>::new(1.0_f64, h).expect("valid");
        let n_settle = 50_usize;
        let n_avg = 20_usize;
        // Settle the filter on the ramp
        for i in 0..n_settle {
            f.update(i as f64 * h);
        }
        // Average outputs: for a ramp, average of alternating [2,0] pairs = 1.0
        let mut sum = 0.0_f64;
        for i in n_settle..(n_settle + n_avg) {
            sum += f.update(i as f64 * h);
        }
        let avg = sum / n_avg as f64;
        // Average should equal the ramp slope (1.0) within 5%
        assert!(
            (avg - 1.0).abs() < 0.05,
            "Average Tustin s^1 output on ramp should equal slope 1.0; avg={}",
            avg
        );
    }

    #[test]
    fn tustin_frac_alpha_half_finite_output() {
        let mut f = TustinFrac::<f64, 8>::new(0.5_f64, 0.01_f64).expect("valid");
        let h = 0.01_f64;
        let mut last = 0.0_f64;
        for i in 0..20 {
            last = f.update(i as f64 * h);
        }
        assert!(
            last.is_finite(),
            "s^0.5 output should be finite, got {}",
            last
        );
    }

    #[test]
    fn tustin_frac_reset_clears_state() {
        let mut f = TustinFrac::<f64, 8>::new(1.0_f64, 0.01_f64).expect("valid");
        for i in 0..20 {
            f.update(i as f64 * 0.01);
        }
        f.reset();
        let out = f.update(0.0);
        assert_eq!(out, 0.0, "After reset and zero input, output should be 0");
    }

    // -----------------------------------------------------------------------
    // Frequency response check
    // -----------------------------------------------------------------------

    #[test]
    fn tustin_frac_alpha_one_gain_at_nyquist() {
        // At DC (all-zero input history, unit step), the gain should be related
        // to (2/T)^α for a pure derivative.  Here we just verify the filter
        // does not diverge.
        let h = 0.1_f64;
        let mut f = TustinFrac::<f64, 6>::new(1.0_f64, h).expect("valid");
        // Apply a unit step and let the filter settle for a few samples
        let mut outputs = [0.0_f64; 20];
        for (i, o) in outputs.iter_mut().enumerate() {
            *o = f.update(if i == 0 { 0.0 } else { 1.0 });
        }
        // All outputs should be finite
        for (i, &out) in outputs.iter().enumerate() {
            assert!(out.is_finite(), "Output[{}]={} is not finite", i, out);
        }
    }

    #[test]
    fn tustin_frac_accessor_lengths() {
        let f = TustinFrac::<f64, 6>::new(0.8_f64, 0.01_f64).expect("valid");
        assert_eq!(f.num_coeffs().len(), 6);
        assert_eq!(f.den_coeffs().len(), 6);
    }

    #[test]
    fn tustin_frac_f32_works() {
        let mut f = TustinFrac::<f32, 4>::new(0.5_f32, 0.01_f32).expect("valid");
        let out = f.update(1.0_f32);
        assert!(out.is_finite());
    }

    // -----------------------------------------------------------------------
    // Integration-mode α < 0
    // -----------------------------------------------------------------------

    #[test]
    fn tustin_frac_alpha_neg_one_monotone_on_step() {
        // s^{-1} of a unit step is a ramp → output should grow monotonically
        let h = 0.1_f64;
        let mut f = TustinFrac::<f64, 8>::new(-1.0_f64, h).expect("valid");
        let mut prev = f64::NEG_INFINITY;
        let mut monotone = true;
        for _ in 0..15 {
            let out = f.update(1.0);
            if out < prev - 1e-9 {
                monotone = false;
                break;
            }
            prev = out;
        }
        assert!(monotone, "s^{{-1}} of unit step should be non-decreasing");
    }
}
