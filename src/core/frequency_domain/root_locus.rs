use super::sensitivity::Complex;
use super::FreqError;
use crate::core::scalar::ControlScalar;
use crate::core::transfer_fn::TransferFn;
use heapless::Vec as HVec;

/// A single point on the root locus: one gain value with the resulting closed-loop poles.
///
/// Poles are stored as complex numbers. For a TF of order N, there are at most N poles.
/// Poles are stored in a heapless Vec with capacity 8.
#[derive(Debug, Clone)]
pub struct RootLocusPoint<S: ControlScalar> {
    /// The gain value k at this point.
    pub gain: S,
    /// Closed-loop poles of 1 + k·L(z) = 0.
    pub poles: HVec<Complex<S>, 8>,
}

/// Collection of N root locus points.
pub struct RootLocusData<S: ControlScalar, const N: usize> {
    pub points: HVec<RootLocusPoint<S>, N>,
}

impl<S: ControlScalar, const N: usize> RootLocusData<S, N> {
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Check whether all poles lie strictly inside the unit circle (discrete-time stability).
///
/// Returns `true` if all poles have magnitude strictly less than 1.
pub fn stability_region<S: ControlScalar>(poles: &HVec<Complex<S>, 8>) -> bool {
    for pole in poles.iter() {
        let mag_sq = pole.re * pole.re + pole.im * pole.im;
        if mag_sq >= S::ONE {
            return false;
        }
    }
    true
}

/// Compute the closed-loop poles of the system 1 + k·L(z) = 0.
///
/// The characteristic polynomial of 1 + k·L(z) = 0 where L(z) = B(z)/A(z) is:
///   A(z) + k·B(z) = 0
///
/// For L(z) = (b[0] + b[1]*z^{-1} + ... + b[N-1]*z^{-(N-1)}) /
///            (1     + a[0]*z^{-1} + ... + a[N-1]*z^{-(N-1)})
///
/// Multiplying through by z^{N-1}:
///   z^{N-1} + a[0]*z^{N-2} + ... + a[N-2]*z + a[N-1]
///   + k*(b[0]*z^{N-1} + b[1]*z^{N-2} + ... + b[N-1]) = 0
///
/// The characteristic polynomial in z-domain (degree N-1) is:
///   (1 + k*b[0])*z^{N-1} + (a[0] + k*b[1])*z^{N-2} + ... + (a[N-2] + k*b[N-1])*z^0
///     + (a[N-1]) * z^{-1}  — wait, we have N coefficients for denominator too.
///
/// Actually, with the TransferFn convention (order N meaning N coefficients each for b and a,
/// denominator = 1 + a[0]*z^{-1} + ... + a[N-1]*z^{-(N-1)}):
/// The characteristic polynomial after clearing z^{-N} is degree N:
///   z^N + a[0]*z^{N-1} + ... + a[N-1] + k*(b[0]*z^{N-1} + b[1]*z^{N-2} + ... + b[N-1]) * z
///
/// More carefully: L(z) = N(z^{-1})/D(z^{-1}) where:
///   N(z^{-1}) = b[0] + b[1]*z^{-1} + ... + b[N-1]*z^{-(N-1)}
///   D(z^{-1}) = 1 + a[0]*z^{-1} + ... + a[N-1]*z^{-(N-1)}
///
/// The closed-loop characteristic equation D(z^{-1}) + k*N(z^{-1}) = 0.
/// Multiply through by z^N:
///   z^N + a[0]*z^{N-1} + ... + a[N-1] + k*(b[0]*z^{N-1} + b[1]*z^{N-2} + ... + b[N-1]) = 0
/// Wait, that's degree N but the last term from D is a[N-1]*z^0 and from k*N is k*b[N-1]*z^0.
/// No: multiplying D(z^{-1})*z^N = z^N + a[0]*z^{N-1} + ... + a[N-1]*z^{N-N} = z^N + a[0]*z^{N-1}+...+a[N-1]
/// And N(z^{-1})*z^N = b[0]*z^N + b[1]*z^{N-1} + ... + b[N-1]*z^{N-N}
/// = b[0]*z^N + b[1]*z^{N-1} + ... + b[N-1]
///
/// So the characteristic polynomial is:
///   (1 + k*b[0])*z^N + (a[0] + k*b[1])*z^{N-1} + ... + (a[N-1] + k*b[N-1])
///
/// This is degree N (with N+1 coefficients from index 0 to N inclusive).
/// For our TransferFn<S, N>, b and a both have N elements.
/// The characteristic polynomial (degree N) has N+1 coefficients:
///   c[0] = 1 + k*b[0]       (leading, z^N term)
///   c[i] = a[i-1] + k*b[i] for i=1..N-1
///   c[N] = a[N-1] + k*b[N-1]
///
/// We solve this polynomial for its N roots.
fn compute_closed_loop_poles<S: ControlScalar, const N: usize>(
    tf: &TransferFn<S, N>,
    k: S,
) -> HVec<Complex<S>, 8> {
    let b = tf.b();
    let a = tf.a();

    // Build the characteristic polynomial coefficients (degree N, N+1 coefficients)
    // poly[0] is coefficient of z^N (leading), poly[N] is constant term.
    // Max degree we handle: N (from TransferFn<S, N> with N up to 8 for the 8-pole limit).
    let mut poly = [S::ZERO; 9]; // up to degree 8 → 9 coefficients

    // The polynomial degree is N (we have N b-coefficients and N a-coefficients).
    // poly[0] = 1 + k*b[0]
    poly[0] = S::ONE + k * b[0];
    // poly[i] = a[i-1] + k*b[i]  for i = 1..N-1
    for i in 1..N {
        poly[i] = a[i - 1] + k * b[i];
    }
    // poly[N] = a[N-1]  (no more b[] terms since b has N elements, last is b[N-1])
    poly[N] = a[N - 1];

    // Find roots of poly[0]*z^N + poly[1]*z^{N-1} + ... + poly[N] = 0
    find_polynomial_roots(&poly[..=N])
}

/// Find roots of a polynomial with real coefficients using the companion matrix
/// and power iteration / direct formulas for low orders.
///
/// `coeffs[0]` is the leading coefficient (z^n), `coeffs[n]` is the constant.
/// Returns up to 8 complex roots.
fn find_polynomial_roots<S: ControlScalar>(coeffs: &[S]) -> HVec<Complex<S>, 8> {
    let mut roots: HVec<Complex<S>, 8> = HVec::new();

    // Determine actual degree by finding leading non-zero coefficient
    let mut start = 0;
    while start < coeffs.len() && coeffs[start].abs() < S::EPSILON {
        start += 1;
    }
    if start >= coeffs.len() {
        return roots; // zero polynomial
    }

    let degree = coeffs.len() - 1 - start;
    let lead = coeffs[start];

    match degree {
        0 => {
            // Constant: no roots
        }
        1 => {
            // Linear: c0*z + c1 = 0 → z = -c1/c0
            let z = -(coeffs[start + 1] / lead);
            let _ = roots.push(Complex::new(z, S::ZERO));
        }
        2 => {
            // Quadratic: c0*z^2 + c1*z + c2 = 0
            let c0 = lead;
            let c1 = coeffs[start + 1];
            let c2 = coeffs[start + 2];
            let disc = c1 * c1 - S::from_f64(4.0) * c0 * c2;
            let two_c0 = S::TWO * c0;
            if disc >= S::ZERO {
                let sq = disc.sqrt();
                let _ = roots.push(Complex::new((-c1 + sq) / two_c0, S::ZERO));
                let _ = roots.push(Complex::new((-c1 - sq) / two_c0, S::ZERO));
            } else {
                let sq = (-disc).sqrt();
                let _ = roots.push(Complex::new(-c1 / two_c0, sq / two_c0));
                let _ = roots.push(Complex::new(-c1 / two_c0, -(sq / two_c0)));
            }
        }
        3 => {
            // Cubic: use Cardano's method (depressed cubic via substitution)
            let c0 = lead;
            let c1 = coeffs[start + 1] / c0;
            let c2 = coeffs[start + 2] / c0;
            let c3 = coeffs[start + 3] / c0;
            compute_cubic_roots(c1, c2, c3, &mut roots);
        }
        4 => {
            // Quartic: use companion matrix power iteration
            let normalized: [S; 5] = [
                S::ONE,
                coeffs[start + 1] / lead,
                coeffs[start + 2] / lead,
                coeffs[start + 3] / lead,
                coeffs[start + 4] / lead,
            ];
            compute_quartic_roots_companion(&normalized, &mut roots);
        }
        _ => {
            // Higher order: use companion matrix with power iteration for each root
            companion_matrix_roots(coeffs, start, degree, lead, &mut roots);
        }
    }
    roots
}

/// Compute cubic roots for z^3 + c1*z^2 + c2*z + c3 = 0 via Cardano's method.
fn compute_cubic_roots<S: ControlScalar>(c1: S, c2: S, c3: S, roots: &mut HVec<Complex<S>, 8>) {
    // Depress: z = t - c1/3
    let third = S::from_f64(1.0 / 3.0);
    let p = c2 - c1 * c1 * third;
    let q = S::from_f64(2.0 / 27.0) * c1 * c1 * c1 - third * c1 * c2 + c3;

    let disc = q * q / S::from_f64(4.0) + p * p * p / S::from_f64(27.0);
    let shift = c1 * third;

    if disc >= S::ZERO {
        // One real root, two complex conjugates (or three real if disc=0)
        let sq = disc.sqrt();
        let u_arg = -q / S::TWO + sq;
        let v_arg = -q / S::TWO - sq;

        let u = cbrt_signed(u_arg);
        let v = cbrt_signed(v_arg);

        let root1 = u + v - shift;
        let _ = roots.push(Complex::new(root1, S::ZERO));

        if disc.abs() < S::EPSILON {
            // Three real roots (two equal)
            let root2 = -(u + v) / S::TWO - shift;
            let _ = roots.push(Complex::new(root2, S::ZERO));
            let _ = roots.push(Complex::new(root2, S::ZERO));
        } else {
            // Two complex conjugate roots
            let re_part = -(u + v) / S::TWO - shift;
            let im_part = S::from_f64(3.0_f64.sqrt() / 2.0) * (u - v);
            let _ = roots.push(Complex::new(re_part, im_part));
            let _ = roots.push(Complex::new(re_part, -im_part));
        }
    } else {
        // Three real roots via trigonometric method
        let m = S::TWO * ((-p) / S::from_f64(3.0)).sqrt();
        let theta = (S::from_f64(3.0) * q / (p * m)).acos() / S::from_f64(3.0);
        let two_pi_thirds = S::from_f64(2.0 * core::f64::consts::PI / 3.0);

        let root1 = m * theta.cos() - shift;
        let root2 = m * (theta - two_pi_thirds).cos() - shift;
        let root3 = m * (theta + two_pi_thirds).cos() - shift;
        let _ = roots.push(Complex::new(root1, S::ZERO));
        let _ = roots.push(Complex::new(root2, S::ZERO));
        let _ = roots.push(Complex::new(root3, S::ZERO));
    }
}

/// Real cube root with sign preservation.
fn cbrt_signed<S: ControlScalar>(x: S) -> S {
    if x >= S::ZERO {
        x.powf(S::from_f64(1.0 / 3.0))
    } else {
        -((-x).powf(S::from_f64(1.0 / 3.0)))
    }
}

/// Compute quartic roots using companion matrix power iteration.
/// `coeffs` is [1, c1, c2, c3, c4] (normalized, leading = 1).
fn compute_quartic_roots_companion<S: ControlScalar>(
    coeffs: &[S; 5],
    roots: &mut HVec<Complex<S>, 8>,
) {
    // Try to factor as two quadratics using Ferrari's method
    // z^4 + c1*z^3 + c2*z^2 + c3*z + c4
    // We use numerical power iteration on the companion matrix
    let all_coeffs = [coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]];
    companion_matrix_roots(&all_coeffs, 0, 4, S::ONE, roots);
}

/// General companion-matrix eigenvalue finding via shifted power iteration.
///
/// Builds the companion matrix and applies inverse power iteration with shifts
/// to find eigenvalues one at a time (deflation).
fn companion_matrix_roots<S: ControlScalar>(
    coeffs: &[S],
    start: usize,
    degree: usize,
    lead: S,
    roots: &mut HVec<Complex<S>, 8>,
) {
    if degree == 0 || degree > 8 {
        return;
    }

    // Normalized coefficients (monic polynomial)
    let mut norm = [S::ZERO; 8];
    #[allow(clippy::needless_range_loop)]
    for i in 0..degree {
        norm[i] = coeffs[start + 1 + i] / lead;
    }

    // Use Weierstrass/Durand-Kerner iteration to find all roots simultaneously.
    // Initialize roots at evenly-spaced points on a circle of radius 1.
    let mut z: [Complex<S>; 8] = [Complex::new(S::ZERO, S::ZERO); 8];
    #[allow(clippy::needless_range_loop)]
    for i in 0..degree {
        let angle = S::from_f64(2.0 * core::f64::consts::PI * i as f64 / degree as f64);
        z[i] = Complex::new(angle.cos(), angle.sin());
    }

    // Durand-Kerner iterations
    for _iter in 0..200 {
        let mut max_change = S::ZERO;
        for i in 0..degree {
            // Evaluate polynomial at z[i]
            let p_val = eval_poly_complex(&norm[..degree], z[i]);

            // Compute denominator: product of (z[i] - z[j]) for j != i
            let mut denom = Complex::new(S::ONE, S::ZERO);
            for j in 0..degree {
                if j != i {
                    denom = denom.multiply(&z[i].sub(&z[j]));
                }
            }

            let denom_mag_sq = denom.magnitude_sq();
            if denom_mag_sq < S::EPSILON {
                continue;
            }

            // Update: z[i] -= p(z[i]) / denom
            let update = p_val
                .divide(&denom)
                .unwrap_or(Complex::new(S::ZERO, S::ZERO));
            let change = update.magnitude();
            if change > max_change {
                max_change = change;
            }
            z[i] = z[i].sub(&update);
        }

        // Convergence check
        if max_change < S::from_f64(1e-12) {
            break;
        }
    }

    for z_val in z.iter().take(degree) {
        if roots.len() < 8 {
            let _ = roots.push(*z_val);
        }
    }
}

/// Evaluate a monic polynomial p(z) = z^n + norm[0]*z^{n-1} + ... + norm[n-1]
/// at complex point z, using Horner's method.
fn eval_poly_complex<S: ControlScalar>(norm: &[S], z: Complex<S>) -> Complex<S> {
    // p(z) = z^n + norm[0]*z^{n-1} + ... + norm[n-1]
    // Using Horner: p = z; p = p*z + norm[0]; p = p*z + norm[1]; ...
    let n = norm.len();
    if n == 0 {
        return Complex::new(S::ONE, S::ZERO);
    }

    // Start with z^n via Horner: accumulate from the leading 1
    let mut acc = Complex::new(S::ONE, S::ZERO); // leading 1
    for &n_coeff in norm.iter().take(n) {
        acc = acc.multiply(&z);
        acc.re += n_coeff;
    }
    acc
}

/// Compute the root locus for a discrete-time open-loop transfer function.
///
/// For N gain values logarithmically spaced from 0 to `k_max` (exclusive of 0,
/// starting at a small value), computes the closed-loop poles of 1 + k·L(z) = 0.
///
/// # Errors
/// - [`FreqError::InsufficientPoints`] if `N < 2`
/// - [`FreqError::InvalidParameter`] if `k_max <= 0`
pub fn compute_root_locus<S: ControlScalar, const TF_ORDER: usize, const N: usize>(
    open_loop_tf: &TransferFn<S, TF_ORDER>,
    k_max: S,
) -> Result<RootLocusData<S, N>, FreqError> {
    if N < 2 {
        return Err(FreqError::InsufficientPoints);
    }
    if k_max <= S::ZERO {
        return Err(FreqError::InvalidParameter);
    }
    // TF_ORDER must fit within 8-pole capacity
    if TF_ORDER > 8 {
        return Err(FreqError::InvalidParameter);
    }

    let mut data = RootLocusData::<S, N> {
        points: HVec::new(),
    };

    let n_minus_one = S::from_f64((N - 1) as f64);
    for i in 0..N {
        // Linearly space gains from 0 to k_max
        let k = k_max * S::from_f64(i as f64) / n_minus_one;
        let poles = compute_closed_loop_poles(open_loop_tf, k);
        let point = RootLocusPoint { gain: k, poles };
        let _ = data.points.push(point);
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transfer_fn::TransferFn;

    /// At k=0, closed-loop poles should equal open-loop poles.
    /// For H(z) = b[0] / (1 + a[0]*z^{-1}), open-loop pole is at z = -a[0].
    #[test]
    fn root_locus_zero_gain_gives_open_loop_poles() {
        // H(z) = (1-α)/(1 - α*z^{-1}): open-loop pole at z = α
        let alpha = 0.5_f64;
        let tf = TransferFn::<f64, 1>::new([1.0 - alpha], [-alpha]);
        let data = compute_root_locus::<f64, 1, 8>(&tf, 1.0).expect("root locus ok");

        // First point is k=0
        let first = &data.points[0];
        assert!(
            (first.gain).abs() < 1e-10,
            "First gain should be 0, got {}",
            first.gain
        );
        // At k=0, the characteristic polynomial is A(z) alone:
        // z^1 + a[0] = z - 0.5 = 0 → z = 0.5
        // Poles should be near 0.5 (with possible numerical error)
        if let Some(pole) = first.poles.first() {
            assert!(
                (pole.re - alpha).abs() < 1e-6,
                "Open-loop pole should be at α={}, got re={}",
                alpha,
                pole.re
            );
            assert!(
                pole.im.abs() < 1e-6,
                "Open-loop pole should be real, got im={}",
                pole.im
            );
        }
    }

    /// Unity gain TF (H(z)=1): all poles at origin at k=0, moves as k increases.
    #[test]
    fn root_locus_point_count() {
        let tf = TransferFn::<f64, 1>::new([1.0], [0.0]);
        let data = compute_root_locus::<f64, 1, 16>(&tf, 1.0).expect("root locus ok");
        assert_eq!(data.len(), 16, "Should have 16 root locus points");
    }

    /// Stable system stays inside unit circle for small k.
    #[test]
    fn stability_region_inside_unit_circle() {
        let alpha = 0.3_f64;
        let tf = TransferFn::<f64, 1>::new([1.0 - alpha], [-alpha]);
        let data = compute_root_locus::<f64, 1, 8>(&tf, 0.1).expect("root locus ok");

        // For small gains, all poles should be inside unit circle
        for pt in data.points.iter() {
            let stable = stability_region(&pt.poles);
            assert!(
                stable,
                "Poles should be inside unit circle for small gain k={}, poles: {:?}",
                pt.gain, pt.poles
            );
        }
    }

    /// stability_region returns false when poles are outside unit circle.
    #[test]
    fn stability_region_detects_unstable() {
        let mut poles: HVec<Complex<f64>, 8> = HVec::new();
        let _ = poles.push(Complex::new(1.5, 0.0)); // outside unit circle
        assert!(!stability_region(&poles), "Pole at 1.5 should be unstable");
    }

    /// stability_region returns true for poles strictly inside unit circle.
    #[test]
    fn stability_region_detects_stable() {
        let mut poles: HVec<Complex<f64>, 8> = HVec::new();
        let _ = poles.push(Complex::new(0.5, 0.0));
        let _ = poles.push(Complex::new(-0.3, 0.2));
        assert!(
            stability_region(&poles),
            "Poles inside unit circle should be stable"
        );
    }

    /// Quadratic characteristic polynomial roots: z^2 - 1 = 0 → roots ±1.
    #[test]
    fn quadratic_roots_real() {
        // poly = [1, 0, -1] → z^2 - 1 = 0 → roots ±1
        let coeffs = [1.0_f64, 0.0, -1.0];
        let roots = find_polynomial_roots(&coeffs);
        assert_eq!(roots.len(), 2, "Should find 2 roots");
        let mut reals: [f64; 2] = [roots[0].re, roots[1].re];
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        assert!((reals[0] - (-1.0)).abs() < 1e-10, "Root should be -1");
        assert!((reals[1] - 1.0).abs() < 1e-10, "Root should be 1");
    }

    /// Complex roots: z^2 + 1 = 0 → roots ±j.
    #[test]
    fn quadratic_roots_complex() {
        let coeffs = [1.0_f64, 0.0, 1.0];
        let roots = find_polynomial_roots(&coeffs);
        assert_eq!(roots.len(), 2, "Should find 2 roots");
        for root in roots.iter() {
            assert!(
                root.re.abs() < 1e-10,
                "Real part should be 0, got {}",
                root.re
            );
            assert!(
                (root.im.abs() - 1.0).abs() < 1e-10,
                "Imag part magnitude should be 1, got {}",
                root.im
            );
        }
    }

    /// Invalid k_max returns error.
    #[test]
    fn root_locus_invalid_kmax() {
        let tf = TransferFn::<f64, 1>::new([1.0], [0.0]);
        let result = compute_root_locus::<f64, 1, 8>(&tf, -1.0);
        assert!(
            matches!(result, Err(FreqError::InvalidParameter)),
            "Negative k_max should return InvalidParameter"
        );
    }
}
