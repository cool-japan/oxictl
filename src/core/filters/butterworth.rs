//! Butterworth IIR filter design via bilinear transform (Tustin method).
//!
//! Analog prototype poles for Nth-order Butterworth:
//!   pk = exp(jπ(2k+N-1)/(2N)),  k = 1..N
//!
//! Bilinear transform with frequency pre-warping:
//!   ωa (analog) ← 2/T · tan(ωd · T/2)
//!   s  ←  2/T · (z-1)/(z+1)
//!
//! Each conjugate pole pair maps to one biquad section (Direct Form II transposed).
//! For odd orders the real pole on the unit circle maps to a first-order section.

use crate::core::filters::FilterError;
use crate::core::scalar::ControlScalar;

// ─────────────────────────────────────────────────────────────
//  Internal biquad (Direct Form II transposed)
// ─────────────────────────────────────────────────────────────

/// Second-order IIR section used internally by Butterworth designs.
/// H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
#[derive(Debug, Clone, Copy)]
pub struct ButterworthBiquad<S: ControlScalar> {
    b0: S,
    b1: S,
    b2: S,
    a1: S,
    a2: S,
    w1: S,
    w2: S,
}

impl<S: ControlScalar> ButterworthBiquad<S> {
    #[inline]
    pub fn new(b0: S, b1: S, b2: S, a1: S, a2: S) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            w1: S::ZERO,
            w2: S::ZERO,
        }
    }

    #[inline]
    pub fn process(&mut self, x: S) -> S {
        let y = self.b0 * x + self.w1;
        self.w1 = self.b1 * x - self.a1 * y + self.w2;
        self.w2 = self.b2 * x - self.a2 * y;
        y
    }

    #[inline]
    pub fn reset(&mut self) {
        self.w1 = S::ZERO;
        self.w2 = S::ZERO;
    }
}

/// First-order IIR section for odd-order Butterworth.
/// H(z) = (b0 + b1·z⁻¹) / (1 + a1·z⁻¹)
#[derive(Debug, Clone, Copy)]
pub struct ButterworthFirstOrder<S: ControlScalar> {
    b0: S,
    b1: S,
    a1: S,
    w: S,
}

impl<S: ControlScalar> ButterworthFirstOrder<S> {
    #[inline]
    pub fn new(b0: S, b1: S, a1: S) -> Self {
        Self {
            b0,
            b1,
            a1,
            w: S::ZERO,
        }
    }

    #[inline]
    pub fn process(&mut self, x: S) -> S {
        let y = self.b0 * x + self.w;
        self.w = self.b1 * x - self.a1 * y;
        y
    }

    #[inline]
    pub fn reset(&mut self) {
        self.w = S::ZERO;
    }
}

// ─────────────────────────────────────────────────────────────
//  Butterworth lowpass  ButterworthLp<S, N>
// ─────────────────────────────────────────────────────────────

/// Nth-order Butterworth lowpass filter.
///
/// `N` is the filter order (1–8).
/// The cascade has `N/2` biquad sections plus one first-order section when N is odd.
///
/// Storage: at most 4 biquads + 1 first-order section (covers up to order 8).
#[derive(Debug, Clone, Copy)]
pub struct ButterworthLp<S: ControlScalar, const N: usize> {
    biquads: [ButterworthBiquad<S>; 4],
    first_order: Option<ButterworthFirstOrder<S>>,
    n_biquads: usize,
    has_first_order: bool,
}

impl<S: ControlScalar, const N: usize> ButterworthLp<S, N> {
    /// Process one input sample.
    #[inline]
    pub fn update(&mut self, x: S) -> S {
        let mut y = x;
        if self.has_first_order {
            y = self.first_order.as_mut().map_or(y, |f| f.process(y));
        }
        for i in 0..self.n_biquads {
            y = self.biquads[i].process(y);
        }
        y
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        for bq in self.biquads.iter_mut() {
            bq.reset();
        }
        if let Some(f) = self.first_order.as_mut() {
            f.reset();
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Butterworth highpass  ButterworthHp<S, N>
// ─────────────────────────────────────────────────────────────

/// Nth-order Butterworth highpass filter.
#[derive(Debug, Clone, Copy)]
pub struct ButterworthHp<S: ControlScalar, const N: usize> {
    biquads: [ButterworthBiquad<S>; 4],
    first_order: Option<ButterworthFirstOrder<S>>,
    n_biquads: usize,
    has_first_order: bool,
}

impl<S: ControlScalar, const N: usize> ButterworthHp<S, N> {
    /// Process one input sample.
    #[inline]
    pub fn update(&mut self, x: S) -> S {
        let mut y = x;
        if self.has_first_order {
            y = self.first_order.as_mut().map_or(y, |f| f.process(y));
        }
        for i in 0..self.n_biquads {
            y = self.biquads[i].process(y);
        }
        y
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        for bq in self.biquads.iter_mut() {
            bq.reset();
        }
        if let Some(f) = self.first_order.as_mut() {
            f.reset();
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Butterworth bandpass  ButterworthBp<S, N>
// ─────────────────────────────────────────────────────────────

/// Nth-order Butterworth bandpass filter (2N biquads total).
///
/// The bandpass is formed by LP→HP frequency transformation applied to
/// the Nth-order Butterworth prototype, yielding 2N poles cascaded as N biquads.
/// `N` is the prototype order; the resulting filter has order 2N.
#[derive(Debug, Clone, Copy)]
pub struct ButterworthBp<S: ControlScalar, const N: usize> {
    /// Up to 8 biquads (covers prototype order up to 4, i.e. bandpass order up to 8).
    biquads: [ButterworthBiquad<S>; 8],
    n_biquads: usize,
}

impl<S: ControlScalar, const N: usize> ButterworthBp<S, N> {
    /// Process one input sample.
    #[inline]
    pub fn update(&mut self, x: S) -> S {
        let mut y = x;
        for i in 0..self.n_biquads {
            y = self.biquads[i].process(y);
        }
        y
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        for bq in self.biquads.iter_mut() {
            bq.reset();
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Helper: bilinear-transform a 2nd-order analog section (s-domain)
//  defined by its pole pair (σ ± jω) into a discrete biquad.
// ─────────────────────────────────────────────────────────────

/// Compute bilinear-transform biquad coefficients for a lowpass prototype
/// second-order section with prototype pole pair (σ ± jω).
///
/// The analog prototype is frequency-scaled to place its cut at ωa = k rad/s:
///   H_a(s/k) = k² / (s² - 2σk·s + p_sq·k²)   where p_sq = σ² + ω²
///
/// After BLT s → (z-1)/(z+1):
///   Numerator z-poly:    k²(z+1)²  →  [k², 2k², k²]
///   Denominator z-poly:  (1-2σk+p_sq·k²)z² + 2(p_sq·k²-1)z + (1+2σk+p_sq·k²)
///
/// Returns (b0, b1, b2, a1, a2) with implicit leading denominator coeff = 1.
fn lp_biquad_from_pole<S: ControlScalar>(sigma: S, omega_im: S, k: S) -> (S, S, S, S, S) {
    let p_sq = sigma * sigma + omega_im * omega_im;
    let k2 = k * k;
    let two = S::TWO;

    // Denominator z-polynomial coefficients (before normalisation)
    let a0 = S::ONE - two * sigma * k + p_sq * k2;
    let a1_z = two * (p_sq * k2 - S::ONE);
    let a2 = S::ONE + two * sigma * k + p_sq * k2;

    // Numerator: k²(1 + z⁻¹)² after normalisation
    let b_all = k2 / a0;
    let b0 = b_all;
    let b1 = two * b_all;
    let b2 = b_all;
    let a1 = a1_z / a0;
    let a2_n = a2 / a0;
    (b0, b1, b2, a1, a2_n)
}

/// Compute bilinear-transform biquad coefficients for a highpass prototype section.
///
/// The HP analog prototype section (frequency-scaled to cut at ωa = k):
///   H_hp(s/k) = s² / (s² - 2σk·s + p_sq·k²)
///
/// After BLT s → (z-1)/(z+1):
///   Numerator z-poly:    (z-1)²  →  [1, -2, 1]   (same for all HP biquads)
///   Denominator: same as LP biquad.
fn hp_biquad_from_pole<S: ControlScalar>(sigma: S, omega_im: S, k: S) -> (S, S, S, S, S) {
    let p_sq = sigma * sigma + omega_im * omega_im;
    let k2 = k * k;
    let two = S::TWO;

    let a0 = S::ONE - two * sigma * k + p_sq * k2;
    let a1_z = two * (p_sq * k2 - S::ONE);
    let a2 = S::ONE + two * sigma * k + p_sq * k2;

    // Numerator: (1-z⁻¹)² after normalisation
    let b0 = S::ONE / a0;
    let b1 = -two / a0;
    let b2 = S::ONE / a0;
    let a1 = a1_z / a0;
    let a2_n = a2 / a0;
    (b0, b1, b2, a1, a2_n)
}

/// First-order lowpass section from the real Butterworth pole at s = -1 (prototype).
///
/// Frequency-scaled prototype: H_a(s/k) = k/(s + k)
/// BLT s → (z-1)/(z+1):
///   H(z) = k/(k+1) · (1 + z⁻¹) / (1 + (k-1)/(k+1)·z⁻¹)
///
/// DC gain: (2k/(k+1)) / (2k/(k+1)) = 1 ✓
fn lp_first_order_from_pole<S: ControlScalar>(k: S) -> (S, S, S) {
    let inv = S::ONE / (k + S::ONE);
    let b0 = k * inv;
    let b1 = k * inv;
    let a1 = (k - S::ONE) * inv; // = (k-1)/(k+1); negative for k<1 (stable LP pole)
    (b0, b1, a1)
}

/// First-order highpass section.
///
/// Frequency-scaled prototype: H_a(s/k) = s/(s + k)
/// BLT s → (z-1)/(z+1):
///   H(z) = 1/(k+1) · (1 - z⁻¹) / (1 + (k-1)/(k+1)·z⁻¹)
///
/// Gain at Nyquist (z=-1): (2/(k+1)) / (2/(k+1)) = 1 ✓ (for normalised input)
fn hp_first_order_from_pole<S: ControlScalar>(k: S) -> (S, S, S) {
    let inv = S::ONE / (k + S::ONE);
    let b0 = inv;
    let b1 = -inv;
    let a1 = (k - S::ONE) * inv;
    (b0, b1, a1)
}

// ─────────────────────────────────────────────────────────────
//  Public design functions
// ─────────────────────────────────────────────────────────────

/// Design an Nth-order Butterworth lowpass filter.
///
/// # Arguments
/// * `cutoff_hz`    — desired −3 dB cutoff frequency in Hz
/// * `sample_rate_hz` — sample rate in Hz
///
/// # Constraints
/// * N must be 1–8 (enforced at runtime via `FilterError`)
/// * `cutoff_hz` must satisfy 0 < fc < fs/2
pub fn design_butterworth_lp<S: ControlScalar, const N: usize>(
    cutoff_hz: S,
    sample_rate_hz: S,
) -> Result<ButterworthLp<S, N>, FilterError> {
    validate_params(cutoff_hz, sample_rate_hz, N)?;

    // Pre-warp: ωd = 2π·fc/fs,  k = tan(ωd/2)  (= 2/T·tan(ωa·T/2) normalised to fs=1)
    let pi = S::PI;
    let omega_d = pi * cutoff_hz / sample_rate_hz; // = π·fc/fs  (half digital frequency)
    let k = omega_d.tan(); // pre-warped prototype frequency gain

    let n_biquads = N / 2;
    let has_first_order = N % 2 == 1;

    let default_bq = ButterworthBiquad::new(S::ONE, S::ZERO, S::ZERO, S::ZERO, S::ZERO);
    let mut biquads = [default_bq; 4];
    let mut first_order: Option<ButterworthFirstOrder<S>> = None;

    // Build biquad sections from conjugate pole pairs.
    // Butterworth poles: pk = exp(jπ(2·m + N - 1)/(2N)) for m = 1..N
    // Real parts σ_m  = cos(π(2m+N-1)/(2N))  (negative — stable LHP)
    // Imag parts ω_m  = sin(π(2m+N-1)/(2N))
    #[allow(clippy::needless_range_loop)]
    for m in 0..n_biquads {
        // Pole index selects conjugate pairs; angles symmetric around -π/2
        let angle = pi * S::from_f64((2 * (m + 1) + N - 1) as f64 / (2 * N) as f64);
        let sigma = angle.cos(); // negative real part of prototype pole
        let omega_im = angle.sin(); // positive imaginary part
        let (b0, b1, b2, a1, a2) = lp_biquad_from_pole(sigma, omega_im, k);
        biquads[m] = ButterworthBiquad::new(b0, b1, b2, a1, a2);
    }

    if has_first_order {
        // Real pole at s = -1 for odd-order prototype
        let (b0, b1, a1) = lp_first_order_from_pole(k);
        first_order = Some(ButterworthFirstOrder::new(b0, b1, a1));
    }

    Ok(ButterworthLp {
        biquads,
        first_order,
        n_biquads,
        has_first_order,
    })
}

/// Design an Nth-order Butterworth highpass filter.
pub fn design_butterworth_hp<S: ControlScalar, const N: usize>(
    cutoff_hz: S,
    sample_rate_hz: S,
) -> Result<ButterworthHp<S, N>, FilterError> {
    validate_params(cutoff_hz, sample_rate_hz, N)?;

    let pi = S::PI;
    let omega_d = pi * cutoff_hz / sample_rate_hz;
    let k = omega_d.tan();

    let n_biquads = N / 2;
    let has_first_order = N % 2 == 1;

    let default_bq = ButterworthBiquad::new(S::ONE, S::ZERO, S::ZERO, S::ZERO, S::ZERO);
    let mut biquads = [default_bq; 4];
    let mut first_order: Option<ButterworthFirstOrder<S>> = None;

    #[allow(clippy::needless_range_loop)]
    for m in 0..n_biquads {
        let angle = pi * S::from_f64((2 * (m + 1) + N - 1) as f64 / (2 * N) as f64);
        let sigma = angle.cos();
        let omega_im = angle.sin();
        let (b0, b1, b2, a1, a2) = hp_biquad_from_pole(sigma, omega_im, k);
        biquads[m] = ButterworthBiquad::new(b0, b1, b2, a1, a2);
    }

    if has_first_order {
        let (b0, b1, a1) = hp_first_order_from_pole(k);
        first_order = Some(ButterworthFirstOrder::new(b0, b1, a1));
    }

    Ok(ButterworthHp {
        biquads,
        first_order,
        n_biquads,
        has_first_order,
    })
}

/// Design an Nth-order Butterworth bandpass filter.
///
/// The bandpass transformation maps a Nth-order LP prototype to a 2N-order BP filter.
/// `low_hz` and `high_hz` define the −3 dB passband edges.
///
/// The prototype order N must be 1–4 (result is up to order 8, i.e. 8 biquads).
pub fn design_butterworth_bp<S: ControlScalar, const N: usize>(
    low_hz: S,
    high_hz: S,
    sample_rate_hz: S,
) -> Result<ButterworthBp<S, N>, FilterError> {
    if N == 0 || N > 4 {
        return Err(FilterError::InvalidOrder);
    }
    if low_hz <= S::ZERO || high_hz <= low_hz || high_hz >= sample_rate_hz * S::HALF {
        return Err(FilterError::InvalidFrequency);
    }
    if sample_rate_hz <= S::ZERO {
        return Err(FilterError::InvalidSampleRate);
    }

    let pi = S::PI;

    // Pre-warp both frequencies
    let omega_l = (pi * low_hz / sample_rate_hz).tan();
    let omega_h = (pi * high_hz / sample_rate_hz).tan();

    // Bandpass parameters
    let bw = omega_h - omega_l; // bandwidth in pre-warped domain
    let omega0_sq = omega_l * omega_h; // center frequency squared

    // For each LP prototype pole pair, the BP transformation produces
    // two biquad sections per LP biquad (and two first-order→biquad for odd N).
    // The bandpass transformation: s_lp → (s² + ω0²) / (bw·s)
    // maps each LP pole to two BP pole pairs.

    let default_bq = ButterworthBiquad::new(S::ZERO, S::ZERO, S::ZERO, S::ZERO, S::ZERO);
    let mut biquads = [default_bq; 8];
    let mut bq_idx = 0usize;

    // Process pole pairs from LP prototype
    let n_biquads_lp = N / 2;
    let has_real_pole = N % 2 == 1;

    for m in 0..n_biquads_lp {
        let angle = pi * S::from_f64((2 * (m + 1) + N - 1) as f64 / (2 * N) as f64);
        let sigma = angle.cos();
        let omega_im = angle.sin();
        // Compute the two BP biquads from this LP pole pair
        let (bq1, bq2) = bp_biquads_from_lp_pole(sigma, omega_im, bw, omega0_sq);
        if bq_idx < 8 {
            biquads[bq_idx] = bq1;
            bq_idx += 1;
        }
        if bq_idx < 8 {
            biquads[bq_idx] = bq2;
            bq_idx += 1;
        }
    }

    if has_real_pole {
        // Real LP pole at s = -1 → one BP biquad
        let bq = bp_biquad_from_lp_real_pole(bw, omega0_sq);
        if bq_idx < 8 {
            biquads[bq_idx] = bq;
            bq_idx += 1;
        }
    }

    Ok(ButterworthBp {
        biquads,
        n_biquads: bq_idx,
    })
}

/// BP biquads derived from a single LP complex pole pair (σ ± jω).
/// The BP transformation maps each complex LP pole to two complex BP pole pairs.
/// Returns two biquad sections.
fn bp_biquads_from_lp_pole<S: ControlScalar>(
    sigma: S,
    omega_im: S,
    bw: S,
    omega0_sq: S,
) -> (ButterworthBiquad<S>, ButterworthBiquad<S>) {
    let two = S::TWO;
    let _four = S::from_f64(4.0);
    let _bw2 = bw * bw;

    // LP pole: s_lp = σ + j·ω
    // BP transform: s_bp = (bw/2)·s_lp ± sqrt((bw/2)²·s_lp² - ω0²)
    // For each LP pole we get two BP poles (plus their conjugates = 4 poles total = 2 biquads).
    // We work with discriminant D = (bw·s_lp/2)² - ω0²
    // s_lp real part contributes σ, imag part ω_im
    // Let p = bw·sigma/2,  q = bw·omega_im/2
    // Then bw²/4·(s_lp²) = bw²/4·((σ²-ω²) + j·2σω)
    // D = bw²/4·(σ²-ω²) - ω0² + j·bw²/2·σ·ω

    let half_bw = bw * S::HALF;
    let p = half_bw * sigma; // real(bw/2 · s_lp)
    let q = half_bw * omega_im; // imag(bw/2 · s_lp)

    // D = p² - q² - ω0²  +  j·2pq
    let d_re = p * p - q * q - omega0_sq;
    let d_im = two * p * q;

    // sqrt of complex D:
    let d_mag = (d_re * d_re + d_im * d_im).sqrt();
    let d_sqrt_re = ((d_mag + d_re) * S::HALF).sqrt();
    // sign of imag sqrt chosen so real(sqrt(D)) > 0 for stability
    let d_sqrt_im = if d_im.abs() > S::ZERO {
        d_im / (two * d_sqrt_re)
    } else {
        S::ZERO
    };

    // Two sets of BP poles:
    // s1 = p + d_sqrt_re + j(q + d_sqrt_im)  → pole 1
    // s2 = p - d_sqrt_re + j(q - d_sqrt_im)  → pole 2 (and their conjugates)
    let s1_re = p + d_sqrt_re;
    let s1_im = q + d_sqrt_im;
    let s2_re = p - d_sqrt_re;
    let s2_im = q - d_sqrt_im;

    // Each pole pair (s_r ± j·s_i) maps to biquad in s-plane:
    // H(s) = bw·s / (s² - 2·s_r·s + (s_r²+s_i²))
    // via BLT s → (z-1)/(z+1) (normalised, k=1 since already in warped domain):
    let bq1 = bp_biquad_from_bp_pole_pair(s1_re, s1_im, bw);
    let bq2 = bp_biquad_from_bp_pole_pair(s2_re, s2_im, bw);
    (bq1, bq2)
}

/// Build a BP biquad from a BP pole pair (σ ± jω) with bandwidth `bw`.
/// H_a(s) = bw·s / (s² - 2σ·s + (σ²+ω²))
/// After BLT with k=1 (since we work in pre-warped space already):
fn bp_biquad_from_bp_pole_pair<S: ControlScalar>(
    sigma: S,
    omega_im: S,
    bw: S,
) -> ButterworthBiquad<S> {
    let two = S::TWO;
    let p_sq = sigma * sigma + omega_im * omega_im;

    // BLT with k=1:
    // a0_d = 1 - 2σ + p_sq
    // a1_d = 2(p_sq - 1) * 2  ... wait, let's derive properly.
    // s = (z-1)/(z+1) → numerator bw·s = bw(z-1)/(z+1)
    // After clearing (z+1)²:
    // Num(z): bw·(z-1)·(z+1) = bw·(z²-1) → [bw, 0, -bw]
    // Den(z): (z-1)² - 2σ(z-1)(z+1) + p_sq(z+1)²
    //       = z²-2z+1 - 2σ(z²-1) + p_sq(z²+2z+1)
    //       = (1-2σ+p_sq)z² + (-2+2p_sq)·2... let me redo:
    // = (1 - 2σ + p_sq)z² + (-2 + 2p_sq - 2·0)z + (1 + 2σ + p_sq)
    // Wait: -2σ(z²-1) = -2σz² + 2σ
    // p_sq(z+1)² = p_sq·z² + 2p_sq·z + p_sq
    // (z-1)² = z² - 2z + 1
    // Sum: (1-2σ+p_sq)z² + (-2+2p_sq)z + (1+2σ+p_sq)
    let a0_d = S::ONE - two * sigma + p_sq;
    let a1_d = -two + two * p_sq;
    let a2_d = S::ONE + two * sigma + p_sq;

    let b0 = bw / a0_d;
    let b1 = S::ZERO;
    let b2 = -(bw / a0_d);
    let a1 = a1_d / a0_d;
    let a2 = a2_d / a0_d;
    ButterworthBiquad::new(b0, b1, b2, a1, a2)
}

/// Build a BP biquad from the real LP pole at s_lp = -1.
/// BP transform: s_bp = -(bw/2) ± sqrt((bw/2)² - ω0²)
/// This gives a real pair or complex pair depending on discriminant.
fn bp_biquad_from_lp_real_pole<S: ControlScalar>(bw: S, omega0_sq: S) -> ButterworthBiquad<S> {
    let half_bw = bw * S::HALF;
    let disc = half_bw * half_bw - omega0_sq;

    if disc >= S::ZERO {
        // Two real BP poles: -bw/2 ± sqrt(disc)
        // Each with conjugate → combine as biquad
        let sq = disc.sqrt();
        let s1 = -half_bw + sq;
        let s2 = -half_bw - sq;
        // H(s) = bw·s / ((s - s1)(s - s2))
        // = bw·s / (s² - (s1+s2)s + s1·s2)
        // s1+s2 = -bw,  s1·s2 = bw²/4 - disc = omega0_sq
        let sigma_sum = s1 + s2; // = -bw
        let p_product = s1 * s2; // = omega0_sq
                                 // BLT: same formula as bp_biquad_from_bp_pole_pair but for real poles
                                 // Treat as pole at (sigma_sum/2, 0) — but that isn't quite right for real poles.
                                 // Better: use explicit transfer function.
        let two = S::TWO;
        let a0_d = S::ONE - sigma_sum + p_product;
        let a1_d = -two + two * p_product;
        let a2_d = S::ONE + sigma_sum + p_product;
        ButterworthBiquad::new(bw / a0_d, S::ZERO, -(bw / a0_d), a1_d / a0_d, a2_d / a0_d)
    } else {
        // Complex BP poles (common case when bw < 2ω0)
        let sq = (-disc).sqrt();
        bp_biquad_from_bp_pole_pair(-half_bw, sq, bw)
    }
}

// ─────────────────────────────────────────────────────────────
//  Parameter validation
// ─────────────────────────────────────────────────────────────

fn validate_params<S: ControlScalar>(
    cutoff_hz: S,
    sample_rate_hz: S,
    order: usize,
) -> Result<(), FilterError> {
    if order == 0 || order > 8 {
        return Err(FilterError::InvalidOrder);
    }
    if sample_rate_hz <= S::ZERO {
        return Err(FilterError::InvalidSampleRate);
    }
    if cutoff_hz <= S::ZERO || cutoff_hz >= sample_rate_hz * S::HALF {
        return Err(FilterError::InvalidFrequency);
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Measure steady-state amplitude after driving at frequency `freq_hz`
    fn measure_gain<F: FnMut(f64) -> f64>(mut filter: F, freq_hz: f64, sample_rate: f64) -> f64 {
        let n_settle = 8000usize;
        let n_measure = 2000usize;
        let mut max_out = 0.0_f64;
        for i in 0..(n_settle + n_measure) {
            let x = (2.0 * core::f64::consts::PI * freq_hz * i as f64 / sample_rate).sin();
            let y = filter(x);
            if i >= n_settle {
                let ya = y.abs();
                if ya > max_out {
                    max_out = ya;
                }
            }
        }
        max_out
    }

    #[test]
    fn butterworth_lp2_dc_gain() {
        let mut filt = design_butterworth_lp::<f64, 2>(100.0, 1000.0).unwrap();
        // Drive DC (freq=0 → constant 1)
        for _ in 0..5000 {
            filt.update(1.0);
        }
        let y = filt.update(1.0);
        assert!((y - 1.0).abs() < 0.01, "DC gain should be ~1, got {y}");
    }

    #[test]
    fn butterworth_lp2_cutoff_attenuation() {
        let fs = 10000.0_f64;
        let fc = 500.0_f64;
        let mut filt = design_butterworth_lp::<f64, 2>(fc, fs).unwrap();
        // At cutoff, gain should be 1/sqrt(2) ≈ 0.707
        let gain = measure_gain(|x| filt.update(x), fc, fs);
        assert!(
            (gain - core::f64::consts::FRAC_1_SQRT_2).abs() < 0.05,
            "At cutoff expected ~0.707, got {gain}"
        );
    }

    #[test]
    fn butterworth_lp4_stopband_attenuation() {
        let fs = 10000.0_f64;
        let fc = 500.0_f64;
        let mut filt = design_butterworth_lp::<f64, 4>(fc, fs).unwrap();
        // At 10× cutoff, 4th-order should attenuate by ≥ 40 dB → gain ≤ 0.01
        let gain = measure_gain(|x| filt.update(x), fc * 10.0, fs);
        assert!(gain < 0.01, "Stopband gain should be < 0.01, got {gain}");
    }

    #[test]
    fn butterworth_lp1_odd_order() {
        let mut filt = design_butterworth_lp::<f64, 1>(200.0, 1000.0).unwrap();
        for _ in 0..5000 {
            filt.update(1.0);
        }
        let y = filt.update(1.0);
        assert!((y - 1.0).abs() < 0.02, "1st-order LP DC gain, got {y}");
    }

    #[test]
    fn butterworth_lp3_odd_order() {
        let fs = 10000.0_f64;
        let fc = 500.0_f64;
        let mut filt = design_butterworth_lp::<f64, 3>(fc, fs).unwrap();
        for _ in 0..5000 {
            filt.update(1.0);
        }
        let y = filt.update(1.0);
        assert!((y - 1.0).abs() < 0.01, "3rd-order LP DC gain, got {y}");
    }

    #[test]
    fn butterworth_hp2_dc_rejection() {
        let mut filt = design_butterworth_hp::<f64, 2>(100.0, 1000.0).unwrap();
        for _ in 0..5000 {
            filt.update(1.0);
        }
        let y = filt.update(1.0);
        assert!(y.abs() < 0.01, "HP should reject DC, got {y}");
    }

    #[test]
    fn butterworth_hp2_passband_gain() {
        let fs = 10000.0_f64;
        let fc = 200.0_f64;
        let mut filt = design_butterworth_hp::<f64, 2>(fc, fs).unwrap();
        // Well above cutoff: should pass with gain ≈ 1
        let gain = measure_gain(|x| filt.update(x), fc * 10.0, fs);
        assert!(gain > 0.9, "HP passband gain should be > 0.9, got {gain}");
    }

    #[test]
    fn butterworth_bp_passband() {
        let fs = 10000.0_f64;
        let flow = 400.0_f64;
        let fhigh = 600.0_f64;
        let mut filt = design_butterworth_bp::<f64, 2>(flow, fhigh, fs).unwrap();
        // Center should pass
        let center = (flow * fhigh).sqrt();
        let gain = measure_gain(|x| filt.update(x), center, fs);
        assert!(
            gain > 0.3,
            "BP center should pass with gain > 0.3, got {gain}"
        );
    }

    #[test]
    fn butterworth_lp_invalid_order() {
        let result = design_butterworth_lp::<f64, 0>(100.0, 1000.0);
        assert!(result.is_err());
        let result = design_butterworth_lp::<f64, 9>(100.0, 1000.0);
        assert!(result.is_err());
    }

    #[test]
    fn butterworth_lp_invalid_frequency() {
        // fc >= fs/2
        let result = design_butterworth_lp::<f64, 2>(500.0, 1000.0);
        assert!(result.is_err());
        // fc <= 0
        let result = design_butterworth_lp::<f64, 2>(-10.0, 1000.0);
        assert!(result.is_err());
    }

    #[test]
    fn butterworth_lp_reset() {
        let mut filt = design_butterworth_lp::<f64, 2>(100.0, 1000.0).unwrap();
        for _ in 0..100 {
            filt.update(1.0);
        }
        filt.reset();
        let y = filt.update(0.0);
        assert_eq!(y, 0.0);
    }
}
