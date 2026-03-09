//! Quantizers for digital control signals.
//!
//! Provides uniform, logarithmic, and dynamic (zoom-based) quantizers
//! suitable for networked and digital control system analysis.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by quantizer constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizerError {
    /// A constructor parameter was out of its valid range.
    InvalidParameter,
    /// The input was outside the representable range (informational).
    OutOfRange,
}

// ---------------------------------------------------------------------------
// UniformQuantizer
// ---------------------------------------------------------------------------

/// Uniform (mid-tread) quantizer with 2^bits levels.
///
/// Q(u) = u_min + Δ · round((u − u_min) / Δ)
/// where Δ = (u_max − u_min) / (2^bits − 1)
pub struct UniformQuantizer<S> {
    bits: u8,
    levels: usize,
    u_min: S,
    u_max: S,
    delta: S,
}

impl<S: ControlScalar> UniformQuantizer<S> {
    /// Create a new uniform quantizer.
    ///
    /// # Errors
    /// Returns [`QuantizerError::InvalidParameter`] if `bits == 0`,
    /// `u_min >= u_max`, or `bits > 63`.
    pub fn new(bits: u8, u_min: S, u_max: S) -> Result<Self, QuantizerError> {
        if bits == 0 || bits > 63 {
            return Err(QuantizerError::InvalidParameter);
        }
        if u_min >= u_max {
            return Err(QuantizerError::InvalidParameter);
        }
        let levels: usize = 1usize << (bits as usize);
        let delta = (u_max - u_min) / S::from_f64((levels - 1) as f64);
        if delta <= S::ZERO {
            return Err(QuantizerError::InvalidParameter);
        }
        Ok(Self {
            bits,
            levels,
            u_min,
            u_max,
            delta,
        })
    }

    /// Quantize input `u` to the nearest quantization level.
    /// Input is clamped to [u_min, u_max] before quantization.
    pub fn quantize(&self, u: S) -> S {
        let u_clamped = u.clamp_val(self.u_min, self.u_max);
        let normalised = (u_clamped - self.u_min) / self.delta;
        // Use libm::round for no_std compatibility
        let rounded = S::from_f64(libm::round(normalised.to_f64()));
        // Clamp index to valid range
        let max_idx = S::from_f64((self.levels - 1) as f64);
        let idx = rounded.clamp_val(S::ZERO, max_idx);
        self.u_min + self.delta * idx
    }

    /// Quantization error: u − Q(u).  The clamped value of u is used.
    pub fn error(&self, u: S) -> S {
        let u_clamped = u.clamp_val(self.u_min, self.u_max);
        u_clamped - self.quantize(u)
    }

    /// Ideal signal-to-noise ratio in dB for a full-scale sinusoid.
    ///
    /// SNR_dB ≈ 6.02 · bits + 1.76
    ///
    /// The `signal_rms` parameter is accepted for API symmetry but the
    /// ideal formula depends only on the number of bits.
    pub fn snr_db(&self, _signal_rms: S) -> S {
        S::from_f64(6.02 * self.bits as f64 + 1.76)
    }

    /// Number of quantization levels.
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Step size Δ.
    pub fn delta(&self) -> S {
        self.delta
    }
}

// ---------------------------------------------------------------------------
// LogQuantizer
// ---------------------------------------------------------------------------

/// Logarithmic quantizer: coarser steps for larger signal magnitudes.
///
/// Quantization boundaries at ±δ_min · ρ^{-k} for k = 0, 1, …, n_levels−1,
/// so inner boundaries are smallest and outer boundaries are largest.
/// A dead-zone of [−δ_min, +δ_min] maps to zero.
pub struct LogQuantizer<S> {
    rho: S,
    delta_min: S,
    n_levels: usize,
}

impl<S: ControlScalar> LogQuantizer<S> {
    /// Create a new logarithmic quantizer.
    ///
    /// # Parameters
    /// - `rho`: ratio between adjacent levels, must be in (0, 1).  Smaller
    ///   values give more compression.
    /// - `delta_min`: smallest (inner) boundary magnitude, must be > 0.
    /// - `n_levels`: number of boundary levels above zero.
    ///
    /// # Errors
    /// Returns [`QuantizerError::InvalidParameter`] if constraints are violated.
    pub fn new(rho: S, delta_min: S, n_levels: usize) -> Result<Self, QuantizerError> {
        if rho <= S::ZERO || rho >= S::ONE {
            return Err(QuantizerError::InvalidParameter);
        }
        if delta_min <= S::ZERO {
            return Err(QuantizerError::InvalidParameter);
        }
        if n_levels == 0 {
            return Err(QuantizerError::InvalidParameter);
        }
        Ok(Self {
            rho,
            delta_min,
            n_levels,
        })
    }

    /// Boundary value for index `k` (0-based):
    /// b_k = delta_min * rho^{-k} = delta_min / rho^k
    fn boundary(&self, k: usize) -> S {
        // delta_min / rho^k  (rho < 1 so this grows with k)
        let pow_val = libm::pow(self.rho.to_f64(), k as f64);
        self.delta_min / S::from_f64(pow_val)
    }

    /// Quantize `u` to the nearest logarithmic level.
    ///
    /// The quantizer is sign-symmetric: Q(−u) = −Q(u).
    /// Signals with |u| < delta_min are mapped to 0.
    /// Signals exceeding the outermost boundary are clamped there.
    pub fn quantize(&self, u: S) -> S {
        let abs_u = S::from_f64(libm::fabs(u.to_f64()));
        let sign = if u < S::ZERO { -S::ONE } else { S::ONE };

        // Dead zone: |u| < delta_min → 0
        if abs_u < self.delta_min {
            return S::ZERO;
        }

        // Find the level k such that boundary(k) <= |u| < boundary(k+1)
        // boundary(k) = delta_min / rho^k  → grows with k (since rho < 1)
        // The quantized output is the midpoint of the two adjacent boundaries,
        // or the outer boundary if we exceed all levels.
        let mut best_k: usize = 0;
        for k in 0..self.n_levels {
            if abs_u >= self.boundary(k) {
                best_k = k;
            } else {
                break;
            }
        }

        // Quantize: midpoint between boundary(best_k) and boundary(best_k+1),
        // or just boundary(best_k) if at the outermost level.
        let lower = self.boundary(best_k);
        let quantized = if best_k + 1 < self.n_levels {
            let upper = self.boundary(best_k + 1);
            // Nearest boundary
            let mid = (lower + upper) * S::HALF;
            if abs_u < mid {
                lower
            } else {
                upper
            }
        } else {
            // Clamp at outermost boundary
            lower
        };

        sign * quantized
    }

    /// Number of levels.
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }
}

// ---------------------------------------------------------------------------
// DynamicQuantizer
// ---------------------------------------------------------------------------

/// Dynamic (zoom-based) quantizer.
///
/// Wraps a uniform quantizer and adaptively scales the zoom factor θ so
/// that the normalised input u/θ stays within the inner quantizer range.
///
/// u_q = θ · Q_inner(u / θ)
pub struct DynamicQuantizer<S> {
    inner: UniformQuantizer<S>,
    zoom: S,
    rho: S,
    zoom_min: S,
    zoom_max: S,
}

impl<S: ControlScalar> DynamicQuantizer<S> {
    /// Create a new dynamic quantizer.
    ///
    /// # Parameters
    /// - `bits`, `u_min`, `u_max`: forwarded to the inner uniform quantizer.
    /// - `rho`: zoom update factor, must be > 1.
    ///
    /// # Errors
    /// Returns [`QuantizerError::InvalidParameter`] for invalid arguments.
    pub fn new(bits: u8, u_min: S, u_max: S, rho: S) -> Result<Self, QuantizerError> {
        if rho <= S::ONE {
            return Err(QuantizerError::InvalidParameter);
        }
        let inner = UniformQuantizer::new(bits, u_min, u_max)?;
        Ok(Self {
            inner,
            zoom: S::ONE,
            rho,
            zoom_min: S::from_f64(1e-6),
            zoom_max: S::from_f64(1e6),
        })
    }

    /// Quantize `u`, updating the zoom factor adaptively.
    ///
    /// The zoom factor θ is updated first, then quantization is applied.
    ///
    /// # Errors
    /// Currently infallible; returns `Ok(S)` for API consistency.
    pub fn quantize(&mut self, u: S) -> Result<S, QuantizerError> {
        let ratio = u / self.zoom;
        let abs_ratio = S::from_f64(libm::fabs(ratio.to_f64()));

        // Zoom out if signal exceeds normalised range
        if abs_ratio > S::ONE {
            self.zoom = (self.zoom * self.rho).clamp_val(self.zoom_min, self.zoom_max);
        } else if abs_ratio < S::HALF {
            // Zoom in if signal is very small relative to zoom
            let new_zoom = self.zoom / self.rho;
            self.zoom = new_zoom.clamp_val(self.zoom_min, self.zoom_max);
        }

        // Re-compute ratio with updated zoom
        let ratio_new = u / self.zoom;
        let q = self.inner.quantize(ratio_new);
        Ok(self.zoom * q)
    }

    /// Current zoom factor θ.
    pub fn zoom_factor(&self) -> S {
        self.zoom
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // UniformQuantizer tests
    // -----------------------------------------------------------------------

    #[test]
    fn uniform_integer_aligned_signal_quantizes_exactly() {
        // 8-bit quantizer over [0, 255]: delta = 1.0, each integer is a level
        let q = UniformQuantizer::new(8, 0.0_f64, 255.0_f64).unwrap();
        assert!((q.quantize(0.0) - 0.0).abs() < 1e-9);
        assert!((q.quantize(1.0) - 1.0).abs() < 1e-9);
        assert!((q.quantize(128.0) - 128.0).abs() < 1e-9);
        assert!((q.quantize(255.0) - 255.0).abs() < 1e-9);
    }

    #[test]
    fn uniform_midpoint_rounds_to_nearest_level() {
        // 2-bit over [0, 3]: levels at 0, 1, 2, 3; delta = 1.0
        let q = UniformQuantizer::new(2, 0.0_f64, 3.0_f64).unwrap();
        // 0.5 should round to 1 (round-half-up via libm::round)
        let v = q.quantize(0.4);
        assert!((v - 0.0).abs() < 1e-9, "0.4 → 0, got {v}");
        let v = q.quantize(0.6);
        assert!((v - 1.0).abs() < 1e-9, "0.6 → 1, got {v}");
    }

    #[test]
    fn uniform_snr_formula_correct() {
        let q = UniformQuantizer::new(8, -1.0_f64, 1.0_f64).unwrap();
        let snr = q.snr_db(0.707);
        let expected = 6.02 * 8.0 + 1.76; // 49.92
        assert!(
            (snr - expected).abs() < 0.01,
            "SNR got {snr}, expected {expected}"
        );
    }

    #[test]
    fn uniform_out_of_range_clamps() {
        let q = UniformQuantizer::new(4, 0.0_f64, 1.0_f64).unwrap();
        // Below u_min
        let v = q.quantize(-5.0);
        assert!((v - 0.0).abs() < 1e-9, "Below min → 0.0, got {v}");
        // Above u_max
        let v = q.quantize(99.0);
        assert!((v - 1.0).abs() < 1e-9, "Above max → 1.0, got {v}");
    }

    #[test]
    fn uniform_error_is_small() {
        let q = UniformQuantizer::new(8, -1.0_f64, 1.0_f64).unwrap();
        let delta = q.delta();
        for i in 0..20 {
            let u = -1.0 + (i as f64) * 0.1;
            let e = q.error(u).abs();
            assert!(
                e <= delta / 2.0 + 1e-9,
                "Error {e} exceeds delta/2 at u={u}"
            );
        }
    }

    #[test]
    fn uniform_invalid_params_rejected() {
        assert!(UniformQuantizer::<f64>::new(0, 0.0, 1.0).is_err());
        assert!(UniformQuantizer::<f64>::new(8, 1.0, 0.0).is_err());
        assert!(UniformQuantizer::<f64>::new(8, 1.0, 1.0).is_err());
    }

    // -----------------------------------------------------------------------
    // LogQuantizer tests
    // -----------------------------------------------------------------------

    #[test]
    fn log_sign_symmetry() {
        let q = LogQuantizer::new(0.5_f64, 0.1, 5).unwrap();
        for i in 1..10 {
            let u = i as f64 * 0.15;
            let pos = q.quantize(u);
            let neg = q.quantize(-u);
            assert!(
                (pos + neg).abs() < 1e-9,
                "Sign symmetry violated at u={u}: Q(u)={pos}, Q(-u)={neg}"
            );
        }
    }

    #[test]
    fn log_dead_zone_maps_to_zero() {
        let q = LogQuantizer::new(0.5_f64, 1.0, 4).unwrap();
        assert!((q.quantize(0.0)).abs() < 1e-9);
        assert!((q.quantize(0.5)).abs() < 1e-9);
        assert!((q.quantize(-0.5)).abs() < 1e-9);
    }

    #[test]
    fn log_monotone_levels_positive() {
        // With rho=0.5, delta_min=1.0, levels at 1, 2, 4, 8, 16 (boundaries)
        let q = LogQuantizer::new(0.5_f64, 1.0, 5).unwrap();
        // Larger input should map to >= quantized value of smaller input
        let mut prev = q.quantize(1.1_f64);
        for i in 2..6 {
            let u = i as f64 * 1.5;
            let v = q.quantize(u);
            assert!(v >= prev - 1e-9, "Not monotone: u={u}, v={v}, prev={prev}");
            prev = v;
        }
    }

    #[test]
    fn log_invalid_params_rejected() {
        assert!(LogQuantizer::<f64>::new(1.5, 0.1, 4).is_err()); // rho >= 1
        assert!(LogQuantizer::<f64>::new(0.5, -1.0, 4).is_err()); // delta_min <= 0
        assert!(LogQuantizer::<f64>::new(0.5, 0.1, 0).is_err()); // n_levels == 0
    }

    // -----------------------------------------------------------------------
    // DynamicQuantizer tests
    // -----------------------------------------------------------------------

    #[test]
    fn dynamic_zoom_grows_with_large_signal() {
        let mut dq = DynamicQuantizer::new(4, -1.0_f64, 1.0_f64, 2.0).unwrap();
        let initial_zoom = dq.zoom_factor();
        // Feed a large signal repeatedly — zoom should increase
        for _ in 0..10 {
            let _ = dq.quantize(100.0);
        }
        assert!(
            dq.zoom_factor() > initial_zoom,
            "Zoom should grow with large signal, got {}",
            dq.zoom_factor()
        );
    }

    #[test]
    fn dynamic_zoom_shrinks_with_small_signal() {
        // Start with a large zoom, feed tiny signal
        let mut dq = DynamicQuantizer::new(4, -1.0_f64, 1.0_f64, 2.0).unwrap();
        // Push zoom up first
        for _ in 0..20 {
            let _ = dq.quantize(50.0);
        }
        let large_zoom = dq.zoom_factor();
        // Now feed a very small signal
        for _ in 0..30 {
            let _ = dq.quantize(0.0);
        }
        assert!(
            dq.zoom_factor() < large_zoom,
            "Zoom should shrink with small signal"
        );
    }

    #[test]
    fn dynamic_zoom_bounded() {
        let mut dq = DynamicQuantizer::new(4, -1.0_f64, 1.0_f64, 2.0).unwrap();
        for _ in 0..100 {
            let _ = dq.quantize(1e9);
        }
        assert!(dq.zoom_factor() <= 1e6 + 1e-6, "Zoom exceeded max");

        let mut dq2 = DynamicQuantizer::new(4, -1.0_f64, 1.0_f64, 2.0).unwrap();
        for _ in 0..200 {
            let _ = dq2.quantize(0.0);
        }
        assert!(dq2.zoom_factor() >= 1e-6 - 1e-12, "Zoom went below min");
    }

    #[test]
    fn dynamic_invalid_rho_rejected() {
        assert!(DynamicQuantizer::<f64>::new(8, -1.0, 1.0, 0.5).is_err()); // rho <= 1
        assert!(DynamicQuantizer::<f64>::new(8, -1.0, 1.0, 1.0).is_err()); // rho == 1
    }
}
