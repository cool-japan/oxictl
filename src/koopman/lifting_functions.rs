//! Lifting functions (observable maps) for Koopman operator theory.
//!
//! A lifting function ψ: ℝᴺ → ℝᴸ embeds a (possibly nonlinear) state space
//! into a higher-dimensional space in which the dynamics become (approximately)
//! linear. Three concrete lifting strategies are provided:
//!
//! - [`PolynomialLifting`]: original state plus quadratic cross-terms.
//! - [`RbfLifting`]: original state plus radial-basis-function activations.
//! - [`DelayEmbedding`]: Takens-style delay-coordinate embedding.

use crate::core::scalar::ControlScalar;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can arise in Koopman lifting and EDMD operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KoopmanError {
    /// The model has not been fitted yet.
    NotFitted,
    /// Fewer data points were provided than the minimum required.
    InsufficientData,
    /// The Gram matrix is singular (cannot be inverted).
    SingularMatrix,
    /// Array dimension mismatch.
    InvalidDimension,
    /// Parameter value is out of valid range (e.g. sigma <= 0, q <= 0).
    InvalidParameter,
}

impl core::fmt::Display for KoopmanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            KoopmanError::NotFitted => write!(f, "model is not fitted"),
            KoopmanError::InsufficientData => write!(f, "insufficient data"),
            KoopmanError::SingularMatrix => write!(f, "matrix is singular"),
            KoopmanError::InvalidDimension => write!(f, "invalid dimension"),
            KoopmanError::InvalidParameter => write!(f, "invalid parameter"),
        }
    }
}

// ── LiftingMap trait ──────────────────────────────────────────────────────────

/// Trait for immutable lifting functions ψ: ℝᴺ → ℝᴸ.
pub trait LiftingMap<S, const N: usize, const L: usize>: Clone {
    /// Compute ψ(x) and return the L-dimensional lifted vector.
    fn lift(&self, x: &[S; N]) -> [S; L];

    /// Return the output dimension L (convenience; equals the const generic).
    #[inline]
    fn output_dim(&self) -> usize {
        L
    }
}

// ── PolynomialLifting ─────────────────────────────────────────────────────────

/// Polynomial (quadratic) lifting.
///
/// The lifted vector is:
/// ```text
///   ψ(x) = [x₀, x₁, …, x_{N-1},   x₀·x₀, x₀·x₁, …, x_{N-1}·x_{N-1}, 0, …]
/// ```
/// The first `N` slots hold the original state; the next `N*(N+1)/2` slots hold
/// the upper-triangular quadratic products (including squares). Any remaining
/// slots up to `L` are filled with zero.
#[derive(Clone, Debug)]
pub struct PolynomialLifting<S, const N: usize, const L: usize> {
    _marker: core::marker::PhantomData<S>,
}

impl<S: ControlScalar, const N: usize, const L: usize> PolynomialLifting<S, N, L> {
    /// Create a new `PolynomialLifting`.
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }

    /// Lift state vector `x` into the L-dimensional feature space.
    #[allow(clippy::needless_range_loop)]
    pub fn lift(&self, x: &[S; N]) -> [S; L] {
        let mut out = [S::ZERO; L];

        // First N entries: linear (original state)
        let linear_end = N.min(L);
        out[..linear_end].copy_from_slice(&x[..linear_end]);

        // Remaining entries: quadratic cross-terms x[i]*x[j] for i ≤ j
        let mut idx = N;
        'outer: for i in 0..N {
            for j in i..N {
                if idx >= L {
                    break 'outer;
                }
                out[idx] = x[i] * x[j];
                idx += 1;
            }
        }

        out
    }
}

impl<S: ControlScalar, const N: usize, const L: usize> Default for PolynomialLifting<S, N, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: ControlScalar, const N: usize, const L: usize> LiftingMap<S, N, L>
    for PolynomialLifting<S, N, L>
{
    fn lift(&self, x: &[S; N]) -> [S; L] {
        PolynomialLifting::lift(self, x)
    }
}

// ── RbfLifting ────────────────────────────────────────────────────────────────

/// Radial-Basis-Function lifting.
///
/// The lifted vector is:
/// ```text
///   ψ(x) = [x₀, …, x_{N-1},   φ₀(x), …, φ_{K-1}(x)]
/// ```
/// where `φₖ(x) = exp(−‖x − cₖ‖² / (2σ²))`.
///
/// The const parameter `L` **must** equal `N + K`.
#[derive(Clone, Debug)]
pub struct RbfLifting<S, const N: usize, const K: usize, const L: usize> {
    /// RBF centres, shape `[K][N]`.
    centers: [[S; N]; K],
    /// Bandwidth parameter σ (must be positive).
    sigma: S,
}

impl<S: ControlScalar, const N: usize, const K: usize, const L: usize> RbfLifting<S, N, K, L> {
    /// Create a new `RbfLifting` from `K` centres and bandwidth `sigma`.
    ///
    /// # Errors
    /// Returns [`KoopmanError::InvalidParameter`] if `sigma <= 0`.
    pub fn new(centers: [[S; N]; K], sigma: S) -> Result<Self, KoopmanError> {
        if sigma <= S::ZERO {
            return Err(KoopmanError::InvalidParameter);
        }
        Ok(Self { centers, sigma })
    }

    /// Lift state vector `x` into the L-dimensional feature space.
    #[allow(clippy::needless_range_loop)]
    pub fn lift(&self, x: &[S; N]) -> [S; L] {
        let mut out = [S::ZERO; L];

        // Linear part: copy state
        let linear_end = N.min(L);
        out[..linear_end].copy_from_slice(&x[..linear_end]);

        // RBF activations: φₖ = exp(−dist² / (2σ²))
        let sigma_f64 = self.sigma.to_f64();
        let denom = 2.0_f64 * sigma_f64 * sigma_f64;

        for k in 0..K {
            let slot = N + k;
            if slot >= L {
                break;
            }
            let mut dist_sq = 0.0_f64;
            for i in 0..N {
                let diff = (x[i] - self.centers[k][i]).to_f64();
                dist_sq += diff * diff;
            }
            out[slot] = S::from_f64(libm::exp(-dist_sq / denom));
        }

        out
    }
}

impl<S: ControlScalar, const N: usize, const K: usize, const L: usize> LiftingMap<S, N, L>
    for RbfLifting<S, N, K, L>
{
    fn lift(&self, x: &[S; N]) -> [S; L] {
        RbfLifting::lift(self, x)
    }
}

// ── DelayEmbedding ────────────────────────────────────────────────────────────

/// Takens delay-coordinate embedding.
///
/// Maintains a circular buffer of the last `D` state snapshots. On each call
/// to `push`, the current state is stored and the full delay window is
/// returned flattened into an `L`-vector where `L == N * D`.
///
/// Layout: `[x[k-D+1], …, x[k-1], x[k]]` — **oldest first** (chronological).
/// Before `D` states have been observed the missing history slots are zero.
#[derive(Clone, Debug)]
pub struct DelayEmbedding<S, const N: usize, const D: usize, const L: usize> {
    /// Circular buffer of the last D snapshots.
    history: [[S; N]; D],
    /// Index of the slot to write *next* (wraps modulo D).
    head: usize,
    /// Number of snapshots pushed so far (saturates at D).
    filled: usize,
}

impl<S: ControlScalar, const N: usize, const D: usize, const L: usize> DelayEmbedding<S, N, D, L> {
    /// Create a new `DelayEmbedding` initialised with all-zero history.
    pub fn new() -> Self {
        Self {
            history: [[S::ZERO; N]; D],
            head: 0,
            filled: 0,
        }
    }

    /// Push a new snapshot `x` and return the current flattened delay vector.
    ///
    /// The output is laid out oldest-first (chronological order):
    /// `[x[k-D+1], …, x[k-1], x[k]]`.
    pub fn push(&mut self, x: &[S; N]) -> [S; L] {
        // Write x into the ring buffer
        self.history[self.head] = *x;
        self.head = (self.head + 1) % D;
        if self.filled < D {
            self.filled += 1;
        }

        // Flatten in chronological (oldest→newest) order.
        // Oldest slot index: (head + D - filled) % D
        let oldest = (self.head + D - self.filled) % D;

        let mut out = [S::ZERO; L];
        for slot in 0..self.filled {
            let ring_idx = (oldest + slot) % D;
            let base = slot * N;
            for i in 0..N {
                if base + i < L {
                    out[base + i] = self.history[ring_idx][i];
                }
            }
        }
        out
    }

    /// Reset the embedding buffer to all zeros.
    pub fn reset(&mut self) {
        self.history = [[S::ZERO; N]; D];
        self.head = 0;
        self.filled = 0;
    }
}

impl<S: ControlScalar, const N: usize, const D: usize, const L: usize> Default
    for DelayEmbedding<S, N, D, L>
{
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PolynomialLifting ─────────────────────────────────────────────────────

    #[test]
    fn polynomial_first_n_slots_equal_x() {
        let lift: PolynomialLifting<f64, 3, 9> = PolynomialLifting::new();
        let x = [1.0_f64, 2.0, 3.0];
        let out = lift.lift(&x);
        assert!((out[0] - 1.0).abs() < 1e-12, "slot 0");
        assert!((out[1] - 2.0).abs() < 1e-12, "slot 1");
        assert!((out[2] - 3.0).abs() < 1e-12, "slot 2");
    }

    #[test]
    fn polynomial_quadratic_terms_correct() {
        // N=2, L=6: x=[2,3]
        // slots 0,1 = x; slots 2,3,4 = x0*x0=4, x0*x1=6, x1*x1=9
        let lift: PolynomialLifting<f64, 2, 6> = PolynomialLifting::new();
        let x = [2.0_f64, 3.0];
        let out = lift.lift(&x);
        assert!((out[2] - 4.0).abs() < 1e-12, "x0*x0: {}", out[2]);
        assert!((out[3] - 6.0).abs() < 1e-12, "x0*x1: {}", out[3]);
        assert!((out[4] - 9.0).abs() < 1e-12, "x1*x1: {}", out[4]);
    }

    #[test]
    fn polynomial_truncated_when_l_smaller() {
        // N=3, L=4: only first 4 slots filled; slot 3 = x[0]*x[0] = 1
        let lift: PolynomialLifting<f64, 3, 4> = PolynomialLifting::new();
        let x = [1.0_f64, 2.0, 3.0];
        let out = lift.lift(&x);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 2.0).abs() < 1e-12);
        assert!((out[2] - 3.0).abs() < 1e-12);
        assert!((out[3] - 1.0).abs() < 1e-12, "x[0]*x[0]=1: {}", out[3]);
    }

    #[test]
    fn polynomial_output_dim_matches_l() {
        let lift: PolynomialLifting<f64, 2, 6> = PolynomialLifting::new();
        assert_eq!(lift.output_dim(), 6);
    }

    // ── RbfLifting ────────────────────────────────────────────────────────────

    #[test]
    fn rbf_at_center_equals_one() {
        // K=1, N=2, L=3; query exactly at center → exp(0) = 1
        let centers = [[1.0_f64, 2.0]];
        let lift: RbfLifting<f64, 2, 1, 3> = RbfLifting::new(centers, 1.0).expect("new");
        let x = [1.0_f64, 2.0];
        let out = lift.lift(&x);
        assert!((out[2] - 1.0).abs() < 1e-12, "rbf at center: {}", out[2]);
    }

    #[test]
    fn rbf_far_from_center_near_zero() {
        // K=1, N=2, L=3; query very far from center → activation ≈ 0
        let centers = [[0.0_f64, 0.0]];
        let lift: RbfLifting<f64, 2, 1, 3> = RbfLifting::new(centers, 1.0).expect("new");
        let x = [100.0_f64, 100.0];
        let out = lift.lift(&x);
        assert!(out[2] < 1e-10, "rbf far from center: {}", out[2]);
    }

    #[test]
    fn rbf_invalid_sigma_returns_error() {
        let result = RbfLifting::<f64, 2, 1, 3>::new([[0.0, 0.0]], 0.0);
        assert!(matches!(result, Err(KoopmanError::InvalidParameter)));
    }

    #[test]
    fn rbf_linear_slots_preserved() {
        let centers = [[0.0_f64]];
        let lift: RbfLifting<f64, 1, 1, 2> = RbfLifting::new(centers, 1.0).expect("new");
        let x = [5.0_f64];
        let out = lift.lift(&x);
        assert!((out[0] - 5.0).abs() < 1e-12);
    }

    // ── DelayEmbedding ────────────────────────────────────────────────────────

    #[test]
    fn delay_push_fills_correctly() {
        // N=2, D=3, L=6; push 3 snapshots; output in chronological order
        let mut emb: DelayEmbedding<f64, 2, 3, 6> = DelayEmbedding::new();
        let _ = emb.push(&[1.0, 2.0]);
        let _ = emb.push(&[3.0, 4.0]);
        let out = emb.push(&[5.0, 6.0]);
        // Oldest first: [1,2, 3,4, 5,6]
        assert!((out[0] - 1.0).abs() < 1e-12, "out[0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-12, "out[1]={}", out[1]);
        assert!((out[2] - 3.0).abs() < 1e-12, "out[2]={}", out[2]);
        assert!((out[3] - 4.0).abs() < 1e-12, "out[3]={}", out[3]);
        assert!((out[4] - 5.0).abs() < 1e-12, "out[4]={}", out[4]);
        assert!((out[5] - 6.0).abs() < 1e-12, "out[5]={}", out[5]);
    }

    #[test]
    fn delay_reset_clears() {
        let mut emb: DelayEmbedding<f64, 2, 3, 6> = DelayEmbedding::new();
        let _ = emb.push(&[10.0, 20.0]);
        let _ = emb.push(&[30.0, 40.0]);
        emb.reset();
        // After reset, a single push of zeros yields only that slot non-zero
        let out = emb.push(&[0.0, 0.0]);
        for v in out.iter() {
            assert!(v.abs() < 1e-12, "expected 0 after reset, got {v}");
        }
    }

    #[test]
    fn delay_initial_zeros_before_filled() {
        let mut emb: DelayEmbedding<f64, 2, 3, 6> = DelayEmbedding::new();
        let out = emb.push(&[1.0, 2.0]);
        // Only 1 snapshot pushed (filled=1): out[0..2]=[1,2], rest=0
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 2.0).abs() < 1e-12);
        assert!((out[2]).abs() < 1e-12);
        assert!((out[3]).abs() < 1e-12);
    }

    #[test]
    fn delay_circular_wrapping() {
        // N=1, D=2, L=2: push 4 values; only most-recent 2 survive
        let mut emb: DelayEmbedding<f64, 1, 2, 2> = DelayEmbedding::new();
        let _ = emb.push(&[10.0]);
        let _ = emb.push(&[20.0]);
        let _ = emb.push(&[30.0]);
        let out = emb.push(&[40.0]);
        // Oldest first: [30, 40]
        assert!((out[0] - 30.0).abs() < 1e-12, "out[0]={}", out[0]);
        assert!((out[1] - 40.0).abs() < 1e-12, "out[1]={}", out[1]);
    }
}
