//! Affine Projection Algorithm (APA) adaptive filter.
//!
//! APA generalizes the Normalized LMS (NLMS) algorithm by using the last `P`
//! input vectors to form an affine projection, yielding faster convergence at
//! the cost of higher computational complexity.
//!
//! # Algorithm
//! At each step, the weight update is:
//! ```text
//! w[n+1] = w[n] + mu · X^T · (X·X^T + eps·I)^{-1} · e
//! ```
//! where `X` is the P×N input matrix (last P input vectors stacked row-wise)
//! and `e` is the P-vector of a posteriori errors.
//!
//! For `P = 1`, APA reduces exactly to NLMS.

use crate::core::adaptive_filters::lms::AdaptiveFilterError;
use crate::core::scalar::ControlScalar;

// ─────────────────────────────────────────────────────────────
//  Gauss elimination solver  (P×P system)
// ─────────────────────────────────────────────────────────────

/// Solve the linear system `A · result = b` using Gaussian elimination
/// with partial (row) pivoting. Works for square P×P systems.
///
/// The arrays `a` and `b` are modified in place during elimination.
///
/// # Errors
/// Returns [`AdaptiveFilterError::Divergence`] if a near-zero pivot is encountered
/// (system is singular or ill-conditioned).
#[allow(clippy::needless_range_loop)]
fn gauss_solve<S: ControlScalar, const P: usize>(
    a: &mut [[S; P]; P],
    b: &mut [S; P],
) -> Result<[S; P], AdaptiveFilterError> {
    let pivot_eps = S::from_f64(1e-30);

    // Forward elimination with partial pivoting
    for col in 0..P {
        // Find the row with the largest absolute value in this column (from col downward)
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..P {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            a.swap(max_row, col);
            b.swap(max_row, col);
        }

        let pivot = a[col][col];
        if pivot.abs() < pivot_eps {
            return Err(AdaptiveFilterError::Divergence);
        }

        // Eliminate below
        for row in (col + 1)..P {
            let factor = a[row][col] / pivot;
            for k in col..P {
                let sub = factor * a[col][k];
                a[row][k] -= sub;
            }
            let sub_b = factor * b[col];
            b[row] -= sub_b;
        }
    }

    // Back substitution
    let mut result = [S::ZERO; P];
    for row in (0..P).rev() {
        let mut sum = b[row];
        for k in (row + 1)..P {
            sum -= a[row][k] * result[k];
        }
        let pivot = a[row][row];
        if pivot.abs() < pivot_eps {
            return Err(AdaptiveFilterError::Divergence);
        }
        result[row] = sum / pivot;
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────
//  ApaFilter
// ─────────────────────────────────────────────────────────────

/// Affine Projection Algorithm (APA) adaptive FIR filter.
///
/// Generalizes NLMS with `P = 1` to `P` concurrent projections.
/// Higher `P` gives faster convergence at O(P²N) cost per step.
///
/// # Type Parameters
/// - `S` — scalar type implementing [`ControlScalar`]
/// - `N` — filter length (number of taps)
/// - `P` — projection order (number of past input vectors used); `P >= 1`
#[derive(Debug, Clone)]
pub struct ApaFilter<S: ControlScalar, const N: usize, const P: usize> {
    weights: [S; N],
    /// Ring buffer of last P input vectors (x_buf[0] = most recent).
    x_buf: [[S; N]; P],
    /// Ring buffer of last P desired signals (d_buf[0] = most recent).
    d_buf: [S; P],
    mu: S,
    eps: S,
}

impl<S: ControlScalar, const N: usize, const P: usize> ApaFilter<S, N, P> {
    /// Create a new APA filter with zero weights and empty input buffer.
    ///
    /// # Arguments
    /// * `mu`  — step size; must be positive
    /// * `eps` — regularization constant; must be positive
    ///
    /// # Errors
    /// Returns [`AdaptiveFilterError::InvalidStepSize`] if `mu <= 0` or `eps <= 0`.
    pub fn new(mu: S, eps: S) -> Result<Self, AdaptiveFilterError> {
        if mu <= S::ZERO || eps <= S::ZERO {
            return Err(AdaptiveFilterError::InvalidStepSize);
        }
        Ok(Self {
            weights: [S::ZERO; N],
            x_buf: [[S::ZERO; N]; P],
            d_buf: [S::ZERO; P],
            mu,
            eps,
        })
    }

    /// Process one sample and return the filter output (before weight update).
    ///
    /// Internally maintains a rolling buffer of the last `P` input vectors and
    /// desired signals to construct the affine projection.
    ///
    /// # Errors
    /// Returns [`AdaptiveFilterError::Divergence`] if the Gram matrix is singular
    /// or weights become non-finite.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, x: &[S; N], d: S) -> Result<S, AdaptiveFilterError> {
        // Step 1: Shift buffer — oldest entry drops, newest enters at index 0.
        for i in (1..P).rev() {
            self.x_buf[i] = self.x_buf[i - 1];
            self.d_buf[i] = self.d_buf[i - 1];
        }
        self.x_buf[0] = *x;
        self.d_buf[0] = d;

        // Step 2: Output y = w^T * x  (pre-update)
        let y: S = self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .fold(S::ZERO, |a, b| a + b);

        // Step 3: Compute error vector e[i] = d_buf[i] - w^T * x_buf[i]
        let mut e = [S::ZERO; P];
        for i in 0..P {
            let dot_wi: S = self
                .weights
                .iter()
                .zip(self.x_buf[i].iter())
                .map(|(&wi, &xij)| wi * xij)
                .fold(S::ZERO, |a, b| a + b);
            e[i] = self.d_buf[i] - dot_wi;
        }

        // Step 4: Gram matrix G = X * X^T  (P×P), G[i][k] = dot(x_buf[i], x_buf[k])
        let mut g = [[S::ZERO; P]; P];
        for i in 0..P {
            for k in 0..P {
                g[i][k] = self.x_buf[i]
                    .iter()
                    .zip(self.x_buf[k].iter())
                    .map(|(&a, &b)| a * b)
                    .fold(S::ZERO, |acc, v| acc + v);
            }
        }

        // Step 5: Regularize: G[i][i] += eps
        let eps = self.eps;
        for (i, g_row) in g.iter_mut().enumerate() {
            g_row[i] += eps;
        }

        // Step 6: Solve (G) * a = e
        let a = gauss_solve::<S, P>(&mut g, &mut e)?;

        // Step 7: Weight update: w[j] += mu * sum_i( a[i] * x_buf[i][j] )
        let mu = self.mu;
        for j in 0..N {
            let correction: S = a
                .iter()
                .zip(self.x_buf.iter())
                .map(|(&ai, xrow)| ai * xrow[j])
                .fold(S::ZERO, |acc, v| acc + v);
            self.weights[j] += mu * correction;
            if !self.weights[j].is_finite() {
                return Err(AdaptiveFilterError::Divergence);
            }
        }

        Ok(y)
    }

    /// Return a reference to the current weight vector.
    pub fn weights(&self) -> &[S; N] {
        &self.weights
    }

    /// Reset weights and input buffers to zero.
    pub fn reset(&mut self) {
        self.weights = [S::ZERO; N];
        self.x_buf = [[S::ZERO; N]; P];
        self.d_buf = [S::ZERO; P];
    }
}

// ─────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::adaptive_filters::lms::NlmsFilter;

    // Deterministic LCG for test signal generation.
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_f64(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (self.state >> 11) as f64 / (1u64 << 53) as f64;
            u * 2.0 - 1.0
        }
    }

    fn apply_fir(x_history: &[f64; 4], coeffs: &[f64; 4]) -> f64 {
        coeffs
            .iter()
            .zip(x_history.iter())
            .map(|(c, x)| c * x)
            .sum()
    }

    #[test]
    fn apa_p1_matches_nlms_behavior() {
        // APA with P=1 and eps regularization should behave identically to NLMS
        // (within floating-point rounding) since the 1×1 Gram system trivially
        // yields a[0] = e / (||x||^2 + eps) and w += mu * a[0] * x.
        let true_coeffs: [f64; 4] = [0.5, 0.3, 0.1, 0.05];
        let mu = 0.5_f64;
        let eps = 1e-6_f64;

        let mut apa = ApaFilter::<f64, 4, 1>::new(mu, eps).expect("valid");
        let mut nlms = NlmsFilter::<f64, 4>::new(mu, eps).expect("valid");

        let mut lcg = Lcg::new(42);
        let mut x_hist = [0.0_f64; 4];
        let mut mse_apa = 0.0_f64;
        let mut mse_nlms = 0.0_f64;
        let n_iter = 2000usize;

        for iter in 0..n_iter {
            x_hist[3] = x_hist[2];
            x_hist[2] = x_hist[1];
            x_hist[1] = x_hist[0];
            x_hist[0] = lcg.next_f64();
            let d = apply_fir(&x_hist, &true_coeffs);

            let y_apa = apa.update(&x_hist, d).expect("apa ok");
            let y_nlms = nlms.update(&x_hist, d).expect("nlms ok");

            if iter >= n_iter - 500 {
                mse_apa += (d - y_apa).powi(2);
                mse_nlms += (d - y_nlms).powi(2);
            }
        }
        mse_apa /= 500.0;
        mse_nlms /= 500.0;

        // Both should converge to similar MSE
        assert!(mse_apa < 1e-4, "APA P=1 MSE too high: {mse_apa}");
        assert!(mse_nlms < 1e-4, "NLMS MSE too high: {mse_nlms}");
    }

    #[test]
    fn apa_p2_faster_convergence_than_p1() {
        // APA with P=2 should converge faster than P=1 (NLMS) in the early phase.
        let true_coeffs: [f64; 4] = [0.5, 0.3, 0.1, 0.05];
        let mu = 0.5_f64;
        let eps = 1e-4_f64;

        let mut apa1 = ApaFilter::<f64, 4, 1>::new(mu, eps).expect("valid");
        let mut apa2 = ApaFilter::<f64, 4, 2>::new(mu, eps).expect("valid");

        let mut lcg1 = Lcg::new(1337);
        let mut lcg2 = Lcg::new(1337);
        let mut x1 = [0.0_f64; 4];
        let mut x2 = [0.0_f64; 4];

        let n_early = 200usize;
        let mut mse_p1 = 0.0_f64;
        let mut mse_p2 = 0.0_f64;

        for iter in 0..n_early {
            let s1 = lcg1.next_f64();
            let s2 = lcg2.next_f64();
            x1[3] = x1[2];
            x1[2] = x1[1];
            x1[1] = x1[0];
            x1[0] = s1;
            x2[3] = x2[2];
            x2[2] = x2[1];
            x2[1] = x2[0];
            x2[0] = s2;

            let d1 = apply_fir(&x1, &true_coeffs);
            let d2 = apply_fir(&x2, &true_coeffs);

            let y1 = apa1.update(&x1, d1).expect("p1 ok");
            let y2 = apa2.update(&x2, d2).expect("p2 ok");

            if iter >= n_early - 100 {
                mse_p1 += (d1 - y1).powi(2);
                mse_p2 += (d2 - y2).powi(2);
            }
        }
        mse_p1 /= 100.0;
        mse_p2 /= 100.0;

        assert!(
            mse_p2 < mse_p1,
            "APA P=2 ({mse_p2:.6}) should converge faster than P=1 ({mse_p1:.6})"
        );
    }

    #[test]
    fn apa_identifies_fir_system() {
        // APA with P=3 identifies a 4-tap FIR system accurately.
        let true_coeffs: [f64; 4] = [0.4, 0.25, 0.2, 0.1];
        let mut apa = ApaFilter::<f64, 4, 3>::new(0.5, 1e-5).expect("valid");

        let mut lcg = Lcg::new(555);
        let mut x_hist = [0.0_f64; 4];
        let n_iter = 3000usize;
        let mut mse = 0.0_f64;

        for iter in 0..n_iter {
            x_hist[3] = x_hist[2];
            x_hist[2] = x_hist[1];
            x_hist[1] = x_hist[0];
            x_hist[0] = lcg.next_f64();
            let d = apply_fir(&x_hist, &true_coeffs);
            let y = apa.update(&x_hist, d).expect("ok");
            if iter >= n_iter - 500 {
                mse += (d - y).powi(2);
            }
        }
        mse /= 500.0;
        assert!(mse < 1e-8, "APA P=3 did not converge: MSE = {mse}");
    }

    #[test]
    fn apa_reset_clears_state() {
        let mut apa = ApaFilter::<f64, 4, 2>::new(0.5, 1e-6).expect("valid");
        let x = [1.0_f64, 0.5, 0.25, 0.1];
        for _ in 0..100 {
            let _ = apa.update(&x, 1.0).expect("ok");
        }
        apa.reset();
        for &w in apa.weights().iter() {
            assert_eq!(w, 0.0, "weight not zero after reset: {w}");
        }
    }

    #[test]
    fn apa_invalid_params() {
        assert!(ApaFilter::<f64, 4, 2>::new(0.0, 1e-6).is_err());
        assert!(ApaFilter::<f64, 4, 2>::new(0.5, 0.0).is_err());
        assert!(ApaFilter::<f64, 4, 2>::new(-0.1, 1e-6).is_err());
    }

    #[test]
    fn gauss_solve_trivial() {
        // 2×2 system: [2 1; 1 3] * [x; y] = [5; 10]
        // Solution: x = 1, y = 3
        let mut a = [[2.0_f64, 1.0], [1.0, 3.0]];
        let mut b = [5.0_f64, 10.0];
        let result = gauss_solve::<f64, 2>(&mut a, &mut b).expect("non-singular");
        assert!((result[0] - 1.0).abs() < 1e-10, "x = {}", result[0]);
        assert!((result[1] - 3.0).abs() < 1e-10, "y = {}", result[1]);
    }
}
