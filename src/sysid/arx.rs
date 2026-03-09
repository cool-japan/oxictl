//! ARX (AutoRegressive with eXogenous input) model identification.
// Suppress doc-formatting and numerical-loop lints that are intentional in this
// signal-processing module.
#![allow(
    clippy::needless_range_loop,
    clippy::doc_overindented_list_items,
    clippy::doc_lazy_continuation,
    clippy::manual_memcpy
)]
//!
//! The ARX model in shift-operator form is:
//! ```text
//!   A(q)·y(t) = B(q)·u(t) + e(t)
//! ```
//! where
//! ```text
//!   A(q) = 1 + a₁q⁻¹ + … + aₙₐq⁻ⁿᵃ
//!   B(q) = b₀q⁻ⁿᵏ + b₁q⁻⁽ⁿᵏ⁺¹⁾ + … + bₙᵦ₋₁q⁻⁽ⁿᵏ⁺ⁿᵇ⁻¹⁾
//! ```
//!
//! Identification is performed via Ordinary Least Squares (batch) or
//! Recursive Least Squares with forgetting factor (online).
//!
//! # Const parameters
//! - `NA`  — number of AR coefficients (order of A polynomial, excluding leading 1).
//! - `NB`  — number of exogenous (B) coefficients.
//! - `P`   — total regressor dimension, **must equal NA + NB**. This is required because
//!           Rust's stable const generics do not yet support const arithmetic on type
//!           parameters in array-size positions.
//! - `NK`  — dead-time (input delay) in samples (baked into regressor construction).
//!           Used as a const parameter only for `ArxIdentifier` and `RecursiveArx`.

use crate::core::scalar::ControlScalar;
use crate::sysid::SysIdError;

// ── Helper: Cholesky solve for symmetric positive-definite system ─────────────

/// Solve the symmetric positive-definite linear system A·x = b in place.
///
/// `a` is the N×N SPD matrix (modified in-place; upper triangle written with L).
/// `b` is the right-hand side (modified in-place to become the solution).
/// Returns `Err(SysIdError::SingularMatrix)` if a diagonal element is ≤ 0 during
/// the Cholesky factorisation (indicating the matrix is not SPD).
pub(crate) fn cholesky_solve_n<S: ControlScalar, const N: usize>(
    a: &mut [[S; N]; N],
    b: &mut [S; N],
) -> Result<(), SysIdError> {
    // Cholesky factorisation: A = L·Lᵀ  (L stored in lower triangle of a)
    for i in 0..N {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            if i == j {
                if s <= S::ZERO {
                    return Err(SysIdError::SingularMatrix);
                }
                a[i][j] = s.sqrt();
            } else {
                let diag = a[j][j];
                if diag == S::ZERO {
                    return Err(SysIdError::SingularMatrix);
                }
                a[i][j] = s / diag;
            }
        }
    }

    // Forward substitution: L·z = b
    for i in 0..N {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i][k] * b[k];
        }
        let diag = a[i][i];
        if diag == S::ZERO {
            return Err(SysIdError::SingularMatrix);
        }
        b[i] = s / diag;
    }

    // Back substitution: Lᵀ·x = z
    let mut i = N;
    while i > 0 {
        i -= 1;
        let mut s = b[i];
        for k in (i + 1)..N {
            s -= a[k][i] * b[k];
        }
        let diag = a[i][i];
        if diag == S::ZERO {
            return Err(SysIdError::SingularMatrix);
        }
        b[i] = s / diag;
    }

    Ok(())
}

// ── ArxModel ─────────────────────────────────────────────────────────────────

/// Identified ARX model.
///
/// Holds the AR coefficients `a[0..NA]` (denominator, excluding leading 1) and
/// the exogenous coefficients `b[0..NB]` (numerator).
///
/// The one-step-ahead prediction is:
/// ```text
///   ŷ(t) = −a₀·y(t-1) − … − aₙₐ₋₁·y(t-NA)
///         + b₀·u(t-NK) + … + bₙᵦ₋₁·u(t-NK-NB+1)
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ArxModel<S: ControlScalar, const NA: usize, const NB: usize> {
    /// AR coefficients a₁ … aₙₐ  (stored raw: the LS solution θ[0..NA]).
    ///
    /// The regressor φ[i] = -y(t-1-i), so the prediction is:
    ///   ŷ = Σ a[i]·φ[i] = -Σ a[i]·y(t-1-i)
    /// i.e. the system pole polynomial is A(q) = 1 + a[0]q⁻¹ + … + a[NA-1]q⁻ᴺᴬ
    pub a: [S; NA],
    /// Exogenous coefficients b₀ … bₙᵦ₋₁.
    pub b: [S; NB],
}

impl<S: ControlScalar, const NA: usize, const NB: usize> ArxModel<S, NA, NB> {
    /// Create a zero-initialised model.
    pub fn zeros() -> Self {
        Self {
            a: [S::ZERO; NA],
            b: [S::ZERO; NB],
        }
    }

    /// One-step-ahead prediction.
    ///
    /// `y_history` — most recent `NA` output samples, **newest first**:
    ///   `y_history[0] = y(t-1)`, `y_history[1] = y(t-2)`, …
    ///
    /// `u_history` — most recent `NB` input samples **starting at the dead-time
    ///   offset**, newest first:
    ///   `u_history[0] = u(t-NK)`, `u_history[1] = u(t-NK-1)`, …
    pub fn predict(&self, y_history: &[S; NA], u_history: &[S; NB]) -> S {
        let mut y_hat = S::ZERO;
        for i in 0..NA {
            y_hat -= self.a[i] * y_history[i];
        }
        for i in 0..NB {
            y_hat += self.b[i] * u_history[i];
        }
        y_hat
    }

    /// Simulate the model over an input sequence of length `M`, returning
    /// predicted outputs.
    ///
    /// `u_seq` — input sequence.  The layout expected is:
    ///   - Index 0 … NB-2: pre-sequence input history (oldest first if NK > 0).
    ///   - Index NB-1 … NB-1+M-1: the M actual inputs to simulate.
    ///   Minimum length: `max(NB, 1) + M - 1`.  For NB=0 length must be ≥ M.
    ///
    /// `y0` — most recent `NA` historical outputs, newest first.
    ///
    /// Returns `Err(InsufficientData)` if `u_seq` is too short.
    pub fn simulate<const M: usize>(
        &self,
        u_seq: &[S],
        y0: &[S; NA],
    ) -> Result<[S; M], SysIdError> {
        let needed = if NB > 0 { NB - 1 + M } else { M };
        if u_seq.len() < needed {
            return Err(SysIdError::InsufficientData);
        }

        let mut y_buf = *y0;
        let mut out = [S::ZERO; M];

        for step in 0..M {
            let base = if NB > 0 { NB - 1 } else { 0 };
            let mut u_hist = [S::ZERO; NB];
            for j in 0..NB {
                let idx = base + step;
                if idx >= j {
                    u_hist[j] = u_seq[idx - j];
                }
            }

            let y_new = self.predict(&y_buf, &u_hist);
            out[step] = y_new;

            // Shift y_buf newest-first
            let mut j = NA;
            while j > 1 {
                y_buf[j - 1] = y_buf[j - 2];
                j -= 1;
            }
            if NA > 0 {
                y_buf[0] = y_new;
            }
        }

        Ok(out)
    }
}

// ── ArxIdentifier (batch LS) ──────────────────────────────────────────────────

/// Batch ARX identifier using Ordinary Least Squares.
///
/// Builds the regression matrix Φ from the provided data and solves
/// θ = (ΦᵀΦ)⁻¹Φᵀy via Cholesky decomposition.
///
/// # Const parameters
/// - `NA` — AR order.
/// - `NB` — exogenous order.
/// - `P`  — regressor dimension = NA + NB  (**caller must ensure P == NA + NB**).
/// - `NK` — dead time (samples of input delay ≥ 0).
pub struct ArxIdentifier<
    S: ControlScalar,
    const NA: usize,
    const NB: usize,
    const P: usize,
    const NK: usize,
> {
    _phantom: core::marker::PhantomData<S>,
}

impl<S: ControlScalar, const NA: usize, const NB: usize, const P: usize, const NK: usize>
    ArxIdentifier<S, NA, NB, P, NK>
{
    /// Identify an ARX model from output sequence `y` and input sequence `u`.
    ///
    /// Both slices must have equal length ≥ `NA + NK + NB`.
    /// Returns `Err(InsufficientData)` if there are not enough samples.
    pub fn fit(y: &[S], u: &[S]) -> Result<ArxModel<S, NA, NB>, SysIdError> {
        let n = y.len();
        if u.len() != n {
            return Err(SysIdError::InvalidData);
        }
        for &v in y.iter().chain(u.iter()) {
            if !v.is_finite() {
                return Err(SysIdError::InvalidData);
            }
        }

        let min_start = if NK + NB > NA { NK + NB } else { NA };
        if n <= min_start {
            return Err(SysIdError::InsufficientData);
        }

        if P == 0 {
            return Ok(ArxModel::zeros());
        }

        let mut ata = [[S::ZERO; P]; P];
        let mut aty = [S::ZERO; P];

        for t in min_start..n {
            let mut phi = [S::ZERO; P];
            for i in 0..NA {
                phi[i] = -y[t - 1 - i];
            }
            for i in 0..NB {
                let delay = NK + i;
                if delay <= t {
                    phi[NA + i] = u[t - delay];
                }
            }
            let yt = y[t];

            for i in 0..P {
                for j in 0..P {
                    ata[i][j] += phi[i] * phi[j];
                }
                aty[i] += phi[i] * yt;
            }
        }

        cholesky_solve_n::<S, P>(&mut ata, &mut aty)?;

        let mut model = ArxModel::zeros();
        for i in 0..NA {
            model.a[i] = aty[i];
        }
        for i in 0..NB {
            model.b[i] = aty[NA + i];
        }
        Ok(model)
    }
}

// ── RecursiveArx (online RLS) ─────────────────────────────────────────────────

/// Online ARX identifier using Recursive Least Squares with forgetting factor.
///
/// # Const parameters
/// - `NA`  — AR order.
/// - `NB`  — exogenous order.
/// - `P`   — regressor dimension = NA + NB (**caller must ensure P == NA + NB**).
/// - `NK`  — dead time (samples of input delay ≥ 0).
/// - `UBN` — input buffer size = NK + NB (**caller must ensure UBN == NK + NB**).
#[derive(Debug, Clone)]
pub struct RecursiveArx<
    S: ControlScalar,
    const NA: usize,
    const NB: usize,
    const P: usize,
    const NK: usize,
    const UBN: usize,
> {
    theta: [S; P],
    cov: [[S; P]; P],
    lambda: S,
    y_buf: [S; NA],
    u_buf: [S; UBN],
    steps: u32,
}

impl<
        S: ControlScalar,
        const NA: usize,
        const NB: usize,
        const P: usize,
        const NK: usize,
        const UBN: usize,
    > RecursiveArx<S, NA, NB, P, NK, UBN>
{
    /// Construct a new online ARX identifier.
    ///
    /// # Arguments
    /// * `lambda` — forgetting factor ∈ (0, 1].
    /// * `p_init` — initial diagonal value of the covariance matrix.
    pub fn new(lambda: S, p_init: S) -> Self {
        let mut cov = [[S::ZERO; P]; P];
        for i in 0..P {
            cov[i][i] = p_init;
        }
        Self {
            theta: [S::ZERO; P],
            cov,
            lambda,
            y_buf: [S::ZERO; NA],
            u_buf: [S::ZERO; UBN],
            steps: 0,
        }
    }

    /// Process a new output/input observation pair.
    pub fn update(&mut self, y_new: S, u_new: S) -> Result<(), SysIdError> {
        if !y_new.is_finite() || !u_new.is_finite() {
            return Err(SysIdError::InvalidData);
        }

        // Shift input buffer newest-first
        {
            let mut i = UBN;
            while i > 1 {
                self.u_buf[i - 1] = self.u_buf[i - 2];
                i -= 1;
            }
            if UBN > 0 {
                self.u_buf[0] = u_new;
            }
        }

        // Build regressor φ = [-y(t-1), …, -y(t-NA), u(t-NK), …, u(t-NK-NB+1)]
        let mut phi = [S::ZERO; P];
        for i in 0..NA {
            phi[i] = -self.y_buf[i];
        }
        for i in 0..NB {
            let delay = NK + i;
            if delay < UBN {
                phi[NA + i] = self.u_buf[delay];
            }
        }

        // P·φ
        let mut p_phi = [S::ZERO; P];
        for i in 0..P {
            for j in 0..P {
                p_phi[i] += self.cov[i][j] * phi[j];
            }
        }

        // φᵀ·P·φ
        let mut phi_t_p_phi = S::ZERO;
        for i in 0..P {
            phi_t_p_phi += phi[i] * p_phi[i];
        }

        let denom = self.lambda + phi_t_p_phi;
        if denom == S::ZERO {
            return Err(SysIdError::SingularMatrix);
        }

        // Gain k
        let mut k = [S::ZERO; P];
        for i in 0..P {
            k[i] = p_phi[i] / denom;
        }

        // Innovation
        let mut y_hat = S::ZERO;
        for i in 0..P {
            y_hat += phi[i] * self.theta[i];
        }
        let innovation = y_new - y_hat;

        // θ update
        for i in 0..P {
            self.theta[i] += k[i] * innovation;
        }

        // P update
        let mut phi_t_p = [S::ZERO; P];
        for j in 0..P {
            for l in 0..P {
                phi_t_p[j] += phi[l] * self.cov[l][j];
            }
        }
        let lambda_inv = S::ONE / self.lambda;
        for i in 0..P {
            for j in 0..P {
                self.cov[i][j] = (self.cov[i][j] - k[i] * phi_t_p[j]) * lambda_inv;
            }
        }

        // Shift output buffer newest-first
        {
            let mut i = NA;
            while i > 1 {
                self.y_buf[i - 1] = self.y_buf[i - 2];
                i -= 1;
            }
            if NA > 0 {
                self.y_buf[0] = y_new;
            }
        }

        self.steps += 1;
        Ok(())
    }

    /// Return the current ARX model estimate.
    pub fn model(&self) -> ArxModel<S, NA, NB> {
        let mut model = ArxModel::zeros();
        for i in 0..NA {
            model.a[i] = self.theta[i];
        }
        for i in 0..NB {
            model.b[i] = self.theta[NA + i];
        }
        model
    }

    /// Number of samples processed so far.
    pub fn steps(&self) -> u32 {
        self.steps
    }
}

// ── fit_percent ───────────────────────────────────────────────────────────────

/// Compute the MATLAB-style FIT% metric.
///
/// ```text
///   FIT% = 100 · (1 − ‖y − ŷ‖₂ / ‖y − mean(y)‖₂)
/// ```
///
/// Returns `S::ZERO` if the denominator is zero (constant signal).
pub fn fit_percent<S: ControlScalar>(model_output: &[S], actual: &[S]) -> S {
    let n = model_output.len().min(actual.len());
    if n == 0 {
        return S::ZERO;
    }

    let mut sum = S::ZERO;
    for i in 0..n {
        sum += actual[i];
    }
    let mean = sum / S::from_f64(n as f64);

    let mut num_sq = S::ZERO;
    let mut den_sq = S::ZERO;
    for i in 0..n {
        let e = actual[i] - model_output[i];
        num_sq += e * e;
        let d = actual[i] - mean;
        den_sq += d * d;
    }

    if den_sq == S::ZERO {
        return S::ZERO;
    }

    let ratio = (num_sq / den_sq).sqrt();
    S::from_f64(100.0) * (S::ONE - ratio)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate data from a known first-order ARX system (noiseless):
    ///   y(t) = −a₁·y(t−1) + b₀·u(t−1)
    ///   true: A polynomial: 1 − 0.7q⁻¹  (pole at 0.7), b₀ = 0.3
    fn generate_arx1(n: usize) -> (heapless::Vec<f64, 4096>, heapless::Vec<f64, 4096>) {
        // True model: y(t) = 0.7·y(t-1) + 0.3·u(t-1)
        // In ARX form: a[0] stores coefficient such that ŷ = -a[0]·y(t-1) + b[0]·u(t-1)
        // i.e. -a[0] = 0.7  →  a[0] = -0.7 (the raw LS coefficient on -y(t-1))
        let a_true = 0.7_f64;
        let b_true = 0.3_f64;
        let mut y: heapless::Vec<f64, 4096> = heapless::Vec::new();
        let mut u: heapless::Vec<f64, 4096> = heapless::Vec::new();
        let _ = y.push(0.0);
        let _ = u.push(0.0);
        for t in 1..n {
            let ut = libm::sin(0.1 * t as f64) + libm::cos(0.07 * t as f64) * 0.5;
            let yt = a_true * y[t - 1] + b_true * u[t - 1];
            let _ = y.push(yt);
            let _ = u.push(ut);
        }
        (y, u)
    }

    #[test]
    fn batch_arx_identifies_first_order() {
        let (y, u) = generate_arx1(800);
        // NA=1, NB=1, P=2, NK=1
        let model = ArxIdentifier::<f64, 1, 1, 2, 1>::fit(y.as_slice(), u.as_slice())
            .expect("batch ARX fit should succeed");

        let mut predicted: heapless::Vec<f64, 4096> = heapless::Vec::new();
        for t in 1..y.len() {
            let y_hist = [y[t - 1]];
            let u_hist = [u[t - 1]];
            let _ = predicted.push(model.predict(&y_hist, &u_hist));
        }
        let fp = fit_percent(predicted.as_slice(), &y.as_slice()[1..]);
        assert!(
            fp > 99.0,
            "FIT% should exceed 99% for noiseless ARX-1, got {fp:.2}%"
        );
    }

    #[test]
    fn recursive_arx_identifies_first_order() {
        let (y, u) = generate_arx1(2000);
        // NA=1, NB=1, P=2, NK=1, UBN=NK+NB=2
        let mut rls = RecursiveArx::<f64, 1, 1, 2, 1, 2>::new(0.99, 1e4);
        for t in 1..y.len() {
            rls.update(y[t], u[t]).expect("update should not error");
        }
        let model = rls.model();

        let mut predicted: heapless::Vec<f64, 4096> = heapless::Vec::new();
        for t in 1..y.len() {
            let y_hist = [y[t - 1]];
            let u_hist = [u[t - 1]];
            let _ = predicted.push(model.predict(&y_hist, &u_hist));
        }
        let fp = fit_percent(predicted.as_slice(), &y.as_slice()[1..]);
        assert!(
            fp > 99.0,
            "RLS ARX FIT% should exceed 99% for noiseless ARX-1, got {fp:.2}%"
        );
    }

    #[test]
    fn fit_percent_perfect_prediction() {
        let y: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let fp = fit_percent(&y, &y);
        assert!(
            (fp - 100.0).abs() < 1e-9,
            "FIT% of perfect prediction should be 100%"
        );
    }

    #[test]
    fn fit_percent_zero_for_mean_prediction() {
        let y: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = [3.0_f64; 5];
        let fp = fit_percent(&mean, &y);
        assert!(fp <= 1e-9, "FIT% of mean-level prediction should be ≤ 0");
    }

    #[test]
    fn cholesky_solve_2x2() {
        // Solve [4, 2; 2, 3]·x = [8; 7]
        // det = 4·3 - 2·2 = 8
        // x[0] = (3·8 - 2·7) / 8 = (24 - 14) / 8 = 10/8 = 1.25
        // x[1] = (4·7 - 2·8) / 8 = (28 - 16) / 8 = 12/8 = 1.5
        let mut a = [[4.0_f64, 2.0], [2.0, 3.0]];
        let mut b = [8.0_f64, 7.0];
        cholesky_solve_n::<f64, 2>(&mut a, &mut b).expect("should succeed");
        assert!((b[0] - 1.25).abs() < 1e-10, "x[0]={}", b[0]);
        assert!((b[1] - 1.5).abs() < 1e-10, "x[1]={}", b[1]);
    }

    #[test]
    fn simulate_produces_finite_output() {
        let model: ArxModel<f64, 1, 1> = ArxModel {
            a: [-0.7],
            b: [0.3],
        };
        let mut u_seq = [0.0_f64; 101];
        for i in 1..=100 {
            u_seq[i] = libm::sin(0.1 * i as f64);
        }
        let y0 = [0.0_f64];
        let out: [f64; 100] = model
            .simulate(&u_seq, &y0)
            .expect("simulate should succeed");
        for i in 0..100 {
            assert!(out[i].is_finite(), "simulate output[{i}] is not finite");
        }
    }
}
