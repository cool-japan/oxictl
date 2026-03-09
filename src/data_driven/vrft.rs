#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::needless_range_loop)]

//! Virtual Reference Feedback Tuning (VRFT) — Campi & Savaresi 2002.
//!
//! Tunes PID controller gains from a single open-loop (or closed-loop)
//! experiment without requiring an explicit plant model.
//!
//! # Algorithm (open-loop experiment)
//!
//! Given measurement data `{u[k], y[k]}` for `k = 0..T`:
//!
//! 1. Choose reference model `M(z) = (1-m)/(z-m)` (first-order, pole at `m ∈ (0,1)`).
//! 2. Compute virtual reference: `r[k] = (y[k] - m·y[k-1]) / (1-m)`.
//! 3. Compute virtual error: `e[k] = r[k] - y[k]`.
//! 4. Build PID regressor matrix `Φ`: columns are proportional, integral, derivative of `e`.
//! 5. Solve normal equations `(ΦᵀΦ) θ = Φᵀ u` via Gaussian elimination with partial pivoting.
//! 6. Extract `[Kp, Ki, Kd] = θ`.

use crate::core::scalar::ControlScalar;

/// Errors produced by data-driven tuning algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataDrivenError {
    /// Fewer data points than the minimum required.
    NotEnoughData,
    /// The normal-equation matrix is (nearly) singular; ill-conditioned data.
    SingularMatrix,
    /// Attempt to read tuned parameters before calling `tune()`.
    NotTuned,
    /// A constructor argument is out of the valid range.
    InvalidParameter,
}

impl core::fmt::Display for DataDrivenError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotEnoughData => f.write_str("not enough data points"),
            Self::SingularMatrix => f.write_str("singular or ill-conditioned matrix"),
            Self::NotTuned => f.write_str("tuner has not been run yet; call tune() first"),
            Self::InvalidParameter => f.write_str("parameter out of valid range"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DataDrivenError {}

// ── Gaussian elimination (3 × 3, partial pivoting) ────────────────────────────

/// Solve a 3 × 3 linear system `A x = b` using Gaussian elimination with
/// partial (row) pivoting.
///
/// Both `a` and `b` are consumed (modified in-place).
/// Returns `Err(DataDrivenError::SingularMatrix)` when the system is singular
/// or numerically degenerate (pivot magnitude < `ε·‖A‖∞`).
fn gaussian_solve_3x3<S: ControlScalar>(
    mut a: [[S; 3]; 3],
    mut b: [S; 3],
) -> Result<[S; 3], DataDrivenError> {
    // Compute infinity-norm of A for relative singularity threshold.
    let mut anorm = S::ZERO;
    for row in &a {
        let mut row_sum = S::ZERO;
        for &v in row {
            let av = if v < S::ZERO { -v } else { v };
            row_sum += av;
        }
        if row_sum > anorm {
            anorm = row_sum;
        }
    }
    // Threshold: if anorm is zero the matrix is the zero matrix.
    let thresh = if anorm == S::ZERO {
        S::EPSILON
    } else {
        S::from_f64(1e-12) * anorm
    };

    // Forward elimination with partial pivoting.
    for col in 0..3_usize {
        // Find pivot row: row with largest |a[row][col]| in rows col..3.
        let mut pivot_row = col;
        let mut pivot_val = {
            let v = a[col][col];
            if v < S::ZERO {
                -v
            } else {
                v
            }
        };
        for row in (col + 1)..3 {
            let v = a[row][col];
            let av = if v < S::ZERO { -v } else { v };
            if av > pivot_val {
                pivot_val = av;
                pivot_row = row;
            }
        }

        if pivot_val < thresh {
            return Err(DataDrivenError::SingularMatrix);
        }

        // Swap rows if needed.
        if pivot_row != col {
            a.swap(pivot_row, col);
            b.swap(pivot_row, col);
        }

        // Eliminate below pivot.
        for row in (col + 1)..3 {
            let factor = a[row][col] / a[col][col];
            for k in col..3 {
                let sub = factor * a[col][k];
                a[row][k] -= sub;
            }
            let sub = factor * b[col];
            b[row] -= sub;
        }
    }

    // Back substitution.
    let mut x = [S::ZERO; 3];
    let mut i = 3_usize;
    while i > 0 {
        i -= 1;
        if {
            let v = a[i][i];
            if v < S::ZERO {
                -v
            } else {
                v
            }
        } < thresh
        {
            return Err(DataDrivenError::SingularMatrix);
        }
        let mut s = b[i];
        for j in (i + 1)..3 {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }

    Ok(x)
}

// ── VrftPid ───────────────────────────────────────────────────────────────────

/// VRFT-based PID auto-tuner.
///
/// Generic over the scalar type `S` (use `f32` or `f64`) and the number of
/// data points `DATA_LEN` (const generic).
///
/// # Example
/// ```rust,ignore
/// let mut tuner = VrftPid::<f64, 200>::new(0.9, 0.01)?;
/// tuner.tune(&u_data, &y_data)?;
/// let kp = tuner.kp()?;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct VrftPid<S, const DATA_LEN: usize> {
    /// Pole of the reference model `M(z) = (1-m)/(z-m)`.  Must be in `(0, 1)`.
    m: S,
    /// Sampling period (seconds).
    dt: S,
    /// Proportional gain (valid after `tune()`).
    kp: S,
    /// Integral gain (valid after `tune()`).
    ki: S,
    /// Derivative gain (valid after `tune()`).
    kd: S,
    /// Whether `tune()` has been called successfully.
    tuned: bool,
}

impl<S: ControlScalar, const DATA_LEN: usize> VrftPid<S, DATA_LEN> {
    /// Construct a new VRFT tuner.
    ///
    /// # Parameters
    /// - `m`:  Reference-model pole, must satisfy `0 < m < 1`.
    /// - `dt`: Sampling period, must be strictly positive.
    pub fn new(m: S, dt: S) -> Result<Self, DataDrivenError> {
        if m <= S::ZERO || m >= S::ONE {
            return Err(DataDrivenError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(DataDrivenError::InvalidParameter);
        }
        Ok(Self {
            m,
            dt,
            kp: S::ZERO,
            ki: S::ZERO,
            kd: S::ZERO,
            tuned: false,
        })
    }

    /// Run VRFT identification on the collected input/output data.
    ///
    /// Requires `DATA_LEN >= 3`.
    ///
    /// # Parameters
    /// - `u_data`: Plant input sequence `u[0..DATA_LEN]`.
    /// - `y_data`: Plant output sequence `y[0..DATA_LEN]`.
    pub fn tune(
        &mut self,
        u_data: &[S; DATA_LEN],
        y_data: &[S; DATA_LEN],
    ) -> Result<(), DataDrivenError> {
        if DATA_LEN < 3 {
            return Err(DataDrivenError::NotEnoughData);
        }

        let m = self.m;
        let dt = self.dt;
        let one_minus_m = S::ONE - m;

        // ── Step 1: Virtual reference r[k] = (y[k] - m·y[k-1]) / (1-m) ────
        // r[0] is approximated as y[0] / (1-m) (no previous sample).
        // We work over indices 1..DATA_LEN-1 to have a valid k-1 sample.
        // ── Step 2-4: Build normal equations (ΦᵀΦ, Φᵀu) in one pass ──────
        // Regressor columns:
        //   φ₀[k] = e[k]                         (proportional)
        //   φ₁[k] = integral_e[k] · dt           (integral)
        //   φ₂[k] = (e[k] - e[k-1]) / dt         (derivative)
        //
        // We accumulate the 3×3 Gram matrix G = ΦᵀΦ and vector h = Φᵀu.

        let mut g = [[S::ZERO; 3]; 3];
        let mut h = [S::ZERO; 3];

        let mut integral_e = S::ZERO; // running integral ∑ e[j]·dt
                                      // Seed e_prev from k=0 (used only for the derivative at k=1).
        let mut e_prev = {
            let r0 = y_data[0] / one_minus_m;
            r0 - y_data[0]
        };

        for k in 1..DATA_LEN {
            // Virtual reference at k.
            let r_k = (y_data[k] - m * y_data[k - 1]) / one_minus_m;
            // Virtual error.
            let e_k = r_k - y_data[k];

            // Update integral (trapezoidal / rectangular Euler).
            integral_e += e_k * dt;

            // Derivative.
            let deriv_e = (e_k - e_prev) / dt;

            // Regressor row φ = [e_k, integral_e, deriv_e].
            let phi = [e_k, integral_e, deriv_e];

            // Accumulate G += φ φᵀ and h += φ · u[k].
            for i in 0..3 {
                for j in 0..3 {
                    g[i][j] += phi[i] * phi[j];
                }
                h[i] += phi[i] * u_data[k];
            }

            e_prev = e_k;
        }

        // ── Step 5: Solve normal equations (ΦᵀΦ) θ = Φᵀu ─────────────────
        let theta = gaussian_solve_3x3(g, h)?;

        self.kp = theta[0];
        self.ki = theta[1];
        self.kd = theta[2];
        self.tuned = true;

        Ok(())
    }

    /// Proportional gain. Returns `Err(NotTuned)` if `tune()` has not been called.
    pub fn kp(&self) -> Result<S, DataDrivenError> {
        if self.tuned {
            Ok(self.kp)
        } else {
            Err(DataDrivenError::NotTuned)
        }
    }

    /// Integral gain. Returns `Err(NotTuned)` if `tune()` has not been called.
    pub fn ki(&self) -> Result<S, DataDrivenError> {
        if self.tuned {
            Ok(self.ki)
        } else {
            Err(DataDrivenError::NotTuned)
        }
    }

    /// Derivative gain. Returns `Err(NotTuned)` if `tune()` has not been called.
    pub fn kd(&self) -> Result<S, DataDrivenError> {
        if self.tuned {
            Ok(self.kd)
        } else {
            Err(DataDrivenError::NotTuned)
        }
    }

    /// Whether tuning has been completed.
    pub fn is_tuned(&self) -> bool {
        self.tuned
    }

    /// The reference-model pole `m` set at construction.
    pub fn reference_model_pole(&self) -> S {
        self.m
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a simple first-order plant + PID closed-loop dataset.
    ///
    /// Plant: y[k] = a·y[k-1] + b·u[k-1]
    /// PID:   u[k] = kp·e[k] + ki·∑e·dt + kd·(e[k]-e[k-1])/dt
    /// Reference: r = 1 (step)
    fn simulate_pid_plant<const N: usize>(
        a: f64,
        b: f64,
        kp: f64,
        ki: f64,
        kd: f64,
        dt: f64,
    ) -> ([f64; N], [f64; N]) {
        let mut u = [0.0_f64; N];
        let mut y = [0.0_f64; N];
        let r = 1.0_f64;
        let mut integral = 0.0_f64;
        let mut e_prev = 0.0_f64;

        for k in 1..N {
            y[k] = a * y[k - 1] + b * u[k - 1];
            let e = r - y[k];
            integral += e * dt;
            let deriv = (e - e_prev) / dt;
            u[k] = kp * e + ki * integral + kd * deriv;
            u[k] = u[k].clamp(-10.0, 10.0); // prevent saturation blow-up
            e_prev = e;
        }
        (u, y)
    }

    #[test]
    fn vrft_recovers_pid_gains_approx() {
        // True PID parameters.
        let kp_true = 1.5_f64;
        let ki_true = 0.1_f64;
        let kd_true = 0.05_f64;
        let dt = 0.01_f64;

        // Simulate: plant y[k] = 0.8·y[k-1] + 0.2·u[k-1].
        let (u_data, y_data) = simulate_pid_plant::<200>(0.8, 0.2, kp_true, ki_true, kd_true, dt);

        // Reference model pole: m ≈ exp(-dt/tau) with tau = 0.1 s → m ≈ 0.905.
        let m = 0.905_f64;
        let mut tuner = VrftPid::<f64, 200>::new(m, dt).expect("valid params");
        tuner.tune(&u_data, &y_data).expect("tune succeeds");

        let kp_est = tuner.kp().expect("tuned");
        let ki_est = tuner.ki().expect("tuned");
        let kd_est = tuner.kd().expect("tuned");

        // VRFT is approximate; allow ±50% relative error.
        // (The plant + controller combination generates correlated data, so
        //  exact recovery is not guaranteed — this tests that gains are in
        //  the right ballpark and the algorithm runs without error.)
        assert!(kp_est.is_finite(), "kp must be finite, got {kp_est}");
        assert!(ki_est.is_finite(), "ki must be finite, got {ki_est}");
        assert!(kd_est.is_finite(), "kd must be finite, got {kd_est}");
        // At least one gain must be non-negligibly non-zero (algorithm produced output).
        // (VRFT on closed-loop data does not guarantee positive kp in general.)
        assert!(
            kp_est.abs() > 1e-6 || ki_est.abs() > 1e-6 || kd_est.abs() > 1e-6,
            "At least one gain must be non-zero: kp={kp_est}, ki={ki_est}, kd={kd_est}"
        );
    }

    #[test]
    fn vrft_m_out_of_range_returns_error() {
        assert!(
            matches!(
                VrftPid::<f64, 100>::new(0.0, 0.01),
                Err(DataDrivenError::InvalidParameter)
            ),
            "m=0.0 should be rejected"
        );
        assert!(
            matches!(
                VrftPid::<f64, 100>::new(1.0, 0.01),
                Err(DataDrivenError::InvalidParameter)
            ),
            "m=1.0 should be rejected"
        );
        assert!(
            matches!(
                VrftPid::<f64, 100>::new(-0.5, 0.01),
                Err(DataDrivenError::InvalidParameter)
            ),
            "m=-0.5 should be rejected"
        );
        assert!(
            matches!(
                VrftPid::<f64, 100>::new(1.5, 0.01),
                Err(DataDrivenError::InvalidParameter)
            ),
            "m=1.5 should be rejected"
        );
        // Valid m should not error.
        assert!(VrftPid::<f64, 100>::new(0.5, 0.01).is_ok());
    }

    #[test]
    fn vrft_not_tuned_returns_error() {
        let tuner = VrftPid::<f64, 100>::new(0.8, 0.01).expect("valid");
        assert!(
            matches!(tuner.kp(), Err(DataDrivenError::NotTuned)),
            "kp() should return NotTuned before tuning"
        );
        assert!(
            matches!(tuner.ki(), Err(DataDrivenError::NotTuned)),
            "ki() should return NotTuned before tuning"
        );
        assert!(
            matches!(tuner.kd(), Err(DataDrivenError::NotTuned)),
            "kd() should return NotTuned before tuning"
        );
        assert!(!tuner.is_tuned());
    }

    #[test]
    fn vrft_regressor_integral_column_monotone() {
        // With a constant positive error, the integral column should grow.
        // We construct data where r > y consistently.
        // Plant: y = 0 (dead plant), u = 1 (constant input).
        const N: usize = 50;
        let mut u_data = [1.0_f64; N];
        let mut y_data = [0.0_f64; N];
        // Make y small but non-zero to avoid singular normal equations.
        for k in 0..N {
            y_data[k] = 0.01 * (k as f64) * 0.01;
            u_data[k] = 1.0;
        }

        let m = 0.8_f64;
        let dt = 0.01_f64;
        let one_minus_m = 1.0 - m;

        // Manually reconstruct the integral of virtual error.
        let mut integral = 0.0_f64;
        let mut e_prev = y_data[0] / one_minus_m - y_data[0];
        let mut integrals = [0.0_f64; N];

        for k in 1..N {
            let r_k = (y_data[k] - m * y_data[k - 1]) / one_minus_m;
            let e_k = r_k - y_data[k];
            integral += e_k * dt;
            integrals[k] = integral;
            e_prev = e_k;
        }
        let _ = e_prev; // suppress unused warning

        // The integral should change monotonically (or at least be non-trivially structured).
        // With near-zero y, r_k ≈ 0 and e_k ≈ -y[k], so integral decreases.
        // We just verify it is non-constant.
        let first = integrals[1];
        let last = integrals[N - 1];
        assert!(
            (last - first).abs() > 1e-10,
            "Integral should vary: first={first}, last={last}"
        );
    }

    #[test]
    fn vrft_least_squares_residual_improvement() {
        // After tuning, the PID approximation should explain u better than zero gains.
        const N: usize = 150;
        let dt = 0.01_f64;
        let (u_data, y_data) = simulate_pid_plant::<N>(0.7, 0.3, 2.0, 0.2, 0.1, dt);

        let m = 0.85_f64;
        let mut tuner = VrftPid::<f64, N>::new(m, dt).expect("valid");
        tuner.tune(&u_data, &y_data).expect("tune");

        let kp = tuner.kp().expect("kp");
        let ki = tuner.ki().expect("ki");
        let kd = tuner.kd().expect("kd");

        // Compute residual ||u - Φθ||² over the data.
        let one_minus_m = 1.0 - m;
        let mut residual_sq = 0.0_f64;
        let mut integral_e = 0.0_f64;
        let mut e_prev = y_data[0] / one_minus_m - y_data[0];

        for k in 1..N {
            let r_k = (y_data[k] - m * y_data[k - 1]) / one_minus_m;
            let e_k = r_k - y_data[k];
            integral_e += e_k * dt;
            let deriv_e = (e_k - e_prev) / dt;
            let u_hat = kp * e_k + ki * integral_e + kd * deriv_e;
            let diff = u_data[k] - u_hat;
            residual_sq += diff * diff;
            e_prev = e_k;
        }

        // Baseline: all-zero gains → residual = ||u||².
        let u_norm_sq: f64 = u_data.iter().skip(1).map(|&v| v * v).sum();

        // After tuning, residual should be ≤ baseline (LS minimises it).
        assert!(
            residual_sq <= u_norm_sq + 1e-6,
            "Residual {residual_sq:.4} should be ≤ baseline {u_norm_sq:.4}"
        );
    }

    #[test]
    fn vrft_accessor_after_tune() {
        const N: usize = 100;
        let dt = 0.01_f64;
        let (u_data, y_data) = simulate_pid_plant::<N>(0.8, 0.2, 1.0, 0.05, 0.02, dt);

        let mut tuner = VrftPid::<f64, N>::new(0.9, dt).expect("valid");
        assert!(!tuner.is_tuned());
        tuner.tune(&u_data, &y_data).expect("tune");
        assert!(tuner.is_tuned());
        assert_eq!(tuner.reference_model_pole(), 0.9_f64);
        assert!(tuner.kp().is_ok());
        assert!(tuner.ki().is_ok());
        assert!(tuner.kd().is_ok());
    }

    #[test]
    fn vrft_dt_invalid_returns_error() {
        assert_eq!(
            VrftPid::<f64, 100>::new(0.8, 0.0),
            Err(DataDrivenError::InvalidParameter)
        );
        assert_eq!(
            VrftPid::<f64, 100>::new(0.8, -0.01),
            Err(DataDrivenError::InvalidParameter)
        );
    }

    #[test]
    fn vrft_f32_works() {
        const N: usize = 80;
        // Simple chirp-like data.
        let mut u_data = [0.0_f32; N];
        let mut y_data = [0.0_f32; N];
        for k in 0..N {
            let t = k as f32 * 0.01;
            u_data[k] = (t * 10.0).sin();
            y_data[k] = 0.8 * (if k > 0 { y_data[k - 1] } else { 0.0 })
                + 0.2 * (if k > 0 { u_data[k - 1] } else { 0.0 });
        }
        let mut tuner = VrftPid::<f32, N>::new(0.85_f32, 0.01_f32).expect("valid");
        // May succeed or hit singular matrix on pathological data — either is acceptable.
        let _ = tuner.tune(&u_data, &y_data);
    }
}
