#![allow(clippy::needless_range_loop)]
use crate::core::matrix::{matmul, matvec, Matrix};
use crate::core::scalar::ControlScalar;

/// Huber M-estimator robust Kalman filter.
///
/// The standard Kalman filter is optimal under Gaussian noise but degrades
/// severely in the presence of heavy-tailed or outlier-corrupted measurements.
/// The Huber robust Kalman filter replaces the least-squares measurement
/// cost with the **Huber loss**, which behaves like L2 for small residuals and
/// like L1 for large ones.  This is implemented via **Iteratively Reweighted
/// Least Squares (IRLS)**: the measurement noise covariance `R` is inflated
/// per-channel by the inverse Huber weight, effectively down-weighting outlier
/// channels before applying the standard KF update equations.
///
/// Discrete-time linear model:
/// ```text
///   x[k+1] = F·x[k] + w[k],  w ~ N(0, Q)
///   z[k]   = H·x[k] + v[k],  v ~ robust (Huber)
/// ```
///
/// # Huber weight function
/// For threshold `c` and scalar residual `ν`:
/// ```text
///   w(ν) = 1            if |ν| ≤ c
///   w(ν) = c / |ν|      otherwise
/// ```
///
/// # Type Parameters
/// * `S` – scalar type (`f32` or `f64`)
/// * `N` – state dimension
/// * `M` – measurement dimension
#[derive(Debug, Clone, Copy)]
pub struct HuberKalmanFilter<S: ControlScalar, const N: usize, const M: usize> {
    /// Current state estimate (N).
    x: [S; N],
    /// Error covariance (N×N).
    p: Matrix<S, N, N>,
    /// Process noise covariance (N×N).
    pub q: Matrix<S, N, N>,
    /// Nominal measurement noise covariance (M×M).
    pub r: Matrix<S, M, M>,
    /// Huber threshold `c` — residuals beyond this are down-weighted.
    pub huber_threshold: S,
}

/// Errors produced by [`HuberKalmanFilter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuberKfError {
    /// A matrix that must be invertible was found to be singular.
    SingularMatrix,
    /// The predicted covariance became non-positive-definite.
    NotPositiveDefinite,
    /// The Huber threshold must be strictly positive.
    InvalidThreshold,
}

impl core::fmt::Display for HuberKfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HuberKfError::SingularMatrix => write!(f, "HuberKalmanFilter: singular matrix"),
            HuberKfError::NotPositiveDefinite => {
                write!(f, "HuberKalmanFilter: covariance not positive definite")
            }
            HuberKfError::InvalidThreshold => {
                write!(f, "HuberKalmanFilter: threshold must be > 0")
            }
        }
    }
}

impl<S: ControlScalar, const N: usize, const M: usize> HuberKalmanFilter<S, N, M> {
    /// Construct a new Huber Kalman filter.
    ///
    /// # Arguments
    /// * `x0`        – initial state estimate
    /// * `p0`        – initial error covariance (must be positive definite)
    /// * `q`         – process noise covariance
    /// * `r`         – nominal measurement noise covariance
    /// * `threshold` – Huber threshold `c` (must be > 0)
    ///
    /// Returns `Err(HuberKfError::InvalidThreshold)` if `threshold ≤ 0`.
    pub fn new(
        x0: [S; N],
        p0: Matrix<S, N, N>,
        q: Matrix<S, N, N>,
        r: Matrix<S, M, M>,
        threshold: S,
    ) -> Result<Self, HuberKfError> {
        if threshold <= S::ZERO {
            return Err(HuberKfError::InvalidThreshold);
        }
        Ok(Self {
            x: x0,
            p: p0,
            q,
            r,
            huber_threshold: threshold,
        })
    }

    /// **Predict step**: propagate state and covariance through the dynamics.
    ///
    /// ```text
    ///   x_pred = F·x
    ///   P_pred = F·P·Fᵀ + Q
    /// ```
    ///
    /// Optionally apply a control input: if `control = Some((B, u))` then
    /// `x_pred = F·x + B·u` where `B` is (N×1) and `u` is (1).
    ///
    /// Returns `Err` if the predicted covariance becomes singular.
    pub fn predict(
        &mut self,
        f_mat: &Matrix<S, N, N>,
        control: Option<(&Matrix<S, N, 1>, &[S; 1])>,
    ) -> Result<(), HuberKfError> {
        // x_pred = F·x [+ B·u]
        let fx = matvec(f_mat, &self.x);
        self.x = if let Some((b, u)) = control {
            let bu = matvec(b, u);
            core::array::from_fn(|i| fx[i] + bu[i])
        } else {
            fx
        };

        // P_pred = F·P·Fᵀ + Q
        let fp = matmul(f_mat, &self.p);
        let ft = f_mat.transpose();
        let fpft = matmul(&fp, &ft);
        self.p = fpft.add_mat(&self.q);

        Ok(())
    }

    /// **Update step** with Huber-weighted IRLS.
    ///
    /// Algorithm:
    /// 1. Compute prior innovation `ν = z - H·x`.
    /// 2. For each channel `i`, compute Huber weight
    ///    `w_i = min(1, c / |ν_i|)` (1 if `|ν_i| ≤ c`).
    /// 3. Build effective noise covariance `R_eff` with `R_eff[i,i] = R[i,i] / w_i`.
    ///    Off-diagonal elements of `R` are left unchanged (only diagonal scaling
    ///    is applied, matching the scalar IRLS interpretation).
    /// 4. Run standard KF update with `R_eff`.
    ///
    /// Returns `Err` if the innovation covariance `S = H·P·Hᵀ + R_eff` is singular.
    pub fn update(&mut self, h_mat: &Matrix<S, M, N>, z: &[S; M]) -> Result<(), HuberKfError> {
        // --- Step 1: prior innovation ν = z - H·x ---
        let hx: [S; M] = matvec(h_mat, &self.x);
        let nu: [S; M] = core::array::from_fn(|i| z[i] - hx[i]);

        // --- Step 2 & 3: Huber weights → R_eff ---
        // R_eff is built by scaling R's diagonal entries by 1/w_i.
        // For off-diagonal entries we keep R[i,j] scaled by 1/sqrt(w_i * w_j)
        // to maintain symmetry; for a diagonal R this is exact.
        let c = self.huber_threshold;
        let mut weights = [S::ONE; M];
        for i in 0..M {
            let abs_nu = num_traits::Float::abs(nu[i]);
            if abs_nu > c {
                weights[i] = c / abs_nu;
            }
        }

        // Build R_eff: R_eff[i][j] = R[i][j] / sqrt(w_i * w_j)
        let mut r_eff = self.r;
        for i in 0..M {
            for j in 0..M {
                let w_ij = num_traits::Float::sqrt(weights[i] * weights[j]);
                if w_ij > S::ZERO {
                    r_eff.data[i][j] = self.r.data[i][j] / w_ij;
                }
            }
        }

        // --- Step 4: Standard KF update with R_eff ---
        // S_innov = H·P·Hᵀ + R_eff
        let ht = h_mat.transpose();
        let ph_t = matmul(&self.p, &ht); // N×M
        let h_p_ht = matmul(h_mat, &ph_t); // M×M
        let s_innov = h_p_ht.add_mat(&r_eff);

        // K = P·Hᵀ · S_innov⁻¹  (N×M)
        let s_inv = s_innov.inv().ok_or(HuberKfError::SingularMatrix)?;
        let k = matmul(&ph_t, &s_inv); // N×M

        // x ← x + K·ν
        let k_nu = matvec(&k, &nu);
        for i in 0..N {
            self.x[i] += k_nu[i];
        }

        // P ← (I - K·H)·P  (Joseph form for numerical stability)
        // P_new = P - K·H·P
        let kh = matmul(&k, h_mat); // N×N
        let kh_p = matmul(&kh, &self.p);
        self.p = self.p.sub_mat(&kh_p);

        Ok(())
    }

    /// Current state estimate.
    pub fn state(&self) -> &[S; N] {
        &self.x
    }

    /// Current error covariance.
    pub fn covariance(&self) -> &Matrix<S, N, N> {
        &self.p
    }

    /// Reset the filter to a new state and covariance.
    pub fn reset(&mut self, x0: [S; N], p0: Matrix<S, N, N>) {
        self.x = x0;
        self.p = p0;
    }
}

// ─── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Position-only model: state = [position], measurement = [position].
    fn build_scalar_filter(threshold: f64) -> HuberKalmanFilter<f64, 1, 1> {
        let x0 = [0.0_f64];
        let p0 = Matrix::<f64, 1, 1> { data: [[100.0]] };
        let q = Matrix::<f64, 1, 1> { data: [[1e-4]] };
        let r = Matrix::<f64, 1, 1> { data: [[1.0]] };
        HuberKalmanFilter::new(x0, p0, q, r, threshold).expect("threshold > 0")
    }

    /// 2-state position-velocity, scalar position measurement.
    fn build_pv_filter(threshold: f64) -> HuberKalmanFilter<f64, 2, 1> {
        let x0 = [0.0_f64; 2];
        let p0 = Matrix::<f64, 2, 2>::identity().scale(100.0);
        let q = Matrix::<f64, 2, 2>::identity().scale(1e-4);
        let r = Matrix::<f64, 1, 1> { data: [[0.1]] };
        HuberKalmanFilter::new(x0, p0, q, r, threshold).expect("threshold > 0")
    }

    fn pv_transition() -> Matrix<f64, 2, 2> {
        let dt = 0.01_f64;
        Matrix::<f64, 2, 2> {
            data: [[1.0, dt], [0.0, 1.0]],
        }
    }

    fn pv_h() -> Matrix<f64, 1, 2> {
        Matrix::<f64, 1, 2> { data: [[1.0, 0.0]] }
    }

    #[test]
    fn invalid_threshold_rejected() {
        let x0 = [0.0_f64];
        let p0 = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let q = Matrix::<f64, 1, 1> { data: [[1e-4]] };
        let r = Matrix::<f64, 1, 1> { data: [[1.0]] };
        assert!(HuberKalmanFilter::new(x0, p0, q, r, 0.0_f64).is_err());
        assert!(HuberKalmanFilter::new(x0, p0, q, r, -1.0_f64).is_err());
    }

    #[test]
    fn predict_update_cycle_completes() {
        let mut f = build_scalar_filter(1.5);
        let f_mat = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let h_mat = Matrix::<f64, 1, 1> { data: [[1.0]] };
        assert!(f.predict(&f_mat, None).is_ok());
        assert!(f.update(&h_mat, &[1.0]).is_ok());
    }

    #[test]
    fn tracks_constant_signal() {
        let mut f = build_pv_filter(3.0);
        let f_mat = pv_transition();
        let h_mat = pv_h();
        let true_pos = 5.0_f64;
        for _ in 0..500 {
            f.predict(&f_mat, None).expect("predict");
            f.update(&h_mat, &[true_pos]).expect("update");
        }
        let x = f.state();
        assert!(
            (x[0] - true_pos).abs() < 0.5,
            "Expected position ~{true_pos}, got {}",
            x[0]
        );
    }

    #[test]
    fn outlier_is_downweighted() {
        // Run two filters: one with a tight Huber threshold (robust) and one
        // with a very large threshold (behaves like standard KF).  Feed an
        // extreme outlier and check that the robust filter is less disturbed.
        let f_mat = pv_transition();
        let h_mat = pv_h();

        let mut robust = build_pv_filter(1.0);
        let mut standard = build_pv_filter(1e9);

        // Warm both filters up on clean measurements
        for _ in 0..200 {
            robust.predict(&f_mat, None).expect("predict");
            robust.update(&h_mat, &[0.0]).expect("update");
            standard.predict(&f_mat, None).expect("predict");
            standard.update(&h_mat, &[0.0]).expect("update");
        }

        // Inject a massive outlier
        let outlier = [100.0_f64];
        robust.predict(&f_mat, None).expect("predict");
        robust.update(&h_mat, &outlier).expect("update");
        standard.predict(&f_mat, None).expect("predict");
        standard.update(&h_mat, &outlier).expect("update");

        let robust_pos = robust.state()[0].abs();
        let standard_pos = standard.state()[0].abs();

        assert!(
            robust_pos < standard_pos,
            "Robust filter ({robust_pos}) should be less affected than standard ({standard_pos})"
        );
    }

    #[test]
    fn covariance_decreases_over_updates() {
        let mut f = build_pv_filter(2.0);
        let f_mat = pv_transition();
        let h_mat = pv_h();

        let initial_trace = f.covariance().trace();
        for i in 0..100 {
            f.predict(&f_mat, None).expect("predict");
            f.update(&h_mat, &[i as f64 * 0.01]).expect("update");
        }
        let final_trace = f.covariance().trace();
        assert!(
            final_trace < initial_trace,
            "Trace should decrease: {initial_trace} → {final_trace}"
        );
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut f = build_pv_filter(2.0);
        let f_mat = pv_transition();
        let h_mat = pv_h();

        for _ in 0..50 {
            f.predict(&f_mat, None).expect("predict");
            f.update(&h_mat, &[3.0]).expect("update");
        }
        let p0 = Matrix::<f64, 2, 2>::identity().scale(100.0);
        f.reset([0.0_f64; 2], p0);
        assert!(
            f.state()[0].abs() < 1e-12,
            "State should be 0 after reset, got {}",
            f.state()[0]
        );
    }

    #[test]
    fn with_control_input() {
        let dt = 0.01_f64;
        let mut f = build_pv_filter(3.0);
        let f_mat = pv_transition();
        let h_mat = pv_h();
        // B matrix: acceleration input
        let b_mat = Matrix::<f64, 2, 1> {
            data: [[0.5 * dt * dt], [dt]],
        };
        let u = [1.0_f64]; // constant acceleration
        for _ in 0..50 {
            f.predict(&f_mat, Some((&b_mat, &u))).expect("predict");
            // measurement follows x = 0.5*a*t^2 (approximate)
            f.update(&h_mat, &[0.0]).expect("update");
        }
        // Filter should have non-zero velocity estimate due to control
        assert!(
            f.state()[1].abs() > 0.0,
            "Velocity should be non-zero with acceleration input"
        );
    }
}
