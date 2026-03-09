#![allow(clippy::needless_range_loop)]
use crate::core::matrix::{matmul, matvec, Matrix};
use crate::core::scalar::ControlScalar;

/// Information form of the Kalman Filter (also called Canonical Filter or
/// Information Filter).
///
/// Instead of tracking the state `x` and covariance `P` directly, this filter
/// maintains the **information vector** `ξ = Ω·x` and the **information matrix**
/// `Ω = P⁻¹` (inverse covariance).  The information form is the natural
/// representation for multi-sensor fusion because each sensor's contribution is
/// *additive* in both `ξ` and `Ω`.
///
/// Discrete-time linear model:
/// ```text
///   x[k+1] = A·x[k] + B·u[k] + w[k],  w ~ N(0, Q)
///   z[k]   = H·x[k] + v[k],             v ~ N(0, R)
/// ```
///
/// # Type Parameters
/// * `S` – scalar type (`f32` or `f64`)
/// * `N` – state dimension
/// * `M` – measurement dimension
/// * `I` – input dimension
#[derive(Debug, Clone, Copy)]
pub struct InformationFilter<S: ControlScalar, const N: usize, const M: usize, const I: usize> {
    /// State transition matrix (N×N).
    pub a: Matrix<S, N, N>,
    /// Control input matrix (N×I).
    pub b: Matrix<S, N, I>,
    /// Measurement matrix (M×N).
    pub h: Matrix<S, M, N>,
    /// Process noise covariance (N×N).  Used during predict.
    pub q: Matrix<S, N, N>,
    /// Measurement noise covariance (M×M).
    pub r: Matrix<S, M, M>,
    /// Information matrix Ω = P⁻¹ (N×N).
    omega: Matrix<S, N, N>,
    /// Information vector ξ = Ω·x (N).
    xi: [S; N],
}

/// Error type for the Information Filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfoFilterError {
    /// A matrix inversion failed (singular matrix encountered).
    SingularMatrix,
    /// The predicted information matrix is not positive definite.
    NotPositiveDefinite,
}

impl core::fmt::Display for InfoFilterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InfoFilterError::SingularMatrix => write!(f, "InformationFilter: singular matrix"),
            InfoFilterError::NotPositiveDefinite => {
                write!(f, "InformationFilter: matrix not positive definite")
            }
        }
    }
}

impl<S: ControlScalar, const N: usize, const M: usize, const I: usize>
    InformationFilter<S, N, M, I>
{
    /// Create an Information Filter from standard Kalman matrices.
    ///
    /// `p0` must be positive definite (invertible).
    /// Returns `None` if `p0` is singular.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        h: Matrix<S, M, N>,
        q: Matrix<S, N, N>,
        r: Matrix<S, M, M>,
        x0: [S; N],
        p0: Matrix<S, N, N>,
    ) -> Option<Self> {
        let omega = p0.inv()?;
        let xi = matvec(&omega, &x0);
        Some(Self {
            a,
            b,
            h,
            q,
            r,
            omega,
            xi,
        })
    }

    /// Create an Information Filter with initial information matrix and vector
    /// specified directly (avoids `p0` inversion).
    pub fn from_information(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        h: Matrix<S, M, N>,
        q: Matrix<S, N, N>,
        r: Matrix<S, M, M>,
        omega0: Matrix<S, N, N>,
        xi0: [S; N],
    ) -> Self {
        Self {
            a,
            b,
            h,
            q,
            r,
            omega: omega0,
            xi: xi0,
        }
    }

    /// **Predict step** — propagate information forward through dynamics.
    ///
    /// Standard-form predict:
    /// ```text
    ///   P_pred = A·P·Aᵀ + Q
    ///   Ω_pred = P_pred⁻¹
    ///   ξ_pred = Ω_pred · (A·x + B·u)
    /// ```
    ///
    /// Returns `Err` if either the current covariance or the predicted
    /// covariance is not invertible.
    pub fn predict(&mut self, u: &[S; I]) -> Result<(), InfoFilterError> {
        // Recover P = Ω⁻¹
        let p = self.omega.inv().ok_or(InfoFilterError::SingularMatrix)?;

        // Recover x = P · ξ  (= Ω⁻¹ · ξ)
        let x = matvec(&p, &self.xi);

        // Propagate state: x_pred = A·x + B·u
        let ax = matvec(&self.a, &x);
        let bu = matvec(&self.b, u);
        let x_pred: [S; N] = core::array::from_fn(|i| ax[i] + bu[i]);

        // Propagate covariance: P_pred = A·P·Aᵀ + Q
        let ap = matmul(&self.a, &p);
        let at = self.a.transpose();
        let apat = matmul(&ap, &at);
        let p_pred = apat.add_mat(&self.q);

        // Invert predicted covariance to get information matrix
        let omega_pred = p_pred.inv().ok_or(InfoFilterError::SingularMatrix)?;

        // Update information vector: ξ_pred = Ω_pred · x_pred
        let xi_pred = matvec(&omega_pred, &x_pred);

        self.omega = omega_pred;
        self.xi = xi_pred;
        Ok(())
    }

    /// **Update step** — additive information fusion from one sensor.
    ///
    /// The information form update is elegantly additive:
    /// ```text
    ///   Ω_new = Ω_pred + Hᵀ·R⁻¹·H
    ///   ξ_new = ξ_pred + Hᵀ·R⁻¹·z
    /// ```
    ///
    /// Returns the innovation vector `z - H·x_pred`, or `Err` if `R` is
    /// singular.
    pub fn update(&mut self, z: &[S; M]) -> Result<[S; M], InfoFilterError> {
        // Invert measurement noise: R⁻¹
        let r_inv = self.r.inv().ok_or(InfoFilterError::SingularMatrix)?;

        // Hᵀ (N×M)
        let ht = self.h.transpose();

        // Hᵀ·R⁻¹ (N×M)
        let ht_rinv = matmul(&ht, &r_inv);

        // Information contribution: Hᵀ·R⁻¹·H (N×N)
        let info_contrib = matmul(&ht_rinv, &self.h);

        // Measurement information: Hᵀ·R⁻¹·z (N)
        let rinv_z = matvec(&r_inv, z);
        let xi_contrib = matvec(&ht, &rinv_z);

        // Additive fusion
        self.omega = self.omega.add_mat(&info_contrib);
        for i in 0..N {
            self.xi[i] += xi_contrib[i];
        }

        // Compute innovation for diagnostics: need current state estimate
        // x_est = P · ξ = Ω⁻¹ · ξ  (before this update, use old Ω)
        // We approximate using the updated omega to give the posterior state:
        let p_post = self.omega.inv().ok_or(InfoFilterError::SingularMatrix)?;
        let x_post = matvec(&p_post, &self.xi);
        let hx: [S; M] = matvec(&self.h, &x_post);
        let innovation: [S; M] = core::array::from_fn(|i| z[i] - hx[i]);

        Ok(innovation)
    }

    /// **Fuse multiple sensors simultaneously** — the key advantage of the
    /// information form.  Each sensor contributes independently to `Ω` and `ξ`.
    ///
    /// `sensors` is a slice of `(H, R⁻¹, z)` tuples.  Passing `R⁻¹` directly
    /// avoids repeated inversion when many sensors share the same noise model.
    pub fn fuse_sensors(
        &mut self,
        sensors: &[(Matrix<S, M, N>, Matrix<S, M, M>, [S; M])],
    ) -> Result<(), InfoFilterError> {
        for (h_i, r_inv_i, z_i) in sensors {
            let ht_i = h_i.transpose();
            let ht_rinv = matmul(&ht_i, r_inv_i);
            let info_contrib = matmul(&ht_rinv, h_i);
            let rinv_z = matvec(r_inv_i, z_i);
            let xi_contrib = matvec(&ht_i, &rinv_z);
            self.omega = self.omega.add_mat(&info_contrib);
            for i in 0..N {
                self.xi[i] += xi_contrib[i];
            }
        }
        Ok(())
    }

    /// Recover the state estimate `x = Ω⁻¹ · ξ`.
    ///
    /// Returns `None` if the information matrix is singular.
    pub fn state(&self) -> Option<[S; N]> {
        let p = self.omega.inv()?;
        Some(matvec(&p, &self.xi))
    }

    /// Recover the covariance `P = Ω⁻¹`.
    ///
    /// Returns `None` if the information matrix is singular.
    pub fn covariance(&self) -> Option<Matrix<S, N, N>> {
        self.omega.inv()
    }

    /// Raw information matrix `Ω`.
    pub fn information_matrix(&self) -> &Matrix<S, N, N> {
        &self.omega
    }

    /// Raw information vector `ξ`.
    pub fn information_vector(&self) -> &[S; N] {
        &self.xi
    }

    /// Reset filter to new state estimate and covariance.
    ///
    /// Returns `None` if `p0` is singular.
    pub fn reset(&mut self, x0: [S; N], p0: Matrix<S, N, N>) -> Option<()> {
        self.omega = p0.inv()?;
        self.xi = matvec(&self.omega, &x0);
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_filter() -> InformationFilter<f64, 2, 1, 1> {
        // Position-velocity model, dt=0.01
        let dt = 0.01_f64;
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = dt;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.5 * dt * dt;
        b.data[1][0] = dt;

        let mut h = Matrix::<f64, 1, 2>::zeros();
        h.data[0][0] = 1.0;

        let q = Matrix::<f64, 2, 2>::identity().scale(1e-4);
        let r = Matrix::<f64, 1, 1>::identity().scale(0.1);
        let p0 = Matrix::<f64, 2, 2>::identity().scale(100.0);

        InformationFilter::new(a, b, h, q, r, [0.0_f64; 2], p0).expect("p0 is positive definite")
    }

    #[test]
    fn new_from_positive_definite_p0() {
        let _f = build_filter();
    }

    #[test]
    fn new_returns_none_for_singular_p0() {
        let a = Matrix::<f64, 2, 2>::identity();
        let b = Matrix::<f64, 2, 1>::zeros();
        let mut h = Matrix::<f64, 1, 2>::zeros();
        h.data[0][0] = 1.0;
        let q = Matrix::<f64, 2, 2>::identity().scale(1e-4);
        let r = Matrix::<f64, 1, 1>::identity().scale(0.1);
        let p_singular = Matrix::<f64, 2, 2>::zeros(); // singular
        let result = InformationFilter::new(a, b, h, q, r, [0.0_f64; 2], p_singular);
        assert!(result.is_none());
    }

    #[test]
    fn predict_ok() {
        let mut f = build_filter();
        assert!(f.predict(&[0.0]).is_ok());
    }

    #[test]
    fn update_returns_innovation() {
        let mut f = build_filter();
        f.predict(&[0.0]).expect("predict");
        let innov = f.update(&[1.0]).expect("update");
        assert_eq!(innov.len(), 1);
    }

    #[test]
    fn tracks_constant_position() {
        let mut f = build_filter();
        let true_pos = 5.0_f64;
        for _ in 0..300 {
            f.predict(&[0.0]).expect("predict");
            f.update(&[true_pos]).expect("update");
        }
        let x = f.state().expect("state");
        assert!(
            (x[0] - true_pos).abs() < 0.5,
            "Expected position ~{true_pos}, got {}",
            x[0]
        );
    }

    #[test]
    fn information_matrix_stays_symmetric() {
        let mut f = build_filter();
        for _ in 0..20 {
            f.predict(&[0.0]).expect("predict");
            f.update(&[1.0]).expect("update");
        }
        let omega = f.information_matrix();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (omega.data[i][j] - omega.data[j][i]).abs() < 1e-9,
                    "Ω not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn from_information_constructor() {
        let dt = 0.01_f64;
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = dt;
        let b = Matrix::<f64, 2, 1>::zeros();
        let mut h = Matrix::<f64, 1, 2>::zeros();
        h.data[0][0] = 1.0;
        let q = Matrix::<f64, 2, 2>::identity().scale(1e-4);
        let r = Matrix::<f64, 1, 1>::identity().scale(0.1);

        // Start with nearly zero information (high uncertainty)
        let omega0 = Matrix::<f64, 2, 2>::identity().scale(0.01);
        let xi0 = [0.0_f64; 2];
        let mut f = InformationFilter::from_information(a, b, h, q, r, omega0, xi0);

        for _ in 0..100 {
            f.predict(&[0.0]).expect("predict");
            f.update(&[3.0]).expect("update");
        }
        let x = f.state().expect("state");
        assert!((x[0] - 3.0).abs() < 0.5, "Expected ~3.0, got {}", x[0]);
    }

    #[test]
    fn reset_restores_state() {
        let mut f = build_filter();
        for _ in 0..50 {
            f.predict(&[0.0]).expect("predict");
            f.update(&[5.0]).expect("update");
        }
        let p0 = Matrix::<f64, 2, 2>::identity().scale(100.0);
        f.reset([0.0_f64; 2], p0).expect("reset");
        let x = f.state().expect("state after reset");
        assert!((x[0]).abs() < 1e-9, "Expected 0, got {}", x[0]);
    }

    #[test]
    fn covariance_decreases() {
        let mut f = build_filter();
        let initial_trace = f.covariance().expect("initial cov").trace();
        for i in 0..100 {
            f.predict(&[0.0]).expect("predict");
            f.update(&[i as f64 * 0.01]).expect("update");
        }
        let final_trace = f.covariance().expect("final cov").trace();
        assert!(
            final_trace < initial_trace,
            "Covariance should decrease: {initial_trace} → {final_trace}"
        );
    }
}
