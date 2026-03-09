use crate::core::matrix::{matmul, matvec, outer, Matrix};
use crate::core::scalar::ControlScalar;

/// Unscented Kalman Filter (UKF) for nonlinear state estimation.
///
/// Uses the unscented transform to propagate the mean and covariance
/// through nonlinear functions without linearization.
///
/// - N: state dimension
/// - M: measurement dimension
/// - I: input dimension
///
/// Default tuning: α=0.001, β=2, κ=0 (minimizes kurtosis error for Gaussian).
pub struct Ukf<S: ControlScalar, const N: usize, const M: usize, const I: usize> {
    /// Process noise covariance (N×N).
    pub q: Matrix<S, N, N>,
    /// Measurement noise covariance (M×M).
    pub r: Matrix<S, M, M>,
    /// State transition function: f(x, u) -> x_next.
    f: fn(&[S; N], &[S; I]) -> [S; N],
    /// Measurement function: h(x) -> z.
    h: fn(&[S; N]) -> [S; M],
    /// UKF spread parameter (typically 1e-3).
    pub alpha: S,
    /// UKF distribution parameter (typically 2 for Gaussian).
    pub beta: S,
    /// UKF secondary scaling (typically 0).
    pub kappa: S,
    /// State estimate.
    x: [S; N],
    /// Error covariance.
    p: Matrix<S, N, N>,
}

impl<S: ControlScalar, const N: usize, const M: usize, const I: usize> Ukf<S, N, M, I> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q: Matrix<S, N, N>,
        r: Matrix<S, M, M>,
        f: fn(&[S; N], &[S; I]) -> [S; N],
        h: fn(&[S; N]) -> [S; M],
        x0: [S; N],
        p0: Matrix<S, N, N>,
        alpha: S,
        beta: S,
        kappa: S,
    ) -> Self {
        Self {
            q,
            r,
            f,
            h,
            alpha,
            beta,
            kappa,
            x: x0,
            p: p0,
        }
    }

    /// Standard UKF with α=0.001, β=2, κ=0.
    pub fn standard(
        q: Matrix<S, N, N>,
        r: Matrix<S, M, M>,
        f: fn(&[S; N], &[S; I]) -> [S; N],
        h: fn(&[S; N]) -> [S; M],
        x0: [S; N],
        p0: Matrix<S, N, N>,
    ) -> Self {
        Self::new(
            q,
            r,
            f,
            h,
            x0,
            p0,
            S::from_f64(0.001),
            S::from_f64(2.0),
            S::ZERO,
        )
    }

    fn weights(&self) -> (S, S, S) {
        let n = S::from_f64(N as f64);
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let c = n + lambda;
        let w_m0 = lambda / c;
        let w_c0 = w_m0 + S::ONE - self.alpha * self.alpha + self.beta;
        let w_i = S::HALF / c;
        (w_m0, w_c0, w_i)
    }

    fn scale_factor(&self) -> S {
        let n = S::from_f64(N as f64);
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        n + lambda
    }

    /// UKF predict step. Returns false if Cholesky fails (non-PD covariance).
    pub fn predict(&mut self, u: &[S; I]) -> bool {
        let c = self.scale_factor();
        let p_scaled = self.p.scale(c);
        let l = match p_scaled.cholesky() {
            Some(l) => l,
            None => return false,
        };

        let (w_m0, w_c0, w_i) = self.weights();

        // Center sigma point propagated through f
        let y0 = (self.f)(&self.x, u);

        // ± sigma points propagated
        let mut y_plus: [[S; N]; N] = core::array::from_fn(|_| [S::ZERO; N]);
        let mut y_minus: [[S; N]; N] = core::array::from_fn(|_| [S::ZERO; N]);

        for j in 0..N {
            let mut xi_plus = self.x;
            let mut xi_minus = self.x;
            for row in 0..N {
                xi_plus[row] += l.data[row][j];
                xi_minus[row] -= l.data[row][j];
            }
            y_plus[j] = (self.f)(&xi_plus, u);
            y_minus[j] = (self.f)(&xi_minus, u);
        }

        // Predicted mean
        let mut x_pred = [S::ZERO; N];
        for k in 0..N {
            x_pred[k] = w_m0 * y0[k];
            for j in 0..N {
                x_pred[k] += w_i * (y_plus[j][k] + y_minus[j][k]);
            }
        }

        // Predicted covariance P_pred = Q + weighted outer products
        let d0: [S; N] = core::array::from_fn(|k| y0[k] - x_pred[k]);
        let mut p_pred = self.q.add_mat(&outer(&d0, &d0).scale(w_c0));
        for j in 0..N {
            let dp: [S; N] = core::array::from_fn(|k| y_plus[j][k] - x_pred[k]);
            let dm: [S; N] = core::array::from_fn(|k| y_minus[j][k] - x_pred[k]);
            p_pred = p_pred
                .add_mat(&outer(&dp, &dp).scale(w_i))
                .add_mat(&outer(&dm, &dm).scale(w_i));
        }

        self.x = x_pred;
        self.p = p_pred;
        true
    }

    /// UKF update step. Returns innovation vector, or None if S is singular.
    pub fn update(&mut self, z_meas: &[S; M]) -> Option<[S; M]> {
        let c = self.scale_factor();
        let p_scaled = self.p.scale(c);
        let l = p_scaled.cholesky()?;

        let (w_m0, w_c0, w_i) = self.weights();

        let z0 = (self.h)(&self.x);

        let mut z_plus: [[S; M]; N] = core::array::from_fn(|_| [S::ZERO; M]);
        let mut z_minus: [[S; M]; N] = core::array::from_fn(|_| [S::ZERO; M]);
        let mut xi_plus_arr: [[S; N]; N] = core::array::from_fn(|_| [S::ZERO; N]);
        let mut xi_minus_arr: [[S; N]; N] = core::array::from_fn(|_| [S::ZERO; N]);

        for j in 0..N {
            let mut xi_plus = self.x;
            let mut xi_minus = self.x;
            for row in 0..N {
                xi_plus[row] += l.data[row][j];
                xi_minus[row] -= l.data[row][j];
            }
            xi_plus_arr[j] = xi_plus;
            xi_minus_arr[j] = xi_minus;
            z_plus[j] = (self.h)(&xi_plus);
            z_minus[j] = (self.h)(&xi_minus);
        }

        // Predicted measurement mean
        let mut z_pred = [S::ZERO; M];
        for k in 0..M {
            z_pred[k] = w_m0 * z0[k];
            for j in 0..N {
                z_pred[k] += w_i * (z_plus[j][k] + z_minus[j][k]);
            }
        }

        // Innovation covariance S_mat (M×M) and cross-covariance C_xz (N×M)
        let dz0: [S; M] = core::array::from_fn(|k| z0[k] - z_pred[k]);
        let mut s_mat = self.r.add_mat(&outer(&dz0, &dz0).scale(w_c0));
        let mut c_xz = Matrix::<S, N, M>::zeros();

        for j in 0..N {
            let dz_p: [S; M] = core::array::from_fn(|k| z_plus[j][k] - z_pred[k]);
            let dz_m: [S; M] = core::array::from_fn(|k| z_minus[j][k] - z_pred[k]);
            let dy_p: [S; N] = core::array::from_fn(|k| xi_plus_arr[j][k] - self.x[k]);
            let dy_m: [S; N] = core::array::from_fn(|k| xi_minus_arr[j][k] - self.x[k]);
            s_mat = s_mat
                .add_mat(&outer(&dz_p, &dz_p).scale(w_i))
                .add_mat(&outer(&dz_m, &dz_m).scale(w_i));
            for row in 0..N {
                for col in 0..M {
                    c_xz.data[row][col] += w_i * (dy_p[row] * dz_p[col] + dy_m[row] * dz_m[col]);
                }
            }
        }

        // Kalman gain K = C_xz * S^{-1}  (N×M)
        let s_inv = s_mat.inv()?;
        let k = matmul(&c_xz, &s_inv);

        // Innovation and state update
        let innovation: [S; M] = core::array::from_fn(|k| z_meas[k] - z_pred[k]);
        let k_innov = matvec(&k, &innovation);
        for (i, &ki) in k_innov.iter().enumerate().take(N) {
            self.x[i] += ki;
        }

        // Covariance update: P = P - K * S * K^T
        let k_s = matmul(&k, &s_mat);
        let k_s_kt = matmul(&k_s, &k.transpose());
        self.p = self.p.sub_mat(&k_s_kt);

        Some(innovation)
    }

    pub fn state(&self) -> &[S; N] {
        &self.x
    }

    pub fn covariance(&self) -> &Matrix<S, N, N> {
        &self.p
    }

    pub fn set_state(&mut self, x: [S; N]) {
        self.x = x;
    }

    pub fn reset(&mut self, x0: [S; N], p0: Matrix<S, N, N>) {
        self.x = x0;
        self.p = p0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f_linear(x: &[f64; 2], u: &[f64; 1]) -> [f64; 2] {
        [x[0] + 0.01 * x[1], x[1] + 0.01 * u[0]]
    }
    fn h_pos(x: &[f64; 2]) -> [f64; 1] {
        [x[0]]
    }

    fn build_ukf() -> Ukf<f64, 2, 1, 1> {
        let q = Matrix::<f64, 2, 2>::identity().scale(1e-4);
        let r = Matrix::<f64, 1, 1>::identity().scale(0.1);
        let p0 = Matrix::<f64, 2, 2>::identity();
        Ukf::standard(q, r, f_linear, h_pos, [0.0_f64; 2], p0)
    }

    #[test]
    fn predict_runs() {
        let mut ukf = build_ukf();
        assert!(ukf.predict(&[0.0]));
    }

    #[test]
    fn update_returns_innovation() {
        let mut ukf = build_ukf();
        ukf.predict(&[0.0]);
        let innov = ukf.update(&[0.1]);
        assert!(innov.is_some());
    }

    #[test]
    fn tracks_constant_position() {
        let mut ukf = build_ukf();
        let true_pos = 5.0_f64;
        for _ in 0..200 {
            ukf.predict(&[0.0]);
            ukf.update(&[true_pos]);
        }
        assert!((ukf.state()[0] - true_pos).abs() < 0.5);
    }

    #[test]
    fn cholesky_consistency() {
        // Verify cholesky of covariance keeps P positive definite
        let mut ukf = build_ukf();
        for _ in 0..10 {
            ukf.predict(&[0.0]);
            ukf.update(&[1.0]);
        }
        // P should remain positive definite
        assert!(ukf.p.cholesky().is_some());
    }
}
