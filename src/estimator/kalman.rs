use crate::core::matrix::{matmul, matvec, vec_add, Matrix};
use crate::core::scalar::ControlScalar;

/// Linear Kalman Filter.
///
/// - N: state dimension
/// - M: measurement dimension
/// - I: input (control) dimension
///
/// Discrete-time model:
///   x[k+1] = A*x[k] + B*u[k] + w[k],   w ~ N(0, Q)
///   z[k]   = H*x[k] + v[k],              v ~ N(0, R)
#[derive(Debug, Clone)]
pub struct KalmanFilter<S: ControlScalar, const N: usize, const M: usize, const I: usize> {
    /// State transition matrix (N×N).
    pub a: Matrix<S, N, N>,
    /// Control input matrix (N×I).
    pub b: Matrix<S, N, I>,
    /// Measurement matrix (M×N).
    pub h: Matrix<S, M, N>,
    /// Process noise covariance (N×N).
    pub q: Matrix<S, N, N>,
    /// Measurement noise covariance (M×M).
    pub r: Matrix<S, M, M>,
    /// State estimate.
    x: [S; N],
    /// Error covariance (N×N).
    p: Matrix<S, N, N>,
}

impl<S: ControlScalar, const N: usize, const M: usize, const I: usize> KalmanFilter<S, N, M, I> {
    /// Create a new Kalman filter.
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        h: Matrix<S, M, N>,
        q: Matrix<S, N, N>,
        r: Matrix<S, M, M>,
        x0: [S; N],
        p0: Matrix<S, N, N>,
    ) -> Self {
        Self {
            a,
            b,
            h,
            q,
            r,
            x: x0,
            p: p0,
        }
    }

    /// Predict step: propagate state and covariance forward.
    pub fn predict(&mut self, u: &[S; I]) {
        // x_pred = A*x + B*u
        let ax = matvec(&self.a, &self.x);
        let bu = matvec(&self.b, u);
        self.x = vec_add(&ax, &bu);

        // P_pred = A*P*A^T + Q
        let ap = matmul(&self.a, &self.p);
        let at = self.a.transpose();
        let apat = matmul(&ap, &at);
        self.p = apat.add_mat(&self.q);
    }

    /// Update step: incorporate measurement z.
    ///
    /// Returns the innovation (z - H*x_pred) for diagnostics.
    pub fn update(&mut self, z: &[S; M]) -> Option<[S; M]> {
        // Innovation: y = z - H*x
        let hx = matvec(&self.h, &self.x);
        let innovation: [S; M] = core::array::from_fn(|i| z[i] - hx[i]);

        // Innovation covariance: S = H*P*H^T + R
        let hp = matmul(&self.h, &self.p);
        let ht = self.h.transpose();
        let hpht = matmul(&hp, &ht);
        let s_mat = hpht.add_mat(&self.r);

        // Kalman gain: K = P*H^T * S^-1
        let s_inv = s_mat.inv()?;
        let pht = matmul(&self.p, &ht);
        let k = matmul(&pht, &s_inv);

        // State update: x = x + K*y
        let ky = matvec(&k, &innovation);
        self.x = vec_add(&self.x, &ky);

        // Covariance update: P = (I - K*H)*P
        let kh = matmul(&k, &self.h);
        let eye = Matrix::<S, N, N>::identity();
        let i_minus_kh = eye.sub_mat(&kh);
        self.p = matmul(&i_minus_kh, &self.p);

        Some(innovation)
    }

    /// Current state estimate.
    pub fn state(&self) -> &[S; N] {
        &self.x
    }

    /// Current error covariance.
    pub fn covariance(&self) -> &Matrix<S, N, N> {
        &self.p
    }

    /// Reset state and covariance.
    pub fn reset(&mut self, x0: [S; N], p0: Matrix<S, N, N>) {
        self.x = x0;
        self.p = p0;
    }
}

/// Builder for common Kalman filter configurations.
impl<S: ControlScalar> KalmanFilter<S, 2, 1, 1> {
    /// Position-velocity tracker with position measurement.
    /// State: [position, velocity], Input: [acceleration], Measurement: [position]
    pub fn position_velocity(dt: S, process_noise: S, measurement_noise: S) -> Self {
        // A = [[1, dt], [0, 1]]
        let mut a = Matrix::<S, 2, 2>::identity();
        a.data[0][1] = dt;

        // B = [[dt^2/2], [dt]]
        let mut b = Matrix::<S, 2, 1>::zeros();
        b.data[0][0] = dt * dt * S::HALF;
        b.data[1][0] = dt;

        // H = [[1, 0]]
        let mut h = Matrix::<S, 1, 2>::zeros();
        h.data[0][0] = S::ONE;

        // Q = process_noise * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
        let q_factor = process_noise;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;
        let mut q = Matrix::<S, 2, 2>::zeros();
        q.data[0][0] = q_factor * dt4 * S::from_f64(0.25);
        q.data[0][1] = q_factor * dt3 * S::HALF;
        q.data[1][0] = q_factor * dt3 * S::HALF;
        q.data[1][1] = q_factor * dt2;

        // R = [[measurement_noise^2]]
        let mut r = Matrix::<S, 1, 1>::zeros();
        r.data[0][0] = measurement_noise * measurement_noise;

        // Initial covariance: large uncertainty
        let p0 = Matrix::<S, 2, 2>::identity().scale(S::from_f64(1000.0));

        Self::new(a, b, h, q, r, [S::ZERO; 2], p0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_1d_kf() -> KalmanFilter<f64, 1, 1, 1> {
        // Simple 1D: x[k+1] = x[k] + u[k], z[k] = x[k]
        let a = Matrix { data: [[1.0]] };
        let b = Matrix { data: [[1.0]] };
        let h = Matrix { data: [[1.0]] };
        let q = Matrix { data: [[0.1]] };
        let r = Matrix { data: [[1.0]] };
        KalmanFilter::new(a, b, h, q, r, [0.0], Matrix::identity())
    }

    #[test]
    fn predict_steps_state() {
        let mut kf = build_1d_kf();
        kf.predict(&[1.0]);
        assert!((kf.state()[0] - 1.0).abs() < 1e-10);
        kf.predict(&[1.0]);
        assert!((kf.state()[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn update_with_noisy_measurement() {
        let mut kf = build_1d_kf();
        kf.predict(&[0.0]);
        let innovation = kf.update(&[1.0]);
        assert!(innovation.is_some());
        // State should move toward measurement 1.0
        assert!(kf.state()[0] > 0.0);
        assert!(kf.state()[0] < 1.0);
    }

    #[test]
    fn position_velocity_tracker() {
        let dt = 0.01;
        let mut kf = KalmanFilter::<f64, 2, 1, 1>::position_velocity(dt, 1.0, 0.5);
        // Feed constant velocity measurements
        let velocity = 2.0;
        for i in 0..1000 {
            let true_pos = velocity * (i as f64 * dt);
            kf.predict(&[0.0]);
            kf.update(&[true_pos]);
        }
        // After convergence, velocity estimate should be close to 2.0
        let state = kf.state();
        assert!(
            (state[1] - velocity).abs() < 0.1,
            "Velocity estimate should converge: got {}",
            state[1]
        );
    }

    #[test]
    fn covariance_decreases_after_updates() {
        let mut kf = build_1d_kf();
        let initial_trace = kf.covariance().trace();
        for i in 0..50 {
            kf.predict(&[0.0]);
            kf.update(&[i as f64 * 0.01]);
        }
        let final_trace = kf.covariance().trace();
        assert!(
            final_trace < initial_trace,
            "Covariance should decrease: {} → {}",
            initial_trace,
            final_trace
        );
    }

    #[test]
    fn handles_singular_innovation_covariance() {
        // R = 0, Q = 0 should not panic (returns None from update)
        let a = Matrix::<f64, 1, 1>::identity();
        let b = Matrix::<f64, 1, 1>::zeros();
        let h = Matrix::<f64, 1, 1>::identity();
        let q = Matrix::<f64, 1, 1>::zeros();
        let r = Matrix::<f64, 1, 1>::zeros();
        let mut kf = KalmanFilter::new(a, b, h, q, r, [0.0], Matrix::zeros());
        kf.predict(&[0.0]);
        // P is all zeros, so S = H*P*H^T + R = 0, which is singular → None
        let result = kf.update(&[1.0]);
        assert!(result.is_none());
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut kf = build_1d_kf();
        for _ in 0..10 {
            kf.predict(&[1.0]);
            kf.update(&[5.0]);
        }
        kf.reset([0.0], Matrix::identity());
        assert_eq!(kf.state(), &[0.0]);
    }
}
