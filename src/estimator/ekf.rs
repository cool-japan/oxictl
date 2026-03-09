use crate::core::matrix::{matmul, matvec, vec_add, Matrix};
use crate::core::scalar::ControlScalar;

/// Extended Kalman Filter for nonlinear systems.
///
/// - N: state dimension
/// - M: measurement dimension
/// - I: input dimension
///
/// Nonlinear discrete-time model:
///   x[k+1] = f(x[k], u[k]) + w[k],  w ~ N(0, Q)
///   z[k]   = h(x[k]) + v[k],          v ~ N(0, R)
///
/// EKF linearizes f and h around the current estimate.
pub struct Ekf<S: ControlScalar, const N: usize, const M: usize, const I: usize> {
    /// Process noise covariance (N×N).
    pub q: Matrix<S, N, N>,
    /// Measurement noise covariance (M×M).
    pub r: Matrix<S, M, M>,
    /// State transition function: f(x, u) -> x_next.
    f: fn(&[S; N], &[S; I]) -> [S; N],
    /// Jacobian of f w.r.t. x: ∂f/∂x evaluated at (x, u) -> N×N matrix.
    f_jac: fn(&[S; N], &[S; I]) -> Matrix<S, N, N>,
    /// Measurement function: h(x) -> z.
    h: fn(&[S; N]) -> [S; M],
    /// Jacobian of h w.r.t. x: ∂h/∂x evaluated at x -> M×N matrix.
    h_jac: fn(&[S; N]) -> Matrix<S, M, N>,
    /// State estimate.
    x: [S; N],
    /// Error covariance (N×N).
    p: Matrix<S, N, N>,
}

impl<S: ControlScalar, const N: usize, const M: usize, const I: usize> Ekf<S, N, M, I> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q: Matrix<S, N, N>,
        r: Matrix<S, M, M>,
        f: fn(&[S; N], &[S; I]) -> [S; N],
        f_jac: fn(&[S; N], &[S; I]) -> Matrix<S, N, N>,
        h: fn(&[S; N]) -> [S; M],
        h_jac: fn(&[S; N]) -> Matrix<S, M, N>,
        x0: [S; N],
        p0: Matrix<S, N, N>,
    ) -> Self {
        Self {
            q,
            r,
            f,
            f_jac,
            h,
            h_jac,
            x: x0,
            p: p0,
        }
    }

    /// EKF predict step.
    pub fn predict(&mut self, u: &[S; I]) {
        // Linearize: F = ∂f/∂x at current state
        let f_k = (self.f_jac)(&self.x, u);

        // Propagate state: x_pred = f(x, u)
        self.x = (self.f)(&self.x, u);

        // Propagate covariance: P_pred = F*P*F^T + Q
        let fp = matmul(&f_k, &self.p);
        let ft = f_k.transpose();
        let fpft = matmul(&fp, &ft);
        self.p = fpft.add_mat(&self.q);
    }

    /// EKF update step. Returns innovation if S matrix is invertible.
    pub fn update(&mut self, z: &[S; M]) -> Option<[S; M]> {
        // Linearize: H_k = ∂h/∂x at current state
        let h_k = (self.h_jac)(&self.x);

        // Innovation: y = z - h(x_pred)
        let hx = (self.h)(&self.x);
        let innovation: [S; M] = core::array::from_fn(|i| z[i] - hx[i]);

        // Innovation covariance: S = H*P*H^T + R
        let hp = matmul(&h_k, &self.p);
        let ht = h_k.transpose();
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
        let kh = matmul(&k, &h_k);
        let eye = Matrix::<S, N, N>::identity();
        let i_minus_kh = eye.sub_mat(&kh);
        self.p = matmul(&i_minus_kh, &self.p);

        Some(innovation)
    }

    pub fn state(&self) -> &[S; N] {
        &self.x
    }

    pub fn covariance(&self) -> &Matrix<S, N, N> {
        &self.p
    }

    pub fn reset(&mut self, x0: [S; N], p0: Matrix<S, N, N>) {
        self.x = x0;
        self.p = p0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Linear system: x[k+1] = A*x[k] + B*u, z = x
    // Use EKF on a linear system (should match KF behavior)
    fn f_linear(x: &[f64; 2], u: &[f64; 1]) -> [f64; 2] {
        [x[0] + 0.01 * x[1] + 0.0 * u[0], x[1] + u[0] * 0.01]
    }

    fn f_jac_linear(_x: &[f64; 2], _u: &[f64; 1]) -> Matrix<f64, 2, 2> {
        let mut m = Matrix::identity();
        m.data[0][1] = 0.01;
        m
    }

    fn h_linear(x: &[f64; 2]) -> [f64; 1] {
        [x[0]]
    }

    fn h_jac_linear(_x: &[f64; 2]) -> Matrix<f64, 1, 2> {
        let mut m = Matrix::zeros();
        m.data[0][0] = 1.0;
        m
    }

    #[test]
    fn ekf_tracks_linear_system() {
        let q = Matrix::<f64, 2, 2>::identity().scale(0.01);
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.5;
        let p0 = Matrix::<f64, 2, 2>::identity().scale(10.0);

        let mut ekf = Ekf::new(
            q,
            r,
            f_linear,
            f_jac_linear,
            h_linear,
            h_jac_linear,
            [0.0, 2.0],
            p0,
        );

        // Simulate constant-velocity motion
        let dt = 0.01;
        let v = 2.0;
        for i in 0..1000 {
            let true_pos = v * (i as f64 * dt);
            ekf.predict(&[0.0]);
            ekf.update(&[true_pos + 0.0]); // noiseless measurement for test stability
        }

        let state = ekf.state();
        assert!(
            (state[0] - v * 10.0).abs() < 0.5,
            "Position should be ~{}, got {}",
            v * 10.0,
            state[0]
        );
        assert!(
            (state[1] - v).abs() < 0.5,
            "Velocity should be ~{}, got {}",
            v,
            state[1]
        );
    }

    // Nonlinear measurement: bearing angle to a fixed target
    fn h_bearing(x: &[f64; 2]) -> [f64; 1] {
        // Bearing from origin to (x[0], x[1])
        [x[1].atan2(x[0])]
    }

    fn h_jac_bearing(x: &[f64; 2]) -> Matrix<f64, 1, 2> {
        let r2 = x[0] * x[0] + x[1] * x[1];
        let mut m = Matrix::zeros();
        if r2 > 1e-10 {
            m.data[0][0] = -x[1] / r2;
            m.data[0][1] = x[0] / r2;
        }
        m
    }

    fn f_id(x: &[f64; 2], _u: &[f64; 0]) -> [f64; 2] {
        *x
    }

    fn f_jac_id(_x: &[f64; 2], _u: &[f64; 0]) -> Matrix<f64, 2, 2> {
        Matrix::identity()
    }

    #[test]
    fn ekf_nonlinear_measurement() {
        let q = Matrix::<f64, 2, 2>::identity().scale(0.001);
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.01;
        let p0 = Matrix::<f64, 2, 2>::identity().scale(100.0);

        let true_pos = [3.0_f64, 4.0_f64]; // at (3,4), bearing = atan2(4,3) ≈ 0.9273 rad

        let mut ekf: Ekf<f64, 2, 1, 0> = Ekf::new(
            q,
            r,
            f_id,
            f_jac_id,
            h_bearing,
            h_jac_bearing,
            [5.0, 5.0], // Start with wrong estimate
            p0,
        );

        let bearing = true_pos[1].atan2(true_pos[0]);
        for _ in 0..500 {
            ekf.predict(&[]);
            ekf.update(&[bearing]);
        }

        let state = ekf.state();
        // Bearing-only measurement is inherently ambiguous w.r.t. distance,
        // but the bearing angle of the estimate should match
        let est_bearing = state[1].atan2(state[0]);
        assert!(
            (est_bearing - bearing).abs() < 0.1,
            "Bearing should converge: est={:.3}, true={:.3}",
            est_bearing,
            bearing
        );
    }

    #[test]
    fn covariance_decreases() {
        let q = Matrix::<f64, 2, 2>::identity().scale(0.01);
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 0.5;
        let p0 = Matrix::<f64, 2, 2>::identity().scale(100.0);
        let initial_trace = p0.trace();

        let mut ekf = Ekf::new(
            q,
            r,
            f_linear,
            f_jac_linear,
            h_linear,
            h_jac_linear,
            [0.0, 0.0],
            p0,
        );

        for i in 0..100 {
            ekf.predict(&[0.0]);
            ekf.update(&[i as f64 * 0.01]);
        }

        assert!(ekf.covariance().trace() < initial_trace);
    }
}
