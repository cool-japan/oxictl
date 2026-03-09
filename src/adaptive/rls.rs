use crate::core::matrix::{matvec, outer, Matrix};
use crate::core::scalar::ControlScalar;

/// Recursive Least Squares (RLS) parameter estimator.
///
/// Estimates parameter vector θ in the linear regression:
///   y[k] = φ[k]^T * θ + e[k]
///
/// RLS update equations:
///   K[k] = P[k-1]*φ[k] / (λ + φ[k]^T*P[k-1]*φ[k])
///   θ[k] = θ[k-1] + K[k] * (y[k] - φ[k]^T*θ[k-1])
///   P[k] = (P[k-1] - K[k]*φ[k]^T*P[k-1]) / λ
///
/// - N: number of parameters
/// - λ: forgetting factor (0 < λ ≤ 1; λ=1 for batch, λ≈0.95-0.99 for tracking)
pub struct Rls<S: ControlScalar, const N: usize> {
    /// Parameter estimate.
    pub theta: [S; N],
    /// Covariance matrix (N×N).
    pub p: Matrix<S, N, N>,
    /// Forgetting factor (0 < λ ≤ 1).
    pub lambda: S,
}

impl<S: ControlScalar, const N: usize> Rls<S, N> {
    /// Create RLS with initial covariance p0*I and forgetting factor lambda.
    ///
    /// `p0`: initial covariance scale (large = high initial uncertainty, e.g. 1e6).
    pub fn new(lambda: S, p0: S) -> Self {
        Self {
            theta: [S::ZERO; N],
            p: Matrix::<S, N, N>::identity().scale(p0),
            lambda,
        }
    }

    /// Update with new measurement y and regressor φ.
    ///
    /// Returns the updated parameter estimate.
    pub fn update(&mut self, phi: &[S; N], y: S) -> &[S; N] {
        // Predicted output
        let y_hat: S = phi
            .iter()
            .zip(self.theta.iter())
            .map(|(&p, &t)| p * t)
            .fold(S::ZERO, |a, b| a + b);
        let error = y - y_hat;

        // Gain: K = P*φ / (λ + φ^T*P*φ)
        let p_phi = matvec(&self.p, phi);
        let phi_p_phi: S = phi
            .iter()
            .zip(p_phi.iter())
            .map(|(&a, &b)| a * b)
            .fold(S::ZERO, |acc, x| acc + x);
        let denom = self.lambda + phi_p_phi;

        if denom.abs() < S::EPSILON {
            return &self.theta;
        }

        let k: [S; N] = core::array::from_fn(|i| p_phi[i] / denom);

        // Update parameter estimate
        for (i, &ki) in k.iter().enumerate().take(N) {
            self.theta[i] += ki * error;
        }

        // Update covariance: P = (P - K*φ^T*P) / λ
        let k_phi_t: Matrix<S, N, N> = outer(&k, phi); // N×N
        let k_phi_t_p = {
            let mut kp = Matrix::<S, N, N>::zeros();
            for r in 0..N {
                for c in 0..N {
                    kp.data[r][c] = k_phi_t.data[r]
                        .iter()
                        .zip(self.p.data.iter())
                        .map(|(&kf, pr)| kf * pr[c])
                        .fold(S::ZERO, |acc, x| acc + x);
                }
            }
            kp
        };
        self.p = self.p.sub_mat(&k_phi_t_p).scale(S::ONE / self.lambda);

        &self.theta
    }

    /// Prediction error for given measurement and regressor.
    pub fn prediction_error(&self, phi: &[S; N], y: S) -> S {
        let y_hat: S = phi
            .iter()
            .zip(self.theta.iter())
            .map(|(&p, &t)| p * t)
            .fold(S::ZERO, |acc, x| acc + x);
        y - y_hat
    }

    pub fn reset(&mut self, p0: S) {
        self.theta = [S::ZERO; N];
        self.p = Matrix::<S, N, N>::identity().scale(p0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_single_parameter() {
        let mut rls = Rls::<f64, 1>::new(1.0, 1e4);
        let true_gain = 3.0_f64;
        for k in 1..=200 {
            let phi = [k as f64];
            let y = true_gain * phi[0];
            rls.update(&phi, y);
        }
        assert!(
            (rls.theta[0] - true_gain).abs() < 0.01,
            "θ={}",
            rls.theta[0]
        );
    }

    #[test]
    fn identifies_two_parameters() {
        let mut rls = Rls::<f64, 2>::new(1.0, 1e4);
        // y = 2*x1 + 3*x2  — use non-collinear regressors [k, k+17]
        let params = [2.0_f64, 3.0];
        for k in 1..=500 {
            let phi = [k as f64, (k + 17) as f64];
            let y = params[0] * phi[0] + params[1] * phi[1];
            rls.update(&phi, y);
        }
        assert!(
            (rls.theta[0] - params[0]).abs() < 0.1,
            "θ0={}",
            rls.theta[0]
        );
        assert!(
            (rls.theta[1] - params[1]).abs() < 0.1,
            "θ1={}",
            rls.theta[1]
        );
    }

    #[test]
    fn forgetting_factor_tracks_change() {
        let mut rls = Rls::<f64, 1>::new(0.95, 1e4);
        // First phase: true gain = 1.0
        for _k in 1..=500 {
            let phi = [1.0_f64];
            rls.update(&phi, 1.0);
        }
        // Second phase: true gain changes to 5.0
        for _k in 1..=500 {
            let phi = [1.0_f64];
            rls.update(&phi, 5.0);
        }
        assert!((rls.theta[0] - 5.0).abs() < 0.5, "θ={}", rls.theta[0]);
    }

    #[test]
    fn prediction_error_zero_on_perfect_fit() {
        let mut rls = Rls::<f64, 1>::new(1.0, 1e4);
        for k in 1..=200 {
            let phi = [k as f64];
            rls.update(&phi, 2.0 * phi[0]);
        }
        let e = rls.prediction_error(&[5.0], 10.0);
        assert!(e.abs() < 0.01);
    }
}
