use crate::core::matrix::{matvec, Matrix};
use crate::core::scalar::ControlScalar;

/// Luenberger state observer for linear systems.
///
/// Observer: x̂[k+1] = A*x̂[k] + B*u[k] + L*(y[k] - C*x̂[k])
///
/// The observer gain L must be designed to place observer poles
/// faster than the plant poles (typically 2-5× faster).
///
/// - N: state dimension
/// - M: output (measurement) dimension
/// - I: input dimension
pub struct LuenbergerObserver<S: ControlScalar, const N: usize, const M: usize, const I: usize> {
    /// State transition matrix.
    a: Matrix<S, N, N>,
    /// Input matrix.
    b: Matrix<S, N, I>,
    /// Output matrix.
    c: Matrix<S, M, N>,
    /// Observer gain matrix L (N×M): places observer poles.
    l: Matrix<S, N, M>,
    /// State estimate.
    x_hat: [S; N],
}

impl<S: ControlScalar, const N: usize, const M: usize, const I: usize>
    LuenbergerObserver<S, N, M, I>
{
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        c: Matrix<S, M, N>,
        l: Matrix<S, N, M>,
    ) -> Self {
        Self {
            a,
            b,
            c,
            l,
            x_hat: [S::ZERO; N],
        }
    }

    pub fn with_initial_state(mut self, x0: [S; N]) -> Self {
        self.x_hat = x0;
        self
    }

    /// Update the observer estimate.
    ///
    /// - `u`: control input applied at the previous step
    /// - `y`: measured output at the current step
    ///
    /// Returns the updated state estimate.
    pub fn update(&mut self, u: &[S; I], y: &[S; M]) -> &[S; N] {
        // Predicted output: ŷ = C * x̂
        let y_hat = matvec(&self.c, &self.x_hat);

        // Innovation: e = y - ŷ
        let innovation: [S; M] = core::array::from_fn(|i| y[i] - y_hat[i]);

        // x̂_new = A*x̂ + B*u + L*e
        let ax = matvec(&self.a, &self.x_hat);
        let bu = matvec(&self.b, u);
        let le = matvec(&self.l, &innovation);

        self.x_hat = core::array::from_fn(|i| ax[i] + bu[i] + le[i]);
        &self.x_hat
    }

    pub fn state(&self) -> &[S; N] {
        &self.x_hat
    }

    pub fn reset(&mut self) {
        self.x_hat = [S::ZERO; N];
    }

    pub fn set_state(&mut self, x: [S; N]) {
        self.x_hat = x;
    }

    /// Compute residual (output error) for diagnostic purposes.
    pub fn residual(&self, y: &[S; M]) -> [S; M] {
        let y_hat = matvec(&self.c, &self.x_hat);
        core::array::from_fn(|i| y[i] - y_hat[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple integrator: x[k+1] = x[k] + u[k], y = x
    fn build_integrator_observer() -> LuenbergerObserver<f64, 1, 1, 1> {
        let a = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let b = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let c = Matrix::<f64, 1, 1> { data: [[1.0]] };
        // Observer pole at z=0.5 (deadbeat-ish): L = 0.5
        let l = Matrix::<f64, 1, 1> { data: [[0.5]] };
        LuenbergerObserver::new(a, b, c, l)
    }

    #[test]
    fn observer_converges_to_true_state() {
        let mut obs = build_integrator_observer();
        let mut x_true = 0.0_f64;
        for _ in 0..100 {
            let u = 0.1_f64;
            // Feed current measurement BEFORE advancing plant (correct observer timing)
            obs.update(&[u], &[x_true]);
            x_true += u; // plant: x[k+1] = x[k] + u
        }
        assert!((obs.state()[0] - x_true).abs() < 0.01);
    }

    #[test]
    fn zero_input_zero_output() {
        let mut obs = build_integrator_observer();
        for _ in 0..10 {
            obs.update(&[0.0], &[0.0]);
        }
        assert!(obs.state()[0].abs() < 1e-10);
    }

    #[test]
    fn reset_clears_state() {
        let mut obs = build_integrator_observer();
        for _ in 0..10 {
            obs.update(&[1.0], &[5.0]);
        }
        obs.reset();
        assert_eq!(obs.state()[0], 0.0);
    }

    #[test]
    fn residual_is_output_error() {
        let mut obs = build_integrator_observer();
        obs.set_state([3.0]);
        let r = obs.residual(&[5.0]);
        assert!((r[0] - 2.0).abs() < 1e-10);
    }
}
