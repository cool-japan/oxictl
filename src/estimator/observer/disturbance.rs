use crate::core::matrix::{matvec, Matrix};
use crate::core::scalar::ControlScalar;

/// Disturbance Observer (DOB) for linear SISO systems.
///
/// Estimates an unknown constant (or slow-varying) disturbance d acting on
/// the plant input channel: x[k+1] = A*x + B*(u + d), y = C*x.
///
/// Augmented state: ξ = [x; d], extended observer tracks both.
///
/// Observer gain L_ext must be designed for the augmented system
/// A_ext = [[A, B], [0, 1]], C_ext = [C, 0].
///
/// - N: original state dimension
/// - M: output dimension
/// - I: input dimension (disturbance estimated on channel 0)
pub struct DisturbanceObserver<S: ControlScalar, const N: usize, const M: usize> {
    /// Original A matrix.
    a: Matrix<S, N, N>,
    /// Input vector (first column of B, or B for SISO).
    b_col: [S; N],
    /// Output matrix.
    c: Matrix<S, M, N>,
    /// Observer gain for state part (N×M).
    l_x: Matrix<S, N, M>,
    /// Observer gain for disturbance part (M elements, one per output).
    l_d: [S; M],
    /// State estimate.
    x_hat: [S; N],
    /// Disturbance estimate.
    d_hat: S,
}

impl<S: ControlScalar, const N: usize, const M: usize> DisturbanceObserver<S, N, M> {
    pub fn new(
        a: Matrix<S, N, N>,
        b_col: [S; N],
        c: Matrix<S, M, N>,
        l_x: Matrix<S, N, M>,
        l_d: [S; M],
    ) -> Self {
        Self {
            a,
            b_col,
            c,
            l_x,
            l_d,
            x_hat: [S::ZERO; N],
            d_hat: S::ZERO,
        }
    }

    /// Update observer with scalar input and measured output.
    ///
    /// - `u`: control input (scalar, SISO)
    /// - `y`: measured output
    ///
    /// Returns (&state_estimate, disturbance_estimate)
    pub fn update(&mut self, u: S, y: &[S; M]) -> (&[S; N], S) {
        let y_hat = matvec(&self.c, &self.x_hat);
        let innovation: [S; M] = core::array::from_fn(|i| y[i] - y_hat[i]);

        // x̂_new = A*x̂ + B*(u + d̂) + L_x*e
        let ax = matvec(&self.a, &self.x_hat);
        let bu_plus_d: S = u + self.d_hat;
        let le = matvec(&self.l_x, &innovation);

        self.x_hat = core::array::from_fn(|i| ax[i] + self.b_col[i] * bu_plus_d + le[i]);

        // d̂_new = d̂ + L_d * e
        let l_d_e: S = self
            .l_d
            .iter()
            .zip(innovation.iter())
            .map(|(&ld, &e)| ld * e)
            .fold(S::ZERO, |a, b| a + b);
        self.d_hat += l_d_e;

        (&self.x_hat, self.d_hat)
    }

    pub fn state(&self) -> &[S; N] {
        &self.x_hat
    }

    pub fn disturbance(&self) -> S {
        self.d_hat
    }

    pub fn reset(&mut self) {
        self.x_hat = [S::ZERO; N];
        self.d_hat = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_dob() -> DisturbanceObserver<f64, 1, 1> {
        // Integrator plant: x[k+1] = x[k] + (u + d)
        let a = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let b = [1.0_f64];
        let c = Matrix::<f64, 1, 1> { data: [[1.0]] };
        // Gains designed for augmented system (place poles at ~0.5)
        let l_x = Matrix::<f64, 1, 1> { data: [[0.6]] };
        let l_d = [0.4_f64];
        DisturbanceObserver::new(a, b, c, l_x, l_d)
    }

    #[test]
    fn estimates_constant_disturbance() {
        let mut dob = build_dob();
        let true_dist = 2.0_f64;
        let mut x_true = 0.0_f64;
        for _ in 0..200 {
            let u = 0.0;
            x_true += u + true_dist;
            dob.update(u, &[x_true]);
        }
        assert!((dob.disturbance() - true_dist).abs() < 0.5);
    }

    #[test]
    fn no_disturbance_zero_estimate() {
        let mut dob = build_dob();
        let mut x_true = 0.0_f64;
        for _ in 0..50 {
            let u = 0.1;
            x_true += u;
            dob.update(u, &[x_true]);
        }
        assert!(dob.disturbance().abs() < 0.5);
    }

    #[test]
    fn reset_clears_estimates() {
        let mut dob = build_dob();
        for _ in 0..20 {
            dob.update(1.0, &[5.0]);
        }
        dob.reset();
        assert_eq!(dob.disturbance(), 0.0);
    }
}
