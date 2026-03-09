use crate::core::matrix::matvec;
use crate::core::matrix::Matrix;
use crate::core::scalar::ControlScalar;

/// LQR with Integral action (LQI) for SISO output tracking.
///
/// Augments the system with an integrator on the tracking error,
/// eliminating steady-state error for step references.
///
/// Control law:
///   z[k+1] = z[k] + dt*(r[k] - y[k])   (integral error state)
///   u[k] = -K_x * x[k] - K_z * z[k]
///
/// The gains K_x and K_z are typically designed by solving the DARE
/// on the augmented system [[A, 0; -C*dt, 1]], [[B; 0]].
///
/// - N: state dimension
/// - I: input dimension
pub struct Lqi<S: ControlScalar, const N: usize, const I: usize> {
    /// State feedback gain (I×N).
    pub k_x: Matrix<S, I, N>,
    /// Integral gain (one per input channel).
    pub k_z: [S; I],
    /// Accumulated integral of tracking error.
    integral: S,
    /// Anti-windup limit for integral state.
    integral_limit: S,
}

impl<S: ControlScalar, const N: usize, const I: usize> Lqi<S, N, I> {
    pub fn new(k_x: Matrix<S, I, N>, k_z: [S; I]) -> Self {
        Self {
            k_x,
            k_z,
            integral: S::ZERO,
            integral_limit: S::from_f64(1e6),
        }
    }

    /// Set anti-windup limit on the integral state.
    pub fn with_integral_limit(mut self, limit: S) -> Self {
        self.integral_limit = limit;
        self
    }

    /// Update: integrate tracking error and compute control output.
    ///
    /// - `x`: current state estimate
    /// - `r`: reference (setpoint for output y)
    /// - `y`: measured output (used for integral error)
    /// - `dt`: time step
    ///
    /// Returns control input u = -K_x*x - K_z*z.
    pub fn update(&mut self, x: &[S; N], r: S, y: S, dt: S) -> [S; I] {
        self.integral =
            (self.integral + dt * (r - y)).clamp_val(-self.integral_limit, self.integral_limit);

        let u_x = matvec(&self.k_x, x);
        core::array::from_fn(|i| -u_x[i] - self.k_z[i] * self.integral)
    }

    pub fn reset(&mut self) {
        self.integral = S::ZERO;
    }

    pub fn integral_state(&self) -> S {
        self.integral
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lqi_eliminates_steady_state_error() {
        // Plant: x[k+1] = 0.9*x + u, y = x
        // LQI gains: K_x=0.4, K_z=-6.0 — designed via Ackermann for poles at 0.7, 0.8
        // (K_z negative because integral grows positive when y < r, so -K_z*z is positive)
        let k_x = Matrix::<f64, 1, 1> { data: [[0.4]] };
        let k_z = [-6.0_f64];
        let mut lqi = Lqi::new(k_x, k_z);

        let mut x = 0.0_f64;
        let r = 1.0_f64;
        let dt = 0.01;

        for _ in 0..2000 {
            let u = lqi.update(&[x], r, x, dt);
            x = 0.9 * x + u[0];
        }
        assert!(
            (x - r).abs() < 0.05,
            "LQI should eliminate SS error: x={:.4}",
            x
        );
    }

    #[test]
    fn reset_clears_integral() {
        let k_x = Matrix::<f64, 1, 1> { data: [[0.3]] };
        let k_z = [0.1_f64];
        let mut lqi = Lqi::new(k_x, k_z);

        for _ in 0..100 {
            lqi.update(&[0.0], 1.0, 0.0, 0.01);
        }
        assert!(lqi.integral_state().abs() > 0.1);
        lqi.reset();
        assert_eq!(lqi.integral_state(), 0.0);
    }

    #[test]
    fn integral_limit_clamps_windup() {
        let k_x = Matrix::<f64, 1, 1> { data: [[0.0]] };
        let k_z = [0.0_f64];
        let mut lqi = Lqi::new(k_x, k_z).with_integral_limit(5.0);

        for _ in 0..10000 {
            lqi.update(&[0.0], 100.0, 0.0, 0.01);
        }
        assert!(lqi.integral_state().abs() <= 5.0 + 1e-10);
    }
}
