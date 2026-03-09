use crate::core::scalar::ControlScalar;

/// Sliding Mode Observer (SMO) for speed and back-EMF estimation.
///
/// Estimates the state by driving the output error to zero using a
/// discontinuous (sign) injection term. Suitable for sensorless motor
/// control where back-EMF (flux linkage) must be estimated.
///
/// Observer:
///   x̂[k+1] = A*x̂[k] + B*u[k] + L*sgn(y[k] - C*x̂[k])
///
/// where `sgn` is replaced by a smooth approximation (sat function)
/// to reduce chattering.
///
/// - N: state dimension
/// - M: output dimension
pub struct SlidingModeObserver<S: ControlScalar, const N: usize, const M: usize> {
    /// State estimate.
    x_hat: [S; N],
    /// Observer gain matrix (N × M), applied to the switching function.
    l: [[S; M]; N],
    /// Switching gain magnitude.
    pub switching_gain: S,
    /// Boundary layer width for saturation smoothing (reduces chattering).
    pub boundary: S,
    /// User-supplied state transition: x̂_new = f(x̂, u, l_term) where l_term = L*sat(e)
    /// Stored as discrete A (N×N) and B (N×1 for scalar input).
    a: [[S; N]; N],
    b: [S; N],
    c: [[S; N]; M],
}

impl<S: ControlScalar, const N: usize, const M: usize> SlidingModeObserver<S, N, M> {
    /// Create a new SMO.
    ///
    /// - `a`: N×N state transition matrix
    /// - `b`: N input vector (scalar input)
    /// - `c`: M×N output matrix
    /// - `l`: N×M observer gain matrix
    /// - `switching_gain`: magnitude of the switching injection
    /// - `boundary`: boundary layer width for saturation smoothing
    pub fn new(
        a: [[S; N]; N],
        b: [S; N],
        c: [[S; N]; M],
        l: [[S; M]; N],
        switching_gain: S,
        boundary: S,
    ) -> Self {
        Self {
            x_hat: [S::ZERO; N],
            l,
            switching_gain,
            boundary,
            a,
            b,
            c,
        }
    }

    /// Saturation function for chattering reduction.
    ///
    /// sat(e, φ) = e/φ if |e| < φ, else sgn(e)
    fn sat(&self, e: S) -> S {
        if self.boundary <= S::ZERO {
            // Pure sign function
            if e > S::ZERO {
                S::ONE
            } else if e < S::ZERO {
                -S::ONE
            } else {
                S::ZERO
            }
        } else if e.abs() < self.boundary {
            e / self.boundary
        } else if e > S::ZERO {
            S::ONE
        } else {
            -S::ONE
        }
    }

    /// Compute output: y_hat = C * x̂
    fn output_estimate(&self) -> [S; M] {
        core::array::from_fn(|i| {
            self.c[i]
                .iter()
                .zip(self.x_hat.iter())
                .map(|(&c_ij, &xj)| c_ij * xj)
                .fold(S::ZERO, |acc, v| acc + v)
        })
    }

    /// Update the observer.
    ///
    /// - `u`: scalar control input
    /// - `y`: measured output (M-dim)
    ///
    /// Returns updated state estimate.
    pub fn update(&mut self, u: S, y: &[S; M]) -> &[S; N] {
        let y_hat = self.output_estimate();

        // Innovation: e_i = y_i - y_hat_i
        let e: [S; M] = core::array::from_fn(|i| y[i] - y_hat[i]);

        // Switching injection: sw_i = switching_gain * sat(e_i)
        let sw: [S; M] = core::array::from_fn(|i| self.switching_gain * self.sat(e[i]));

        // x̂_new = A*x̂ + B*u + L*sw
        self.x_hat = core::array::from_fn(|i| {
            let ax_i: S = self.a[i]
                .iter()
                .zip(self.x_hat.iter())
                .map(|(&a_ij, &xj)| a_ij * xj)
                .fold(S::ZERO, |acc, v| acc + v);
            let l_sw_i: S = self.l[i]
                .iter()
                .zip(sw.iter())
                .map(|(&l_ij, &sw_j)| l_ij * sw_j)
                .fold(S::ZERO, |acc, v| acc + v);
            ax_i + self.b[i] * u + l_sw_i
        });

        &self.x_hat
    }

    pub fn state(&self) -> &[S; N] {
        &self.x_hat
    }

    pub fn set_state(&mut self, x: [S; N]) {
        self.x_hat = x;
    }

    pub fn reset(&mut self) {
        self.x_hat = [S::ZERO; N];
    }

    /// Output error (innovation) at current estimate.
    pub fn innovation(&self, y: &[S; M]) -> [S; M] {
        let y_hat = self.output_estimate();
        core::array::from_fn(|i| y[i] - y_hat[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_integrator_smo() -> SlidingModeObserver<f64, 1, 1> {
        // x[k+1] = x[k] + u, y = x
        let a = [[1.0_f64]];
        let b = [1.0_f64];
        let c = [[1.0_f64]];
        let l = [[1.5_f64]]; // high gain injection
        SlidingModeObserver::new(a, b, c, l, 2.0, 0.1)
    }

    #[test]
    fn tracks_ramp_input() {
        let mut obs = build_integrator_smo();
        let mut x_true = 0.0_f64;
        for _ in 0..200 {
            let u = 0.1_f64;
            obs.update(u, &[x_true]);
            x_true += u;
        }
        // SMO should track the true state within the boundary layer
        assert!(
            (obs.state()[0] - x_true).abs() < 1.0,
            "x̂={:.3}",
            obs.state()[0]
        );
    }

    #[test]
    fn zero_input_stays_at_zero() {
        let mut obs = build_integrator_smo();
        for _ in 0..50 {
            obs.update(0.0, &[0.0]);
        }
        assert!(obs.state()[0].abs() < 1e-10);
    }

    #[test]
    fn reset_clears_state() {
        let mut obs = build_integrator_smo();
        for _ in 0..20 {
            obs.update(1.0, &[5.0]);
        }
        obs.reset();
        assert_eq!(obs.state()[0], 0.0);
    }

    #[test]
    fn boundary_layer_smooth() {
        let obs = build_integrator_smo();
        // Inside boundary: linear
        let v = obs.sat(0.05); // boundary = 0.1
        assert!((v - 0.5).abs() < 1e-10);
        // Outside boundary: saturate to ±1
        assert_eq!(obs.sat(5.0), 1.0);
        assert_eq!(obs.sat(-5.0), -1.0);
    }
}
