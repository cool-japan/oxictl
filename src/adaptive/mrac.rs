use crate::core::scalar::ControlScalar;

/// Model Reference Adaptive Control (MRAC) for SISO first-order plants.
///
/// Reference model: y_m[k+1] = a_m * y_m[k] + b_m * r[k]
/// Plant:           y_p[k+1] = a_p * y_p[k] + b_p * u[k]
///
/// Adaptive controller: u[k] = θ_r[k] * r[k] + θ_y[k] * y_p[k]
///
/// MIT rule adaptation (gradient descent on 1/2 * e^2):
///   e[k] = y_p[k] - y_m[k]  (tracking error)
///   θ_r[k+1] = θ_r[k] - γ * e[k] * r[k]
///   θ_y[k+1] = θ_y[k] - γ * e[k] * y_p[k]
///
/// The learning rate γ controls adaptation speed vs. stability.
#[derive(Debug, Clone, Copy)]
pub struct Mrac<S: ControlScalar> {
    /// Reference model: y_m[k+1] = a_m*y_m + b_m*r
    pub a_m: S,
    pub b_m: S,
    /// Learning rate (γ).
    pub gamma: S,
    /// Adaptive feedforward gain (θ_r).
    pub theta_r: S,
    /// Adaptive feedback gain (θ_y).
    pub theta_y: S,
    /// Reference model state.
    y_m: S,
    /// Tracking error.
    error: S,
}

impl<S: ControlScalar> Mrac<S> {
    /// Create MRAC controller.
    ///
    /// `a_m`, `b_m`: reference model parameters.
    /// `gamma`: adaptation rate (try 0.01 to 0.5; too large → instability).
    pub fn new(a_m: S, b_m: S, gamma: S) -> Self {
        Self {
            a_m,
            b_m,
            gamma,
            theta_r: b_m,     // Initial: guess plant gain = reference model
            theta_y: S::ZERO, // Initial: no feedback
            y_m: S::ZERO,
            error: S::ZERO,
        }
    }

    /// Update one step.
    ///
    /// - `r`: reference input
    /// - `y_p`: measured plant output
    ///
    /// Returns control output u[k].
    pub fn update(&mut self, r: S, y_p: S) -> S {
        // Update reference model
        self.y_m = self.a_m * self.y_m + self.b_m * r;

        // Tracking error
        self.error = y_p - self.y_m;

        // MIT rule adaptation
        self.theta_r -= self.gamma * self.error * r;
        self.theta_y -= self.gamma * self.error * y_p;

        // Control law
        self.theta_r * r + self.theta_y * y_p
    }

    pub fn reference_output(&self) -> S {
        self.y_m
    }

    pub fn tracking_error(&self) -> S {
        self.error
    }

    pub fn reset(&mut self) {
        self.y_m = S::ZERO;
        self.error = S::ZERO;
        self.theta_r = self.b_m;
        self.theta_y = S::ZERO;
    }
}

/// MRAC with normalization (prevents parameter drift for large signals).
///
/// Normalized MIT rule: θ̇ = -γ * e * φ / (1 + φ^T*φ)
#[derive(Debug, Clone, Copy)]
pub struct MracNormalized<S: ControlScalar> {
    inner: Mrac<S>,
    /// Small constant for normalization denominator (prevents div-by-zero).
    epsilon: S,
}

impl<S: ControlScalar> MracNormalized<S> {
    pub fn new(a_m: S, b_m: S, gamma: S) -> Self {
        Self {
            inner: Mrac::new(a_m, b_m, gamma),
            epsilon: S::from_f64(1e-4),
        }
    }

    pub fn update(&mut self, r: S, y_p: S) -> S {
        self.inner.y_m = self.inner.a_m * self.inner.y_m + self.inner.b_m * r;
        self.inner.error = y_p - self.inner.y_m;

        let norm = S::ONE + r * r + y_p * y_p + self.epsilon;

        self.inner.theta_r -= self.inner.gamma * self.inner.error * r / norm;
        self.inner.theta_y -= self.inner.gamma * self.inner.error * y_p / norm;

        self.inner.theta_r * r + self.inner.theta_y * y_p
    }

    pub fn reference_output(&self) -> S {
        self.inner.y_m
    }

    pub fn tracking_error(&self) -> S {
        self.inner.error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrac_tracks_reference_model() {
        // Reference model: y_m[k+1] = 0.8*y_m + 0.2*r
        // Plant: y_p[k+1] = 0.7*y_p + 2.0*u  (different from model)
        let mut mrac = Mrac::<f64>::new(0.8, 0.2, 0.01);
        let mut y_p = 0.0_f64;

        for _ in 0..2000 {
            let r = 1.0;
            let u = mrac.update(r, y_p);
            y_p = 0.7 * y_p + 2.0 * u;
        }

        // Should track reference model output at steady state
        let y_m_ss = mrac.reference_output();
        assert!(
            (y_p - y_m_ss).abs() < 0.5,
            "y_p={:.4}, y_m={:.4}",
            y_p,
            y_m_ss
        );
    }

    #[test]
    fn mrac_normalized_stable() {
        let mut mrac = MracNormalized::<f64>::new(0.8, 0.2, 0.1);
        let mut y_p = 0.0_f64;
        let mut bounded = true;

        for _ in 0..5000 {
            let r = 1.0;
            let u = mrac.update(r, y_p);
            y_p = 0.7 * y_p + 2.0 * u;
            if y_p.abs() > 1000.0 {
                bounded = false;
                break;
            }
        }
        assert!(bounded, "Output should remain bounded");
    }

    #[test]
    fn zero_input_stays_zero() {
        let mut mrac = Mrac::<f64>::new(0.8, 0.2, 0.01);
        let u = mrac.update(0.0, 0.0);
        assert_eq!(u, 0.0);
        assert_eq!(mrac.reference_output(), 0.0);
    }

    #[test]
    fn reset_returns_to_initial() {
        let mut mrac = Mrac::<f64>::new(0.8, 0.2, 0.01);
        for _ in 0..100 {
            mrac.update(1.0, 0.5);
        }
        mrac.reset();
        assert_eq!(mrac.reference_output(), 0.0);
        assert_eq!(mrac.tracking_error(), 0.0);
    }
}
