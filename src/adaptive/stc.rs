use crate::core::scalar::ControlScalar;

/// Self-Tuning Controller (STC) using Recursive Least Squares + Pole Placement.
///
/// Combines:
///   1. **RLS estimator** — online identification of process parameters
///   2. **Pole placement** — controller design based on estimated model
///
/// Assumes an ARX (AutoRegressive with eXogenous input) model:
///   y[k] = a1*y[k-1] + a2*y[k-2] + b1*u[k-1] + b2*u[k-2]
///
/// The controller is a discrete-time RST structure:
///   R*u[k] = T*r[k] - S*y[k]
///
/// Simplified here to a PID-like pole placement for first-order ARX:
///   y[k] = a*y[k-1] + b*u[k-1]
///
/// with desired closed-loop pole at `p_d` (|p_d| < 1).
///
/// N_PARAMS: number of ARX parameters being estimated (typically 2-4).
pub struct SelfTuningController<S: ControlScalar, const N_PARAMS: usize> {
    /// RLS parameter estimate: [a1, a2, ..., b1, b2, ...]
    pub theta: [S; N_PARAMS],
    /// RLS covariance matrix (diagonal approximation).
    p: [S; N_PARAMS],
    /// Forgetting factor.
    pub lambda: S,
    /// Desired closed-loop poles (one per DOF).
    pub desired_poles: [S; 2],
    /// Regressor: [y[k-1], y[k-2], u[k-1], u[k-2]]
    regressor: [S; N_PARAMS],
    /// Previous output.
    prev_y: S,
    /// Previous input.
    prev_u: S,
    /// Controller output limit.
    pub output_limit: S,
    /// Current control output.
    u: S,
}

impl<S: ControlScalar, const N_PARAMS: usize> SelfTuningController<S, N_PARAMS> {
    /// Create a self-tuning controller.
    ///
    /// - `lambda`: forgetting factor (0.9-1.0)
    /// - `p0`: initial RLS covariance (large = high uncertainty, e.g. 1e4)
    /// - `desired_poles`: two desired closed-loop poles (inside unit circle)
    /// - `output_limit`: maximum control signal magnitude
    pub fn new(lambda: S, p0: S, desired_poles: [S; 2], output_limit: S) -> Self {
        Self {
            theta: [S::ZERO; N_PARAMS],
            p: [p0; N_PARAMS],
            lambda,
            desired_poles,
            regressor: [S::ZERO; N_PARAMS],
            prev_y: S::ZERO,
            prev_u: S::ZERO,
            output_limit,
            u: S::ZERO,
        }
    }

    /// Update: identify model and compute control output.
    ///
    /// - `y`: current plant output (measurement)
    /// - `r`: reference (setpoint)
    ///
    /// Returns control signal.
    pub fn update(&mut self, y: S, r: S) -> S {
        // Build regressor from history
        // For N_PARAMS >= 2: regressor = [prev_y, prev_u]
        // For N_PARAMS >= 4: regressor = [prev_y, prev2_y, prev_u, prev2_u]
        if N_PARAMS >= 1 {
            self.regressor[0] = self.prev_y;
        }
        if N_PARAMS >= 2 {
            self.regressor[1] = self.prev_u;
        }
        // Extended regressors would require more history storage; use zero for unused
        // (kept as previous values)

        // RLS update: diagonal covariance approximation
        let phi = &self.regressor;
        let y_hat: S = phi
            .iter()
            .zip(self.theta.iter())
            .map(|(&p, &t)| p * t)
            .fold(S::ZERO, |a, b| a + b);
        let error = y - y_hat;

        // Gain: k_i = p_i * phi_i / (lambda + sum(p_i * phi_i^2))
        let denom = self.lambda
            + phi
                .iter()
                .zip(self.p.iter())
                .map(|(&ph, &pi)| ph * ph * pi)
                .fold(S::ZERO, |a, b| a + b);

        if denom.abs() > S::EPSILON {
            for (i, &ph) in phi.iter().enumerate().take(N_PARAMS) {
                let k_i = self.p[i] * ph / denom;
                self.theta[i] += k_i * error;
                self.p[i] = (self.p[i] - k_i * ph * self.p[i]) / self.lambda;
                // Clamp P to prevent divergence
                self.p[i] = self.p[i].clamp_val(S::EPSILON, S::from_f64(1e8));
            }
        }

        // Pole placement: compute controller gains from estimated model
        // For first-order ARX: y[k] = a*y[k-1] + b*u[k-1]
        // a ≈ theta[0], b ≈ theta[1]
        let a_est = if N_PARAMS >= 1 {
            self.theta[0]
        } else {
            S::ZERO
        };
        let b_est = if N_PARAMS >= 2 { self.theta[1] } else { S::ONE };

        // Desired characteristic polynomial: (z - p1)(z - p2) = z² - (p1+p2)z + p1*p2
        // Simple first-order design: place closed-loop pole at p_d = desired_poles[0]
        // u[k] = (p_d - a) / b * y[k] + (1 - p_d) / b * r[k]   ← deadbeat-like
        let p_d = self.desired_poles[0];

        let control = if b_est.abs() > S::from_f64(0.001) {
            // Pole placement: closed-loop pole at p_d
            // y[k] = (a - b*fb_gain)*y[k-1] + b*ref_gain*r  →  a - b*fb_gain = p_d
            let feedback_gain = (a_est - p_d) / b_est;
            let reference_gain = (S::ONE - p_d) / b_est;
            reference_gain * r - feedback_gain * y
        } else {
            // Model not identified yet; use simple proportional
            S::from_f64(0.5) * (r - y)
        };

        self.u = control.clamp_val(-self.output_limit, self.output_limit);
        self.prev_y = y;
        self.prev_u = self.u;

        self.u
    }

    pub fn control_output(&self) -> S {
        self.u
    }

    pub fn reset(&mut self) {
        self.theta = [S::ZERO; N_PARAMS];
        self.p = [S::from_f64(1e4); N_PARAMS];
        self.regressor = [S::ZERO; N_PARAMS];
        self.prev_y = S::ZERO;
        self.prev_u = S::ZERO;
        self.u = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_and_tracks_setpoint() {
        // Plant: y[k] = 0.8*y[k-1] + 0.5*u[k-1]
        let mut stc = SelfTuningController::<f64, 2>::new(1.0, 1e4, [0.3, 0.1], 10.0);
        let mut y = 0.0_f64;
        let r = 1.0_f64;

        for _ in 0..500 {
            let u = stc.update(y, r);
            y = 0.8 * y + 0.5 * u;
        }
        assert!((y - r).abs() < 0.2, "y={:.4}, r={:.4}", y, r);
    }

    #[test]
    fn control_is_clamped() {
        let mut stc = SelfTuningController::<f64, 2>::new(1.0, 1e4, [0.5, 0.5], 5.0);
        let u = stc.update(0.0, 100.0); // huge reference
        assert!(u.abs() <= 5.0 + 1e-10);
    }

    #[test]
    fn reset_clears_state() {
        let mut stc = SelfTuningController::<f64, 2>::new(0.99, 1e4, [0.5, 0.5], 10.0);
        for _ in 0..100 {
            stc.update(1.0, 2.0);
        }
        stc.reset();
        assert_eq!(stc.theta[0], 0.0);
        assert_eq!(stc.theta[1], 0.0);
    }
}
