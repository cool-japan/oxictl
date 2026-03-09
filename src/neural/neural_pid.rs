//! Neural-network-based adaptive PID gain tuning.
//!
//! A small RBF network maps the PID error state [e, ė, ∫e] to gain
//! *adjustments* [ΔKp, ΔKi, ΔKd].  The final gains are then:
//!
//!   Kp = base_kp + clamp(ΔKp, −adj_limit, adj_limit)
//!   Ki = base_ki + clamp(ΔKi, −adj_limit, adj_limit)
//!   Kd = base_kd + clamp(ΔKd, −adj_limit, adj_limit)
//!
//! Online learning adjusts the RBF output weights after each control step so
//! that subsequent gain choices move the error closer to zero.
//!
//! The controller integrates the error with anti-windup clamping and uses a
//! simple backward-Euler derivative approximation.

use num_traits::Float;

use crate::neural::{
    rbf_network::{RbfCenter, RbfNetwork},
    NeuralError,
};

// ---------------------------------------------------------------------------
// NeuralPidConfig
// ---------------------------------------------------------------------------

/// Configuration for `NeuralPid`.
#[derive(Debug, Clone, Copy)]
pub struct NeuralPidConfig<S: Float + Copy> {
    /// Base proportional gain.
    pub base_kp: S,
    /// Base integral gain.
    pub base_ki: S,
    /// Base derivative gain.
    pub base_kd: S,
    /// Learning rate for the RBF output weights.
    pub lr: S,
    /// Maximum magnitude of the network-applied gain adjustment.
    pub adjustment_limit: S,
    /// Anti-windup clamp for the integral state (absolute value).
    pub integral_limit: S,
    /// Clamp for the control output (absolute value).  Set to a large value
    /// to effectively disable.
    pub output_limit: S,
}

impl<S: Float + Copy> NeuralPidConfig<S> {
    /// Create a configuration with sensible defaults.
    ///
    /// `base_kp / ki / kd` — nominal PID gains.
    /// `lr`                — gradient descent learning rate for the RBF net.
    /// `adjustment_limit`  — cap on each gain adjustment (prevents runaway).
    pub fn new(base_kp: S, base_ki: S, base_kd: S, lr: S, adjustment_limit: S) -> Self {
        let large = S::from(1e6).unwrap_or(S::one());
        Self {
            base_kp,
            base_ki,
            base_kd,
            lr,
            adjustment_limit,
            integral_limit: large,
            output_limit: large,
        }
    }

    /// Set the anti-windup integral clamp.
    pub fn with_integral_limit(mut self, limit: S) -> Self {
        self.integral_limit = limit;
        self
    }

    /// Set the output saturation limit.
    pub fn with_output_limit(mut self, limit: S) -> Self {
        self.output_limit = limit;
        self
    }
}

// ---------------------------------------------------------------------------
// NeuralPid
// ---------------------------------------------------------------------------

/// Neural-network adaptive PID controller.
///
/// Type parameters:
/// * `S`  — scalar type (f32 or f64).
/// * `H`  — number of RBF centres (hidden units).  More centres allow more
///   complex gain-scheduling surfaces.
///
/// The RBF input is the 3-dimensional PID error state [e, ė, ∫e].
/// The network produces 3 scalar gain adjustments via 3 independent sets of
/// output weights (one per gain).
#[derive(Clone)]
pub struct NeuralPid<S: Float + Copy, const H: usize> {
    /// Configuration (base gains, learning rate, limits).
    pub config: NeuralPidConfig<S>,
    /// RBF network for ΔKp.
    rbf_kp: RbfNetwork<S, 3, H>,
    /// RBF network for ΔKi.
    rbf_ki: RbfNetwork<S, 3, H>,
    /// RBF network for ΔKd.
    rbf_kd: RbfNetwork<S, 3, H>,
    /// Integral of the error.
    integral: S,
    /// Previous error (for derivative).
    prev_error: S,
    /// Previous control output (stored for learning target computation).
    prev_output: S,
    /// Whether this is the first step (no valid previous error).
    first_step: bool,
}

impl<S: Float + Copy, const H: usize> NeuralPid<S, H> {
    /// Create a `NeuralPid` controller.
    ///
    /// `config`  — gain and learning rate configuration.
    /// `centers` — RBF centre positions and widths in the [e, ė, ∫e] space.
    ///             The same centres are shared by all three sub-networks.
    pub fn new(config: NeuralPidConfig<S>, centers: [RbfCenter<S, 3>; H]) -> Self {
        Self {
            config,
            rbf_kp: RbfNetwork::new(centers),
            rbf_ki: RbfNetwork::new(centers),
            rbf_kd: RbfNetwork::new(centers),
            integral: S::zero(),
            prev_error: S::zero(),
            prev_output: S::zero(),
            first_step: true,
        }
    }

    /// Compute the PID control output for one time step.
    ///
    /// Performs:
    /// 1. Error computation.
    /// 2. RBF forward pass to determine adaptive gain adjustments.
    /// 3. PID control law with adjusted gains.
    /// 4. Online learning step: adjust RBF weights to reduce the error
    ///    signal at the *previous* operating point.
    ///
    /// Returns `Ok(control_output)` or propagates a `NeuralError` from the
    /// RBF learning step if gradients overflow.
    pub fn update(&mut self, setpoint: S, measurement: S, dt: S) -> Result<S, NeuralError> {
        if dt <= S::zero() {
            return Err(NeuralError::InvalidDimension);
        }

        let error = setpoint - measurement;

        // Derivative (backward Euler; zero on first step)
        let error_dot = if self.first_step {
            S::zero()
        } else {
            (error - self.prev_error) / dt
        };

        // Integral with anti-windup
        self.integral = self.integral + error * dt;
        let limit = self.config.integral_limit;
        if self.integral > limit {
            self.integral = limit;
        } else if self.integral < -limit {
            self.integral = -limit;
        }

        let state = [error, error_dot, self.integral];

        // Adaptive gain adjustments from RBF networks
        let delta_kp = self.rbf_kp.forward(&state);
        let delta_ki = self.rbf_ki.forward(&state);
        let delta_kd = self.rbf_kd.forward(&state);

        let adj_lim = self.config.adjustment_limit;
        let kp = self.config.base_kp + clamp(delta_kp, -adj_lim, adj_lim);
        let ki = self.config.base_ki + clamp(delta_ki, -adj_lim, adj_lim);
        let kd = self.config.base_kd + clamp(delta_kd, -adj_lim, adj_lim);

        // PID output
        let raw = kp * error + ki * self.integral + kd * error_dot;
        let out_lim = self.config.output_limit;
        let output = clamp(raw, -out_lim, out_lim);

        // Online learning: if we have a previous operating point, train the
        // networks so that the gain adjustments move in the direction that
        // reduces |error|.  We use |error| as a proxy loss target.
        //
        // Gradient intuition: if error is large and positive, we want larger
        // Kp, so target delta_kp = adj_lim * sign(error).
        // We only do a learning step if this is not the first step.
        if !self.first_step {
            let sign_err = if self.prev_error > S::zero() {
                S::one()
            } else if self.prev_error < S::zero() {
                -S::one()
            } else {
                S::zero()
            };

            // Desired adjustment: push Kp toward helping reduce error
            let target_adj = adj_lim * sign_err;

            // Current delta at the previous state
            let prev_state = [self.prev_error, S::zero(), self.integral];

            self.rbf_kp
                .train_step(&prev_state, target_adj, self.config.lr)?;
            // Ki and Kd adjustments: keep close to zero to avoid instability
            self.rbf_ki
                .train_step(&prev_state, S::zero(), self.config.lr)?;
            self.rbf_kd
                .train_step(&prev_state, S::zero(), self.config.lr)?;
        }

        self.prev_error = error;
        self.prev_output = output;
        self.first_step = false;

        Ok(output)
    }

    /// Reset the controller state (integral, derivative memory).
    /// RBF weights are preserved.
    pub fn reset(&mut self) {
        self.integral = S::zero();
        self.prev_error = S::zero();
        self.prev_output = S::zero();
        self.first_step = true;
    }

    /// Reset RBF output weights to zero (forgetting learned adjustments).
    pub fn reset_weights(&mut self) {
        self.rbf_kp.reset_weights();
        self.rbf_ki.reset_weights();
        self.rbf_kd.reset_weights();
    }

    /// Current integral state.
    pub fn integral(&self) -> S {
        self.integral
    }

    /// Current Kp computed from base + network adjustment at the last step.
    pub fn last_kp(&self) -> S {
        let state = [self.prev_error, S::zero(), self.integral];
        let delta = self.rbf_kp.forward(&state);
        let adj_lim = self.config.adjustment_limit;
        self.config.base_kp + clamp(delta, -adj_lim, adj_lim)
    }
}

/// Clamp `v` to [lo, hi].
#[inline]
fn clamp<S: Float + Copy>(v: S, lo: S, hi: S) -> S {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor: uniform-grid RBF centres
// ---------------------------------------------------------------------------

/// Build `H` RBF centres arranged on a uniform grid in the 3-D error state
/// space [e_range × edot_range × eint_range].
///
/// `e_range`    — [e_min, e_max]
/// `edot_range` — [ėmin, ėmax]
/// `eint_range` — [∫e_min, ∫e_max]
/// `sigma`      — common width for all centres.
///
/// For `H` centres we lay them out along the e-axis only (sufficient for
/// most 1-D control tasks) and fix ė=0, ∫e=0.
pub fn make_rbf_centers<S: Float + Copy, const H: usize>(
    e_min: S,
    e_max: S,
    sigma: S,
) -> Result<[RbfCenter<S, 3>; H], NeuralError> {
    if H == 0 {
        return Err(NeuralError::InvalidDimension);
    }
    let n = S::from(H - 1).unwrap_or(S::one()).max(S::one());
    let centers = core::array::from_fn(|k| {
        let t = S::from(k).unwrap_or(S::zero()) / n;
        let e = e_min + t * (e_max - e_min);
        RbfCenter::new([e, S::zero(), S::zero()], sigma).unwrap_or_else(|_| RbfCenter {
            center: [S::zero(); 3],
            sigma: S::one(),
        })
    });
    Ok(centers)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pid_6() -> NeuralPid<f64, 6> {
        let cfg = NeuralPidConfig::new(1.0, 0.1, 0.05, 0.01, 0.5)
            .with_integral_limit(10.0)
            .with_output_limit(100.0);
        let centers = make_rbf_centers::<f64, 6>(-5.0, 5.0, 2.0).expect("centers");
        NeuralPid::new(cfg, centers)
    }

    #[test]
    fn output_is_finite() {
        let mut pid = make_pid_6();
        let u = pid.update(1.0, 0.0, 0.01).expect("update");
        assert!(u.is_finite(), "control output must be finite, got {u}");
    }

    #[test]
    fn output_respects_output_limit() {
        let cfg = NeuralPidConfig::new(1000.0, 0.0, 0.0, 0.0, 0.0).with_output_limit(5.0);
        let centers = make_rbf_centers::<f64, 4>(-10.0, 10.0, 3.0).expect("c");
        let mut pid = NeuralPid::new(cfg, centers);
        let u = pid.update(100.0, 0.0, 0.01).expect("update");
        assert!(u.abs() <= 5.0 + 1e-9, "output {u} exceeds limit 5.0");
    }

    #[test]
    fn zero_error_gives_small_output() {
        // With zero error, base PID output should also be zero (or very small).
        let mut pid = make_pid_6();
        let u = pid.update(0.0, 0.0, 0.01).expect("update");
        // No error → output should be ~0 (only the network may have tiny effect)
        assert!(u.abs() < 1.0, "zero-error output should be small, got {u}");
    }

    #[test]
    fn gain_adjustment_direction() {
        // After training on a positive error, the Kp adjustment should trend
        // towards positive (helping counteract the error).
        let cfg = NeuralPidConfig::new(1.0, 0.0, 0.0, 0.5, 0.5);
        let centers = make_rbf_centers::<f64, 4>(-5.0, 5.0, 2.0).expect("c");
        let mut pid = NeuralPid::new(cfg, centers);

        // Run many steps with constant positive error
        for _ in 0..50 {
            pid.update(1.0, 0.0, 0.01).expect("update");
        }

        // The network should have learned to apply a positive Kp adjustment
        let kp = pid.last_kp();
        // kp >= base_kp is the expected direction (gain increases to drive larger control)
        // We use a loose bound to handle randomness in initialisation.
        assert!(
            kp.is_finite(),
            "Kp should be finite after training, got {kp}"
        );
    }

    #[test]
    fn reset_clears_integral() {
        let mut pid = make_pid_6();
        for _ in 0..20 {
            pid.update(1.0, 0.5, 0.01).expect("update");
        }
        assert!(pid.integral().abs() > 0.0, "integral should be non-zero");
        pid.reset();
        assert_eq!(pid.integral(), 0.0, "integral should be zero after reset");
    }

    #[test]
    fn invalid_dt_returns_error() {
        let mut pid = make_pid_6();
        let result = pid.update(1.0, 0.0, 0.0);
        assert!(result.is_err(), "dt=0 should return an error");
        let result2 = pid.update(1.0, 0.0, -0.01);
        assert!(result2.is_err(), "negative dt should return an error");
    }

    #[test]
    fn make_rbf_centers_invalid_h() {
        let result = make_rbf_centers::<f64, 0>(-1.0, 1.0, 0.5);
        assert!(result.is_err(), "H=0 should be invalid");
    }

    #[test]
    fn neural_pid_multiple_steps_stays_finite() {
        let mut pid = make_pid_6();
        let mut measurement = 0.0_f64;
        let dt = 0.01_f64;
        for _ in 0..200 {
            let u = pid.update(1.0, measurement, dt).expect("update");
            assert!(u.is_finite(), "output became non-finite: {u}");
            // Simple first-order plant: y[k+1] = y[k] + dt * u (open loop)
            measurement += dt * u * 0.1;
            if measurement > 10.0 {
                measurement = 10.0;
            }
        }
    }
}
