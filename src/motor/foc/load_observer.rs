//! Load torque disturbance observer and friction model.
//!
//! Implements a full-order Luenberger-type disturbance observer for estimating
//! load torque, combined with a Coulomb + viscous friction model.
//!
//! State equation (motor + load):
//!   J·dω/dt = u - τ_f(ω) - τ_load
//!
//! The observer adds an auxiliary integral state to track τ_load:
//!   dω̂/dt = (u - τ_f(ω̂) - τ̂_load)/J + L·(ω - ω̂)
//!   dτ̂_load/dt = -L²·J·(ω - ω̂) / some_gain  [simplified Luenberger]
#![allow(clippy::excessive_precision)]

use crate::core::scalar::ControlScalar;

/// Coulomb + viscous friction model.
///
/// τ_f = fc·sign(ω) + fv·ω
///
/// In the stiction region (|ω| < ω_stiction) a linear interpolation
/// avoids discontinuity: τ_f = (fc/ω_stiction)·ω + fv·ω.
#[derive(Debug, Clone, Copy)]
pub struct FrictionModel<S: ControlScalar> {
    /// fc: Coulomb friction coefficient (N·m).
    pub coulomb_coeff: S,
    /// fv: Viscous damping coefficient (N·m·s/rad).
    pub viscous_coeff: S,
    /// Stiction threshold (rad/s); below this, linear interpolation is used.
    pub stiction_thresh: S,
}

impl<S: ControlScalar> FrictionModel<S> {
    pub fn new(coulomb_coeff: S, viscous_coeff: S, stiction_thresh: S) -> Self {
        Self {
            coulomb_coeff,
            viscous_coeff,
            stiction_thresh,
        }
    }

    /// Compute friction torque.
    ///
    /// For |ω| ≥ stiction_thresh: τ_f = fc·sign(ω) + fv·ω
    /// For |ω| < stiction_thresh: τ_f = (fc/stiction_thresh + fv)·ω  (linear)
    pub fn torque(&self, omega: S) -> S {
        let abs_omega = omega.abs();
        if abs_omega >= self.stiction_thresh && self.stiction_thresh >= S::ZERO {
            self.coulomb_coeff * omega.signum() + self.viscous_coeff * omega
        } else if self.stiction_thresh > S::EPSILON {
            // Linear stiction region
            let gain = self.coulomb_coeff / self.stiction_thresh + self.viscous_coeff;
            gain * omega
        } else {
            // No stiction region: just viscous
            self.viscous_coeff * omega
        }
    }

    /// Adapt viscous coefficient via gradient descent on torque error.
    ///
    /// Update rule: fv += lr · torque_error · ω
    ///
    /// This allows online identification of viscous friction from torque
    /// residuals.
    pub fn adapt_viscous(&mut self, omega: S, torque_error: S, learning_rate: S) {
        self.viscous_coeff += learning_rate * torque_error * omega;
    }
}

/// Full-order disturbance observer for load torque estimation.
///
/// Uses a reduced-order (second-state) Luenberger observer:
///
/// Plant: J·ω̇ = u - τ_f(ω) - τ_load
///
/// Observer:
///   ω̂̇ = (u - τ_f(ω̂) - τ̂_load)/J + L·e_ω
///   τ̂̇_load = −L²·J·e_ω   (ESO disturbance integral)
///
/// where e_ω = ω_meas - ω̂.
///
/// The observer gain L sets the bandwidth: poles at -L (faster → larger L).
#[derive(Debug, Clone, Copy)]
pub struct LoadObserver<S: ControlScalar> {
    /// J: moment of inertia (kg·m²).
    pub inertia: S,
    /// Friction model.
    pub friction: FrictionModel<S>,
    /// L: observer gain (rad/s).
    pub observer_gain: S,
    /// ω̂: estimated angular velocity (rad/s).
    omega_hat: S,
    /// τ̂_load: estimated load torque (N·m).
    tau_load_hat: S,
}

impl<S: ControlScalar> LoadObserver<S> {
    pub fn new(inertia: S, friction: FrictionModel<S>, observer_gain: S) -> Self {
        Self {
            inertia,
            friction,
            observer_gain,
            omega_hat: S::ZERO,
            tau_load_hat: S::ZERO,
        }
    }

    /// Estimate load torque.
    ///
    /// Runs one step of the observer using forward Euler integration.
    ///
    /// # Arguments
    /// * `omega_meas` - Measured angular velocity (rad/s).
    /// * `u` - Applied electromagnetic torque (N·m).
    /// * `dt` - Time step (s).
    ///
    /// # Returns
    /// Updated estimated load torque (N·m).
    pub fn estimate(&mut self, omega_meas: S, u: S, dt: S) -> S {
        // Observer error
        let e_omega = omega_meas - self.omega_hat;

        // Friction torque at estimated speed
        let tau_friction = self.friction.torque(self.omega_hat);

        // Observer equations (forward Euler)
        let l = self.observer_gain;
        let j = self.inertia;

        // dω̂/dt = (u - τ_f - τ̂_load)/J + L·e_ω
        let omega_hat_dot = (u - tau_friction - self.tau_load_hat) / j + l * e_omega;

        // dτ̂_load/dt = −L²·J·e_ω  (ESO: ẑ₂̇ = −β₂·e → τ̂̇ = −J·β₂·e_ω)
        let tau_load_dot = -(l * l * j * e_omega);

        self.omega_hat += omega_hat_dot * dt;
        self.tau_load_hat += tau_load_dot * dt;

        self.tau_load_hat
    }

    /// Return current estimated speed.
    pub fn omega_hat(&self) -> S {
        self.omega_hat
    }

    /// Return current estimated load torque.
    pub fn tau_load_hat(&self) -> S {
        self.tau_load_hat
    }

    /// Reset observer states to zero.
    pub fn reset(&mut self) {
        self.omega_hat = S::ZERO;
        self.tau_load_hat = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_friction_model_positive_speed() {
        let fm = FrictionModel::<f32>::new(0.5, 0.1, 0.01);
        // ω = 10 rad/s → τ = 0.5 + 0.1*10 = 1.5 N·m
        let tau = fm.torque(10.0);
        assert!((tau - 1.5_f32).abs() < 1e-5, "tau={tau}");
    }

    #[test]
    fn test_friction_model_negative_speed() {
        let fm = FrictionModel::<f32>::new(0.5, 0.1, 0.01);
        // ω = -10 → τ = -0.5 + 0.1*(-10) = -1.5
        let tau = fm.torque(-10.0);
        assert!((tau + 1.5_f32).abs() < 1e-5, "tau={tau}");
    }

    #[test]
    fn test_friction_model_stiction_region() {
        let fm = FrictionModel::<f32>::new(1.0, 0.0, 1.0);
        // ω = 0.5 (within stiction_thresh=1.0): τ = (1/1 + 0)*0.5 = 0.5
        let tau = fm.torque(0.5);
        assert!((tau - 0.5_f32).abs() < 1e-5, "tau={tau}");
    }

    #[test]
    fn test_load_observer_converges() {
        let fm = FrictionModel::<f32>::new(0.0, 0.0, 0.0);
        let mut obs = LoadObserver::<f32>::new(0.01, fm, 100.0);

        // True load torque = 1.0 N·m, applied torque = 1.0 (steady state)
        // The observer should converge: ω → 0, τ̂_load → 1.0
        // Apply u=1.0, but motor is stalled (omega_meas=0): observer integrates
        for _ in 0..2000 {
            obs.estimate(0.0, 1.0, 0.0001);
        }
        // After convergence τ̂_load should approach applied torque
        let tau_hat = obs.tau_load_hat();
        assert!(
            tau_hat > 0.5,
            "Observer should converge toward 1.0, got {tau_hat}"
        );
    }

    #[test]
    fn test_load_observer_reset() {
        let fm = FrictionModel::<f32>::new(0.0, 0.0, 0.0);
        let mut obs = LoadObserver::<f32>::new(0.01, fm, 10.0);
        for _ in 0..100 {
            obs.estimate(1.0, 0.0, 0.001);
        }
        obs.reset();
        assert_eq!(obs.omega_hat(), 0.0_f32);
        assert_eq!(obs.tau_load_hat(), 0.0_f32);
    }
}
