use crate::antiwindup::aw_compensator::AntiWindupError;
/// Back-Calculation (Conditioning) Anti-Windup and Tracking-Mode Anti-Windup.
///
/// Two implementations:
/// - `ConditioningController`: PI with back-calculation (Hanus conditioning).
///   The integrator is driven back toward feasibility by a gain `ka` on the
///   saturation error `v - u_unsat`.
/// - `TrackingAntiWindup`: PID with tracking-mode AW. A tracking time constant
///   `tt` determines how fast the integrator is driven toward the saturated
///   value: `integrator += dt * (ki * e + (v - u_unsat) / tt)`.
use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// ConditioningController
// ---------------------------------------------------------------------------

/// PI controller with back-calculation (conditioning-technique) anti-windup.
///
/// Discrete update:
/// ```text
/// e         = r - y
/// u_unsat   = kp * e + integrator
/// v         = clamp(u_unsat, u_min, u_max)
/// integrator += dt * (ki * e + ka * (v - u_unsat))
/// ```
///
/// The gain `ka` (back-calculation gain) controls how aggressively the
/// integrator is returned to the feasible range. A common choice is
/// `ka = 1 / Ti` where `Ti = kp / ki` is the integral time constant.
#[derive(Debug, Clone, Copy)]
pub struct ConditioningController<S: ControlScalar> {
    /// Proportional gain.
    kp: S,
    /// Integral gain.
    ki: S,
    /// Back-calculation anti-windup gain (must be > 0).
    ka: S,
    /// Integrator state.
    integrator: S,
    /// Lower saturation limit.
    u_min: S,
    /// Upper saturation limit.
    u_max: S,
    /// Sample period.
    dt: S,
}

impl<S: ControlScalar> ConditioningController<S> {
    /// Construct a new `ConditioningController`.
    ///
    /// # Arguments
    /// * `kp` – Proportional gain (must be > 0).
    /// * `ki` – Integral gain (must be ≥ 0).
    /// * `ka` – Back-calculation gain (must be > 0).
    /// * `u_min`, `u_max` – Saturation limits (`u_min < u_max`).
    /// * `dt` – Sample period (must be > 0).
    pub fn new(kp: S, ki: S, ka: S, u_min: S, u_max: S, dt: S) -> Result<Self, AntiWindupError> {
        if u_min >= u_max {
            return Err(AntiWindupError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(AntiWindupError::InvalidParameter);
        }
        if kp <= S::ZERO {
            return Err(AntiWindupError::InvalidParameter);
        }
        if ki < S::ZERO {
            return Err(AntiWindupError::InvalidParameter);
        }
        if ka <= S::ZERO {
            return Err(AntiWindupError::InvalidParameter);
        }
        Ok(Self {
            kp,
            ki,
            ka,
            integrator: S::ZERO,
            u_min,
            u_max,
            dt,
        })
    }

    /// Perform one discrete-time control update.
    ///
    /// # Arguments
    /// * `r` – Reference (setpoint).
    /// * `y` – Plant output (measurement).
    ///
    /// # Returns
    /// Saturated control output `v`.
    pub fn update(&mut self, r: S, y: S) -> Result<S, AntiWindupError> {
        let e = r - y;
        let u_unsat = self.kp * e + self.integrator;
        let v = u_unsat.clamp_val(self.u_min, self.u_max);
        let sat_error = v - u_unsat;
        // Integrator with back-calculation AW
        self.integrator += self.dt * (self.ki * e + self.ka * sat_error);
        Ok(v)
    }

    /// Reset integrator to zero.
    pub fn reset(&mut self) {
        self.integrator = S::ZERO;
    }

    /// Read current integrator value.
    #[inline]
    pub fn integrator(&self) -> S {
        self.integrator
    }
}

// ---------------------------------------------------------------------------
// TrackingAntiWindup
// ---------------------------------------------------------------------------

/// PID controller with tracking-mode anti-windup.
///
/// The derivative term uses output-derivative (derivative on measurement only)
/// to avoid derivative kick on setpoint changes:
/// ```text
/// e         = r - y
/// dy        = y - y_prev
/// u_unsat   = kp * e + integrator - kd * dy / dt
/// v         = clamp(u_unsat, u_min, u_max)
/// integrator += dt * (ki * e + (v - u_unsat) / tt)
/// y_prev    = y
/// ```
///
/// The tracking time constant `tt` (T_t) sets how fast the integrator
/// tracks the saturated output. Typically `sqrt(Ti * Td) ≤ tt ≤ Ti`.
#[derive(Debug, Clone, Copy)]
pub struct TrackingAntiWindup<S: ControlScalar> {
    /// Proportional gain.
    kp: S,
    /// Integral gain.
    ki: S,
    /// Derivative gain.
    kd: S,
    /// Integrator state.
    integrator: S,
    /// Previous measurement (for derivative-on-measurement).
    y_prev: S,
    /// Lower saturation limit.
    u_min: S,
    /// Upper saturation limit.
    u_max: S,
    /// Tracking time constant (must be > 0).
    tt: S,
    /// Sample period.
    dt: S,
}

impl<S: ControlScalar> TrackingAntiWindup<S> {
    /// Construct a new `TrackingAntiWindup` PID controller.
    ///
    /// # Arguments
    /// * `kp` – Proportional gain (must be > 0).
    /// * `ki` – Integral gain (must be ≥ 0).
    /// * `kd` – Derivative gain (must be ≥ 0).
    /// * `tt` – Tracking time constant (must be > 0).
    /// * `u_min`, `u_max` – Saturation limits (`u_min < u_max`).
    /// * `dt` – Sample period (must be > 0).
    pub fn new(
        kp: S,
        ki: S,
        kd: S,
        tt: S,
        u_min: S,
        u_max: S,
        dt: S,
    ) -> Result<Self, AntiWindupError> {
        if u_min >= u_max {
            return Err(AntiWindupError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(AntiWindupError::InvalidParameter);
        }
        if tt <= S::ZERO {
            return Err(AntiWindupError::InvalidParameter);
        }
        if kp <= S::ZERO {
            return Err(AntiWindupError::InvalidParameter);
        }
        if ki < S::ZERO || kd < S::ZERO {
            return Err(AntiWindupError::InvalidParameter);
        }
        Ok(Self {
            kp,
            ki,
            kd,
            integrator: S::ZERO,
            y_prev: S::ZERO,
            u_min,
            u_max,
            tt,
            dt,
        })
    }

    /// Perform one discrete-time PID+tracking-AW step.
    ///
    /// # Arguments
    /// * `r` – Reference (setpoint).
    /// * `y` – Plant output (measurement).
    ///
    /// # Returns
    /// Saturated control output `v`.
    pub fn update(&mut self, r: S, y: S) -> Result<S, AntiWindupError> {
        let e = r - y;
        // Derivative on measurement to avoid setpoint kick
        let dy = y - self.y_prev;
        let deriv = if self.dt > S::ZERO {
            self.kd * dy / self.dt
        } else {
            S::ZERO
        };
        let u_unsat = self.kp * e + self.integrator - deriv;
        let v = u_unsat.clamp_val(self.u_min, self.u_max);
        let sat_error = v - u_unsat;
        // Integrator with tracking AW
        self.integrator += self.dt * (self.ki * e + sat_error / self.tt);
        self.y_prev = y;
        Ok(v)
    }

    /// Reset integrator and stored measurement to zero.
    pub fn reset(&mut self) {
        self.integrator = S::ZERO;
        self.y_prev = S::ZERO;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ConditioningController tests
    // -----------------------------------------------------------------------

    #[test]
    fn conditioning_unsaturated_integrator_grows() {
        // With no saturation, integrator should grow each step.
        let mut ctrl =
            ConditioningController::<f64>::new(1.0, 1.0, 1.0, -100.0, 100.0, 0.01).unwrap();
        let v0 = ctrl.update(1.0, 0.0).unwrap();
        let i0 = ctrl.integrator();
        let _v1 = ctrl.update(1.0, 0.0).unwrap();
        let i1 = ctrl.integrator();
        assert!(v0.abs() < 100.0, "should not be saturated: {v0}");
        assert!(i1 > i0, "integrator should grow: {i0} → {i1}");
    }

    #[test]
    fn conditioning_saturated_back_calc_bounds_integrator() {
        // Large setpoint step → saturation; back-calculation should bound integrator.
        let mut ctrl = ConditioningController::<f64>::new(1.0, 1.0, 2.0, -1.0, 1.0, 0.01).unwrap();
        for _ in 0..500 {
            let _ = ctrl.update(100.0, 0.0).unwrap();
        }
        assert!(
            ctrl.integrator().abs() < 100.0,
            "integrator not bounded: {}",
            ctrl.integrator()
        );
    }

    #[test]
    fn conditioning_ka_must_be_positive() {
        let res = ConditioningController::<f64>::new(1.0, 1.0, 0.0, -10.0, 10.0, 0.01);
        assert!(
            matches!(res, Err(AntiWindupError::InvalidParameter)),
            "expected InvalidParameter for ka=0"
        );
        let res2 = ConditioningController::<f64>::new(1.0, 1.0, -1.0, -10.0, 10.0, 0.01);
        assert!(
            matches!(res2, Err(AntiWindupError::InvalidParameter)),
            "expected InvalidParameter for ka<0"
        );
    }

    #[test]
    fn conditioning_reset() {
        let mut ctrl =
            ConditioningController::<f64>::new(1.0, 1.0, 1.0, -10.0, 10.0, 0.01).unwrap();
        for _ in 0..20 {
            let _ = ctrl.update(5.0, 0.0).unwrap();
        }
        ctrl.reset();
        assert_eq!(ctrl.integrator(), 0.0);
    }

    #[test]
    fn conditioning_invalid_limits() {
        let res = ConditioningController::<f64>::new(1.0, 1.0, 1.0, 5.0, 3.0, 0.01);
        assert!(
            matches!(res, Err(AntiWindupError::InvalidParameter)),
            "expected InvalidParameter for u_min > u_max"
        );
    }

    // -----------------------------------------------------------------------
    // TrackingAntiWindup tests
    // -----------------------------------------------------------------------

    #[test]
    fn tracking_aw_saturated_bounded() {
        // Persistent large setpoint: integrator must be bounded by tracking AW.
        let mut ctrl = TrackingAntiWindup::<f64>::new(1.0, 1.0, 0.0, 0.1, -1.0, 1.0, 0.01).unwrap();
        for _ in 0..500 {
            let _ = ctrl.update(50.0, 0.0).unwrap();
        }
        assert!(
            ctrl.integrator.abs() < 200.0,
            "integrator not bounded: {}",
            ctrl.integrator
        );
    }

    #[test]
    fn tracking_aw_reset() {
        let mut ctrl =
            TrackingAntiWindup::<f64>::new(1.0, 1.0, 0.0, 0.5, -10.0, 10.0, 0.01).unwrap();
        for _ in 0..20 {
            let _ = ctrl.update(5.0, 0.0).unwrap();
        }
        ctrl.reset();
        assert_eq!(ctrl.integrator, 0.0);
        assert_eq!(ctrl.y_prev, 0.0);
    }

    #[test]
    fn tracking_aw_invalid_tt() {
        let res = TrackingAntiWindup::<f64>::new(1.0, 1.0, 0.0, 0.0, -10.0, 10.0, 0.01);
        assert!(
            matches!(res, Err(AntiWindupError::InvalidParameter)),
            "expected InvalidParameter for tt=0"
        );
    }

    #[test]
    fn tracking_aw_output_clamped() {
        let mut ctrl =
            TrackingAntiWindup::<f64>::new(10.0, 0.0, 0.0, 1.0, -2.0, 2.0, 0.01).unwrap();
        // kp*e = 10*5 = 50, but clamped to 2
        let v = ctrl.update(5.0, 0.0).unwrap();
        assert!((v - 2.0).abs() < 1e-12, "v={v}");
    }
}
