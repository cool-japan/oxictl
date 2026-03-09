use crate::antiwindup::aw_compensator::AntiWindupError;
/// Observer-Based Anti-Windup.
///
/// An N-th order state-feedback controller combined with a full-order
/// observer (Luenberger). The key anti-windup mechanism is that the
/// observer update uses the **actual saturated input** `v` rather than
/// the commanded (unsaturated) input `u_lin`.  This ensures the observer
/// state tracks the true plant state even when the actuator is saturated.
///
/// Discrete update (Euler):
/// ```text
/// u_lin  = -K · x̂ + n_gain · r
/// v      = clamp(u_lin, u_min, u_max)
/// innov  = y − C · x̂
/// x̂    += dt * (A · x̂ + B · v + L · innov)   ← v, not u_lin
/// ```
use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// ObserverAntiWindup<S, N>
// ---------------------------------------------------------------------------

/// Observer-based anti-windup for a SISO, N-th order plant.
///
/// The observer uses the saturated plant input `v` in the prediction step,
/// which prevents the simulated and real plant states from diverging during
/// actuator saturation.
#[derive(Debug, Clone)]
pub struct ObserverAntiWindup<S: ControlScalar, const N: usize> {
    /// Current state estimate x̂.
    x_hat: [S; N],
    /// Plant A matrix (N × N), row-major.
    a: [[S; N]; N],
    /// Plant B column vector (N × 1, scalar input).
    b: [S; N],
    /// Plant C row vector (1 × N, scalar output).
    c: [S; N],
    /// Observer gain vector L (N × 1).
    l: [S; N],
    /// State-feedback gain vector K (1 × N) — u = -K·x̂ + n_gain·r.
    k: [S; N],
    /// Reference pre-filter scalar gain.
    n_gain: S,
    /// Lower saturation limit.
    u_min: S,
    /// Upper saturation limit.
    u_max: S,
    /// True when output was saturated on the last update.
    saturated: bool,
    /// Sample period.
    dt: S,
}

impl<S: ControlScalar, const N: usize> ObserverAntiWindup<S, N> {
    /// Construct a new `ObserverAntiWindup`.
    ///
    /// # Arguments
    /// * `a`       – Plant A matrix (N × N), row-major.
    /// * `b`       – Plant B vector (N,).
    /// * `c`       – Plant C vector (N,).
    /// * `l`       – Observer gain vector (N,).
    /// * `k`       – State-feedback gain vector (N,).
    /// * `n_gain`  – Reference pre-filter gain.
    /// * `u_min`   – Lower saturation bound.
    /// * `u_max`   – Upper saturation bound.
    /// * `dt`      – Sample period (must be > 0).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a: [[S; N]; N],
        b: [S; N],
        c: [S; N],
        l: [S; N],
        k: [S; N],
        n_gain: S,
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
        Ok(Self {
            x_hat: [S::ZERO; N],
            a,
            b,
            c,
            l,
            k,
            n_gain,
            u_min,
            u_max,
            saturated: false,
            dt,
        })
    }

    /// Perform one discrete-time observer-based AW step.
    ///
    /// # Arguments
    /// * `r` – Reference signal.
    /// * `y` – Plant output measurement.
    ///
    /// # Returns
    /// Saturated control output `v`.
    pub fn update(&mut self, r: S, y: S) -> Result<S, AntiWindupError> {
        // u_lin = -K · x̂ + n_gain · r
        let mut u_lin = self.n_gain * r;
        for i in 0..N {
            u_lin -= self.k[i] * self.x_hat[i];
        }

        // v = sat(u_lin)
        let v = u_lin.clamp_val(self.u_min, self.u_max);
        self.saturated = (v - u_lin) * (v - u_lin) > S::EPSILON * S::EPSILON;

        // innov = y - C · x̂
        let mut c_xhat = S::ZERO;
        for i in 0..N {
            c_xhat += self.c[i] * self.x_hat[i];
        }
        let innov = y - c_xhat;

        // x̂ += dt * (A · x̂ + B · v + L · innov)
        // Compute A · x̂  (2D index requires range loop)
        let mut ax = [S::ZERO; N];
        #[allow(clippy::needless_range_loop)]
        for i in 0..N {
            for j in 0..N {
                ax[i] += self.a[i][j] * self.x_hat[j];
            }
        }

        #[allow(clippy::needless_range_loop)]
        for i in 0..N {
            let dx = ax[i] + self.b[i] * v + self.l[i] * innov;
            self.x_hat[i] += self.dt * dx;
        }

        Ok(v)
    }

    /// Read-only view of the current state estimate.
    #[inline]
    pub fn state_estimate(&self) -> &[S; N] {
        &self.x_hat
    }

    /// Returns `true` if the output was saturated on the last update.
    #[inline]
    pub fn is_saturated(&self) -> bool {
        self.saturated
    }

    /// Reset state estimate to a given initial condition.
    ///
    /// # Arguments
    /// * `x0` – Initial state estimate.
    pub fn reset(&mut self, x0: [S; N]) {
        self.x_hat = x0;
        self.saturated = false;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple 1st-order observer AW controller.
    ///
    /// Plant: A=[[a]], B=[b], C=[c].
    /// Observer gain: L=[l].
    /// State feedback: K=[k], n_gain.
    #[allow(clippy::too_many_arguments)]
    fn make_obs_aw(
        a: f64,
        b: f64,
        c: f64,
        l: f64,
        k: f64,
        n_gain: f64,
        u_min: f64,
        u_max: f64,
        dt: f64,
    ) -> ObserverAntiWindup<f64, 1> {
        ObserverAntiWindup::new([[a]], [b], [c], [l], [k], n_gain, u_min, u_max, dt).unwrap()
    }

    // -----------------------------------------------------------------------
    // 1. Observer uses v (not u_lin) in the update
    // -----------------------------------------------------------------------
    #[test]
    fn observer_uses_saturated_input_v() {
        // Plant: A=0, B=1, C=1.  K=1, n_gain=1. Limits [-1, 1].
        // Reference = 10 → u_lin = 1*10 - 1*x̂ will be >> u_max.
        // With observer using v = u_max = 1, state estimate advances differently
        // than if u_lin were used.

        let mut obs = make_obs_aw(0.0, 1.0, 1.0, 0.5, 1.0, 1.0, -1.0, 1.0, 0.01);

        // Step with large reference → saturates
        let v = obs.update(10.0, 0.0).unwrap();
        assert!((v - 1.0).abs() < 1e-12, "expected v=1.0, got {v}");
        assert!(obs.is_saturated());

        // x̂ advance: dt * (A*0 + B*v + L*(y - C*0))
        //           = 0.01 * (0 + 1*1 + 0.5*(0 - 0)) = 0.01
        let expected = 0.01_f64;
        assert!(
            (obs.state_estimate()[0] - expected).abs() < 1e-12,
            "x_hat={} expected={expected}",
            obs.state_estimate()[0]
        );
    }

    // -----------------------------------------------------------------------
    // 2. Saturation flag is set correctly
    // -----------------------------------------------------------------------
    #[test]
    fn observer_saturation_flag() {
        // Tiny reference → no saturation
        let mut obs = make_obs_aw(0.0, 1.0, 1.0, 0.5, 1.0, 1.0, -10.0, 10.0, 0.01);
        obs.update(0.5, 0.0).unwrap();
        assert!(!obs.is_saturated(), "should not be saturated");

        // Large reference → saturation
        obs.reset([0.0]);
        obs.update(100.0, 0.0).unwrap();
        assert!(obs.is_saturated(), "should be saturated");
    }

    // -----------------------------------------------------------------------
    // 3. Invalid parameter validation
    // -----------------------------------------------------------------------
    #[test]
    fn observer_invalid_params() {
        // u_min >= u_max
        let res = ObserverAntiWindup::<f64, 1>::new(
            [[0.0]],
            [1.0],
            [1.0],
            [0.5],
            [1.0],
            1.0,
            5.0,
            3.0,
            0.01,
        );
        assert!(
            matches!(res, Err(AntiWindupError::InvalidParameter)),
            "expected InvalidParameter for u_min >= u_max"
        );

        // dt <= 0
        let res2 = ObserverAntiWindup::<f64, 1>::new(
            [[0.0]],
            [1.0],
            [1.0],
            [0.5],
            [1.0],
            1.0,
            -10.0,
            10.0,
            0.0,
        );
        assert!(
            matches!(res2, Err(AntiWindupError::InvalidParameter)),
            "expected InvalidParameter for dt <= 0"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Reset restores initial condition
    // -----------------------------------------------------------------------
    #[test]
    fn observer_reset() {
        let mut obs = make_obs_aw(0.0, 1.0, 1.0, 0.5, 1.0, 1.0, -10.0, 10.0, 0.01);
        for _ in 0..20 {
            obs.update(5.0, 0.0).unwrap();
        }
        obs.reset([core::f64::consts::PI]);
        assert!(
            (obs.state_estimate()[0] - core::f64::consts::PI).abs() < 1e-12,
            "state after reset: {}",
            obs.state_estimate()[0]
        );
        assert!(!obs.is_saturated());
    }

    // -----------------------------------------------------------------------
    // 5. Unsaturated: observer state converges toward plant output
    // -----------------------------------------------------------------------
    #[test]
    fn observer_unsaturated_tracks_output() {
        // Stable plant: A=-1, B=1, C=1. Observer gain L=2.
        // K=2, n_gain=3. Limits wide: [-100, 100].
        let mut obs = make_obs_aw(-1.0, 1.0, 1.0, 2.0, 2.0, 3.0, -100.0, 100.0, 0.001);
        // Run 200 steps toward a unit step reference.
        // y is assumed 0 (open-loop; we just check observer doesn't blow up).
        for _ in 0..200 {
            let v = obs.update(1.0, 0.0).unwrap();
            assert!(v.abs() < 100.0, "output grew unbounded: {v}");
        }
        // x̂ should be a finite number.
        assert!(
            obs.state_estimate()[0].is_finite(),
            "x_hat diverged: {}",
            obs.state_estimate()[0]
        );
    }
}
