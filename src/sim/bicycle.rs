use crate::core::scalar::ControlScalar;

/// Error type for bicycle model operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BicycleError {
    /// A model parameter is invalid (e.g. zero wheelbase).
    InvalidParameter,
    /// Longitudinal speed is zero (dynamic model requires vx > 0).
    ZeroLongitudinalSpeed,
}

impl core::fmt::Display for BicycleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BicycleError::InvalidParameter => write!(f, "invalid bicycle parameter"),
            BicycleError::ZeroLongitudinalSpeed => {
                write!(f, "longitudinal speed must be nonzero for dynamic model")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kinematic bicycle model
// ---------------------------------------------------------------------------
/// Kinematic bicycle model for path planning and low-speed manoeuvres.
///
/// State vector: `[x, y, psi, delta]`
/// - `x`     : longitudinal position (m)
/// - `y`     : lateral position (m)
/// - `psi`   : heading angle (rad)
/// - `delta` : front-wheel steering angle (rad)
///
/// Continuous-time kinematics (single-track, rear-axle reference point):
/// ```text
///   ẋ     = v · cos(ψ)
///   ẏ     = v · sin(ψ)
///   ψ̇     = v · tan(δ) / L
///   δ̇     = delta_dot          (steering rate, direct input)
/// ```
///
/// Integration: first-order Euler with fixed timestep `dt`.
#[derive(Debug, Clone, Copy)]
pub struct KinematicBicycle<S: ControlScalar> {
    /// Current state `[x, y, psi, delta]`.
    pub state: [S; 4],
    /// Wheelbase L (m) — distance from rear to front axle.
    pub wheelbase: S,
    /// Integration timestep (s).
    pub dt: S,
}

impl<S: ControlScalar> KinematicBicycle<S> {
    /// Construct a kinematic bicycle model.
    ///
    /// # Parameters
    /// - `wheelbase`: distance between axles (m); must be positive.
    /// - `dt`       : integration step (s); must be positive.
    /// - `x0`, `y0` : initial position (m); `psi` and `delta` start at zero.
    pub fn new(wheelbase: S, dt: S, x0: S, y0: S) -> Result<Self, BicycleError> {
        if wheelbase <= S::ZERO {
            return Err(BicycleError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(BicycleError::InvalidParameter);
        }
        Ok(Self {
            state: [x0, y0, S::ZERO, S::ZERO],
            wheelbase,
            dt,
        })
    }

    /// Compute state derivatives.
    ///
    /// `s = [x, y, psi, delta]`, `v` = longitudinal speed (m/s),
    /// `delta_dot` = steering rate (rad/s).
    fn derivatives(s: &[S; 4], v: S, delta_dot: S, wheelbase: S) -> [S; 4] {
        let psi = s[2];
        let delta = s[3];
        let dx = v * psi.cos();
        let dy = v * psi.sin();
        let dpsi = v * delta.tan() / wheelbase;
        let ddelta = delta_dot;
        [dx, dy, dpsi, ddelta]
    }

    /// Advance the model by one timestep using Euler integration.
    ///
    /// Returns the new state `[x, y, psi, delta]`.
    pub fn step(&mut self, v: S, delta_dot: S) -> Result<[S; 4], BicycleError> {
        let d = Self::derivatives(&self.state, v, delta_dot, self.wheelbase);
        let dt = self.dt;
        for (s, di) in self.state.iter_mut().zip(d.iter()) {
            *s += dt * *di;
        }
        Ok(self.state)
    }

    /// Return the current state `[x, y, psi, delta]`.
    pub fn state(&self) -> [S; 4] {
        self.state
    }
}

// ---------------------------------------------------------------------------
// Dynamic bicycle model
// ---------------------------------------------------------------------------
/// Dynamic bicycle model for lateral vehicle control at higher speeds.
///
/// State vector: `[y, psi, vy, r]`
/// - `y`   : lateral position (m)
/// - `psi` : heading/yaw angle (rad)
/// - `vy`  : lateral (side-slip) velocity (m/s)
/// - `r`   : yaw rate (rad/s)
///
/// Continuous-time equations (linear lateral dynamics, constant longitudinal speed `vx`):
/// ```text
///   ẏ    = vy + vx · ψ
///   ψ̇    = r
///   m · v̇y = −2(Cf+Cr)/vx · vy − [2(Cf·lf − Cr·lr)/vx + m·vx] · r + 2Cf·δ
///   Iz · ṙ = −2(Cf·lf−Cr·lr)/vx · vy − 2(Cf·lf²+Cr·lr²)/vx · r + 2Cf·lf·δ
/// ```
///
/// Integration: 4th-order Runge-Kutta with fixed timestep `dt`.
#[derive(Debug, Clone, Copy)]
pub struct DynamicBicycle<S: ControlScalar> {
    /// Current state `[y, psi, vy, r]`.
    pub state: [S; 4],
    /// Vehicle mass (kg).
    pub m: S,
    /// Yaw moment of inertia (kg·m²).
    pub iz: S,
    /// Distance from CoG to front axle (m).
    pub lf: S,
    /// Distance from CoG to rear axle (m).
    pub lr: S,
    /// Front cornering stiffness (N/rad).
    pub cf: S,
    /// Rear cornering stiffness (N/rad).
    pub cr: S,
    /// Longitudinal speed (m/s); held constant during simulation.
    pub vx: S,
    /// Integration timestep (s).
    pub dt: S,
}

impl<S: ControlScalar> DynamicBicycle<S> {
    /// Construct a dynamic bicycle model.
    ///
    /// All stiffness, mass, inertia, and distance parameters must be strictly
    /// positive.  `vx` must be nonzero (the model is singular at zero speed).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: S,
        iz: S,
        lf: S,
        lr: S,
        cf: S,
        cr: S,
        vx: S,
        dt: S,
    ) -> Result<Self, BicycleError> {
        if m <= S::ZERO || iz <= S::ZERO {
            return Err(BicycleError::InvalidParameter);
        }
        if lf <= S::ZERO || lr <= S::ZERO {
            return Err(BicycleError::InvalidParameter);
        }
        if cf <= S::ZERO || cr <= S::ZERO {
            return Err(BicycleError::InvalidParameter);
        }
        if vx.abs() < S::EPSILON {
            return Err(BicycleError::ZeroLongitudinalSpeed);
        }
        if dt <= S::ZERO {
            return Err(BicycleError::InvalidParameter);
        }
        Ok(Self {
            state: [S::ZERO; 4],
            m,
            iz,
            lf,
            lr,
            cf,
            cr,
            vx,
            dt,
        })
    }

    /// Compute lateral dynamics derivatives for state `s = [y, psi, vy, r]`
    /// and steering input `delta`.
    fn derivatives(&self, s: &[S; 4], delta: S) -> [S; 4] {
        let psi = s[1];
        let vy = s[2];
        let r = s[3];

        let two = S::TWO;
        let vx = self.vx;

        let cf_plus_cr = self.cf + self.cr;
        let cf_lf_minus_cr_lr = self.cf * self.lf - self.cr * self.lr;
        let cf_lf2_plus_cr_lr2 = self.cf * self.lf * self.lf + self.cr * self.lr * self.lr;

        // dy/dt = vy + vx * psi
        let dy = vy + vx * psi;

        // dpsi/dt = r
        let dpsi = r;

        // m · dvy/dt = -2(Cf+Cr)/vx · vy - (2(Cf·lf-Cr·lr)/vx + m·vx)·r + 2Cf·delta
        let dvy = (-(two * cf_plus_cr / vx) * vy
            - (two * cf_lf_minus_cr_lr / vx + self.m * vx) * r
            + two * self.cf * delta)
            / self.m;

        // Iz · dr/dt = -2(Cf·lf-Cr·lr)/vx · vy - 2(Cf·lf²+Cr·lr²)/vx · r + 2Cf·lf·delta
        let dr = (-(two * cf_lf_minus_cr_lr / vx) * vy - (two * cf_lf2_plus_cr_lr2 / vx) * r
            + two * self.cf * self.lf * delta)
            / self.iz;

        [dy, dpsi, dvy, dr]
    }

    /// Advance the model by one timestep using RK4 integration.
    ///
    /// Returns the new state `[y, psi, vy, r]`.
    pub fn step(&mut self, delta: S) -> Result<[S; 4], BicycleError> {
        let s = self.state;
        let dt = self.dt;

        let k1 = self.derivatives(&s, delta);
        let s2: [S; 4] = core::array::from_fn(|i| s[i] + S::HALF * dt * k1[i]);
        let k2 = self.derivatives(&s2, delta);
        let s3: [S; 4] = core::array::from_fn(|i| s[i] + S::HALF * dt * k2[i]);
        let k3 = self.derivatives(&s3, delta);
        let s4: [S; 4] = core::array::from_fn(|i| s[i] + dt * k3[i]);
        let k4 = self.derivatives(&s4, delta);

        let sixth = S::ONE / S::from_f64(6.0);
        for i in 0..4 {
            self.state[i] += sixth * dt * (k1[i] + S::TWO * k2[i] + S::TWO * k3[i] + k4[i]);
        }
        Ok(self.state)
    }

    /// Return the current state `[y, psi, vy, r]`.
    pub fn state(&self) -> [S; 4] {
        self.state
    }

    /// Lateral position error — returns state component `y`.
    pub fn lateral_error(&self) -> S {
        self.state[0]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // --- KinematicBicycle ---------------------------------------------------

    #[test]
    fn kinematic_zero_input_straight_line() {
        // v=1 m/s, delta_dot=0, initial heading=0 → x increases, y stays 0
        let mut bike = KinematicBicycle::<f64>::new(2.5, 0.01, 0.0, 0.0).expect("valid params");
        for _ in 0..100 {
            bike.step(1.0, 0.0).expect("step ok");
        }
        let s = bike.state();
        // After 1 s at 1 m/s → x ≈ 1 m
        assert!(s[0] > 0.9, "x should increase: {}", s[0]);
        assert!(s[1].abs() < 1e-10, "y should stay 0: {}", s[1]);
        assert!(s[2].abs() < 1e-10, "psi should stay 0: {}", s[2]);
    }

    #[test]
    fn kinematic_turning_radius() {
        // Constant speed and constant steering → circular arc.
        // After one complete revolution the vehicle should return near its origin.
        let l = 2.5_f64;
        let delta = 0.1_f64; // rad — moderate steering angle
        let v = 5.0_f64;
        let dt = 0.001_f64;

        let mut bike = KinematicBicycle::<f64>::new(l, dt, 0.0, 0.0).expect("valid params");
        // Pre-set the steering angle (no steering-rate input needed)
        bike.state[3] = delta;

        let r_expected = l / delta.tan();
        let period = 2.0 * core::f64::consts::PI * r_expected / v;
        let steps = (period / dt) as usize;
        for _ in 0..steps {
            bike.step(v, 0.0).expect("step ok");
        }
        let s = bike.state();
        let dist = (s[0] * s[0] + s[1] * s[1]).sqrt();
        // Allow 5% of radius as closure error (Euler accumulation)
        assert!(
            dist < r_expected * 0.05,
            "after full circle dist={:.3}, expected < {:.3}",
            dist,
            r_expected * 0.05
        );
    }

    #[test]
    fn kinematic_steering_rate_changes_heading() {
        // Positive delta_dot → steering angle grows → vehicle curves left
        let mut bike = KinematicBicycle::<f64>::new(2.5, 0.01, 0.0, 0.0).expect("valid params");
        for _ in 0..50 {
            bike.step(1.0, 0.02).expect("step ok");
        }
        let s = bike.state();
        // steering angle should be ~0.02*50*0.01 = 0.01 rad, heading should grow
        assert!(s[3] > 0.0, "steering angle should be positive: {}", s[3]);
        assert!(s[2] > 0.0, "heading should be positive: {}", s[2]);
        assert!(s[1] > 0.0, "y should drift left: {}", s[1]);
    }

    #[test]
    fn kinematic_invalid_params() {
        assert!(KinematicBicycle::<f64>::new(0.0, 0.01, 0.0, 0.0).is_err());
        assert!(KinematicBicycle::<f64>::new(-1.0, 0.01, 0.0, 0.0).is_err());
        assert!(KinematicBicycle::<f64>::new(2.5, 0.0, 0.0, 0.0).is_err());
        assert!(KinematicBicycle::<f64>::new(2.5, -0.01, 0.0, 0.0).is_err());
    }

    // --- DynamicBicycle -----------------------------------------------------

    fn make_dynamic() -> DynamicBicycle<f64> {
        // Typical mid-size passenger car parameters
        DynamicBicycle::<f64>::new(
            1500.0,   // m   kg
            2500.0,   // iz  kg·m²
            1.2,      // lf  m
            1.4,      // lr  m
            80_000.0, // cf  N/rad
            80_000.0, // cr  N/rad
            20.0,     // vx  m/s
            0.001,    // dt  s
        )
        .expect("valid params")
    }

    #[test]
    fn dynamic_zero_input_decays() {
        // Non-zero initial vy and r with zero steering → system should damp
        let mut bike = make_dynamic();
        bike.state[2] = 1.0; // vy = 1 m/s
        bike.state[3] = 0.1; // r  = 0.1 rad/s
        let initial_energy = bike.state[2].powi(2) + bike.state[3].powi(2);
        for _ in 0..2000 {
            bike.step(0.0).expect("step ok");
        }
        let final_energy = bike.state[2].powi(2) + bike.state[3].powi(2);
        assert!(
            final_energy < initial_energy,
            "lateral energy should decay: initial={:.4}, final={:.4}",
            initial_energy,
            final_energy
        );
    }

    #[test]
    fn dynamic_step_steer_response() {
        // Step steering input → vy and r become nonzero
        let mut bike = make_dynamic();
        let delta = 0.02_f64; // 0.02 rad
        for _ in 0..500 {
            bike.step(delta).expect("step ok");
        }
        let s = bike.state();
        assert!(
            s[2].abs() > 1e-6,
            "vy should be nonzero after step steer: {}",
            s[2]
        );
        assert!(
            s[3].abs() > 1e-6,
            "r should be nonzero after step steer: {}",
            s[3]
        );
    }

    #[test]
    fn dynamic_lateral_error_accessor() {
        let mut bike = make_dynamic();
        bike.step(0.01).expect("step ok");
        assert!(
            (bike.lateral_error() - bike.state()[0]).abs() < f64::EPSILON,
            "lateral_error() must equal state[0]: {} vs {}",
            bike.lateral_error(),
            bike.state()[0]
        );
    }

    #[test]
    fn dynamic_invalid_params() {
        // zero vx
        assert!(DynamicBicycle::<f64>::new(
            1500.0, 2500.0, 1.2, 1.4, 80_000.0, 80_000.0, 0.0, 0.001
        )
        .is_err());
        // zero mass
        assert!(
            DynamicBicycle::<f64>::new(0.0, 2500.0, 1.2, 1.4, 80_000.0, 80_000.0, 20.0, 0.001)
                .is_err()
        );
        // negative lf
        assert!(DynamicBicycle::<f64>::new(
            1500.0, 2500.0, -1.0, 1.4, 80_000.0, 80_000.0, 20.0, 0.001
        )
        .is_err());
        // zero dt
        assert!(DynamicBicycle::<f64>::new(
            1500.0, 2500.0, 1.2, 1.4, 80_000.0, 80_000.0, 20.0, 0.0
        )
        .is_err());
        // zero cornering stiffness
        assert!(
            DynamicBicycle::<f64>::new(1500.0, 2500.0, 1.2, 1.4, 0.0, 80_000.0, 20.0, 0.001)
                .is_err()
        );
    }
}
