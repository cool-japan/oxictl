use crate::core::scalar::ControlScalar;

/// Error type for vehicle platoon operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlatoonError {
    /// A constructor parameter is invalid.
    InvalidParameter,
    /// The platoon has zero vehicles (N = 0).
    EmptyPlatoon,
}

impl core::fmt::Display for PlatoonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PlatoonError::InvalidParameter => write!(f, "invalid platoon parameter"),
            PlatoonError::EmptyPlatoon => write!(f, "platoon must have at least one vehicle"),
        }
    }
}

/// N-vehicle longitudinal platoon with PD spacing control.
///
/// # Vehicle dynamics (first-order speed model)
/// Each follower `i` (1 ≤ i < N) obeys:
/// ```text
///   ẋᵢ = −a · xᵢ + a · uᵢ          (speed dynamics, time constant 1/a)
/// ```
/// where `xᵢ` is the speed (m/s) and `uᵢ` is the desired speed command.
///
/// # Lead vehicle (i = 0)
/// Speed is set exogenously each step: `x₀ = v_lead`.
///
/// # Spacing policy
/// Two policies are supported via the `headway` parameter:
/// - **Constant Distance (CD)**: `headway = 0`; desired gap = `desired_spacing`.
/// - **Constant Time Headway (CTH)**: `headway > 0`; desired gap = `desired_spacing + headway · xᵢ`.
///
/// # Spacing error and PD control
/// ```text
///   eᵢ      = (pos_{i-1} − posᵢ) − d_des − headway · xᵢ
///   ė̇ᵢ     = xᵢ₋₁ − xᵢ
///   uᵢ      = kp · eᵢ + kd · ė̇ᵢ
/// ```
///
/// # Integration
/// Euler with fixed timestep `dt`.  Positions are integrated from speeds each step.
///
/// # Const generic `N`
/// The number of vehicles is fixed at compile time.  `N = 0` is rejected at
/// runtime (returns [`PlatoonError::EmptyPlatoon`]).
#[derive(Debug, Clone, Copy)]
pub struct VehiclePlatoon<S: ControlScalar, const N: usize> {
    /// Current speed of each vehicle (m/s).
    speeds: [S; N],
    /// Current position of each vehicle (m), increasing for forward motion.
    positions: [S; N],
    /// First-order vehicle bandwidth (1/τ, s⁻¹); must be positive.
    a: S,
    /// Proportional spacing gain.
    kp: S,
    /// Derivative spacing gain.
    kd: S,
    /// Nominal inter-vehicle spacing (m); must be non-negative.
    desired_spacing: S,
    /// Time-headway coefficient (s); 0 for constant-distance policy.
    headway: S,
    /// Integration timestep (s).
    dt: S,
}

impl<S: ControlScalar, const N: usize> VehiclePlatoon<S, N> {
    /// Construct a platoon initialised at equilibrium.
    ///
    /// All vehicles start at `v_lead_0` m/s, equally spaced by `d_des` m.
    /// Vehicle 0 is at position 0; vehicle `i` is at `−i · d_des`.
    ///
    /// # Constraints
    /// - `N ≥ 1`
    /// - `a > 0`, `dt > 0`, `d_des ≥ 0`, `headway ≥ 0`
    pub fn new(
        a: S,
        kp: S,
        kd: S,
        d_des: S,
        headway: S,
        dt: S,
        v_lead_0: S,
    ) -> Result<Self, PlatoonError> {
        if N == 0 {
            return Err(PlatoonError::EmptyPlatoon);
        }
        if a <= S::ZERO {
            return Err(PlatoonError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(PlatoonError::InvalidParameter);
        }
        if d_des < S::ZERO {
            return Err(PlatoonError::InvalidParameter);
        }
        if headway < S::ZERO {
            return Err(PlatoonError::InvalidParameter);
        }
        // Initialise all vehicles at equilibrium: same speed, spaced by d_des.
        let mut speeds = [v_lead_0; N];
        let mut positions = [S::ZERO; N];
        for i in 0..N {
            speeds[i] = v_lead_0;
            positions[i] = -S::from_f64(i as f64) * d_des;
        }
        Ok(Self {
            speeds,
            positions,
            a,
            kp,
            kd,
            desired_spacing: d_des,
            headway,
            dt,
        })
    }

    /// Advance the platoon by one timestep.
    ///
    /// - `v_lead` : exogenous speed command for the lead vehicle (vehicle 0).
    ///
    /// Returns an array of spacing errors `[e₁, e₂, …, e_{N-1}, 0]`.
    /// The last entry (index 0 / lead) is always zero.
    pub fn step(&mut self, v_lead: S) -> Result<[S; N], PlatoonError> {
        // Snapshot positions before any update so that spacing errors are computed
        // consistently (all positions from the same time instant).
        let pos_prev = self.positions;

        // --- 1. Lead vehicle: speed is set directly; integrate position.
        self.speeds[0] = v_lead;
        self.positions[0] += self.dt * v_lead;

        let mut errors = [S::ZERO; N];

        // --- 2. Compute spacing errors and PD commands for followers using
        //        positions from *before* this step (pos_prev), ensuring a
        //        consistent Euler discretisation with no integration-order bias.
        let mut speed_commands = [S::ZERO; N];
        speed_commands[0] = v_lead;

        for i in 1..N {
            let gap = pos_prev[i - 1] - pos_prev[i];
            let d_desired = self.desired_spacing + self.headway * self.speeds[i];
            let e_i = gap - d_desired;
            let e_dot_i = self.speeds[i - 1] - self.speeds[i];
            // Feedforward: command equals the preceding vehicle's speed plus
            // the PD correction so that at equilibrium (e=0, ė=0) the command
            // equals the current speed and the first-order model holds speed.
            let u_i = self.speeds[i - 1] + self.kp * e_i + self.kd * e_dot_i;
            speed_commands[i] = u_i;
            errors[i] = e_i;
        }

        // --- 3. Integrate follower speeds and positions (Euler).
        #[allow(clippy::needless_range_loop)]
        for i in 1..N {
            let dxi = -self.a * self.speeds[i] + self.a * speed_commands[i];
            self.speeds[i] += self.dt * dxi;
            self.positions[i] += self.dt * self.speeds[i];
        }

        Ok(errors)
    }

    /// Reference to the current speed array.
    pub fn speeds(&self) -> &[S; N] {
        &self.speeds
    }

    /// Reference to the current position array.
    pub fn positions(&self) -> &[S; N] {
        &self.positions
    }

    /// Heuristic string-stability check.
    ///
    /// Clones `self`, applies a speed perturbation of `perturbation_amp` to the
    /// lead vehicle for 10 steps, then runs 200 settling steps.  Returns `true`
    /// if the maximum absolute spacing error among all followers is strictly less
    /// than `perturbation_amp` (errors do not amplify downstream).
    ///
    /// This is a necessary (not sufficient) condition for string stability.
    pub fn is_string_stable(&self, perturbation_amp: S) -> bool {
        if N < 2 {
            // Single-vehicle platoon — trivially stable (no followers).
            return true;
        }
        // Clone to avoid mutating self.
        let mut sim = *self;

        // Nominal lead speed (first vehicle's current speed).
        let v_nominal = sim.speeds[0];
        let v_perturb = v_nominal + perturbation_amp;

        // Apply perturbation for 10 steps.
        for _ in 0..10 {
            let _ = sim.step(v_perturb);
        }
        // Return to nominal for 200 settling steps.
        for _ in 0..200 {
            let _ = sim.step(v_nominal);
        }

        // Measure maximum absolute spacing error among followers.
        let mut max_err = S::ZERO;
        for i in 1..N {
            let gap = sim.positions[i - 1] - sim.positions[i];
            let d_desired = sim.desired_spacing + sim.headway * sim.speeds[i];
            let abs_err = (gap - d_desired).abs();
            if abs_err > max_err {
                max_err = abs_err;
            }
        }
        max_err < perturbation_amp
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a 4-vehicle platoon with typical CD-policy parameters.
    fn make_platoon() -> VehiclePlatoon<f64, 4> {
        VehiclePlatoon::<f64, 4>::new(
            1.0,  // a   — vehicle bandwidth (s⁻¹)
            0.5,  // kp
            1.0,  // kd
            10.0, // d_des (m)
            0.0,  // headway=0 → constant-distance policy
            0.01, // dt (s)
            20.0, // v_lead_0 (m/s)
        )
        .expect("valid params")
    }

    #[test]
    fn equilibrium_spacing_holds() {
        // At equilibrium all vehicles share the same speed and are spaced d_des apart.
        // Running the platoon at constant lead speed should keep errors near zero.
        let mut p = make_platoon();
        let mut max_err = 0.0_f64;
        for _ in 0..500 {
            let errs = p.step(20.0).expect("step ok");
            for &e in errs.iter().skip(1) {
                if e.abs() > max_err {
                    max_err = e.abs();
                }
            }
        }
        assert!(
            max_err < 1e-6,
            "spacing errors should stay near zero at equilibrium: max={:.2e}",
            max_err
        );
    }

    #[test]
    fn lead_speed_step_followers_track() {
        // Lead speed jumps from 20 → 25 m/s; followers should eventually match.
        let mut p = make_platoon();
        // Let the system settle first
        for _ in 0..200 {
            p.step(20.0).expect("ok");
        }
        // Step change in lead speed
        for _ in 0..2000 {
            p.step(25.0).expect("ok");
        }
        let speeds = p.speeds();
        for (i, &s) in speeds.iter().enumerate() {
            assert!(
                (s - 25.0).abs() < 1.0,
                "vehicle {} speed should track lead (25 m/s): got {:.3}",
                i,
                s
            );
        }
    }

    #[test]
    fn spacing_errors_bounded() {
        // After a transient perturbation spacing errors should not grow unboundedly.
        let mut p = make_platoon();
        // Apply a 5 m/s speed perturbation for 50 steps, then return to nominal.
        for _ in 0..50 {
            p.step(25.0).expect("ok");
        }
        let mut max_err_after = 0.0_f64;
        for _ in 0..3000 {
            let errs = p.step(20.0).expect("ok");
            for &e in errs.iter().skip(1) {
                if e.abs() > max_err_after {
                    max_err_after = e.abs();
                }
            }
        }
        // Errors must be bounded (< 50 m is a very conservative bound)
        assert!(
            max_err_after < 50.0,
            "spacing errors should stay bounded: max={:.3}",
            max_err_after
        );
    }

    #[test]
    fn invalid_params_err() {
        // a = 0 → InvalidParameter
        assert!(VehiclePlatoon::<f64, 4>::new(0.0, 0.5, 1.0, 10.0, 0.0, 0.01, 20.0).is_err());
        // d_des < 0 → InvalidParameter
        assert!(VehiclePlatoon::<f64, 4>::new(1.0, 0.5, 1.0, -1.0, 0.0, 0.01, 20.0).is_err());
        // dt = 0 → InvalidParameter
        assert!(VehiclePlatoon::<f64, 4>::new(1.0, 0.5, 1.0, 10.0, 0.0, 0.0, 20.0).is_err());
        // headway < 0 → InvalidParameter
        assert!(VehiclePlatoon::<f64, 4>::new(1.0, 0.5, 1.0, 10.0, -0.5, 0.01, 20.0).is_err());
    }

    #[test]
    fn cth_policy_spacing_grows_with_speed() {
        // With CTH policy (headway > 0), desired gap = d_des + h*speed.
        // At equilibrium higher speed → larger gaps between vehicles.
        let mut p_fast =
            VehiclePlatoon::<f64, 3>::new(1.0, 0.5, 1.0, 5.0, 0.5, 0.01, 30.0).expect("ok");
        let mut p_slow =
            VehiclePlatoon::<f64, 3>::new(1.0, 0.5, 1.0, 5.0, 0.5, 0.01, 10.0).expect("ok");
        // Settle both platoons
        for _ in 0..3000 {
            p_fast.step(30.0).expect("ok");
            p_slow.step(10.0).expect("ok");
        }
        let gap_fast = p_fast.positions()[0] - p_fast.positions()[1];
        let gap_slow = p_slow.positions()[0] - p_slow.positions()[1];
        assert!(
            gap_fast > gap_slow,
            "CTH: fast platoon gap ({:.2}) should exceed slow platoon gap ({:.2})",
            gap_fast,
            gap_slow
        );
    }
}
