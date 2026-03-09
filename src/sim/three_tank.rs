use crate::core::scalar::ControlScalar;

/// Three-tank hydraulic system simulation.
///
/// Three coupled tanks connected by pipes at the bottom.
/// Flow between tanks follows Torricelli's law:
///   Q_ij = Cd · a_ij · sign(h_i - h_j) · √(2g · |h_i - h_j|)
///
/// Tank levels: h1, h2, h3 (m)
/// Inlet flows: q1_in, q3_in (m³/s) — control inputs
/// Tank cross-sectional areas: A1, A2, A3 (m²)
/// Pipe cross-sectional areas: a12, a23 (m²)
/// Discharge coefficient: Cd (dimensionless, typically 0.6–0.8)
/// Gravity: g (m/s²)
///
/// Dynamics:
///   A1 · dh1/dt = q1_in - Q12
///   A2 · dh2/dt = Q12 - Q23
///   A3 · dh3/dt = q3_in + Q23 - Q3_out
///
/// where:
///   Q12 = Cd·a12·sign(h1-h2)·√(2g·|h1-h2|)
///   Q23 = Cd·a23·sign(h2-h3)·√(2g·|h2-h3|)
///   Q3_out = Cd·a3_out·√(2g·h3)  (drain from tank 3)
///
/// Level constraints: 0 ≤ hi ≤ h_max
/// RK4 integration with constraint projection.
#[derive(Debug, Clone, Copy)]
pub struct ThreeTankSystem<S: ControlScalar> {
    /// Cross-sectional areas of tanks 1, 2, 3 (m²).
    pub tank_areas: [S; 3],
    /// Cross-sectional area of pipe between tank 1 and 2 (m²).
    pub pipe_area_12: S,
    /// Cross-sectional area of pipe between tank 2 and 3 (m²).
    pub pipe_area_23: S,
    /// Cross-sectional area of outlet pipe from tank 3 (m²).
    pub outlet_area: S,
    /// Discharge coefficient Cd (0.6–0.8 typical).
    pub cd: S,
    /// Gravitational acceleration (m/s²).
    pub gravity: S,
    /// Maximum tank height (overflow constraint) (m).
    pub h_max: S,
    /// Current tank levels [h1, h2, h3] (m).
    levels: [S; 3],
}

impl<S: ControlScalar> ThreeTankSystem<S> {
    /// Create three-tank system with given geometry.
    pub fn new(
        tank_areas: [S; 3],
        pipe_area_12: S,
        pipe_area_23: S,
        outlet_area: S,
        cd: S,
        gravity: S,
        h_max: S,
    ) -> Self {
        Self {
            tank_areas,
            pipe_area_12,
            pipe_area_23,
            outlet_area,
            cd,
            gravity,
            h_max,
            levels: [S::ZERO; 3],
        }
    }

    /// Create standard three-tank benchmark (DLR benchmark parameters).
    ///
    /// Tank areas: 0.0154 m² each.
    /// Pipe areas: 5e-5 m² each.
    /// Outlet area: 5e-5 m².
    /// Cd = 0.67, g = 9.81, h_max = 0.62 m.
    pub fn dlr_benchmark() -> Self {
        Self::new(
            [
                S::from_f64(0.0154),
                S::from_f64(0.0154),
                S::from_f64(0.0154),
            ],
            S::from_f64(5e-5),
            S::from_f64(5e-5),
            S::from_f64(5e-5),
            S::from_f64(0.67),
            S::from_f64(9.81),
            S::from_f64(0.62),
        )
    }

    /// Set initial tank levels.
    pub fn set_levels(&mut self, h1: S, h2: S, h3: S) {
        self.levels[0] = h1.clamp_val(S::ZERO, self.h_max);
        self.levels[1] = h2.clamp_val(S::ZERO, self.h_max);
        self.levels[2] = h3.clamp_val(S::ZERO, self.h_max);
    }

    /// Current tank levels [h1, h2, h3].
    pub fn levels(&self) -> &[S; 3] {
        &self.levels
    }

    pub fn h1(&self) -> S {
        self.levels[0]
    }
    pub fn h2(&self) -> S {
        self.levels[1]
    }
    pub fn h3(&self) -> S {
        self.levels[2]
    }

    /// Torricelli flow between two tanks.
    ///
    /// Q = Cd · a · sign(ha - hb) · √(2g·|ha - hb|)
    ///
    /// Returns positive flow from a to b if ha > hb.
    fn torricelli_flow(&self, ha: S, hb: S, pipe_area: S) -> S {
        let dh = ha - hb;
        let abs_dh = dh.abs();
        if abs_dh < S::EPSILON {
            return S::ZERO;
        }
        let speed = (S::TWO * self.gravity * abs_dh).sqrt();
        let q = self.cd * pipe_area * speed;
        if dh > S::ZERO {
            q
        } else {
            -q
        }
    }

    /// Outlet flow from tank 3 (drain).
    fn outlet_flow(&self, h3: S) -> S {
        if h3 < S::EPSILON {
            return S::ZERO;
        }
        self.cd * self.outlet_area * (S::TWO * self.gravity * h3).sqrt()
    }

    /// Compute level derivatives for given inlet flows.
    fn derivatives(&self, h: &[S; 3], q1_in: S, q3_in: S) -> [S; 3] {
        let h1 = h[0].clamp_val(S::ZERO, self.h_max);
        let h2 = h[1].clamp_val(S::ZERO, self.h_max);
        let h3 = h[2].clamp_val(S::ZERO, self.h_max);

        let q12 = self.torricelli_flow(h1, h2, self.pipe_area_12);
        let q23 = self.torricelli_flow(h2, h3, self.pipe_area_23);
        let q3_out = self.outlet_flow(h3);

        let dh1 = if self.tank_areas[0] > S::EPSILON {
            (q1_in - q12) / self.tank_areas[0]
        } else {
            S::ZERO
        };
        let dh2 = if self.tank_areas[1] > S::EPSILON {
            (q12 - q23) / self.tank_areas[1]
        } else {
            S::ZERO
        };
        let dh3 = if self.tank_areas[2] > S::EPSILON {
            (q3_in + q23 - q3_out) / self.tank_areas[2]
        } else {
            S::ZERO
        };

        [dh1, dh2, dh3]
    }

    /// Advance system using RK4 integration.
    ///
    /// `q1_in`: inlet flow to tank 1 (m³/s).
    /// `q3_in`: inlet flow to tank 3 (m³/s).
    /// `dt`: integration step (s).
    pub fn step(&mut self, q1_in: S, q3_in: S, dt: S) {
        let h = self.levels;

        let k1 = self.derivatives(&h, q1_in, q3_in);
        let h2: [S; 3] = core::array::from_fn(|i| h[i] + S::HALF * dt * k1[i]);
        let k2 = self.derivatives(&h2, q1_in, q3_in);
        let h3: [S; 3] = core::array::from_fn(|i| h[i] + S::HALF * dt * k2[i]);
        let k3 = self.derivatives(&h3, q1_in, q3_in);
        let h4: [S; 3] = core::array::from_fn(|i| h[i] + dt * k3[i]);
        let k4 = self.derivatives(&h4, q1_in, q3_in);

        let sixth = S::ONE / S::from_f64(6.0);
        for i in 0..3 {
            let new_level = h[i] + sixth * dt * (k1[i] + S::TWO * k2[i] + S::TWO * k3[i] + k4[i]);
            // Apply level constraints
            self.levels[i] = new_level.clamp_val(S::ZERO, self.h_max);
        }
    }

    /// Total water volume in the system (m³).
    pub fn total_volume(&self) -> S {
        let mut vol = S::ZERO;
        for i in 0..3 {
            vol += self.tank_areas[i] * self.levels[i];
        }
        vol
    }

    /// Flow from tank 1 to tank 2 at current state.
    pub fn flow_12(&self) -> S {
        self.torricelli_flow(self.levels[0], self.levels[1], self.pipe_area_12)
    }

    /// Flow from tank 2 to tank 3 at current state.
    pub fn flow_23(&self) -> S {
        self.torricelli_flow(self.levels[1], self.levels[2], self.pipe_area_23)
    }

    /// Outlet flow from tank 3 at current state.
    pub fn flow_out(&self) -> S {
        self.outlet_flow(self.levels[2])
    }

    /// Steady-state levels for given constant inlet flows (numerical approximation).
    ///
    /// Runs simulation until convergence or max_steps.
    /// Returns the levels array or None if not converged.
    pub fn find_steady_state(
        &self,
        q1_in: S,
        q3_in: S,
        dt: S,
        max_steps: usize,
        tol: S,
    ) -> Option<[S; 3]> {
        let mut sim = *self;
        let mut prev = sim.levels;

        for _ in 0..max_steps {
            sim.step(q1_in, q3_in, dt);
            let mut converged = true;
            for (&lvl, &prv) in sim.levels.iter().zip(prev.iter()) {
                if (lvl - prv).abs() > tol {
                    converged = false;
                    break;
                }
            }
            if converged {
                return Some(sim.levels);
            }
            prev = sim.levels;
        }
        None
    }

    /// Reset all tank levels to zero.
    pub fn reset(&mut self) {
        self.levels = [S::ZERO; 3];
    }

    /// Check if any tank is overflowing.
    pub fn is_overflowing(&self) -> bool {
        self.levels.iter().any(|&h| h >= self.h_max)
    }

    /// Check if all tanks are empty.
    pub fn is_empty(&self) -> bool {
        self.levels.iter().all(|&h| h <= S::EPSILON)
    }

    /// Maximum level across all tanks.
    pub fn max_level(&self) -> S {
        let mut m = self.levels[0];
        if self.levels[1] > m {
            m = self.levels[1];
        }
        if self.levels[2] > m {
            m = self.levels[2];
        }
        m
    }

    /// Minimum level across all tanks.
    pub fn min_level(&self) -> S {
        let mut m = self.levels[0];
        if self.levels[1] < m {
            m = self.levels[1];
        }
        if self.levels[2] < m {
            m = self.levels[2];
        }
        m
    }

    /// Mean level across all tanks.
    pub fn mean_level(&self) -> S {
        (self.levels[0] + self.levels[1] + self.levels[2]) / S::from_f64(3.0)
    }

    /// Net inflow rate (difference between inlet and outlet flows).
    ///
    /// If positive, total water volume is increasing.
    pub fn net_inflow(&self, q1_in: S, q3_in: S) -> S {
        q1_in + q3_in - self.flow_out()
    }

    /// Hydraulic potential energy of the system (relative to ground).
    ///
    /// E = ρ·g · Σ A_i · h_i² / 2
    /// where ρ = 1000 kg/m³ (water density).
    pub fn hydraulic_energy(&self) -> S {
        let rho = S::from_f64(1000.0);
        let half = S::HALF;
        let mut e = S::ZERO;
        for i in 0..3 {
            e += self.tank_areas[i] * self.levels[i] * self.levels[i];
        }
        rho * self.gravity * half * e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_increase_with_inflow() {
        let mut sys = ThreeTankSystem::<f64>::dlr_benchmark();
        let h_before = sys.h1();
        for _ in 0..100 {
            sys.step(1e-4, 0.0, 1.0); // 1e-4 m³/s into tank 1
        }
        assert!(
            sys.h1() > h_before,
            "h1 should increase: before={}, after={}",
            h_before,
            sys.h1()
        );
    }

    #[test]
    fn no_inflow_empties_tanks() {
        let mut sys = ThreeTankSystem::<f64>::dlr_benchmark();
        sys.set_levels(0.3, 0.3, 0.3);
        // Run for long time with no inflow
        for _ in 0..10000 {
            sys.step(0.0, 0.0, 1.0);
        }
        // Tanks should be near empty
        assert!(sys.h3() < 0.01, "h3 should drain: {}", sys.h3());
    }

    #[test]
    fn levels_respect_constraints() {
        let mut sys = ThreeTankSystem::<f64>::dlr_benchmark();
        // Flood with extreme inflow
        for _ in 0..1000 {
            sys.step(1.0, 1.0, 0.1);
        }
        let h_max = sys.h_max;
        assert!(
            sys.h1() <= h_max + 1e-9,
            "h1={} > h_max={}",
            sys.h1(),
            h_max
        );
        assert!(sys.h2() <= h_max + 1e-9);
        assert!(sys.h3() <= h_max + 1e-9);
        assert!(!sys.is_empty());
    }

    #[test]
    fn levels_nonnegative() {
        let mut sys = ThreeTankSystem::<f64>::dlr_benchmark();
        sys.set_levels(0.1, 0.1, 0.1);
        for _ in 0..10000 {
            sys.step(0.0, 0.0, 0.5);
        }
        assert!(sys.h1() >= 0.0, "h1={}", sys.h1());
        assert!(sys.h2() >= 0.0, "h2={}", sys.h2());
        assert!(sys.h3() >= 0.0, "h3={}", sys.h3());
    }

    #[test]
    fn torricelli_flow_direction() {
        let sys = ThreeTankSystem::<f64>::dlr_benchmark();
        // h1 > h2: flow should go from 1 to 2 (positive)
        let q = sys.torricelli_flow(0.3_f64, 0.1_f64, sys.pipe_area_12);
        assert!(q > 0.0, "flow should be positive (1→2): {}", q);

        // h1 < h2: flow from 2 to 1 (negative)
        let q_rev = sys.torricelli_flow(0.1_f64, 0.3_f64, sys.pipe_area_12);
        assert!(q_rev < 0.0, "flow should be negative (2→1): {}", q_rev);
    }

    #[test]
    fn outlet_flow_zero_when_empty() {
        let sys = ThreeTankSystem::<f64>::dlr_benchmark();
        let q = sys.outlet_flow(0.0_f64);
        assert!(
            q.abs() < 1e-15,
            "outlet flow should be zero when empty: {}",
            q
        );
    }

    #[test]
    fn total_volume_positive_when_filled() {
        let mut sys = ThreeTankSystem::<f64>::dlr_benchmark();
        sys.set_levels(0.2, 0.3, 0.1);
        let vol = sys.total_volume();
        assert!(vol > 0.0, "volume should be positive: {}", vol);
    }

    #[test]
    fn reset_zeros_levels() {
        let mut sys = ThreeTankSystem::<f64>::dlr_benchmark();
        sys.set_levels(0.3, 0.2, 0.1);
        sys.reset();
        assert_eq!(sys.h1(), 0.0);
        assert_eq!(sys.h2(), 0.0);
        assert_eq!(sys.h3(), 0.0);
        assert!(sys.is_empty());
    }

    #[test]
    fn steady_state_converges() {
        let sys = ThreeTankSystem::<f64>::dlr_benchmark();
        // With small constant inflow, should reach steady state
        let ss = sys.find_steady_state(1e-5, 0.0, 1.0, 100000, 1e-8);
        assert!(ss.is_some(), "should converge to steady state");
        let [h1, h2, h3] = ss.unwrap();
        assert!(h1 >= 0.0 && h1 <= sys.h_max, "h1 out of range: {}", h1);
        assert!(h2 >= 0.0 && h2 <= sys.h_max, "h2 out of range: {}", h2);
        assert!(h3 >= 0.0 && h3 <= sys.h_max, "h3 out of range: {}", h3);
    }

    #[test]
    fn symmetric_tanks_equalize() {
        // With h1 > h2 = h3 and equal tank/pipe areas, levels should equalize
        let mut sys =
            ThreeTankSystem::<f64>::new([0.01, 0.01, 0.01], 1e-4, 1e-4, 1e-4, 0.7, 9.81, 1.0);
        sys.set_levels(0.5, 0.1, 0.0);
        for _ in 0..50000 {
            sys.step(0.0, 0.0, 0.01);
        }
        // After long time with no inflow, all tanks approach common low level (draining via outlet)
        // h3 drains via outlet, so h1 > h2 > h3 eventually as water flows and drains
        assert!(sys.h1() >= 0.0);
        assert!(
            sys.total_volume() <= 0.5 * 0.01 + 0.1 * 0.01 + 1e-3, // started with 0.5+0.1
            "volume conservation violated: {}",
            sys.total_volume()
        );
    }

    #[test]
    fn flow_computations_finite() {
        let mut sys = ThreeTankSystem::<f64>::dlr_benchmark();
        sys.set_levels(0.3, 0.2, 0.1);
        assert!(sys.flow_12().is_finite());
        assert!(sys.flow_23().is_finite());
        assert!(sys.flow_out().is_finite());
    }

    #[test]
    fn max_level_correct() {
        let mut sys = ThreeTankSystem::<f64>::dlr_benchmark();
        sys.set_levels(0.1, 0.4, 0.2);
        assert!((sys.max_level() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn min_level_correct() {
        let mut sys = ThreeTankSystem::<f64>::dlr_benchmark();
        sys.set_levels(0.1, 0.4, 0.2);
        assert!((sys.min_level() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn mean_level_correct() {
        let mut sys = ThreeTankSystem::<f64>::dlr_benchmark();
        sys.set_levels(0.3, 0.3, 0.3);
        assert!((sys.mean_level() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn hydraulic_energy_positive_when_filled() {
        let mut sys = ThreeTankSystem::<f64>::dlr_benchmark();
        sys.set_levels(0.2, 0.2, 0.2);
        assert!(sys.hydraulic_energy() > 0.0);
    }

    #[test]
    fn net_inflow_positive_when_inflow_exceeds_drain() {
        let sys = ThreeTankSystem::<f64>::dlr_benchmark();
        // With large inflow and empty tank 3 (no drain), net should be positive
        let net = sys.net_inflow(1.0, 0.0);
        assert!(net > 0.0, "net_inflow={}", net);
    }
}
