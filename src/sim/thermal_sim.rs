use crate::core::traits::Plant;

/// First-order thermal plant model.
///
/// Dynamics: dT/dt = (1/tau) * (K * u - (T - T_ambient))
///
/// - T: temperature (state)
/// - u: heater power input (0..1)
/// - K: heater gain (degrees per unit input at steady state)
/// - tau: thermal time constant (seconds)
/// - T_ambient: ambient temperature
pub struct ThermalPlant {
    temperature: f64,
    tau: f64,
    gain: f64,
    ambient: f64,
}

impl ThermalPlant {
    pub fn new(initial_temp: f64, tau: f64, gain: f64, ambient: f64) -> Self {
        Self {
            temperature: initial_temp,
            tau,
            gain,
            ambient,
        }
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Analytical steady-state temperature for a given constant input.
    pub fn steady_state(&self, u: f64) -> f64 {
        self.ambient + self.gain * u
    }

    /// Inject a disturbance (additive temperature change).
    pub fn add_disturbance(&mut self, delta_t: f64) {
        self.temperature += delta_t;
    }
}

impl Plant<f64> for ThermalPlant {
    fn step(&mut self, u: f64, dt: f64) {
        // Forward Euler: T += dt * dT/dt
        let dt_dt = (self.gain * u - (self.temperature - self.ambient)) / self.tau;
        self.temperature += dt_dt * dt;
    }

    fn output(&self) -> f64 {
        self.temperature
    }

    fn state(&self) -> &[f64] {
        core::slice::from_ref(&self.temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_conditions() {
        let plant = ThermalPlant::new(25.0, 10.0, 100.0, 25.0);
        assert_eq!(plant.output(), 25.0);
    }

    #[test]
    fn step_response_direction() {
        let mut plant = ThermalPlant::new(25.0, 10.0, 100.0, 25.0);
        plant.step(1.0, 0.01);
        // With heater on, temperature should increase
        assert!(plant.output() > 25.0);
    }

    #[test]
    fn steady_state_analytical() {
        let plant = ThermalPlant::new(25.0, 10.0, 100.0, 25.0);
        // At u=1.0, steady state = 25 + 100*1 = 125
        assert_eq!(plant.steady_state(1.0), 125.0);
    }

    #[test]
    fn converges_to_steady_state() {
        let mut plant = ThermalPlant::new(25.0, 10.0, 100.0, 25.0);
        let u = 0.5;
        let expected = plant.steady_state(u);
        let dt = 0.01;

        // Run for 5*tau = 50 seconds (99.3% of final value)
        for _ in 0..5000 {
            plant.step(u, dt);
        }

        assert!(
            (plant.output() - expected).abs() < 1.0,
            "Should converge to {}, got {}",
            expected,
            plant.output()
        );
    }

    #[test]
    fn time_constant_accuracy() {
        // After 1 tau, should reach ~63.2% of final value
        let mut plant = ThermalPlant::new(0.0, 1.0, 100.0, 0.0);
        let dt = 0.001;
        let steps = 1000; // 1 second = 1 tau

        for _ in 0..steps {
            plant.step(1.0, dt);
        }

        let expected_at_tau = 100.0 * (1.0 - (-1.0_f64).exp()); // ~63.2
        assert!(
            (plant.output() - expected_at_tau).abs() < 1.0,
            "At t=tau, expected ~{:.1}, got {:.1}",
            expected_at_tau,
            plant.output()
        );
    }

    #[test]
    fn disturbance_injection() {
        let mut plant = ThermalPlant::new(25.0, 10.0, 100.0, 25.0);
        plant.add_disturbance(-5.0);
        assert_eq!(plant.output(), 20.0);
    }

    #[test]
    fn cooling_when_above_equilibrium() {
        let mut plant = ThermalPlant::new(100.0, 10.0, 50.0, 25.0);
        // With u=0, equilibrium is 25. Starting at 100, should cool down.
        plant.step(0.0, 0.01);
        assert!(plant.output() < 100.0);
    }
}
