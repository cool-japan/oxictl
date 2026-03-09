//! 3-DOF bicycle vehicle dynamics model.
//!
//! Implements a simplified bicycle model with linear Pacejka tire forces,
//! aerodynamic drag, and forward-Euler integration.
#![cfg(feature = "std")]

use crate::core::scalar::ControlScalar;

/// Simplified Pacejka tire model (linear cornering stiffness approximation).
#[derive(Debug, Clone, Copy)]
pub struct TireModel<S: ControlScalar> {
    /// Cornering stiffness (N/rad).
    pub cornering_stiffness: S,
    /// Peak friction coefficient.
    pub mu: S,
    /// Tire normal load (N).
    pub fz: S,
}

impl<S: ControlScalar> TireModel<S> {
    pub fn new(cornering_stiffness: S, mu: S, fz: S) -> Self {
        Self {
            cornering_stiffness,
            mu,
            fz,
        }
    }

    /// Lateral force = stiffness * slip_angle (linear region), clamped by friction limit.
    pub fn lateral_force(&self, slip_angle: S) -> S {
        let f_linear = self.cornering_stiffness * slip_angle;
        let f_max = self.mu * self.fz;
        f_linear.clamp_val(-f_max, f_max)
    }

    /// Longitudinal force with friction limit: F = mu * Fz * slip_ratio, clamped.
    pub fn longitudinal_force(&self, slip_ratio: S) -> S {
        let f_raw = self.mu * self.fz * slip_ratio;
        let f_max = self.mu * self.fz;
        f_raw.clamp_val(-f_max, f_max)
    }
}

/// 3-DOF bicycle model state: [x, y, yaw, vx, vy, yaw_rate].
#[derive(Debug, Clone, Copy)]
pub struct VehicleState<S: ControlScalar> {
    pub x: S,
    pub y: S,
    pub yaw: S,
    pub vx: S,
    pub vy: S,
    pub yaw_rate: S,
}

impl<S: ControlScalar> VehicleState<S> {
    pub fn zero() -> Self {
        Self {
            x: S::ZERO,
            y: S::ZERO,
            yaw: S::ZERO,
            vx: S::ZERO,
            vy: S::ZERO,
            yaw_rate: S::ZERO,
        }
    }
}

/// Vehicle dynamics parameters.
#[derive(Debug, Clone, Copy)]
pub struct VehicleParams<S: ControlScalar> {
    /// Vehicle mass (kg).
    pub mass: S,
    /// Yaw moment of inertia (kg*m^2).
    pub inertia_z: S,
    /// Distance from CG to front axle (m).
    pub lf: S,
    /// Distance from CG to rear axle (m).
    pub lr: S,
    /// Aerodynamic drag coefficient.
    pub cd: S,
    /// Frontal area (m^2).
    pub frontal_area: S,
    /// Air density (kg/m^3).
    pub air_density: S,
}

/// 3-DOF bicycle vehicle dynamics simulator.
///
/// Uses forward-Euler integration of the bicycle model equations:
///   m * ax = Fx_f + Fx_r - F_drag
///   m * ay = Fy_f + Fy_r
///   Iz * yaw_ddot = lf * Fy_f - lr * Fy_r
pub struct VehicleDynamics<S: ControlScalar> {
    pub state: VehicleState<S>,
    pub params: VehicleParams<S>,
    pub front_tire: TireModel<S>,
    pub rear_tire: TireModel<S>,
}

impl<S: ControlScalar> VehicleDynamics<S> {
    pub fn new(
        params: VehicleParams<S>,
        front_tire: TireModel<S>,
        rear_tire: TireModel<S>,
    ) -> Self {
        Self {
            state: VehicleState::zero(),
            params,
            front_tire,
            rear_tire,
        }
    }

    /// Aerodynamic drag force (N), opposing velocity.
    fn aero_drag(&self) -> S {
        let half = S::HALF;
        let vx2 = self.state.vx * self.state.vx;
        half * self.params.air_density * self.params.cd * self.params.frontal_area * vx2
    }

    /// Total speed (m/s).
    pub fn speed(&self) -> S {
        let vx2 = self.state.vx * self.state.vx;
        let vy2 = self.state.vy * self.state.vy;
        (vx2 + vy2).sqrt()
    }

    /// Simulate one step.
    ///
    /// - `throttle`: normalized drive force request [0, 1]
    /// - `brake`: normalized braking force request [0, 1]
    /// - `steer`: front wheel steering angle (rad)
    /// - `dt`: time step (s)
    pub fn step(&mut self, throttle: S, brake: S, steer: S, dt: S) {
        let s = &self.state;
        let p = &self.params;

        // Front slip angle (small angle approximation)
        // alpha_f = steer - (vy + lf * yaw_rate) / vx
        let vx_safe = if s.vx.abs() < S::from_f64(0.1) {
            S::from_f64(0.1)
        } else {
            s.vx
        };
        let alpha_f = steer - (s.vy + p.lf * s.yaw_rate) / vx_safe;
        let alpha_r = -(s.vy - p.lr * s.yaw_rate) / vx_safe;

        // Tire forces in tire frame
        let fy_f = self.front_tire.lateral_force(alpha_f);
        let fy_r = self.rear_tire.lateral_force(alpha_r);

        // Drive/brake longitudinal force (applied at rear axle for RWD)
        let f_drive = throttle * self.rear_tire.mu * self.rear_tire.fz;
        let f_brake_mag = brake
            * (self.front_tire.mu * self.front_tire.fz + self.rear_tire.mu * self.rear_tire.fz);
        let fx_net = f_drive - f_brake_mag;

        // Aerodynamic drag (opposes forward motion)
        let f_drag = self.aero_drag();

        // Equations of motion (body frame)
        let ax = (fx_net - f_drag) / p.mass + s.vy * s.yaw_rate;
        let ay = (fy_f + fy_r) / p.mass - s.vx * s.yaw_rate;
        let yaw_ddot = (p.lf * fy_f - p.lr * fy_r) / p.inertia_z;

        // Forward Euler integration
        let new_vx = s.vx + ax * dt;
        let new_vy = s.vy + ay * dt;
        let new_yaw_rate = s.yaw_rate + yaw_ddot * dt;
        let new_yaw = s.yaw + s.yaw_rate * dt;

        // Global position update
        let cos_yaw = s.yaw.cos();
        let sin_yaw = s.yaw.sin();
        let new_x = s.x + (s.vx * cos_yaw - s.vy * sin_yaw) * dt;
        let new_y = s.y + (s.vx * sin_yaw + s.vy * cos_yaw) * dt;

        self.state = VehicleState {
            x: new_x,
            y: new_y,
            yaw: new_yaw,
            vx: new_vx.clamp_val(S::ZERO, S::from_f64(100.0)),
            vy: new_vy,
            yaw_rate: new_yaw_rate,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vehicle() -> VehicleDynamics<f64> {
        let params = VehicleParams {
            mass: 1500.0,
            inertia_z: 2500.0,
            lf: 1.2,
            lr: 1.4,
            cd: 0.3,
            frontal_area: 2.2,
            air_density: 1.225,
        };
        let front_tire = TireModel::new(80_000.0, 0.9, 7357.5);
        let rear_tire = TireModel::new(80_000.0, 0.9, 7357.5);
        VehicleDynamics::new(params, front_tire, rear_tire)
    }

    #[test]
    fn test_vehicle_accelerates_from_rest() {
        let mut veh = make_vehicle();
        // Apply throttle for 1 second with small steps
        for _ in 0..100 {
            veh.step(0.3, 0.0, 0.0, 0.01);
        }
        // Should have accelerated
        assert!(veh.state.vx > 0.5, "vx={}", veh.state.vx);
        assert!(veh.state.x > 0.1, "x={}", veh.state.x);
    }

    #[test]
    fn test_tire_lateral_force_linear() {
        let tire = TireModel::<f64>::new(80_000.0, 1.0, 5000.0);
        // Small slip angle: linear region
        let f = tire.lateral_force(0.05);
        let expected = 80_000.0 * 0.05;
        assert!((f - expected).abs() < 1.0, "f={f}");
    }

    #[test]
    fn test_tire_lateral_force_saturates() {
        let tire = TireModel::<f64>::new(80_000.0, 1.0, 5000.0);
        // Large slip: should saturate at mu * fz
        let f = tire.lateral_force(1.0);
        assert!((f - 5000.0).abs() < 1.0, "f={f}");
    }

    #[test]
    fn test_aero_drag_increases_with_speed() {
        let mut veh = make_vehicle();
        veh.state.vx = 10.0;
        let drag_low = veh.aero_drag();
        veh.state.vx = 30.0;
        let drag_high = veh.aero_drag();
        assert!(
            drag_high > drag_low * 8.0,
            "drag_low={drag_low} drag_high={drag_high}"
        );
    }

    #[test]
    fn test_yaw_rate_from_steering() {
        let mut veh = make_vehicle();
        veh.state.vx = 10.0;
        let steer = 0.05_f64;
        for _ in 0..50 {
            veh.step(0.0, 0.0, steer, 0.01);
        }
        // Should develop a nonzero yaw rate
        assert!(
            veh.state.yaw_rate.abs() > 1e-3,
            "yaw_rate={}",
            veh.state.yaw_rate
        );
    }
}
