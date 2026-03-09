use crate::core::scalar::ControlScalar;

/// Hall sensor state (3 bits: H3, H2, H1).
pub type HallState = u8;

/// Decode Hall sensor state to sector index 0..5.
///
/// Standard 6-step commutation table:
/// Hall (H3H2H1) → sector → electrical angle (center)
///   5 (101) → 0 →   0°
///   1 (001) → 1 →  60°
///   3 (011) → 2 → 120°
///   2 (010) → 3 → 180°
///   6 (110) → 4 → 240°
///   4 (100) → 5 → 300°
///
/// Returns None for invalid states (0 or 7).
pub fn hall_to_sector(hall: HallState) -> Option<u8> {
    match hall & 0x07 {
        5 => Some(0),
        1 => Some(1),
        3 => Some(2),
        2 => Some(3),
        6 => Some(4),
        4 => Some(5),
        _ => None, // 0 (no sensor) or 7 (all on) are fault states
    }
}

/// Electrical angle at center of each Hall sector (radians, 0..2π).
pub fn sector_to_angle<S: ControlScalar>(sector: u8) -> S {
    let deg_per_sector = S::from_f64(60.0_f64.to_radians());
    S::from_f64(sector as f64) * deg_per_sector + S::from_f64(30.0_f64.to_radians())
}

/// Hall sensor decoder with speed estimation.
///
/// Decodes 3-bit Hall sensor output to rotor sector and angle.
/// Estimates rotational speed by measuring time between sector transitions.
#[derive(Debug, Clone, Copy)]
pub struct HallSensor<S: ControlScalar> {
    /// Current Hall state (H3H2H1 bitmask).
    pub hall: HallState,
    /// Current sector (0..5).
    pub sector: u8,
    /// Estimated electrical angle (rad), interpolated within sector.
    theta: S,
    /// Estimated electrical speed (rad/s), EMA filtered.
    omega: S,
    /// Time accumulator in current sector (s).
    sector_time: S,
    /// Speed filter coefficient.
    alpha: S,
    /// Number of pole pairs (for mechanical speed: omega_mech = omega_elec / p).
    pub pole_pairs: u8,
    /// Direction of rotation: +1 or -1.
    direction: i8,
    /// Previous sector for direction detection.
    prev_sector: u8,
}

impl<S: ControlScalar> HallSensor<S> {
    /// Create Hall sensor decoder.
    ///
    /// - `pole_pairs`: number of pole pairs (p), typically 2–10 for BLDC motors
    /// - `alpha`: speed EMA filter coefficient (0 < α ≤ 1)
    pub fn new(pole_pairs: u8, alpha: S) -> Self {
        Self {
            hall: 5,
            sector: 0,
            theta: S::ZERO,
            omega: S::ZERO,
            sector_time: S::ZERO,
            alpha,
            pole_pairs,
            direction: 1,
            prev_sector: 0,
        }
    }

    /// Update with new Hall state and elapsed time since last call.
    ///
    /// - `hall`: 3-bit Hall sensor reading (H3H2H1)
    /// - `dt`: elapsed time since last update (s)
    pub fn update(&mut self, hall: HallState, dt: S) {
        self.sector_time += dt;

        let new_sector = match hall_to_sector(hall) {
            Some(s) => s,
            None => return, // Fault state, ignore
        };

        if new_sector != self.sector {
            // Sector transition: update speed estimate
            if self.sector_time > S::ZERO {
                // One sector = 60° = π/3 electrical radians
                let pi_over_3 = S::PI / S::from_f64(3.0);
                let omega_raw = pi_over_3 / self.sector_time;

                // Determine direction (CW or CCW)
                // Forward: sector increments by 1 (mod 6)
                let expected_fwd = (self.sector + 1) % 6;
                self.direction = if new_sector == expected_fwd { 1 } else { -1 };

                let omega_signed = if self.direction > 0 {
                    omega_raw
                } else {
                    -omega_raw
                };
                self.omega += self.alpha * (omega_signed - self.omega);
            }
            self.prev_sector = self.sector;
            self.sector = new_sector;
            self.sector_time = S::ZERO;
        }

        self.hall = hall;

        // Interpolated angle within sector
        let sector_start = S::from_f64(self.sector as f64) * (S::PI / S::from_f64(3.0));
        let interp = if self.omega.abs() > S::ZERO && self.sector_time < S::from_f64(0.1) {
            (self.omega.abs() * self.sector_time).clamp_val(S::ZERO, S::PI / S::from_f64(3.0))
        } else {
            S::PI / S::from_f64(6.0) // Center of sector as fallback
        };
        self.theta = sector_start + interp;
    }

    /// Estimated electrical angle (rad), interpolated within sector.
    pub fn theta_e(&self) -> S {
        self.theta
    }

    /// Estimated electrical speed (rad/s).
    pub fn omega_e(&self) -> S {
        self.omega
    }

    /// Estimated mechanical speed (rad/s) = omega_e / pole_pairs.
    pub fn omega_mech(&self) -> S {
        if self.pole_pairs > 0 {
            self.omega / S::from_f64(self.pole_pairs as f64)
        } else {
            self.omega
        }
    }

    /// Direction: +1 = CW (forward), -1 = CCW (reverse).
    pub fn direction(&self) -> i8 {
        self.direction
    }

    /// Check for invalid Hall state (fault).
    pub fn is_fault(&self) -> bool {
        let h = self.hall & 0x07;
        h == 0 || h == 7
    }

    pub fn reset(&mut self) {
        self.hall = 5;
        self.sector = 0;
        self.theta = S::ZERO;
        self.omega = S::ZERO;
        self.sector_time = S::ZERO;
        self.direction = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn decode_all_valid_states() {
        let valid = [(5, 0), (1, 1), (3, 2), (2, 3), (6, 4), (4, 5)];
        for (hall, expected) in &valid {
            assert_eq!(hall_to_sector(*hall), Some(*expected), "hall={}", hall);
        }
    }

    #[test]
    fn invalid_states_return_none() {
        assert_eq!(hall_to_sector(0), None);
        assert_eq!(hall_to_sector(7), None);
    }

    #[test]
    fn sector_angles_spaced_60_degrees() {
        for i in 0u8..5 {
            let a0 = sector_to_angle::<f64>(i);
            let a1 = sector_to_angle::<f64>(i + 1);
            let diff = (a1 - a0).abs();
            assert!((diff - PI / 3.0).abs() < 1e-10, "diff={:.4}", diff);
        }
    }

    #[test]
    fn speed_estimation_from_transitions() {
        let mut hall = HallSensor::new(1_u8, 1.0_f64); // p=1, no filter
                                                       // Simulate 6 transitions at 1ms each → one electrical revolution in 6ms
                                                       // Speed = 2π / 6ms = 1047 rad/s
        let sequence = [5u8, 1, 3, 2, 6, 4, 5];
        let dt = 0.001_f64;

        for &h in &sequence {
            hall.update(h, dt);
        }

        // After all transitions, should estimate ~1047 rad/s
        assert!(
            hall.omega_e().abs() > 500.0,
            "omega={:.1} rad/s",
            hall.omega_e()
        );
    }

    #[test]
    fn direction_detection() {
        let mut hall = HallSensor::new(1_u8, 1.0_f64);
        // Forward sequence
        let fwd = [5u8, 1, 3, 2, 6, 4];
        for &h in &fwd {
            hall.update(h, 0.001);
        }
        assert_eq!(hall.direction(), 1, "Forward sequence should give +1");
    }

    #[test]
    fn fault_detection() {
        let hall = HallSensor::new(1_u8, 1.0_f64);
        // Only check is_fault() based on stored hall state (default=5, valid)
        assert!(!hall.is_fault());
    }
}
