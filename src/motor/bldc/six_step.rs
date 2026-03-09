use crate::core::scalar::ControlScalar;

/// Hall sensor state (bits: H3, H2, H1).
pub type HallState = u8;

/// Six-step commutation output for BLDC motor.
/// Each phase can be: High (PWM), Low (GND), or Off (high-Z).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseState {
    High,
    Low,
    Off,
}

/// Six-step commutation table (hall sensor → phase states).
/// Standard BLDC commutation for clockwise rotation.
/// Index = hall state (1..=6), [phase_a, phase_b, phase_c].
///
/// Hall: 1=001, 2=010, 3=011, 4=100, 5=101, 6=110
const COMMUTATION_TABLE: [[PhaseState; 3]; 8] = {
    use PhaseState::*;
    [
        [Off, Off, Off],  // 0: invalid
        [High, Low, Off], // 1: H=001
        [Off, High, Low], // 2: H=010
        [High, Off, Low], // 3: H=011
        [Low, Off, High], // 4: H=100
        [Low, High, Off], // 5: H=101
        [Off, Low, High], // 6: H=110
        [Off, Off, Off],  // 7: invalid
    ]
};

/// Commutation table for counter-clockwise (reversed) rotation.
const COMMUTATION_TABLE_REV: [[PhaseState; 3]; 8] = {
    use PhaseState::*;
    [
        [Off, Off, Off],  // 0: invalid
        [Off, Low, High], // 1
        [High, Off, Low], // 2
        [Off, High, Low], // 3
        [High, Low, Off], // 4
        [Low, Off, High], // 5 (reversed)
        [Low, High, Off], // 6 (reversed)
        [Off, Off, Off],  // 7: invalid
    ]
};

/// Six-step BLDC commutation controller.
pub struct SixStepCommutator<S: ControlScalar> {
    /// PWM duty cycle (0..1).
    duty: S,
    /// Forward (true) or reverse (false) direction.
    forward: bool,
}

impl<S: ControlScalar> SixStepCommutator<S> {
    pub fn new() -> Self {
        Self {
            duty: S::ZERO,
            forward: true,
        }
    }

    pub fn set_duty(&mut self, duty: S) {
        self.duty = duty.clamp_val(S::ZERO, S::ONE);
    }

    pub fn set_direction(&mut self, forward: bool) {
        self.forward = forward;
    }

    /// Get commutation states and duty cycle for given hall sensor state.
    /// Returns ([phase_a, phase_b, phase_c], duty_cycle).
    pub fn commutate(&self, hall: HallState) -> ([PhaseState; 3], S) {
        let idx = (hall & 0x07) as usize;
        let table = if self.forward {
            &COMMUTATION_TABLE
        } else {
            &COMMUTATION_TABLE_REV
        };
        (table[idx], self.duty)
    }

    /// Returns effective duty cycle per phase (0..duty for High, 0 for Low/Off).
    pub fn phase_duties(&self, hall: HallState) -> [S; 3] {
        let (states, duty) = self.commutate(hall);
        core::array::from_fn(|i| match states[i] {
            PhaseState::High => duty,
            PhaseState::Low => S::ZERO,
            PhaseState::Off => S::ZERO,
        })
    }
}

impl<S: ControlScalar> Default for SixStepCommutator<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Electrical sector from hall sensor state.
pub fn hall_to_sector(hall: HallState) -> Option<u8> {
    match hall & 0x07 {
        1 => Some(1),
        2 => Some(2),
        3 => Some(3),
        4 => Some(4),
        5 => Some(5),
        6 => Some(6),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_hall_states_produce_output() {
        let mut comm = SixStepCommutator::<f64>::new();
        comm.set_duty(0.8);
        for hall in 1..=6_u8 {
            let (states, duty) = comm.commutate(hall);
            assert_eq!(duty, 0.8);
            // Exactly one High and one Low, one Off
            let highs = states.iter().filter(|&&s| s == PhaseState::High).count();
            let lows = states.iter().filter(|&&s| s == PhaseState::Low).count();
            assert_eq!(highs, 1, "hall={}: should have 1 High", hall);
            assert_eq!(lows, 1, "hall={}: should have 1 Low", hall);
        }
    }

    #[test]
    fn invalid_hall_gives_all_off() {
        let comm = SixStepCommutator::<f64>::new();
        let (states, _) = comm.commutate(0);
        assert!(states.iter().all(|&s| s == PhaseState::Off));
        let (states7, _) = comm.commutate(7);
        assert!(states7.iter().all(|&s| s == PhaseState::Off));
    }

    #[test]
    fn reverse_differs_from_forward() {
        let mut fwd = SixStepCommutator::<f64>::new();
        fwd.set_duty(1.0);
        let mut rev = SixStepCommutator::<f64>::new();
        rev.set_direction(false);
        rev.set_duty(1.0);

        let (fwd_states, _) = fwd.commutate(1);
        let (rev_states, _) = rev.commutate(1);
        assert_ne!(fwd_states, rev_states);
    }

    #[test]
    fn duty_clamped_to_zero_one() {
        let mut comm = SixStepCommutator::<f64>::new();
        comm.set_duty(2.0);
        let (_, d) = comm.commutate(1);
        assert_eq!(d, 1.0);

        comm.set_duty(-1.0);
        let (_, d) = comm.commutate(1);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn hall_to_sector_mapping() {
        for h in 1..=6_u8 {
            assert!(hall_to_sector(h).is_some());
        }
        assert!(hall_to_sector(0).is_none());
        assert!(hall_to_sector(7).is_none());
    }
}
