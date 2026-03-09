use crate::core::scalar::ControlScalar;

/// Trapezoidal (sinusoidal-like) current reference generator for BLDC motors.
///
/// Produces three-phase current references with a trapezoidal profile
/// as a function of electrical angle. Each phase:
///   - Rises linearly over `rise_angle` rad
///   - Is flat at ±1 (×i_ref) for the middle sector
///   - Falls linearly over `rise_angle` rad
///   - Is zero (off) for the complementary half-cycle
///
/// Standard trapezoidal: rise/flat/fall each = π/3 rad (120° trapezoidal)
#[derive(Debug, Clone, Copy)]
pub struct TrapezoidalCommutator<S: ControlScalar> {
    /// Electrical angle over which current ramps up/down (rad). Default π/3.
    pub rise_angle: S,
    /// Peak current reference (A).
    pub i_ref: S,
}

impl<S: ControlScalar> TrapezoidalCommutator<S> {
    pub fn new(i_ref: S) -> Self {
        Self {
            rise_angle: S::PI / S::from_f64(3.0),
            i_ref,
        }
    }

    /// Compute three-phase current references given electrical angle θ_e (rad).
    ///
    /// Phase offsets: B lags A by 2π/3, C lags A by 4π/3.
    pub fn current_references(&self, theta_e: S) -> [S; 3] {
        core::array::from_fn(|i| {
            let offset = S::from_f64(i as f64) * S::TWO * S::PI / S::from_f64(3.0);
            let angle = theta_e - offset;
            self.i_ref * trapezoidal_profile(angle, self.rise_angle)
        })
    }

    /// Voltage duty cycles from current error (simple proportional).
    pub fn duties(&self, theta_e: S, i_measured: &[S; 3], kp: S) -> [S; 3] {
        let i_ref = self.current_references(theta_e);
        core::array::from_fn(|i| {
            let duty = kp * (i_ref[i] - i_measured[i]);
            duty.clamp_val(-S::ONE, S::ONE)
        })
    }
}

/// Trapezoidal profile [-1, 1] as a function of electrical angle.
///
/// The period is 2π. The shape (for one electrical cycle):
///   - [0, rise]: linear 0 → +1
///   - [rise, π-rise]: flat +1
///   - [π-rise, π]: linear +1 → 0
///   - [π, π+rise]: linear 0 → -1
///   - [π+rise, 2π-rise]: flat -1
///   - [2π-rise, 2π]: linear -1 → 0
fn trapezoidal_profile<S: ControlScalar>(angle: S, rise: S) -> S {
    // Normalize angle to [0, 2π)
    let two_pi = S::TWO * S::PI;
    let a = normalize_angle(angle, two_pi);
    let flat = S::PI - rise;

    if a < rise {
        // Rising edge: 0 → +1
        a / rise
    } else if a < flat {
        // Flat top: +1
        S::ONE
    } else if a < S::PI {
        // Falling edge: +1 → 0
        (S::PI - a) / rise
    } else {
        let a2 = a - S::PI;
        if a2 < rise {
            // Rising edge negative: 0 → -1
            -(a2 / rise)
        } else if a2 < flat {
            // Flat bottom: -1
            -S::ONE
        } else if a2 < S::PI {
            // Falling edge negative: -1 → 0
            -(S::PI - a2) / rise
        } else {
            S::ZERO
        }
    }
}

fn normalize_angle<S: ControlScalar>(angle: S, period: S) -> S {
    let mut a = angle;
    while a < S::ZERO {
        a += period;
    }
    while a >= period {
        a -= period;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_refs_sum_to_zero() {
        let comm = TrapezoidalCommutator::new(10.0_f64);
        for k in 0..12 {
            let theta = k as f64 * core::f64::consts::PI / 6.0;
            let refs = comm.current_references(theta);
            let sum = refs[0] + refs[1] + refs[2];
            assert!(sum.abs() < 1e-10, "sum={:.6} at theta={:.2}", sum, theta);
        }
    }

    #[test]
    fn peak_magnitude_is_i_ref() {
        let i_ref = 5.0_f64;
        let comm = TrapezoidalCommutator::new(i_ref);
        let half_pi = core::f64::consts::PI / 2.0;
        // At θ = π/2, phase A should be near peak
        let refs = comm.current_references(half_pi);
        assert!(refs[0].abs() <= i_ref + 1e-10, "ia={:.4}", refs[0]);
    }

    #[test]
    fn trapezoidal_profile_symmetric() {
        let rise = core::f64::consts::PI / 3.0;
        let pos = trapezoidal_profile(core::f64::consts::PI / 2.0, rise);
        let neg = trapezoidal_profile(3.0 * core::f64::consts::PI / 2.0, rise);
        assert!(
            (pos + neg).abs() < 1e-10,
            "not anti-symmetric: {pos:.4} {neg:.4}"
        );
    }

    #[test]
    fn flat_section_is_one() {
        let rise = core::f64::consts::PI / 3.0;
        // Middle of flat region: π/3 < angle < 2π/3 → value = 1
        let v = trapezoidal_profile(core::f64::consts::PI / 2.0, rise);
        assert!((v - 1.0).abs() < 1e-10, "v={v:.6}");
    }
}
