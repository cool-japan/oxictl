use crate::core::scalar::ControlScalar;

/// Sinusoidal PWM (SPWM) duty cycle generator.
///
/// Compares sinusoidal reference signals with a triangular carrier
/// to generate three-phase PWM duties.
///
/// For a balanced three-phase system with modulation index m ∈ [0, 1]:
///   d_a(θ) = 0.5 + 0.5·m·cos(θ)
///   d_b(θ) = 0.5 + 0.5·m·cos(θ − 2π/3)
///   d_c(θ) = 0.5 + 0.5·m·cos(θ − 4π/3)
///
/// Returns duties ∈ [0, 1] (leg high-side switch duty cycle).
pub fn spwm_duties<S: ControlScalar>(theta: S, m: S) -> [S; 3] {
    let m_clamped = m.clamp_val(S::ZERO, S::ONE);
    let two_pi_3 = S::TWO * S::PI / S::from_f64(3.0);
    let half = S::from_f64(0.5);
    [
        half + half * m_clamped * theta.cos(),
        half + half * m_clamped * (theta - two_pi_3).cos(),
        half + half * m_clamped * (theta - S::TWO * two_pi_3).cos(),
    ]
}

/// Single-phase SPWM duty cycle.
///
/// Returns duty ∈ [0, 1] for one half-bridge leg.
pub fn spwm_single<S: ControlScalar>(theta: S, m: S) -> S {
    let m_clamped = m.clamp_val(S::ZERO, S::ONE);
    let half = S::from_f64(0.5);
    half + half * m_clamped * theta.cos()
}

/// Overmodulation mode: extend modulation index beyond 1.0 (into square-wave region).
///
/// With third-harmonic injection, the effective linear range extends to m = 1.155 (2/√3).
/// This function applies third-harmonic injection for increased DC bus utilization.
pub fn spwm_with_third_harmonic<S: ControlScalar>(theta: S, m: S) -> [S; 3] {
    let m_clamped = m.clamp_val(S::ZERO, S::from_f64(1.155));
    let two_pi_3 = S::TWO * S::PI / S::from_f64(3.0);
    let half = S::from_f64(0.5);
    // Third harmonic: v_3h = -m * sin(3θ) / 6
    let v3h = -m_clamped * (S::from_f64(3.0) * theta).sin() / S::from_f64(6.0);

    let raw = [
        half + half * m_clamped * theta.cos() + v3h,
        half + half * m_clamped * (theta - two_pi_3).cos() + v3h,
        half + half * m_clamped * (theta - S::TWO * two_pi_3).cos() + v3h,
    ];
    core::array::from_fn(|i| raw[i].clamp_val(S::ZERO, S::ONE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spwm_sums_to_three_halves() {
        // Sum of three duties should equal 3/2 (each centered at 0.5)
        let sum: f64 = spwm_duties(0.0_f64, 0.8).iter().sum();
        assert!((sum - 1.5).abs() < 1e-10, "sum={sum:.6}");
    }

    #[test]
    fn spwm_duties_in_range() {
        for k in 0..36 {
            let theta = k as f64 * core::f64::consts::PI / 18.0;
            let duties = spwm_duties(theta, 0.9);
            for (i, &d) in duties.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&d),
                    "duties[{i}]={d:.4} out of [0,1] at theta={theta:.2}"
                );
            }
        }
    }

    #[test]
    fn zero_modulation_gives_half_duty() {
        let duties = spwm_duties(0.5_f64, 0.0);
        for &d in &duties {
            assert!((d - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn third_harmonic_duties_in_range() {
        // With 3rd harmonic injection, m up to ~1.155 should stay in [0,1]
        for k in 0..36 {
            let theta = k as f64 * core::f64::consts::PI / 18.0;
            let duties = spwm_with_third_harmonic(theta, 1.1);
            for (i, &d) in duties.iter().enumerate() {
                assert!(
                    (-0.01..=1.01).contains(&d),
                    "duties[{i}]={d:.4} out of range at theta={theta:.2}"
                );
            }
        }
    }

    #[test]
    fn spwm_120_degree_phase_offset() {
        let theta = 0.0_f64;
        let duties = spwm_duties(theta, 1.0);
        // At theta=0: d_a = 1.0, d_b = 0.5 + 0.5*cos(-120°) = 0.5 - 0.25 = 0.25
        assert!((duties[0] - 1.0).abs() < 1e-10, "d_a={:.4}", duties[0]);
        let expected_b = 0.5 + 0.5 * (-2.0 * core::f64::consts::PI / 3.0_f64).cos();
        assert!(
            (duties[1] - expected_b).abs() < 1e-10,
            "d_b={:.4}",
            duties[1]
        );
    }
}
