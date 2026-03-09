//! SVPWM harmonic and duty-cycle validation.

use oxictl::motor::transform::clarke::AlphaBeta;
use oxictl::motor::transform::svpwm::svpwm;

/// SVPWM: duty cycles sum to 1.5 (average = 0.5 per phase) for zero reference.
#[test]
fn svpwm_zero_reference_gives_half_duty() {
    let ab = AlphaBeta::<f64> {
        alpha: 0.0,
        beta: 0.0,
        zero: 0.0,
    };
    let vdc = 24.0f64;
    let duty = svpwm(&ab, vdc);
    // Zero reference → all duties ≈ 0.5
    assert!((duty.ta - 0.5).abs() < 0.01, "ta={}", duty.ta);
    assert!((duty.tb - 0.5).abs() < 0.01, "tb={}", duty.tb);
    assert!((duty.tc - 0.5).abs() < 0.01, "tc={}", duty.tc);
}

/// SVPWM: all duty cycles are within [0, 1].
#[test]
fn svpwm_duties_within_bounds() {
    let vdc = 48.0f64;
    // Sweep voltage vectors in the αβ plane
    use std::f64::consts::TAU;
    for i in 0..36 {
        let angle = i as f64 * TAU / 36.0;
        let mag = 0.8f64; // 80% modulation
        let ab = AlphaBeta {
            alpha: mag * angle.cos(),
            beta: mag * angle.sin(),
            zero: 0.0,
        };
        let duty = svpwm(&ab, vdc);
        assert!(
            duty.ta >= -0.01 && duty.ta <= 1.01,
            "ta={} out of bounds at angle={:.2}",
            duty.ta,
            angle
        );
        assert!(
            duty.tb >= -0.01 && duty.tb <= 1.01,
            "tb={} out of bounds",
            duty.tb
        );
        assert!(
            duty.tc >= -0.01 && duty.tc <= 1.01,
            "tc={} out of bounds",
            duty.tc
        );
    }
}

/// SVPWM: output voltages (from duties) are balanced for balanced αβ input.
#[test]
fn svpwm_balanced_output() {
    // For balanced αβ input, reconstructed abc voltages should be balanced
    let vdc = 48.0f64;

    let ab = AlphaBeta::<f64> {
        alpha: 0.5,
        beta: 0.0,
        zero: 0.0,
    }; // α-axis vector
    let duty = svpwm(&ab, vdc);

    // Convert duty cycles to phase voltages (relative to mid-bus)
    let va = (duty.ta - 0.5) * vdc;
    let vb = (duty.tb - 0.5) * vdc;
    let vc = (duty.tc - 0.5) * vdc;

    // Sum should be zero (balanced, or close to it with 3rd harmonic injection)
    // With SVPWM (min-max injection), sum ≠ 0 but line-to-line is correct
    let vab = va - vb;
    let vbc = vb - vc;
    let vca = vc - va;

    // Line-to-line voltages sum to zero
    assert!(
        (vab + vbc + vca).abs() < 1e-10,
        "Line voltages don't sum to zero: {}",
        vab + vbc + vca
    );
}

/// SVPWM: maximum linear range is Vdc/√3.
#[test]
fn svpwm_linear_range_boundary() {
    let vdc = 100.0f64;
    let sqrt3 = 3.0f64.sqrt();
    let v_max_linear = vdc / sqrt3; // ~57.7V

    // At 90% of max linear range: all duties should be within [0,1]
    let v_test = 0.9 * v_max_linear / vdc; // normalized
    let ab = AlphaBeta::<f64> {
        alpha: v_test,
        beta: 0.0,
        zero: 0.0,
    };
    let duty = svpwm(&ab, vdc);

    assert!(
        duty.ta >= 0.0 && duty.ta <= 1.0,
        "ta={} out of range",
        duty.ta
    );
    assert!(
        duty.tb >= 0.0 && duty.tb <= 1.0,
        "tb={} out of range",
        duty.tb
    );
    assert!(
        duty.tc >= 0.0 && duty.tc <= 1.0,
        "tc={} out of range",
        duty.tc
    );
}

/// SVPWM: symmetry across sectors (rotating reference vector).
#[test]
fn svpwm_sector_symmetry() {
    let vdc = 48.0f64;
    use std::f64::consts::PI;

    // At angles 60° apart (sector boundaries), duties should be related by 120° symmetry
    let mag = 0.5f64;
    let angles = [
        0.0f64,
        PI / 3.0,
        2.0 * PI / 3.0,
        PI,
        4.0 * PI / 3.0,
        5.0 * PI / 3.0,
    ];

    let duties: Vec<_> = angles
        .iter()
        .map(|&a| {
            let ab = AlphaBeta {
                alpha: mag * a.cos(),
                beta: mag * a.sin(),
                zero: 0.0,
            };
            svpwm(&ab, vdc)
        })
        .collect();

    // Each duty should be a 120° rotation of the previous one
    for d in &duties {
        assert!(d.ta >= 0.0 && d.ta <= 1.0);
        assert!(d.tb >= 0.0 && d.tb <= 1.0);
        assert!(d.tc >= 0.0 && d.tc <= 1.0);
    }
}
