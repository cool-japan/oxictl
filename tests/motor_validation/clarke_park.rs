//! Clarke/Park transform numerical validation.

use oxictl::motor::transform::clarke::{clarke, clarke_2ph, clarke_inverse, AlphaBeta};
use oxictl::motor::transform::park::{park, park_inverse, Dq};

const EPS: f64 = 1e-10;

/// Clarke transform: energy invariance (power balance).
#[test]
fn clarke_amplitude_invariant() {
    // For balanced 3-phase: a + b + c = 0
    let a = 1.0f64;
    let b = -0.5f64;
    let c = -0.5f64;
    let ab = clarke(a, b, c);

    // Amplitude-invariant: |Vαβ| = |Va| for fundamental
    // |alpha|² + |beta|² ≈ a² (for balanced)
    let mag_abc = (a * a + b * b + c * c) / 1.5; // RMS scale
    let mag_ab = ab.alpha * ab.alpha + ab.beta * ab.beta;
    assert!(
        (mag_abc - mag_ab).abs() < 1e-10,
        "Amplitude invariance: mag_abc={}, mag_ab={}",
        mag_abc,
        mag_ab
    );
}

/// Clarke round-trip: abc → αβ → abc = identity.
#[test]
fn clarke_inverse_roundtrip() {
    let test_cases = [
        (1.0f64, -0.5, -0.5),
        (0.866, 0.0, -0.866),
        (0.5, 0.866, -1.366),
        (0.0, 0.0, 0.0),
    ];

    for (a, b, c) in test_cases {
        let ab = clarke(a, b, c);
        let (a2, b2, c2) = clarke_inverse(&ab);
        // Note: for unbalanced input, inverse only reconstructs balanced part
        // For balanced (a+b+c=0), full reconstruction
        let balanced = (a + b + c).abs() < 1e-10;
        if balanced {
            assert!((a - a2).abs() < EPS, "a mismatch: {} vs {}", a, a2);
            assert!((b - b2).abs() < EPS, "b mismatch: {} vs {}", b, b2);
            assert!((c - c2).abs() < EPS, "c mismatch: {} vs {}", c, c2);
        }
    }
}

/// Clarke 2-phase (c=-a-b): matches full 3-phase.
#[test]
fn clarke_2ph_matches_3ph() {
    let test_cases = [(1.5f64, -0.75), (0.0, 1.0), (-1.0, 0.5)];
    for (a, b) in test_cases {
        let c = -a - b;
        let ab3 = clarke(a, b, c);
        let ab2 = clarke_2ph(a, b);
        assert!((ab3.alpha - ab2.alpha).abs() < EPS, "alpha mismatch");
        assert!((ab3.beta - ab2.beta).abs() < EPS, "beta mismatch");
    }
}

/// Park transform at known angles (reference values from appendix).
#[test]
fn park_transform_known_angles() {
    use std::f64::consts::PI;

    // At theta=0: d=alpha, q=beta
    let ab = AlphaBeta {
        alpha: 1.0f64,
        beta: 0.0,
        zero: 0.0,
    };
    let dq = park(&ab, 0.0);
    assert!((dq.d - 1.0).abs() < 1e-10, "d at theta=0: {}", dq.d);
    assert!((dq.q - 0.0).abs() < 1e-10, "q at theta=0: {}", dq.q);

    // At theta=π/2: d=-beta, q=alpha
    let dq90 = park(&ab, PI / 2.0);
    assert!((dq90.d - 0.0).abs() < 1e-10, "d at 90deg: {}", dq90.d);
    assert!(
        (dq90.q + 1.0).abs() < 1e-10,
        "q at 90deg: {} (expected -1)",
        dq90.q
    );

    // At theta=π: d=-alpha, q=-beta
    let dq180 = park(&ab, PI);
    assert!((dq180.d + 1.0).abs() < 1e-10, "d at 180deg: {}", dq180.d);
    assert!((dq180.q - 0.0).abs() < 1e-10, "q at 180deg: {}", dq180.q);
}

/// Park round-trip: αβ → dq → αβ = identity.
#[test]
fn park_inverse_roundtrip() {
    use std::f64::consts::PI;

    let test_cases = [
        0.0f64,
        PI / 6.0,
        PI / 4.0,
        PI / 3.0,
        PI / 2.0,
        PI,
        3.0 * PI / 2.0,
    ];
    let ab_in = AlphaBeta {
        alpha: 1.5f64,
        beta: -0.8,
        zero: 0.0,
    };

    for theta in test_cases {
        let dq = park(&ab_in, theta);
        let ab_out = park_inverse(&Dq { d: dq.d, q: dq.q }, theta);
        assert!(
            (ab_in.alpha - ab_out.alpha).abs() < 1e-10,
            "alpha roundtrip failed at theta={}: {} vs {}",
            theta,
            ab_in.alpha,
            ab_out.alpha
        );
        assert!(
            (ab_in.beta - ab_out.beta).abs() < 1e-10,
            "beta roundtrip failed at theta={}: {} vs {}",
            theta,
            ab_in.beta,
            ab_out.beta
        );
    }
}

/// Full chain: abc → αβ → dq → αβ → abc (round-trip).
#[test]
fn full_chain_roundtrip() {
    use std::f64::consts::PI;

    let a = 1.0f64;
    let b = -0.5f64;
    let c = -0.5f64;
    let theta = PI / 4.0;

    let ab = clarke(a, b, c);
    let dq = park(&ab, theta);
    let ab2 = park_inverse(&Dq { d: dq.d, q: dq.q }, theta);
    let (a2, b2, c2) = clarke_inverse(&ab2);

    assert!((a - a2).abs() < 1e-10, "a: {} vs {}", a, a2);
    assert!((b - b2).abs() < 1e-10, "b: {} vs {}", b, b2);
    assert!((c - c2).abs() < 1e-10, "c: {} vs {}", c, c2);
}
