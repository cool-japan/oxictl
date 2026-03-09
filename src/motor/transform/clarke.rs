use crate::core::scalar::ControlScalar;

/// Clarke transform result: αβ stationary frame.
#[derive(Debug, Clone, Copy)]
pub struct AlphaBeta<S: ControlScalar> {
    pub alpha: S,
    pub beta: S,
    pub zero: S,
}

/// Clarke transform: 3-phase abc → αβ stationary frame.
///
/// Amplitude-invariant form:
///   α =  (2/3)*[a - (1/2)*b - (1/2)*c]
///   β =  (2/3)*[     (√3/2)*b - (√3/2)*c]
///   0 =  (1/3)*[a + b + c]
///
/// For balanced three-phase: a + b + c = 0, so zero component = 0.
pub fn clarke<S: ControlScalar>(a: S, b: S, c: S) -> AlphaBeta<S> {
    let two_thirds = S::from_f64(2.0 / 3.0);
    let one_third = S::from_f64(1.0 / 3.0);
    let half = S::HALF;
    let sqrt3_over2 = S::from_f64(0.8660254037844386);

    let alpha = two_thirds * (a - half * b - half * c);
    let beta = two_thirds * (sqrt3_over2 * b - sqrt3_over2 * c);
    let zero = one_third * (a + b + c);

    AlphaBeta { alpha, beta, zero }
}

/// Clarke transform for balanced 2-measurement case (c = -a - b).
pub fn clarke_2ph<S: ControlScalar>(a: S, b: S) -> AlphaBeta<S> {
    clarke(a, b, -a - b)
}

/// Inverse Clarke transform: αβ → abc.
/// For balanced systems (zero = 0).
pub fn clarke_inverse<S: ControlScalar>(ab: &AlphaBeta<S>) -> (S, S, S) {
    let sqrt3_over2 = S::from_f64(0.8660254037844386);
    let half = S::HALF;

    let a = ab.alpha + ab.zero;
    let b = -half * ab.alpha + sqrt3_over2 * ab.beta + ab.zero;
    let c = -half * ab.alpha - sqrt3_over2 * ab.beta + ab.zero;
    (a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clarke_balanced_a_axis() {
        // a=1, b=-0.5, c=-0.5 (balanced, a-axis aligned)
        let ab = clarke(1.0_f64, -0.5, -0.5);
        assert!((ab.alpha - 1.0).abs() < 1e-10, "alpha={}", ab.alpha);
        assert!((ab.beta).abs() < 1e-10, "beta={}", ab.beta);
        assert!((ab.zero).abs() < 1e-10, "zero={}", ab.zero);
    }

    #[test]
    fn clarke_balanced_sum_zero() {
        let ab = clarke(1.0_f64, -0.5, -0.5);
        assert!((ab.zero).abs() < 1e-10);
    }

    #[test]
    fn clarke_90_degree() {
        // Rotate by 90°: b = 1, a = -0.5, c = -0.5 (shifted by 120°)
        let phase = core::f64::consts::PI / 2.0;
        let a = phase.cos();
        let b = (phase - 2.0 * core::f64::consts::PI / 3.0).cos();
        let c = (phase - 4.0 * core::f64::consts::PI / 3.0).cos();
        let ab = clarke(a, b, c);
        // α ≈ 0, β ≈ 1
        assert!(ab.alpha.abs() < 1e-10, "alpha={}", ab.alpha);
        assert!((ab.beta - 1.0).abs() < 1e-10, "beta={}", ab.beta);
    }

    #[test]
    fn inverse_clarke_roundtrip() {
        let ab = clarke(1.0_f64, -0.5, -0.5);
        let (a2, b2, c2) = clarke_inverse(&ab);
        assert!((a2 - 1.0).abs() < 1e-10, "a={}", a2);
        assert!((b2 - (-0.5)).abs() < 1e-10, "b={}", b2);
        assert!((c2 - (-0.5)).abs() < 1e-10, "c={}", c2);
    }

    #[test]
    fn clarke_2ph_matches_3ph() {
        let a = 1.0_f64;
        let b = -0.5;
        let ab2 = clarke_2ph(a, b);
        let ab3 = clarke(a, b, -a - b);
        assert!((ab2.alpha - ab3.alpha).abs() < 1e-10);
        assert!((ab2.beta - ab3.beta).abs() < 1e-10);
    }

    #[test]
    fn amplitude_preserved() {
        // Verify that amplitude is preserved (amplitude-invariant form)
        let amp = 5.0_f64;
        let ab = clarke(amp, -amp / 2.0, -amp / 2.0);
        let reconstructed_amp = (ab.alpha * ab.alpha + ab.beta * ab.beta).sqrt();
        assert!(
            (reconstructed_amp - amp).abs() < 1e-10,
            "amp={}, reconstructed={}",
            amp,
            reconstructed_amp
        );
    }
}
