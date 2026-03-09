use crate::core::scalar::ControlScalar;
use crate::motor::transform::clarke::AlphaBeta;

/// Park transform result: dq rotating frame.
#[derive(Debug, Clone, Copy)]
pub struct Dq<S: ControlScalar> {
    /// Direct axis (flux-producing).
    pub d: S,
    /// Quadrature axis (torque-producing).
    pub q: S,
}

/// Park transform: αβ stationary → dq rotating frame.
///
///   d = α*cos(θ) + β*sin(θ)
///   q = -α*sin(θ) + β*cos(θ)
///
/// θ = rotor electrical angle (radians).
pub fn park<S: ControlScalar>(ab: &AlphaBeta<S>, theta: S) -> Dq<S> {
    let (sin_t, cos_t) = theta.sin_cos();
    Dq {
        d: ab.alpha * cos_t + ab.beta * sin_t,
        q: -ab.alpha * sin_t + ab.beta * cos_t,
    }
}

/// Inverse Park transform: dq rotating → αβ stationary frame.
pub fn park_inverse<S: ControlScalar>(dq: &Dq<S>, theta: S) -> AlphaBeta<S> {
    let (sin_t, cos_t) = theta.sin_cos();
    AlphaBeta {
        alpha: dq.d * cos_t - dq.q * sin_t,
        beta: dq.d * sin_t + dq.q * cos_t,
        zero: S::ZERO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motor::transform::clarke::clarke;

    #[test]
    fn park_at_zero_angle() {
        // θ=0: d=α, q=β
        let ab = AlphaBeta {
            alpha: 1.0_f64,
            beta: 0.0,
            zero: 0.0,
        };
        let dq = park(&ab, 0.0);
        assert!((dq.d - 1.0).abs() < 1e-10);
        assert!((dq.q).abs() < 1e-10);
    }

    #[test]
    fn park_at_90_degrees() {
        let ab = AlphaBeta {
            alpha: 1.0_f64,
            beta: 0.0,
            zero: 0.0,
        };
        let dq = park(&ab, core::f64::consts::PI / 2.0);
        // d = cos(90°) = 0, q = -sin(90°) = -1
        assert!((dq.d).abs() < 1e-10, "d={}", dq.d);
        assert!((dq.q - (-1.0)).abs() < 1e-10, "q={}", dq.q);
    }

    #[test]
    fn inverse_park_roundtrip() {
        let ab = AlphaBeta {
            alpha: 0.8_f64,
            beta: 0.6,
            zero: 0.0,
        };
        let theta = core::f64::consts::PI / 4.0;
        let dq = park(&ab, theta);
        let ab2 = park_inverse(&dq, theta);
        assert!((ab2.alpha - ab.alpha).abs() < 1e-10, "alpha={}", ab2.alpha);
        assert!((ab2.beta - ab.beta).abs() < 1e-10, "beta={}", ab2.beta);
    }

    #[test]
    fn full_abc_to_dq_roundtrip() {
        use crate::motor::transform::clarke::clarke_inverse;
        let a = 1.0_f64;
        let b = -0.5;
        let c = -0.5;
        let theta = 0.3;

        let ab = clarke(a, b, c);
        let dq = park(&ab, theta);
        let ab2 = park_inverse(&dq, theta);
        let (a2, b2, c2) = clarke_inverse(&ab2);

        assert!((a2 - a).abs() < 1e-10, "a={}", a2);
        assert!((b2 - b).abs() < 1e-10, "b={}", b2);
        assert!((c2 - c).abs() < 1e-10, "c={}", c2);
    }

    #[test]
    fn field_oriented_d_aligned() {
        // At θ = angle of current phasor, d component should be amplitude, q = 0
        let amplitude = 3.0_f64;
        let theta = 0.7_f64;
        let a = amplitude * theta.cos();
        let b = amplitude * (theta - 2.0 * core::f64::consts::PI / 3.0).cos();
        let c = amplitude * (theta - 4.0 * core::f64::consts::PI / 3.0).cos();

        let ab = clarke(a, b, c);
        let dq = park(&ab, theta);

        assert!((dq.d - amplitude).abs() < 1e-8, "d={}", dq.d);
        assert!(dq.q.abs() < 1e-8, "q={}", dq.q);
    }
}
