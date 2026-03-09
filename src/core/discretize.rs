use crate::core::{
    matrix::{matmul, Matrix},
    scalar::ControlScalar,
};

/// Discretize a continuous-time LTI system using the Forward Euler method.
///
/// Ad = I + Ac·dt
/// Bd = Bc·dt
pub fn discretize_euler<S: ControlScalar, const N: usize, const M: usize>(
    ac: &Matrix<S, N, N>,
    bc: &Matrix<S, N, M>,
    dt: S,
) -> (Matrix<S, N, N>, Matrix<S, N, M>) {
    // Ad = I + Ac*dt
    let mut ad = ac.scale(dt);
    for i in 0..N {
        ad.data[i][i] += S::ONE;
    }
    let bd = bc.scale(dt);
    (ad, bd)
}

/// Discretize using Zero-Order Hold via truncated matrix exponential series.
///
/// Ad = exp(Ac·dt) ≈ Σ_{k=0}^{terms} (Ac·dt)^k / k!
/// Bd = (Σ_{k=0}^{terms-1} (Ac·dt)^k / (k+1)!) · Bc·dt
///
/// `terms` = 10 is sufficient for stable systems with small dt.
pub fn discretize_zoh<S: ControlScalar, const N: usize, const M: usize>(
    ac: &Matrix<S, N, N>,
    bc: &Matrix<S, N, M>,
    dt: S,
    terms: usize,
) -> (Matrix<S, N, N>, Matrix<S, N, M>) {
    let a_dt = ac.scale(dt);

    // Ad = exp(Ac*dt) via series: I + A_dt + A_dt²/2! + ...
    let mut ad = Matrix::<S, N, N>::identity();
    let mut power = Matrix::<S, N, N>::identity(); // A_dt^k at start of each iteration

    // b_integral = Σ_{k=0}^{terms-1} A_dt^k / (k+1)!
    // Used to compute Bd = b_integral * Bc * dt
    let mut b_integral = Matrix::<S, N, N>::zeros();
    let mut factorial = S::ONE;

    for k in 0..terms {
        let k_f = S::from_f64((k + 1) as f64);
        // b_integral term: A_dt^k / (k+1)!
        let b_fact = factorial * k_f;
        b_integral = b_integral.add_mat(&power.scale(S::ONE / b_fact));

        // Advance: power = A_dt^{k+1}
        power = matmul(&power, &a_dt);
        factorial *= k_f;
        ad = ad.add_mat(&power.scale(S::ONE / factorial));
    }

    // Bd = b_integral * Bc * dt
    let bd = matmul(&b_integral, &bc.scale(dt));
    (ad, bd)
}

/// Discretize using Tustin (bilinear / trapezoidal) method.
///
/// Ad = (I − Ac·dt/2)⁻¹ · (I + Ac·dt/2)
/// Bd = (I − Ac·dt/2)⁻¹ · Bc·dt
///
/// Returns `None` if (I − Ac·dt/2) is singular.
pub fn discretize_tustin<S: ControlScalar, const N: usize, const M: usize>(
    ac: &Matrix<S, N, N>,
    bc: &Matrix<S, N, M>,
    dt: S,
) -> Option<(Matrix<S, N, N>, Matrix<S, N, M>)> {
    let half_dt = dt / S::from_f64(2.0);

    // A_plus  = I + Ac * dt/2
    let mut a_plus = ac.scale(half_dt);
    for i in 0..N {
        a_plus.data[i][i] += S::ONE;
    }
    // A_minus = I - Ac * dt/2
    let mut a_minus = ac.scale(-half_dt);
    for i in 0..N {
        a_minus.data[i][i] += S::ONE;
    }

    let a_minus_inv = a_minus.inv()?;

    let ad = matmul(&a_minus_inv, &a_plus);
    let bd = matmul(&a_minus_inv, &bc.scale(dt));

    Some((ad, bd))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ac_2x2() -> Matrix<f64, 2, 2> {
        Matrix {
            data: [[-1.0, 0.0], [0.0, -2.0]],
        }
    }

    fn make_bc_2x1() -> Matrix<f64, 2, 1> {
        Matrix {
            data: [[1.0], [1.0]],
        }
    }

    #[test]
    fn euler_ad_is_i_plus_a_dt() {
        let ac = make_ac_2x2();
        let bc = make_bc_2x1();
        let dt = 0.01_f64;
        let (ad, bd) = discretize_euler(&ac, &bc, dt);
        assert!((ad.data[0][0] - 0.99).abs() < 1e-10);
        assert!((ad.data[1][1] - 0.98).abs() < 1e-10);
        assert!((bd.data[0][0] - 0.01).abs() < 1e-10);
    }

    #[test]
    fn zoh_stable_first_order() {
        let ac = Matrix::<f64, 1, 1> { data: [[-1.0]] };
        let bc = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let dt = 0.1_f64;
        let (ad, _bd) = discretize_zoh(&ac, &bc, dt, 20);
        let expected = (-0.1_f64).exp();
        assert!(
            (ad.data[0][0] - expected).abs() < 1e-6,
            "ad={:.8}, exp={:.8}",
            ad.data[0][0],
            expected
        );
    }

    #[test]
    fn tustin_returns_stable_ad() {
        let ac = Matrix::<f64, 1, 1> { data: [[-1.0]] };
        let bc = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let dt = 0.1_f64;
        let (ad, _bd) = discretize_tustin(&ac, &bc, dt).expect("should succeed");
        let expected = (1.0 - 0.05) / (1.0 + 0.05);
        assert!(
            (ad.data[0][0] - expected).abs() < 1e-10,
            "ad={:.8}, exp={:.8}",
            ad.data[0][0],
            expected
        );
    }

    #[test]
    fn inv_2x2_correct() {
        let m = Matrix::<f64, 2, 2> {
            data: [[2.0, 1.0], [1.0, 3.0]],
        };
        let inv = m.inv().expect("should be invertible");
        let prod = matmul(&m, &inv);
        assert!((prod.data[0][0] - 1.0).abs() < 1e-10);
        assert!((prod.data[0][1]).abs() < 1e-10);
        assert!((prod.data[1][0]).abs() < 1e-10);
        assert!((prod.data[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn singular_inv_returns_none() {
        let m = Matrix::<f64, 2, 2> {
            data: [[1.0, 2.0], [2.0, 4.0]],
        };
        assert!(m.inv().is_none());
    }
}
