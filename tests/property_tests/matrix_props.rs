//! Property-based tests for Matrix<S,R,C> algebraic laws.

use oxictl::core::matrix::{matmul, matvec, Matrix};
use proptest::prelude::*;

fn finite(v: f64) -> bool {
    v.is_finite()
}

proptest! {
    /// Transpose is involutory: (A^T)^T = A
    #[test]
    fn transpose_involutory(
        a00 in -1e3f64..1e3, a01 in -1e3f64..1e3,
        a10 in -1e3f64..1e3, a11 in -1e3f64..1e3,
    ) {
        prop_assume!(finite(a00) && finite(a01) && finite(a10) && finite(a11));
        let m = Matrix::<f64, 2, 2> { data: [[a00, a01], [a10, a11]] };
        let tt = m.transpose().transpose();
        for r in 0..2 {
            for c in 0..2 {
                prop_assert!((m.data[r][c] - tt.data[r][c]).abs() < 1e-10);
            }
        }
    }

    /// Addition commutative: A + B = B + A
    #[test]
    fn add_commutative(
        a00 in -1e3f64..1e3, a01 in -1e3f64..1e3,
        b00 in -1e3f64..1e3, b01 in -1e3f64..1e3,
    ) {
        prop_assume!(finite(a00) && finite(a01) && finite(b00) && finite(b01));
        let a = Matrix::<f64, 1, 2> { data: [[a00, a01]] };
        let b = Matrix::<f64, 1, 2> { data: [[b00, b01]] };
        let ab = a.add_mat(&b);
        let ba = b.add_mat(&a);
        for c in 0..2 {
            prop_assert!((ab.data[0][c] - ba.data[0][c]).abs() < 1e-10);
        }
    }

    /// Multiply by identity gives same matrix: A * I = A
    #[test]
    fn matmul_identity_right(
        a00 in -1e3f64..1e3, a01 in -1e3f64..1e3,
        a10 in -1e3f64..1e3, a11 in -1e3f64..1e3,
    ) {
        prop_assume!(finite(a00) && finite(a01) && finite(a10) && finite(a11));
        let a = Matrix::<f64, 2, 2> { data: [[a00, a01], [a10, a11]] };
        let i = Matrix::<f64, 2, 2>::identity();
        let ai = matmul(&a, &i);
        for r in 0..2 {
            for c in 0..2 {
                prop_assert!((ai.data[r][c] - a.data[r][c]).abs() < 1e-9 * a.data[r][c].abs().max(1.0));
            }
        }
    }

    /// Scale by zero gives zeros matrix
    #[test]
    fn scale_by_zero(
        a00 in -1e3f64..1e3, a01 in -1e3f64..1e3,
    ) {
        prop_assume!(finite(a00) && finite(a01));
        let a = Matrix::<f64, 1, 2> { data: [[a00, a01]] };
        let z = a.scale(0.0);
        for c in 0..2 {
            prop_assert!(z.data[0][c].abs() < 1e-14);
        }
    }

    /// Matvec: zero vector gives zero output
    #[test]
    fn matvec_zero_input(
        a00 in -1e3f64..1e3, a01 in -1e3f64..1e3,
        a10 in -1e3f64..1e3, a11 in -1e3f64..1e3,
    ) {
        prop_assume!(finite(a00) && finite(a01) && finite(a10) && finite(a11));
        let a = Matrix::<f64, 2, 2> { data: [[a00, a01], [a10, a11]] };
        let v = [0.0f64, 0.0];
        let r = matvec(&a, &v);
        prop_assert!(r[0].abs() < 1e-14);
        prop_assert!(r[1].abs() < 1e-14);
    }

    /// 1x1 inverse: inv(a)*a = 1.0
    #[test]
    fn inv_1x1_correct(a in 0.01f64..1e3) {
        let m = Matrix::<f64, 1, 1> { data: [[a]] };
        let inv = m.inv().expect("1x1 invertible");
        let prod = matmul(&m, &inv);
        prop_assert!((prod.data[0][0] - 1.0).abs() < 1e-9);
    }

    /// Negate and negate again gives original
    #[test]
    fn double_negate(
        a00 in -1e3f64..1e3, a01 in -1e3f64..1e3,
    ) {
        prop_assume!(finite(a00) && finite(a01));
        let a = Matrix::<f64, 1, 2> { data: [[a00, a01]] };
        let nn = a.neg().neg();
        for c in 0..2 {
            prop_assert!((nn.data[0][c] - a.data[0][c]).abs() < 1e-14);
        }
    }

    /// sub_mat is anti-commutative: A - B = -(B - A)
    #[test]
    fn sub_mat_anti_commutative(
        a00 in -1e3f64..1e3,
        b00 in -1e3f64..1e3,
    ) {
        prop_assume!(finite(a00) && finite(b00));
        let a = Matrix::<f64, 1, 1> { data: [[a00]] };
        let b = Matrix::<f64, 1, 1> { data: [[b00]] };
        let ab = a.sub_mat(&b);
        let ba_neg = b.sub_mat(&a).neg();
        prop_assert!((ab.data[0][0] - ba_neg.data[0][0]).abs() < 1e-10);
    }
}
