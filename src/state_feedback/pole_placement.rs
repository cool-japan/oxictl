use crate::core::matrix::{matmul, matvec, Matrix};
use crate::core::scalar::ControlScalar;

/// Ackermann's formula for pole placement (SISO systems).
///
/// Computes the state-feedback gain K such that the closed-loop system
/// A - B*K has eigenvalues equal to `desired_poles`.
///
/// Only valid for SISO (single input) systems. The system must be controllable.
///
/// # Arguments
/// - `a`: N×N state matrix
/// - `b`: N×1 input matrix (column vector)
/// - `desired_poles`: desired closed-loop eigenvalues (real)
///
/// # Returns
/// `Some(K)` where K is a 1×N gain row-vector, or `None` if the system
/// is not controllable (controllability matrix is singular).
pub fn ackermann<S: ControlScalar, const N: usize>(
    a: &Matrix<S, N, N>,
    b: &Matrix<S, N, 1>,
    desired_poles: &[S; N],
) -> Option<[S; N]> {
    // Build controllability matrix W_c = [B, AB, A²B, ..., A^{N-1}B]  (N×N)
    let mut w_c = Matrix::<S, N, N>::zeros();
    let mut a_pow_b: [S; N] = core::array::from_fn(|i| b.data[i][0]);

    for col in 0..N {
        for (row, &val) in a_pow_b.iter().enumerate().take(N) {
            w_c.data[row][col] = val;
        }
        if col < N - 1 {
            a_pow_b = matvec(a, &a_pow_b);
        }
    }

    let w_c_inv = w_c.inv()?;

    // Compute φ(A) = (A - r_0*I)(A - r_1*I)...(A - r_{N-1}*I)
    let phi_a = char_poly_at_matrix(a, desired_poles);

    // K = e_N^T * W_c^{-1} * φ(A)
    // = (last row of W_c^{-1}) * φ(A)
    let last_row = w_c_inv.data[N - 1]; // last row: [S; N]
    let k = matvec(&phi_a.transpose(), &last_row);
    Some(k)
}

/// Evaluate the characteristic polynomial φ(A) = prod(A - r_i * I).
fn char_poly_at_matrix<S: ControlScalar, const N: usize>(
    a: &Matrix<S, N, N>,
    roots: &[S; N],
) -> Matrix<S, N, N> {
    let mut result = Matrix::<S, N, N>::identity();
    for &r in roots {
        // Compute (A - r*I)
        let mut a_shifted = *a;
        for i in 0..N {
            a_shifted.data[i][i] -= r;
        }
        result = matmul(&result, &a_shifted);
    }
    result
}

/// Full-state feedback controller using a pre-computed gain vector.
///
/// Control law: u = -K * (x - x_ref)
pub struct StateFeedback<S: ControlScalar, const N: usize> {
    /// Gain row-vector (1×N).
    pub k: [S; N],
}

impl<S: ControlScalar, const N: usize> StateFeedback<S, N> {
    pub fn new(k: [S; N]) -> Self {
        Self { k }
    }

    /// Compute control input u = -K*(x - x_ref).
    pub fn control(&self, x: &[S; N], x_ref: &[S; N]) -> S {
        let mut u = S::ZERO;
        for i in 0..N {
            u -= self.k[i] * (x[i] - x_ref[i]);
        }
        u
    }

    /// Regulation: u = -K*x.
    pub fn regulate(&self, x: &[S; N]) -> S {
        self.k
            .iter()
            .zip(x.iter())
            .fold(S::ZERO, |acc, (&ki, &xi)| acc - ki * xi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Double integrator: A = [[0,1],[0,0]], B = [[0],[1]]
    fn double_integrator() -> (Matrix<f64, 2, 2>, Matrix<f64, 2, 1>) {
        let a = Matrix::<f64, 2, 2> {
            data: [[0.0, 1.0], [0.0, 0.0]],
        };
        let b = Matrix::<f64, 2, 1> {
            data: [[0.0], [1.0]],
        };
        (a, b)
    }

    #[test]
    fn ackermann_double_integrator() {
        let (a, b) = double_integrator();
        // Desired poles at z = -1, -2 (continuous equivalent)
        let poles = [-1.0_f64, -2.0];
        let k = ackermann(&a, &b, &poles).expect("Should be controllable");
        // Closed-loop A-BK should have eigenvalues at desired poles
        // Verify by checking that characteristic polynomial matches
        // For now, just verify that K is computed
        assert!(k[0].abs() > 0.0 || k[1].abs() > 0.0);
    }

    #[test]
    fn ackermann_poles_stabilize_system() {
        let (a, b) = double_integrator();
        // Desired poles at z = 0.5, 0.5 (stable discrete-time poles)
        let poles = [0.5_f64, 0.5];
        let k = ackermann(&a, &b, &poles).unwrap();

        // Simulate closed-loop response
        let mut x = [1.0_f64, 0.0]; // initial position error
        for _ in 0..100 {
            let u = -(k[0] * x[0] + k[1] * x[1]);
            let x_new = [
                a.data[0][0] * x[0] + a.data[0][1] * x[1] + b.data[0][0] * u,
                a.data[1][0] * x[0] + a.data[1][1] * x[1] + b.data[1][0] * u,
            ];
            x = x_new;
        }
        assert!(x[0].abs() < 0.01, "Should converge: x={:.4}", x[0]);
    }

    #[test]
    fn uncontrollable_returns_none() {
        // Uncontrollable: B = [0, 0]^T
        let a = Matrix::<f64, 2, 2> {
            data: [[1.0, 0.0], [0.0, 1.0]],
        };
        let b = Matrix::<f64, 2, 1> {
            data: [[0.0], [0.0]],
        };
        let poles = [0.5_f64, 0.5];
        assert!(ackermann(&a, &b, &poles).is_none());
    }

    #[test]
    fn state_feedback_control_law() {
        let sf = StateFeedback::new([2.0_f64, 3.0]);
        let u = sf.control(&[1.0, 2.0], &[0.0, 0.0]);
        assert!((u + 8.0).abs() < 1e-10); // u = -(2*1 + 3*2) = -8
    }
}
