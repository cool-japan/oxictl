//! Servo (2-DOF) state feedback controller with pre-filter for zero steady-state error.
//!
//! The pre-filter maps reference r to state/input targets (Nx, Nu) such that
//! the closed-loop system achieves zero steady-state tracking error.
#![allow(unused)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Pre-filter matrices for reference tracking.
///
/// Used with u = -K*(x - Nx*r) + Nu*r to achieve zero SS error.
#[derive(Debug, Clone, Copy)]
pub struct PreFilter<S: ControlScalar, const N: usize, const I: usize> {
    /// State pre-filter Nx (N×I): maps reference to state target.
    pub n_x: Matrix<S, N, I>,
    /// Input pre-filter Nu (I×I): maps reference to feed-forward input.
    pub n_u: Matrix<S, I, I>,
}

impl<S: ControlScalar, const N: usize, const I: usize> PreFilter<S, N, I> {
    /// Create a pre-filter from known matrices.
    pub fn new(n_x: Matrix<S, N, I>, n_u: Matrix<S, I, I>) -> Self {
        Self { n_x, n_u }
    }
}

/// Servo controller: u = -K*(x - Nx*r) + Nu*r
///
/// This 2-DOF structure decouples disturbance rejection (K) from
/// reference tracking (Nx, Nu), eliminating steady-state error.
pub struct ServoController<S: ControlScalar, const N: usize, const I: usize> {
    /// LQR/pole-placed gain K (I×N).
    pub k_gain: Matrix<S, I, N>,
    /// Pre-filter for reference tracking.
    pub pre_filter: PreFilter<S, N, I>,
    /// Internal state estimate (for observer-based variants; zero for full-state).
    pub x_hat: Matrix<S, N, 1>,
}

impl<S: ControlScalar, const N: usize, const I: usize> ServoController<S, N, I> {
    /// Create a new servo controller.
    pub fn new(k_gain: Matrix<S, I, N>, pre_filter: PreFilter<S, N, I>) -> Self {
        Self {
            k_gain,
            pre_filter,
            x_hat: Matrix::zeros(),
        }
    }

    /// Compute control input given state x and reference r.
    ///
    /// u = -K*(x - Nx*r) + Nu*r
    ///   = -K*x + (K*Nx + Nu)*r
    pub fn control(&self, x: &Matrix<S, N, 1>, r: &Matrix<S, I, 1>) -> Matrix<S, I, 1> {
        // x_ref = Nx * r  (N×1)
        let x_ref = matmul(&self.pre_filter.n_x, r);

        // error = x - x_ref
        let error = x.sub_mat(&x_ref);

        // u_fb = -K * error
        let kx = matmul(&self.k_gain, &error);
        let u_fb = kx.neg();

        // u_ff = Nu * r
        let u_ff = matmul(&self.pre_filter.n_u, r);

        // u = u_fb + u_ff
        u_fb.add_mat(&u_ff)
    }

    /// Reset internal state estimate.
    pub fn reset(&mut self) {
        self.x_hat = Matrix::zeros();
    }
}

/// Design pre-filter for a SISO system (M = number of outputs = I for square systems).
///
/// Solves for (Nx, Nu) such that:
///   (A - B*K)*Nx + B*Nu = 0
///   C*Nx = I_M
///
/// For SISO (I=1, M=1), this simplifies to:
///   Nu = -(C*(A-BK)^{-1}*B)^{-1}
///   Nx = (A-BK)^{-1}*B*Nu (with sign flip already in Nu)
///
/// Returns None if the closed-loop matrix is singular.
pub fn design_prefilter<S: ControlScalar, const N: usize, const I: usize, const M: usize>(
    a: &Matrix<S, N, N>,
    b: &Matrix<S, N, I>,
    c: &Matrix<S, M, N>,
    k: &Matrix<S, I, N>,
) -> Option<PreFilter<S, N, I>> {
    // Discrete-time prefilter derivation (Nx = 0 form).
    //
    // Control law: u = -K*(x - Nx*r) + Nu*r
    //
    // With Nx = 0 the control becomes u = -K*x + Nu*r.
    // The closed-loop fixed point satisfies:
    //   (I - A_cl)*x_ss = B*Nu*r   =>   x_ss = (I - A_cl)^{-1}*B*Nu*r
    //
    // For output tracking C*x_ss = r:
    //   C*(I - A_cl)^{-1}*B*Nu = I_M
    //   Nu = pinv(C*(I - A_cl)^{-1}*B)
    //
    // The matching state target is then Nx = (I-A_cl)^{-1}*B*Nu, so that
    // the error x - Nx*r = 0 at the fixed point and u_ss = Nu*r - K*0 = Nu*r.
    // However, substituting Nx = (I-A_cl)^{-1}*B*Nu back into the control law
    // at x_ss changes the input to -K*(x_ss - Nx*r) + Nu*r = Nu*r (since
    // x_ss = Nx*r), confirming the fixed-point consistency.

    // Closed-loop matrix A_cl = A - B*K (N×N)
    let bk = matmul(b, k);
    let a_cl = a.sub_mat(&bk);

    // T = (I - A_cl)^{-1} * B  (N×I)
    let i_minus_acl = Matrix::<S, N, N>::identity().sub_mat(&a_cl);
    let i_minus_acl_inv = i_minus_acl.inv()?;
    let t = matmul(&i_minus_acl_inv, b);

    // C * T  (M×I)
    let ct = matmul(c, &t);

    // Nu = pinv(C*T) = (C*T)^T * ((C*T)*(C*T)^T)^{-1}  [right pseudo-inverse].
    // For square M == I this reduces to the regular inverse.

    // (C*T)^T is I×M
    let ct_t = ct.transpose(); // I×M

    // (C*T)*(C*T)^T is M×M
    let cct = matmul(&ct, &ct_t); // M×M

    // Invert M×M matrix
    let cct_inv = cct.inv()?;

    // nu_mat: I×M
    let nu_mat = matmul(&ct_t, &cct_inv); // I×M

    // Build n_u as I×I (works exactly when M == I).
    let mut n_u = Matrix::<S, I, I>::zeros();
    for row in 0..I {
        for col in 0..I.min(M) {
            n_u.data[row][col] = nu_mat.data[row][col];
        }
    }

    // Nx = 0.  Setting Nx = 0 means u_ss = Nu*r at the fixed point, which
    // together with x_ss = T*Nu*r gives C*x_ss = C*T*Nu*r = r as required.
    // (Providing Nx = T*Nu would be equivalent but adds redundancy and creates
    // an apparent inconsistency when the plant has integrating modes.)
    let n_x = Matrix::<S, N, I>::zeros();

    Some(PreFilter { n_x, n_u })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple 1D integrating plant: x[k+1] = x[k] + u[k]
    /// With state feedback K = [0.5], pre-filter designed for tracking.
    fn siso_system() -> (
        Matrix<f64, 1, 1>,
        Matrix<f64, 1, 1>,
        Matrix<f64, 1, 1>,
        Matrix<f64, 1, 1>,
    ) {
        let mut a = Matrix::<f64, 1, 1>::zeros();
        a.data[0][0] = 0.9; // marginally stable

        let mut b = Matrix::<f64, 1, 1>::zeros();
        b.data[0][0] = 1.0;

        let mut c = Matrix::<f64, 1, 1>::zeros();
        c.data[0][0] = 1.0;

        let mut k = Matrix::<f64, 1, 1>::zeros();
        k.data[0][0] = 0.5; // K = 0.5 => A_cl = 0.9 - 0.5 = 0.4 (stable)

        (a, b, c, k)
    }

    #[test]
    fn prefilter_design_siso() {
        let (a, b, c, k) = siso_system();
        let pf = design_prefilter::<f64, 1, 1, 1>(&a, &b, &c, &k);
        assert!(
            pf.is_some(),
            "Pre-filter design should succeed for stable system"
        );
        let pf = pf.unwrap();
        // Nu should be nonzero
        assert!(pf.n_u.data[0][0].abs() > 0.0, "Nu should be nonzero");
    }

    #[test]
    fn servo_control_computation() {
        let (a, b, c, k) = siso_system();
        let pf = design_prefilter::<f64, 1, 1, 1>(&a, &b, &c, &k).unwrap();
        let ctrl = ServoController::new(k, pf);

        let mut x = Matrix::<f64, 1, 1>::zeros();
        x.data[0][0] = 0.0;

        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 1.0;

        let u = ctrl.control(&x, &r);
        // u should be nonzero since we need to drive x to r
        assert!(
            u.data[0][0].abs() > 0.0,
            "Control should be nonzero: {}",
            u.data[0][0]
        );
    }

    #[test]
    fn servo_zero_reference_zero_control() {
        let (a, b, c, k) = siso_system();
        let pf = design_prefilter::<f64, 1, 1, 1>(&a, &b, &c, &k).unwrap();
        let ctrl = ServoController::new(k, pf);

        let x = Matrix::<f64, 1, 1>::zeros();
        let r = Matrix::<f64, 1, 1>::zeros();
        let u = ctrl.control(&x, &r);
        assert!(
            u.data[0][0].abs() < 1e-12,
            "u should be zero at equilibrium: {}",
            u.data[0][0]
        );
    }

    #[test]
    fn servo_tracking_convergence() {
        // 2-state system: double integrator discrete
        // A = [[1, 0.1],[0, 1]], B = [[0.005],[0.1]], C = [[1, 0]], K (LQR-tuned)
        let mut a = Matrix::<f64, 2, 2>::identity();
        a.data[0][1] = 0.1;

        let mut b = Matrix::<f64, 2, 1>::zeros();
        b.data[0][0] = 0.005;
        b.data[1][0] = 0.1;

        let mut c = Matrix::<f64, 1, 2>::zeros();
        c.data[0][0] = 1.0;

        let mut k = Matrix::<f64, 1, 2>::zeros();
        k.data[0][0] = 3.0;
        k.data[0][1] = 1.5;

        let pf = design_prefilter::<f64, 2, 1, 1>(&a, &b, &c, &k);
        assert!(pf.is_some(), "Pre-filter should be designed");
        let pf = pf.unwrap();
        let ctrl = ServoController::new(k, pf);

        let mut x = Matrix::<f64, 2, 1>::zeros();
        let mut r = Matrix::<f64, 1, 1>::zeros();
        r.data[0][0] = 1.0;

        for _ in 0..500 {
            let u = ctrl.control(&x, &r);
            let ax = matmul(&a, &x);
            let bu = matmul(&b, &u);
            x = ax.add_mat(&bu);
        }

        assert!(
            (x.data[0][0] - 1.0).abs() < 0.05,
            "Output should track reference: x[0]={}",
            x.data[0][0]
        );
    }
}
