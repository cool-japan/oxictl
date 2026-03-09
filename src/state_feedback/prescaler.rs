//! Steady-state pre-scaler (reference gain) for state feedback.
//!
//! Computes the feed-forward gain N_bar such that the closed-loop system
//! achieves unity DC gain: y_ss = r for step reference r.
#![allow(unused)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Compute the feed-forward (pre-scaler) gain N_bar for a SISO system.
///
/// N_bar = -(C * (A - B*K)^{-1} * B)^{-1}
///
/// Such that u = -K*x + N_bar*r achieves y_ss = r (zero steady-state error)
/// for a step reference r.
///
/// Returns None if (A-BK) or the final scalar expression is singular/zero.
pub fn compute_feed_forward_gain<S: ControlScalar, const N: usize, const I: usize>(
    a: &Matrix<S, N, N>,
    b: &Matrix<S, N, I>,
    c: &Matrix<S, 1, N>,
    k: &Matrix<S, I, N>,
) -> Option<S> {
    // Closed-loop A_cl = A - B*K  (N×N)
    let bk = matmul(b, k);
    let a_cl = a.sub_mat(&bk);

    // (A_cl - I)^{-1}: for discrete-time, DC gain uses (I - A_cl)^{-1}
    // DC gain = C * (I - A_cl)^{-1} * B
    let i_minus_acl = Matrix::<S, N, N>::identity().sub_mat(&a_cl);
    let i_minus_acl_inv = i_minus_acl.inv()?;

    // C * (I - A_cl)^{-1}  (1×N)
    let ci = matmul(c, &i_minus_acl_inv);

    // DC gain scalar: C * (I - A_cl)^{-1} * B  (1×I, then sum for SISO I=1)
    let dc_gain_mat = matmul(&ci, b); // 1×I

    // For SISO we take the [0][0] element
    let dc_gain = dc_gain_mat.data[0][0];

    if dc_gain.abs() < S::EPSILON * S::from_f64(1e6) {
        return None; // singular
    }

    Some(S::ONE / dc_gain)
}

/// Steady-state pre-scaler with optional integral action.
///
/// Combines a scalar feed-forward gain N_bar with an integrator to eliminate
/// steady-state error even in the presence of model uncertainty.
///
/// Control law: u = -K*x + N_bar*r + Ki * integrator
pub struct Prescaler<S: ControlScalar, const N: usize, const I: usize> {
    /// Pre-filter gain N_bar (scalar).
    pub n_bar: S,
    /// LQR/pole-placed gain K (I×N).
    pub k_gain: Matrix<S, I, N>,
    /// Integral state (accumulated tracking error).
    pub integrator: S,
    /// Integral gain Ki.
    pub ki: S,
}

impl<S: ControlScalar, const N: usize, const I: usize> Prescaler<S, N, I> {
    /// Create a new prescaler.
    ///
    /// - `n_bar`: pre-computed DC-gain pre-scaler (use `compute_feed_forward_gain`).
    /// - `k_gain`: state feedback gain.
    /// - `ki`: integral gain (set to zero to disable integration).
    pub fn new(n_bar: S, k_gain: Matrix<S, I, N>, ki: S) -> Self {
        Self {
            n_bar,
            k_gain,
            integrator: S::ZERO,
            ki,
        }
    }

    /// Compute control and update the integrator.
    ///
    /// - `x`: current state vector (N×1 column).
    /// - `r`: scalar reference value.
    /// - `y`: scalar measured output.
    /// - `dt`: time step.
    ///
    /// Returns the scalar control signal u[0].
    pub fn update(&mut self, x: &Matrix<S, N, 1>, r: S, y: S, dt: S) -> S {
        // State feedback: u_fb = -K * x  (I×1), take first element for SISO
        let kx = matmul(&self.k_gain, x);
        let u_fb = -kx.data[0][0];

        // Feed-forward: u_ff = N_bar * r
        let u_ff = self.n_bar * r;

        // Integral action: u_int = Ki * integrator
        let u_int = self.ki * self.integrator;

        // Total control
        let u = u_fb + u_ff + u_int;

        // Update integrator with error e = r - y
        let error = r - y;
        self.integrator += error * dt;

        u
    }

    /// Reset the integrator to zero.
    pub fn reset(&mut self) {
        self.integrator = S::ZERO;
    }

    /// Manually set the integrator state (for anti-windup or initialisation).
    pub fn set_integrator(&mut self, val: S) {
        self.integrator = val;
    }

    /// Compute feed-forward contribution only: N_bar * r.
    pub fn feed_forward(&self, r: S) -> S {
        self.n_bar * r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D plant: x[k+1] = a*x[k] + b*u[k], y = x
    fn siso_plant() -> (
        Matrix<f64, 1, 1>,
        Matrix<f64, 1, 1>,
        Matrix<f64, 1, 1>,
        Matrix<f64, 1, 1>,
    ) {
        let mut a = Matrix::<f64, 1, 1>::zeros();
        a.data[0][0] = 0.9;

        let mut b = Matrix::<f64, 1, 1>::zeros();
        b.data[0][0] = 1.0;

        let mut c = Matrix::<f64, 1, 1>::zeros();
        c.data[0][0] = 1.0;

        let mut k = Matrix::<f64, 1, 1>::zeros();
        k.data[0][0] = 0.5; // A_cl = 0.9 - 0.5 = 0.4

        (a, b, c, k)
    }

    #[test]
    fn feed_forward_gain_computed() {
        let (a, b, c, k) = siso_plant();
        let n_bar = compute_feed_forward_gain(&a, &b, &c, &k);
        assert!(n_bar.is_some(), "N_bar should be computable");
        let n_bar = n_bar.unwrap();
        // For A=0.9, B=1, K=0.5: A_cl=0.4, (I-A_cl)=0.6, (I-A_cl)^{-1}=1/0.6
        // DC gain = C*(I-A_cl)^{-1}*B = 1/0.6 ≈ 1.667; N_bar = 0.6
        assert!(
            (n_bar - 0.6).abs() < 1e-8,
            "N_bar should be ~0.6: {}",
            n_bar
        );
    }

    #[test]
    fn prescaler_zero_ref_zero_output() {
        let (a, b, c, k) = siso_plant();
        let n_bar = compute_feed_forward_gain(&a, &b, &c, &k).unwrap();
        let mut ps = Prescaler::new(n_bar, k, 0.0_f64);

        let x = Matrix::<f64, 1, 1>::zeros();
        let u = ps.update(&x, 0.0, 0.0, 0.01);
        assert!(u.abs() < 1e-12, "Control should be zero: {}", u);
    }

    #[test]
    fn prescaler_tracking_with_integral() {
        let (a, b, c, k) = siso_plant();
        let n_bar = compute_feed_forward_gain(&a, &b, &c, &k).unwrap();
        let ki = 0.1_f64;
        let mut ps = Prescaler::new(n_bar, k, ki);

        let mut x = Matrix::<f64, 1, 1>::zeros();
        let r = 1.0_f64;
        let dt = 0.01_f64;

        for _ in 0..500 {
            let y = x.data[0][0];
            let u = ps.update(&x, r, y, dt);
            // Plant step: x = a*x + b*u
            x.data[0][0] = a.data[0][0] * x.data[0][0] + b.data[0][0] * u;
        }

        assert!(
            (x.data[0][0] - r).abs() < 0.05,
            "Output should track reference with integral: y={}, r={}",
            x.data[0][0],
            r
        );
    }

    #[test]
    fn prescaler_reset_clears_integrator() {
        let (a, b, c, k) = siso_plant();
        let n_bar = compute_feed_forward_gain(&a, &b, &c, &k).unwrap();
        let mut ps = Prescaler::new(n_bar, k, 0.1_f64);

        // Accumulate integrator
        let x = Matrix::<f64, 1, 1>::zeros();
        for _ in 0..10 {
            ps.update(&x, 1.0, 0.0, 0.01);
        }
        assert!(ps.integrator.abs() > 0.0);

        ps.reset();
        assert!(
            ps.integrator.abs() < 1e-15,
            "Integrator should be zero after reset"
        );
    }
}
