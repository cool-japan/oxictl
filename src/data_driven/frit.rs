#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::needless_range_loop)]

//! Fictitious Reference Iterative Tuning (FRIT) — Soma et al.
//!
//! Tunes a PID controller using closed-loop experimental data
//! `{r[k], u[k], y[k]}` without an explicit plant model.
//!
//! # Method
//!
//! Define the FRIT cost:
//! ```text
//! J_FRIT(θ) = (1/T) · ∑_k δy[k]²
//! ```
//! where `δy[k]` is the output of the reference model `M(z)` driven by the
//! input discrepancy `v[k] = u_data[k] - u_θ[k]`:
//! ```text
//! δy[k] = m · δy[k-1] + (1-m) · v[k]
//! u_θ[k] = Kp·e[k] + Ki·I_e[k] + Kd·D_e[k],   e[k] = r[k] - y[k]
//! ```
//!
//! Analytic gradients are derived by propagating sensitivities through `M(z)`.

use crate::core::scalar::ControlScalar;
use crate::data_driven::vrft::DataDrivenError;

/// FRIT-based iterative PID tuner using closed-loop data.
///
/// Generic over scalar type `S` (`f32` or `f64`) and `DATA_LEN`.
///
/// # Example
/// ```rust,ignore
/// let mut tuner = FritTuner::<f64, 200>::new(1.0, 0.0, 0.0, 0.9, 0.01, 0.1)?;
/// for _ in 0..20 {
///     let cost = tuner.step(&r_data, &u_data, &y_data)?;
/// }
/// let (kp, ki, kd) = tuner.parameters();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FritTuner<S, const DATA_LEN: usize> {
    /// Proportional gain.
    kp: S,
    /// Integral gain.
    ki: S,
    /// Derivative gain.
    kd: S,
    /// Reference-model pole `m ∈ (0, 1)`.
    m: S,
    /// Sampling period (seconds).
    dt: S,
    /// Gradient-descent step size.
    mu: S,
    /// Number of completed gradient steps.
    iteration: usize,
}

impl<S: ControlScalar, const DATA_LEN: usize> FritTuner<S, DATA_LEN> {
    /// Construct a FRIT tuner with initial PID parameters.
    ///
    /// # Parameters
    /// - `kp0`, `ki0`, `kd0`: Initial PID gains.
    /// - `m`:  Reference-model pole, must satisfy `0 < m < 1`.
    /// - `dt`: Sampling period, must be strictly positive.
    /// - `mu`: Gradient-descent step size, must be strictly positive.
    pub fn new(kp0: S, ki0: S, kd0: S, m: S, dt: S, mu: S) -> Result<Self, DataDrivenError> {
        if m <= S::ZERO || m >= S::ONE {
            return Err(DataDrivenError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(DataDrivenError::InvalidParameter);
        }
        if mu <= S::ZERO {
            return Err(DataDrivenError::InvalidParameter);
        }
        Ok(Self {
            kp: kp0,
            ki: ki0,
            kd: kd0,
            m,
            dt,
            mu,
            iteration: 0,
        })
    }

    /// Compute the FRIT cost `J` and its gradient, then perform one
    /// gradient-descent step.
    ///
    /// # Parameters
    /// - `r_data`: Reference sequence used in the original experiment.
    /// - `u_data`: Plant input from the original experiment.
    /// - `y_data`: Plant output from the original experiment.
    ///
    /// # Returns
    /// The FRIT cost `J_FRIT` **before** the parameter update.
    pub fn step(
        &mut self,
        r_data: &[S; DATA_LEN],
        u_data: &[S; DATA_LEN],
        y_data: &[S; DATA_LEN],
    ) -> Result<S, DataDrivenError> {
        if DATA_LEN < 2 {
            return Err(DataDrivenError::NotEnoughData);
        }

        let (cost, grad_kp, grad_ki, grad_kd) =
            self.compute_cost_and_gradient(r_data, u_data, y_data);

        let mu = self.mu;
        self.kp -= mu * grad_kp;
        self.ki -= mu * grad_ki;
        self.kd -= mu * grad_kd;
        self.iteration += 1;

        Ok(cost)
    }

    /// Compute the FRIT cost without updating parameters.
    pub fn frit_cost(
        &self,
        r_data: &[S; DATA_LEN],
        u_data: &[S; DATA_LEN],
        y_data: &[S; DATA_LEN],
    ) -> Result<S, DataDrivenError> {
        if DATA_LEN < 2 {
            return Err(DataDrivenError::NotEnoughData);
        }
        let (cost, _, _, _) = self.compute_cost_and_gradient(r_data, u_data, y_data);
        Ok(cost)
    }

    /// Core computation: returns `(J, ∂J/∂Kp, ∂J/∂Ki, ∂J/∂Kd)`.
    ///
    /// # Algorithm
    ///
    /// 1. Compute PID output with current gains: `u_θ[k]`.
    /// 2. Discrepancy: `v[k] = u_data[k] - u_θ[k]`.
    /// 3. Filter `v` through reference model `M(z)` to get `δy[k]`:
    ///    `δy[k] = m·δy[k-1] + (1-m)·v[k]`.
    /// 4. Cost: `J = (1/T)·∑ δy[k]²`.
    /// 5. Gradient via sensitivity propagation:
    ///    `∂δy/∂Kp[k] = m·∂δy/∂Kp[k-1] + (1-m)·(∂u_θ/∂Kp[k])·(-1)`
    ///    (negative because `v = u_data - u_θ` and `∂v/∂Kp = -∂u_θ/∂Kp`).
    ///    `∂u_θ/∂Kp[k] = e[k]`,  `∂u_θ/∂Ki[k] = I_e[k]`,  `∂u_θ/∂Kd[k] = D_e[k]`.
    fn compute_cost_and_gradient(
        &self,
        r_data: &[S; DATA_LEN],
        u_data: &[S; DATA_LEN],
        y_data: &[S; DATA_LEN],
    ) -> (S, S, S, S) {
        let m = self.m;
        let one_minus_m = S::ONE - m;
        let dt = self.dt;
        let t_inv = S::ONE / S::from_f64(DATA_LEN as f64);

        let kp = self.kp;
        let ki = self.ki;
        let kd = self.kd;

        // Accumulators for cost and gradients.
        let mut cost = S::ZERO;
        let mut grad_kp = S::ZERO;
        let mut grad_ki = S::ZERO;
        let mut grad_kd = S::ZERO;

        // State variables.
        let mut dy = S::ZERO; // δy[k-1]
        let mut s_kp = S::ZERO; // ∂δy/∂Kp[k-1]
        let mut s_ki = S::ZERO; // ∂δy/∂Ki[k-1]
        let mut s_kd = S::ZERO; // ∂δy/∂Kd[k-1]

        let mut integral_e = S::ZERO;
        let mut e_prev = S::ZERO;

        for k in 0..DATA_LEN {
            // Tracking error.
            let e_k = r_data[k] - y_data[k];

            // Running integral and derivative of e.
            integral_e += e_k * dt;
            let deriv_e = if k == 0 { S::ZERO } else { (e_k - e_prev) / dt };

            // PID output with current gains.
            let u_theta = kp * e_k + ki * integral_e + kd * deriv_e;

            // Input discrepancy.
            let v_k = u_data[k] - u_theta;

            // δy through reference model M(z): δy[k] = m·δy[k-1] + (1-m)·v[k].
            dy = m * dy + one_minus_m * v_k;

            // Sensitivity propagation (chain rule through M):
            // ∂v/∂Kp = -e_k,  ∂v/∂Ki = -integral_e,  ∂v/∂Kd = -deriv_e.
            s_kp = m * s_kp + one_minus_m * (-e_k);
            s_ki = m * s_ki + one_minus_m * (-integral_e);
            s_kd = m * s_kd + one_minus_m * (-deriv_e);

            // Accumulate cost: J = (1/T)·∑ δy²
            cost += dy * dy;

            // Accumulate gradient: ∂J/∂θ_i = (2/T)·∑ δy · ∂δy/∂θ_i
            grad_kp += dy * s_kp;
            grad_ki += dy * s_ki;
            grad_kd += dy * s_kd;

            e_prev = e_k;
        }

        let two_t_inv = S::TWO * t_inv;
        (
            cost * t_inv,
            grad_kp * two_t_inv,
            grad_ki * two_t_inv,
            grad_kd * two_t_inv,
        )
    }

    /// Return current PID parameters `(Kp, Ki, Kd)`.
    pub fn parameters(&self) -> (S, S, S) {
        (self.kp, self.ki, self.kd)
    }

    /// Number of gradient steps completed so far.
    pub fn iteration(&self) -> usize {
        self.iteration
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate a simple closed-loop experiment.
    ///
    /// Plant: `y[k] = a·y[k-1] + b·u[k-1]`
    /// Initial controller: pure proportional `u[k] = Kp0·(r[k]-y[k])`
    fn generate_closed_loop_data<const N: usize>(
        a: f64,
        b: f64,
        kp0: f64,
        r_val: f64,
    ) -> ([f64; N], [f64; N], [f64; N]) {
        let r_data = [r_val; N];
        let mut u_data = [0.0_f64; N];
        let mut y_data = [0.0_f64; N];
        for k in 1..N {
            y_data[k] = a * y_data[k - 1] + b * u_data[k - 1];
            let e = r_data[k] - y_data[k];
            u_data[k] = kp0 * e;
            // Clamp to prevent blow-up.
            u_data[k] = u_data[k].clamp(-10.0, 10.0);
        }
        // Suppress unused mut warning for r_data.
        let _ = r_data[0];
        (r_data, u_data, y_data)
    }

    #[test]
    fn frit_cost_is_computable() {
        const N: usize = 100;
        let (r_data, u_data, y_data) = generate_closed_loop_data::<N>(0.7, 0.3, 1.0, 1.0);
        let tuner = FritTuner::<f64, N>::new(1.0, 0.0, 0.0, 0.8, 0.01, 0.1).expect("valid");
        let cost = tuner.frit_cost(&r_data, &u_data, &y_data).expect("cost ok");
        assert!(cost.is_finite(), "FRIT cost must be finite, got {cost}");
        assert!(cost >= 0.0, "FRIT cost must be non-negative, got {cost}");
    }

    #[test]
    fn frit_parameters_update_after_step() {
        const N: usize = 80;
        let (r_data, u_data, y_data) = generate_closed_loop_data::<N>(0.7, 0.3, 1.0, 1.0);

        let mut tuner = FritTuner::<f64, N>::new(1.0, 0.1, 0.02, 0.8, 0.01, 0.5).expect("valid");
        let (kp0, ki0, kd0) = tuner.parameters();

        tuner.step(&r_data, &u_data, &y_data).expect("step ok");

        let (kp1, ki1, kd1) = tuner.parameters();
        // At least one parameter must have changed (unless gradient is exactly zero,
        // which is extremely unlikely for non-trivial data).
        let changed =
            (kp1 - kp0).abs() > 1e-14 || (ki1 - ki0).abs() > 1e-14 || (kd1 - kd0).abs() > 1e-14;
        assert!(changed, "Parameters should update after a step");
    }

    #[test]
    fn frit_step_size_effect() {
        // Larger mu → larger parameter change per step.
        const N: usize = 100;
        let (r_data, u_data, y_data) = generate_closed_loop_data::<N>(0.7, 0.3, 0.5, 1.0);

        let mut tuner_small =
            FritTuner::<f64, N>::new(1.0, 0.0, 0.0, 0.8, 0.01, 0.01).expect("valid");
        let mut tuner_large =
            FritTuner::<f64, N>::new(1.0, 0.0, 0.0, 0.8, 0.01, 1.0).expect("valid");

        tuner_small.step(&r_data, &u_data, &y_data).expect("step");
        tuner_large.step(&r_data, &u_data, &y_data).expect("step");

        let (kp_small, _, _) = tuner_small.parameters();
        let (kp_large, _, _) = tuner_large.parameters();

        let change_small = (kp_small - 1.0_f64).abs();
        let change_large = (kp_large - 1.0_f64).abs();

        assert!(
            change_large >= change_small,
            "Larger mu should produce larger change: small={change_small}, large={change_large}"
        );
    }

    #[test]
    fn frit_cost_does_not_increase_unboundedly_over_iterations() {
        // Run FRIT on a simple stable plant for several iterations and verify
        // the cost trajectory stays finite and the final cost is not
        // dramatically worse than the initial cost.
        const N: usize = 120;
        let (r_data, u_data, y_data) = generate_closed_loop_data::<N>(0.7, 0.3, 1.0, 1.0);

        let mut tuner = FritTuner::<f64, N>::new(1.0, 0.0, 0.0, 0.85, 0.01, 0.05).expect("valid");

        let cost0 = tuner
            .frit_cost(&r_data, &u_data, &y_data)
            .expect("initial cost");

        let mut final_cost = cost0;
        for _ in 0..10 {
            final_cost = tuner.step(&r_data, &u_data, &y_data).expect("step");
        }

        assert!(
            final_cost.is_finite(),
            "Cost must remain finite, got {final_cost}"
        );
        // With a small step size, cost should not blow up by more than 100×.
        assert!(
            final_cost <= cost0 * 100.0 + 1.0,
            "Cost should stay bounded: initial={cost0}, final={final_cost}"
        );
    }

    #[test]
    fn frit_iteration_counter() {
        const N: usize = 60;
        let (r_data, u_data, y_data) = generate_closed_loop_data::<N>(0.8, 0.2, 1.0, 1.0);

        let mut tuner = FritTuner::<f64, N>::new(0.5, 0.0, 0.0, 0.9, 0.01, 0.1).expect("valid");

        assert_eq!(tuner.iteration(), 0);
        for i in 1..=5 {
            tuner.step(&r_data, &u_data, &y_data).expect("step");
            assert_eq!(
                tuner.iteration(),
                i,
                "iteration counter mismatch at step {i}"
            );
        }
    }

    #[test]
    fn frit_invalid_params_rejected() {
        // m out of range.
        assert_eq!(
            FritTuner::<f64, 100>::new(1.0, 0.0, 0.0, 0.0, 0.01, 0.1),
            Err(DataDrivenError::InvalidParameter)
        );
        assert_eq!(
            FritTuner::<f64, 100>::new(1.0, 0.0, 0.0, 1.0, 0.01, 0.1),
            Err(DataDrivenError::InvalidParameter)
        );
        // dt = 0.
        assert_eq!(
            FritTuner::<f64, 100>::new(1.0, 0.0, 0.0, 0.8, 0.0, 0.1),
            Err(DataDrivenError::InvalidParameter)
        );
        // mu = 0.
        assert_eq!(
            FritTuner::<f64, 100>::new(1.0, 0.0, 0.0, 0.8, 0.01, 0.0),
            Err(DataDrivenError::InvalidParameter)
        );
        // Valid.
        assert!(FritTuner::<f64, 100>::new(1.0, 0.0, 0.0, 0.8, 0.01, 0.1).is_ok());
    }

    #[test]
    fn frit_f32_works() {
        const N: usize = 50;
        let r_data = [1.0_f32; N];
        let u_data = [0.5_f32; N];
        let y_data = {
            let mut y = [0.0_f32; N];
            for k in 1..N {
                y[k] = 0.7 * y[k - 1] + 0.3 * u_data[k - 1];
            }
            y
        };
        let mut tuner = FritTuner::<f32, N>::new(1.0, 0.0, 0.0, 0.8, 0.01, 0.05).expect("valid");
        let cost = tuner.step(&r_data, &u_data, &y_data).expect("step");
        assert!(cost.is_finite());
    }
}
