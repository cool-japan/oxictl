use crate::core::matrix::{matvec, outer, Matrix};
use crate::core::scalar::ControlScalar;

/// Self-Tuning Regulator (STR) for SISO discrete-time plants.
///
/// Plant model (ARX):
///   A(q)*y[k] = B(q)*u[k] + e[k]
///
/// With A(q) = 1 + a1*q^{-1} + a2*q^{-2}  (nb_a poles)
///      B(q) =     b1*q^{-1} + b2*q^{-2}  (nb_b zeros)
///
/// Online identification: RLS estimates [a1, a2, b1, b2] from I/O data.
///
/// Minimum Variance Control (MV):
///   u[k] = (r[k] - a1*y[k] - a2*y[k-1]) / b1
///   This minimizes E[(y[k+1] - r[k+1])²]  for one-step-ahead prediction.
///
/// Pole-Zero Placement:
///   Assigns closed-loop poles to desired locations via Diophantine equation.
///
/// # Type parameters
/// - `S`: scalar type
/// - `NA`: AR order (number of a-coefficients)
/// - `NB`: MA order (number of b-coefficients)
/// - `NP`: NA + NB (total parameters for RLS)
#[derive(Debug)]
pub struct SelfTuningRegulator<S: ControlScalar, const NA: usize, const NB: usize, const NP: usize>
{
    /// Estimated plant parameters: [a1..aNA, b1..bNB].
    pub theta: [S; NP],
    /// RLS covariance matrix.
    pub p: Matrix<S, NP, NP>,
    /// RLS forgetting factor.
    pub lambda: S,
    /// Output history y[k-1], y[k-2], …
    y_hist: [S; NA],
    /// Input history u[k-1], u[k-2], …
    u_hist: [S; NB],
    /// Control output saturation limit.
    pub u_limit: S,
    /// Desired closed-loop poles for pole-placement mode.
    pub desired_poles: [S; NA],
    /// Control mode selection.
    pub mode: StrMode,
    /// Minimum |b1| to prevent divide-by-zero in MV control.
    pub b_min: S,
    /// Sample counter for diagnostics.
    sample_count: u32,
    /// One-step-ahead prediction error from the most recent `update_estimate` call.
    last_prediction_error: S,
}

/// Control law selection for the STR.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StrMode {
    /// Minimum variance control: minimizes one-step-ahead prediction error.
    MinimumVariance,
    /// Pole-zero placement: closed-loop poles at `desired_poles`.
    PolePlacement,
}

impl<S: ControlScalar, const NA: usize, const NB: usize, const NP: usize>
    SelfTuningRegulator<S, NA, NB, NP>
{
    /// Create a new STR.
    ///
    /// `lambda`: RLS forgetting factor (0 < λ ≤ 1).
    /// `p0`: initial RLS covariance scale (large = uninformative prior, e.g. 1e6).
    /// `u_limit`: control output saturation.
    /// `b_min`: minimum |b1| threshold to avoid singularity.
    ///
    /// Panics (in debug) if NP != NA + NB.
    pub fn new(lambda: S, p0: S, u_limit: S, b_min: S) -> Self {
        debug_assert_eq!(NP, NA + NB, "NP must equal NA + NB");
        Self {
            theta: [S::ZERO; NP],
            p: Matrix::<S, NP, NP>::identity().scale(p0),
            lambda,
            y_hist: [S::ZERO; NA],
            u_hist: [S::ZERO; NB],
            u_limit,
            desired_poles: [S::ZERO; NA],
            mode: StrMode::MinimumVariance,
            b_min,
            sample_count: 0,
            last_prediction_error: S::ZERO,
        }
    }

    /// Update RLS estimate with current I/O pair.
    ///
    /// `y` is the measured plant output at time k.
    /// `u_prev` is the control input u[k-1] that was applied to produce `y`.
    ///
    /// Must be called before `compute_control` to incorporate measurement.
    pub fn update_estimate(&mut self, y: S, u_prev: S) {
        // Shift u_hist and insert u_prev BEFORE building the regressor so that
        // phi[NA..] = [u[k-1], u[k-2], …] (not one step stale).
        for i in (1..NB).rev() {
            self.u_hist[i] = self.u_hist[i - 1];
        }
        if NB > 0 {
            self.u_hist[0] = u_prev;
        }

        // Build regressor: φ = [-y[k-1], …, -y[k-NA], u[k-1], …, u[k-NB]]
        let phi = self.build_regressor();

        // RLS update (inline to avoid borrow issues)
        let y_hat: S = phi
            .iter()
            .zip(self.theta.iter())
            .map(|(&p, &t)| p * t)
            .fold(S::ZERO, |a, b| a + b);
        let error = y - y_hat;

        let p_phi = matvec(&self.p, &phi);
        let phi_p_phi: S = phi
            .iter()
            .zip(p_phi.iter())
            .map(|(&a, &b)| a * b)
            .fold(S::ZERO, |acc, x| acc + x);
        let denom = self.lambda + phi_p_phi;

        if denom.abs() >= S::EPSILON {
            let k: [S; NP] = core::array::from_fn(|i| p_phi[i] / denom);

            for (i, &ki) in k.iter().enumerate() {
                self.theta[i] += ki * error;
            }

            let k_phi_t: Matrix<S, NP, NP> = outer(&k, &phi);
            let mut kp = Matrix::<S, NP, NP>::zeros();
            for r in 0..NP {
                for c in 0..NP {
                    kp.data[r][c] = k_phi_t.data[r]
                        .iter()
                        .zip(self.p.data.iter())
                        .map(|(&kf, pr)| kf * pr[c])
                        .fold(S::ZERO, |acc, x| acc + x);
                }
            }
            self.p = self.p.sub_mat(&kp).scale(S::ONE / self.lambda);
        }

        // Shift y history (newest measurement to front) after RLS update
        for i in (1..NA).rev() {
            self.y_hist[i] = self.y_hist[i - 1];
        }
        if NA > 0 {
            self.y_hist[0] = y;
        }
        // NOTE: u_hist was already shifted at the top of this function.

        // Store the pre-update prediction error for external access.
        self.last_prediction_error = error;
        self.sample_count += 1;
    }

    /// Compute the control output for the given reference `r`.
    ///
    /// Returns saturated control signal u[k].
    pub fn compute_control(&self, r: S) -> S {
        let u = match self.mode {
            StrMode::MinimumVariance => self.minimum_variance_law(r),
            StrMode::PolePlacement => self.pole_placement_law(r),
        };
        u.clamp_val(-self.u_limit, self.u_limit)
    }

    /// Minimum variance control law.
    ///
    /// Derives u[k] by setting the one-step-ahead ARX prediction equal to r:
    ///
    ///   y_hat[k+1] = Σ_i theta[i]*(-y[k-i]) + Σ_j theta[NA+j]*u[k-j]  = r
    ///
    /// Solving for u[k] (the j=0 term in the MA part):
    ///
    ///   u[k] = (r + Σ_i theta[i]*y[k-i] - Σ_{j≥1} theta[NA+j]*u[k-j]) / theta[NA]
    ///
    /// Note: theta[i] ≈ -a_i_plant (negative, because phi[i] = -y[k-i]).
    fn minimum_variance_law(&self, r: S) -> S {
        // AR contribution: Σ_i theta[i] * y[k-i]   (theta[i] = -a_i_plant)
        let mut ar_sum = S::ZERO;
        for i in 0..NA {
            ar_sum += self.theta[i] * self.y_hist[i];
        }

        // b1 = theta[NA] is the first MA coefficient (≈ b1_plant > 0)
        let b1 = if NB > 0 { self.theta[NA] } else { S::ONE };
        let b1_guarded = if b1.abs() < self.b_min {
            if b1 >= S::ZERO {
                self.b_min
            } else {
                -self.b_min
            }
        } else {
            b1
        };

        // Higher-order MA contributions from past inputs (j ≥ 1 in MA part)
        // u_hist[j-1] = u[k-j] after u_hist has been shifted with u_prev
        let mut bu_sum = S::ZERO;
        for j in 1..NB {
            bu_sum += self.theta[NA + j] * self.u_hist[j];
        }

        // u[k] = (r + ar_sum - bu_sum) / b1
        (r + ar_sum - bu_sum) / b1_guarded
    }

    /// Pole placement control law (simplified first-order case).
    ///
    /// Solves: A(q)*S(q) + B(q)*R(q) = T(q)*Ac(q)
    /// For NA=NB=1: u = (t0*r - s0*y) / r0
    ///
    /// For higher orders a full Diophantine solver would be needed;
    /// here we use a direct state-feedback approximation.
    fn pole_placement_law(&self, r: S) -> S {
        if NA == 0 || NB == 0 {
            return self.minimum_variance_law(r);
        }

        // Desired closed-loop characteristic polynomial coefficient
        // Ac(q) = q^NA + pc1*q^{NA-1} + ... + pcNA
        // For NA=1: Ac(z) = z - p1, so ac1 = -p1
        let pc1 = if NA > 0 {
            self.desired_poles[0]
        } else {
            S::ZERO
        };

        // A(q) estimated: A(z) = z + a1
        let a1 = if NA > 0 { self.theta[0] } else { S::ZERO };
        let b1 = if NB > 0 { self.theta[NA] } else { S::ONE };

        // Diophantine for order 1:
        //   (z + a1)*(1) + b1*r0 = z - pc1
        //   => r0 = (-pc1 - a1) / b1
        //   feedforward t0 = -pc1
        let b1_guarded = if b1.abs() < self.b_min {
            if b1 >= S::ZERO {
                self.b_min
            } else {
                -self.b_min
            }
        } else {
            b1
        };

        let r0 = (-pc1 - a1) / b1_guarded;
        let t0 = -pc1;

        let y_now = if NA > 0 { self.y_hist[0] } else { S::ZERO };
        t0 * r - r0 * y_now
    }

    /// Build the regressor vector from current history.
    fn build_regressor(&self) -> [S; NP] {
        let mut phi = [S::ZERO; NP];
        // AR part: -y[k-1], -y[k-2], …
        for (phi_i, &y_i) in phi.iter_mut().zip(self.y_hist.iter()).take(NA) {
            *phi_i = -y_i;
        }
        // MA part: u[k-1], u[k-2], …
        for (phi_j, &u_j) in phi[NA..].iter_mut().zip(self.u_hist.iter()).take(NB) {
            *phi_j = u_j;
        }
        phi
    }

    /// Return the estimated plant parameters.
    /// Layout: [a1, …, aNA, b1, …, bNB]
    pub fn parameters(&self) -> &[S; NP] {
        &self.theta
    }

    /// Return the estimated a-coefficients.
    pub fn a_params(&self) -> &[S] {
        &self.theta[..NA]
    }

    /// Return the estimated b-coefficients.
    pub fn b_params(&self) -> &[S] {
        &self.theta[NA..]
    }

    /// Reset estimator and controller states.
    pub fn reset(&mut self, p0: S) {
        self.theta = [S::ZERO; NP];
        self.p = Matrix::<S, NP, NP>::identity().scale(p0);
        self.y_hist = [S::ZERO; NA];
        self.u_hist = [S::ZERO; NB];
        self.sample_count = 0;
        self.last_prediction_error = S::ZERO;
    }

    /// Number of samples processed.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// One-step-ahead prediction error from the most recent `update_estimate` call.
    ///
    /// Returns the innovation `y[k] - y_hat[k]` that was computed and used in the
    /// last RLS update. This is the correct residual after the model has converged.
    ///
    /// Note: the `y` argument is ignored; it is kept for API compatibility.
    pub fn prediction_error(&self, _y: S) -> S {
        self.last_prediction_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plant: y[k] = 0.8*y[k-1] + 0.5*u[k-1]  => A=[0.8?]... ARX form:
    /// y[k] - 0.8*y[k-1] = 0.5*u[k-1]
    /// theta = [-a1, b1] where a1 stored as theta[0] via regressor = -y[k-1]
    /// so theta[0] = 0.8 (since phi[0] = -y[k-1], and ARX: y = theta[0]*(-y[k-1]) + theta[1]*u[k-1])
    /// Wait: y[k] = 0.8*y[k-1] + 0.5*u[k-1]
    ///            = -(-0.8)*y[k-1] + 0.5*u[k-1]
    /// phi = [-y[k-1], u[k-1]]
    /// theta = [a1_est, b1_est] where model: y = a1_est*(-y[k-1]) + b1_est*u[k-1]
    ///   => a1_est = -0.8, b1_est = 0.5
    #[test]
    fn identifies_first_order_plant() {
        // NP = NA + NB = 1 + 1 = 2
        let mut str_ctrl = SelfTuningRegulator::<f64, 1, 1, 2>::new(1.0, 1e6, 100.0, 0.01);
        let mut y = 0.0_f64;
        let mut u = 0.0_f64;

        for k in 1..=500 {
            // Plant simulation
            let y_next = 0.8 * y + 0.5 * u;
            str_ctrl.update_estimate(y_next, u);
            // Use a fixed persistently exciting input
            u = if k % 10 < 5 { 1.0 } else { -1.0 };
            y = y_next;
        }

        // theta[0] ≈ -(-0.8) = phi[0]*theta[0] contributes -theta[0]*y[k-1]
        // ARX: y[k] = theta[0]*phi[0] + theta[1]*phi[1]
        //           = theta[0]*(-y[k-1]) + theta[1]*u[k-1]
        // Plant: y[k] = 0.8*y[k-1] + 0.5*u[k-1]
        // So theta[0] = -0.8, theta[1] = 0.5
        let a1 = str_ctrl.a_params()[0];
        let b1 = str_ctrl.b_params()[0];
        assert!(
            (a1 - (-0.8)).abs() < 0.05,
            "a1 est={:.4} (expected -0.8)",
            a1
        );
        assert!((b1 - 0.5).abs() < 0.05, "b1 est={:.4} (expected 0.5)", b1);
    }

    #[test]
    fn mv_control_tracks_reference() {
        let mut str_ctrl = SelfTuningRegulator::<f64, 1, 1, 2>::new(0.98, 1e6, 50.0, 0.05);
        let mut y = 0.0_f64;
        let mut u = 0.0_f64;
        let r = 1.0_f64;

        for _ in 0..1000 {
            let y_next = 0.7 * y + 0.8 * u;
            str_ctrl.update_estimate(y_next, u);
            u = str_ctrl.compute_control(r);
            y = y_next;
        }

        assert!(
            (y - r).abs() < 0.2,
            "STR MV tracking: y={:.4}, r={:.4}",
            y,
            r
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut str_ctrl = SelfTuningRegulator::<f64, 1, 1, 2>::new(1.0, 1e4, 100.0, 0.01);
        let mut y = 0.0_f64;
        let u = 1.0_f64;
        for _ in 0..100 {
            let y_next = 0.8 * y + 0.5 * u;
            str_ctrl.update_estimate(y_next, u);
            y = y_next;
        }
        str_ctrl.reset(1e4);
        assert_eq!(str_ctrl.sample_count(), 0);
        assert!((str_ctrl.parameters()[0]).abs() < 1e-10);
    }

    #[test]
    fn control_saturates() {
        let str_ctrl = SelfTuningRegulator::<f64, 1, 1, 2>::new(1.0, 1e4, 5.0, 0.01);
        // With zero estimates and huge reference, MV uses b_min in denominator
        let u = str_ctrl.compute_control(1000.0);
        assert!(u.abs() <= 5.0 + 1e-9, "u={}", u);
    }

    #[test]
    fn prediction_error_zero_on_exact_model() {
        let mut str_ctrl = SelfTuningRegulator::<f64, 1, 1, 2>::new(1.0, 1e6, 100.0, 0.01);
        let mut y = 0.0_f64;
        let mut u = 0.0_f64;
        // Excite with PE input and long run
        for k in 1..=800 {
            let y_next = 0.5 * y + 1.0 * u;
            str_ctrl.update_estimate(y_next, u);
            u = if k % 7 < 3 { 1.0 } else { -1.0 };
            y = y_next;
        }
        // After convergence, prediction error should be small
        let e = str_ctrl.prediction_error(y);
        assert!(e.abs() < 0.2, "prediction error={:.4}", e);
    }

    #[test]
    fn second_order_identification() {
        // Plant: y[k] - 1.3*y[k-1] + 0.4*y[k-2] = 0.5*u[k-1] + 0.2*u[k-2]
        // NP = NA + NB = 2 + 2 = 4
        let mut str_ctrl = SelfTuningRegulator::<f64, 2, 2, 4>::new(0.99, 1e5, 100.0, 0.01);
        let mut y = [0.0_f64; 3];
        let mut u = [0.0_f64; 3];

        for k in 1..=2000 {
            let y_next = 1.3 * y[0] - 0.4 * y[1] + 0.5 * u[0] + 0.2 * u[1];
            str_ctrl.update_estimate(y_next, u[0]);
            u[1] = u[0];
            u[0] = if k % 13 < 6 { 1.5 } else { -1.0 };
            y[2] = y[1];
            y[1] = y[0];
            y[0] = y_next;
        }

        // a1_est ≈ -(-1.3) = -1.3 via phi[0]=-y[k-1]
        // phi[0]=-y[k-1], phi[1]=-y[k-2], phi[2]=u[k-1], phi[3]=u[k-2]
        // y_next = theta[0]*(-y[0]) + theta[1]*(-y[1]) + theta[2]*u[0] + theta[3]*u[1]
        //        = -theta[0]*y[0] - theta[1]*y[1] + theta[2]*u[0] + theta[3]*u[1]
        // So theta[0]=-1.3, theta[1]=0.4, theta[2]=0.5, theta[3]=0.2
        let params = str_ctrl.parameters();
        assert!((params[0] - (-1.3)).abs() < 0.1, "a1={:.4}", params[0]);
        assert!((params[1] - 0.4).abs() < 0.1, "a2={:.4}", params[1]);
        assert!((params[2] - 0.5).abs() < 0.1, "b1={:.4}", params[2]);
        assert!((params[3] - 0.2).abs() < 0.1, "b2={:.4}", params[3]);
    }
}
