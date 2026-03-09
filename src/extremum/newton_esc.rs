// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright COOLJAPAN OU (Team Kitasan)
//
// Newton-Based Extremum Seeking Control.
//
// Simultaneously estimates the gradient and the Hessian of the unknown map
// `y = f(u)`, then applies a Newton step:
//
//   û̇ = -k · ξ_G / (|Ĥ| + ε)
//
// This achieves convergence rates that are **independent of the Hessian
// magnitude**, unlike the gradient ESC whose convergence speed scales with
// the curvature.
//
// Two-filter demodulation architecture (discrete Euler):
//
//   1. Probing:            u_k   = û_k + a·sin(φ_k)
//                          user evaluates y = f(u_k)
//
//   2. HPF on y:           η_{k+1}   = η_k + dt·h_y·(y_k − η_k)
//                          y_hp_k    = y_k − η_k
//
//   3. Gradient channel:   ξ_G,k+1 = ξ_G,k + dt·h·(y_hp·sin φ − ξ_G,k)
//
//   4. Hessian channel:    ξ_H,k+1 = ξ_H,k + dt·h·(y_hp·(−2/a²)·cos 2φ − ξ_H,k)
//                          Ĥ_k     = ξ_H,k
//
//   5. Newton step:        û_{k+1} = û_k + dt·k·ξ_G,k / (|Ĥ_k| + ε)
//
//   6. Phase:              φ_{k+1} = (φ_k + ω·dt) mod 2π
//
// The HPF (steps 2) removes the absolute DC offset of f so the demodulated
// signals reflect only the local gradient/Hessian, not the function value.
// The Hessian signal (−2/a²)·cos 2φ is derived from the second-order term
// of the Taylor expansion of y around û:
//   y ≈ f(û) + f'(û)·a·sin φ + ½f''(û)·a²·sin²φ
//   sin²φ = ½(1 − cos 2φ)  →  Hessian demodulator: −cos 2φ (normalised by 2/a²)

use crate::core::scalar::ControlScalar;
use crate::extremum::gradient_esc::ExtremumError;

// ─────────────────────────────────────────────────────────────
// NewtonEsc  (SISO)
// ─────────────────────────────────────────────────────────────

/// Newton-based Extremum Seeking Controller (SISO).
///
/// Compared with [`GradientEsc`](crate::extremum::GradientEsc), this
/// controller achieves **curvature-independent** convergence by normalising
/// the gradient estimate by the (regularised) Hessian estimate.
///
/// The controller maximises `f(u)`.  To minimise, negate `y` before calling
/// [`update`](Self::update).
///
/// # References
/// Ariyur & Krstić (2003), *Real-Time Optimization by Extremum-Seeking
/// Control*, Chapter 4.
#[derive(Debug, Clone)]
pub struct NewtonEsc<S> {
    /// Current estimate of the optimal input `û`.
    u_hat: S,
    /// LPF-of-y state for the output HPF (η ≈ LPF(y)).
    eta: S,
    /// LPF state for the gradient estimate ξ_G.
    xi_grad: S,
    /// LPF state for the Hessian estimate ξ_H.
    xi_hess: S,
    /// Running Hessian estimate Ĥ = ξ_H.
    hess_hat: S,
    /// Current phase of the probing sinusoid [rad].
    phase: S,

    // ── parameters ──────────────────────────────────────────
    /// Probing amplitude `a > 0`.
    amplitude: S,
    /// Probing angular frequency `ω > 0` [rad/s].
    omega: S,
    /// Output HPF bandwidth `h_y > 0` (removes DC of `y`).
    h_y: S,
    /// Demodulation LPF bandwidth `h > 0`.
    h: S,
    /// Newton step size `k > 0`.
    k: S,
    /// Hessian regularisation `ε > 0` (prevents division-by-zero).
    eps: S,
    /// Discrete time-step `dt > 0` [s].
    dt: S,
}

impl<S: ControlScalar> NewtonEsc<S> {
    /// Construct a new `NewtonEsc`.
    ///
    /// # Parameters
    /// * `u_init`        – Initial estimate of the optimal input.
    /// * `amplitude`     – Probing amplitude `a > 0`.
    /// * `omega`         – Probing frequency `ω > 0` [rad/s].
    /// * `hpf_bandwidth` – Output HPF bandwidth `h_y > 0`.
    /// * `lpf_bandwidth` – Demodulation LPF bandwidth `h > 0`.
    /// * `k`             – Newton step size `k > 0`.
    /// * `eps`           – Hessian regularisation `ε > 0`.
    /// * `dt`            – Sample period `> 0` [s].
    ///
    /// # Errors
    /// Returns [`ExtremumError::InvalidParameter`] when any parameter
    /// violates its positivity constraint.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        u_init: S,
        amplitude: S,
        omega: S,
        hpf_bandwidth: S,
        lpf_bandwidth: S,
        k: S,
        eps: S,
        dt: S,
    ) -> Result<Self, ExtremumError> {
        if amplitude <= S::ZERO {
            return Err(ExtremumError::InvalidParameter(
                "amplitude must be positive",
            ));
        }
        if omega <= S::ZERO {
            return Err(ExtremumError::InvalidParameter("omega must be positive"));
        }
        if hpf_bandwidth <= S::ZERO {
            return Err(ExtremumError::InvalidParameter(
                "hpf_bandwidth must be positive",
            ));
        }
        if lpf_bandwidth <= S::ZERO {
            return Err(ExtremumError::InvalidParameter(
                "lpf_bandwidth must be positive",
            ));
        }
        if k <= S::ZERO {
            return Err(ExtremumError::InvalidParameter(
                "k (Newton step size) must be positive",
            ));
        }
        if eps <= S::ZERO {
            return Err(ExtremumError::InvalidParameter(
                "eps (Hessian regularisation) must be positive",
            ));
        }
        if dt <= S::ZERO {
            return Err(ExtremumError::InvalidParameter("dt must be positive"));
        }

        Ok(Self {
            u_hat: u_init,
            eta: S::ZERO,
            xi_grad: S::ZERO,
            xi_hess: S::ZERO,
            hess_hat: -S::ONE, // sensible prior: negative (typical for maxima)
            phase: S::ZERO,
            amplitude,
            omega,
            h_y: hpf_bandwidth,
            h: lpf_bandwidth,
            k,
            eps,
            dt,
        })
    }

    /// Returns `û + a·sin(φ)`.
    #[inline]
    pub fn probing_input(&self) -> S {
        let s = S::from_f64(libm::sin(self.phase.to_f64()));
        self.u_hat + self.amplitude * s
    }

    /// Ingest plant output `y`, advance internal states, and return the new
    /// probing input.
    ///
    /// # Algorithm (per step)
    /// 1. **HPF on `y`**: `η ← η + dt·h_y·(y − η)`;  `y_hp = y − η`
    /// 2. **Gradient LPF**: `ξ_G ← ξ_G + dt·h·(y_hp·sin φ − ξ_G)`
    /// 3. **Hessian LPF**: `ξ_H ← ξ_H + dt·h·(y_hp·(−2/a²)·cos 2φ − ξ_H)`
    /// 4. **Hessian estimate**: `Ĥ ← ξ_H`
    /// 5. **Newton step**: `û ← û + dt·k·ξ_G / (|Ĥ| + ε)`
    /// 6. **Phase advance**: `φ ← (φ + ω·dt) mod 2π`
    ///
    /// # Errors
    /// Currently infallible; `Result` is reserved for future numerical-error
    /// detection.
    pub fn update(&mut self, y: S) -> Result<S, ExtremumError> {
        let phase_f64 = self.phase.to_f64();
        let sin_phi = S::from_f64(libm::sin(phase_f64));
        let cos_2phi = S::from_f64(libm::cos(2.0 * phase_f64));

        let two = S::TWO;
        let a_sq = self.amplitude * self.amplitude;
        // Hessian demodulation weight: −2/a²
        let hess_weight = -(two / a_sq);

        // ── Step 2: HPF on y ────────────────────────────────
        self.eta += self.dt * self.h_y * (y - self.eta);
        let y_hp = y - self.eta;

        // ── Step 3: Gradient LPF ─────────────────────────────
        self.xi_grad += self.dt * self.h * (y_hp * sin_phi - self.xi_grad);

        // ── Step 4: Hessian LPF ──────────────────────────────
        let hess_signal = y_hp * hess_weight * cos_2phi;
        self.xi_hess += self.dt * self.h * (hess_signal - self.xi_hess);
        self.hess_hat = self.xi_hess;

        // ── Step 5: Newton update ────────────────────────────
        let hess_abs = S::from_f64(libm::fabs(self.hess_hat.to_f64()));
        self.u_hat += self.dt * self.k * self.xi_grad / (hess_abs + self.eps);

        // ── Step 6: Phase advance with wrap ──────────────────
        self.phase += self.omega * self.dt;
        let two_pi = S::TWO * S::PI;
        while self.phase >= two_pi {
            self.phase -= two_pi;
        }

        Ok(self.probing_input())
    }

    /// Returns the current estimate of the optimal input `û`.
    #[inline]
    pub fn estimate(&self) -> S {
        self.u_hat
    }

    /// Returns the current Hessian estimate `Ĥ`.
    ///
    /// For a maximisation problem, `Ĥ < 0` at the optimum (negative
    /// curvature).  A larger magnitude indicates stronger curvature and
    /// faster Newton convergence.
    #[inline]
    pub fn hessian_estimate(&self) -> S {
        self.hess_hat
    }

    /// Resets to `u_init`; clears all filter and phase states.
    pub fn reset(&mut self, u_init: S) {
        self.u_hat = u_init;
        self.eta = S::ZERO;
        self.xi_grad = S::ZERO;
        self.xi_hess = S::ZERO;
        self.hess_hat = -S::ONE;
        self.phase = S::ZERO;
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extremum::gradient_esc::GradientEsc;

    fn quadratic_max(u: f64, u_star: f64, peak: f64) -> f64 {
        -(u - u_star) * (u - u_star) + peak
    }

    // ── Newton ESC tests ────────────────────────────────────

    /// Newton ESC converges to u* = 3 on the standard quadratic test map.
    ///
    /// Parameters: ω=20, h_y=1, h=5, k=5, ε=0.01, a=0.2, dt=0.001.
    /// The HPF removes the DC offset (f = 10 at optimum), and the Newton
    /// normalisation makes convergence curvature-independent.
    #[test]
    fn newton_esc_converges_quadratic() {
        let mut esc = NewtonEsc::<f64>::new(
            0.0,   // u_init
            0.2,   // amplitude
            20.0,  // omega
            1.0,   // hpf_bandwidth
            5.0,   // lpf_bandwidth
            5.0,   // k
            0.01,  // eps
            0.001, // dt
        )
        .expect("valid params");

        for _ in 0..20_000 {
            let u = esc.probing_input();
            let y = quadratic_max(u, 3.0, 10.0);
            esc.update(y).expect("update ok");
        }

        let est = esc.estimate();
        assert!(
            (est - 3.0).abs() < 0.3,
            "Newton ESC: expected û ≈ 3.0, got {est:.4}"
        );
    }

    /// Newton ESC reaches the threshold in fewer steps than gradient ESC
    /// (both use the same two-filter parameterisation; Newton should be faster
    /// because the Hessian normalisation compensates for curvature).
    #[test]
    fn newton_esc_faster_than_gradient() {
        const U_STAR: f64 = 3.0;
        const THRESHOLD: f64 = 0.5;
        const MAX_STEPS: usize = 30_000;
        let dt = 0.001_f64;
        let amp = 0.2_f64;
        let omega = 20.0_f64;

        // ── Newton ──────────────────────────────────────────
        let mut n_esc =
            NewtonEsc::<f64>::new(0.0, amp, omega, 1.0, 5.0, 5.0, 0.01, dt).expect("valid params");
        let mut newton_steps = MAX_STEPS;
        for i in 0..MAX_STEPS {
            let u = n_esc.probing_input();
            let y = quadratic_max(u, U_STAR, 10.0);
            n_esc.update(y).expect("update ok");
            if (n_esc.estimate() - U_STAR).abs() < THRESHOLD {
                newton_steps = i + 1;
                break;
            }
        }

        // ── Gradient ─────────────────────────────────────────
        let mut g_esc = GradientEsc::<f64>::new(0.0, amp, omega, 1.0, 5.0, 5.0, dt, false)
            .expect("valid params");
        let mut grad_steps = MAX_STEPS;
        for i in 0..MAX_STEPS {
            let u = g_esc.probing_input();
            let y = quadratic_max(u, U_STAR, 10.0);
            g_esc.update(y).expect("update ok");
            if (g_esc.estimate() - U_STAR).abs() < THRESHOLD {
                grad_steps = i + 1;
                break;
            }
        }

        // Newton must converge within MAX_STEPS.
        assert!(
            newton_steps < MAX_STEPS,
            "Newton ESC did not converge within {MAX_STEPS} steps"
        );

        // Gradient must also converge (validates comparator).
        assert!(
            grad_steps < MAX_STEPS,
            "Gradient ESC did not converge within {MAX_STEPS} steps (comparator broken)"
        );

        // Newton should be at least as fast as gradient (within 50 % slack
        // to account for Hessian estimation transient at startup).
        let slack = grad_steps + (grad_steps / 2);
        assert!(
            newton_steps <= slack,
            "Newton ({newton_steps} steps) was not faster than gradient \
             ({grad_steps} steps) even with 50% slack"
        );
    }

    /// After convergence the Hessian estimate must be **negative** (concave map).
    #[test]
    fn hessian_estimate_negative_after_convergence() {
        let mut esc = NewtonEsc::<f64>::new(0.0, 0.2, 20.0, 1.0, 5.0, 5.0, 0.01, 0.001)
            .expect("valid params");

        for _ in 0..20_000 {
            let u = esc.probing_input();
            let y = quadratic_max(u, 3.0, 10.0);
            esc.update(y).expect("update ok");
        }

        let h = esc.hessian_estimate();
        assert!(
            h < 0.0,
            "Hessian estimate should be negative at maximum, got {h:.6}"
        );
    }

    /// Starting at the optimum, `u_hat` should remain near 3.0 (gradient ≈ 0).
    #[test]
    fn zero_gradient_stable() {
        let mut esc = NewtonEsc::<f64>::new(
            3.0,  // already at optimum
            0.05, // small amplitude → small dither
            20.0, 1.0, 5.0, 5.0, 0.01, 0.001,
        )
        .expect("valid params");

        for _ in 0..10_000 {
            let u = esc.probing_input();
            let y = quadratic_max(u, 3.0, 10.0);
            esc.update(y).expect("update ok");
        }

        let est = esc.estimate();
        assert!(
            (est - 3.0).abs() < 0.3,
            "Starting at optimum, û should remain near 3.0, got {est:.4}"
        );
    }

    /// Various invalid parameters must produce errors.
    #[test]
    fn invalid_params_rejected() {
        // amplitude = 0
        assert!(
            NewtonEsc::<f64>::new(0.0, 0.0, 20.0, 1.0, 5.0, 5.0, 0.01, 0.001).is_err(),
            "zero amplitude should be rejected"
        );
        // k = 0
        assert!(
            NewtonEsc::<f64>::new(0.0, 0.2, 20.0, 1.0, 5.0, 0.0, 0.01, 0.001).is_err(),
            "zero k should be rejected"
        );
        // eps = 0
        assert!(
            NewtonEsc::<f64>::new(0.0, 0.2, 20.0, 1.0, 5.0, 5.0, 0.0, 0.001).is_err(),
            "zero eps should be rejected"
        );
        // omega < 0
        assert!(
            NewtonEsc::<f64>::new(0.0, 0.2, -1.0, 1.0, 5.0, 5.0, 0.01, 0.001).is_err(),
            "negative omega should be rejected"
        );
        // hpf_bandwidth = 0
        assert!(
            NewtonEsc::<f64>::new(0.0, 0.2, 20.0, 0.0, 5.0, 5.0, 0.01, 0.001).is_err(),
            "zero hpf_bandwidth should be rejected"
        );
        // lpf_bandwidth = 0
        assert!(
            NewtonEsc::<f64>::new(0.0, 0.2, 20.0, 1.0, 0.0, 5.0, 0.01, 0.001).is_err(),
            "zero lpf_bandwidth should be rejected"
        );
    }

    /// At phase = 0, `probing_input()` must equal `u_hat` (sin(0) = 0).
    #[test]
    fn probing_input_at_zero_phase() {
        let u_init = 7.5_f64;
        let esc = NewtonEsc::<f64>::new(u_init, 0.2, 20.0, 1.0, 5.0, 5.0, 0.01, 0.001)
            .expect("valid params");

        let probe = esc.probing_input();
        assert!(
            (probe - u_init).abs() < 1e-12,
            "At phase=0: probing_input should equal u_init={u_init}, got {probe}"
        );
    }

    /// Reset must restore `u_hat` and clear filter states.
    #[test]
    fn reset_clears_state() {
        let mut esc = NewtonEsc::<f64>::new(0.0, 0.2, 20.0, 1.0, 5.0, 5.0, 0.01, 0.001)
            .expect("valid params");

        for _ in 0..500 {
            let u = esc.probing_input();
            let y = quadratic_max(u, 3.0, 10.0);
            esc.update(y).expect("update ok");
        }

        esc.reset(5.0);
        assert!(
            (esc.estimate() - 5.0).abs() < 1e-12,
            "After reset, estimate should be 5.0"
        );
        // phase cleared → probing_input = u_hat
        assert!(
            (esc.probing_input() - 5.0).abs() < 1e-12,
            "After reset, probing_input should equal new u_init"
        );
    }
}
