// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright COOLJAPAN OU (Team Kitasan)
//
// Gradient-based Extremum Seeking Control (Krstić & Wang 2000).
//
// Two-filter demodulation architecture (discrete Euler):
//
//   1. Probing:       u_k = û_k + a·sin(φ_k)
//                     user evaluates y = f(u_k)
//
//   2. HPF on y:      η_{k+1} = η_k + dt·h_y·(y_k - η_k)    (LPF state)
//                     y_hp_k  = y_k - η_k                     (y minus LPF = HPF)
//
//   3. Demodulate:    d_k = y_hp_k · sin(φ_k)
//
//   4. LPF on d:      ξ_{k+1} = ξ_k + dt·h·(d_k - ξ_k)     (gradient estimate)
//
//   5. Integrate:     û_{k+1} = û_k + dt·k_int·ξ_k
//
//   6. Phase:         φ_{k+1} = (φ_k + ω·dt) mod 2π
//
// The HPF on y removes the absolute DC offset of f (which would otherwise
// create a spurious constant term in the demodulated signal and produce
// steady-state tracking error when the LPF's DC gain is not exactly 1).
// The LPF on the demodulated product (with unity DC gain) then provides a
// clean estimate of the gradient direction.

use crate::core::scalar::ControlScalar;

// ─────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────

/// Errors produced by extremum seeking controllers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtremumError {
    /// A configuration parameter is outside its valid range.
    InvalidParameter(&'static str),
}

impl core::fmt::Display for ExtremumError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ExtremumError::InvalidParameter(msg) => {
                write!(f, "ExtremumError::InvalidParameter: {msg}")
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// GradientEsc  (SISO)
// ─────────────────────────────────────────────────────────────

/// Perturbation-based sinusoidal Extremum Seeking Controller (SISO).
///
/// Finds the input `u*` that **maximises** the unknown static map `y = f(u)`.
/// Set `minimize = true` to find the **minimum** instead.
///
/// # Algorithm
/// Uses a two-filter demodulation chain (HPF on output + LPF on product)
/// following the Krstić & Wang (2000) averaged analysis.  The HPF removes
/// the absolute DC offset of `f` so the gradient signal is cleanly extracted
/// regardless of the function's absolute value.
///
/// # Tuning guidelines
/// * `amplitude` `a`   – small relative to the basin of attraction; larger
///   → stronger gradient signal but bigger dithering at steady state.
/// * `omega` `ω`       – probing frequency [rad/s]; must satisfy ω >> k_int
///   (time-scale separation).
/// * `hpf_bandwidth`   – bandwidth `h_y` of the output HPF; should be ≪ ω
///   so that it removes slow output drift but not the probing-period variation.
/// * `lpf_bandwidth`   – bandwidth `h` of the gradient LPF; should be ≪ ω
///   to reject 2ω harmonics but ≫ integrator bandwidth.
/// * `integrator_gain` `k_int` – drives û toward u*; keep ω >> k_int.
#[derive(Debug, Clone)]
pub struct GradientEsc<S> {
    /// Current estimate of the optimal input `û`.
    u_hat: S,
    /// LPF state used to build the HPF on `y`: η ≈ LPF(y).
    eta: S,
    /// LPF state for the gradient estimate: ξ ≈ LPF(y_hp · sin φ).
    xi: S,
    /// Current phase of the probing sinusoid [rad].
    phase: S,

    // ── parameters ──────────────────────────────────────────
    /// Probing amplitude `a > 0`.
    amplitude: S,
    /// Probing angular frequency `ω > 0` [rad/s].
    omega: S,
    /// Output high-pass filter bandwidth `h_y > 0` [rad/s].
    h_y: S,
    /// Gradient low-pass filter bandwidth `h > 0` [rad/s].
    h: S,
    /// Integrator gain `k_int > 0`.
    k_int: S,
    /// Discrete time-step `dt > 0` [s].
    dt: S,
    /// `true` → minimise `f`; `false` → maximise `f`.
    minimize: bool,
}

impl<S: ControlScalar> GradientEsc<S> {
    /// Construct a new `GradientEsc`.
    ///
    /// # Parameters
    /// * `u_init`          – Initial estimate of the optimal input.
    /// * `amplitude`       – Probing amplitude `a > 0`.
    /// * `omega`           – Probing frequency `ω > 0` [rad/s].
    /// * `hpf_bandwidth`   – Output HPF bandwidth `h_y > 0` (removes DC of `y`).
    /// * `lpf_bandwidth`   – Gradient LPF bandwidth `h > 0` (smooths demodulated signal).
    /// * `integrator_gain` – Integrator gain `k_int > 0`.
    /// * `dt`              – Sample period `> 0` [s].
    /// * `minimize`        – Minimise when `true`, maximise when `false`.
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
        integrator_gain: S,
        dt: S,
        minimize: bool,
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
        if integrator_gain <= S::ZERO {
            return Err(ExtremumError::InvalidParameter(
                "integrator_gain must be positive",
            ));
        }
        if dt <= S::ZERO {
            return Err(ExtremumError::InvalidParameter("dt must be positive"));
        }

        Ok(Self {
            u_hat: u_init,
            eta: S::ZERO,
            xi: S::ZERO,
            phase: S::ZERO,
            amplitude,
            omega,
            h_y: hpf_bandwidth,
            h: lpf_bandwidth,
            k_int: integrator_gain,
            dt,
            minimize,
        })
    }

    /// Returns the current probing input `û + a·sin(φ)`.
    ///
    /// The caller evaluates `y = f(probing_input())` and feeds `y` back via
    /// [`update`](Self::update) **before** calling `probing_input` again.
    #[inline]
    pub fn probing_input(&self) -> S {
        let s = S::from_f64(libm::sin(self.phase.to_f64()));
        self.u_hat + self.amplitude * s
    }

    /// Ingest the plant output `y`, advance internal states, and return the
    /// new probing input.
    ///
    /// # Errors
    /// Currently infallible; `Result` is reserved for future numerical-error
    /// detection.
    pub fn update(&mut self, y: S) -> Result<S, ExtremumError> {
        let sin_phase = S::from_f64(libm::sin(self.phase.to_f64()));
        let sign = if self.minimize { -S::ONE } else { S::ONE };

        // ── Step 2: HPF on y ────────────────────────────────
        // η is a LPF of y; y_hp = y - η is the high-pass output.
        self.eta += self.dt * self.h_y * (y - self.eta);
        let y_hp = y - self.eta;

        // ── Step 3+4: Demodulate and LPF ────────────────────
        // ξ is LPF(y_hp · sin φ); unity-DC-gain LPF form.
        let d = sign * y_hp * sin_phase;
        self.xi += self.dt * self.h * (d - self.xi);

        // ── Step 5: Integrate ────────────────────────────────
        self.u_hat += self.dt * self.k_int * self.xi;

        // ── Step 6: Phase advance with wrap ─────────────────
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

    /// Resets to a new initial estimate; clears all filter and phase states.
    pub fn reset(&mut self, u_init: S) {
        self.u_hat = u_init;
        self.eta = S::ZERO;
        self.xi = S::ZERO;
        self.phase = S::ZERO;
    }
}

// ─────────────────────────────────────────────────────────────
// GradientEsc2D  (2-input)
// ─────────────────────────────────────────────────────────────

/// Perturbation-based Extremum Seeking Controller for two simultaneous inputs.
///
/// Each input dimension `i` uses an independent probing frequency `ω_i` and
/// its own HPF + LPF + integrator chain.  The shared output `y` is fed to
/// both channels; the differing frequencies `ω_1 ≠ ω_2` make the gradient
/// estimates statistically orthogonal.
#[derive(Debug, Clone)]
pub struct GradientEsc2D<S> {
    /// Current estimates of the optimal inputs `[û₁, û₂]`.
    u_hat: [S; 2],
    /// LPF-of-y states for HPF (per channel): `[η₁, η₂]`.
    eta: [S; 2],
    /// Gradient LPF states: `[ξ₁, ξ₂]`.
    xi: [S; 2],
    /// Current phases `[φ₁, φ₂]` [rad].
    phase: [S; 2],

    // ── parameters ──────────────────────────────────────────
    /// Probing amplitudes `[a₁, a₂]`.
    amplitude: [S; 2],
    /// Probing angular frequencies `[ω₁, ω₂]` [rad/s].
    omega: [S; 2],
    /// Output HPF bandwidth (shared).
    h_y: S,
    /// Gradient LPF bandwidth (shared).
    h: S,
    /// Integrator gain (shared).
    k_int: S,
    /// Discrete time-step [s].
    dt: S,
    /// `true` → minimise `f`; `false` → maximise `f`.
    minimize: bool,
}

impl<S: ControlScalar> GradientEsc2D<S> {
    /// Construct a new `GradientEsc2D`.
    ///
    /// # Errors
    /// Returns [`ExtremumError::InvalidParameter`] when any required parameter
    /// is non-positive, or when the two probing frequencies coincide.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        u_init: [S; 2],
        amplitude: [S; 2],
        omega: [S; 2],
        hpf_bandwidth: S,
        lpf_bandwidth: S,
        k_int: S,
        dt: S,
        minimize: bool,
    ) -> Result<Self, ExtremumError> {
        for &a in &amplitude {
            if a <= S::ZERO {
                return Err(ExtremumError::InvalidParameter(
                    "all amplitudes must be positive",
                ));
            }
        }
        for &w in &omega {
            if w <= S::ZERO {
                return Err(ExtremumError::InvalidParameter(
                    "all omega values must be positive",
                ));
            }
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
        if k_int <= S::ZERO {
            return Err(ExtremumError::InvalidParameter(
                "integrator_gain must be positive",
            ));
        }
        if dt <= S::ZERO {
            return Err(ExtremumError::InvalidParameter("dt must be positive"));
        }
        // Require distinct probing frequencies to prevent channel cross-talk.
        let diff = S::from_f64(libm::fabs((omega[0] - omega[1]).to_f64()));
        if diff < S::EPSILON {
            return Err(ExtremumError::InvalidParameter(
                "omega[0] and omega[1] must differ to avoid channel cross-talk",
            ));
        }

        Ok(Self {
            u_hat: u_init,
            eta: [S::ZERO; 2],
            xi: [S::ZERO; 2],
            phase: [S::ZERO; 2],
            amplitude,
            omega,
            h_y: hpf_bandwidth,
            h: lpf_bandwidth,
            k_int,
            dt,
            minimize,
        })
    }

    /// Returns `[û₁ + a₁·sin(φ₁), û₂ + a₂·sin(φ₂)]`.
    #[inline]
    pub fn probing_input(&self) -> [S; 2] {
        [
            self.u_hat[0] + self.amplitude[0] * S::from_f64(libm::sin(self.phase[0].to_f64())),
            self.u_hat[1] + self.amplitude[1] * S::from_f64(libm::sin(self.phase[1].to_f64())),
        ]
    }

    /// Ingest the scalar plant output `y` (shared cost function), advance
    /// both gradient channels, and return new probing inputs.
    pub fn update(&mut self, y: S) -> Result<[S; 2], ExtremumError> {
        let sign = if self.minimize { -S::ONE } else { S::ONE };
        let two_pi = S::TWO * S::PI;

        for i in 0..2 {
            let sin_phi = S::from_f64(libm::sin(self.phase[i].to_f64()));

            // HPF on y (per channel, each sees the same y but different φ)
            self.eta[i] += self.dt * self.h_y * (y - self.eta[i]);
            let y_hp = y - self.eta[i];

            // Demodulate + LPF
            let d = sign * y_hp * sin_phi;
            self.xi[i] += self.dt * self.h * (d - self.xi[i]);

            // Gradient-ascent integrator
            self.u_hat[i] += self.dt * self.k_int * self.xi[i];

            // Phase advance with wrap
            self.phase[i] += self.omega[i] * self.dt;
            while self.phase[i] >= two_pi {
                self.phase[i] -= two_pi;
            }
        }

        Ok(self.probing_input())
    }

    /// Returns `[û₁, û₂]`.
    #[inline]
    pub fn estimate(&self) -> [S; 2] {
        self.u_hat
    }

    /// Resets to `u_init`; clears filter and phase states.
    pub fn reset(&mut self, u_init: [S; 2]) {
        self.u_hat = u_init;
        self.eta = [S::ZERO; 2];
        self.xi = [S::ZERO; 2];
        self.phase = [S::ZERO; 2];
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────

    fn quadratic_max(u: f64, u_star: f64, peak: f64) -> f64 {
        -(u - u_star) * (u - u_star) + peak
    }

    fn quadratic_min(u: f64, u_star: f64) -> f64 {
        (u - u_star) * (u - u_star)
    }

    // ── SISO tests ──────────────────────────────────────────

    /// ESC must converge to u* = 3 on y = -(u-3)^2 + 10 (maximise).
    ///
    /// Parameter rationale:
    ///   ω = 20 rad/s (probing), h_y = 1 rad/s (output HPF), h = 5 rad/s
    ///   (gradient LPF), k_int = 5 (integrator).  The HPF removes the
    ///   absolute DC offset of f so the 2-filter structure converges cleanly.
    #[test]
    fn quadratic_1d_maximize() {
        let mut esc = GradientEsc::<f64>::new(
            0.0,   // u_init  (far from optimum)
            0.2,   // amplitude
            20.0,  // omega  [rad/s]
            1.0,   // hpf_bandwidth (removes DC of y)
            5.0,   // lpf_bandwidth (smooths demodulated gradient)
            5.0,   // integrator_gain
            0.001, // dt
            false, // maximise
        )
        .expect("valid params");

        for _ in 0..30_000 {
            let u = esc.probing_input();
            let y = quadratic_max(u, 3.0, 10.0);
            esc.update(y).expect("update ok");
        }

        let est = esc.estimate();
        assert!(
            (est - 3.0).abs() < 0.3,
            "Expected u_hat ≈ 3.0, got {est:.4}"
        );
    }

    /// ESC must converge to u* = 3 on y = (u-3)^2 (minimise).
    #[test]
    fn quadratic_1d_minimize() {
        let mut esc = GradientEsc::<f64>::new(
            0.0,   // u_init
            0.2,   // amplitude
            20.0,  // omega
            1.0,   // hpf_bandwidth
            5.0,   // lpf_bandwidth
            5.0,   // integrator_gain
            0.001, // dt
            true,  // minimise
        )
        .expect("valid params");

        for _ in 0..30_000 {
            let u = esc.probing_input();
            let y = quadratic_min(u, 3.0);
            esc.update(y).expect("update ok");
        }

        let est = esc.estimate();
        assert!(
            (est - 3.0).abs() < 0.3,
            "Expected u_hat ≈ 3.0 (min), got {est:.4}"
        );
    }

    /// 2-D ESC: y = -(u1-2)^2 - (u2-5)^2  →  optimal (2, 5).
    ///
    /// Start close to the optimum (1.0, 4.0) with conservative gains to
    /// prevent the integrator from diverging on the large initial gradient
    /// from a distant starting point.
    #[test]
    fn quadratic_2d_maximize() {
        let mut esc = GradientEsc2D::<f64>::new(
            [1.0, 4.0],   // u_init (close to optimum to avoid divergence)
            [0.2, 0.2],   // amplitude
            [20.0, 30.0], // omega (distinct frequencies)
            1.0,          // hpf_bandwidth
            5.0,          // lpf_bandwidth
            5.0,          // k_int
            0.001,        // dt
            false,        // maximise
        )
        .expect("valid params");

        for _ in 0..30_000 {
            let [u1, u2] = esc.probing_input();
            let y = -(u1 - 2.0) * (u1 - 2.0) - (u2 - 5.0) * (u2 - 5.0);
            esc.update(y).expect("update ok");
        }

        let [e1, e2] = esc.estimate();
        assert!((e1 - 2.0).abs() < 0.5, "Expected û₁ ≈ 2.0, got {e1:.4}");
        assert!((e2 - 5.0).abs() < 0.5, "Expected û₂ ≈ 5.0, got {e2:.4}");
    }

    /// Invalid amplitude (zero) must return an error.
    #[test]
    fn invalid_amplitude_zero() {
        let result = GradientEsc::<f64>::new(0.0, 0.0, 20.0, 1.0, 5.0, 5.0, 0.001, false);
        assert!(result.is_err(), "zero amplitude should be rejected");
        assert_eq!(
            result.unwrap_err(),
            ExtremumError::InvalidParameter("amplitude must be positive")
        );
    }

    /// Negative omega must return an error.
    #[test]
    fn invalid_omega_negative() {
        let result = GradientEsc::<f64>::new(0.0, 0.2, -1.0, 1.0, 5.0, 5.0, 0.001, false);
        assert!(result.is_err(), "negative omega should be rejected");
        assert_eq!(
            result.unwrap_err(),
            ExtremumError::InvalidParameter("omega must be positive")
        );
    }

    /// Zero HPF bandwidth must return an error.
    #[test]
    fn invalid_hpf_zero() {
        let result = GradientEsc::<f64>::new(0.0, 0.2, 20.0, 0.0, 5.0, 5.0, 0.001, false);
        assert!(result.is_err(), "zero hpf_bandwidth should be rejected");
        assert_eq!(
            result.unwrap_err(),
            ExtremumError::InvalidParameter("hpf_bandwidth must be positive")
        );
    }

    /// At phase = 0 (initial state), `probing_input()` == `u_hat + a·sin(0)` == `u_hat`.
    #[test]
    fn probing_amplitude_at_zero_phase() {
        let amp = 0.2_f64;
        let esc = GradientEsc::<f64>::new(5.0, amp, 20.0, 1.0, 5.0, 5.0, 0.001, false)
            .expect("valid params");

        let probe = esc.probing_input();
        let est = esc.estimate();
        assert!(
            (probe - est).abs() < 1e-12,
            "At phase=0 probe should equal u_hat: probe={probe}, est={est}"
        );
    }

    /// Phase accumulates as ω·dt per step (modular).
    #[test]
    fn phase_accumulation() {
        let omega = 20.0_f64;
        let dt = 0.001_f64;
        let mut esc = GradientEsc::<f64>::new(0.0, 0.2, omega, 1.0, 5.0, 5.0, dt, false)
            .expect("valid params");

        let n_steps = 50_usize;
        for _ in 0..n_steps {
            let u = esc.probing_input();
            let y = quadratic_max(u, 3.0, 10.0);
            esc.update(y).expect("update ok");
        }

        let expected_phase = (omega * dt * n_steps as f64).rem_euclid(2.0 * core::f64::consts::PI);

        // Verify indirectly: probing_input = u_hat + a·sin(phase)
        let probe = esc.probing_input();
        let u_hat = esc.estimate();
        let sin_phase_actual = (probe - u_hat) / 0.2;
        let sin_phase_expected = libm::sin(expected_phase);
        assert!(
            (sin_phase_actual - sin_phase_expected).abs() < 0.02,
            "Phase mismatch: sin(actual)={sin_phase_actual:.6}, \
             sin(expected)={sin_phase_expected:.6}"
        );
    }

    /// Reset must restore `u_hat` and clear internal state.
    #[test]
    fn reset_restores_state() {
        let mut esc = GradientEsc::<f64>::new(0.0, 0.2, 20.0, 1.0, 5.0, 5.0, 0.001, false)
            .expect("valid params");

        for _ in 0..500 {
            let u = esc.probing_input();
            let y = quadratic_max(u, 3.0, 10.0);
            esc.update(y).expect("update ok");
        }

        esc.reset(10.0);
        assert!(
            (esc.estimate() - 10.0).abs() < 1e-12,
            "After reset u_hat should be 10.0"
        );
        // phase = 0 → probing_input = u_hat
        assert!(
            (esc.probing_input() - 10.0).abs() < 1e-12,
            "After reset probing_input should equal new u_init"
        );
    }

    /// Equal omega values in 2D must be rejected.
    #[test]
    fn invalid_2d_equal_omega() {
        let result = GradientEsc2D::<f64>::new(
            [0.0, 0.0],
            [0.2, 0.2],
            [20.0, 20.0], // identical → cross-talk
            1.0,
            5.0,
            5.0,
            0.001,
            false,
        );
        assert!(result.is_err(), "Equal omega should be rejected");
    }
}
