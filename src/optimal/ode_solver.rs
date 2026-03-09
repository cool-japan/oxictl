/// ODE integrators for direct shooting methods in optimal control.
///
/// Provides:
/// - `OdeSolver` trait — one-step integrator interface
/// - `Euler` — forward Euler (first-order, reference only)
/// - `RungeKutta4` — classical RK4 (primary solver)
/// - `RungeKuttaFehlberg` — RK45 adaptive step with Fehlberg error estimate
/// - `integrate` — utility to roll an ODE forward over [t0, tf]
use crate::core::scalar::ControlScalar;

use super::OptimalError;

// ───────────────────────────────────────────── trait ──────────────────────────

/// One-step ODE integrator.
///
/// Advances the state `x` from time `t` by `dt` under vector field `f`.
///
/// # Type parameters
/// - `S`: scalar type (`f32` or `f64`)
/// - `N`: state dimension (const generic)
pub trait OdeSolver<S: ControlScalar, const N: usize> {
    /// Compute one fixed-step integration step.
    ///
    /// Returns the estimated next state `x(t + dt)`.
    fn step<F>(&self, f: F, x: &[S; N], t: S, dt: S) -> [S; N]
    where
        F: Fn(&[S; N], S) -> [S; N];
}

// ─────────────────────────────────────────── forward Euler ────────────────────

/// Forward (explicit) Euler integrator — first-order, O(dt) local error.
///
/// Provided as a reference baseline; prefer `RungeKutta4` for practical use.
#[derive(Debug, Clone, Copy, Default)]
pub struct Euler<S: ControlScalar, const N: usize> {
    _marker: core::marker::PhantomData<S>,
}

impl<S: ControlScalar, const N: usize> Euler<S, N> {
    /// Create a new Euler integrator.
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<S: ControlScalar, const N: usize> OdeSolver<S, N> for Euler<S, N> {
    #[inline]
    fn step<F>(&self, f: F, x: &[S; N], t: S, dt: S) -> [S; N]
    where
        F: Fn(&[S; N], S) -> [S; N],
    {
        let k = f(x, t);
        core::array::from_fn(|i| x[i] + dt * k[i])
    }
}

// ──────────────────────────────────────────── RK4 ─────────────────────────────

/// Classical 4th-order Runge-Kutta integrator — O(dt⁴) local error.
///
/// This is the primary solver for fixed-step shooting methods.
#[derive(Debug, Clone, Copy, Default)]
pub struct RungeKutta4<S: ControlScalar, const N: usize> {
    _marker: core::marker::PhantomData<S>,
}

impl<S: ControlScalar, const N: usize> RungeKutta4<S, N> {
    /// Create a new RK4 integrator.
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<S: ControlScalar, const N: usize> OdeSolver<S, N> for RungeKutta4<S, N> {
    fn step<F>(&self, f: F, x: &[S; N], t: S, dt: S) -> [S; N]
    where
        F: Fn(&[S; N], S) -> [S; N],
    {
        let half = S::HALF;
        let sixth = S::from_f64(1.0 / 6.0);
        let dt_half = dt * half;

        // k1 = f(x, t)
        let k1 = f(x, t);

        // k2 = f(x + dt/2 * k1, t + dt/2)
        let x2: [S; N] = core::array::from_fn(|i| x[i] + dt_half * k1[i]);
        let k2 = f(&x2, t + dt_half);

        // k3 = f(x + dt/2 * k2, t + dt/2)
        let x3: [S; N] = core::array::from_fn(|i| x[i] + dt_half * k2[i]);
        let k3 = f(&x3, t + dt_half);

        // k4 = f(x + dt * k3, t + dt)
        let x4: [S; N] = core::array::from_fn(|i| x[i] + dt * k3[i]);
        let k4 = f(&x4, t + dt);

        // x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        core::array::from_fn(|i| {
            x[i] + dt * sixth * (k1[i] + S::TWO * k2[i] + S::TWO * k3[i] + k4[i])
        })
    }
}

// ────────────────────────────────────────── RK45 (Fehlberg) ──────────────────

/// Adaptive-step Runge-Kutta-Fehlberg (RK45) integrator.
///
/// Uses the Fehlberg tableau to produce both a 4th-order and a 5th-order
/// estimate, then controls step size based on the local truncation error.
///
/// Call `step_adaptive` instead of `step` for variable-step integration.
/// `step` (from the trait) uses a single fixed-dt RK4-like sub-step.
#[derive(Debug, Clone, Copy)]
pub struct RungeKuttaFehlberg<S: ControlScalar, const N: usize> {
    /// Minimum allowed step size (prevents infinite loops).
    pub dt_min: S,
    /// Safety factor for step-size control (typical: 0.9).
    pub safety: S,
    _marker: core::marker::PhantomData<S>,
}

impl<S: ControlScalar, const N: usize> Default for RungeKuttaFehlberg<S, N> {
    fn default() -> Self {
        Self {
            dt_min: S::from_f64(1e-10),
            safety: S::from_f64(0.9),
            _marker: core::marker::PhantomData,
        }
    }
}

impl<S: ControlScalar, const N: usize> RungeKuttaFehlberg<S, N> {
    /// Create with explicit minimum step and safety factor.
    pub fn new(dt_min: S, safety: S) -> Self {
        Self {
            dt_min,
            safety,
            _marker: core::marker::PhantomData,
        }
    }

    /// Attempt one adaptive RK45 step.
    ///
    /// Returns `(x_next, dt_used)` where `dt_used ≤ dt_max`.
    /// If the local error estimate exceeds `tol`, the step is retried with a
    /// smaller `dt` (halved each attempt) until `dt < dt_min`, at which point
    /// the best available result is returned to avoid infinite loops.
    ///
    /// # Parameters
    /// - `f`      — vector field `ẋ = f(x, t)`
    /// - `x`      — current state
    /// - `t`      — current time
    /// - `dt_max` — maximum step size to try
    /// - `tol`    — scalar error tolerance (applied to the ∞-norm of the error)
    pub fn step_adaptive<F>(&self, f: F, x: &[S; N], t: S, dt_max: S, tol: S) -> ([S; N], S)
    where
        F: Fn(&[S; N], S) -> [S; N] + Copy,
    {
        let mut dt = dt_max;

        loop {
            let (x4, x5) = self.fehlberg_stages(f, x, t, dt);

            // ∞-norm of the error estimate (|x5 - x4|)
            let mut err = S::ZERO;
            for i in 0..N {
                let e = (x5[i] - x4[i]).abs();
                if e > err {
                    err = e;
                }
            }

            if err <= tol || dt <= self.dt_min {
                // Accept step — use the higher-order (5th-order) estimate
                return (x5, dt);
            }

            // Compute optimal step-size scaling: h_new = h * safety * (tol/err)^0.2
            let ratio = tol / err;
            // 0.2 = 1/5; use powf from Float trait
            let scale = self.safety * ratio.powf(S::from_f64(0.2));
            // Clamp scale to avoid extreme changes [0.1, 5.0]
            let scale = scale.clamp_val(S::from_f64(0.1), S::from_f64(5.0));
            dt = (dt * scale).clamp_val(self.dt_min, dt_max);
        }
    }

    /// Internal: compute all six RK45 Fehlberg stages and return
    /// `(x_4th_order, x_5th_order)`.
    ///
    /// Uses the original Fehlberg (1970) coefficients.
    fn fehlberg_stages<F>(&self, f: F, x: &[S; N], t: S, dt: S) -> ([S; N], [S; N])
    where
        F: Fn(&[S; N], S) -> [S; N],
    {
        // Butcher tableau coefficients — Fehlberg RK45
        // c nodes
        let c2 = S::from_f64(1.0 / 4.0);
        let c3 = S::from_f64(3.0 / 8.0);
        let c4 = S::from_f64(12.0 / 13.0);
        // c5 = 1, c6 = 1/2

        // a coefficients (lower-triangular)
        let a21 = S::from_f64(1.0 / 4.0);
        let a31 = S::from_f64(3.0 / 32.0);
        let a32 = S::from_f64(9.0 / 32.0);
        let a41 = S::from_f64(1932.0 / 2197.0);
        let a42 = S::from_f64(-7200.0 / 2197.0);
        let a43 = S::from_f64(7296.0 / 2197.0);
        let a51 = S::from_f64(439.0 / 216.0);
        let a52 = S::from_f64(-8.0);
        let a53 = S::from_f64(3680.0 / 513.0);
        let a54 = S::from_f64(-845.0 / 4104.0);
        let a61 = S::from_f64(-8.0 / 27.0);
        let a62 = S::from_f64(2.0);
        let a63 = S::from_f64(-3544.0 / 2565.0);
        let a64 = S::from_f64(1859.0 / 4104.0);
        let a65 = S::from_f64(-11.0 / 40.0);

        // 4th-order b coefficients
        let b4_1 = S::from_f64(25.0 / 216.0);
        let b4_3 = S::from_f64(1408.0 / 2565.0);
        let b4_4 = S::from_f64(2197.0 / 4104.0);
        let b4_5 = S::from_f64(-1.0 / 5.0);

        // 5th-order b coefficients
        let b5_1 = S::from_f64(16.0 / 135.0);
        let b5_3 = S::from_f64(6656.0 / 12825.0);
        let b5_4 = S::from_f64(28561.0 / 56430.0);
        let b5_5 = S::from_f64(-9.0 / 50.0);
        let b5_6 = S::from_f64(2.0 / 55.0);

        // Stage evaluations
        let k1 = f(x, t);

        let x2: [S; N] = core::array::from_fn(|i| x[i] + dt * a21 * k1[i]);
        let k2 = f(&x2, t + dt * c2);

        let x3: [S; N] = core::array::from_fn(|i| x[i] + dt * (a31 * k1[i] + a32 * k2[i]));
        let k3 = f(&x3, t + dt * c3);

        let x4_node: [S; N] =
            core::array::from_fn(|i| x[i] + dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]));
        let k4 = f(&x4_node, t + dt * c4);

        let x5_node: [S; N] = core::array::from_fn(|i| {
            x[i] + dt * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i])
        });
        let k5 = f(&x5_node, t + dt);

        let x6_node: [S; N] = core::array::from_fn(|i| {
            x[i] + dt * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i])
        });
        let k6 = f(&x6_node, t + dt * S::HALF);

        // 4th-order estimate (b4)
        let x_4th: [S; N] = core::array::from_fn(|i| {
            x[i] + dt * (b4_1 * k1[i] + b4_3 * k3[i] + b4_4 * k4[i] + b4_5 * k5[i])
        });

        // 5th-order estimate (b5)
        let x_5th: [S; N] = core::array::from_fn(|i| {
            x[i] + dt * (b5_1 * k1[i] + b5_3 * k3[i] + b5_4 * k4[i] + b5_5 * k5[i] + b5_6 * k6[i])
        });

        (x_4th, x_5th)
    }
}

impl<S: ControlScalar, const N: usize> OdeSolver<S, N> for RungeKuttaFehlberg<S, N> {
    /// Fixed-step RK4 (delegates to the 4th-order Fehlberg estimate).
    fn step<F>(&self, f: F, x: &[S; N], t: S, dt: S) -> [S; N]
    where
        F: Fn(&[S; N], S) -> [S; N],
    {
        let (x4, _) = self.fehlberg_stages(f, x, t, dt);
        x4
    }
}

// ──────────────────────────────────────── integrate utility ───────────────────

/// Integrate `ẋ = f(x, t)` from `t0` to `tf` with fixed step `dt`.
///
/// Uses any solver implementing [`OdeSolver`].  The final step is adjusted
/// so it does not overshoot `tf`.
///
/// Returns the state at time `tf`, or `OptimalError::IntegrationFailed` if the
/// time span is non-positive.
pub fn integrate<S, const N: usize, Solver>(
    f: impl Fn(&[S; N], S) -> [S; N],
    x0: [S; N],
    t0: S,
    tf: S,
    dt: S,
    solver: &Solver,
) -> Result<[S; N], OptimalError>
where
    S: ControlScalar,
    Solver: OdeSolver<S, N>,
{
    if tf <= t0 {
        return Err(OptimalError::IntegrationFailed(
            "tf must be greater than t0",
        ));
    }
    if dt <= S::ZERO {
        return Err(OptimalError::IntegrationFailed("dt must be positive"));
    }

    let mut x = x0;
    let mut t = t0;

    while t < tf {
        // Clamp the last step to avoid overshooting tf
        let step = if t + dt > tf { tf - t } else { dt };
        x = solver.step(&f, &x, t, step);
        t += step;
    }

    Ok(x)
}

// ──────────────────────────────────────────── tests ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── exact linear ODE ──────────────────────────────────────────────────────
    // ẋ = -x, x(0) = 1  =>  x(t) = exp(-t)

    fn linear_decay(x: &[f64; 1], _t: f64) -> [f64; 1] {
        [-x[0]]
    }

    #[test]
    fn euler_linear_ode_accuracy() {
        let solver = Euler::<f64, 1>::new();
        let x0 = [1.0_f64];
        let tf = 1.0_f64;
        let dt = 1e-4;
        let result =
            integrate(linear_decay, x0, 0.0, tf, dt, &solver).expect("integration should succeed");
        let exact = (-tf).exp();
        // Euler is O(dt), so expect ~1% error at dt=1e-4 over 1 second
        assert!(
            (result[0] - exact).abs() < 1e-3,
            "Euler result={:.6} exact={:.6}",
            result[0],
            exact
        );
    }

    #[test]
    fn rk4_linear_ode_accuracy() {
        let solver = RungeKutta4::<f64, 1>::new();
        let x0 = [1.0_f64];
        let tf = 1.0_f64;
        // RK4 global error ≈ C·dt⁴; with dt=0.01 over 100 steps → ~1e-9
        let dt = 0.01;
        let result =
            integrate(linear_decay, x0, 0.0, tf, dt, &solver).expect("integration should succeed");
        let exact = (-tf).exp();
        assert!(
            (result[0] - exact).abs() < 1e-9,
            "RK4 result={:.12} exact={:.12}",
            result[0],
            exact
        );
    }

    // ── harmonic oscillator energy conservation ──────────────────────────────
    // State: [q, p], ẋ = [p, -q]  =>  H = (q² + p²)/2 = const
    // Exact solution: q(t)=cos(t), p(t)=-sin(t) starting from [1,0]

    fn harmonic(x: &[f64; 2], _t: f64) -> [f64; 2] {
        [x[1], -x[0]]
    }

    #[test]
    fn rk4_harmonic_oscillator_energy() {
        let solver = RungeKutta4::<f64, 2>::new();
        let x0 = [1.0_f64, 0.0];
        let tf = 10.0 * core::f64::consts::PI; // 5 full periods
        let dt = 0.01;
        let result =
            integrate(harmonic, x0, 0.0, tf, dt, &solver).expect("integration should succeed");

        let energy_initial = 0.5 * (x0[0] * x0[0] + x0[1] * x0[1]);
        let energy_final = 0.5 * (result[0] * result[0] + result[1] * result[1]);

        assert!(
            (energy_final - energy_initial).abs() < 1e-6,
            "Energy drift = {:.2e}",
            (energy_final - energy_initial).abs()
        );
    }

    // ── RKF45 adaptive step ───────────────────────────────────────────────────

    #[test]
    fn rkf45_linear_ode_accuracy() {
        let solver = RungeKuttaFehlberg::<f64, 1>::default();
        let f = linear_decay;
        let x0 = [1.0_f64];
        let t = 0.0_f64;
        let dt_max = 0.5;
        let tol = 1e-8;
        let (x_next, dt_used) = solver.step_adaptive(f, &x0, t, dt_max, tol);
        let exact = (-dt_used).exp();
        assert!(
            (x_next[0] - exact).abs() < tol * 10.0,
            "RKF45 result={:.10} exact={:.10}",
            x_next[0],
            exact
        );
    }

    #[test]
    fn integrate_rejects_non_positive_span() {
        let solver = RungeKutta4::<f64, 1>::new();
        let result = integrate(linear_decay, [1.0], 1.0, 0.0, 0.1, &solver);
        assert!(result.is_err());
    }

    #[test]
    fn integrate_rejects_non_positive_dt() {
        let solver = RungeKutta4::<f64, 1>::new();
        let result = integrate(linear_decay, [1.0], 0.0, 1.0, -0.1, &solver);
        assert!(result.is_err());
    }

    #[test]
    fn rkf45_step_uses_dt_min_when_tight_tol() {
        // With an extremely tight tolerance, the solver should still return
        // without hanging — it respects dt_min.
        let solver = RungeKuttaFehlberg::<f64, 1> {
            dt_min: 1e-6,
            safety: 0.9,
            _marker: core::marker::PhantomData,
        };
        let x0 = [1.0_f64];
        // tol so tight it can never be satisfied — must terminate at dt_min
        let (x_next, dt_used) = solver.step_adaptive(linear_decay, &x0, 0.0, 1.0, 1e-300);
        assert!(dt_used >= 1e-7, "dt_used={:.2e}", dt_used);
        assert!(x_next[0] < x0[0]); // state decreased as expected
    }
}
