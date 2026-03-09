//! Switched Linear Time-Invariant (LTI) System.
//!
//! Implements discrete-time switched dynamics:
//!   x\[k+1\] = A_σ * x\[k\] + B_σ * u\[k\]
//!   y\[k\]   = C_σ · x\[k\]   (scalar output)
//!
//! Mode switching is gated by a minimum dwell-time constraint.
//! An optional Lyapunov matrix P can be set for stability monitoring.
#![cfg_attr(not(feature = "std"), no_std)]

use crate::core::scalar::ControlScalar;

/// Errors produced by [`SwitchedLti`] operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchedError {
    /// The requested mode index is out of range.
    InvalidMode,
    /// A mode switch was attempted before the minimum dwell time was satisfied.
    DwellViolation,
    /// A configuration parameter is invalid (e.g. M == 0).
    InvalidParameter,
}

/// Discrete-time switched LTI system with `M` modes, `N`-dimensional state,
/// and `I`-dimensional input.
///
/// # Type Parameters
/// * `S` — scalar type implementing [`ControlScalar`]
/// * `N` — state dimension
/// * `I` — input dimension
/// * `M` — number of modes
pub struct SwitchedLti<S, const N: usize, const I: usize, const M: usize> {
    /// State matrices A_m for each mode: a_modes[m][row][col]
    a_modes: [[[S; N]; N]; M],
    /// Input matrices B_m for each mode: b_modes[m][row][col]  (row = state dim)
    b_modes: [[[S; I]; N]; M],
    /// Output row vectors C_m for each mode: c_modes[m][col]
    c_modes: [[S; N]; M],
    state: [S; N],
    mode: usize,
    min_dwell: usize,
    time_in_mode: usize,
    total_switches: usize,
    /// Lyapunov matrix P (N×N), stored row-major: p_lyapunov[row][col]
    p_lyapunov: [[S; N]; N],
    p_set: bool,
}

impl<S: ControlScalar, const N: usize, const I: usize, const M: usize> SwitchedLti<S, N, I, M> {
    /// Create a new switched LTI system.
    ///
    /// # Arguments
    /// * `a_modes`   — state transition matrices per mode
    /// * `b_modes`   — input matrices per mode
    /// * `c_modes`   — output row vectors per mode
    /// * `min_dwell` — minimum time steps before a mode switch is allowed
    ///
    /// # Errors
    /// Returns [`SwitchedError::InvalidParameter`] if `M == 0` or `N == 0`.
    pub fn new(
        a_modes: [[[S; N]; N]; M],
        b_modes: [[[S; I]; N]; M],
        c_modes: [[S; N]; M],
        min_dwell: usize,
    ) -> Result<Self, SwitchedError> {
        if M == 0 || N == 0 {
            return Err(SwitchedError::InvalidParameter);
        }
        let state = [S::ZERO; N];
        let p_lyapunov = [[S::ZERO; N]; N];
        Ok(Self {
            a_modes,
            b_modes,
            c_modes,
            state,
            mode: 0,
            min_dwell,
            time_in_mode: 0,
            total_switches: 0,
            p_lyapunov,
            p_set: false,
        })
    }

    /// Attempt to switch to `new_mode`.
    ///
    /// The switch is rejected if fewer than `min_dwell` steps have elapsed in
    /// the current mode.
    ///
    /// # Errors
    /// * [`SwitchedError::InvalidMode`] — `new_mode >= M`
    /// * [`SwitchedError::DwellViolation`] — minimum dwell not satisfied
    pub fn switch_to(&mut self, new_mode: usize) -> Result<(), SwitchedError> {
        if new_mode >= M {
            return Err(SwitchedError::InvalidMode);
        }
        if self.time_in_mode < self.min_dwell {
            return Err(SwitchedError::DwellViolation);
        }
        if new_mode != self.mode {
            self.mode = new_mode;
            self.time_in_mode = 0;
            self.total_switches += 1;
        }
        Ok(())
    }

    /// Advance the system by one step with input vector `u`.
    ///
    /// Computes:
    ///   x_new\[i\] = sum_j A\[mode\]\[i\]\[j\] * x\[j\]  +  sum_k B\[mode\]\[i\]\[k\] * u\[k\]
    ///   y        = C\[mode\] · x_new
    ///
    /// # Returns
    /// The scalar output `y`.
    pub fn step(&mut self, u: &[S; I]) -> Result<S, SwitchedError> {
        let m = self.mode;

        let mut x_new = [S::ZERO; N];
        for (i, x_new_i) in x_new.iter_mut().enumerate() {
            let mut acc = S::ZERO;
            for (j, x_j) in self.state.iter().enumerate() {
                acc += self.a_modes[m][i][j] * *x_j;
            }
            for (k, u_k) in u.iter().enumerate() {
                acc += self.b_modes[m][i][k] * *u_k;
            }
            *x_new_i = acc;
        }
        self.state = x_new;
        self.time_in_mode += 1;

        // y = C[m] · x_new
        let mut y = S::ZERO;
        for (j, x_j) in self.state.iter().enumerate() {
            y += self.c_modes[m][j] * *x_j;
        }
        Ok(y)
    }

    /// Set the Lyapunov matrix P for stability monitoring (x^T P x).
    pub fn set_lyapunov(&mut self, p: [[S; N]; N]) {
        self.p_lyapunov = p;
        self.p_set = true;
    }

    /// Compute the current Lyapunov value V(x) = x^T P x.
    ///
    /// Returns `None` if no Lyapunov matrix has been set.
    pub fn lyapunov_value(&self) -> Option<S> {
        if !self.p_set {
            return None;
        }
        // v = x^T P x = sum_i sum_j x[i] * P[i][j] * x[j]
        let mut v = S::ZERO;
        for i in 0..N {
            for j in 0..N {
                v += self.state[i] * self.p_lyapunov[i][j] * self.state[j];
            }
        }
        Some(v)
    }

    /// Check whether the Lyapunov function is decreasing compared to a
    /// previous state `x_prev`.
    ///
    /// Returns `None` if no Lyapunov matrix has been set.
    /// Returns `Some(true)` if V(x_current) < V(x_prev), `Some(false)` otherwise.
    pub fn lyapunov_decreasing(&self, x_prev: &[S; N]) -> Option<bool> {
        if !self.p_set {
            return None;
        }
        // V_prev = x_prev^T P x_prev
        let mut v_prev = S::ZERO;
        for i in 0..N {
            for j in 0..N {
                v_prev += x_prev[i] * self.p_lyapunov[i][j] * x_prev[j];
            }
        }
        let v_curr = self.lyapunov_value()?;
        Some(v_curr < v_prev)
    }

    /// Return the current discrete mode index.
    #[inline]
    pub fn mode(&self) -> usize {
        self.mode
    }

    /// Return the total number of successful mode switches.
    #[inline]
    pub fn total_switches(&self) -> usize {
        self.total_switches
    }

    /// Return a reference to the current state vector.
    #[inline]
    pub fn state(&self) -> &[S; N] {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stable mode 0: A0 = 0.5*I, B0 = [[1],[0]], C0 = [1,0]
    /// Active mode 1: A1 = 0.9*I, B1 = [[0],[1]], C1 = [0,1]
    fn make_2s1i2m() -> SwitchedLti<f64, 2, 1, 2> {
        let a0 = [[0.5, 0.0], [0.0, 0.5]];
        let a1 = [[0.9, 0.0], [0.0, 0.9]];
        let b0 = [[1.0], [0.0]];
        let b1 = [[0.0], [1.0]];
        let c0 = [1.0, 0.0];
        let c1 = [0.0, 1.0];
        SwitchedLti::new([a0, a1], [b0, b1], [c0, c1], 1).unwrap()
    }

    #[test]
    fn dynamics_correct_mode0() {
        let mut sys = make_2s1i2m();
        // x = [0,0], u = [2.0]: x_new = A0*[0,0] + B0*[2] = [2,0], y = C0·x_new = 2
        let y = sys.step(&[2.0]).unwrap();
        assert!((y - 2.0).abs() < 1e-12, "Expected output 2.0, got {y}");
        assert!((sys.state()[0] - 2.0).abs() < 1e-12);
        assert!((sys.state()[1]).abs() < 1e-12);
    }

    #[test]
    fn dynamics_correct_mode1() {
        let mut sys = make_2s1i2m();
        // Switch to mode 1 (min_dwell=1, time_in_mode starts at 0 so need 1 step first)
        sys.step(&[0.0]).unwrap(); // time_in_mode becomes 1
        sys.switch_to(1).unwrap();
        // x=[0,0], u=[3]: x_new = A1*[0,0] + B1*[3] = [0,3], y = C1·[0,3] = 3
        let y = sys.step(&[3.0]).unwrap();
        assert!(
            (y - 3.0).abs() < 1e-12,
            "Expected output 3.0 in mode1, got {y}"
        );
    }

    #[test]
    fn dwell_violation_returns_error() {
        let mut sys = make_2s1i2m();
        // time_in_mode = 0, min_dwell = 1 → DwellViolation
        let err = sys.switch_to(1);
        assert_eq!(err.err(), Some(SwitchedError::DwellViolation));
    }

    #[test]
    fn mode_switch_changes_output() {
        let mut sys = make_2s1i2m();
        // Drive x[0]=2 in mode 0
        sys.step(&[4.0]).unwrap(); // time_in_mode→1
        sys.switch_to(1).unwrap(); // switch to mode 1 (C1=[0,1])
                                   // x=[4,0] (approx) in mode1: x_new = A1*[4,0]+B1*[0] = [3.6,0], y=C1·[3.6,0]=0
        let y = sys.step(&[0.0]).unwrap();
        assert!(
            (y).abs() < 1e-12,
            "Output should be 0 in mode1 with C1=[0,1]"
        );
    }

    #[test]
    fn lyapunov_value_computed() {
        let mut sys = make_2s1i2m();
        // P = I (identity)
        sys.set_lyapunov([[1.0, 0.0], [0.0, 1.0]]);
        // x = A0*[0,0] + B0*[3] = [3,0]: V = 3^2 + 0^2 = 9
        sys.step(&[3.0]).unwrap();
        let v = sys.lyapunov_value().unwrap();
        assert!(
            (v - 9.0).abs() < 1e-9,
            "Lyapunov value should be 9.0, got {v}"
        );
    }

    #[test]
    fn lyapunov_none_when_not_set() {
        let sys = make_2s1i2m();
        assert!(sys.lyapunov_value().is_none());
        assert!(sys.lyapunov_decreasing(&[1.0, 0.0]).is_none());
    }

    #[test]
    fn total_switches_increments() {
        let mut sys = make_2s1i2m();
        assert_eq!(sys.total_switches(), 0);
        sys.step(&[0.0]).unwrap(); // time_in_mode → 1
        sys.switch_to(1).unwrap();
        assert_eq!(sys.total_switches(), 1);
        sys.step(&[0.0]).unwrap(); // time_in_mode → 1
        sys.switch_to(0).unwrap();
        assert_eq!(sys.total_switches(), 2);
    }

    #[test]
    fn siso_output_matches_manual() {
        // Single-state single-input: A=[0.8], B=[1.0], C=[1.0]
        // x[k+1] = 0.8*x[k] + u[k], y = x[k+1]
        let a = [[[0.8_f64]]];
        let b = [[[1.0_f64]]];
        let c = [[1.0_f64]];
        let mut sys = SwitchedLti::<f64, 1, 1, 1>::new(a, b, c, 0).unwrap();
        // Step 1: x=0, u=2 → x=2, y=2
        let y1 = sys.step(&[2.0]).unwrap();
        assert!((y1 - 2.0).abs() < 1e-12);
        // Step 2: x=2, u=0 → x=1.6, y=1.6
        let y2 = sys.step(&[0.0]).unwrap();
        assert!((y2 - 1.6).abs() < 1e-12);
    }

    #[test]
    fn lyapunov_decreasing_check() {
        let mut sys = make_2s1i2m();
        sys.set_lyapunov([[1.0, 0.0], [0.0, 1.0]]);
        // x = A0*[0,0]+B0*[4] = [4,0]
        sys.step(&[4.0]).unwrap();
        let x_after_first = *sys.state(); // [4, 0]
                                          // x = A0*[4,0]+B0*[0] = [2,0]
        sys.step(&[0.0]).unwrap();
        let decreasing = sys.lyapunov_decreasing(&x_after_first).unwrap();
        assert!(decreasing, "Lyapunov should be decreasing (stable mode)");
    }

    #[test]
    fn invalid_mode_error() {
        let mut sys =
            SwitchedLti::<f64, 1, 1, 1>::new([[[0.5_f64]]], [[[1.0_f64]]], [[1.0_f64]], 0).unwrap();
        assert_eq!(sys.switch_to(5).err(), Some(SwitchedError::InvalidMode));
    }
}
