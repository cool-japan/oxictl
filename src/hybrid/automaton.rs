//! Hybrid Automaton with M discrete modes and N-dimensional continuous state.
//!
//! Each mode has its own continuous dynamics function `f(x, u) -> dx/dt`.
//! Transitions between modes are governed by guard conditions (fn pointers).
//! Optional reset maps can modify the state on transition.
//! Minimum dwell time (in steps) is enforced before any transition is taken.
#![cfg_attr(not(feature = "std"), no_std)]

use core::mem::MaybeUninit;

use crate::core::scalar::ControlScalar;

/// Errors for HybridAutomaton operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridError {
    /// The requested mode index is out of range.
    InvalidMode,
    /// A mode switch was requested but minimum dwell time not satisfied.
    DwellTimeViolation,
    /// A parameter value is invalid (e.g. zero dt, M=0).
    InvalidParameter,
}

/// Guard function type: maps current state to a transition trigger decision.
pub type GuardFn<S, const N: usize> = fn(&[S; N]) -> bool;

/// Reset map type: maps pre-transition state to post-transition state.
pub type ResetFn<S, const N: usize> = fn(&[S; N]) -> [S; N];

/// Dynamics function type: maps (state, input) to state derivative.
pub type DynamicsFn<S, const N: usize> = fn(&[S; N], S) -> [S; N];

/// Row of guard options for a source mode: guards[from][to].
type GuardRow<S, const N: usize, const M: usize> = [Option<GuardFn<S, N>>; M];

/// Row of reset options for a source mode: resets[from][to].
type ResetRow<S, const N: usize, const M: usize> = [Option<ResetFn<S, N>>; M];

/// Initialize a 2D array of `Option<GuardFn>` to all `None`.
///
/// # Safety
/// Every element is explicitly written before `assume_init` is called.
unsafe fn init_guards<S, const N: usize, const M: usize>() -> [GuardRow<S, N, M>; M] {
    let mut arr: MaybeUninit<[GuardRow<S, N, M>; M]> = MaybeUninit::uninit();
    let ptr = arr.as_mut_ptr() as *mut Option<GuardFn<S, N>>;
    for i in 0..(M * M) {
        ptr.add(i).write(None);
    }
    arr.assume_init()
}

/// Initialize a 2D array of `Option<ResetFn>` to all `None`.
///
/// # Safety
/// Every element is explicitly written before `assume_init` is called.
unsafe fn init_resets<S, const N: usize, const M: usize>() -> [ResetRow<S, N, M>; M] {
    let mut arr: MaybeUninit<[ResetRow<S, N, M>; M]> = MaybeUninit::uninit();
    let ptr = arr.as_mut_ptr() as *mut Option<ResetFn<S, N>>;
    for i in 0..(M * M) {
        ptr.add(i).write(None);
    }
    arr.assume_init()
}

/// Hybrid automaton with M discrete modes and N-dimensional continuous state.
///
/// State evolves via Euler integration of the active mode's dynamics.
/// Guard conditions are checked after each integration step; if a guard fires
/// and the minimum dwell counter is satisfied, the automaton transitions to
/// the target mode (applying the optional reset map).
pub struct HybridAutomaton<S, const N: usize, const M: usize> {
    state: [S; N],
    mode: usize,
    /// Per-mode dynamics: dynamics[m](x, u) -> derivative [S; N]
    dynamics: [DynamicsFn<S, N>; M],
    /// guards[from][to] = Some(fn) => check guard for from→to transition
    guards: [GuardRow<S, N, M>; M],
    /// resets[from][to] = Some(fn) => applied on from→to transition
    resets: [ResetRow<S, N, M>; M],
    dwell_counter: usize,
    min_dwell: usize,
    dt: S,
    n_transitions: usize,
}

impl<S: ControlScalar, const N: usize, const M: usize> HybridAutomaton<S, N, M> {
    /// Create a new hybrid automaton.
    ///
    /// # Arguments
    /// * `state0`    — initial continuous state
    /// * `mode0`     — initial discrete mode (must be < M)
    /// * `dynamics`  — per-mode derivative functions
    /// * `dt`        — Euler integration step size
    /// * `min_dwell` — minimum steps before a transition can fire
    ///
    /// # Errors
    /// Returns [`HybridError::InvalidParameter`] if `M == 0`.
    /// Returns [`HybridError::InvalidMode`] if `mode0 >= M`.
    pub fn new(
        state0: [S; N],
        mode0: usize,
        dynamics: [DynamicsFn<S, N>; M],
        dt: S,
        min_dwell: usize,
    ) -> Result<Self, HybridError> {
        if M == 0 {
            return Err(HybridError::InvalidParameter);
        }
        if mode0 >= M {
            return Err(HybridError::InvalidMode);
        }
        // SAFETY: all elements are written before assume_init.
        let guards = unsafe { init_guards::<S, N, M>() };
        let resets = unsafe { init_resets::<S, N, M>() };
        Ok(Self {
            state: state0,
            mode: mode0,
            dynamics,
            guards,
            resets,
            dwell_counter: 0,
            min_dwell,
            dt,
            n_transitions: 0,
        })
    }

    /// Register a guard function for the from→to transition.
    ///
    /// # Errors
    /// Returns [`HybridError::InvalidMode`] if `from >= M` or `to >= M`.
    pub fn add_guard(
        &mut self,
        from: usize,
        to: usize,
        guard: GuardFn<S, N>,
    ) -> Result<(), HybridError> {
        if from >= M || to >= M {
            return Err(HybridError::InvalidMode);
        }
        self.guards[from][to] = Some(guard);
        Ok(())
    }

    /// Register a reset map for the from→to transition.
    ///
    /// # Errors
    /// Returns [`HybridError::InvalidMode`] if `from >= M` or `to >= M`.
    pub fn add_reset(
        &mut self,
        from: usize,
        to: usize,
        reset: ResetFn<S, N>,
    ) -> Result<(), HybridError> {
        if from >= M || to >= M {
            return Err(HybridError::InvalidMode);
        }
        self.resets[from][to] = Some(reset);
        Ok(())
    }

    /// Advance the automaton by one time step with scalar input `u`.
    ///
    /// Algorithm:
    /// 1. Euler integration: `x += dynamics[mode](x, u) * dt`
    /// 2. Guard check: iterate candidate modes; take the first firing guard
    ///    whose dwell condition is satisfied, apply reset, switch mode.
    /// 3. Increment dwell counter.
    ///
    /// # Returns
    /// `(active_mode, new_state)`
    pub fn step(&mut self, u: S) -> Result<(usize, [S; N]), HybridError> {
        // Step 1: Euler integration
        let dx = (self.dynamics[self.mode])(&self.state, u);
        for (x_i, dx_i) in self.state.iter_mut().zip(dx.iter()) {
            *x_i += *dx_i * self.dt;
        }

        // Step 2: Check guards for the current mode
        let current = self.mode;
        for j in 0..M {
            if j == current {
                continue;
            }
            if let Some(guard) = self.guards[current][j] {
                if guard(&self.state) && self.dwell_counter >= self.min_dwell {
                    // Apply reset if registered
                    if let Some(reset) = self.resets[current][j] {
                        self.state = reset(&self.state);
                    }
                    self.mode = j;
                    self.dwell_counter = 0;
                    self.n_transitions += 1;
                    break;
                }
            }
        }

        // Step 3: Increment dwell counter
        self.dwell_counter += 1;

        Ok((self.mode, self.state))
    }

    /// Return a reference to the current continuous state.
    #[inline]
    pub fn state(&self) -> &[S; N] {
        &self.state
    }

    /// Return the current discrete mode index.
    #[inline]
    pub fn mode(&self) -> usize {
        self.mode
    }

    /// Return the total number of mode transitions that have occurred.
    #[inline]
    pub fn n_transitions(&self) -> usize {
        self.n_transitions
    }

    /// Forcibly switch to the specified mode, resetting the dwell counter.
    ///
    /// # Errors
    /// Returns [`HybridError::InvalidMode`] if `mode >= M`.
    pub fn force_mode(&mut self, mode: usize) -> Result<(), HybridError> {
        if mode >= M {
            return Err(HybridError::InvalidMode);
        }
        self.mode = mode;
        self.dwell_counter = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Dynamics helpers ----

    /// Constant derivative: dx/dt = [1.0, 0.0]
    fn dyn_const_one(x: &[f64; 2], _u: f64) -> [f64; 2] {
        let _ = x;
        [1.0, 0.0]
    }

    /// Constant derivative: dx/dt = [2.0, 0.0]
    fn dyn_const_two(x: &[f64; 2], _u: f64) -> [f64; 2] {
        let _ = x;
        [2.0, 0.0]
    }

    /// Zero derivative
    fn dyn_zero(x: &[f64; 2], _u: f64) -> [f64; 2] {
        let _ = x;
        [0.0, 0.0]
    }

    // ---- Guard helpers ----

    /// Guard: fires when x[0] > 0.5
    fn guard_x0_gt_half(x: &[f64; 2]) -> bool {
        x[0] > 0.5
    }

    /// Guard: always fires
    fn guard_always(x: &[f64; 2]) -> bool {
        let _ = x;
        true
    }

    // ---- Reset helpers ----

    /// Reset: set x[0] = 0.0, keep x[1]
    fn reset_zero_x0(x: &[f64; 2]) -> [f64; 2] {
        [0.0, x[1]]
    }

    // ---- Tests ----

    #[test]
    fn single_mode_no_transition() {
        // One mode, dynamics adds dt to x[0] each step.
        let dynamics: [DynamicsFn<f64, 2>; 1] = [dyn_const_one];
        let mut ha = HybridAutomaton::<f64, 2, 1>::new([0.0, 0.0], 0, dynamics, 0.1, 0).unwrap();

        for _ in 0..5 {
            ha.step(0.0).unwrap();
        }
        // x[0] should be 5 * 0.1 = 0.5
        assert!((ha.state()[0] - 0.5).abs() < 1e-9);
        assert_eq!(ha.mode(), 0);
        assert_eq!(ha.n_transitions(), 0);
    }

    #[test]
    fn guard_triggers_mode_switch() {
        // Mode 0: dx=[1,0], Mode 1: dx=[0,0].
        // Guard 0→1 fires when x[0] > 0.5. dt=0.1, min_dwell=0.
        let dynamics: [DynamicsFn<f64, 2>; 2] = [dyn_const_one, dyn_zero];
        let mut ha = HybridAutomaton::<f64, 2, 2>::new([0.0, 0.0], 0, dynamics, 0.1, 0).unwrap();
        ha.add_guard(0, 1, guard_x0_gt_half).unwrap();

        // Step until mode switches (should happen after x[0] > 0.5, i.e. 6 steps)
        let mut switched = false;
        for _ in 0..20 {
            let (mode, _) = ha.step(0.0).unwrap();
            if mode == 1 {
                switched = true;
                break;
            }
        }
        assert!(switched, "Should have switched to mode 1");
        assert_eq!(ha.n_transitions(), 1);
    }

    #[test]
    fn dwell_prevents_immediate_switch() {
        // Guard always fires from step 0, but min_dwell=3 means switch can
        // only happen at dwell_counter >= 3, i.e. after at least 3 steps.
        let dynamics: [DynamicsFn<f64, 2>; 2] = [dyn_zero, dyn_zero];
        let mut ha = HybridAutomaton::<f64, 2, 2>::new([1.0, 0.0], 0, dynamics, 0.01, 3).unwrap();
        ha.add_guard(0, 1, guard_always).unwrap();

        // Steps 1, 2, 3: dwell_counter is 0,1,2 before check → no switch yet
        let (m1, _) = ha.step(0.0).unwrap(); // dwell_counter was 0 → incremented to 1 after step
        let (m2, _) = ha.step(0.0).unwrap();
        let (m3, _) = ha.step(0.0).unwrap();
        // dwell_counter before check at step 1 = 0 < 3, step 2 = 1 < 3, step 3 = 2 < 3
        assert_eq!(m1, 0, "Should still be mode 0 after step 1");
        assert_eq!(m2, 0, "Should still be mode 0 after step 2");
        assert_eq!(m3, 0, "Should still be mode 0 after step 3");

        // Step 4: dwell_counter before check = 3 >= 3 → switch fires
        let (m4, _) = ha.step(0.0).unwrap();
        assert_eq!(m4, 1, "Should have switched to mode 1 at step 4");
        assert_eq!(ha.n_transitions(), 1);
    }

    #[test]
    fn reset_applied_on_transition() {
        // Mode 0 increments x[0]; guard fires when x[0] > 0.5; reset zeroes x[0].
        let dynamics: [DynamicsFn<f64, 2>; 2] = [dyn_const_one, dyn_zero];
        let mut ha = HybridAutomaton::<f64, 2, 2>::new([0.0, 0.0], 0, dynamics, 0.1, 0).unwrap();
        ha.add_guard(0, 1, guard_x0_gt_half).unwrap();
        ha.add_reset(0, 1, reset_zero_x0).unwrap();

        for _ in 0..20 {
            let (mode, _) = ha.step(0.0).unwrap();
            if mode == 1 {
                break;
            }
        }
        assert_eq!(ha.mode(), 1);
        // After reset, x[0] should be 0.0
        assert!(ha.state()[0].abs() < 1e-9, "Reset should zero x[0]");
    }

    #[test]
    fn invalid_mode_returns_error() {
        let dynamics: [DynamicsFn<f64, 2>; 2] = [dyn_zero, dyn_zero];
        // mode0 = 2 >= M=2 → InvalidMode
        let result = HybridAutomaton::<f64, 2, 2>::new([0.0, 0.0], 2, dynamics, 0.1, 0);
        assert_eq!(result.err(), Some(HybridError::InvalidMode));
    }

    #[test]
    fn n_transitions_counts_correctly() {
        // Mode 0→1 via always-guard (dwell=0), Mode 1→0 via always-guard.
        // After each step the automaton will switch back and forth.
        let dynamics: [DynamicsFn<f64, 2>; 2] = [dyn_zero, dyn_zero];
        let mut ha = HybridAutomaton::<f64, 2, 2>::new([0.0, 0.0], 0, dynamics, 0.01, 0).unwrap();
        ha.add_guard(0, 1, guard_always).unwrap();
        ha.add_guard(1, 0, guard_always).unwrap();

        // Each step fires a transition (dwell resets to 0 after each switch,
        // so dwell_counter=0 >= min_dwell=0 is immediately true again on next step).
        for _ in 0..3 {
            ha.step(0.0).unwrap();
        }
        assert_eq!(ha.n_transitions(), 3);
    }

    #[test]
    fn euler_step_correct() {
        // Dynamics: constant derivative [1.0, 0.0], dt=0.1, no guard.
        let dynamics: [DynamicsFn<f64, 2>; 1] = [dyn_const_one];
        let mut ha = HybridAutomaton::<f64, 2, 1>::new([0.0, 0.0], 0, dynamics, 0.1, 0).unwrap();
        let (_, state) = ha.step(0.0).unwrap();
        assert!(
            (state[0] - 0.1).abs() < 1e-12,
            "x[0] should be 0.1 after one step"
        );
        assert!(state[1].abs() < 1e-12, "x[1] should remain 0.0");
    }

    #[test]
    fn force_mode_and_invalid_force() {
        let dynamics: [DynamicsFn<f64, 2>; 2] = [dyn_zero, dyn_zero];
        let mut ha = HybridAutomaton::<f64, 2, 2>::new([0.0, 0.0], 0, dynamics, 0.01, 10).unwrap();
        // Force to mode 1
        ha.force_mode(1).unwrap();
        assert_eq!(ha.mode(), 1);
        // Force to invalid mode
        assert_eq!(ha.force_mode(2).err(), Some(HybridError::InvalidMode));
    }

    #[test]
    fn add_guard_invalid_mode_error() {
        let dynamics: [DynamicsFn<f64, 2>; 2] = [dyn_zero, dyn_zero];
        let mut ha = HybridAutomaton::<f64, 2, 2>::new([0.0, 0.0], 0, dynamics, 0.01, 0).unwrap();
        assert_eq!(
            ha.add_guard(0, 5, guard_always).err(),
            Some(HybridError::InvalidMode)
        );
        assert_eq!(
            ha.add_reset(5, 0, reset_zero_x0).err(),
            Some(HybridError::InvalidMode)
        );
    }

    #[test]
    fn two_mode_dynamics_correct() {
        // Verify that after mode switch, mode-1 dynamics ([2,0]) are applied.
        let dynamics: [DynamicsFn<f64, 2>; 2] = [dyn_const_one, dyn_const_two];
        let mut ha = HybridAutomaton::<f64, 2, 2>::new([0.0, 0.0], 0, dynamics, 0.1, 0).unwrap();
        ha.add_guard(0, 1, guard_x0_gt_half).unwrap();

        // Run until switch
        loop {
            let (mode, _) = ha.step(0.0).unwrap();
            if mode == 1 {
                break;
            }
        }
        let state_at_switch = *ha.state();
        // One more step in mode 1: derivative is 2.0, dt=0.1 → x[0] += 0.2
        ha.step(0.0).unwrap();
        let after = ha.state()[0];
        assert!((after - (state_at_switch[0] + 0.2)).abs() < 1e-9);
    }
}
