//! Piecewise Affine (PWA) Systems and Controllers.
//!
//! A PWA system partitions the state space into polyhedral regions via
//! halfspace inequalities. Within each region `r` the discrete-time dynamics are:
//!
//!   x[k+1] = A[r] * x[k] + B[r] * u[k] + f[r]
//!
//! where `f[r]` is the affine offset. Regions are characterised by:
//!
//!   R_r = { x : h_coeffs[r] · x ≤ h_bounds[r] }
//!
//! Region selection returns the *first* region whose halfspace condition is
//! satisfied by the current state.
//!
//! A PWA controller implements per-region state-feedback:
//!
//!   u_i = -(K[r][i,:] · x)
#![cfg_attr(not(feature = "std"), no_std)]

use crate::core::scalar::ControlScalar;

/// Errors produced by PWA system and controller operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PwaError {
    /// No region's halfspace condition is satisfied by the current state.
    NoActiveRegion,
    /// The requested region index is out of range.
    InvalidRegion,
    /// A configuration parameter is invalid (e.g. R == 0 or N == 0).
    InvalidParameter,
}

/// Compute the dot product of two fixed-size arrays.
#[inline]
fn dot<S: ControlScalar, const N: usize>(a: &[S; N], b: &[S; N]) -> S {
    let mut acc = S::ZERO;
    for (ai, bi) in a.iter().zip(b.iter()) {
        acc += *ai * *bi;
    }
    acc
}

/// Discrete-time Piecewise Affine system.
///
/// # Type Parameters
/// * `S` — scalar type implementing [`ControlScalar`]
/// * `N` — state dimension
/// * `I` — input dimension
/// * `R` — number of polyhedral regions
pub struct PwaSystem<S, const N: usize, const I: usize, const R: usize> {
    /// State matrices per region: a_regions[r][row][col]
    a_regions: [[[S; N]; N]; R],
    /// Input matrices per region: b_regions[r][row][col]
    b_regions: [[[S; I]; N]; R],
    /// Affine offsets per region: f_regions[r][i]
    f_regions: [[S; N]; R],
    /// Halfspace normal vectors per region: h_coeffs[r][i]
    h_coeffs: [[S; N]; R],
    /// Halfspace right-hand side values per region: h_bounds[r]
    h_bounds: [S; R],
    state: [S; N],
    active_region: usize,
}

impl<S: ControlScalar, const N: usize, const I: usize, const R: usize> PwaSystem<S, N, I, R> {
    /// Create a new PWA system.
    ///
    /// The initial active region is determined from `x0`; if no region contains
    /// `x0` the first region (index 0) is used as a fallback.
    ///
    /// # Errors
    /// Returns [`PwaError::InvalidParameter`] if `R == 0` or `N == 0`.
    pub fn new(
        a_regions: [[[S; N]; N]; R],
        b_regions: [[[S; I]; N]; R],
        f_regions: [[S; N]; R],
        h_coeffs: [[S; N]; R],
        h_bounds: [S; R],
        x0: [S; N],
    ) -> Result<Self, PwaError> {
        if R == 0 || N == 0 {
            return Err(PwaError::InvalidParameter);
        }
        // Find first region containing x0; fall back to region 0.
        let initial_region = (0..R)
            .find(|&r| dot(&h_coeffs[r], &x0) <= h_bounds[r])
            .unwrap_or(0);
        Ok(Self {
            a_regions,
            b_regions,
            f_regions,
            h_coeffs,
            h_bounds,
            state: x0,
            active_region: initial_region,
        })
    }

    /// Find the first region index `r` such that `h_coeffs[r] · x ≤ h_bounds[r]`.
    ///
    /// Returns `None` if no region satisfies the condition.
    pub fn find_region(&self, x: &[S; N]) -> Option<usize> {
        (0..R).find(|&r| dot(&self.h_coeffs[r], x) <= self.h_bounds[r])
    }

    /// Advance the system by one discrete step with input `u`.
    ///
    /// 1. Finds the active region for the current state.
    /// 2. Applies `x_new = A[r]*x + B[r]*u + f[r]`.
    /// 3. Updates the stored state and active region.
    ///
    /// # Returns
    /// `(active_region_used, new_state)`
    ///
    /// # Errors
    /// Returns [`PwaError::NoActiveRegion`] if no region contains the current state.
    pub fn step(&mut self, u: &[S; I]) -> Result<(usize, [S; N]), PwaError> {
        let r = self
            .find_region(&self.state)
            .ok_or(PwaError::NoActiveRegion)?;
        self.active_region = r;

        // x_new[i] = sum_j A[r][i][j]*x[j] + sum_k B[r][i][k]*u[k] + f[r][i]
        let mut x_new = [S::ZERO; N];
        for (i, x_new_i) in x_new.iter_mut().enumerate() {
            let mut acc = S::ZERO;
            for (j, x_j) in self.state.iter().enumerate() {
                acc += self.a_regions[r][i][j] * *x_j;
            }
            for (k, u_k) in u.iter().enumerate() {
                acc += self.b_regions[r][i][k] * *u_k;
            }
            acc += self.f_regions[r][i];
            *x_new_i = acc;
        }
        self.state = x_new;
        // Update active region for new state (best-effort; fallback unchanged)
        if let Some(new_r) = self.find_region(&self.state) {
            self.active_region = new_r;
        }
        Ok((r, self.state))
    }

    /// Return a reference to the current state.
    #[inline]
    pub fn state(&self) -> &[S; N] {
        &self.state
    }

    /// Return the index of the region active at the last step.
    #[inline]
    pub fn active_region(&self) -> usize {
        self.active_region
    }
}

/// Piecewise Affine state-feedback controller.
///
/// In region `r`, the control law is:
///   u_i = -(K[r][i,:] · x)   for i = 0..I
///
/// # Type Parameters
/// * `S` — scalar type implementing [`ControlScalar`]
/// * `N` — state dimension
/// * `I` — input (control) dimension
/// * `R` — number of polyhedral regions
pub struct PwaController<S, const N: usize, const I: usize, const R: usize> {
    /// Per-region gain matrices: k_gains[r][i][j] → K[r] has I rows, N cols
    k_gains: [[[S; N]; I]; R],
    h_coeffs: [[S; N]; R],
    h_bounds: [S; R],
}

impl<S: ControlScalar, const N: usize, const I: usize, const R: usize> PwaController<S, N, I, R> {
    /// Create a new PWA controller.
    ///
    /// # Errors
    /// Returns [`PwaError::InvalidParameter`] if `R == 0` or `N == 0`.
    pub fn new(
        k_gains: [[[S; N]; I]; R],
        h_coeffs: [[S; N]; R],
        h_bounds: [S; R],
    ) -> Result<Self, PwaError> {
        if R == 0 || N == 0 {
            return Err(PwaError::InvalidParameter);
        }
        Ok(Self {
            k_gains,
            h_coeffs,
            h_bounds,
        })
    }

    /// Find the first region index `r` such that `h_coeffs[r] · x ≤ h_bounds[r]`.
    pub fn find_region(&self, x: &[S; N]) -> Option<usize> {
        (0..R).find(|&r| dot(&self.h_coeffs[r], x) <= self.h_bounds[r])
    }

    /// Compute the control input `u` for state `x`.
    ///
    /// Selects the active region, then applies u_i = -(K[r][i,:] · x).
    ///
    /// # Errors
    /// Returns [`PwaError::NoActiveRegion`] if no region contains `x`.
    pub fn control(&self, x: &[S; N]) -> Result<[S; I], PwaError> {
        let r = self.find_region(x).ok_or(PwaError::NoActiveRegion)?;
        let mut u = [S::ZERO; I];
        for (i, u_i) in u.iter_mut().enumerate() {
            *u_i = -dot(&self.k_gains[r][i], x);
        }
        Ok(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === PwaSystem tests ===

    /// Build a simple 2-region, 1-state, 1-input PWA system:
    ///   Region 0: h=[1]*x ≤ 0  (x ≤ 0)  → A0=[[0.5]], B0=[[1]], f0=[0]
    ///   Region 1: h=[-1]*x ≤ 1e12 (always for x < 1e12) → A1=[[0.8]], B1=[[1]], f1=[0]
    ///
    /// For PwaSystem<f64, 1, 1, 2>:
    ///   a_regions: [[[f64; 1]; 1]; 2] → two matrices [[f64;1];1]
    fn make_1s1i2r() -> PwaSystem<f64, 1, 1, 2> {
        // a0: [[f64; 1]; 1] — 1×1 state matrix
        let a0: [[f64; 1]; 1] = [[0.5_f64]];
        let a1: [[f64; 1]; 1] = [[0.8_f64]];
        // b0: [[f64; 1]; 1] — 1×1 input matrix (N rows, I cols)
        let b0: [[f64; 1]; 1] = [[1.0_f64]];
        let b1: [[f64; 1]; 1] = [[1.0_f64]];
        // f0: [f64; 1] — affine offset
        let f0: [f64; 1] = [0.0_f64];
        let f1: [f64; 1] = [0.0_f64];
        // h_coeffs: [[f64; 1]; 2] — halfspace normals
        let h_coeffs: [[f64; 1]; 2] = [[1.0_f64], [-1.0_f64]];
        let h_bounds: [f64; 2] = [0.0_f64, 1e12_f64];
        // x0: [f64; 1]
        PwaSystem::new([a0, a1], [b0, b1], [f0, f1], h_coeffs, h_bounds, [-1.0_f64]).unwrap()
    }

    #[test]
    fn region_selection_correct() {
        let sys = make_1s1i2r();
        // x = -1: h[0]·x = 1*(-1) = -1 ≤ 0 → region 0
        assert_eq!(sys.find_region(&[-1.0_f64]), Some(0));
        // x = 1: h[0]·x = 1 > 0, h[1]·x = -1 ≤ 1e12 → region 1
        assert_eq!(sys.find_region(&[1.0_f64]), Some(1));
    }

    #[test]
    fn two_region_step_different_dynamics() {
        let mut sys = make_1s1i2r();
        // Start x=-1 (region 0): x_new = 0.5*(-1) + 1*0 + 0 = -0.5
        let (r, state) = sys.step(&[0.0_f64]).unwrap();
        assert_eq!(r, 0, "Should be in region 0");
        assert!((state[0] - (-0.5)).abs() < 1e-12);

        // Now x=-0.5 still in region 0: x_new = 0.5*(-0.5) = -0.25
        let (r2, state2) = sys.step(&[0.0_f64]).unwrap();
        assert_eq!(r2, 0);
        assert!((state2[0] - (-0.25)).abs() < 1e-12);
    }

    #[test]
    fn region1_dynamics_applied() {
        // Start at x=1.0 → region 1, A1=[[0.8]]
        let a0: [[f64; 1]; 1] = [[0.5_f64]];
        let a1: [[f64; 1]; 1] = [[0.8_f64]];
        let b0: [[f64; 1]; 1] = [[1.0_f64]];
        let b1: [[f64; 1]; 1] = [[1.0_f64]];
        let f0: [f64; 1] = [0.0_f64];
        let f1: [f64; 1] = [0.0_f64];
        let h_coeffs: [[f64; 1]; 2] = [[1.0_f64], [-1.0_f64]];
        let h_bounds: [f64; 2] = [0.0_f64, 1e12_f64];
        let mut sys =
            PwaSystem::new([a0, a1], [b0, b1], [f0, f1], h_coeffs, h_bounds, [1.0_f64]).unwrap();
        let (r, state) = sys.step(&[0.0_f64]).unwrap();
        assert_eq!(r, 1, "Should be in region 1 for x=1");
        assert!((state[0] - 0.8).abs() < 1e-12, "x_new should be 0.8*1.0");
    }

    #[test]
    fn affine_offset_applied() {
        // f[0] = 2.0: check offset is added when A=0, B=0
        let a: [[[f64; 1]; 1]; 2] = [[[0.0_f64]], [[0.0_f64]]];
        let b: [[[f64; 1]; 1]; 2] = [[[0.0_f64]], [[0.0_f64]]];
        let f: [[f64; 1]; 2] = [[2.0_f64], [5.0_f64]];
        let h: [[f64; 1]; 2] = [[1.0_f64], [-1.0_f64]];
        let hb: [f64; 2] = [0.0_f64, 1e12];
        let mut sys = PwaSystem::new(a, b, f, h, hb, [-1.0_f64]).unwrap();
        // x=-1 → region 0, x_new = 0*(-1) + 0*0 + 2 = 2.0
        let (r, state) = sys.step(&[0.0_f64]).unwrap();
        assert_eq!(r, 0);
        assert!(
            (state[0] - 2.0).abs() < 1e-12,
            "Affine offset f[0]=2 should be added"
        );
    }

    #[test]
    fn no_active_region_returns_error() {
        // Both regions have h·x ≤ -1e12 (never satisfied for finite x)
        let a: [[[f64; 1]; 1]; 2] = [[[0.5_f64]], [[0.5_f64]]];
        let b: [[[f64; 1]; 1]; 2] = [[[1.0_f64]], [[1.0_f64]]];
        let f: [[f64; 1]; 2] = [[0.0_f64], [0.0_f64]];
        let h: [[f64; 1]; 2] = [[1.0_f64], [1.0_f64]];
        let hb: [f64; 2] = [-1e12_f64, -1e12_f64];
        // Initial state falls back to region 0 (find returns None, fallback=0)
        let mut sys = PwaSystem::new(a, b, f, h, hb, [0.0_f64]).unwrap();
        // step() calls find_region which returns None → NoActiveRegion
        let err = sys.step(&[0.0_f64]);
        assert_eq!(err.err(), Some(PwaError::NoActiveRegion));
    }

    // === PwaController tests ===

    /// Build a 2-region, 2-state, 1-input PWA controller.
    /// Region 0: x[0] ≤ 0, K0 = [[1.0, 0.0]]  → u = -x[0]
    /// Region 1: catches x[0] > 0, K1 = [[0.0, 2.0]] → u = -2*x[1]
    ///
    /// For PwaController<f64, 2, 1, 2>:
    ///   k_gains: [[[f64; 2]; 1]; 2]
    ///     k_gains[r]: [[f64; 2]; 1] — I=1 row of N=2 coefficients
    fn make_ctrl_2s1i2r() -> PwaController<f64, 2, 1, 2> {
        // k0: [[f64; 2]; 1] — 1 row with 2 cols
        let k0: [[f64; 2]; 1] = [[1.0_f64, 0.0_f64]];
        let k1: [[f64; 2]; 1] = [[0.0_f64, 2.0_f64]];
        let h: [[f64; 2]; 2] = [[1.0_f64, 0.0_f64], [-1.0_f64, 0.0_f64]];
        let hb: [f64; 2] = [0.0_f64, 1e12_f64];
        PwaController::new([k0, k1], h, hb).unwrap()
    }

    #[test]
    fn pwa_controller_picks_right_gain() {
        let ctrl = make_ctrl_2s1i2r();
        // x = [-2, 5]: h[0]·x = 1*(-2) = -2 ≤ 0 → region 0, u = -(1*(-2)+0*5) = 2
        let u = ctrl.control(&[-2.0, 5.0]).unwrap();
        assert!((u[0] - 2.0).abs() < 1e-12, "Expected u=2.0, got {}", u[0]);
    }

    #[test]
    fn pwa_controller_region1_gain() {
        let ctrl = make_ctrl_2s1i2r();
        // x = [3, 4]: h[0]·x = 3 > 0 → fail; h[1]·x = -3 ≤ 1e12 → region 1
        // u = -(0*3 + 2*4) = -8
        let u = ctrl.control(&[3.0, 4.0]).unwrap();
        assert!(
            (u[0] - (-8.0)).abs() < 1e-12,
            "Expected u=-8.0, got {}",
            u[0]
        );
    }

    #[test]
    fn pwa_controller_no_region_error() {
        // Both regions impossible: h·x ≤ -1e12 never holds for finite x
        // PwaController<f64, 1, 1, 2>: k_gains [[[f64;1];1];2], h [[f64;1];2], hb [f64;2]
        let k: [[[f64; 1]; 1]; 2] = [[[1.0_f64]], [[1.0_f64]]];
        let h: [[f64; 1]; 2] = [[1.0_f64], [1.0_f64]];
        let hb: [f64; 2] = [-1e12_f64, -1e12_f64];
        let ctrl = PwaController::<f64, 1, 1, 2>::new(k, h, hb).unwrap();
        assert_eq!(
            ctrl.control(&[0.0_f64]).err(),
            Some(PwaError::NoActiveRegion)
        );
    }

    #[test]
    fn pwa_system_active_region_tracks() {
        let mut sys = make_1s1i2r();
        // Start: x=-1, region 0. Step to x=-0.5, still region 0.
        sys.step(&[0.0_f64]).unwrap();
        assert_eq!(sys.active_region(), 0);
    }

    #[test]
    fn pwa_controller_multi_input() {
        // PwaController<f64, 2, 2, 1>:
        //   k_gains: [[[f64;2];2];1] — 1 region, 2 input rows, 2 state cols
        //   K0 = [[1,0],[0,1]]: u_0=-x[0], u_1=-x[1]
        let k0: [[f64; 2]; 2] = [[1.0_f64, 0.0_f64], [0.0_f64, 1.0_f64]];
        let h: [[f64; 2]; 1] = [[-1.0_f64, 0.0_f64]]; // always satisfied
        let hb: [f64; 1] = [1e12_f64];
        let ctrl = PwaController::<f64, 2, 2, 1>::new([k0], h, hb).unwrap();
        let u = ctrl.control(&[3.0_f64, 5.0_f64]).unwrap();
        assert!((u[0] - (-3.0)).abs() < 1e-12);
        assert!((u[1] - (-5.0)).abs() < 1e-12);
    }

    #[test]
    fn pwa_system_input_drives_state() {
        // x[0] starts at -1 (region 0), u=2: x_new = 0.5*(-1) + 1*2 = 1.5
        let mut sys = make_1s1i2r();
        let (r, state) = sys.step(&[2.0_f64]).unwrap();
        assert_eq!(r, 0, "Started in region 0");
        assert!((state[0] - 1.5).abs() < 1e-12, "x_new should be 1.5");
    }
}
