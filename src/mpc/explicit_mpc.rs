use crate::core::scalar::ControlScalar;

/// A single affine control region for explicit MPC.
///
/// Control law within this region: u = K·x + k0
/// Region defined by box: x_min[i] ≤ x[i] ≤ x_max[i]
#[derive(Debug, Clone, Copy)]
pub struct MpcRegion<S: ControlScalar, const N: usize, const M: usize> {
    /// Feedback gain matrix (M×N).
    pub gain: [[S; N]; M],
    /// Affine offset (M).
    pub offset: [S; M],
    /// Box region lower bounds.
    pub x_min: [S; N],
    /// Box region upper bounds.
    pub x_max: [S; N],
}

impl<S: ControlScalar, const N: usize, const M: usize> MpcRegion<S, N, M> {
    pub fn new(gain: [[S; N]; M], offset: [S; M], x_min: [S; N], x_max: [S; N]) -> Self {
        Self {
            gain,
            offset,
            x_min,
            x_max,
        }
    }

    /// Check if state `x` falls in this region.
    pub fn contains(&self, x: &[S; N]) -> bool {
        x.iter()
            .enumerate()
            .all(|(i, &xi)| xi >= self.x_min[i] && xi <= self.x_max[i])
    }

    /// Compute control action: u = K·x + k0
    pub fn compute(&self, x: &[S; N]) -> [S; M] {
        core::array::from_fn(|i| {
            self.offset[i]
                + self.gain[i]
                    .iter()
                    .zip(x.iter())
                    .fold(S::ZERO, |acc, (&g, &xi)| acc + g * xi)
        })
    }
}

/// Explicit (pre-computed) MPC with R piecewise-affine regions.
///
/// At runtime, finds the matching region for the current state and
/// applies the affine control law. O(R) lookup — O(1) per region check.
///
/// `R` = number of regions.
#[derive(Debug, Clone, Copy)]
pub struct ExplicitMpc<S: ControlScalar, const N: usize, const M: usize, const R: usize> {
    regions: [Option<MpcRegion<S, N, M>>; R],
    /// Fallback control (used when no region matches — should not happen in a
    /// well-designed explicit MPC, but provides a safe default).
    pub fallback: [S; M],
    /// Input saturation.
    pub u_min: [S; M],
    pub u_max: [S; M],
}

impl<S: ControlScalar, const N: usize, const M: usize, const R: usize> ExplicitMpc<S, N, M, R> {
    pub fn new(u_min: [S; M], u_max: [S; M]) -> Self {
        Self {
            regions: core::array::from_fn(|_| None),
            fallback: [S::ZERO; M],
            u_min,
            u_max,
        }
    }

    /// Add a region. Returns false if no free slots remain.
    pub fn add_region(&mut self, region: MpcRegion<S, N, M>) -> bool {
        for slot in self.regions.iter_mut() {
            if slot.is_none() {
                *slot = Some(region);
                return true;
            }
        }
        false
    }

    /// Evaluate the explicit MPC law for state `x`.
    ///
    /// Searches regions in order, returns first match.
    /// Falls back to `self.fallback` if no region matches.
    pub fn compute(&self, x: &[S; N]) -> [S; M] {
        for region in self.regions.iter().flatten() {
            if region.contains(x) {
                let u = region.compute(x);
                return core::array::from_fn(|i| u[i].clamp_val(self.u_min[i], self.u_max[i]));
            }
        }
        self.fallback
    }

    /// Number of active regions.
    pub fn region_count(&self) -> usize {
        self.regions.iter().filter(|r| r.is_some()).count()
    }
}

/// Build a simple explicit MPC for a scalar first-order system
/// by partitioning the state space into sign-based regions.
///
/// Plant: x_{k+1} = a*x + b*u
/// Optimal (unconstrained LQR-like) gain: K such that u = -K*x
///
/// Two regions: x > 0 → u = -K*x, x ≤ 0 → u = -K*x (same gain, two regions)
pub fn build_scalar_explicit_mpc<S: ControlScalar>(
    gain_k: S,
    x_min: S,
    x_max: S,
    u_min: S,
    u_max: S,
) -> ExplicitMpc<S, 1, 1, 2> {
    let mut empc = ExplicitMpc::new([u_min], [u_max]);

    // Region 1: x ≥ 0
    let r1 = MpcRegion::new([[-gain_k]], [S::ZERO], [S::ZERO], [x_max]);
    // Region 2: x < 0
    let r2 = MpcRegion::new([[-gain_k]], [S::ZERO], [x_min], [S::ZERO]);
    empc.add_region(r1);
    empc.add_region(r2);
    empc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_contains_check() {
        let region = MpcRegion::new([[-1.0_f64]], [0.0], [-5.0], [5.0]);
        assert!(region.contains(&[0.0_f64]));
        assert!(region.contains(&[-5.0_f64]));
        assert!(!region.contains(&[6.0_f64]));
    }

    #[test]
    fn explicit_mpc_computes_correctly() {
        let empc = build_scalar_explicit_mpc(0.5_f64, -10.0, 10.0, -1.0, 1.0);
        // x = 2.0 → u = -0.5 * 2.0 = -1.0 (clamped to u_max? no: -1.0 >= -1.0)
        let u = empc.compute(&[2.0_f64]);
        assert!((u[0] - (-1.0)).abs() < 1e-10, "u={:.4}", u[0]);

        // x = -1.0 → u = -0.5 * (-1.0) = 0.5
        let u2 = empc.compute(&[-1.0_f64]);
        assert!((u2[0] - 0.5).abs() < 1e-10, "u2={:.4}", u2[0]);
    }

    #[test]
    fn explicit_mpc_fallback_on_no_match() {
        let mut empc = ExplicitMpc::<f64, 1, 1, 1>::new([-1.0], [1.0]);
        // One region only covers [0, 5]
        let r = MpcRegion::new([[-0.5]], [0.0], [0.0], [5.0]);
        empc.add_region(r);
        empc.fallback = [0.0];
        // State outside all regions
        let u = empc.compute(&[-2.0_f64]);
        assert_eq!(u[0], 0.0);
    }

    #[test]
    fn region_count_correct() {
        let empc = build_scalar_explicit_mpc(0.5_f64, -10.0, 10.0, -1.0, 1.0);
        assert_eq!(empc.region_count(), 2);
    }

    #[test]
    fn explicit_mpc_stabilizes_first_order() {
        // x_{k+1} = 0.9*x + u, gain K=0.5 → poles at 0.9-0.5=0.4 (stable)
        let empc = build_scalar_explicit_mpc(0.5_f64, -100.0, 100.0, -5.0, 5.0);
        let mut x = 3.0_f64;
        for _ in 0..50 {
            let u = empc.compute(&[x]);
            x = 0.9 * x + u[0];
        }
        assert!(x.abs() < 0.1, "x={:.4} should converge to 0", x);
    }
}
