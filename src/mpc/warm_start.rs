use crate::core::scalar::ControlScalar;

/// Warm-start manager for MPC optimization.
///
/// Stores previous optimal input sequence and provides a warm-started
/// initial guess for the next optimization step by:
///   1. Shifting: u_k = u*_{k+1} for k = 0..H-2 (receding horizon shift)
///   2. Tail fill: u_{H-1} = u*_{H-1} (hold last, or zero, or extrapolate)
///
/// Reduces iterations needed for convergence by ~60-80% in practice.
#[derive(Debug, Clone, Copy)]
pub struct WarmStart<S: ControlScalar, const M: usize, const H: usize> {
    /// Previous optimal input sequence.
    u_prev: [[S; M]; H],
    /// Whether a valid previous solution exists.
    pub has_solution: bool,
    /// Tail fill strategy.
    pub strategy: WarmStartStrategy,
}

/// Strategy for filling the last element after the shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmStartStrategy {
    /// Hold the last control input (u[H-1] = u*[H-1]).
    HoldLast,
    /// Zero fill (u[H-1] = 0).
    ZeroFill,
    /// Linear extrapolate from last two values.
    Extrapolate,
}

impl<S: ControlScalar, const M: usize, const H: usize> WarmStart<S, M, H> {
    pub fn new(strategy: WarmStartStrategy) -> Self {
        Self {
            u_prev: [[S::ZERO; M]; H],
            has_solution: false,
            strategy,
        }
    }

    /// Record new solution for next warm-start.
    pub fn record(&mut self, u_opt: &[[S; M]; H]) {
        self.u_prev = *u_opt;
        self.has_solution = true;
    }

    /// Generate warm-started initial guess by shifting u_prev by one step.
    pub fn shifted(&self) -> [[S; M]; H] {
        if !self.has_solution {
            return [[S::ZERO; M]; H];
        }

        let mut u_init = [[S::ZERO; M]; H];

        // Shift: u_init[k] = u_prev[k+1] for k < H-1
        let n = H.saturating_sub(1);
        u_init[..n].copy_from_slice(&self.u_prev[1..n + 1]);

        // Fill tail based on strategy
        if H > 0 {
            let tail = H - 1;
            match self.strategy {
                WarmStartStrategy::HoldLast => {
                    u_init[tail] = self.u_prev[H - 1];
                }
                WarmStartStrategy::ZeroFill => {
                    u_init[tail] = [S::ZERO; M];
                }
                WarmStartStrategy::Extrapolate => {
                    if H >= 2 {
                        // Linear extrapolation: u[H-1] = 2*u_prev[H-1] - u_prev[H-2]
                        u_init[tail] = core::array::from_fn(|i| {
                            S::TWO * self.u_prev[H - 1][i] - self.u_prev[H - 2][i]
                        });
                    } else {
                        u_init[tail] = self.u_prev[H - 1];
                    }
                }
            }
        }

        u_init
    }

    /// Reset warm-start state (clear previous solution).
    pub fn reset(&mut self) {
        self.u_prev = [[S::ZERO; M]; H];
        self.has_solution = false;
    }

    /// Check if the warm-started guess is within bounds.
    pub fn clip_to_bounds(u_init: &mut [[S; M]; H], u_min: &[S; M], u_max: &[S; M]) {
        for uk in u_init.iter_mut() {
            for (i, ui) in uk.iter_mut().enumerate() {
                *ui = ui.clamp_val(u_min[i], u_max[i]);
            }
        }
    }
}

/// Multi-step warm-start cache: stores multiple candidate solutions.
///
/// Useful when the problem has multiple local optima.
/// At each step, pick the candidate with lowest cost as warm start.
#[derive(Debug, Clone, Copy)]
pub struct MultiSolutionCache<S: ControlScalar, const M: usize, const H: usize, const C: usize> {
    solutions: [[[S; M]; H]; C],
    costs: [S; C],
    n_stored: usize,
}

impl<S: ControlScalar, const M: usize, const H: usize, const C: usize>
    MultiSolutionCache<S, M, H, C>
{
    pub fn new() -> Self {
        Self {
            solutions: [[[S::ZERO; M]; H]; C],
            costs: [S::from_f64(f64::MAX / 2.0); C],
            n_stored: 0,
        }
    }

    /// Store a solution with its cost. Replaces worst if full.
    pub fn store(&mut self, u_seq: [[S; M]; H], cost: S) {
        if self.n_stored < C {
            let idx = self.n_stored;
            self.solutions[idx] = u_seq;
            self.costs[idx] = cost;
            self.n_stored += 1;
        } else {
            // Replace worst
            let mut worst_idx = 0;
            let mut worst_cost = self.costs[0];
            for (i, &c) in self.costs.iter().enumerate().skip(1) {
                if c > worst_cost {
                    worst_cost = c;
                    worst_idx = i;
                }
            }
            if cost < worst_cost {
                self.solutions[worst_idx] = u_seq;
                self.costs[worst_idx] = cost;
            }
        }
    }

    /// Get the best (lowest cost) stored solution.
    pub fn best(&self) -> Option<([[S; M]; H], S)> {
        if self.n_stored == 0 {
            return None;
        }
        let mut best_idx = 0;
        let mut best_cost = self.costs[0];
        for (i, &c) in self.costs[..self.n_stored].iter().enumerate().skip(1) {
            if c < best_cost {
                best_cost = c;
                best_idx = i;
            }
        }
        Some((self.solutions[best_idx], best_cost))
    }

    pub fn clear(&mut self) {
        self.n_stored = 0;
        self.costs = [S::from_f64(f64::MAX / 2.0); C];
    }
}

impl<S: ControlScalar, const M: usize, const H: usize, const C: usize> Default
    for MultiSolutionCache<S, M, H, C>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_start_shift_hold_last() {
        let mut ws = WarmStart::<f64, 1, 4>::new(WarmStartStrategy::HoldLast);
        let u_opt = [[1.0], [2.0], [3.0], [4.0]];
        ws.record(&u_opt);
        let shifted = ws.shifted();
        assert_eq!(shifted[0], [2.0]);
        assert_eq!(shifted[1], [3.0]);
        assert_eq!(shifted[2], [4.0]);
        assert_eq!(shifted[3], [4.0]); // hold last
    }

    #[test]
    fn warm_start_shift_zero_fill() {
        let mut ws = WarmStart::<f64, 1, 3>::new(WarmStartStrategy::ZeroFill);
        ws.record(&[[1.0], [2.0], [3.0]]);
        let shifted = ws.shifted();
        assert_eq!(shifted[0], [2.0]);
        assert_eq!(shifted[1], [3.0]);
        assert_eq!(shifted[2], [0.0]); // zero fill
    }

    #[test]
    fn warm_start_extrapolate() {
        let mut ws = WarmStart::<f64, 1, 4>::new(WarmStartStrategy::Extrapolate);
        ws.record(&[[1.0], [2.0], [3.0], [4.0]]);
        let shifted = ws.shifted();
        // tail = 2*4 - 3 = 5
        assert!(
            (shifted[3][0] - 5.0).abs() < 1e-10,
            "tail={:.4}",
            shifted[3][0]
        );
    }

    #[test]
    fn warm_start_no_solution_returns_zeros() {
        let ws = WarmStart::<f64, 2, 3>::new(WarmStartStrategy::HoldLast);
        let shifted = ws.shifted();
        for row in &shifted {
            assert_eq!(*row, [0.0, 0.0]);
        }
    }

    #[test]
    fn multi_cache_best_solution() {
        let mut cache = MultiSolutionCache::<f64, 1, 2, 3>::new();
        cache.store([[1.0], [2.0]], 10.0);
        cache.store([[3.0], [4.0]], 5.0);
        cache.store([[5.0], [6.0]], 15.0);
        let (sol, cost) = cache.best().unwrap();
        assert!((cost - 5.0).abs() < 1e-10);
        assert_eq!(sol[0], [3.0]);
    }

    #[test]
    fn multi_cache_replaces_worst() {
        let mut cache = MultiSolutionCache::<f64, 1, 1, 2>::new();
        cache.store([[0.0]], 100.0);
        cache.store([[1.0]], 50.0);
        cache.store([[2.0]], 1.0); // should replace 100.0
        let (_, cost) = cache.best().unwrap();
        assert!((cost - 1.0).abs() < 1e-10);
    }

    #[test]
    fn clip_to_bounds() {
        let mut u = [[5.0_f64], [-5.0]];
        WarmStart::<f64, 1, 2>::clip_to_bounds(&mut u, &[-2.0], &[2.0]);
        assert_eq!(u[0], [2.0]);
        assert_eq!(u[1], [-2.0]);
    }
}
