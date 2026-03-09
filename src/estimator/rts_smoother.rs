use crate::core::matrix::{matmul, matvec, Matrix};
use crate::core::scalar::ControlScalar;

/// Error type for the RTS smoother.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmootherError {
    /// A required matrix inversion failed (singular matrix).
    SingularMatrix,
    /// The store buffer is full (index out of range).
    BufferFull,
    /// The requested index is beyond what was stored.
    IndexOutOfRange,
    /// Not enough states have been stored to perform smoothing.
    InsufficientData,
}

impl core::fmt::Display for SmootherError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SmootherError::SingularMatrix => write!(f, "RtsSmoother: singular matrix"),
            SmootherError::BufferFull => write!(f, "RtsSmoother: buffer full"),
            SmootherError::IndexOutOfRange => write!(f, "RtsSmoother: index out of range"),
            SmootherError::InsufficientData => {
                write!(f, "RtsSmoother: insufficient data for smoothing")
            }
        }
    }
}

/// One slot of forward-filter data needed for the backward smoothing pass.
///
/// Stores the **posterior** (filtered) and **prior** (predicted) state and
/// covariance at a single time step k.
#[derive(Debug, Clone, Copy)]
pub struct FilteredState<S: ControlScalar, const N: usize> {
    /// Posterior state estimate x_{k|k}.
    pub x: [S; N],
    /// Posterior covariance P_{k|k}.
    pub p: Matrix<S, N, N>,
    /// Prior (predicted) state estimate x_{k|k-1}.
    pub x_pred: [S; N],
    /// Prior (predicted) covariance P_{k|k-1}.
    pub p_pred: Matrix<S, N, N>,
}

impl<S: ControlScalar, const N: usize> FilteredState<S, N> {
    /// Create a new `FilteredState` explicitly.
    pub fn new(x: [S; N], p: Matrix<S, N, N>, x_pred: [S; N], p_pred: Matrix<S, N, N>) -> Self {
        Self {
            x,
            p,
            x_pred,
            p_pred,
        }
    }
}

/// Output of the RTS backward smoothing pass.
///
/// Contains T smoothed states, each with a smoothed state vector and
/// smoothed covariance matrix.
#[derive(Debug, Clone, Copy)]
pub struct SmoothedState<S: ControlScalar, const N: usize> {
    /// Smoothed state estimate x_{k|N}.
    pub x: [S; N],
    /// Smoothed covariance P_{k|N}.
    pub p: Matrix<S, N, N>,
}

/// Container for T smoothed states.
#[derive(Debug, Clone, Copy)]
pub struct SmoothedData<S: ControlScalar, const N: usize, const T: usize> {
    /// Array of smoothed states, indexed 0..T.
    pub states: [SmoothedState<S, N>; T],
    /// Number of valid entries.
    pub len: usize,
}

/// Rauch-Tung-Striebel (RTS) Kalman Smoother.
///
/// Performs offline smoothing over a stored sequence of forward Kalman filter
/// passes.  The smoother improves all past estimates by using future
/// measurements.
///
/// ## Algorithm
///
/// **Forward pass** (done externally, results stored via `store_forward`):
/// Run a standard Kalman filter, storing at each step k:
/// - x_{k|k}, P_{k|k}  (posterior)
/// - x_{k+1|k}, P_{k+1|k}  (predicted, i.e. the *next* step's prior)
///
/// **Backward pass** (performed by `smooth`):
/// Starting from k = T-1 downto 0:
/// ```text
///   G_k  = P_{k|k} · Aᵀ · P_{k+1|k}⁻¹            (smoother gain)
///   x_{k|N} = x_{k|k} + G_k · (x_{k+1|N} - x_{k+1|k})
///   P_{k|N} = P_{k|k} + G_k · (P_{k+1|N} - P_{k+1|k}) · G_kᵀ
/// ```
///
/// # Type Parameters
/// * `S` — scalar type (`f32` or `f64`)
/// * `N` — state dimension
/// * `T` — maximum number of time steps to store (compile-time constant)
#[derive(Debug, Clone, Copy)]
pub struct RtsSmoother<S: ControlScalar, const N: usize, const T: usize> {
    /// Stored forward-pass states.
    buffer: [FilteredState<S, N>; T],
    /// Number of states currently stored.
    count: usize,
}

impl<S: ControlScalar, const N: usize, const T: usize> RtsSmoother<S, N, T> {
    /// Create a new empty smoother.
    pub fn new() -> Self {
        // Safety: FilteredState is composed of Copy types; zero-initialise.
        let zero_state = FilteredState {
            x: [S::ZERO; N],
            p: Matrix::zeros(),
            x_pred: [S::ZERO; N],
            p_pred: Matrix::zeros(),
        };
        Self {
            buffer: [zero_state; T],
            count: 0,
        }
    }

    /// Reset the smoother, clearing all stored states.
    pub fn reset(&mut self) {
        self.count = 0;
    }

    /// Store one filtered state produced by the forward Kalman pass.
    ///
    /// Call this after each forward predict+update cycle in time order
    /// (k = 0, 1, 2, …).
    ///
    /// `state.x_pred` / `state.p_pred` should be the *predicted* quantities
    /// for the **same** time step k (i.e. x_{k|k-1} and P_{k|k-1}).
    ///
    /// Returns `Err(SmootherError::BufferFull)` if T slots are already used.
    pub fn store_forward(&mut self, state: FilteredState<S, N>) -> Result<(), SmootherError> {
        if self.count >= T {
            return Err(SmootherError::BufferFull);
        }
        self.buffer[self.count] = state;
        self.count += 1;
        Ok(())
    }

    /// Number of states currently stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true when no states are stored.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Access a stored `FilteredState` by index.
    ///
    /// Returns `None` if `k >= self.len()`.
    pub fn get_state(&self, k: usize) -> Option<&FilteredState<S, N>> {
        if k < self.count {
            Some(&self.buffer[k])
        } else {
            None
        }
    }

    /// Run the RTS backward smoothing pass over all stored states.
    ///
    /// `a` is the state-transition matrix used during the forward filter.
    ///
    /// Returns `SmoothedData` with `len == self.count` smoothed states,
    /// or an error if any matrix inversion fails.
    pub fn smooth(&self, a: &Matrix<S, N, N>) -> Result<SmoothedData<S, N, T>, SmootherError> {
        let n = self.count;
        if n == 0 {
            return Err(SmootherError::InsufficientData);
        }

        // Initialise output array (zero-fill placeholders).
        let zero_smoothed = SmoothedState {
            x: [S::ZERO; N],
            p: Matrix::zeros(),
        };
        let mut out = SmoothedData {
            states: [zero_smoothed; T],
            len: n,
        };

        // Terminal condition: smoothed = filtered at last step.
        out.states[n - 1] = SmoothedState {
            x: self.buffer[n - 1].x,
            p: self.buffer[n - 1].p,
        };

        // Special case: only one stored state — nothing to smooth.
        if n == 1 {
            return Ok(out);
        }

        // Aᵀ — used repeatedly.
        let at = a.transpose();

        // Backward pass: k = n-2 downto 0.
        // At iteration k we need x_{k+1|N} and P_{k+1|N} from the step
        // just computed (stored in out.states[k+1]).
        // We also need x_{k+1|k} = buffer[k].x_pred and
        //              P_{k+1|k} = buffer[k].p_pred.
        //
        // Note: buffer[k].x_pred holds the *predicted* quantities
        //       for time k+1 (i.e. computed after the k-th update and before
        //       the (k+1)-th update). This is the standard storage convention.
        for k in (0..n - 1).rev() {
            let fk = &self.buffer[k];

            // P_{k+1|k} — predicted covariance one step ahead.
            // Stored in buffer[k].p_pred.
            let p_pred_kp1 = &fk.p_pred;

            // Invert P_{k+1|k}.
            let p_pred_kp1_inv = p_pred_kp1.inv().ok_or(SmootherError::SingularMatrix)?;

            // G_k = P_{k|k} · Aᵀ · P_{k+1|k}⁻¹
            let p_at = matmul(&fk.p, &at);
            let g_k = matmul(&p_at, &p_pred_kp1_inv);

            // diff_x = x_{k+1|N} - x_{k+1|k}
            let x_smooth_kp1 = out.states[k + 1].x;
            let x_pred_kp1 = fk.x_pred;
            let diff_x: [S; N] = core::array::from_fn(|i| x_smooth_kp1[i] - x_pred_kp1[i]);

            // x_{k|N} = x_{k|k} + G_k · diff_x
            let g_diff = matvec(&g_k, &diff_x);
            let x_smooth_k: [S; N] = core::array::from_fn(|i| fk.x[i] + g_diff[i]);

            // diff_P = P_{k+1|N} - P_{k+1|k}
            let p_smooth_kp1 = out.states[k + 1].p;
            let diff_p = p_smooth_kp1.sub_mat(p_pred_kp1);

            // P_{k|N} = P_{k|k} + G_k · diff_P · G_kᵀ
            let g_diff_p = matmul(&g_k, &diff_p);
            let gkt = g_k.transpose();
            let correction = matmul(&g_diff_p, &gkt);
            let p_smooth_k = fk.p.add_mat(&correction);

            out.states[k] = SmoothedState {
                x: x_smooth_k,
                p: p_smooth_k,
            };
        }

        Ok(out)
    }
}

impl<S: ControlScalar, const N: usize, const T: usize> Default for RtsSmoother<S, N, T> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::matrix::matmul;

    /// Build a simple 1-D constant-state model (A=1, H=1, Q=q, R=r).
    /// Runs a forward Kalman filter for `steps` steps and stores results.
    fn run_forward_1d(
        steps: usize,
        q: f64,
        r: f64,
    ) -> (RtsSmoother<f64, 1, 64>, Matrix<f64, 1, 1>) {
        let a = Matrix::<f64, 1, 1>::identity();
        let h = Matrix::<f64, 1, 1>::identity();
        let q_mat = Matrix::<f64, 1, 1> { data: [[q]] };
        let r_mat = Matrix::<f64, 1, 1> { data: [[r]] };

        let mut x = [0.0_f64; 1];
        let mut p = Matrix::<f64, 1, 1> { data: [[10.0]] };

        let mut smoother = RtsSmoother::<f64, 1, 64>::new();

        let measurement = 5.0_f64;

        for _ in 0..steps {
            // Predict
            let x_pred = x; // A=I
            let ap = matmul(&a, &p);
            let at = a.transpose();
            let apat = matmul(&ap, &at);
            let p_pred = apat.add_mat(&q_mat);

            // Update
            let hx = [x_pred[0]]; // H=I
            let innov = [measurement - hx[0]];
            let hp = matmul(&h, &p_pred);
            let ht = h.transpose();
            let hpht = matmul(&hp, &ht);
            let s_mat = hpht.add_mat(&r_mat);
            let s_inv = s_mat.inv().expect("S invertible");
            let pht = matmul(&p_pred, &ht);
            let k = matmul(&pht, &s_inv);
            let kv = crate::core::matrix::matvec(&k, &innov);
            let x_post: [f64; 1] = core::array::from_fn(|i| x_pred[i] + kv[i]);
            let kh = matmul(&k, &h);
            let eye = Matrix::<f64, 1, 1>::identity();
            let i_kh = eye.sub_mat(&kh);
            let p_post = matmul(&i_kh, &p_pred);

            smoother
                .store_forward(FilteredState::new(x_post, p_post, x_pred, p_pred))
                .expect("store");

            x = x_post;
            p = p_post;
        }

        (smoother, a)
    }

    #[test]
    fn smoother_covariance_le_filter_covariance() {
        let steps = 10_usize;
        let (smoother, a) = run_forward_1d(steps, 1e-4, 1.0);
        let smoothed = smoother.smooth(&a).expect("smooth");

        for k in 0..steps {
            let p_filter = smoother.buffer[k].p.trace();
            let p_smooth = smoothed.states[k].p.trace();
            assert!(
                p_smooth <= p_filter + 1e-10,
                "Smoothed variance must not exceed filtered variance at k={k}: \
                 p_smooth={p_smooth}, p_filter={p_filter}"
            );
        }
    }

    #[test]
    fn smoother_with_single_step_equals_filter() {
        let (smoother, a) = run_forward_1d(1, 1e-4, 1.0);
        let smoothed = smoother.smooth(&a).expect("smooth");

        assert_eq!(smoothed.len, 1);
        let x_filter = smoother.buffer[0].x[0];
        let x_smooth = smoothed.states[0].x[0];
        assert!(
            (x_filter - x_smooth).abs() < 1e-12,
            "Single step: smoothed must equal filtered. filter={x_filter}, smooth={x_smooth}"
        );
    }

    #[test]
    fn smoother_len_matches_stored_count() {
        let (smoother, a) = run_forward_1d(8, 1e-4, 0.5);
        let smoothed = smoother.smooth(&a).expect("smooth");
        assert_eq!(smoothed.len, 8);
    }

    #[test]
    fn store_forward_returns_error_when_full() {
        let mut smoother = RtsSmoother::<f64, 1, 2>::new();
        let fs = FilteredState::new(
            [0.0_f64; 1],
            Matrix::identity(),
            [0.0_f64; 1],
            Matrix::identity(),
        );
        smoother.store_forward(fs).expect("first store ok");
        smoother.store_forward(fs).expect("second store ok");
        let res = smoother.store_forward(fs);
        assert!(
            matches!(res, Err(SmootherError::BufferFull)),
            "Expected BufferFull"
        );
    }

    #[test]
    fn empty_smoother_returns_insufficient_data() {
        let smoother = RtsSmoother::<f64, 2, 16>::new();
        let a = Matrix::<f64, 2, 2>::identity();
        let res = smoother.smooth(&a);
        assert!(matches!(res, Err(SmootherError::InsufficientData)));
    }

    #[test]
    fn reset_clears_stored_states() {
        let mut smoother = RtsSmoother::<f64, 1, 8>::new();
        let fs = FilteredState::new(
            [1.0_f64; 1],
            Matrix::identity(),
            [1.0_f64; 1],
            Matrix::identity(),
        );
        smoother.store_forward(fs).expect("store");
        assert_eq!(smoother.len(), 1);
        smoother.reset();
        assert_eq!(smoother.len(), 0);
        assert!(smoother.is_empty());
    }
}
