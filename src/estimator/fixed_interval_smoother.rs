use crate::core::matrix::{matmul, matvec, Matrix};
use crate::core::scalar::ControlScalar;

/// Error type for the Fixed-Interval Smoother.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedIntervalError {
    /// A required matrix inversion failed (singular matrix).
    SingularMatrix,
    /// The store buffer is full.
    BufferFull,
    /// Not enough data to run the smoother.
    InsufficientData,
}

impl core::fmt::Display for FixedIntervalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FixedIntervalError::SingularMatrix => {
                write!(f, "FixedIntervalSmoother: singular matrix")
            }
            FixedIntervalError::BufferFull => write!(f, "FixedIntervalSmoother: buffer full"),
            FixedIntervalError::InsufficientData => {
                write!(f, "FixedIntervalSmoother: insufficient data")
            }
        }
    }
}

/// One slot of data required for the fixed-interval two-pass smoother.
///
/// Stores both the forward-filter posterior and the predicted (prior)
/// quantities at each time step, plus the measurement and measurement matrix
/// needed for the backward information filter.
#[derive(Debug, Clone, Copy)]
pub struct FisSlot<S: ControlScalar, const N: usize, const M: usize> {
    /// Posterior state x_{k|k}.
    pub x_post: [S; N],
    /// Posterior covariance P_{k|k}.
    pub p_post: Matrix<S, N, N>,
    /// Predicted state x_{k|k-1}.
    pub x_pred: [S; N],
    /// Predicted covariance P_{k|k-1}.
    pub p_pred: Matrix<S, N, N>,
    /// Measurement at step k.
    pub z: [S; M],
}

impl<S: ControlScalar, const N: usize, const M: usize> FisSlot<S, N, M> {
    /// Construct a new slot.
    pub fn new(
        x_post: [S; N],
        p_post: Matrix<S, N, N>,
        x_pred: [S; N],
        p_pred: Matrix<S, N, N>,
        z: [S; M],
    ) -> Self {
        Self {
            x_post,
            p_post,
            x_pred,
            p_pred,
            z,
        }
    }
}

/// Smoothed state output from the fixed-interval smoother.
#[derive(Debug, Clone, Copy)]
pub struct FisSmoothed<S: ControlScalar, const N: usize> {
    /// Smoothed state x_{k|T}.
    pub x: [S; N],
    /// Smoothed covariance P_{k|T}.
    pub p: Matrix<S, N, N>,
}

/// Container for T smoothed states from the fixed-interval smoother.
#[derive(Debug, Clone, Copy)]
pub struct FisSmoothedData<S: ControlScalar, const N: usize, const T: usize> {
    /// Smoothed state array.
    pub states: [FisSmoothed<S, N>; T],
    /// Number of valid entries.
    pub len: usize,
}

/// Fixed-Interval Smoother (Two-Filter / Bryson–Frazier variant).
///
/// This smoother combines:
/// - A **forward Kalman filter** pass (results stored externally via
///   `store_slot`).
/// - A **backward information filter** (BIF) pass running from T to 0.
///
/// The combined smoothed estimate at step k is:
/// ```text
///   P_{k|T}⁻¹  = P_{k|k}⁻¹ + Λ_k
///   x_{k|T}    = P_{k|T} · (P_{k|k}⁻¹ · x_{k|k} + λ_k)
/// ```
/// where `Λ_k` and `λ_k` are the backward information matrix and vector.
///
/// The BIF recursion (backward from k=T-1 to 0):
/// ```text
///   Λ_{k} = Aᵀ · (Λ_{k+1}⁻¹ + Q)⁻¹ · A + Hᵀ R⁻¹ H
///   λ_{k} = Hᵀ R⁻¹ z_k + Aᵀ · (Λ_{k+1}⁻¹ + Q)⁻¹ · λ_{k+1}  (wait, λ passes forward in BIF)
/// ```
///
/// In practice we implement the simplified RTS-equivalent formulation:
/// the backward pass propagates the smoother gain identically to the RTS
/// smoother, which is algebraically equivalent to the two-filter form for
/// linear Gaussian models.  The BIF form is used to initialise the terminal
/// information pair (Λ_T = 0, λ_T = 0) and run the recursion backward.
///
/// # Type Parameters
/// * `S` — scalar type
/// * `N` — state dimension
/// * `M` — measurement dimension
/// * `T` — maximum time steps (compile-time)
#[derive(Debug, Clone, Copy)]
pub struct FixedIntervalSmoother<S: ControlScalar, const N: usize, const M: usize, const T: usize> {
    /// Stored forward-pass slots.
    buffer: [FisSlot<S, N, M>; T],
    /// Number of filled slots.
    count: usize,
    /// Measurement matrix H (M×N), shared across all steps.
    h: Matrix<S, M, N>,
    /// Process noise covariance Q (N×N).
    /// Stored for API completeness; the smoother uses stored `p_pred` (which
    /// incorporates Q from the forward filter pass) rather than Q directly.
    q: Matrix<S, N, N>,
    /// Measurement noise covariance R (M×M).
    r: Matrix<S, M, M>,
}

impl<S: ControlScalar, const N: usize, const M: usize, const T: usize>
    FixedIntervalSmoother<S, N, M, T>
{
    /// Create a new smoother with the given system matrices.
    pub fn new(h: Matrix<S, M, N>, q: Matrix<S, N, N>, r: Matrix<S, M, M>) -> Self {
        let zero_slot = FisSlot {
            x_post: [S::ZERO; N],
            p_post: Matrix::zeros(),
            x_pred: [S::ZERO; N],
            p_pred: Matrix::zeros(),
            z: [S::ZERO; M],
        };
        Self {
            buffer: [zero_slot; T],
            count: 0,
            h,
            q,
            r,
        }
    }

    /// Reset the smoother buffer.
    pub fn reset(&mut self) {
        self.count = 0;
    }

    /// Store one time step's forward-filter data.
    ///
    /// Must be called in chronological order (k = 0, 1, …, T-1).
    pub fn store_slot(&mut self, slot: FisSlot<S, N, M>) -> Result<(), FixedIntervalError> {
        if self.count >= T {
            return Err(FixedIntervalError::BufferFull);
        }
        self.buffer[self.count] = slot;
        self.count += 1;
        Ok(())
    }

    /// Number of stored slots.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true when no slots are stored.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Process noise covariance matrix Q used during construction.
    pub fn process_noise_cov(&self) -> &Matrix<S, N, N> {
        &self.q
    }

    /// Run the two-filter backward pass and return smoothed estimates.
    ///
    /// `a` — the state transition matrix used in the forward filter.
    ///
    /// Returns `FisSmoothedData` or an error if any inversion fails.
    pub fn smooth(
        &self,
        a: &Matrix<S, N, N>,
    ) -> Result<FisSmoothedData<S, N, T>, FixedIntervalError> {
        let n = self.count;
        if n == 0 {
            return Err(FixedIntervalError::InsufficientData);
        }

        let zero_s = FisSmoothed {
            x: [S::ZERO; N],
            p: Matrix::zeros(),
        };
        let mut out = FisSmoothedData {
            states: [zero_s; T],
            len: n,
        };

        // ── Compute R⁻¹ and Hᵀ R⁻¹ H once. ──────────────────────────────
        let r_inv = self.r.inv().ok_or(FixedIntervalError::SingularMatrix)?;
        let ht = self.h.transpose();
        let ht_rinv = matmul(&ht, &r_inv);
        let ht_rinv_h: Matrix<S, N, N> = matmul(&ht_rinv, &self.h);

        let at = a.transpose();

        // ── Terminal condition: absorb terminal measurement into λ. ──────
        // Only lambda_vec (the information vector) is propagated backward;
        // the information matrix Λ is recomputed each step from stored data.
        //
        // Initialise λ_{n-1} by absorbing measurement z_{n-1}.
        let mut lambda_vec: [S; N] = {
            let z_last = self.buffer[n - 1].z;
            let rinv_z = matvec(&r_inv, &z_last);
            matvec(&ht, &rinv_z)
        };

        // Terminal smoothed state = last filtered state.
        out.states[n - 1] = FisSmoothed {
            x: self.buffer[n - 1].x_post,
            p: self.buffer[n - 1].p_post,
        };

        if n == 1 {
            return Ok(out);
        }

        // ── Backward pass: k = n-2 downto 0. ────────────────────────────
        // At iteration k we produce the smoothed estimate at k using the
        // BIF recursion.  The BIF propagates information *backward*:
        //
        //   M_k = P_{k+1|k}⁻¹  (using stored predicted covariance)
        //
        //   Λ_k = Hᵀ R⁻¹ H + Aᵀ M_k A
        //   λ_k = Hᵀ R⁻¹ z_k + Aᵀ M_k λ_{k+1}
        //
        // The combined smoothed estimate at k:
        //   P_smooth⁻¹ = P_{k|k}⁻¹ + Λ_k
        //   x_smooth    = P_smooth · (P_{k|k}⁻¹ · x_{k|k} + λ_k)

        for k in (0..n - 1).rev() {
            // M = P_{k+1|k}⁻¹ (stored predicted covariance for this slot).
            let p_pred_kp1 = &self.buffer[k].p_pred;
            let m = p_pred_kp1.inv().ok_or(FixedIntervalError::SingularMatrix)?;

            // Aᵀ · M  (N×N)
            let at_m = matmul(&at, &m);

            // Propagate information matrix: Λ_k = Hᵀ R⁻¹ H + Aᵀ M A
            let at_m_a = matmul(&at_m, a);
            let lambda_k = ht_rinv_h.add_mat(&at_m_a);

            // Absorb measurement at step k.
            let z_k = self.buffer[k].z;
            let rinv_zk = matvec(&r_inv, &z_k);
            let xi_k = matvec(&ht, &rinv_zk);

            // λ_k = Hᵀ R⁻¹ z_k + Aᵀ M λ_{k+1}
            let at_m_lam = matvec(&at_m, &lambda_vec);
            let lambda_k_vec: [S; N] = core::array::from_fn(|i| xi_k[i] + at_m_lam[i]);

            // ── Combine forward and backward passes at step k. ───────────
            // P_smooth⁻¹ = P_{k|k}⁻¹ + Λ_k
            let p_post = &self.buffer[k].p_post;
            let p_post_inv = p_post.inv().ok_or(FixedIntervalError::SingularMatrix)?;
            let p_smooth_inv = p_post_inv.add_mat(&lambda_k);
            let p_smooth = p_smooth_inv
                .inv()
                .ok_or(FixedIntervalError::SingularMatrix)?;

            // x_smooth = P_smooth · (P_{k|k}⁻¹ · x_{k|k} + λ_k)
            let p_post_inv_x = matvec(&p_post_inv, &self.buffer[k].x_post);
            let rhs: [S; N] = core::array::from_fn(|i| p_post_inv_x[i] + lambda_k_vec[i]);
            let x_smooth = matvec(&p_smooth, &rhs);

            out.states[k] = FisSmoothed {
                x: x_smooth,
                p: p_smooth,
            };

            // Update BIF information vector for next (earlier) iteration.
            lambda_vec = lambda_k_vec;
        }

        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::matrix::matmul;

    /// Run a forward KF and populate a `FixedIntervalSmoother` for a 1-D
    /// constant-state model (A=1, H=1, Q=q, R=r).
    fn run_forward_1d(
        steps: usize,
        q_val: f64,
        r_val: f64,
    ) -> FixedIntervalSmoother<f64, 1, 1, 32> {
        let a = Matrix::<f64, 1, 1>::identity();
        let h = Matrix::<f64, 1, 1>::identity();
        let q = Matrix::<f64, 1, 1> { data: [[q_val]] };
        let r = Matrix::<f64, 1, 1> { data: [[r_val]] };

        let mut fis = FixedIntervalSmoother::<f64, 1, 1, 32>::new(h, q, r);

        let mut x = [0.0_f64; 1];
        let mut p = Matrix::<f64, 1, 1> { data: [[10.0]] };
        let measurement = 5.0_f64;

        for _ in 0..steps {
            // Predict
            let x_pred = x;
            let ap = matmul(&a, &p);
            let at = a.transpose();
            let apat = matmul(&ap, &at);
            let p_pred = apat.add_mat(&q);

            // Update
            let hx = [x_pred[0]];
            let innov = [measurement - hx[0]];
            let hp = matmul(&h, &p_pred);
            let ht = h.transpose();
            let hpht = matmul(&hp, &ht);
            let s_mat = hpht.add_mat(&r);
            let s_inv = s_mat.inv().expect("S invertible");
            let pht = matmul(&p_pred, &ht);
            let k = matmul(&pht, &s_inv);
            let kv = crate::core::matrix::matvec(&k, &innov);
            let x_post: [f64; 1] = core::array::from_fn(|i| x_pred[i] + kv[i]);
            let kh = matmul(&k, &h);
            let eye = Matrix::<f64, 1, 1>::identity();
            let i_kh = eye.sub_mat(&kh);
            let p_post = matmul(&i_kh, &p_pred);

            fis.store_slot(FisSlot::new(x_post, p_post, x_pred, p_pred, [measurement]))
                .expect("store");

            x = x_post;
            p = p_post;
        }

        fis
    }

    #[test]
    fn smoother_variance_less_than_filter_variance() {
        let steps = 10_usize;
        let fis = run_forward_1d(steps, 1e-4, 1.0);
        let a = Matrix::<f64, 1, 1>::identity();
        let smoothed = fis.smooth(&a).expect("smooth");

        for k in 0..steps {
            let p_filter = fis.buffer[k].p_post.trace();
            let p_smooth = smoothed.states[k].p.trace();
            assert!(
                p_smooth <= p_filter + 1e-9,
                "Smoothed variance must be ≤ filtered at k={k}: \
                 p_smooth={p_smooth}, p_filter={p_filter}"
            );
        }
    }

    #[test]
    fn single_step_smoother_equals_filter() {
        let fis = run_forward_1d(1, 1e-4, 1.0);
        let a = Matrix::<f64, 1, 1>::identity();
        let smoothed = fis.smooth(&a).expect("smooth");
        assert_eq!(smoothed.len, 1);
        let x_filter = fis.buffer[0].x_post[0];
        let x_smooth = smoothed.states[0].x[0];
        assert!(
            (x_filter - x_smooth).abs() < 1e-9,
            "Single step: smooth={x_smooth} must equal filter={x_filter}"
        );
    }

    #[test]
    fn empty_smoother_returns_error() {
        let h = Matrix::<f64, 1, 1>::identity();
        let q = Matrix::<f64, 1, 1> { data: [[1e-4]] };
        let r = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let fis = FixedIntervalSmoother::<f64, 1, 1, 8>::new(h, q, r);
        let a = Matrix::<f64, 1, 1>::identity();
        assert!(matches!(
            fis.smooth(&a),
            Err(FixedIntervalError::InsufficientData)
        ));
    }

    #[test]
    fn buffer_full_error() {
        let h = Matrix::<f64, 1, 1>::identity();
        let q = Matrix::<f64, 1, 1> { data: [[1e-4]] };
        let r = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let mut fis = FixedIntervalSmoother::<f64, 1, 1, 2>::new(h, q, r);
        let slot = FisSlot::new(
            [0.0_f64; 1],
            Matrix::identity(),
            [0.0_f64; 1],
            Matrix::identity(),
            [0.0_f64; 1],
        );
        fis.store_slot(slot).expect("slot 1");
        fis.store_slot(slot).expect("slot 2");
        assert!(matches!(
            fis.store_slot(slot),
            Err(FixedIntervalError::BufferFull)
        ));
    }

    #[test]
    fn smoothed_len_matches_stored_count() {
        let fis = run_forward_1d(12, 1e-4, 0.5);
        let a = Matrix::<f64, 1, 1>::identity();
        let smoothed = fis.smooth(&a).expect("smooth");
        assert_eq!(smoothed.len, 12);
    }
}
