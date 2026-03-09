//! 2D EKF-SLAM — Extended Kalman Filter Simultaneous Localisation and Mapping.
//!
//! Estimates a vehicle pose `[x, y, theta]` and up to `(STATE_DIM - 3) / 2`
//! landmark positions `[lx_i, ly_i]` simultaneously.
//!
//! # State vector (length `STATE_DIM`)
//! ```text
//! [ x, y, theta,  lx_0, ly_0,  lx_1, ly_1, ...,  lx_{N-1}, ly_{N-1} ]
//!   0  1    2       3     4      5     6           3+2*(N-1)
//! ```
//!
//! # Const generic contract
//! The caller must supply `STATE_DIM = 3 + 2 * N` where `N` is the number of
//! landmarks.  Construction validates this at runtime.
//!
//! # Motion model (Euler, unicycle)
//! ```text
//! x'     = x     + v * cos(theta) * dt
//! y'     = y     + v * sin(theta) * dt
//! theta' = theta + omega * dt
//! ```
//!
//! # Measurement model (range–bearing from vehicle to landmark `i`)
//! ```text
//! dx = lx_i - x,   dy = ly_i - y
//! range   = sqrt(dx^2 + dy^2)
//! bearing = atan2(dy, dx) - theta
//! ```

use crate::core::scalar::ControlScalar;
use crate::navigation::dead_reckoning::NavigationError;

/// 2D EKF-SLAM with `STATE_DIM = 3 + 2 * N` landmarks.
///
/// Use `STATE_DIM = 3` for pose-only (no landmarks). `STATE_DIM` must satisfy
/// `(STATE_DIM - 3) % 2 == 0`.
#[derive(Debug)]
pub struct EkfSlam2D<S: ControlScalar, const STATE_DIM: usize> {
    /// Full state vector.
    state: [S; STATE_DIM],
    /// Full covariance matrix (STATE_DIM × STATE_DIM), row-major.
    covariance: [[S; STATE_DIM]; STATE_DIM],
    /// Flags indicating which landmarks have been initialised from a measurement.
    /// Index `i` corresponds to landmark `i` (offset 3 + 2*i in the state).
    landmark_seen: [bool; STATE_DIM], // length STATE_DIM — we only use first n_landmarks entries
    /// Number of landmarks: `(STATE_DIM - 3) / 2`.
    n_landmarks: usize,
    /// Process noise variance for linear velocity.
    q_v: S,
    /// Process noise variance for angular velocity.
    q_omega: S,
    /// Measurement noise variance for range.
    r_range: S,
    /// Measurement noise variance for bearing.
    r_bearing: S,
    /// Fixed integration time step (s).
    dt: S,
}

impl<S: ControlScalar, const STATE_DIM: usize> EkfSlam2D<S, STATE_DIM> {
    /// Construct a new EKF-SLAM instance.
    ///
    /// # Errors
    /// - [`NavigationError::InvalidParameter`] if `STATE_DIM < 3`, if
    ///   `(STATE_DIM - 3) % 2 != 0`, or if any noise / dt value is non-positive.
    pub fn new(
        q_v: S,
        q_omega: S,
        r_range: S,
        r_bearing: S,
        dt: S,
    ) -> Result<Self, NavigationError> {
        if STATE_DIM < 3 {
            return Err(NavigationError::InvalidParameter);
        }
        if (STATE_DIM - 3) % 2 != 0 {
            return Err(NavigationError::InvalidParameter);
        }
        if q_v <= S::ZERO || q_omega <= S::ZERO {
            return Err(NavigationError::InvalidParameter);
        }
        if r_range <= S::ZERO || r_bearing <= S::ZERO {
            return Err(NavigationError::InvalidParameter);
        }
        if dt <= S::ZERO {
            return Err(NavigationError::InvalidParameter);
        }

        let n_landmarks = (STATE_DIM - 3) / 2;

        // Initialise covariance: large uncertainty for vehicle pose, even
        // larger for unobserved landmarks.
        let mut cov = [[S::ZERO; STATE_DIM]; STATE_DIM];
        let p_vehicle = S::from_f64(0.1);
        let p_landmark = S::from_f64(1e6);
        cov[0][0] = p_vehicle;
        cov[1][1] = p_vehicle;
        cov[2][2] = p_vehicle;
        for i in 0..n_landmarks {
            let base = 3 + 2 * i;
            cov[base][base] = p_landmark;
            cov[base + 1][base + 1] = p_landmark;
        }

        Ok(Self {
            state: [S::ZERO; STATE_DIM],
            covariance: cov,
            landmark_seen: [false; STATE_DIM],
            n_landmarks,
            q_v,
            q_omega,
            r_range,
            r_bearing,
            dt,
        })
    }

    /// EKF prediction step using unicycle motion model.
    ///
    /// Propagates the vehicle pose only; landmark states are unchanged.
    ///
    /// # Parameters
    /// - `v`     — commanded linear velocity (m/s).
    /// - `omega` — commanded angular velocity (rad/s).
    ///
    /// # Errors
    /// Returns [`NavigationError::InvalidMeasurement`] if either value is
    /// non-finite.
    pub fn predict(&mut self, v: S, omega: S) -> Result<(), NavigationError> {
        if !v.is_finite() || !omega.is_finite() {
            return Err(NavigationError::InvalidMeasurement);
        }

        let theta = self.state[2];
        let cos_th = S::from_f64(libm::cos(theta.to_f64()));
        let sin_th = S::from_f64(libm::sin(theta.to_f64()));

        // Propagate vehicle state.
        self.state[0] += v * cos_th * self.dt;
        self.state[1] += v * sin_th * self.dt;
        self.state[2] += omega * self.dt;

        // Build motion Jacobian F (STATE_DIM × STATE_DIM).
        // Only the vehicle 3×3 block has off-diagonal entries.
        // F = I + small perturbation in (0,2) and (1,2) entries.
        let mut f = [[S::ZERO; STATE_DIM]; STATE_DIM];
        for (i, row) in f.iter_mut().enumerate() {
            row[i] = S::ONE;
        }
        // ∂(x') / ∂theta = -v * sin(theta) * dt
        f[0][2] = S::from_f64((-v.to_f64()) * theta.to_f64().sin() * self.dt.to_f64());
        // ∂(y') / ∂theta =  v * cos(theta) * dt
        f[1][2] = S::from_f64(v.to_f64() * theta.to_f64().cos() * self.dt.to_f64());

        // Build process noise Q (STATE_DIM × STATE_DIM).
        // Only the vehicle block is nonzero.
        let mut q = [[S::ZERO; STATE_DIM]; STATE_DIM];
        let dt2 = self.dt * self.dt;
        q[0][0] = self.q_v * dt2;
        q[1][1] = self.q_v * dt2;
        q[2][2] = self.q_omega * dt2;

        // P = F * P * F^T + Q
        self.covariance = Self::mat_triple_product_plus_q(&f, &self.covariance, &q);

        Ok(())
    }

    /// EKF measurement update for a single landmark observation.
    ///
    /// If the landmark has not been seen before, it is initialised from the
    /// current vehicle pose and the measurement.
    ///
    /// # Parameters
    /// - `landmark_id` — zero-based landmark index (0 ≤ id < n_landmarks).
    /// - `range`       — measured range to landmark (m), must be > 0.
    /// - `bearing`     — measured bearing to landmark (rad), world-frame.
    ///
    /// # Errors
    /// - [`NavigationError::InvalidLandmarkId`] if the index is out of range.
    /// - [`NavigationError::InvalidMeasurement`] if range ≤ 0 or values are non-finite.
    /// - [`NavigationError::SingularSystem`] if the innovation covariance is
    ///   not invertible (2×2 matrix with zero determinant).
    pub fn update(
        &mut self,
        landmark_id: usize,
        range: S,
        bearing: S,
    ) -> Result<(), NavigationError> {
        if landmark_id >= self.n_landmarks {
            return Err(NavigationError::InvalidLandmarkId);
        }
        if range <= S::ZERO || !range.is_finite() || !bearing.is_finite() {
            return Err(NavigationError::InvalidMeasurement);
        }

        let lm_base = 3 + 2 * landmark_id;

        // Initialise landmark if not yet seen.
        if !self.landmark_seen[landmark_id] {
            let vx = self.state[0];
            let vy = self.state[1];
            let vth = self.state[2];
            let abs_bearing = bearing + vth;
            let cos_b = S::from_f64(libm::cos(abs_bearing.to_f64()));
            let sin_b = S::from_f64(libm::sin(abs_bearing.to_f64()));
            self.state[lm_base] = vx + range * cos_b;
            self.state[lm_base + 1] = vy + range * sin_b;
            self.landmark_seen[landmark_id] = true;
            // Covariance for this landmark is already set to large value.
            // After initialisation, proceed with the update step immediately.
        }

        // Compute expected measurement from current state.
        let vx = self.state[0];
        let vy = self.state[1];
        let vth = self.state[2];
        let lx = self.state[lm_base];
        let ly = self.state[lm_base + 1];

        let dx = lx - vx;
        let dy = ly - vy;
        let r2 = dx * dx + dy * dy;
        let r = S::from_f64(libm::sqrt(r2.to_f64()));

        if r <= S::from_f64(1e-9) {
            // Vehicle is at the landmark — skip update to avoid division by zero.
            return Ok(());
        }

        let expected_range = r;
        let expected_bearing = S::from_f64(libm::atan2(dy.to_f64(), dx.to_f64())) - vth;

        // Innovation.
        let innov_range = range - expected_range;
        let innov_bearing = Self::wrap_angle(bearing - expected_bearing);

        // Measurement Jacobian H (2 × STATE_DIM).
        // Row 0: ∂range / ∂state
        // Row 1: ∂bearing / ∂state
        let mut h = [[S::ZERO; STATE_DIM]; 2];
        let r2_inv = S::ONE / r2;
        let r_inv = S::ONE / r;

        // Vehicle position partial derivatives for range.
        h[0][0] = -dx * r_inv; // ∂range / ∂x
        h[0][1] = -dy * r_inv; // ∂range / ∂y
                               // Vehicle heading: no direct effect on range.

        // Vehicle position partial derivatives for bearing.
        h[1][0] = dy * r2_inv; // ∂bearing / ∂x
        h[1][1] = -dx * r2_inv; // ∂bearing / ∂y
        h[1][2] = S::from_f64(-1.0); // ∂bearing / ∂theta

        // Landmark partial derivatives.
        h[0][lm_base] = dx * r_inv; // ∂range / ∂lx
        h[0][lm_base + 1] = dy * r_inv; // ∂range / ∂ly
        h[1][lm_base] = -dy * r2_inv; // ∂bearing / ∂lx
        h[1][lm_base + 1] = dx * r2_inv; // ∂bearing / ∂ly

        // Innovation covariance: S_mat = H * P * H^T + R  (2 × 2).
        let hp = Self::mat2xn_times_nxn(&h, &self.covariance);
        let hpht = Self::mat2xn_times_nx2(&hp, &h);

        // Measurement noise R (2 × 2 diagonal).
        let s00 = hpht[0][0] + self.r_range;
        let s01 = hpht[0][1];
        let s10 = hpht[1][0];
        let s11 = hpht[1][1] + self.r_bearing;

        // Invert 2×2 S matrix.
        let det = s00 * s11 - s01 * s10;
        if det.to_f64().abs() < 1e-30 {
            return Err(NavigationError::SingularSystem);
        }
        let det_inv = S::ONE / det;
        let si00 = s11 * det_inv;
        let si01 = -s01 * det_inv;
        let si10 = -s10 * det_inv;
        let si11 = s00 * det_inv;

        // Kalman gain: K = P * H^T * S^-1  (STATE_DIM × 2).
        // P * H^T  (STATE_DIM × 2).
        let pht = Self::matnxn_times_nx2_transposed(&self.covariance, &h);
        // K = pht * S_inv  (STATE_DIM × 2).
        let mut k = [[S::ZERO; 2]; STATE_DIM];
        for (i, ki) in k.iter_mut().enumerate() {
            ki[0] = pht[i][0] * si00 + pht[i][1] * si10;
            ki[1] = pht[i][0] * si01 + pht[i][1] * si11;
        }

        // State update: x += K * innov.
        for (i, si) in self.state.iter_mut().enumerate() {
            *si += k[i][0] * innov_range + k[i][1] * innov_bearing;
        }

        // Covariance update: P = (I - K * H) * P.
        // Compute K*H  (STATE_DIM × STATE_DIM).
        let mut kh = [[S::ZERO; STATE_DIM]; STATE_DIM];
        for (i, khi) in kh.iter_mut().enumerate() {
            for (j, khi_j) in khi.iter_mut().enumerate() {
                *khi_j = k[i][0] * h[0][j] + k[i][1] * h[1][j];
            }
        }
        // Compute (I - KH) * P properly.
        // Indices i, j, k_idx are all required for the conditional diagonal logic.
        #[allow(clippy::needless_range_loop)]
        let mut new_p = [[S::ZERO; STATE_DIM]; STATE_DIM];
        #[allow(clippy::needless_range_loop)]
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                let mut sum = S::ZERO;
                for k_idx in 0..STATE_DIM {
                    let i_kh = if i == k_idx {
                        S::ONE - kh[i][k_idx]
                    } else {
                        -kh[i][k_idx]
                    };
                    sum += i_kh * self.covariance[k_idx][j];
                }
                new_p[i][j] = sum;
            }
        }
        self.covariance = new_p;

        Ok(())
    }

    /// Return the current vehicle pose estimate `[x, y, theta]`.
    #[inline]
    pub fn vehicle_pose(&self) -> [S; 3] {
        [self.state[0], self.state[1], self.state[2]]
    }

    /// Return the estimated position of landmark `id` as `[lx, ly]`.
    ///
    /// # Errors
    /// Returns [`NavigationError::InvalidLandmarkId`] if the index is out of range.
    pub fn landmark(&self, id: usize) -> Result<[S; 2], NavigationError> {
        if id >= self.n_landmarks {
            return Err(NavigationError::InvalidLandmarkId);
        }
        let base = 3 + 2 * id;
        Ok([self.state[base], self.state[base + 1]])
    }

    /// Return a reference to the full covariance matrix (for testing/inspection).
    #[inline]
    pub fn covariance(&self) -> &[[S; STATE_DIM]; STATE_DIM] {
        &self.covariance
    }

    // -----------------------------------------------------------------------
    // Internal matrix helpers (no heap allocation, fixed-size stack arrays).
    // -----------------------------------------------------------------------

    /// Compute `F * P * F^T + Q` where all matrices are `STATE_DIM × STATE_DIM`.
    fn mat_triple_product_plus_q(
        f: &[[S; STATE_DIM]; STATE_DIM],
        p: &[[S; STATE_DIM]; STATE_DIM],
        q: &[[S; STATE_DIM]; STATE_DIM],
    ) -> [[S; STATE_DIM]; STATE_DIM] {
        // fp = F * P.
        let mut fp = [[S::ZERO; STATE_DIM]; STATE_DIM];
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                let mut s = S::ZERO;
                for k in 0..STATE_DIM {
                    s += f[i][k] * p[k][j];
                }
                fp[i][j] = s;
            }
        }
        // result = fp * F^T + Q.
        let mut result = [[S::ZERO; STATE_DIM]; STATE_DIM];
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                let mut s = S::ZERO;
                for k in 0..STATE_DIM {
                    s += fp[i][k] * f[j][k]; // F^T[k][j] = F[j][k]
                }
                result[i][j] = s + q[i][j];
            }
        }
        result
    }

    /// `H` (2 × STATE_DIM) × `P` (STATE_DIM × STATE_DIM) → (2 × STATE_DIM).
    fn mat2xn_times_nxn(
        h: &[[S; STATE_DIM]; 2],
        p: &[[S; STATE_DIM]; STATE_DIM],
    ) -> [[S; STATE_DIM]; 2] {
        let mut result = [[S::ZERO; STATE_DIM]; 2];
        for i in 0..2 {
            for j in 0..STATE_DIM {
                let mut s = S::ZERO;
                for k in 0..STATE_DIM {
                    s += h[i][k] * p[k][j];
                }
                result[i][j] = s;
            }
        }
        result
    }

    /// `A` (2 × STATE_DIM) × `H^T` (STATE_DIM × 2) → (2 × 2).
    fn mat2xn_times_nx2(a: &[[S; STATE_DIM]; 2], h: &[[S; STATE_DIM]; 2]) -> [[S; 2]; 2] {
        let mut result = [[S::ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let mut s = S::ZERO;
                for k in 0..STATE_DIM {
                    s += a[i][k] * h[j][k]; // H^T[k][j] = H[j][k]
                }
                result[i][j] = s;
            }
        }
        result
    }

    /// `P` (STATE_DIM × STATE_DIM) × `H^T` (STATE_DIM × 2) → (STATE_DIM × 2).
    fn matnxn_times_nx2_transposed(
        p: &[[S; STATE_DIM]; STATE_DIM],
        h: &[[S; STATE_DIM]; 2],
    ) -> [[S; 2]; STATE_DIM] {
        let mut result = [[S::ZERO; 2]; STATE_DIM];
        for i in 0..STATE_DIM {
            for j in 0..2 {
                let mut s = S::ZERO;
                for k in 0..STATE_DIM {
                    s += p[i][k] * h[j][k]; // H^T[k][j] = H[j][k]
                }
                result[i][j] = s;
            }
        }
        result
    }

    /// Wrap angle to `(-pi, pi]`.
    fn wrap_angle(a: S) -> S {
        let mut v = a.to_f64();
        use core::f64::consts::PI;
        while v > PI {
            v -= 2.0 * PI;
        }
        while v <= -PI {
            v += 2.0 * PI;
        }
        S::from_f64(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Slam7 = EkfSlam2D<f64, 7>; // 3 + 2*2 = 7, 2 landmarks

    fn make_slam() -> Slam7 {
        EkfSlam2D::<f64, 7>::new(0.01, 0.01, 0.1, 0.05, 0.1).unwrap()
    }

    /// Predict-only: state changes and covariance grows.
    #[test]
    fn predict_changes_pose_and_grows_covariance() {
        let mut slam = make_slam();
        let p_before = slam.covariance[0][0];
        slam.predict(1.0, 0.0).unwrap();
        let pose = slam.vehicle_pose();
        // After 0.1 s at 1 m/s with theta=0, x should increase.
        assert!(
            pose[0] > 0.0,
            "x should increase after predict: {}",
            pose[0]
        );
        let p_after = slam.covariance[0][0];
        assert!(
            p_after > p_before,
            "covariance should grow: {} <= {}",
            p_after,
            p_before
        );
    }

    /// Update initialises a landmark when first observed.
    #[test]
    fn update_initialises_landmark() {
        let mut slam = make_slam();
        // Vehicle at origin, landmark at range=5, bearing=0.
        slam.update(0, 5.0, 0.0).unwrap();
        let lm = slam.landmark(0).unwrap();
        // Expected landmark at (5, 0) relative to origin with theta=0.
        assert!(
            (lm[0] - 5.0).abs() < 1e-6,
            "landmark x: expected ~5.0, got {}",
            lm[0]
        );
        assert!(
            lm[1].abs() < 1e-6,
            "landmark y: expected ~0.0, got {}",
            lm[1]
        );
    }

    /// Two updates on the same landmark reduce position uncertainty.
    #[test]
    fn two_updates_reduce_landmark_uncertainty() {
        let mut slam = make_slam();
        slam.update(0, 5.0, 0.0).unwrap();
        let var0 = slam.covariance[3][3]; // lx_0 variance
        slam.update(0, 5.0, 0.0).unwrap();
        let var1 = slam.covariance[3][3];
        assert!(
            var1 < var0,
            "second update should reduce landmark uncertainty: {} >= {}",
            var1,
            var0
        );
    }

    /// Invalid landmark ID returns error.
    #[test]
    fn invalid_landmark_id_returns_error() {
        let mut slam = make_slam();
        let result = slam.update(10, 5.0, 0.0);
        assert_eq!(result.unwrap_err(), NavigationError::InvalidLandmarkId);
    }

    /// Invalid STATE_DIM (not divisible by 2 after subtracting 3) → error.
    #[test]
    fn invalid_state_dim_returns_error() {
        // STATE_DIM = 4 → (4-3) % 2 = 1 ≠ 0 → invalid.
        let result = EkfSlam2D::<f64, 4>::new(0.01, 0.01, 0.1, 0.05, 0.1);
        assert_eq!(result.unwrap_err(), NavigationError::InvalidParameter);
    }

    /// Known vehicle pose: after many updates landmark estimate converges.
    #[test]
    fn landmark_converges_with_known_vehicle_pose() {
        let mut slam = EkfSlam2D::<f64, 5>::new(1e-6, 1e-6, 0.01, 0.01, 0.01).unwrap();
        // Vehicle stationary at origin. Landmark at (3, 4).
        let true_range = 5.0_f64; // sqrt(9+16)
        let true_bearing = libm::atan2(4.0, 3.0);
        for _ in 0..50 {
            slam.update(0, true_range, true_bearing).unwrap();
        }
        let lm = slam.landmark(0).unwrap();
        assert!(
            (lm[0] - 3.0).abs() < 0.15,
            "lx should be ~3.0, got {}",
            lm[0]
        );
        assert!(
            (lm[1] - 4.0).abs() < 0.15,
            "ly should be ~4.0, got {}",
            lm[1]
        );
    }

    /// Predict with non-finite input returns error.
    #[test]
    fn predict_nan_returns_error() {
        let mut slam = make_slam();
        let result = slam.predict(f64::NAN, 0.0);
        assert_eq!(result.unwrap_err(), NavigationError::InvalidMeasurement);
    }

    /// Negative range returns error.
    #[test]
    fn negative_range_returns_error() {
        let mut slam = make_slam();
        let result = slam.update(0, -1.0, 0.0);
        assert_eq!(result.unwrap_err(), NavigationError::InvalidMeasurement);
    }

    /// Out-of-range landmark query returns error.
    #[test]
    fn landmark_query_out_of_range() {
        let slam = make_slam();
        assert_eq!(
            slam.landmark(5).unwrap_err(),
            NavigationError::InvalidLandmarkId
        );
    }
}
