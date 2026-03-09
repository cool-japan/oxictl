//! Parity Space Fault Detection & Isolation
//!
//! Detects faults using model-based residuals. For a discrete system
//! x[k+1] = A*x[k] + B*u[k], y[k] = C*x[k], a one-step predictor computes
//! ŷ[k] = C*x̂[k], and the residual r[k] = y[k] - ŷ[k] is monitored.
//! Under no fault, ||r[k]|| ≈ 0; under a fault, ||r[k]|| grows detectably.
//! A consecutive-fault counter prevents false alarms from transient noise.

#![cfg_attr(not(feature = "std"), no_std)]

use crate::core::scalar::ControlScalar;

/// Errors returned by the FDI subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdiError {
    /// A parameter value is out of its valid range.
    InvalidParameter,
    /// Dimension mismatch between matrices or vectors.
    DimensionMismatch,
}

/// Status returned after each FDI update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultStatus {
    /// No fault declared — residual within normal bounds.
    Normal,
    /// Fault declared — residual exceeded threshold for `consecutive_needed` steps.
    FaultDetected,
}

/// Parity-space fault detector using an open-loop model predictor.
///
/// # Type Parameters
/// * `S`     — scalar type implementing [`ControlScalar`] (f32 or f64).
/// * `N`     — state dimension.
/// * `M`     — output (measurement) dimension.
/// * `I`     — input dimension.
///
/// The detector propagates a state estimate with the nominal model and compares
/// predicted outputs against measured outputs. Faults manifest as persistent
/// non-zero residuals.
#[derive(Debug, Clone)]
pub struct ParitySpaceDetector<S: ControlScalar, const N: usize, const M: usize, const I: usize> {
    /// System matrix A (N×N).
    a: [[S; N]; N],
    /// Input matrix B (N×I).
    b: [[S; N]; I],
    /// Output matrix C (M×N).
    c: [[S; M]; N],
    /// Open-loop state estimate x̂.
    x_hat: [S; N],
    /// Residual norm threshold for fault declaration.
    threshold: S,
    /// Number of consecutive threshold-exceeding steps before declaring a fault.
    consecutive_needed: usize,
    /// Current consecutive fault-exceeding counter.
    fault_count: usize,
    /// Residual norm from the most recent update.
    last_residual_norm: S,
}

impl<S: ControlScalar, const N: usize, const M: usize, const I: usize>
    ParitySpaceDetector<S, N, M, I>
{
    /// Construct a new [`ParitySpaceDetector`].
    ///
    /// # Arguments
    /// * `a` — discrete system matrix A (N×N).
    /// * `b` — input matrix B (N×I).
    /// * `c` — output matrix C (M×N).
    /// * `threshold` — positive residual-norm threshold for fault detection.
    /// * `consecutive_needed` — how many consecutive threshold exceedances
    ///   are required before a fault is declared (≥ 1).
    ///
    /// # Errors
    /// Returns [`FdiError::InvalidParameter`] if `threshold ≤ 0` or
    /// `consecutive_needed == 0`.
    pub fn new(
        a: [[S; N]; N],
        b: [[S; N]; I],
        c: [[S; M]; N],
        threshold: S,
        consecutive_needed: usize,
    ) -> Result<Self, FdiError> {
        if threshold <= S::ZERO || consecutive_needed == 0 {
            return Err(FdiError::InvalidParameter);
        }
        Ok(Self {
            a,
            b,
            c,
            x_hat: [S::ZERO; N],
            threshold,
            consecutive_needed,
            fault_count: 0,
            last_residual_norm: S::ZERO,
        })
    }

    /// Process one measurement sample.
    ///
    /// Computes ŷ = C·x̂, residual r = y − ŷ, then propagates
    /// x̂ ← A·x̂ + B·u.  Returns [`FaultStatus::FaultDetected`] when
    /// `fault_count ≥ consecutive_needed`.
    ///
    /// # Arguments
    /// * `u` — current input vector (length I).
    /// * `y` — current measurement vector (length M).
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, u: &[S; I], y: &[S; M]) -> Result<FaultStatus, FdiError> {
        // Predicted output: ŷ[i] = Σ_j C[i][j] · x̂[j]
        let mut y_hat = [S::ZERO; M];
        for i in 0..M {
            for j in 0..N {
                y_hat[i] += self.c[j][i] * self.x_hat[j];
            }
        }

        // Residual r[i] = y[i] − ŷ[i], then ||r||
        let mut norm_sq = S::ZERO;
        for i in 0..M {
            let ri = y[i] - y_hat[i];
            norm_sq += ri * ri;
        }
        let norm = S::from_f64(libm::sqrt(norm_sq.to_f64()));
        self.last_residual_norm = norm;

        // Propagate state: x̂_new[j] = Σ_k A[j][k]·x̂[k] + Σ_k B[j][k]·u[k]
        let mut x_new = [S::ZERO; N];
        for j in 0..N {
            for k in 0..N {
                x_new[j] += self.a[j][k] * self.x_hat[k];
            }
            for k in 0..I {
                x_new[j] += self.b[k][j] * u[k];
            }
        }
        self.x_hat = x_new;

        // Update consecutive fault counter
        if norm > self.threshold {
            self.fault_count += 1;
        } else {
            self.fault_count = 0;
        }

        if self.fault_count >= self.consecutive_needed {
            Ok(FaultStatus::FaultDetected)
        } else {
            Ok(FaultStatus::Normal)
        }
    }

    /// Return the residual norm computed in the most recent [`update`] call.
    pub fn residual_norm(&self) -> S {
        self.last_residual_norm
    }

    /// Reset the detector state.
    ///
    /// Sets the internal state estimate to `x_init` and clears the fault counter
    /// and last residual norm.
    pub fn reset(&mut self, x_init: [S; N]) {
        self.x_hat = x_init;
        self.fault_count = 0;
        self.last_residual_norm = S::ZERO;
    }

    /// Current consecutive fault-exceeding count.
    pub fn fault_count(&self) -> usize {
        self.fault_count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: identity system A=I, C=I, B=0, scalar (N=1, M=1, I=1)
    fn make_scalar_detector(threshold: f64, consec: usize) -> ParitySpaceDetector<f64, 1, 1, 1> {
        let a = [[1.0_f64]];
        let b = [[0.0_f64]];
        let c = [[1.0_f64]];
        ParitySpaceDetector::new(a, b, c, threshold, consec).expect("valid params")
    }

    #[test]
    fn zero_residual_perfect_model() {
        // A=I, C=I: x̂ starts at 0, y[k]=0 → residual=0 always → Normal
        let mut det = make_scalar_detector(0.1, 1);
        for _ in 0..10 {
            let status = det.update(&[0.0], &[0.0]).expect("update ok");
            assert_eq!(status, FaultStatus::Normal);
            assert!(det.residual_norm() < 1e-12);
        }
    }

    #[test]
    fn step_fault_detected_after_consecutive() {
        // threshold=0.5, need 3 consecutive
        let mut det = make_scalar_detector(0.5, 3);
        // Inject large fault (y=10, x_hat=0 → residual=10)
        let s1 = det.update(&[0.0], &[10.0]).expect("update ok");
        assert_eq!(s1, FaultStatus::Normal); // count=1, need 3
        let s2 = det.update(&[0.0], &[10.0]).expect("update ok");
        assert_eq!(s2, FaultStatus::Normal); // count=2
        let s3 = det.update(&[0.0], &[10.0]).expect("update ok");
        assert_eq!(s3, FaultStatus::FaultDetected); // count=3 ≥ 3
    }

    #[test]
    fn fault_count_resets_on_normal() {
        let mut det = make_scalar_detector(0.5, 3);
        // Two fault steps, then normal
        det.update(&[0.0], &[10.0]).expect("ok");
        det.update(&[0.0], &[10.0]).expect("ok");
        assert_eq!(det.fault_count(), 2);
        // Normal measurement → count resets
        det.update(&[0.0], &[0.0]).expect("ok");
        assert_eq!(det.fault_count(), 0);
        // One more fault → count=1
        det.update(&[0.0], &[10.0]).expect("ok");
        assert_eq!(det.fault_count(), 1);
    }

    #[test]
    fn reset_clears_state() {
        let mut det = make_scalar_detector(0.5, 1);
        // Trigger fault
        det.update(&[0.0], &[10.0]).expect("ok");
        assert_eq!(det.fault_count(), 1);
        // Reset
        det.reset([0.0]);
        assert_eq!(det.fault_count(), 0);
        assert_eq!(det.residual_norm(), 0.0);
        // After reset, same fault step again counts from 1
        let status = det.update(&[0.0], &[0.0]).expect("ok");
        assert_eq!(status, FaultStatus::Normal);
    }

    #[test]
    fn threshold_boundary() {
        // threshold = 1.0, residual exactly 1.0 → NOT exceeding (norm > threshold is false)
        let mut det = make_scalar_detector(1.0, 1);
        let status = det.update(&[0.0], &[1.0]).expect("ok"); // residual = 1.0, not > 1.0
        assert_eq!(status, FaultStatus::Normal);
        // residual 1.001 > 1.0 → fault count increments
        let mut det2 = make_scalar_detector(1.0, 1);
        let status2 = det2.update(&[0.0], &[1.001]).expect("ok");
        assert_eq!(status2, FaultStatus::FaultDetected);
    }

    #[test]
    fn multiple_outputs_residual_norm() {
        // 2D output (N=2, M=2, I=1): A=I2, C=I2, B=zero (1 input → 2 states)
        // b has shape [[S; N]; I] = [[f64; 2]; 1]
        let a = [[1.0_f64, 0.0], [0.0, 1.0]];
        let b: [[f64; 2]; 1] = [[0.0_f64, 0.0]];
        let c = [[1.0_f64, 0.0], [0.0, 1.0]];
        let mut det: ParitySpaceDetector<f64, 2, 2, 1> =
            ParitySpaceDetector::new(a, b, c, 0.5, 1).expect("ok");
        // y=[3,4], x_hat=0 → r=[3,4], ||r||=5
        let status = det.update(&[0.0], &[3.0, 4.0]).expect("ok");
        assert_eq!(status, FaultStatus::FaultDetected);
        let norm = det.residual_norm();
        assert!((norm - 5.0).abs() < 1e-9, "expected norm=5, got {norm}");
    }

    #[test]
    fn invalid_parameter_rejection() {
        // threshold = 0 → error
        let a = [[1.0_f64]];
        let b = [[0.0_f64]];
        let c = [[1.0_f64]];
        assert!(
            ParitySpaceDetector::<f64, 1, 1, 1>::new(a, b, c, 0.0, 1).is_err(),
            "zero threshold should fail"
        );
        // consecutive_needed = 0 → error
        assert!(
            ParitySpaceDetector::<f64, 1, 1, 1>::new(a, b, c, 1.0, 0).is_err(),
            "zero consecutive_needed should fail"
        );
    }

    #[test]
    fn noise_free_prediction_with_input() {
        // x[k+1] = x[k] + u[k], y = x, x_hat starts 0
        // Feed u=1 each step → x_hat grows as 0,1,2,...
        // Feed y matching true trajectory → residual=0
        let a = [[1.0_f64]];
        let b = [[1.0_f64]]; // b[input_idx][state_idx]
        let c = [[1.0_f64]];
        let mut det: ParitySpaceDetector<f64, 1, 1, 1> =
            ParitySpaceDetector::new(a, b, c, 0.5, 1).expect("ok");
        let mut x_true = 0.0_f64;
        for _ in 0..5 {
            let u = 1.0_f64;
            let y = x_true;
            let status = det.update(&[u], &[y]).expect("ok");
            assert_eq!(status, FaultStatus::Normal, "residual should be zero");
            x_true += u;
        }
    }
}
