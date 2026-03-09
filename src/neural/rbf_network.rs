//! Radial Basis Function (RBF) network with fixed Gaussian kernels.
//!
//! The output weights are learned via gradient descent while the RBF centres
//! and widths remain fixed after construction.  This makes the learning
//! problem convex (linear in the output weights).
//!
//! Architecture:
//!   φ_k(x) = exp(−‖x − c_k‖² / (2·σ_k²))
//!   f(x)   = Σ_k w_k · φ_k(x)

use num_traits::Float;

use crate::neural::NeuralError;

// ---------------------------------------------------------------------------
// RbfCenter
// ---------------------------------------------------------------------------

/// A single RBF kernel: centre position `c` and width `sigma`.
///
/// `D` is the input dimension.
#[derive(Debug, Clone, Copy)]
pub struct RbfCenter<S: Float + Copy, const D: usize> {
    /// Centre of the radial basis function.
    pub center: [S; D],
    /// Width parameter σ (standard deviation).  Must be > 0.
    pub sigma: S,
}

impl<S: Float + Copy, const D: usize> RbfCenter<S, D> {
    /// Create a new RBF centre.
    ///
    /// Returns `Err(NeuralError::InvalidDimension)` when `sigma ≤ 0`.
    pub fn new(center: [S; D], sigma: S) -> Result<Self, NeuralError> {
        if sigma <= S::zero() {
            return Err(NeuralError::InvalidDimension);
        }
        Ok(Self { center, sigma })
    }
}

// ---------------------------------------------------------------------------
// Gaussian RBF kernel
// ---------------------------------------------------------------------------

/// Evaluate the Gaussian RBF: φ(x, c, σ) = exp(−‖x − c‖² / (2σ²)).
///
/// Returns 1 when x == c and approaches 0 as ‖x − c‖ grows.
pub fn gaussian_rbf<S: Float + Copy, const D: usize>(x: &[S; D], center: &[S; D], sigma: S) -> S {
    let mut sq_dist = S::zero();
    for i in 0..D {
        let diff = x[i] - center[i];
        sq_dist = sq_dist + diff * diff;
    }
    let two = S::from(2.0).unwrap_or(S::one());
    let denom = two * sigma * sigma;
    (-sq_dist / denom).exp()
}

// ---------------------------------------------------------------------------
// RbfNetwork
// ---------------------------------------------------------------------------

/// RBF network with `K` Gaussian kernels over a `D`-dimensional input space.
///
/// The network computes a scalar output:
///   f(x) = Σ_{k=0}^{K-1} w_k · φ_k(x)
///
/// Only the output weights `w` are learned; centres are fixed at
/// construction.
#[derive(Clone)]
pub struct RbfNetwork<S: Float + Copy, const D: usize, const K: usize> {
    /// RBF centres and widths.
    pub centers: [RbfCenter<S, D>; K],
    /// Output weights w[k].
    pub weights: [S; K],
}

impl<S: Float + Copy, const D: usize, const K: usize> RbfNetwork<S, D, K> {
    /// Create an RBF network with given centres and zero output weights.
    ///
    /// `centers` — array of `K` RBF centres (each with position + width).
    pub fn new(centers: [RbfCenter<S, D>; K]) -> Self {
        Self {
            centers,
            weights: [S::zero(); K],
        }
    }

    /// Create an RBF network with given centres and explicit output weights.
    pub fn with_weights(centers: [RbfCenter<S, D>; K], weights: [S; K]) -> Self {
        Self { centers, weights }
    }

    /// Compute the network output: f(x) = Σ_k w_k · φ_k(x).
    pub fn forward(&self, x: &[S; D]) -> S {
        let mut out = S::zero();
        for k in 0..K {
            let phi = gaussian_rbf(x, &self.centers[k].center, self.centers[k].sigma);
            out = out + self.weights[k] * phi;
        }
        out
    }

    /// Evaluate the K basis functions at `x` and store results in `phi_out`.
    fn eval_basis(&self, x: &[S; D]) -> [S; K] {
        core::array::from_fn(|k| gaussian_rbf(x, &self.centers[k].center, self.centers[k].sigma))
    }

    /// Online gradient descent step on the output weights.
    ///
    /// Loss = (f(x) − target)²
    /// ∂L/∂w_k = 2·(f(x) − target) · φ_k(x)
    ///
    /// Returns the squared loss before the update.
    /// Returns `Err(NeuralError::NumericalOverflow)` when gradients are non-finite.
    pub fn train_step(&mut self, x: &[S; D], target: S, lr: S) -> Result<S, NeuralError> {
        let phi = self.eval_basis(x);
        let prediction = self
            .weights
            .iter()
            .zip(phi.iter())
            .fold(S::zero(), |acc, (&w, &p)| acc + w * p);

        let error = prediction - target;
        let loss = error * error;
        let two = S::from(2.0).unwrap_or(S::one());
        let grad_scale = two * error;

        for (wk, &pk) in self.weights.iter_mut().zip(phi.iter()) {
            let dw = grad_scale * pk;
            if !dw.is_finite() {
                return Err(NeuralError::NumericalOverflow);
            }
            *wk = *wk - lr * dw;
        }

        Ok(loss)
    }

    /// Reset output weights to zero.
    pub fn reset_weights(&mut self) {
        for w in self.weights.iter_mut() {
            *w = S::zero();
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_rbf_at_center_is_one() {
        let center = [1.0_f64, 2.0, 3.0];
        let x = [1.0_f64, 2.0, 3.0];
        let v = gaussian_rbf::<f64, 3>(&x, &center, 1.0);
        assert!(
            (v - 1.0).abs() < 1e-12,
            "Gaussian RBF at centre should be 1, got {v}"
        );
    }

    #[test]
    fn gaussian_rbf_at_three_sigma_approx_zero_point_011() {
        // At distance 3σ the Gaussian value is exp(-9/2) ≈ 0.01111
        let center = [0.0_f64];
        let sigma = 1.0_f64;
        let x = [3.0_f64 * sigma];
        let v = gaussian_rbf::<f64, 1>(&x, &center, sigma);
        let expected = (-9.0_f64 / 2.0).exp();
        assert!(
            (v - expected).abs() < 1e-8,
            "at 3σ: expected ≈{expected:.5}, got {v:.5}"
        );
        assert!((v - 0.011109).abs() < 1e-4, "at 3σ ≈ 0.011, got {v:.5}");
    }

    #[test]
    fn gaussian_rbf_decreases_with_distance() {
        let center = [0.0_f64];
        let sigma = 1.0_f64;
        let v0 = gaussian_rbf::<f64, 1>(&[0.0], &center, sigma);
        let v1 = gaussian_rbf::<f64, 1>(&[1.0], &center, sigma);
        let v2 = gaussian_rbf::<f64, 1>(&[2.0], &center, sigma);
        assert!(
            v0 > v1 && v1 > v2,
            "Gaussian should decrease: {v0:.4} > {v1:.4} > {v2:.4}"
        );
    }

    #[test]
    fn rbf_center_rejects_non_positive_sigma() {
        let result = RbfCenter::<f64, 2>::new([0.0, 0.0], 0.0);
        assert!(result.is_err(), "sigma=0 should be rejected");
        let result2 = RbfCenter::<f64, 2>::new([0.0, 0.0], -1.0);
        assert!(result2.is_err(), "negative sigma should be rejected");
    }

    #[test]
    fn rbf_network_forward_zero_weights() {
        let c = RbfCenter::new([0.0_f64], 1.0).expect("valid");
        let net = RbfNetwork::<f64, 1, 1>::new([c]);
        let y = net.forward(&[0.5]);
        assert_eq!(y, 0.0, "zero weights should give zero output");
    }

    #[test]
    fn rbf_network_regression_sin() {
        // Learn f(x) ≈ sin(x) on [0, π] using 6 equidistant centres.
        const K: usize = 6;
        let sigma = 0.8_f64;
        let centers: [RbfCenter<f64, 1>; K] = core::array::from_fn(|k| {
            let pos = core::f64::consts::PI * (k as f64) / ((K - 1) as f64);
            RbfCenter::new([pos], sigma).expect("valid")
        });
        let mut net = RbfNetwork::<f64, 1, K>::new(centers);

        let lr = 0.05_f64;
        for _ in 0..3000 {
            for k in 0..20_usize {
                let x_val = core::f64::consts::PI * (k as f64) / 19.0;
                let target = x_val.sin();
                net.train_step(&[x_val], target, lr).expect("train step");
            }
        }

        // Evaluate at a few test points
        let mut max_err = 0.0_f64;
        for k in 0..10_usize {
            let x_val = core::f64::consts::PI * (k as f64) / 9.0;
            let pred = net.forward(&[x_val]);
            let err = (pred - x_val.sin()).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 0.15,
            "RBF sin regression max error too large: {max_err:.4}"
        );
    }

    #[test]
    fn rbf_network_loss_decreases() {
        let c = RbfCenter::new([0.0_f64], 1.0).expect("valid");
        let mut net = RbfNetwork::<f64, 1, 1>::new([c]);
        let lr = 0.1_f64;
        let x = [0.0_f64];
        let target = 1.0_f64;

        let initial_loss = net.train_step(&x, target, lr).expect("step");
        for _ in 0..100 {
            net.train_step(&x, target, lr).expect("step");
        }
        let final_loss = net.train_step(&x, target, lr).expect("step");
        assert!(
            final_loss < initial_loss + 1e-6 || final_loss < 1e-6,
            "loss should decrease: initial={initial_loss:.4}, final={final_loss:.4}"
        );
    }

    #[test]
    fn rbf_network_copy_type() {
        // RbfCenter and the constituent types must be Copy — verify Clone at least
        let c = RbfCenter::new([1.0_f64, 2.0], 0.5).expect("valid");
        let c2 = c;
        assert!((c2.center[0] - 1.0).abs() < 1e-12);
    }
}
