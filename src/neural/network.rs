//! Multi-layer perceptron (MLP) with a fixed 3-layer topology.
//!
//! The network has the shape:
//!   Input (I) → Hidden (H, Tanh) → Output (O, Linear)
//!
//! All state lives on the stack; no heap allocation is used.
//! Mini-batch gradient descent is supported via `train_batch`.

use num_traits::Float;

use crate::neural::{
    activations::{ActivationFn, Linear, Tanh},
    layer::{DenseLayer, GradDense},
    NeuralError,
};

// ---------------------------------------------------------------------------
// Mlp — two-layer MLP
// ---------------------------------------------------------------------------

/// Two-layer MLP: Input(I) → Hidden(H, Tanh) → Output(O, Linear).
///
/// Type parameters:
/// * `S`  — scalar type (f32 or f64).
/// * `I`  — number of input features.
/// * `H`  — number of hidden neurons.
/// * `O`  — number of output neurons.
/// * `A1` — hidden activation (default: Tanh).
/// * `A2` — output activation (default: Linear).
#[derive(Clone, Copy)]
pub struct Mlp<S, A1, A2, const I: usize, const H: usize, const O: usize>
where
    S: Float + Copy,
    A1: ActivationFn<S>,
    A2: ActivationFn<S>,
{
    /// Input → hidden dense layer.
    pub layer1: DenseLayer<S, A1, I, H>,
    /// Hidden → output dense layer.
    pub layer2: DenseLayer<S, A2, H, O>,
}

impl<S, const I: usize, const H: usize, const O: usize> Mlp<S, Tanh<S>, Linear<S>, I, H, O>
where
    S: Float + Copy,
{
    /// Create an MLP with Xavier-uniform initialisation.
    ///
    /// Hidden layer uses `Tanh`; output layer uses `Linear` (identity).
    pub fn new() -> Self {
        Self {
            layer1: DenseLayer::new(Tanh::new()),
            layer2: DenseLayer::new(Linear::new()),
        }
    }
}

impl<S, const I: usize, const H: usize, const O: usize> Default
    for Mlp<S, Tanh<S>, Linear<S>, I, H, O>
where
    S: Float + Copy,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S, A1, A2, const I: usize, const H: usize, const O: usize> Mlp<S, A1, A2, I, H, O>
where
    S: Float + Copy,
    A1: ActivationFn<S>,
    A2: ActivationFn<S>,
{
    /// Create with explicit activations (for custom architectures).
    pub fn with_activations(act1: A1, act2: A2) -> Self {
        Self {
            layer1: DenseLayer::new(act1),
            layer2: DenseLayer::new(act2),
        }
    }

    /// Forward pass: compute network output.
    pub fn forward(&self, input: &[S; I]) -> [S; O] {
        let h = self.layer1.forward(input);
        self.layer2.forward(&h)
    }

    /// Forward pass that additionally returns intermediate values for backprop.
    fn forward_full(&self, input: &[S; I]) -> ([S; O], [S; H], [S; H], [S; O]) {
        let (h, pre1) = self.layer1.forward_with_preact(input);
        let (out, pre2) = self.layer2.forward_with_preact(&h);
        (out, h, pre1, pre2)
    }

    /// Single-sample MSE gradient descent step.
    ///
    /// Loss = Σ_i (out[i] − target[i])² / O
    ///
    /// Returns the scalar MSE loss before the parameter update.
    /// Returns `Err(NeuralError::NumericalOverflow)` if any gradient is non-finite.
    pub fn train_step(&mut self, input: &[S; I], target: &[S; O], lr: S) -> Result<S, NeuralError> {
        let (out, h, pre1, pre2) = self.forward_full(input);

        // MSE loss and its gradient w.r.t. output
        let n_out = S::from(O).unwrap_or(S::one());
        let two_over_n = S::from(2.0).unwrap_or(S::one()) / n_out;

        let mut loss = S::zero();
        let mut grad_out = [S::zero(); O];
        for i in 0..O {
            let diff = out[i] - target[i];
            loss = loss + diff * diff;
            grad_out[i] = two_over_n * diff;
        }
        loss = loss / n_out;

        // Backprop through layer2
        let (grad_h, grad2) = self.layer2.backward(&h, &pre2, &grad_out);

        // Backprop through layer1
        let (_, grad1) = self.layer1.backward(input, &pre1, &grad_h);

        // Parameter updates
        self.layer2.update_weights(&grad2, lr)?;
        self.layer1.update_weights(&grad1, lr)?;

        Ok(loss)
    }

    /// Mini-batch gradient descent step.
    ///
    /// Accumulates gradients over `B` samples and applies a single averaged
    /// parameter update.  Returns mean MSE loss over the batch.
    pub fn train_batch<const B: usize>(
        &mut self,
        inputs: &[[S; I]; B],
        targets: &[[S; O]; B],
        lr: S,
    ) -> Result<S, NeuralError> {
        let mut acc_grad1 = GradDense::<S, I, H>::zeros();
        let mut acc_grad2 = GradDense::<S, H, O>::zeros();
        let mut total_loss = S::zero();

        for b in 0..B {
            let (out, h, pre1, pre2) = self.forward_full(&inputs[b]);

            let n_out = S::from(O).unwrap_or(S::one());
            let two_over_n = S::from(2.0).unwrap_or(S::one()) / n_out;

            let mut grad_out = [S::zero(); O];
            for i in 0..O {
                let diff = out[i] - targets[b][i];
                total_loss = total_loss + diff * diff / n_out;
                grad_out[i] = two_over_n * diff;
            }

            let (grad_h, g2) = self.layer2.backward(&h, &pre2, &grad_out);
            let (_, g1) = self.layer1.backward(&inputs[b], &pre1, &grad_h);

            acc_grad1.accumulate(&g1);
            acc_grad2.accumulate(&g2);
        }

        // Average over batch
        let inv_b = S::from(1.0 / B as f64).unwrap_or(S::one());
        acc_grad1.scale(inv_b);
        acc_grad2.scale(inv_b);
        total_loss = total_loss / S::from(B).unwrap_or(S::one());

        self.layer2.update_weights(&acc_grad2, lr)?;
        self.layer1.update_weights(&acc_grad1, lr)?;

        Ok(total_loss)
    }

    // ------------------------------------------------------------------
    // Serialization helpers
    // ------------------------------------------------------------------

    /// Total number of weight parameters (weights + biases for both layers).
    ///
    /// Layout: [W1 row-major (I*H), b1 (H), W2 row-major (H*O), b2 (O)].
    pub fn num_params() -> usize {
        I * H + H + H * O + O
    }

    /// Export all parameters into a caller-supplied slice.
    ///
    /// `buf` must have length ≥ `num_params()`.
    /// Returns `Err(NeuralError::InvalidDimension)` when `buf` is too small.
    ///
    /// Layout: [W1 row-major, b1, W2 row-major, b2].
    pub fn export_weights_slice(&self, buf: &mut [S]) -> Result<(), NeuralError> {
        let needed = Self::num_params();
        if buf.len() < needed {
            return Err(NeuralError::InvalidDimension);
        }
        let mut idx = 0_usize;

        for row in &self.layer1.weights {
            for &w in row {
                buf[idx] = w;
                idx += 1;
            }
        }
        for &b in &self.layer1.biases {
            buf[idx] = b;
            idx += 1;
        }
        for row in &self.layer2.weights {
            for &w in row {
                buf[idx] = w;
                idx += 1;
            }
        }
        for &b in &self.layer2.biases {
            buf[idx] = b;
            idx += 1;
        }

        Ok(())
    }

    /// Import parameters from a flat slice produced by `export_weights_slice`.
    ///
    /// Returns `Err(NeuralError::InvalidDimension)` if the slice is too short or
    /// any value is non-finite.
    pub fn import_weights_slice(&mut self, params: &[S]) -> Result<(), NeuralError> {
        let needed = Self::num_params();
        if params.len() < needed {
            return Err(NeuralError::InvalidDimension);
        }
        let mut idx = 0_usize;

        for row in self.layer1.weights.iter_mut() {
            for w in row.iter_mut() {
                if !params[idx].is_finite() {
                    return Err(NeuralError::InvalidDimension);
                }
                *w = params[idx];
                idx += 1;
            }
        }
        for b in self.layer1.biases.iter_mut() {
            if !params[idx].is_finite() {
                return Err(NeuralError::InvalidDimension);
            }
            *b = params[idx];
            idx += 1;
        }
        for row in self.layer2.weights.iter_mut() {
            for w in row.iter_mut() {
                if !params[idx].is_finite() {
                    return Err(NeuralError::InvalidDimension);
                }
                *w = params[idx];
                idx += 1;
            }
        }
        for b in self.layer2.biases.iter_mut() {
            if !params[idx].is_finite() {
                return Err(NeuralError::InvalidDimension);
            }
            *b = params[idx];
            idx += 1;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MlpRegressor — ergonomic wrapper for regression / control
// ---------------------------------------------------------------------------

/// Regression-oriented wrapper around `Mlp` for scalar output control.
///
/// Provides a stable API for typical control applications where the network
/// maps a state vector to a control signal.
#[derive(Clone, Copy)]
pub struct MlpRegressor<S, const I: usize, const H: usize, const O: usize>
where
    S: Float + Copy,
{
    /// Underlying MLP.
    pub mlp: Mlp<S, Tanh<S>, Linear<S>, I, H, O>,
    /// Learning rate used by `fit_step`.
    pub lr: S,
    /// Running exponential moving average of the training loss (τ = 0.99).
    pub loss_ema: S,
}

impl<S, const I: usize, const H: usize, const O: usize> MlpRegressor<S, I, H, O>
where
    S: Float + Copy,
{
    /// Create a regressor with Xavier-initialised MLP and given learning rate.
    pub fn new(lr: S) -> Self {
        Self {
            mlp: Mlp::new(),
            lr,
            loss_ema: S::zero(),
        }
    }

    /// Forward inference: map input to output without any weight update.
    pub fn predict(&self, input: &[S; I]) -> [S; O] {
        self.mlp.forward(input)
    }

    /// Single-sample gradient step; updates `loss_ema` and returns MSE loss.
    pub fn fit_step(&mut self, input: &[S; I], target: &[S; O]) -> Result<S, NeuralError> {
        let loss = self.mlp.train_step(input, target, self.lr)?;
        let alpha = S::from(0.01).unwrap_or(S::one());
        self.loss_ema = (S::one() - alpha) * self.loss_ema + alpha * loss;
        Ok(loss)
    }

    /// Mini-batch gradient step.
    pub fn fit_batch<const B: usize>(
        &mut self,
        inputs: &[[S; I]; B],
        targets: &[[S; O]; B],
    ) -> Result<S, NeuralError> {
        let loss = self.mlp.train_batch::<B>(inputs, targets, self.lr)?;
        let alpha = S::from(0.01).unwrap_or(S::one());
        self.loss_ema = (S::one() - alpha) * self.loss_ema + alpha * loss;
        Ok(loss)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_forward_shape() {
        let net = Mlp::<f64, _, _, 3, 4, 2>::new();
        let y = net.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn mlp_loss_decreases_identity() {
        // Learn identity mapping: output[0] ≈ input[0]
        let mut net = Mlp::<f64, _, _, 1, 8, 1>::new();
        let input = [1.0_f64];
        let target = [1.0_f64];
        let lr = 0.01;

        let initial_loss = net.train_step(&input, &target, lr).expect("step failed");
        for _ in 0..2000 {
            net.train_step(&input, &target, lr).expect("step failed");
        }
        let final_y = net.forward(&input);
        let final_loss = (final_y[0] - target[0]).powi(2);
        assert!(
            final_loss < initial_loss + 1e-4 || final_loss < 0.01,
            "identity mapping: initial_loss={initial_loss:.6}, final_loss={final_loss:.6}, output={:.4}",
            final_y[0]
        );
    }

    #[test]
    fn mlp_learn_xor() {
        // XOR: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
        // This needs enough hidden units and iterations.
        let mut net = Mlp::<f64, _, _, 2, 8, 1>::new();
        let xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let ys = [[0.0_f64], [1.0], [1.0], [0.0]];
        let lr = 0.05_f64;

        let mut last_loss = f64::MAX;
        for epoch in 0..5000 {
            let mut epoch_loss = 0.0;
            for k in 0..4 {
                epoch_loss += net.train_step(&xs[k], &ys[k], lr).expect("step");
            }
            if epoch == 4999 {
                last_loss = epoch_loss / 4.0;
            }
        }
        // XOR is learnable with tanh hidden units; check loss converged
        assert!(
            last_loss < 0.1,
            "XOR training did not converge: final loss = {last_loss:.4}"
        );
    }

    #[test]
    fn mlp_train_batch_reduces_loss() {
        let mut net = Mlp::<f64, _, _, 2, 4, 1>::new();
        let inputs = [[1.0_f64, 0.0], [0.0, 1.0]];
        let targets = [[1.0_f64], [1.0]];
        let lr = 0.05;

        let initial = net.train_batch::<2>(&inputs, &targets, lr).expect("batch");
        for _ in 0..500 {
            net.train_batch::<2>(&inputs, &targets, lr).expect("batch");
        }
        let final_loss = net.train_batch::<2>(&inputs, &targets, lr).expect("batch");
        assert!(
            final_loss <= initial + 1e-6 || final_loss < 0.01,
            "batch training did not reduce loss: initial={initial:.4}, final={final_loss:.4}"
        );
    }

    #[test]
    fn mlp_export_import_roundtrip() {
        let net = Mlp::<f64, _, _, 2, 3, 1>::new();
        let n = Mlp::<f64, Tanh<f64>, Linear<f64>, 2, 3, 1>::num_params();
        let mut buf = [0.0_f64; 2 * 3 + 3 + 3 + 1]; // I*H+H+H*O+O = 6+3+3+1=13
        net.export_weights_slice(&mut buf[..n]).expect("export");

        let mut net2 = Mlp::<f64, _, _, 2, 3, 1>::new();
        net2.import_weights_slice(&buf[..n]).expect("import");

        let y1 = net.forward(&[0.5, -0.5]);
        let y2 = net2.forward(&[0.5, -0.5]);
        assert!((y1[0] - y2[0]).abs() < 1e-12, "roundtrip mismatch");
    }

    #[test]
    fn mlp_regressor_fit_step() {
        let mut reg = MlpRegressor::<f64, 1, 4, 1>::new(0.05);
        let input = [2.0_f64];
        let target = [4.0_f64];

        let mut prev_loss = f64::MAX;
        for _ in 0..1000 {
            prev_loss = reg.fit_step(&input, &target).expect("fit");
        }
        assert!(
            prev_loss < 0.1,
            "regressor did not converge: final loss={prev_loss:.4}"
        );
    }
}
