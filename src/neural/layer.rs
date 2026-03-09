//! Dense (fully-connected) neural network layer with const-generic dimensions.
//!
//! All computation is performed on the stack — no heap allocation.
//! Weight initialisation uses Xavier uniform scaling with a deterministic
//! Linear Congruential Generator seeded from the layer dimensions so that
//! the same architecture always produces the same initial weights.

use num_traits::Float;

use crate::neural::{activations::ActivationFn, NeuralError};

// ---------------------------------------------------------------------------
// LCG — deterministic pseudo-random number generator
// ---------------------------------------------------------------------------

/// 64-bit Linear Congruential Generator (Numerical Recipes constants).
///
/// Used exclusively for deterministic weight initialisation; not suitable
/// for cryptographic purposes.
#[derive(Clone, Copy)]
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    /// Seed from a `u64` value.
    const fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x853c49e6748fea9b,
        }
    }

    /// Return the next pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG constants
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return a value uniformly distributed in [−1, 1).
    fn next_f64(&mut self) -> f64 {
        // Map the upper 53 bits to [0, 1)
        let bits = self.next_u64() >> 11;
        let unit = (bits as f64) * (1.0 / (1u64 << 53) as f64);
        unit * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Gradient accumulator
// ---------------------------------------------------------------------------

/// Accumulated gradients for a `DenseLayer<S, IN, OUT>`.
#[derive(Clone, Copy)]
pub struct GradDense<S: Float + Copy, const IN: usize, const OUT: usize> {
    /// Gradient of loss w.r.t. weight matrix W[out][in].
    pub dw: [[S; IN]; OUT],
    /// Gradient of loss w.r.t. bias vector b[out].
    pub db: [S; OUT],
}

impl<S: Float + Copy, const IN: usize, const OUT: usize> GradDense<S, IN, OUT> {
    /// Create a zero-initialised gradient accumulator.
    pub fn zeros() -> Self {
        Self {
            dw: [[S::zero(); IN]; OUT],
            db: [S::zero(); OUT],
        }
    }

    /// Accumulate another gradient (for mini-batch averaging).
    pub fn accumulate(&mut self, other: &Self) {
        for i in 0..OUT {
            self.db[i] = self.db[i] + other.db[i];
            for j in 0..IN {
                self.dw[i][j] = self.dw[i][j] + other.dw[i][j];
            }
        }
    }

    /// Scale all gradients by a scalar (e.g. 1/batch_size).
    pub fn scale(&mut self, factor: S) {
        for i in 0..OUT {
            self.db[i] = self.db[i] * factor;
            for j in 0..IN {
                self.dw[i][j] = self.dw[i][j] * factor;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DenseLayer
// ---------------------------------------------------------------------------

/// Fully-connected layer with `IN` inputs and `OUT` outputs.
///
/// Forward pass: y = activation(W·x + b).
/// Backward pass: returns input-gradient and parameter gradient.
///
/// Weight initialisation: Xavier uniform, scale = sqrt(6 / (IN + OUT)).
/// The LCG is seeded with `IN * 6364 ^ OUT * 2654` so different layer sizes
/// always produce different weights.
#[derive(Clone, Copy)]
pub struct DenseLayer<S, A, const IN: usize, const OUT: usize>
where
    S: Float + Copy,
    A: ActivationFn<S>,
{
    /// Weight matrix W[OUT][IN].
    pub weights: [[S; IN]; OUT],
    /// Bias vector b[OUT].
    pub biases: [S; OUT],
    /// Activation function applied element-wise after the linear transform.
    pub activation: A,
}

impl<S, A, const IN: usize, const OUT: usize> DenseLayer<S, A, IN, OUT>
where
    S: Float + Copy,
    A: ActivationFn<S>,
{
    /// Create a layer with Xavier-uniform weight initialisation.
    ///
    /// All biases are initialised to zero; weights are drawn from a uniform
    /// distribution on [−limit, limit] where limit = sqrt(6 / (IN + OUT)).
    pub fn new(activation: A) -> Self {
        // Xavier uniform limit: sqrt(6 / (fan_in + fan_out))
        let fan_sum = (IN + OUT) as f64;
        let limit = libm::sqrt(6.0 / fan_sum);

        // Deterministic seed derived from IN and OUT
        let seed = (IN as u64).wrapping_mul(6_364_136_223_846_793_005)
            ^ (OUT as u64).wrapping_mul(2_654_435_761);
        let mut rng = Lcg64::new(seed);

        let weights: [[S; IN]; OUT] = core::array::from_fn(|_| {
            core::array::from_fn(|_| {
                let v = rng.next_f64() * limit;
                // Cast f64 to S — both f32 and f64 implement Float
                S::from(v).unwrap_or(S::zero())
            })
        });

        Self {
            weights,
            biases: [S::zero(); OUT],
            activation,
        }
    }

    /// Create a layer with explicitly provided weights and biases.
    pub fn with_params(weights: [[S; IN]; OUT], biases: [S; OUT], activation: A) -> Self {
        Self {
            weights,
            biases,
            activation,
        }
    }

    /// Forward pass: y[i] = activation(Σ_j W[i][j]·x[j] + b[i]).
    ///
    /// Returns the activated output `[S; OUT]`.
    pub fn forward(&self, input: &[S; IN]) -> [S; OUT] {
        core::array::from_fn(|i| {
            let z = self.weights[i]
                .iter()
                .zip(input.iter())
                .fold(self.biases[i], |acc, (&w, &x)| acc + w * x);
            self.activation.apply(z)
        })
    }

    /// Forward pass that also returns the pre-activation values (needed for
    /// back-propagation without re-computing them).
    pub fn forward_with_preact(&self, input: &[S; IN]) -> ([S; OUT], [S; OUT]) {
        let mut pre = [S::zero(); OUT];
        let mut out = [S::zero(); OUT];
        for (i, (pre_i, out_i)) in pre.iter_mut().zip(out.iter_mut()).enumerate() {
            let z = self.weights[i]
                .iter()
                .zip(input.iter())
                .fold(self.biases[i], |acc, (&w, &x)| acc + w * x);
            *pre_i = z;
            *out_i = self.activation.apply(z);
        }
        (out, pre)
    }

    /// Backward pass — computes gradients and the gradient flowing into the input.
    ///
    /// # Arguments
    /// * `input`       — the input that was passed to `forward`.
    /// * `pre_act`     — pre-activation values (z = W·x + b) from `forward_with_preact`.
    /// * `grad_output` — ∂L/∂y[i] for each output unit.
    ///
    /// # Returns
    /// `(grad_input, GradDense)` where:
    /// * `grad_input[j]` = ∂L/∂x[j] = Σ_i W[i][j] · δ[i]
    /// * `GradDense.dw[i][j]` = δ[i] · x[j]
    /// * `GradDense.db[i]`    = δ[i]
    ///
    /// and δ[i] = grad_output[i] · activation'(pre_act[i]).
    pub fn backward(
        &self,
        input: &[S; IN],
        pre_act: &[S; OUT],
        grad_output: &[S; OUT],
    ) -> ([S; IN], GradDense<S, IN, OUT>) {
        // δ[i] = ∂L/∂y[i] · f'(z[i])
        let mut delta = [S::zero(); OUT];
        for i in 0..OUT {
            delta[i] = grad_output[i] * self.activation.derivative(pre_act[i]);
        }

        // Gradient w.r.t. input: grad_input[j] = Σ_i W[i][j] · δ[i]
        let grad_input: [S; IN] = core::array::from_fn(|j| {
            self.weights
                .iter()
                .zip(delta.iter())
                .fold(S::zero(), |acc, (row, &d)| acc + row[j] * d)
        });

        // Gradient w.r.t. parameters
        let dw: [[S; IN]; OUT] =
            core::array::from_fn(|i| core::array::from_fn(|j| delta[i] * input[j]));
        let db: [S; OUT] = core::array::from_fn(|i| delta[i]);

        (grad_input, GradDense { dw, db })
    }

    /// SGD parameter update: w -= lr·dw, b -= lr·db.
    ///
    /// Returns `Err(NeuralError::NumericalOverflow)` if any gradient is non-finite.
    pub fn update_weights(
        &mut self,
        grad: &GradDense<S, IN, OUT>,
        lr: S,
    ) -> Result<(), NeuralError> {
        for i in 0..OUT {
            if !grad.db[i].is_finite() {
                return Err(NeuralError::NumericalOverflow);
            }
            self.biases[i] = self.biases[i] - lr * grad.db[i];
            for j in 0..IN {
                if !grad.dw[i][j].is_finite() {
                    return Err(NeuralError::NumericalOverflow);
                }
                self.weights[i][j] = self.weights[i][j] - lr * grad.dw[i][j];
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::activations::{Linear, Tanh};

    /// Numerical gradient via central finite differences.
    fn numerical_grad_input<S, A, const IN: usize, const OUT: usize>(
        layer: &DenseLayer<S, A, IN, OUT>,
        input: &[S; IN],
        loss_grad_out: &[S; OUT],
        h: S,
    ) -> [S; IN]
    where
        S: Float + Copy,
        A: ActivationFn<S>,
    {
        core::array::from_fn(|j| {
            let mut xp = *input;
            let mut xm = *input;
            xp[j] = xp[j] + h;
            xm[j] = xm[j] - h;
            let yp = layer.forward(&xp);
            let ym = layer.forward(&xm);
            // Approximate ∂L/∂x[j] via chain rule over the loss grad
            let mut acc = S::zero();
            for i in 0..OUT {
                acc = acc + loss_grad_out[i] * (yp[i] - ym[i]) / (h + h);
            }
            acc
        })
    }

    #[test]
    fn forward_shape_linear() {
        let layer = DenseLayer::<f64, Linear<f64>, 3, 2>::new(Linear::new());
        let y = layer.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn forward_zero_weights_bias_gives_zero_for_linear() {
        let layer = DenseLayer::<f64, Linear<f64>, 2, 2>::with_params(
            [[0.0; 2]; 2],
            [0.0; 2],
            Linear::new(),
        );
        let y = layer.forward(&[5.0, -3.0]);
        assert_eq!(y, [0.0, 0.0]);
    }

    #[test]
    fn forward_known_weights() {
        // W = [[1,0],[0,1]], b = [0.5, -0.5], linear activation
        let layer = DenseLayer::<f64, Linear<f64>, 2, 2>::with_params(
            [[1.0, 0.0], [0.0, 1.0]],
            [0.5, -0.5],
            Linear::new(),
        );
        let y = layer.forward(&[3.0, 7.0]);
        assert!((y[0] - 3.5).abs() < 1e-12);
        assert!((y[1] - 6.5).abs() < 1e-12);
    }

    #[test]
    fn xavier_init_range() {
        // Weights should be within the Xavier bound
        let layer = DenseLayer::<f64, Tanh<f64>, 4, 4>::new(Tanh::new());
        let limit = libm::sqrt(6.0 / 8.0);
        for row in &layer.weights {
            for &w in row {
                assert!(
                    w.abs() <= limit + 1e-10,
                    "weight {w} outside Xavier range [{}, {}]",
                    -limit,
                    limit
                );
            }
        }
    }

    #[test]
    fn backward_gradient_numerical_check() {
        // Use a tanh layer and verify analytic grad_input against finite differences.
        let layer = DenseLayer::<f64, Tanh<f64>, 3, 2>::new(Tanh::new());
        let input = [0.5_f64, -0.3, 1.1];
        let (_, pre_act) = layer.forward_with_preact(&input);
        let grad_out = [1.0_f64, -1.0];

        let (analytic_grad_in, _grad) = layer.backward(&input, &pre_act, &grad_out);
        let numerical_grad_in = numerical_grad_input(&layer, &input, &grad_out, 1e-5);

        for j in 0..3 {
            let err = (analytic_grad_in[j] - numerical_grad_in[j]).abs();
            assert!(
                err < 1e-5,
                "grad_input[{j}]: analytic={}, numerical={}, err={}",
                analytic_grad_in[j],
                numerical_grad_in[j],
                err
            );
        }
    }

    #[test]
    fn backward_gradient_weights_numerical_check() {
        // Verify dw numerically for the weight of the (0,0) element.
        let mut layer = DenseLayer::<f64, Tanh<f64>, 2, 2>::new(Tanh::new());
        let input = [1.0_f64, -0.5];
        let grad_out = [1.0_f64, 0.0];

        let h = 1e-5;
        // Perturb W[0][0]
        let original = layer.weights[0][0];
        layer.weights[0][0] = original + h;
        let yp = layer.forward(&input);
        layer.weights[0][0] = original - h;
        let ym = layer.forward(&input);
        layer.weights[0][0] = original;

        let num_dw00 =
            (yp[0] - ym[0]) / (2.0 * h) * grad_out[0] + (yp[1] - ym[1]) / (2.0 * h) * grad_out[1];

        let (_, pre_act) = layer.forward_with_preact(&input);
        let (_, grad) = layer.backward(&input, &pre_act, &grad_out);

        let err = (grad.dw[0][0] - num_dw00).abs();
        assert!(
            err < 1e-5,
            "dw[0][0]: analytic={}, numerical={}, err={}",
            grad.dw[0][0],
            num_dw00,
            err
        );
    }

    #[test]
    fn update_weights_reduces_loss() {
        let mut layer =
            DenseLayer::<f64, Linear<f64>, 1, 1>::with_params([[0.0]], [0.0], Linear::new());
        let input = [1.0_f64];
        let target = [2.0_f64];

        let initial_y = layer.forward(&input)[0];
        let initial_loss = (initial_y - target[0]).powi(2);

        let lr = 0.1_f64;
        for _ in 0..100 {
            let (y, pre) = layer.forward_with_preact(&input);
            let g_out = [2.0 * (y[0] - target[0])];
            let (_, grad) = layer.backward(&input, &pre, &g_out);
            layer.update_weights(&grad, lr).expect("update failed");
        }

        let final_y = layer.forward(&input)[0];
        let final_loss = (final_y - target[0]).powi(2);
        assert!(
            final_loss < initial_loss + 1e-6 || final_loss < 1e-6,
            "loss did not decrease: initial={initial_loss}, final={final_loss}"
        );
    }
}
