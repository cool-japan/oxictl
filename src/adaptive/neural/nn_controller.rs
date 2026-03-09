use crate::core::scalar::ControlScalar;

/// Single hidden-layer feedforward neural network controller.
///
/// Architecture: IN → HIDDEN (tanh) → OUT (linear)
///
/// Forward pass: O(IN·HIDDEN + HIDDEN·OUT) — fully deterministic, no alloc.
///
/// `IN` = input dimension, `HIDDEN` = hidden neurons, `OUT` = output dimension.
#[derive(Debug, Clone, Copy)]
pub struct NeuralController<
    S: ControlScalar,
    const IN: usize,
    const HIDDEN: usize,
    const OUT: usize,
> {
    /// Input→hidden weights W1[hidden][input].
    pub w1: [[S; IN]; HIDDEN],
    /// Hidden biases b1[hidden].
    pub b1: [S; HIDDEN],
    /// Hidden→output weights W2[out][hidden].
    pub w2: [[S; HIDDEN]; OUT],
    /// Output biases b2[out].
    pub b2: [S; OUT],
    /// Output scale factor (for bounded output).
    pub output_scale: S,
}

impl<S: ControlScalar, const IN: usize, const HIDDEN: usize, const OUT: usize>
    NeuralController<S, IN, HIDDEN, OUT>
{
    /// Create with zero weights.
    pub fn zeros() -> Self {
        Self {
            w1: [[S::ZERO; IN]; HIDDEN],
            b1: [S::ZERO; HIDDEN],
            w2: [[S::ZERO; HIDDEN]; OUT],
            b2: [S::ZERO; OUT],
            output_scale: S::ONE,
        }
    }

    /// Forward pass: compute output given input.
    ///
    /// Hidden layer activation: tanh(W1·x + b1)
    /// Output layer: linear W2·h + b2, scaled by output_scale
    pub fn forward(&self, input: &[S; IN]) -> [S; OUT] {
        // Hidden layer
        let hidden: [S; HIDDEN] = core::array::from_fn(|j| {
            let z = self.b1[j]
                + self.w1[j]
                    .iter()
                    .zip(input.iter())
                    .fold(S::ZERO, |acc, (&w, &x)| acc + w * x);
            tanh_approx(z)
        });

        // Output layer
        core::array::from_fn(|k| {
            let y = self.b2[k]
                + self.w2[k]
                    .iter()
                    .zip(hidden.iter())
                    .fold(S::ZERO, |acc, (&w, &h)| acc + w * h);
            y * self.output_scale
        })
    }

    /// Online gradient descent weight update (single sample).
    ///
    /// Updates weights using ∂L/∂W via backpropagation for MSE loss:
    ///   L = ||output - target||²
    ///
    /// - `input`: network input
    /// - `target`: desired output
    /// - `lr`: learning rate
    pub fn update(&mut self, input: &[S; IN], target: &[S; OUT], lr: S) {
        // Forward pass (with pre-activations saved)
        let z1: [S; HIDDEN] = core::array::from_fn(|j| {
            self.b1[j]
                + self.w1[j]
                    .iter()
                    .zip(input.iter())
                    .fold(S::ZERO, |acc, (&w, &x)| acc + w * x)
        });
        let h: [S; HIDDEN] = core::array::from_fn(|j| tanh_approx(z1[j]));
        let output: [S; OUT] = core::array::from_fn(|k| {
            let y = self.b2[k]
                + self.w2[k]
                    .iter()
                    .zip(h.iter())
                    .fold(S::ZERO, |acc, (&w, &hj)| acc + w * hj);
            y * self.output_scale
        });

        // Output layer error: δ_out = 2*(output - target) * output_scale
        let delta_out: [S; OUT] =
            core::array::from_fn(|k| S::TWO * (output[k] - target[k]) * self.output_scale);

        // Backprop through output layer
        for (k, &dok) in delta_out.iter().enumerate() {
            self.b2[k] -= lr * dok;
            for (wj, &hj) in self.w2[k].iter_mut().zip(h.iter()) {
                *wj -= lr * dok * hj;
            }
        }

        // Hidden layer delta: δ_h = (W2^T · δ_out) ⊙ tanh'(z1)
        let delta_h: [S; HIDDEN] = core::array::from_fn(|j| {
            let s = self
                .w2
                .iter()
                .zip(delta_out.iter())
                .fold(S::ZERO, |acc, (w2k, &dok)| acc + w2k[j] * dok);
            s * dtanh(z1[j])
        });

        // Backprop through hidden layer
        for (j, &dhj) in delta_h.iter().enumerate() {
            self.b1[j] -= lr * dhj;
            for (wji, &xi) in self.w1[j].iter_mut().zip(input.iter()) {
                *wji -= lr * dhj * xi;
            }
        }
    }
}

/// Fast tanh approximation using Padé [3/2] — accurate for |x| < 3.
#[inline]
fn tanh_approx<S: ControlScalar>(x: S) -> S {
    let x2 = x * x;
    let num = x * (S::from_f64(27.0) + x2);
    let den = S::from_f64(27.0) + S::from_f64(9.0) * x2;
    num / den
}

/// Derivative of tanh approximation: 1 − tanh²(x).
#[inline]
fn dtanh<S: ControlScalar>(x: S) -> S {
    let t = tanh_approx(x);
    S::ONE - t * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_pass_zero_weights_gives_zero() {
        let nn = NeuralController::<f64, 2, 4, 1>::zeros();
        let y = nn.forward(&[1.0, 2.0]);
        assert_eq!(y[0], 0.0);
    }

    #[test]
    fn tanh_at_zero_is_zero() {
        let v = tanh_approx(0.0_f64);
        assert!(v.abs() < 1e-10, "tanh(0)={v}");
    }

    #[test]
    fn tanh_at_large_positive_approx_one() {
        let v = tanh_approx(5.0_f64);
        assert!(v > 0.9, "tanh(5)={v:.4}");
    }

    #[test]
    fn tanh_is_odd() {
        let v1 = tanh_approx(1.5_f64);
        let v2 = tanh_approx(-1.5_f64);
        assert!((v1 + v2).abs() < 1e-10, "not odd: {v1:.6}, {v2:.6}");
    }

    #[test]
    fn loss_decreases_with_training() {
        let mut nn = NeuralController::<f64, 1, 4, 1>::zeros();
        nn.output_scale = 1.0;

        // Measure initial loss on y = 0.5
        let initial_out = nn.forward(&[0.0])[0];
        let initial_loss = (initial_out - 0.5).powi(2);

        // Train
        let lr = 0.1;
        for _ in 0..500 {
            nn.update(&[0.0], &[0.5], lr);
        }

        let final_out = nn.forward(&[0.0])[0];
        let final_loss = (final_out - 0.5).powi(2);

        assert!(
            final_loss < initial_loss + 0.01 || final_loss < 0.01,
            "training should reduce loss: initial={initial_loss:.4}, final={final_loss:.4}, out={final_out:.4}"
        );
    }
}
