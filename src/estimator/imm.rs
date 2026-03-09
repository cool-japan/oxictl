//! Interacting Multiple Model (IMM) estimator: K-model bank.
//!
//! The IMM estimator maintains a bank of K Kalman filters, each describing
//! a different dynamics hypothesis. Model probabilities are updated
//! from measurement likelihoods and blended via a Markov transition matrix.
//!
//! Reference: Bar-Shalom, Li, Kirubarajan — "Estimation with Applications to
//! Tracking and Navigation" Ch. 11.
#![allow(clippy::needless_range_loop)]

use crate::core::matrix::{matmul, Matrix};
use crate::core::scalar::ControlScalar;

/// Single model in the IMM bank (Kalman filter snapshot).
#[derive(Debug, Clone, Copy)]
pub struct ImmModel<S: ControlScalar, const N: usize, const M: usize> {
    /// State transition matrix (N×N) for this model.
    pub a: Matrix<S, N, N>,
    /// State estimate (N×1).
    pub x_hat: Matrix<S, N, 1>,
    /// Error covariance (N×N).
    pub p: Matrix<S, N, N>,
    /// Process noise covariance (N×N).
    pub q: Matrix<S, N, N>,
    /// Unnormalised likelihood of current measurement under this model.
    pub likelihood: S,
}

/// IMM Estimator with K models.
///
/// N: state dim, M: measurement dim, K: number of models.
pub struct ImmEstimator<S: ControlScalar, const N: usize, const M: usize, const K: usize> {
    /// Bank of K model-specific Kalman filters.
    pub models: [ImmModel<S, N, M>; K],
    /// Markov transition matrix p_{ij} = P(model j | model i) — K×K row-stochastic.
    pub transition: [[S; K]; K],
    /// Model probabilities μ_k (sum to 1).
    pub mu: [S; K],
    /// Measurement matrix C (M×N) shared across models.
    pub c: Matrix<S, M, N>,
    /// Measurement noise covariance R (M×M) shared across models.
    pub r: Matrix<S, M, M>,
}

impl<S: ControlScalar, const N: usize, const M: usize, const K: usize> ImmEstimator<S, N, M, K> {
    /// Create a new IMM estimator.
    ///
    /// `transition[i][j]` = P(mode j | mode i).  Rows must sum to 1.
    /// Initial `mu` should also sum to 1.
    pub fn new(
        models: [ImmModel<S, N, M>; K],
        transition: [[S; K]; K],
        c: Matrix<S, M, N>,
        r: Matrix<S, M, M>,
    ) -> Self {
        let mu: [S; K] = core::array::from_fn(|_| S::ONE / S::from_f64(K as f64));
        Self {
            models,
            transition,
            mu,
            c,
            r,
        }
    }

    /// IMM mixing step: compute mixed initial conditions before model-conditional filtering.
    ///
    /// For each model j: compute mixed mean x_{0j} and covariance P_{0j} from
    /// all model estimates weighted by mixing probabilities μ_{i|j}.
    pub fn mix_estimates(&mut self) {
        // Predicted model probabilities: c_j = sum_i mu_i * p_{ij}
        let mut c_bar: [S; K] = [S::ZERO; K];
        for j in 0..K {
            for i in 0..K {
                c_bar[j] += self.mu[i] * self.transition[i][j];
            }
        }

        // Mixing probabilities: mu_{i|j} = mu_i * p_{ij} / c_bar_j
        let mut mu_ij: [[S; K]; K] = [[S::ZERO; K]; K];
        for j in 0..K {
            if c_bar[j] > S::EPSILON {
                for i in 0..K {
                    mu_ij[i][j] = self.mu[i] * self.transition[i][j] / c_bar[j];
                }
            }
        }

        // Compute mixed means x_{0j} = sum_i mu_{i|j} * x_hat_i
        let mut mixed_x: [Matrix<S, N, 1>; K] =
            core::array::from_fn(|_| Matrix::<S, N, 1>::zeros());
        for j in 0..K {
            for i in 0..K {
                let contrib = self.models[i].x_hat.scale(mu_ij[i][j]);
                mixed_x[j] = mixed_x[j].add_mat(&contrib);
            }
        }

        // Compute mixed covariances P_{0j}
        let mut mixed_p: [Matrix<S, N, N>; K] =
            core::array::from_fn(|_| Matrix::<S, N, N>::zeros());
        for j in 0..K {
            for i in 0..K {
                let w = mu_ij[i][j];
                // Spread: dx = x_hat_i - x_{0j}
                let dx = self.models[i].x_hat.sub_mat(&mixed_x[j]);
                let dxt = dx.transpose();
                let spread = matmul(&dx, &dxt);
                // Contribution: w * (P_i + spread)
                let p_plus_spread = self.models[i].p.add_mat(&spread);
                let contrib = p_plus_spread.scale(w);
                mixed_p[j] = mixed_p[j].add_mat(&contrib);
            }
        }

        // Write mixed conditions back to models
        for j in 0..K {
            self.models[j].x_hat = mixed_x[j];
            self.models[j].p = mixed_p[j];
        }

        // Update predicted probabilities
        self.mu = c_bar;
    }

    /// Run a standard Kalman predict+update for each model with measurement y.
    pub fn update_all(&mut self, y: &Matrix<S, M, 1>) {
        let ct = self.c.transpose();
        for model in self.models.iter_mut() {
            // Predict: x_hat = A * x_hat
            let ax = matmul(&model.a, &model.x_hat);
            model.x_hat = ax;

            // Predict covariance: P = A * P * A^T + Q
            let ap = matmul(&model.a, &model.p);
            let at = model.a.transpose();
            let apat = matmul(&ap, &at);
            model.p = apat.add_mat(&model.q);

            // Innovation covariance: S = C*P*C^T + R
            let cp = matmul(&self.c, &model.p);
            let cpct = matmul(&cp, &ct);
            let s_mat = cpct.add_mat(&self.r);

            let s_inv = match s_mat.inv() {
                Some(inv) => inv,
                None => {
                    model.likelihood = S::EPSILON;
                    continue;
                }
            };

            // Kalman gain: K = P*C^T*S^{-1}
            let pct = matmul(&model.p, &ct);
            let k = matmul(&pct, &s_inv);

            // Innovation
            let cx = matmul(&self.c, &model.x_hat);
            let innov = y.sub_mat(&cx);

            // Likelihood: approx Gaussian L ≈ exp(-0.5 * innov^T * S^{-1} * innov)
            // Compute innovation Mahalanobis distance
            let sinv_innov = matmul(&s_inv, &innov);
            let innov_t = innov.transpose();
            let mahal = matmul(&innov_t, &sinv_innov);
            let dist_sq = mahal.data[0][0];
            // Use approximation: exp(-0.5*d^2), clamp to avoid underflow
            let neg_half = S::from_f64(-0.5);
            let exponent = neg_half * dist_sq;
            // Safe exp: clamp exponent to [-50, 0]
            let exponent_clamped = exponent.clamp_val(S::from_f64(-50.0), S::ZERO);
            model.likelihood = exponent_clamped.exp();

            // State update
            let k_innov = matmul(&k, &innov);
            model.x_hat = model.x_hat.add_mat(&k_innov);

            // Covariance update (I - K*C)*P
            let kc = matmul(&k, &self.c);
            let eye = Matrix::<S, N, N>::identity();
            let i_minus_kc = eye.sub_mat(&kc);
            model.p = matmul(&i_minus_kc, &model.p);
        }
    }

    /// Update model probabilities from likelihoods.
    ///
    /// μ_k ← μ_k * L_k / (sum_j μ_j * L_j)
    pub fn update_probabilities(&mut self) {
        let mut total = S::ZERO;
        for k in 0..K {
            total += self.mu[k] * self.models[k].likelihood;
        }
        if total < S::EPSILON {
            // Degenerate: reset to uniform
            for k in 0..K {
                self.mu[k] = S::ONE / S::from_f64(K as f64);
            }
            return;
        }
        for k in 0..K {
            self.mu[k] = self.mu[k] * self.models[k].likelihood / total;
        }
    }

    /// Fused state estimate: weighted sum of model estimates.
    pub fn fused_estimate(&self) -> Matrix<S, N, 1> {
        let mut result = Matrix::<S, N, 1>::zeros();
        for k in 0..K {
            let contrib = self.models[k].x_hat.scale(self.mu[k]);
            result = result.add_mat(&contrib);
        }
        result
    }

    /// Model probability accessor.
    pub fn model_probability(&self, k: usize) -> S {
        self.mu[k]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_2model_imm() -> ImmEstimator<f64, 1, 1, 2> {
        // Model 0: slow dynamics a=0.9
        let model0 = ImmModel::<f64, 1, 1> {
            a: Matrix { data: [[0.9]] },
            x_hat: Matrix::<f64, 1, 1>::zeros(),
            p: Matrix::<f64, 1, 1>::identity(),
            q: Matrix { data: [[0.01]] },
            likelihood: 0.5,
        };
        // Model 1: fast dynamics a=0.5
        let model1 = ImmModel::<f64, 1, 1> {
            a: Matrix { data: [[0.5]] },
            x_hat: Matrix::<f64, 1, 1>::zeros(),
            p: Matrix::<f64, 1, 1>::identity(),
            q: Matrix { data: [[0.1]] },
            likelihood: 0.5,
        };
        let transition = [[0.9, 0.1], [0.1, 0.9]];
        let c = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let r = Matrix::<f64, 1, 1> { data: [[0.1]] };
        ImmEstimator::new([model0, model1], transition, c, r)
    }

    #[test]
    fn initial_probabilities_uniform() {
        let imm = make_2model_imm();
        assert!((imm.model_probability(0) - 0.5).abs() < 1e-12);
        assert!((imm.model_probability(1) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn fused_estimate_is_weighted_mean() {
        let mut imm = make_2model_imm();
        // Set different state estimates
        imm.models[0].x_hat.data[0][0] = 2.0;
        imm.models[1].x_hat.data[0][0] = 4.0;
        imm.mu = [0.5, 0.5];
        let fused = imm.fused_estimate();
        assert!(
            (fused.data[0][0] - 3.0).abs() < 1e-12,
            "fused={}",
            fused.data[0][0]
        );
    }

    #[test]
    fn update_all_changes_state() {
        let mut imm = make_2model_imm();
        let y = Matrix::<f64, 1, 1> { data: [[5.0]] };
        imm.update_all(&y);
        // After update, at least one model estimate should be non-zero
        let sum = imm.models[0].x_hat.data[0][0].abs() + imm.models[1].x_hat.data[0][0].abs();
        assert!(sum > 0.0);
    }

    #[test]
    fn probabilities_sum_to_one_after_update() {
        let mut imm = make_2model_imm();
        let y = Matrix::<f64, 1, 1> { data: [[1.0]] };
        imm.update_all(&y);
        imm.update_probabilities();
        let total = imm.mu[0] + imm.mu[1];
        assert!((total - 1.0).abs() < 1e-10, "total={total}");
    }
}
