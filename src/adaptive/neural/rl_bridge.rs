use crate::core::scalar::ControlScalar;

/// Reinforcement Learning policy bridge for control integration.
///
/// Provides a zero-allocation stub for plugging in a pre-trained RL policy
/// (e.g., from Python/PyTorch training) as a Rust control component.
///
/// The policy maps observations → actions:
///   π: O^N → A^M
///
/// Three operating modes:
///   1. `Stub`: returns zeros (placeholder, for compilation/testing)
///   2. `Linear`: linear policy u = W·obs (useful for simple trained LQR-like policies)
///   3. `Table`: lookup table over discretized observation grid (1D)
///
/// Policy operating mode.
#[derive(Debug, Clone, Copy)]
pub enum PolicyMode {
    /// Always return zero actions (stub/placeholder).
    Stub,
    /// Linear policy: u = W·obs.
    Linear,
    /// Lookup table (1D observation discretized).
    Table,
}

/// RL policy bridge with N-dim observation and M-dim action.
///
/// `OBS` = observation dimension, `ACT` = action dimension, `TABLE_SIZE` = LUT entries.
pub struct RlPolicyBridge<
    S: ControlScalar,
    const OBS: usize,
    const ACT: usize,
    const TABLE_SIZE: usize,
> {
    /// Policy mode.
    pub mode: PolicyMode,
    /// Linear policy weight matrix W (ACT × OBS).
    pub weights: [[S; OBS]; ACT],
    /// Lookup table: each entry is an action vector.
    pub table: [[S; ACT]; TABLE_SIZE],
    /// Observation range for table indexing: [obs_min, obs_max].
    pub obs_min: S,
    pub obs_max: S,
    /// Action clipping bounds.
    pub action_min: [S; ACT],
    pub action_max: [S; ACT],
    /// Cumulative episode reward (for monitoring).
    pub episode_reward: S,
    /// Discount factor γ for reward computation.
    pub gamma: S,
}

impl<S: ControlScalar, const OBS: usize, const ACT: usize, const TABLE_SIZE: usize>
    RlPolicyBridge<S, OBS, ACT, TABLE_SIZE>
{
    pub fn new_stub() -> Self {
        Self {
            mode: PolicyMode::Stub,
            weights: [[S::ZERO; OBS]; ACT],
            table: [[S::ZERO; ACT]; TABLE_SIZE],
            obs_min: -S::ONE,
            obs_max: S::ONE,
            action_min: [-S::ONE; ACT],
            action_max: [S::ONE; ACT],
            episode_reward: S::ZERO,
            gamma: S::from_f64(0.99),
        }
    }

    /// Load a linear policy from weight matrix (ACT × OBS).
    pub fn load_linear(weights: [[S; OBS]; ACT]) -> Self {
        let mut s = Self::new_stub();
        s.mode = PolicyMode::Linear;
        s.weights = weights;
        s
    }

    /// Load a lookup-table policy.
    pub fn load_table(table: [[S; ACT]; TABLE_SIZE], obs_min: S, obs_max: S) -> Self {
        let mut s = Self::new_stub();
        s.mode = PolicyMode::Table;
        s.table = table;
        s.obs_min = obs_min;
        s.obs_max = obs_max;
        s
    }

    /// Query the policy: map observation → action.
    pub fn act(&self, obs: &[S; OBS]) -> [S; ACT] {
        let raw = match self.mode {
            PolicyMode::Stub => [S::ZERO; ACT],
            PolicyMode::Linear => core::array::from_fn(|i| {
                obs.iter()
                    .zip(self.weights[i].iter())
                    .fold(S::ZERO, |acc, (&o, &w)| acc + w * o)
            }),
            PolicyMode::Table => {
                if TABLE_SIZE == 0 {
                    return [S::ZERO; ACT];
                }
                // Use first observation dimension for 1D table lookup
                let o = obs[0];
                let range = self.obs_max - self.obs_min;
                let idx = if range.abs() < S::from_f64(1e-15) {
                    0
                } else {
                    let t = (o - self.obs_min) / range;
                    let t_clamped = t.clamp_val(S::ZERO, S::ONE);
                    let idx_f = t_clamped * S::from_f64((TABLE_SIZE - 1) as f64);
                    // Round to nearest: add 0.5, truncate
                    let idx_rounded = idx_f + S::from_f64(0.5);
                    // Convert to usize via f64
                    let idx_u = (idx_rounded * S::ONE).abs();
                    // Use comparison to extract integer
                    let mut best = 0usize;
                    let mut best_diff = (idx_u - S::ZERO).abs();
                    for k in 0..TABLE_SIZE {
                        let diff = (idx_u - S::from_f64(k as f64)).abs();
                        if diff < best_diff {
                            best_diff = diff;
                            best = k;
                        }
                    }
                    best
                };
                self.table[idx]
            }
        };

        // Clip actions to bounds
        core::array::from_fn(|i| raw[i].clamp_val(self.action_min[i], self.action_max[i]))
    }

    /// Update cumulative episode reward with current step reward.
    pub fn update_reward(&mut self, reward: S) {
        self.episode_reward = self.episode_reward * self.gamma + reward;
    }

    /// Reset episode reward tracking.
    pub fn reset_episode(&mut self) {
        self.episode_reward = S::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_returns_zeros() {
        let policy = RlPolicyBridge::<f64, 3, 2, 10>::new_stub();
        let obs = [1.0, -0.5, 0.3];
        let act = policy.act(&obs);
        assert_eq!(act, [0.0, 0.0]);
    }

    #[test]
    fn linear_policy_correct() {
        // Simple policy: u[0] = -obs[0], u[1] = 0.5*obs[1]
        let weights = [[-1.0_f64, 0.0], [0.0, 0.5]];
        let mut policy = RlPolicyBridge::<f64, 2, 2, 1>::load_linear(weights);
        policy.action_min = [-10.0, -10.0];
        policy.action_max = [10.0, 10.0];
        let obs = [3.0, 4.0];
        let act = policy.act(&obs);
        assert!((act[0] - (-3.0)).abs() < 1e-10, "act[0]={:.4}", act[0]);
        assert!((act[1] - 2.0).abs() < 1e-10, "act[1]={:.4}", act[1]);
    }

    #[test]
    fn linear_policy_action_clipped() {
        let weights = [[-10.0_f64, 0.0]]; // very large gain
        let mut policy = RlPolicyBridge::<f64, 2, 1, 1>::load_linear(weights);
        policy.action_min = [-2.0];
        policy.action_max = [2.0];
        let obs = [5.0, 0.0]; // u = -50, should be clipped to -2
        let act = policy.act(&obs);
        assert_eq!(act[0], -2.0);
    }

    #[test]
    fn table_policy_lookup() {
        // 5-entry table: obs ∈ [-1, 1] → action
        let table = [[-2.0_f64], [-1.0], [0.0], [1.0], [2.0]];
        let mut policy = RlPolicyBridge::<f64, 1, 1, 5>::load_table(table, -1.0, 1.0);
        policy.action_min = [-5.0];
        policy.action_max = [5.0];

        // obs = -1.0 → idx=0 → action = -2.0
        let act = policy.act(&[-1.0]);
        assert!((act[0] - (-2.0)).abs() < 0.01, "act={:.4}", act[0]);

        // obs = 0.0 → idx=2 → action = 0.0
        let act2 = policy.act(&[0.0]);
        assert!((act2[0] - 0.0).abs() < 0.01, "act={:.4}", act2[0]);

        // obs = 1.0 → idx=4 → action = 2.0
        let act3 = policy.act(&[1.0]);
        assert!((act3[0] - 2.0).abs() < 0.01, "act={:.4}", act3[0]);
    }

    #[test]
    fn reward_accumulation() {
        let mut policy = RlPolicyBridge::<f64, 1, 1, 1>::new_stub();
        policy.update_reward(1.0);
        policy.update_reward(1.0);
        // γ=0.99: R = 0*0.99 + 1 = 1, then 1*0.99 + 1 = 1.99
        assert!((policy.episode_reward - 1.99).abs() < 1e-6);
        policy.reset_episode();
        assert_eq!(policy.episode_reward, 0.0);
    }
}
