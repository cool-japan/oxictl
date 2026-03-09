use crate::core::scalar::ControlScalar;

/// Prediction horizon configuration for MPC.
///
/// Manages the relationship between:
///   - Prediction horizon Hp: how far ahead we predict state evolution
///   - Control horizon Hc: how many control moves we optimize (Hc ≤ Hp)
///   - Blocking factor: groups of steps using the same control input
///
/// Blocking reduces the optimization dimension: Hc blocked inputs instead of Hp.
#[derive(Debug, Clone, Copy)]
pub struct HorizonConfig {
    /// Prediction horizon (steps).
    pub prediction: usize,
    /// Control horizon (steps, ≤ prediction).
    pub control: usize,
    /// Sample time (s) — time between steps.
    pub dt: f64,
}

impl HorizonConfig {
    /// Create a symmetric horizon (control = prediction).
    pub fn new(horizon: usize, dt: f64) -> Self {
        Self {
            prediction: horizon,
            control: horizon,
            dt,
        }
    }

    /// Create with separate prediction and control horizons.
    pub fn with_control_horizon(prediction: usize, control: usize, dt: f64) -> Self {
        let control = control.min(prediction);
        Self {
            prediction,
            control,
            dt,
        }
    }

    /// Total prediction time (seconds).
    pub fn prediction_time_s(&self) -> f64 {
        self.prediction as f64 * self.dt
    }

    /// Control time span (seconds).
    pub fn control_time_s(&self) -> f64 {
        self.control as f64 * self.dt
    }

    /// Number of repeated control steps (tail where u = u_last).
    pub fn tail_steps(&self) -> usize {
        self.prediction.saturating_sub(self.control)
    }
}

/// Variable prediction horizon (Economic MPC / shrinking horizon).
///
/// Shrinking horizon: reduces prediction window as the end of a batch approaches.
/// Receding horizon: always keeps the same window (standard MPC).
#[derive(Debug, Clone, Copy)]
pub struct VariableHorizon {
    /// Maximum prediction horizon.
    pub max_horizon: usize,
    /// Current horizon (changes with time).
    pub current: usize,
    /// Mode: shrinking (true) or receding (false).
    pub shrinking: bool,
    /// Remaining steps until end-of-batch (for shrinking horizon).
    pub steps_remaining: usize,
}

impl VariableHorizon {
    /// Create a receding (standard MPC) horizon.
    pub fn receding(horizon: usize) -> Self {
        Self {
            max_horizon: horizon,
            current: horizon,
            shrinking: false,
            steps_remaining: 0,
        }
    }

    /// Create a shrinking horizon for batch processes.
    pub fn shrinking(max_horizon: usize, batch_steps: usize) -> Self {
        let current = max_horizon.min(batch_steps);
        Self {
            max_horizon,
            current,
            shrinking: true,
            steps_remaining: batch_steps,
        }
    }

    /// Advance one step. Updates current horizon if shrinking.
    pub fn step(&mut self) {
        if self.shrinking && self.steps_remaining > 0 {
            self.steps_remaining -= 1;
            self.current = self.max_horizon.min(self.steps_remaining);
        }
    }

    /// True if batch is complete (shrinking horizon only).
    pub fn is_complete(&self) -> bool {
        self.shrinking && self.steps_remaining == 0
    }

    /// Reset shrinking horizon for a new batch.
    pub fn reset(&mut self, batch_steps: usize) {
        self.steps_remaining = batch_steps;
        self.current = self.max_horizon.min(batch_steps);
    }
}

/// Reference trajectory over the prediction horizon.
///
/// Maps each step k ∈ [0, H) to a reference state.
/// Three modes: constant, step change, linear interpolation.
#[derive(Debug, Clone, Copy)]
pub enum HorizonReference<S: ControlScalar, const N: usize, const H: usize> {
    /// Same reference for all steps.
    Constant([S; N]),
    /// Step change at step `k_step`.
    Step {
        before: [S; N],
        after: [S; N],
        k_step: usize,
    },
    /// Linear interpolation from `start` to `end`.
    Linear { start: [S; N], end: [S; N] },
}

impl<S: ControlScalar, const N: usize, const H: usize> HorizonReference<S, N, H> {
    /// Get reference state at step k.
    pub fn at(&self, k: usize) -> [S; N] {
        match *self {
            Self::Constant(r) => r,
            Self::Step {
                before,
                after,
                k_step,
            } => {
                if k < k_step {
                    before
                } else {
                    after
                }
            }
            Self::Linear { start, end } => {
                if H == 0 || H == 1 {
                    return end;
                }
                let t = S::from_f64(k as f64 / (H - 1) as f64);
                core::array::from_fn(|i| start[i] + t * (end[i] - start[i]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizon_config_times() {
        let h = HorizonConfig::with_control_horizon(10, 5, 0.1);
        assert_eq!(h.prediction, 10);
        assert_eq!(h.control, 5);
        assert!((h.prediction_time_s() - 1.0).abs() < 1e-10);
        assert!((h.control_time_s() - 0.5).abs() < 1e-10);
        assert_eq!(h.tail_steps(), 5);
    }

    #[test]
    fn horizon_control_clamped_to_prediction() {
        let h = HorizonConfig::with_control_horizon(5, 10, 0.1);
        assert_eq!(h.control, 5); // clamped
    }

    #[test]
    fn receding_horizon_no_change() {
        let mut h = VariableHorizon::receding(10);
        h.step();
        h.step();
        assert_eq!(h.current, 10); // unchanged
    }

    #[test]
    fn shrinking_horizon_decreases() {
        let mut h = VariableHorizon::shrinking(5, 8);
        assert_eq!(h.current, 5);
        h.step(); // remaining = 7 → current = min(5,7) = 5
        h.step(); // remaining = 6 → current = 5
        h.step(); // remaining = 5 → current = 5
        h.step(); // remaining = 4 → current = 4
        assert_eq!(h.current, 4);
    }

    #[test]
    fn shrinking_horizon_completes() {
        let mut h = VariableHorizon::shrinking(3, 3);
        for _ in 0..3 {
            h.step();
        }
        assert!(h.is_complete());
    }

    #[test]
    fn reference_constant() {
        let r = HorizonReference::<f64, 2, 10>::Constant([1.0, 2.0]);
        assert_eq!(r.at(0), [1.0, 2.0]);
        assert_eq!(r.at(9), [1.0, 2.0]);
    }

    #[test]
    fn reference_step_change() {
        let r = HorizonReference::<f64, 1, 10>::Step {
            before: [0.0],
            after: [5.0],
            k_step: 5,
        };
        assert_eq!(r.at(4), [0.0]);
        assert_eq!(r.at(5), [5.0]);
    }

    #[test]
    fn reference_linear_endpoints() {
        let r = HorizonReference::<f64, 1, 11>::Linear {
            start: [0.0],
            end: [10.0],
        };
        assert!((r.at(0)[0] - 0.0).abs() < 1e-10);
        assert!((r.at(10)[0] - 10.0).abs() < 1e-10);
        assert!((r.at(5)[0] - 5.0).abs() < 1e-10);
    }
}
