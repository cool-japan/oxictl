//! Priority-based cascaded control allocation.
//!
//! Multiple control objectives are assigned priorities. The highest-priority
//! task is satisfied first using the full actuator set. The null space of the
//! achieved solution is then used to satisfy lower-priority tasks.
//!
//! Each task specifies a 1×M effectiveness row and a desired scalar value.
//! Tasks are processed in order: index 0 is the highest priority.
//!
//! # Algorithm
//!
//! For each task i with row b_i and desired value v_i:
//! 1. Compute the weighted least-squares step for b_i: alpha = residual / (b_i W^{-1} b_i^T).
//! 2. Scale alpha so that no actuator violates remaining bounds.
//! 3. Apply scaled step; tighten remaining delta bounds.
use crate::core::scalar::ControlScalar;

pub use crate::allocation::weighted_pseudo::AllocationError;

/// A single control allocation task (one row of the effectiveness matrix).
#[derive(Debug, Clone, Copy)]
pub struct AllocationTask<S, const M: usize> {
    /// Effectiveness row b_i (1×M vector).
    pub b_row: [S; M],
    /// Desired virtual control value for this task.
    pub v_des: S,
}

impl<S: ControlScalar, const M: usize> AllocationTask<S, M> {
    /// Create a new task.
    pub fn new(b_row: [S; M], v_des: S) -> Self {
        Self { b_row, v_des }
    }
}

/// Priority-based cascaded control allocator.
///
/// # Type Parameters
/// - `S`: scalar type
/// - `M`: number of actuators
/// - `TASKS`: number of priority levels (tasks)
pub struct PriorityAllocator<S, const M: usize, const TASKS: usize> {
    /// Lower actuator bounds.
    u_min: [S; M],
    /// Upper actuator bounds.
    u_max: [S; M],
    /// Per-actuator weights (default all 1).
    weights: [S; M],
    /// Singularity threshold.
    singular_tol: S,
}

impl<S: ControlScalar, const M: usize, const TASKS: usize> PriorityAllocator<S, M, TASKS> {
    /// Create a new priority allocator.
    ///
    /// # Errors
    /// Returns `AllocationError::InvalidBounds` if any `u_min[i] > u_max[i]`.
    pub fn new(u_min: [S; M], u_max: [S; M]) -> Result<Self, AllocationError> {
        for i in 0..M {
            if u_min[i] > u_max[i] {
                return Err(AllocationError::InvalidBounds);
            }
        }
        Ok(Self {
            u_min,
            u_max,
            weights: [S::ONE; M],
            singular_tol: S::from_f64(1e-10),
        })
    }

    /// Set per-actuator weights (must all be positive).
    ///
    /// # Errors
    /// Returns `AllocationError::InvalidWeight` if any weight ≤ 0.
    pub fn set_weights(&mut self, weights: [S; M]) -> Result<(), AllocationError> {
        for &wi in weights.iter() {
            if wi <= S::ZERO {
                return Err(AllocationError::InvalidWeight);
            }
        }
        self.weights = weights;
        Ok(())
    }

    /// Perform cascaded priority allocation.
    ///
    /// Tasks are processed in order; index 0 has highest priority.
    ///
    /// Returns the M-dimensional actuator command vector.
    pub fn allocate(
        &self,
        tasks: &[AllocationTask<S, M>; TASKS],
    ) -> Result<[S; M], AllocationError> {
        let mut u = [S::ZERO; M];
        // Clamp initial u to feasible region
        for (j, u_j) in u.iter_mut().enumerate() {
            *u_j = u_j.clamp_val(self.u_min[j], self.u_max[j]);
        }

        // Remaining slack bounds for each actuator
        let mut delta_min = [S::ZERO; M];
        let mut delta_max = [S::ZERO; M];
        for j in 0..M {
            delta_min[j] = self.u_min[j] - u[j];
            delta_max[j] = self.u_max[j] - u[j];
        }

        for task in tasks.iter() {
            // Current virtual control residual
            let bu: S = task
                .b_row
                .iter()
                .zip(u.iter())
                .fold(S::ZERO, |acc, (&bij, &uj)| acc + bij * uj);
            let residual = task.v_des - bu;

            // WLS gain: b W^{-1} b^T (scalar)
            let b_winv_b: S = task
                .b_row
                .iter()
                .zip(self.weights.iter())
                .fold(S::ZERO, |acc, (&bij, &wj)| acc + bij * bij / wj);

            if b_winv_b < self.singular_tol {
                continue;
            }

            let alpha = residual / b_winv_b;

            // Compute unconstrained delta_u = alpha * W^{-1} b^T
            let mut delta_u = [S::ZERO; M];
            for (j, (du, (&bij, &wj))) in delta_u
                .iter_mut()
                .zip(task.b_row.iter().zip(self.weights.iter()))
                .enumerate()
            {
                let _ = j; // suppress unused warning from enumerate
                *du = alpha * bij / wj;
            }

            // Find maximum feasible scale factor in [0, 1]
            let mut scale = S::ONE;
            for (j, &du) in delta_u.iter().enumerate() {
                if du > S::ZERO && du > delta_max[j] && delta_max[j] >= S::ZERO {
                    let s = delta_max[j] / du;
                    if s < scale {
                        scale = s;
                    }
                } else if du < S::ZERO && du < delta_min[j] && delta_min[j] <= S::ZERO {
                    let s = delta_min[j] / du;
                    if s < scale {
                        scale = s;
                    }
                }
            }

            // Apply scaled step and tighten slack bounds
            for j in 0..M {
                let applied = scale * delta_u[j];
                u[j] += applied;
                delta_min[j] -= applied;
                delta_max[j] -= applied;
            }
        }

        // Final clamp for numerical safety
        for (j, u_j) in u.iter_mut().enumerate() {
            *u_j = u_j.clamp_val(self.u_min[j], self.u_max[j]);
        }

        Ok(u)
    }

    /// Evaluate how well each task is satisfied.
    ///
    /// Returns an array of residuals |b_i · u - v_des_i| for each task.
    pub fn task_residuals(&self, u: &[S; M], tasks: &[AllocationTask<S, M>; TASKS]) -> [S; TASKS] {
        let mut residuals = [S::ZERO; TASKS];
        for (i, task) in tasks.iter().enumerate() {
            let bu: S = task
                .b_row
                .iter()
                .zip(u.iter())
                .fold(S::ZERO, |acc, (&bij, &uj)| acc + bij * uj);
            let diff = bu - task.v_des;
            residuals[i] = if diff < S::ZERO { -diff } else { diff };
        }
        residuals
    }

    /// Check that u is within bounds.
    pub fn is_feasible(&self, u: &[S; M]) -> bool {
        u.iter().enumerate().all(|(j, &uj)| {
            uj >= self.u_min[j] - S::from_f64(1e-9) && uj <= self.u_max[j] + S::from_f64(1e-9)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Single task: result should match weighted pseudo-inverse with one row.
    #[test]
    fn test_single_task_pseudo_inverse() {
        let u_min = [0.0f64; 3];
        let u_max = [10.0f64; 3];
        let alloc = PriorityAllocator::<f64, 3, 1>::new(u_min, u_max).expect("ok");

        let tasks = [AllocationTask::new([1.0f64, 0.0, 1.0], 2.0)];
        let u = alloc.allocate(&tasks).expect("ok");

        let v = u[0] + u[2];
        assert!(approx(v, 2.0, 1e-8), "v={}", v);
        assert!(approx(u[1], 0.0, 1e-8));
    }

    /// Two tasks with orthogonal rows — both should be satisfied exactly.
    #[test]
    fn test_two_orthogonal_tasks() {
        let u_min = [-10.0f64; 2];
        let u_max = [10.0f64; 2];
        let alloc = PriorityAllocator::<f64, 2, 2>::new(u_min, u_max).expect("ok");

        let tasks = [
            AllocationTask::new([1.0f64, 0.0], 3.0),
            AllocationTask::new([0.0f64, 1.0], 2.0),
        ];
        let u = alloc.allocate(&tasks).expect("ok");

        assert!(approx(u[0], 3.0, 1e-6), "u[0]={}", u[0]);
        assert!(approx(u[1], 2.0, 1e-6), "u[1]={}", u[1]);
    }

    /// Priority ordering: higher-priority task dominates when actuators share capacity.
    #[test]
    fn test_priority_ordering() {
        let u_min = [0.0f64; 2];
        let u_max = [5.0f64; 2];
        let alloc = PriorityAllocator::<f64, 2, 2>::new(u_min, u_max).expect("ok");

        let tasks = [
            AllocationTask::new([1.0f64, 1.0], 4.0),
            AllocationTask::new([1.0f64, -1.0], 0.0),
        ];
        let u = alloc.allocate(&tasks).expect("ok");

        let v0 = u[0] + u[1];
        assert!(approx(v0, 4.0, 1e-5), "high-priority residual: v0={}", v0);
        assert!(alloc.is_feasible(&u), "u={:?} out of bounds", u);
    }

    /// Saturation: desired virtual control exceeds actuator limits.
    #[test]
    fn test_saturation() {
        let u_min = [0.0f64];
        let u_max = [5.0f64];
        let alloc = PriorityAllocator::<f64, 1, 1>::new(u_min, u_max).expect("ok");

        let tasks = [AllocationTask::new([1.0f64], 10.0)];
        let u = alloc.allocate(&tasks).expect("ok");

        assert!(u[0] <= 5.0 + 1e-9);
        assert!(u[0] >= -1e-9);
    }

    /// Task residuals are computed correctly.
    #[test]
    fn test_task_residuals() {
        let u_min = [-10.0f64; 2];
        let u_max = [10.0f64; 2];
        let alloc = PriorityAllocator::<f64, 2, 2>::new(u_min, u_max).expect("ok");

        let tasks = [
            AllocationTask::new([1.0f64, 0.0], 3.0),
            AllocationTask::new([0.0f64, 1.0], 2.0),
        ];
        let u = alloc.allocate(&tasks).expect("ok");
        let residuals = alloc.task_residuals(&u, &tasks);

        assert!(residuals[0] < 1e-6, "task 0 residual: {}", residuals[0]);
        assert!(residuals[1] < 1e-6, "task 1 residual: {}", residuals[1]);
    }

    /// is_feasible correctly identifies boundary violations.
    #[test]
    fn test_is_feasible() {
        let u_min = [0.0f64; 2];
        let u_max = [5.0f64; 2];
        let alloc = PriorityAllocator::<f64, 2, 2>::new(u_min, u_max).expect("ok");

        assert!(alloc.is_feasible(&[2.5f64, 3.0]));
        assert!(!alloc.is_feasible(&[6.0f64, 3.0]));
        assert!(!alloc.is_feasible(&[-1.0f64, 3.0]));
    }
}
