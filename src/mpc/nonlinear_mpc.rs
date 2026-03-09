use crate::core::scalar::ControlScalar;

/// Nonlinear MPC using single-shooting + gradient descent.
///
/// Minimizes:
///   J = Σ_{k=0}^{H-1} [ ||x_k − x_ref||_Q² + ||u_k||_R² ] + ||x_H − x_ref||_Qt²
///
/// Subject to:
///   x_{k+1} = f(x_k, u_k)   (user-supplied discrete-time plant)
///   u_min ≤ u_k ≤ u_max
///
/// Gradient is computed by finite differences on the cost function.
/// `N` = state dim, `M` = input dim, `H` = prediction horizon.
pub struct NonlinearMpc<S: ControlScalar, const N: usize, const M: usize, const H: usize> {
    /// State tracking weight vector (diagonal Q).
    pub q: [S; N],
    /// Terminal state tracking weight (diagonal Qt).
    pub q_terminal: [S; N],
    /// Input weight vector (diagonal R).
    pub r: [S; M],
    /// Input lower bound.
    pub u_min: [S; M],
    /// Input upper bound.
    pub u_max: [S; M],
    /// Finite difference step for gradient computation.
    pub fd_step: S,
    /// Warm-started input sequence.
    u_seq: [[S; M]; H],
    /// Discrete-time plant model: f(x, u) → x_next.
    plant_fn: fn(&[S; N], &[S; M]) -> [S; N],
}

impl<S: ControlScalar, const N: usize, const M: usize, const H: usize> NonlinearMpc<S, N, M, H> {
    pub fn new(
        q: [S; N],
        q_terminal: [S; N],
        r: [S; M],
        u_min: [S; M],
        u_max: [S; M],
        plant_fn: fn(&[S; N], &[S; M]) -> [S; N],
    ) -> Self {
        Self {
            q,
            q_terminal,
            r,
            u_min,
            u_max,
            fd_step: S::from_f64(1e-5),
            u_seq: [[S::ZERO; M]; H],
            plant_fn,
        }
    }

    /// Solve the NMPC problem and return the first control action.
    ///
    /// - `x`: current state
    /// - `x_ref`: reference state trajectory (constant reference)
    /// - `step_size`: gradient descent step size
    /// - `iterations`: number of gradient steps
    pub fn update(
        &mut self,
        x: &[S; N],
        x_ref: &[S; N],
        step_size: S,
        iterations: usize,
    ) -> [S; M] {
        // Warm-start shift
        for k in 0..H.saturating_sub(1) {
            self.u_seq[k] = self.u_seq[k + 1];
        }

        let j0 = self.compute_cost(x, x_ref, &self.u_seq);
        let _ = j0;

        for _ in 0..iterations {
            // Compute gradient by finite differences
            let mut grad = [[S::ZERO; M]; H];
            for (k, gk) in grad.iter_mut().enumerate() {
                for (i, gi) in gk.iter_mut().enumerate() {
                    let mut u_plus = self.u_seq;
                    u_plus[k][i] += self.fd_step;
                    let j_plus = self.compute_cost(x, x_ref, &u_plus);
                    let mut u_minus = self.u_seq;
                    u_minus[k][i] -= self.fd_step;
                    let j_minus = self.compute_cost(x, x_ref, &u_minus);
                    *gi = (j_plus - j_minus) / (S::TWO * self.fd_step);
                }
            }

            // Gradient step + clamp to constraints
            for (k, uk) in self.u_seq.iter_mut().enumerate() {
                for (i, ui) in uk.iter_mut().enumerate() {
                    *ui -= step_size * grad[k][i];
                    *ui = ui.clamp_val(self.u_min[i], self.u_max[i]);
                }
            }
        }

        self.u_seq[0]
    }

    fn compute_cost(&self, x0: &[S; N], x_ref: &[S; N], u_seq: &[[S; M]; H]) -> S {
        let mut x = *x0;
        let mut cost = S::ZERO;

        for uk in u_seq.iter() {
            // State tracking cost
            for (&xi, (&qi, &ri_ref)) in x.iter().zip(self.q.iter().zip(x_ref.iter())) {
                let e = xi - ri_ref;
                cost += qi * e * e;
            }
            // Input cost
            for (&ui, &ri) in uk.iter().zip(self.r.iter()) {
                cost += ri * ui * ui;
            }
            x = (self.plant_fn)(&x, uk);
        }

        // Terminal cost
        for i in 0..N {
            let e = x[i] - x_ref[i];
            cost += self.q_terminal[i] * e * e;
        }
        cost
    }

    pub fn reset(&mut self) {
        self.u_seq = [[S::ZERO; M]; H];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple discrete double integrator: x_{k+1} = A*x + B*u
    fn double_integrator(x: &[f64; 2], u: &[f64; 1]) -> [f64; 2] {
        let dt = 0.1;
        [x[0] + x[1] * dt, x[1] + u[0] * dt]
    }

    #[test]
    fn nmpc_drives_state_to_zero() {
        let q = [10.0_f64, 1.0];
        let qt = [10.0_f64, 1.0];
        let r = [0.1_f64];

        let mut mpc =
            NonlinearMpc::<f64, 2, 1, 10>::new(q, qt, r, [-5.0], [5.0], double_integrator);

        let x_ref = [0.0_f64; 2];
        let mut x = [2.0_f64, 0.0]; // start at position 2

        for _ in 0..100 {
            let u = mpc.update(&x, &x_ref, 0.01, 10);
            x = double_integrator(&x, &u);
        }

        assert!(x[0].abs() < 0.5, "state x[0]={:.4} should be near 0", x[0]);
    }

    #[test]
    fn input_constraints_respected() {
        let mut mpc = NonlinearMpc::<f64, 2, 1, 5>::new(
            [1.0, 1.0],
            [1.0, 1.0],
            [0.1],
            [-1.0],
            [1.0],
            double_integrator,
        );
        let x = [5.0_f64, 0.0];
        let u = mpc.update(&x, &[0.0; 2], 0.01, 5);
        assert!(u[0] >= -1.0 - 1e-10 && u[0] <= 1.0 + 1e-10, "u={:.4}", u[0]);
    }
}
