use crate::core::matrix::{matmul, matvec, vec_add, Matrix};
use crate::core::scalar::ControlScalar;

/// Discrete-time linear state-space model:
///   x[k+1] = A*x[k] + B*u[k]
///   y[k]   = C*x[k] + D*u[k]
///
/// N = state dimension, I = input dimension, O = output dimension.
#[derive(Debug, Clone, Copy)]
pub struct StateSpace<S: ControlScalar, const N: usize, const I: usize, const O: usize> {
    pub a: Matrix<S, N, N>,
    pub b: Matrix<S, N, I>,
    pub c: Matrix<S, O, N>,
    pub d: Matrix<S, O, I>,
    state: [S; N],
}

impl<S: ControlScalar, const N: usize, const I: usize, const O: usize> StateSpace<S, N, I, O> {
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        c: Matrix<S, O, N>,
        d: Matrix<S, O, I>,
    ) -> Self {
        Self {
            a,
            b,
            c,
            d,
            state: core::array::from_fn(|_| S::ZERO),
        }
    }

    /// Step the system with input u, return output y.
    pub fn step(&mut self, u: &[S; I]) -> [S; O] {
        let ax = matvec(&self.a, &self.state);
        let bu = matvec(&self.b, u);
        let cx = matvec(&self.c, &self.state);
        let du = matvec(&self.d, u);

        // Next state
        self.state = vec_add(&ax, &bu);

        // Output
        vec_add(&cx, &du)
    }

    /// Current state.
    pub fn state(&self) -> &[S; N] {
        &self.state
    }

    /// Reset state to zero.
    pub fn reset(&mut self) {
        self.state = core::array::from_fn(|_| S::ZERO);
    }

    /// Set initial state.
    pub fn set_state(&mut self, x0: [S; N]) {
        self.state = x0;
    }

    /// Compute output without stepping (uses current state).
    pub fn output(&self, u: &[S; I]) -> [S; O] {
        let cx = matvec(&self.c, &self.state);
        let du = matvec(&self.d, u);
        vec_add(&cx, &du)
    }
}

/// Continuous-time state-space model. Can be discretized via ZOH or Tustin.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousStateSpace<S: ControlScalar, const N: usize, const I: usize, const O: usize> {
    pub a: Matrix<S, N, N>,
    pub b: Matrix<S, N, I>,
    pub c: Matrix<S, O, N>,
    pub d: Matrix<S, O, I>,
}

impl<S: ControlScalar, const N: usize, const I: usize, const O: usize>
    ContinuousStateSpace<S, N, I, O>
{
    pub fn new(
        a: Matrix<S, N, N>,
        b: Matrix<S, N, I>,
        c: Matrix<S, O, N>,
        d: Matrix<S, O, I>,
    ) -> Self {
        Self { a, b, c, d }
    }

    /// Discretize via Euler (Forward Euler: Ad = I + A*dt, Bd = B*dt).
    /// Not recommended for accuracy-critical applications, but fast.
    pub fn discretize_euler(&self, dt: S) -> StateSpace<S, N, I, O> {
        let eye = Matrix::identity();
        let ad = eye.add_mat(&self.a.scale(dt));
        let bd = self.b.scale(dt);
        StateSpace::new(ad, bd, self.c, self.d)
    }

    /// Discretize via Tustin (bilinear transform / trapezoidal rule).
    /// Ad = (I + A*dt/2) * (I - A*dt/2)^-1
    /// Bd = (I - A*dt/2)^-1 * B * dt
    /// Returns None if (I - A*dt/2) is singular.
    pub fn discretize_tustin(&self, dt: S) -> Option<StateSpace<S, N, I, O>> {
        let half_dt = dt * S::HALF;
        let eye = Matrix::<S, N, N>::identity();
        let a_half = self.a.scale(half_dt);
        let m_plus = eye.add_mat(&a_half); // (I + A*dt/2)
        let m_minus = eye.sub_mat(&a_half); // (I - A*dt/2)

        let m_minus_inv = m_minus.inv()?;

        let ad = matmul(&m_plus, &m_minus_inv);
        let bd = matmul(&m_minus_inv, &self.b.scale(dt));

        Some(StateSpace::new(ad, bd, self.c, self.d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_order_system() -> ContinuousStateSpace<f64, 1, 1, 1> {
        // dx/dt = -x + u, y = x
        let a = Matrix { data: [[-1.0]] };
        let b = Matrix { data: [[1.0]] };
        let c = Matrix { data: [[1.0]] };
        let d = Matrix { data: [[0.0]] };
        ContinuousStateSpace::new(a, b, c, d)
    }

    #[test]
    fn euler_step() {
        let cont = first_order_system();
        let mut sys = cont.discretize_euler(0.001);
        let y = sys.step(&[1.0]);
        assert_eq!(y[0], 0.0); // initial state is 0
        let y2 = sys.step(&[1.0]);
        assert!(y2[0] > 0.0); // should increase
    }

    #[test]
    fn tustin_step() {
        let cont = first_order_system();
        let mut sys = cont.discretize_tustin(0.01).unwrap();
        sys.step(&[1.0]);
        let y = sys.step(&[1.0]);
        assert!(y[0] > 0.0);
    }

    #[test]
    fn state_space_direct() {
        let a = Matrix::<f64, 2, 2>::identity();
        let b = Matrix::<f64, 2, 1>::zeros();
        let c = Matrix::<f64, 1, 2>::zeros();
        let d = Matrix::<f64, 1, 1>::zeros();
        let mut sys = StateSpace::new(a, b, c, d);
        sys.set_state([1.0, 2.0]);
        let y = sys.step(&[0.0]);
        assert_eq!(y[0], 0.0);
        assert_eq!(sys.state(), &[1.0, 2.0]);
    }

    #[test]
    fn reset_zeros_state() {
        let a = Matrix::<f64, 1, 1>::identity();
        let b = Matrix::<f64, 1, 1>::identity();
        let c = Matrix::<f64, 1, 1>::identity();
        let d = Matrix::<f64, 1, 1>::zeros();
        let mut sys = StateSpace::new(a, b, c, d);
        sys.step(&[1.0]);
        sys.reset();
        assert_eq!(sys.state(), &[0.0]);
    }

    #[test]
    fn first_order_convergence_euler() {
        // dx/dt = -x + u, y = x, step input u=1, tau=1
        // Steady state: x_ss = 1
        let cont = first_order_system();
        let mut sys = cont.discretize_euler(0.001);
        for _ in 0..10_000 {
            sys.step(&[1.0]);
        }
        assert!((sys.state()[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn direct_feedthrough() {
        let a = Matrix::<f64, 1, 1>::zeros();
        let b = Matrix::<f64, 1, 1>::zeros();
        let c = Matrix::<f64, 1, 1>::zeros();
        let mut d = Matrix::<f64, 1, 1>::zeros();
        d.data[0][0] = 5.0;
        let mut sys = StateSpace::new(a, b, c, d);
        let y = sys.step(&[3.0]);
        assert_eq!(y[0], 15.0); // D * u = 5 * 3
    }
}
