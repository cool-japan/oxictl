//! Activation functions and their derivatives for neural network layers.
//!
//! All activations are generic over scalar types implementing `num_traits::Float`
//! so the same code compiles for f32 and f64 without heap allocation.

use num_traits::Float;

/// Trait for activation functions used in neural network layers.
///
/// Implementors must provide a forward pass (`apply`) and the analytic
/// derivative (`derivative`) used during back-propagation.
pub trait ActivationFn<S: Float + Copy>: Copy {
    /// Apply the activation function: out = f(x).
    fn apply(&self, x: S) -> S;

    /// Derivative of the activation with respect to its input: df/dx at x.
    fn derivative(&self, x: S) -> S;
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

/// Rectified Linear Unit: f(x) = max(0, x).
///
/// Derivative: 1 if x > 0, else 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct Relu;

impl<S: Float + Copy> ActivationFn<S> for Relu {
    #[inline]
    fn apply(&self, x: S) -> S {
        if x > S::zero() {
            x
        } else {
            S::zero()
        }
    }

    #[inline]
    fn derivative(&self, x: S) -> S {
        if x > S::zero() {
            S::one()
        } else {
            S::zero()
        }
    }
}

// ---------------------------------------------------------------------------
// LeakyReLU
// ---------------------------------------------------------------------------

/// Leaky ReLU: f(x) = x if x ≥ 0, else α·x.
///
/// `α` is the negative-slope coefficient (commonly 0.01).
#[derive(Debug, Clone, Copy)]
pub struct LeakyRelu<S: Float + Copy> {
    /// Negative slope coefficient α (e.g. 0.01).
    pub alpha: S,
}

impl<S: Float + Copy> LeakyRelu<S> {
    /// Create a `LeakyRelu` with given negative-slope coefficient.
    pub fn new(alpha: S) -> Self {
        Self { alpha }
    }
}

impl<S: Float + Copy> ActivationFn<S> for LeakyRelu<S> {
    #[inline]
    fn apply(&self, x: S) -> S {
        if x >= S::zero() {
            x
        } else {
            self.alpha * x
        }
    }

    #[inline]
    fn derivative(&self, x: S) -> S {
        if x >= S::zero() {
            S::one()
        } else {
            self.alpha
        }
    }
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

/// Sigmoid: σ(x) = 1 / (1 + exp(−x)).
///
/// Derivative: σ(x) · (1 − σ(x)).
#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid<S: Float + Copy> {
    _phantom: core::marker::PhantomData<S>,
}

impl<S: Float + Copy> Sigmoid<S> {
    /// Create a `Sigmoid` activation.
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<S: Float + Copy> ActivationFn<S> for Sigmoid<S> {
    #[inline]
    fn apply(&self, x: S) -> S {
        S::one() / (S::one() + (-x).exp())
    }

    #[inline]
    fn derivative(&self, x: S) -> S {
        let s = self.apply(x);
        s * (S::one() - s)
    }
}

// ---------------------------------------------------------------------------
// Tanh
// ---------------------------------------------------------------------------

/// Hyperbolic tangent: f(x) = tanh(x).
///
/// Derivative: 1 − tanh²(x).
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh<S: Float + Copy> {
    _phantom: core::marker::PhantomData<S>,
}

impl<S: Float + Copy> Tanh<S> {
    /// Create a `Tanh` activation.
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<S: Float + Copy> ActivationFn<S> for Tanh<S> {
    #[inline]
    fn apply(&self, x: S) -> S {
        x.tanh()
    }

    #[inline]
    fn derivative(&self, x: S) -> S {
        let t = x.tanh();
        S::one() - t * t
    }
}

// ---------------------------------------------------------------------------
// Linear (identity)
// ---------------------------------------------------------------------------

/// Identity / linear activation: f(x) = x.
///
/// Derivative: 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct Linear<S: Float + Copy> {
    _phantom: core::marker::PhantomData<S>,
}

impl<S: Float + Copy> Linear<S> {
    /// Create a `Linear` (identity) activation.
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<S: Float + Copy> ActivationFn<S> for Linear<S> {
    #[inline]
    fn apply(&self, x: S) -> S {
        x
    }

    #[inline]
    fn derivative(&self, _x: S) -> S {
        S::one()
    }
}

// ---------------------------------------------------------------------------
// Swish
// ---------------------------------------------------------------------------

/// Swish: f(x) = x · σ(x).
///
/// Derivative: σ(x) + x · σ(x) · (1 − σ(x))
///           = σ(x) · (1 + x · (1 − σ(x))).
#[derive(Debug, Clone, Copy, Default)]
pub struct Swish<S: Float + Copy> {
    _phantom: core::marker::PhantomData<S>,
}

impl<S: Float + Copy> Swish<S> {
    /// Create a `Swish` activation.
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<S: Float + Copy> ActivationFn<S> for Swish<S> {
    #[inline]
    fn apply(&self, x: S) -> S {
        let sigma = S::one() / (S::one() + (-x).exp());
        x * sigma
    }

    #[inline]
    fn derivative(&self, x: S) -> S {
        let sigma = S::one() / (S::one() + (-x).exp());
        // σ(x) + x·σ(x)·(1−σ(x))
        sigma + x * sigma * (S::one() - sigma)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Numerically verify that `derivative` matches a central finite difference
    /// of `apply`.  We check several representative points.
    fn check_derivative_numerically<S, A>(act: &A, xs: &[S], h: S, tol: S)
    where
        S: Float + Copy + core::fmt::Debug,
        A: ActivationFn<S>,
    {
        for &x in xs {
            let fd = (act.apply(x + h) - act.apply(x - h)) / (h + h);
            let analytic = act.derivative(x);
            let err = (fd - analytic).abs();
            assert!(
                err < tol,
                "derivative mismatch at x={x:?}: fd={fd:?}, analytic={analytic:?}, err={err:?}"
            );
        }
    }

    #[test]
    fn relu_values() {
        let r = Relu;
        assert_eq!(r.apply(2.0_f64), 2.0);
        assert_eq!(r.apply(-1.0_f64), 0.0);
        assert_eq!(r.apply(0.0_f64), 0.0);
        assert_eq!(r.derivative(1.0_f64), 1.0);
        assert_eq!(r.derivative(-1.0_f64), 0.0);
    }

    #[test]
    fn relu_derivative_numerical() {
        let act = Relu;
        // Avoid x=0 (non-differentiable)
        let xs = [0.5_f64, 1.0, 2.0, -0.5, -1.0];
        check_derivative_numerically(&act, &xs, 1e-5, 1e-4);
    }

    #[test]
    fn leaky_relu_values() {
        let lr = LeakyRelu::new(0.01_f64);
        assert!((lr.apply(2.0) - 2.0).abs() < 1e-12);
        assert!((lr.apply(-2.0) - (-0.02)).abs() < 1e-12);
        assert_eq!(lr.derivative(1.0), 1.0);
        assert!((lr.derivative(-1.0) - 0.01).abs() < 1e-12);
    }

    #[test]
    fn leaky_relu_derivative_numerical() {
        let act = LeakyRelu::new(0.01_f64);
        let xs = [0.5_f64, 1.5, -0.5, -1.5];
        check_derivative_numerically(&act, &xs, 1e-5, 1e-4);
    }

    #[test]
    fn sigmoid_values() {
        let s = Sigmoid::<f64>::new();
        let v = s.apply(0.0);
        assert!((v - 0.5).abs() < 1e-12);
        let v_large = s.apply(100.0);
        assert!((v_large - 1.0).abs() < 1e-6);
        let v_neg = s.apply(-100.0);
        assert!(v_neg.abs() < 1e-6);
    }

    #[test]
    fn sigmoid_derivative_numerical() {
        let act = Sigmoid::<f64>::new();
        let xs = [-2.0_f64, -0.5, 0.0, 0.5, 2.0];
        check_derivative_numerically(&act, &xs, 1e-5, 1e-4);
    }

    #[test]
    fn tanh_values() {
        let t = Tanh::<f64>::new();
        assert!((t.apply(0.0) - 0.0).abs() < 1e-12);
        assert!((t.apply(1.0) - 1.0_f64.tanh()).abs() < 1e-12);
        assert_eq!(t.derivative(0.0), 1.0);
    }

    #[test]
    fn tanh_derivative_numerical() {
        let act = Tanh::<f64>::new();
        let xs = [-2.0_f64, -1.0, 0.0, 1.0, 2.0];
        check_derivative_numerically(&act, &xs, 1e-5, 1e-4);
    }

    #[test]
    fn linear_values() {
        let l = Linear::<f64>::new();
        assert_eq!(l.apply(core::f64::consts::PI), core::f64::consts::PI);
        assert_eq!(l.derivative(42.0), 1.0);
        assert_eq!(l.derivative(-99.0), 1.0);
    }

    #[test]
    fn linear_derivative_numerical() {
        let act = Linear::<f64>::new();
        let xs = [-5.0_f64, 0.0, 5.0];
        check_derivative_numerically(&act, &xs, 1e-5, 1e-4);
    }

    #[test]
    fn swish_values() {
        let sw = Swish::<f64>::new();
        // swish(0) = 0 * sigmoid(0) = 0
        assert!(sw.apply(0.0).abs() < 1e-12);
        // swish should be monotonically increasing for large positive x
        assert!(sw.apply(5.0) > sw.apply(1.0));
    }

    #[test]
    fn swish_derivative_numerical() {
        let act = Swish::<f64>::new();
        let xs = [-2.0_f64, -0.5, 0.0, 0.5, 2.0];
        check_derivative_numerically(&act, &xs, 1e-5, 1e-4);
    }
}
