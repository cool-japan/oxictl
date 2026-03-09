//! Property-based tests for kinematics invariants.

use oxictl::kinematics::forward::{Transform2D, Transform3D};
use oxictl::kinematics::jacobian::Jacobian2R;
use proptest::prelude::*;

proptest! {
    /// Transform2D identity is left and right unit of compose.
    #[test]
    fn transform2d_identity_is_unit(
        x in -10.0f64..10.0,
        y in -10.0f64..10.0,
        theta in -core::f64::consts::PI..core::f64::consts::PI,
    ) {
        let t = Transform2D::<f64>::new(x, y, theta);
        let id = Transform2D::<f64>::identity();
        let left = id.compose(&t);
        let right = t.compose(&id);
        prop_assert!((left.x - t.x).abs() < 1e-12);
        prop_assert!((left.y - t.y).abs() < 1e-12);
        prop_assert!((left.theta - t.theta).abs() < 1e-12);
        prop_assert!((right.x - t.x).abs() < 1e-12);
        prop_assert!((right.y - t.y).abs() < 1e-12);
        prop_assert!((right.theta - t.theta).abs() < 1e-12);
    }

    /// Transform2D compose is associative.
    #[test]
    fn transform2d_compose_associative(
        x1 in -5.0f64..5.0, y1 in -5.0f64..5.0, t1 in -3.0f64..3.0,
        x2 in -5.0f64..5.0, y2 in -5.0f64..5.0, t2 in -3.0f64..3.0,
        x3 in -5.0f64..5.0, y3 in -5.0f64..5.0, t3 in -3.0f64..3.0,
    ) {
        let a = Transform2D::<f64>::new(x1, y1, t1);
        let b = Transform2D::<f64>::new(x2, y2, t2);
        let c = Transform2D::<f64>::new(x3, y3, t3);
        let lhs = a.compose(&b).compose(&c);
        let rhs = a.compose(&b.compose(&c));
        prop_assert!((lhs.x - rhs.x).abs() < 1e-10, "x: {} vs {}", lhs.x, rhs.x);
        prop_assert!((lhs.y - rhs.y).abs() < 1e-10, "y: {} vs {}", lhs.y, rhs.y);
        prop_assert!((lhs.theta - rhs.theta).abs() < 1e-10);
    }

    /// Transform2D inverse undoes the transform.
    #[test]
    fn transform2d_inverse_is_inverse(
        x in -10.0f64..10.0,
        y in -10.0f64..10.0,
        theta in -core::f64::consts::PI..core::f64::consts::PI,
    ) {
        let t = Transform2D::<f64>::new(x, y, theta);
        let inv = t.inverse();
        let composed = t.compose(&inv);
        prop_assert!(composed.x.abs() < 1e-10, "x={}", composed.x);
        prop_assert!(composed.y.abs() < 1e-10, "y={}", composed.y);
        prop_assert!(composed.theta.abs() < 1e-10, "theta={}", composed.theta);
    }

    /// Transform2D transform_point is invertible.
    #[test]
    fn transform2d_point_roundtrip(
        tx in -5.0f64..5.0, ty in -5.0f64..5.0, theta in -core::f64::consts::PI..core::f64::consts::PI,
        px in -5.0f64..5.0, py in -5.0f64..5.0,
    ) {
        let t = Transform2D::<f64>::new(tx, ty, theta);
        let (fx, fy) = t.transform_point(px, py);
        let (rx, ry) = t.inverse().transform_point(fx, fy);
        prop_assert!((rx - px).abs() < 1e-9, "rx={rx}, px={px}");
        prop_assert!((ry - py).abs() < 1e-9, "ry={ry}, py={py}");
    }

    /// Transform3D rot_z inverse round-trips.
    #[test]
    fn transform3d_rot_z_invertible(theta in -core::f64::consts::PI..core::f64::consts::PI) {
        let t = Transform3D::<f64>::rot_z(theta);
        let inv = t.inverse();
        let composed = t.compose(&inv);
        let p = composed.transform_point([1.0, 0.5, -0.3]);
        prop_assert!((p[0] - 1.0).abs() < 1e-10);
        prop_assert!((p[1] - 0.5).abs() < 1e-10);
        prop_assert!((p[2] + 0.3).abs() < 1e-10);
    }

    /// Jacobian2R: compute returns finite values.
    #[test]
    fn jacobian2r_compute_finite(
        l1 in 0.1f64..2.0,
        l2 in 0.1f64..2.0,
        q1 in -core::f64::consts::PI..core::f64::consts::PI,
        q2 in -core::f64::consts::PI..core::f64::consts::PI,
    ) {
        let jac = Jacobian2R::<f64>::new(l1, l2);
        let j = jac.compute(q1, q2);
        for row in &j {
            for &v in row {
                prop_assert!(v.is_finite(), "Jacobian element is not finite: {v}");
            }
        }
    }

    /// Jacobian2R: determinant magnitude bounded by l1*l2.
    #[test]
    fn jacobian2r_determinant_bounded(
        l1 in 0.1f64..2.0,
        l2 in 0.1f64..2.0,
        q1 in -core::f64::consts::PI..core::f64::consts::PI,
        q2 in -core::f64::consts::PI..core::f64::consts::PI,
    ) {
        let jac = Jacobian2R::<f64>::new(l1, l2);
        let det = jac.determinant(q1, q2);
        // |det| = l1*l2*|sin(q2)| ≤ l1*l2
        prop_assert!(det.abs() <= l1 * l2 + 1e-9, "det={det}, l1*l2={}", l1*l2);
    }

    /// Jacobian2R: apply gives finite velocity output.
    #[test]
    fn jacobian2r_apply_finite(
        l1 in 0.1f64..2.0,
        l2 in 0.1f64..2.0,
        q1 in -core::f64::consts::PI..core::f64::consts::PI,
        q2 in -core::f64::consts::PI..core::f64::consts::PI,
        dq1 in -1.0f64..1.0,
        dq2 in -1.0f64..1.0,
    ) {
        let jac = Jacobian2R::<f64>::new(l1, l2);
        let (vx, vy) = jac.apply(q1, q2, dq1, dq2);
        prop_assert!(vx.is_finite() && vy.is_finite());
    }
}
