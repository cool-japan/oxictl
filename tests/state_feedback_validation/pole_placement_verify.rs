//! Pole placement validation: eigenvalue-based settling time verification.
//!
//! When poles are placed at desired locations, the closed-loop system response
//! should match the theoretical convergence rate derived from those poles.
//! For a discrete-time system with dominant pole at z = r, the state envelope
//! decays as r^k, so settling time is determined by the slowest pole magnitude.

#[cfg(feature = "state_feedback")]
mod inner {
    use oxictl::core::matrix::Matrix;
    use oxictl::state_feedback::pole_placement::{ackermann, StateFeedback};

    /// First-order system: pole placed at z=0.5 → state halves every step.
    #[test]
    fn pole_at_half_gives_half_decay_per_step() {
        let a = Matrix::<f64, 1, 1> { data: [[0.9]] };
        let b = Matrix::<f64, 1, 1> { data: [[1.0]] };
        let desired = [0.5_f64];

        let k = ackermann(&a, &b, &desired)
            .expect("Ackermann should succeed for 1D controllable system");
        let sf = StateFeedback::new(k);

        let mut x = [1.0_f64];
        for _ in 0..10 {
            let u = sf.control(&x, &[0.0]);
            // State update: x[k+1] = a*x - b*K*(x - 0) = (a - b*K)*x
            x[0] = a.data[0][0] * x[0] + b.data[0][0] * u;
        }

        // After 10 steps with pole at 0.5: theoretical value = 0.5^10 = 1/1024
        let theoretical = 0.5_f64.powi(10);
        assert!(
            (x[0] - theoretical).abs() < 1e-9,
            "State after 10 steps: {:.2e} (theory: {:.2e})",
            x[0],
            theoretical
        );
    }

    /// Second-order system: poles placed at desired locations, settling time matches theory.
    #[test]
    fn second_order_poles_determine_settling_time() {
        // Unstable open-loop plant: double integrator
        let a = Matrix::<f64, 2, 2> {
            data: [[1.0, 0.1], [0.0, 1.0]],
        };
        let b = Matrix::<f64, 2, 1> {
            data: [[0.0], [0.1]],
        };
        // Place poles at z = 0.7, 0.75 → moderate decay
        let desired = [0.70_f64, 0.75];

        let k = ackermann(&a, &b, &desired).expect("Ackermann pole placement should succeed");
        let sf = StateFeedback::new(k);

        let mut x = [1.0_f64, 0.5];
        // Simulate until ||x|| < threshold_theory
        let dominant_pole = 0.75_f64; // slowest pole
        let threshold = 0.01_f64;
        // Theoretical steps to reach threshold: n = log(threshold / ||x0||) / log(dominant)
        let x0_norm = (x[0] * x[0] + x[1] * x[1]).sqrt();
        let theoretical_steps = (threshold / x0_norm).ln() / dominant_pole.ln();
        let theoretical_steps = theoretical_steps.ceil() as usize;

        let mut n_steps = 0_usize;
        for _ in 0..500 {
            let norm = (x[0] * x[0] + x[1] * x[1]).sqrt();
            if norm < threshold {
                break;
            }
            let u = sf.control(&x, &[0.0, 0.0]);
            let x0_new = a.data[0][0] * x[0] + a.data[0][1] * x[1] + b.data[0][0] * u;
            let x1_new = a.data[1][0] * x[0] + a.data[1][1] * x[1] + b.data[1][0] * u;
            x = [x0_new, x1_new];
            n_steps += 1;
        }

        assert!(
            n_steps <= theoretical_steps * 2 + 20,
            "Actual settling ({} steps) should be within 2x of theory ({} steps)",
            n_steps,
            theoretical_steps
        );

        let final_norm = (x[0] * x[0] + x[1] * x[1]).sqrt();
        assert!(
            final_norm < threshold * 5.0,
            "State norm {:.4} should be near zero after settling",
            final_norm
        );
    }

    /// Fast poles (near origin) settle faster than slow poles.
    #[test]
    fn faster_poles_settle_sooner() {
        let a = Matrix::<f64, 2, 2> {
            data: [[1.0, 0.1], [0.0, 1.0]],
        };
        let b = Matrix::<f64, 2, 1> {
            data: [[0.0], [0.1]],
        };

        let k_fast = ackermann(&a, &b, &[0.3_f64, 0.35]).expect("Fast pole placement");
        let k_slow = ackermann(&a, &b, &[0.7_f64, 0.75]).expect("Slow pole placement");

        let sf_fast = StateFeedback::new(k_fast);
        let sf_slow = StateFeedback::new(k_slow);

        let x0 = [1.0_f64, 0.5];

        // Run fast controller
        let mut x_fast = x0;
        for _ in 0..100 {
            let u = sf_fast.control(&x_fast, &[0.0, 0.0]);
            let x0_new = a.data[0][0] * x_fast[0] + a.data[0][1] * x_fast[1] + b.data[0][0] * u;
            let x1_new = a.data[1][0] * x_fast[0] + a.data[1][1] * x_fast[1] + b.data[1][0] * u;
            x_fast = [x0_new, x1_new];
        }

        // Run slow controller
        let mut x_slow = x0;
        for _ in 0..100 {
            let u = sf_slow.control(&x_slow, &[0.0, 0.0]);
            let x0_new = a.data[0][0] * x_slow[0] + a.data[0][1] * x_slow[1] + b.data[0][0] * u;
            let x1_new = a.data[1][0] * x_slow[0] + a.data[1][1] * x_slow[1] + b.data[1][0] * u;
            x_slow = [x0_new, x1_new];
        }

        let norm_fast = (x_fast[0] * x_fast[0] + x_fast[1] * x_fast[1]).sqrt();
        let norm_slow = (x_slow[0] * x_slow[0] + x_slow[1] * x_slow[1]).sqrt();

        assert!(
            norm_fast < norm_slow,
            "Fast poles ({:.4}) should yield smaller residual than slow poles ({:.4})",
            norm_fast,
            norm_slow
        );
    }

    /// Ackermann: closed-loop pole at exact desired location for 1D system.
    #[test]
    fn ackermann_exact_pole_location_1d() {
        for &desired_pole in &[0.1_f64, 0.3, 0.5, 0.8, -0.4] {
            let a = Matrix::<f64, 1, 1> { data: [[0.95]] };
            let b = Matrix::<f64, 1, 1> { data: [[1.0]] };
            let k = ackermann(&a, &b, &[desired_pole]).expect("Ackermann should succeed");
            // A_cl = A - B*K = 0.95 - k[0]
            let a_cl = 0.95 - k[0];
            assert!(
                (a_cl - desired_pole).abs() < 1e-9,
                "For desired pole {}, got closed-loop pole {:.8}",
                desired_pole,
                a_cl
            );
        }
    }
}

#[cfg(not(feature = "state_feedback"))]
#[test]
fn pole_placement_skipped_without_feature() {
    // Skipped: requires state_feedback feature.
}
