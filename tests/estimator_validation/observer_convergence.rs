//! Observer convergence validation: Luenberger, Disturbance Observer.

use oxictl::core::matrix::Matrix;
use oxictl::estimator::observer::disturbance::DisturbanceObserver;
use oxictl::estimator::observer::luenberger::LuenbergerObserver;

/// Luenberger observer converges to true state from wrong initial estimate.
#[test]
fn luenberger_converges_from_wrong_init() {
    let a = Matrix { data: [[0.9f64]] };
    let b = Matrix { data: [[1.0f64]] };
    let c = Matrix { data: [[1.0f64]] };
    let l = Matrix { data: [[0.5f64]] };

    let mut obs = LuenbergerObserver::new(a, b, c, l).with_initial_state([5.0f64]); // Wrong init: 5 vs 0

    let x_true = 0.0f64;
    for _ in 0..100 {
        let y = [x_true];
        let u = [0.0f64];
        obs.update(&u, &y);
    }

    assert!(
        (obs.state()[0] - x_true).abs() < 0.01,
        "Luenberger should converge: est={:.4}, true={}",
        obs.state()[0],
        x_true
    );
}

/// Luenberger observer tracks slowly varying state.
#[test]
fn luenberger_tracks_ramp() {
    let dt = 0.01f64;
    let a = Matrix { data: [[1.0f64]] };
    let b = Matrix { data: [[dt]] };
    let c = Matrix { data: [[1.0f64]] };
    let l = Matrix { data: [[0.9f64]] };

    let mut obs = LuenbergerObserver::new(a, b, c, l);

    let mut x_true = 0.0f64;
    for _ in 0..500 {
        let u = [1.0f64];
        x_true += dt * u[0];
        obs.update(&u, &[x_true]);
    }

    let err = (obs.state()[0] - x_true).abs();
    assert!(err < 0.05, "Luenberger ramp tracking error: {:.4}", err);
}

/// Disturbance observer estimates constant disturbance.
#[test]
fn disturbance_observer_estimates_constant_disturbance() {
    let a = Matrix { data: [[0.95f64]] };
    let b_col = [0.05f64];
    let c = Matrix { data: [[1.0f64]] };
    let l_x = Matrix { data: [[0.3f64]] };
    let l_d = [0.4f64];

    let mut dob = DisturbanceObserver::new(a, b_col, c, l_x, l_d);

    let true_disturbance = 2.0f64;
    let mut x_true = 0.0f64;

    for _ in 0..500 {
        let u = 0.0f64;
        x_true = 0.95 * x_true + 0.05 * (u + true_disturbance);
        dob.update(u, &[x_true]);
    }

    let est_disturbance = dob.disturbance();
    assert!(
        (est_disturbance - true_disturbance).abs() < 0.5,
        "DOB should estimate disturbance: est={:.3}, true={}",
        est_disturbance,
        true_disturbance
    );
}

/// Disturbance observer: zero disturbance → estimate near zero.
#[test]
fn disturbance_observer_zero_disturbance() {
    let a = Matrix { data: [[0.8f64]] };
    let b_col = [0.2f64];
    let c = Matrix { data: [[1.0f64]] };
    let l_x = Matrix { data: [[0.3f64]] };
    let l_d = [0.2f64];

    let mut dob = DisturbanceObserver::new(a, b_col, c, l_x, l_d);

    let mut x_true = 1.0f64;
    for _ in 0..200 {
        x_true *= 0.8;
        dob.update(0.0f64, &[x_true]);
    }

    let est_d = dob.disturbance();
    assert!(
        est_d.abs() < 0.1,
        "No disturbance → est should be ~0: {:.4}",
        est_d
    );
}

/// Observer error: 2-state system.
#[test]
fn luenberger_2state_convergence() {
    let dt = 0.01f64;
    let a = Matrix {
        data: [[1.0f64, dt], [0.0, 0.9]],
    };
    let b = Matrix::<f64, 2, 1>::zeros();
    let c = Matrix {
        data: [[1.0f64, 0.0]],
    };
    let l = Matrix {
        data: [[0.9f64], [5.0]],
    };

    let mut obs = LuenbergerObserver::new(a, b, c, l).with_initial_state([0.0f64, 5.0]); // Wrong velocity init

    let mut x_true = [1.0f64, 0.0];
    for _ in 0..300 {
        let x0 = x_true[0] + dt * x_true[1];
        let x1 = 0.9 * x_true[1];
        x_true = [x0, x1];
        obs.update(&[0.0f64], &[x_true[0]]);
    }

    let pos_err = (obs.state()[0] - x_true[0]).abs();
    let vel_err = (obs.state()[1] - x_true[1]).abs();
    assert!(
        pos_err < 0.05,
        "Position observer error too large: {:.4}",
        pos_err
    );
    assert!(
        vel_err < 0.5,
        "Velocity observer error too large: {:.4}",
        vel_err
    );
}
