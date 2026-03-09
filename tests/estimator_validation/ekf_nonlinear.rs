//! EKF nonlinear estimation validation.

use oxictl::core::matrix::Matrix;
use oxictl::estimator::ekf::Ekf;

/// Simple pendulum angle estimation via EKF.
/// State: [theta, theta_dot], Observation: [sin(theta)]
/// f(x, u) = [x[0] + x[1]*dt, x[1] - (g/l)*sin(x[0])*dt]
/// h(x) = [sin(x[0])]
const DT: f64 = 0.01;
const G: f64 = 9.81;
const L: f64 = 1.0;

fn pendulum_f(x: &[f64; 2], _u: &[f64; 1]) -> [f64; 2] {
    [x[0] + x[1] * DT, x[1] - (G / L) * x[0].sin() * DT]
}

fn pendulum_h(x: &[f64; 2]) -> [f64; 1] {
    [x[0].sin()]
}

fn pendulum_f_jac(_x: &[f64; 2], _u: &[f64; 1]) -> Matrix<f64, 2, 2> {
    Matrix {
        data: [[1.0, DT], [-(G / L) * DT, 1.0]],
    }
}

fn pendulum_h_jac(x: &[f64; 2]) -> Matrix<f64, 1, 2> {
    Matrix {
        data: [[x[0].cos(), 0.0]],
    }
}

/// EKF tracks pendulum angle accurately.
#[test]
fn ekf_tracks_pendulum_angle() {
    let q = Matrix {
        data: [[1e-6f64, 0.0], [0.0, 1e-5]],
    };
    let r = Matrix { data: [[0.01f64]] };
    let p0 = Matrix {
        data: [[0.1f64, 0.0], [0.0, 0.1]],
    };

    let mut ekf = Ekf::new(
        q,
        r,
        pendulum_f,
        pendulum_f_jac,
        pendulum_h,
        pendulum_h_jac,
        [0.2f64, 0.0], // Initial: theta=0.2 rad, theta_dot=0
        p0,
    );

    let mut x_true = [0.2f64, 0.0];
    let mut rmse_sum = 0.0f64;
    let n_steps = 300usize;

    for step in 0..n_steps {
        let theta_new = x_true[0] + x_true[1] * DT;
        let thetad_new = x_true[1] - (G / L) * x_true[0].sin() * DT;
        x_true = [theta_new, thetad_new];

        let z = [x_true[0].sin()];
        ekf.predict(&[0.0f64]);
        ekf.update(&z);

        if step > 50 {
            let err = ekf.state()[0] - x_true[0];
            rmse_sum += err * err;
        }
    }

    let rmse = (rmse_sum / (n_steps - 50) as f64).sqrt();
    assert!(
        rmse < 0.05,
        "EKF angle RMSE={:.6} should be < 0.05 rad",
        rmse
    );
}

/// EKF covariance stays positive definite (diagonal elements > 0).
#[test]
fn ekf_covariance_stays_positive() {
    let q = Matrix {
        data: [[1e-4f64, 0.0], [0.0, 1e-3]],
    };
    let r = Matrix { data: [[0.05f64]] };
    let p0 = Matrix {
        data: [[1.0f64, 0.0], [0.0, 1.0]],
    };

    let mut ekf = Ekf::new(
        q,
        r,
        pendulum_f,
        pendulum_f_jac,
        pendulum_h,
        pendulum_h_jac,
        [0.5f64, 0.0],
        p0,
    );

    let mut x_true = [0.5f64, 0.0];
    for _ in 0..100 {
        let theta_new = x_true[0] + x_true[1] * DT;
        let thetad_new = x_true[1] - (G / L) * x_true[0].sin() * DT;
        x_true = [theta_new, thetad_new];

        ekf.predict(&[0.0f64]);
        ekf.update(&[x_true[0].sin()]);

        let p = ekf.covariance();
        assert!(
            p.data[0][0] > 0.0,
            "P[0,0] must be positive: {}",
            p.data[0][0]
        );
        assert!(
            p.data[1][1] > 0.0,
            "P[1,1] must be positive: {}",
            p.data[1][1]
        );
    }
}

/// EKF reset: state and covariance are properly reset.
#[test]
fn ekf_reset_restores_initial_state() {
    let q = Matrix {
        data: [[1e-4f64, 0.0], [0.0, 1e-3]],
    };
    let r = Matrix { data: [[0.05f64]] };
    let p0 = Matrix {
        data: [[2.0f64, 0.0], [0.0, 2.0]],
    };

    let mut ekf = Ekf::new(
        q,
        r,
        pendulum_f,
        pendulum_f_jac,
        pendulum_h,
        pendulum_h_jac,
        [0.3f64, 0.1],
        p0,
    );

    for _ in 0..50 {
        ekf.predict(&[0.0f64]);
        ekf.update(&[0.3f64.sin()]);
    }

    let x_new = [0.0f64, 0.0];
    let p_new = Matrix {
        data: [[5.0f64, 0.0], [0.0, 5.0]],
    };
    ekf.reset(x_new, p_new);

    assert!((ekf.state()[0] - 0.0).abs() < 1e-10, "State not reset");
    assert!(
        (ekf.covariance().data[0][0] - 5.0).abs() < 1e-10,
        "Cov not reset"
    );
}
