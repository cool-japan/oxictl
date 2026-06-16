# OxiCtl — Pure Rust Real-Time Control Systems Framework

[![Crates.io](https://img.shields.io/crates/v/oxictl.svg)](https://crates.io/crates/oxictl)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-nightly-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-2909%20passing-brightgreen.svg)]()

> Comprehensive, `no_std`-compatible control systems framework for embedded robotics and industrial automation.
> Pure Rust — no C/Fortran dependencies.

## Status

| Metric | Value |
|--------|-------|
| Version | 0.1.1 |
| Tests | 2,909 passing |
| Lines of Code | 102,351 |
| Rust Files | 610 |
| Clippy Warnings | 0 |
| Release Date | 2026-06-16 |

## Features

- **`no_std` compatible** — Core, PID, estimator, motor, and most modules run on bare metal with zero heap allocation (uses `heapless` for fixed-size buffers, `libm` for math)
- **Pure Rust** — No C/Fortran dependencies; no `openblas`, no `bindgen`
- **Generic scalar** — All algorithms are generic over `ControlScalar` (`f32` or `f64`); choose precision per use-case
- **No `unwrap()`** — All public APIs return `Result<_, E>` or `Option<_>`; safe for safety-critical embedded targets
- **Industrial protocols** — Modbus RTU/TCP, CANopen NMT/SDO/PDO/OD, EtherCAT master/slave/DC
- **40+ control domains** — PID, LQR, H∞, MPC (8 variants), FOC, MRAC, ILC, Koopman, geometric, passivity, fuzzy, neural, and more

## Modules

| Module | Feature Flag | Description |
|--------|-------------|-------------|
| `core` | *(always)* | `ControlScalar` trait, `Matrix<S,R,C>`, state-space, transfer functions, linearization, Bode/Nyquist/root locus, Butterworth/Chebyshev/FIR/moving-average/median filters, LMS/NLMS/VSLMS/RLS/APA adaptive filters |
| `pid` | `pid` | Standard PID (2-DOF), cascade, auto-tune (relay feedback), gain scheduling, incremental, anti-windup, bumpless transfer, fractional FOPID (GL operator, Tustin) |
| `state_feedback` | `state_feedback` | LQR (DARE), H∞, pole placement, output feedback, servo, ADRC, ISMC, backstepping, super-twisting SMC, terminal SMC, prescribed-time control, model-free control |
| `estimator` | `estimator` | KF, EKF, UKF, information filter, sqrt-KF, ensemble KF, marginalized particle filter, RTS smoother, fixed-interval smoother, batch ML, EM, Huber KF, variational Bayes, Cauchy estimator |
| `mpc` | `mpc` | Linear, economic, tracking, tube, robust (min-max), stochastic (scenario), multi-objective, multi-stage MPC, MHE, MPPI |
| `motor` | `motor` | FOC (dq current, speed, sensorless back-EMF, MTPA, DTC, overmodulation, direct thrust), BLDC six-step, stepper S-curve, SRM model, encoder, PMSM/induction param_id |
| `adaptive` | `adaptive` | MRAC (MIT rule + Lyapunov), gain scheduling, self-tuning regulator (STR), adaptive KF |
| `trajectory` | `trajectory` | Bezier (de Casteljau), clothoid, B-spline (de Boor), polynomial (min-jerk/snap), RRT/RRT*, Dubins paths, time-optimal TOTP (bang-bang) |
| `sim` | `sim` | Thermal, DC motor, nonlinear pendulum, three-tank hydraulic, quadrotor 6-DOF, cart-pole (Lagrangian), 2-DOF robotic arm, Thevenin battery, PEMFC fuel cell, hybrid energy storage, kinematic/dynamic bicycle, differential drive, vehicle platoon |
| `safety` | `safety` | Watchdog, fault handler, range/rate/timeout monitors, SIL 1-4 classification, DualChannel/TMR redundancy, fault tree diagnostics, safe state machine |
| `power` | `power` | SRF-PLL, boost/buck/buck-boost converters, THD analysis, VSI inverter with LCL filter, MPPT (P&O/InC/FracOCV), active power filter, SPWM, SVPWM 3-level |
| `scheduler` | `scheduler` | Fixed-rate task, multi-rate + priority scheduler, task timing (EMA), deadline monitor |
| `kinematics` | `kinematics` | Forward/inverse (geometric Pieper + Levenberg-Marquardt numerical), Jacobian, SCARA 4-DOF, 6-DOF dynamics (inertia/Coriolis), workspace reachability analysis |
| `protocol` | `protocol` | Modbus RTU/TCP, CANopen NMT/SDO/PDO/OD/LSS, EtherCAT master/slave/DC; see `dds-*` features for RTPS 2.3/ROS2 DDS stack |
| `fuzzy` | `fuzzy` | Triangular/trapezoidal/Gaussian/sigmoid membership functions, Mamdani/Sugeno inference, CoG/MOM/bisector defuzzification, fuzzy PID |
| `optimal` | `optimal` | ODE solvers (Euler/RK4/RK45 Fehlberg), single/multiple shooting, Pontryagin principle, bang-bang control |
| `neural` | `neural` | Activations (ReLU/Swish/…), DenseLayer (Xavier init, backprop), MLP, RBF network, neural PID (3-network online adjustment) |
| `imc` | `imc` | Q-filter IMC controller, Smith predictor (compile-time dead-time), PFC (predictive functional control) |
| `flatness` | `flatness` | Quadrotor/unicycle/2-DOF manipulator flat maps, min-snap trajectory |
| `networked` | `networked` | Static/dynamic event-triggered control (Tabuada/Girard), self-triggered LQR, multi-agent average consensus, leader-following |
| `geometric` | `geometric` | SO(3) exponential/log/hat/vee, unit quaternion (SLERP), geometric PD on SO(3) (Lee 2010), SE(3) adjoint/wrench |
| `passivity` | `passivity` | Port-Hamiltonian structure (J/R/g), IDA-PBC energy shaping + damping injection, Lyapunov storage function verifier |
| `disturbance` | `disturbance` | Q-filter DOB, nonlinear DOB (NDOB/ESO-based), UDE (uncertainty & disturbance estimator) |
| `gp` | `gp` | RBF/Matern 5-2/linear kernels, exact GP regression (Cholesky), sparse GP-FITC (inducing points) |
| `allocation` | `allocation` | Weighted pseudo-inverse CA, prioritized cascaded CA, LP-based (simplex) CA |
| `ilc` | `ilc` | P-type ILC (Arimoto), D-type ILC, norm-optimal ILC |
| `navigation` | `navigation` | Wheel odometry + IMU dead reckoning, 2D EKF-SLAM (landmark-based), linear pose-graph optimization |
| `fdi` | `fdi` | Parity-space FDI, observer-based structured residuals, chi-squared test, SPRT sequential detection |
| `extremum` | `extremum` | Perturbation-based gradient ESC (1D/2D), Newton-based ESC (Hessian estimation) |
| `comm` | `comm` | Uniform/log/dynamic quantizers, Bernoulli/Markov dropout model, Padé delay approximation, finite-history delay buffer |
| `repetitive` | `repetitive` | Plug-in repetitive controller, modified RC (3-tap FIR Q-filter), 2-DOF PID, inversion/polynomial feedforward |
| `optim` | `optim` | PSO (particle swarm, LCG-based), genetic algorithm (tournament/crossover/mutation), simulated annealing |
| `data_driven` | `data_driven` | VRFT (Campi & Savaresi), correlation-based tuning (CbT), FRIT |
| `koopman` | `koopman` | Polynomial/RBF/delay-embedding lifting functions, EDMD data-driven Koopman, greedy Koopman-MPC |
| `antiwindup` | `antiwindup` | Linear AW compensator (Teel-Praly), conditioning technique (I-PD + tracking), observer-based AW |
| `hybrid` | `hybrid` | Hybrid automaton (guards/resets/invariants), switched LTI (dwell-time stability), PWA system + controller |
| `sysid` | `sysid` | ARX (batch LS + online RLS), ARMAX (ELS), IV/Refined IV, N4SID subspace, validation (FIT%, Ljung-Box) |
| `io` | *(always)* | Kizzasi bridge, JSON state export |
| `core/fixed_point` | `fixed_point` | Q15.16/Q3.29/Q7.24 fixed-point types; `PidScalar` trait enabling PID on embedded targets without FPU |

## Quickstart

Add to `Cargo.toml`:

```toml
[dependencies]
oxictl = { version = "0.1", features = ["pid", "sim", "safety"] }
```

Basic closed-loop temperature control:

```rust
use oxictl::pid::{PidConfig, AntiWindup};
use oxictl::safety::SafetyMonitor;
use oxictl::sim::ThermalPlant;

let mut pid = PidConfig::<f64>::new(2.0, 0.5, 0.1)
    .with_limits(-100.0, 100.0)
    .with_anti_windup(AntiWindup::BackCalculation { gain: 0.1 })
    .build();

let mut plant = ThermalPlant::new(1.0, 10.0, 20.0); // tau, gain, ambient
let dt = 0.01_f64;
let setpoint = 80.0_f64;

for _ in 0..1000 {
    let temp = plant.temperature();
    let u = pid.update(setpoint, temp, dt);
    plant.step(u, dt);
}
```

Kalman filter for position tracking:

```rust
use oxictl::estimator::KalmanFilter;
use oxictl::core::Matrix;

// State: [position, velocity], Measurement: [position]
let mut kf = KalmanFilter::<f64, 2, 1, 1>::new(
    Matrix::identity(),     // F (state transition)
    Matrix::from([[1.0, 0.0]]), // H (observation)
    Matrix::identity() * 0.01, // Q (process noise)
    Matrix::identity() * 0.1,  // R (measurement noise)
    Matrix::identity(),     // P0 (initial covariance)
)?;

let measurement = Matrix::from([[3.5_f64]]);
kf.predict(None)?;
kf.update(&measurement)?;
```

## Feature Flags

| Feature | Default | Requires `std` | Description |
|---------|---------|---------------|-------------|
| `std` | yes | — | Enables `thiserror`; required by `sim` |
| `pid` | yes | no | PID family (standard, cascade, fractional, …) |
| `safety` | yes | no | Watchdog, fault handling, SIL, redundancy |
| `sim` | no | yes | Simulation plant models |
| `estimator` | no | no | Kalman filter family |
| `state_feedback` | no | no | LQR, H∞, ADRC, SMC variants (implies `estimator`) |
| `motor` | no | no | FOC, BLDC, stepper, SRM |
| `scheduler` | no | no | Fixed-rate and multi-rate task scheduler |
| `adaptive` | no | no | MRAC, gain scheduling, STR |
| `trajectory` | no | no | Path planning (Bezier, clothoid, RRT, Dubins, …) |
| `mpc` | no | no | MPC family (linear, economic, robust, MPPI, …) |
| `kinematics` | no | no | Forward/inverse kinematics, Jacobian |
| `power` | no | no | PLL, converters, MPPT, active filter |
| `protocol` | no | no | Modbus, CANopen, EtherCAT, ROS2 |
| `fuzzy` | no | no | Fuzzy logic and fuzzy PID |
| `optimal` | no | no | ODE solvers, shooting methods, Pontryagin |
| `neural` | no | no | Neural networks and neural PID |
| `imc` | no | no | IMC, Smith predictor, PFC |
| `flatness` | no | no | Differential flatness maps |
| `networked` | no | no | Event-triggered and consensus control |
| `geometric` | no | no | SO(3)/SE(3) geometric control |
| `passivity` | no | no | Port-Hamiltonian and IDA-PBC |
| `disturbance` | no | no | DOB, NDOB, UDE |
| `gp` | no | no | Gaussian process regression |
| `allocation` | no | no | Control allocation |
| `ilc` | no | no | Iterative learning control |
| `navigation` | no | no | Dead reckoning, EKF-SLAM, pose graph |
| `fdi` | no | no | Fault detection and isolation |
| `extremum` | no | no | Extremum seeking control |
| `comm` | no | no | Quantization and communication effects |
| `repetitive` | no | no | Repetitive and 2-DOF control |
| `optim` | no | no | PSO, genetic algorithm, simulated annealing |
| `data_driven` | no | no | VRFT, CbT, FRIT data-driven tuning |
| `koopman` | no | no | Koopman operator methods |
| `antiwindup` | no | no | Advanced anti-windup compensators |
| `hybrid` | no | no | Hybrid automata and switched systems |
| `sysid` | no | no | System identification (ARX, ARMAX, N4SID) |
| `fixed_point` | no | no | Q-format fixed-point arithmetic (Q15.16, Q3.29, Q7.24); `PidScalar` trait for embedded PID |
| `dds` | no | no | RTPS 2.3 wire-protocol parser/serializer (`no_std`, zero-alloc) |
| `dds-transport` | no | yes | UDPv4 transport for DDS (requires `dds`) |
| `dds-discovery` | no | yes | SPDP/SEDP participant + endpoint discovery (requires `dds-transport`) |
| `dds-stateless` | no | yes | Best-effort StatelessWriter/StatelessReader (requires `dds-discovery`) |
| `dds-stateful` | no | yes | Reliable StatefulWriter/StatefulReader with HEARTBEAT/ACKNACK (requires `dds-stateless`) |
| `dds-ros2` | no | yes | ROS2 bridge — topic naming, builtins, CDR codecs (requires `dds-stateful`) |
| `dds-api` | no | yes | High-level DDS API: `Participant`, `Publisher<T>`, `Subscription<T>`, `ServiceClient<S>`, `ServiceServer<S>`, `ActionClient<A>`, `ActionServer<A>` (requires `dds-ros2`) |

Enable all features for full functionality:

```toml
[dependencies]
oxictl = { version = "0.1", features = ["__all"] }
```

Or in `Cargo.toml` dev/example context:

```bash
cargo build --all-features
cargo nextest run --all-features
```

## `no_std` Usage

Most modules are `no_std` compatible. Only `sim` requires `std` (for file I/O in scope recorder).

```toml
[dependencies]
oxictl = { version = "0.1", default-features = false, features = ["pid", "safety", "estimator", "motor"] }
```

Internals use:
- `heapless::Vec` for fixed-size buffers (no heap allocation)
- `libm` for transcendental math (`sin`, `cos`, `sqrt`, `exp`, `ln`, …)
- `num-traits` with `libm` backend for generic float operations

## Examples

| Example | Features | Description |
|---------|----------|-------------|
| `pid_temperature` | `sim`, `pid`, `safety` | Closed-loop PID temperature regulation |
| `foc_motor` | `motor`, `sim` | Field-oriented control for PMSM |
| `kalman_tracking` | `estimator`, `sim` | Kalman filter position/velocity tracking |
| `mpc_inverted_pendulum` | `mpc`, `state_feedback`, `sim` | Linear MPC for inverted pendulum |
| `ethercat_servo` | `protocol`, `motor` | EtherCAT servo drive communication |
| `safety_watchdog` | `safety` | Watchdog + fault handler demo |
| `trajectory_planning` | `trajectory` | Bezier path planning |
| `adrc_servo` | `state_feedback` | ADRC servo with ESO-based disturbance rejection |
| `robust_mpc_pendulum` | `mpc` | Min-max robust MPC for uncertain pendulum |
| `multi_sensor_fusion` | `estimator` | Information filter fusing 3 IMUs |
| `bode_analysis` | `std` | Bode/Nyquist stability margins for lead-lag compensator |
| `fuzzy_temperature` | `fuzzy` | Mamdani fuzzy thermostat |
| `geometric_attitude` | `geometric` | Geometric PD on SO(3) with disturbance torque |
| `mppi_obstacle` | `mpc` | MPPI stochastic control for obstacle avoidance |
| `passive_control` | `passivity` | IDA-PBC for magnetic levitation |
| `super_twisting_motor` | `state_feedback` | Super-twisting SMC with matched disturbance |
| `battery_simulation` | `sim` | Thevenin battery charge/discharge cycle |
| `ekf_slam_demo` | `estimator` | 2D EKF-SLAM with 3 landmarks |
| `koopman_pendulum` | `koopman` | Koopman linearization of nonlinear pendulum |
| `switched_controller` | `hybrid` | Mode-switching control for hybrid plant |
| `antiwindup_demo` | `antiwindup` | AW compensator on saturated actuator |
| `pso_pid_tuning` | `optim` | PSO auto-tuning PID gains |
| `vrft_tuning` | `data_driven` | VRFT data-driven controller tuning |
| `fixed_point_pid` | `fixed_point`, `pid` | PID controller with Q15.16 fixed-point arithmetic, first-order plant convergence |
| `ros2_chatter` | `dds-api` | ROS2-compatible pub/sub with `std_msgs::String` on `rt/chatter`, in-process loopback |
| `ros2_imu_publisher` | `dds-api` | `sensor_msgs::Imu` publisher at simulated 100 Hz with quaternion orientation |
| `ros2_twist_subscriber` | `dds-api` | `geometry_msgs::Twist` subscriber driving a simulated unicycle model |

Run any example:

```bash
cargo run --all-features --example pid_temperature
cargo run --features "state_feedback" --example adrc_servo
```

## Protocol Support

### Modbus RTU/TCP

```rust
use oxictl::protocol::modbus::{RtuMaster, TcpSession, RegisterBank};
```

### CANopen

```rust
use oxictl::protocol::canopen::{NmtState, SdoServer, PdoMapper, ObjectDictionary};
```

### EtherCAT

```rust
use oxictl::protocol::ethercat::{EtherCatMaster, SlaveConfig, DcConfig, MailboxProtocol};
```

### ROS2 CDR

```rust
use oxictl::protocol::ros2::{CdrSerializer, QosProfile, SpscTopic};
```

### DDS / RTPS 2.3 (ROS2-compatible)

```rust
use oxictl::protocol::dds::api::{Participant, create_client, create_server};
use oxictl::protocol::dds::discovery::qos_profile::QosProfile;
use oxictl::protocol::dds::ros2::msgs::std_msgs::StdString;
use oxictl::protocol::dds::types::guid::GuidPrefix;

// Two participants in the same process (loopback demo)
let qos = QosProfile::ros2_default();
let mut p1 = Participant::new(0, GuidPrefix([0x01; 12]), qos)?;
let mut p2 = Participant::new(0, GuidPrefix([0x02; 12]), qos)?;

// Explicit peer registration (auto-discovery via SPDP multicast also works)
let addr2 = p2.local_metatraffic_addr()?;
p1.add_peer(addr2)?;

let pub_ = p1.create_publisher::<StdString>("rt/chatter", &qos)?;
let sub_ = p2.create_subscription::<StdString>("rt/chatter", &qos)?;
// ...publish, spin_once, take
```

## Architecture

```
oxictl/src/
  core/               ControlScalar trait, Matrix, state-space, transfer functions
  core/filters/       Butterworth, Chebyshev, FIR, moving-average, median
  core/adaptive_filters/ LMS, NLMS, VSLMS, RLS, APA
  core/frequency_domain/ Bode, Nyquist, sensitivity, root locus
  pid/                Standard, cascade, auto-tune, gain-schedule, incremental, fractional
  pid/fractional/     GL operator, PI^λD^μ, Tustin approximation
  state_feedback/     LQR, H∞, ADRC, ISMC, backstepping, SMC variants, servo
  estimator/          KF, EKF, UKF, information, sqrt-KF, ensemble, particle, smoothers
  mpc/                Linear, economic, tracking, tube, robust, stochastic, MPPI, MHE
  motor/              FOC pipeline, BLDC, stepper, SRM, encoder, param_id
  adaptive/           MRAC, gain scheduling, STR, adaptive KF
  trajectory/         Bezier, clothoid, B-spline, polynomial, RRT*, Dubins, TOTP
  sim/                Thermal, DC motor, pendulum, quadrotor, battery, bicycle, platoon
  safety/             Watchdog, monitors, SIL, redundancy, fault tree, safe state
  power/              PLL, converters, MPPT, VSI/LCL, active filter, SPWM/SVPWM
  scheduler/          Fixed-rate, multi-rate, timing, deadline monitor
  kinematics/         FK/IK, Jacobian, SCARA, 6-DOF, dynamics, workspace
  protocol/           Modbus RTU/TCP, CANopen NMT/SDO/PDO/OD, EtherCAT, ROS2
  fuzzy/              Membership functions, Mamdani/Sugeno, defuzzification, fuzzy PID
  optimal/            ODE solvers, single/multiple shooting, Pontryagin
  neural/             Activations, DenseLayer, MLP, RBF, neural PID
  imc/                IMC, Smith predictor, PFC
  flatness/           Quadrotor/unicycle/manipulator flat maps, min-snap
  networked/          Event-triggered, self-triggered, consensus
  geometric/          SO(3), quaternion, geometric PD, SE(3)
  passivity/          Port-Hamiltonian, IDA-PBC, Lyapunov verifier
  disturbance/        Q-filter DOB, NDOB, UDE
  gp/                 RBF/Matern/linear kernels, exact GP, sparse FITC
  allocation/         Weighted pseudo-inverse, prioritized, LP-based
  ilc/                P-type, D-type, norm-optimal ILC
  navigation/         Dead reckoning, EKF-SLAM, pose graph
  fdi/                Parity space, observer-based, chi-squared, SPRT
  extremum/           Gradient ESC, Newton ESC
  comm/               Quantizers, dropout, delay, Padé
  repetitive/         Plug-in RC, 2-DOF PID, feedforward
  optim/              PSO, genetic algorithm, simulated annealing
  data_driven/        VRFT, CbT, FRIT
  koopman/            Lifting functions, EDMD, Koopman-MPC
  antiwindup/         Linear AW compensator, conditioning, observer-based
  hybrid/             Hybrid automaton, switched LTI, PWA
  sysid/              ARX, ARMAX, IV, N4SID, validation
  io/                 Kizzasi bridge, JSON export
  core/fixed_point/   Q-format fixed-point types (Q15.16, Q3.29, Q7.24), PidScalar trait
  protocol/dds/       RTPS 2.3 wire protocol (no_std), UDPv4 transport, SPDP/SEDP discovery
  protocol/dds/api/   Participant, Publisher<T>, Subscription<T>, ServiceClient/Server, ActionClient/Server
  protocol/dds/ros2/  ROS2 bridge: topic naming, CDR codecs, 35 standard message types
```

## License

Apache-2.0 — Copyright COOLJAPAN OU (Team Kitasan)
