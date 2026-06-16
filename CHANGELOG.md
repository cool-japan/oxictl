# Changelog

All notable changes to oxictl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-06-16

### Added
- Phase 23: High-level DDS user API (`dds-api` feature) — `Participant`, `Publisher<T>`, `Subscription<T>`, `DdsType` trait, `Sample<T>`, `EntityIdAllocator`, `LogOwned`, `ParameterEventOwned`
- CDR string helpers promoted to public API: `ByteWriter::write_cdr_string`, `ByteCursor::read_cdr_string`
- `IncomingResult` re-exported from `discovery::` public API
- 5 integration tests in `tests/dds_api_integration/`
- Phase 22 (22.1–22.9): Complete RTPS 2.3 / DDS stack — transport, SPDP/SEDP discovery, QoS matching, stateless/stateful endpoints, ROS2 CDR bridge
- ROS2 Service layer (`dds-api` feature): `ServiceClient<S>` and `ServiceServer<S>` — ROS2-compatible request/reply over DDS; `create_client` and `create_server` factory functions; `SampleIdentity` for correlating request/response pairs
- 4 integration tests in `tests/dds_service_integration/` (add_two_ints, two_clients_no_cross_talk, server_handles_multiple_sequential, unmatched_service_no_reply)
- ROS2 Action layer (`dds-api` feature): `ActionClient<A>`, `ActionServer<A>`, `ActionHandler<A>` — ROS2 Action protocol (goal/feedback/result/cancel); `create_action_client` and `create_action_server` factory functions
- 5 integration tests in `tests/dds_action_integration/` (fibonacci_goal_accept_and_result, feedback_flows_to_client, status_array_reflects_lifecycle, cancel_goal_marks_canceling, two_action_clients_isolated)
- New ROS2 message types (`dds-api` feature): `unique_identifier_msgs::Uuid` (16-byte fixed array); `action_msgs::{GoalInfo, GoalStatus, GoalStatusArray, CancelGoal}` with goal_status constants; `example_interfaces::{AddTwoIntsRequest, AddTwoIntsResponse}` (AddTwoInts service); `example_interfaces_action::{FibonacciGoal, FibonacciResult, FibonacciFeedback}` (Fibonacci action)
- Examples: `ros2_chatter` (Publisher/Subscription with StdString on `rt/chatter`, in-process loopback via explicit `add_peer`), `ros2_imu_publisher` (Publisher/Subscription with `sensor_msgs::Imu` at simulated 100 Hz), `ros2_twist_subscriber` (Publisher/Subscription with `geometry_msgs::Twist`, integrates unicycle model), `fixed_point_pid` (PID controller with Q15.16 fixed-point arithmetic)

### Changed
- Promoted `write_param_string` / `read_param_cdr_string` CDR helpers from private to public in `byte_cursor.rs`
- Fixed-point PID support via `PidScalar` trait (`fixed_point` feature)
- Internal DDS stack cleanup and robustness improvements across RTPS transport, SPDP/SEDP discovery, stateless/stateful endpoints, CDR codec, and QoS handling
- `proptest` updated from 1.8 to 1.11

## [0.1.0] - 2026-03-09

### Added
- Core control systems framework with generic scalar trait (`ControlScalar`)
- PID controller family: standard, cascade, auto-tune, gain-scheduling, incremental, anti-windup, bumpless transfer, fractional (FOPID)
- State feedback controllers: LQR, H∞, pole placement, output feedback, ADRC, ISMC, backstepping, super-twisting SMC, terminal SMC, prescribed-time control, model-free control
- State estimators: Kalman Filter, EKF, UKF, information filter, sqrt-KF, ensemble KF, marginalized particle filter, RTS smoother, fixed-interval smoother, Huber M-estimator KF, variational Bayes filter, Cauchy estimator, EM algorithm
- Model Predictive Control: linear, economic, tracking, tube, robust, stochastic, multi-objective, multi-stage MPC, MHE, MPPI
- Motor control: FOC (dq current, speed, sensorless, MTPA, DTC, overmodulation), BLDC, stepper, SRM, parameter identification
- Adaptive control: MRAC, gain scheduling, self-tuning regulator, adaptive Kalman filter
- Trajectory planning: Bezier, clothoid, B-spline, polynomial (min-jerk/snap), RRT/RRT*, Dubins paths, time-optimal (TOTP)
- Simulation models: thermal, DC motor, nonlinear pendulum, three-tank, quadrotor (6-DOF), cart-pole, robotic arm, Thevenin battery, PEMFC fuel cell, hybrid energy storage, bicycle, differential drive, vehicle platoon
- Safety: watchdog, fault handler, monitors, SIL classification, redundancy (DualChannel/TMR), diagnostic fault tree, safe state management
- Power electronics: PLL, boost/buck converters, harmonic/THD analysis, inverter/VSI, MPPT (P&O/InC/FracOCV), active filter
- System identification: ARX, ARMAX, IV, N4SID subspace identification
- Kinematics: forward/inverse kinematics, Jacobian, SCARA, 6-DOF (geometric Pieper + Levenberg-Marquardt)
- Protocols: ROS2 CDR/QoS, Modbus RTU/TCP, CANopen NMT/SDO/PDO/OD, EtherCAT mailbox
- Frequency domain: Bode, Nyquist, sensitivity, root locus
- Fuzzy logic: membership functions, Mamdani/Sugeno inference, fuzzy PID
- Optimal control: ODE solvers (RK4/RK45), single/multiple shooting, Pontryagin
- Neural networks: activations, DenseLayer, MLP, RBF, neural PID
- IMC: Q-filter controller, Smith predictor, predictive functional control
- Geometric control: SO(3), quaternion, geometric PD (Lee 2010), SE(3) wrench transform
- Passivity-based control: port-Hamiltonian, IDA-PBC, Lyapunov storage function verifier
- Disturbance rejection: Q-filter DOB, NDOB, UDE controller
- Adaptive filters: LMS, NLMS, VSLMS, RLS, APA
- Flatness-based control: quadrotor/unicycle/manipulator flat maps, min-snap trajectory
- Networked control: event-triggered, self-triggered LQR, multi-agent consensus
- Iterative learning control (ILC): P-type, D-type, norm-optimal
- Navigation: dead reckoning, 2D EKF-SLAM, pose graph optimization
- Fault detection and isolation (FDI): parity space, observer-based, chi-squared test, SPRT
- Extremum seeking: gradient ESC (1D/2D), Newton ESC
- Communication: quantizers, packet dropout, delay buffer, Pade delay
- Repetitive control: plug-in RC, modified RC, 2-DOF PID, feedforward
- Optimization: PSO, genetic algorithm, simulated annealing (pure Rust LCG)
- Data-driven control: VRFT, CbT, FRIT
- Koopman operator: polynomial/RBF/delay lifting, EDMD, Koopman-MPC
- Anti-windup: linear compensator, back-calculation, observer-based
- Hybrid systems: hybrid automaton, switched LTI, PWA systems
- Kizzasi I/O bridge
- `no_std` compatible throughout (using `libm`, `heapless`, `core`)

[0.1.0]: https://github.com/cool-japan/oxictl/releases/tag/v0.1.0
[0.1.1]: https://github.com/cool-japan/oxictl/compare/v0.1.0...v0.1.1
