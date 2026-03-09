# Changelog

All notable changes to oxictl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
