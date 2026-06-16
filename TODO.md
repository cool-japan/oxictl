# OxiCtl Development TODO

**Current Release: v0.1.1 — 2026-06-16**
**Status: All phases complete (321 items ✅)**

## Phase 1: PID + Safety + Simulation ✅ COMPLETE

- [x] Project scaffold (Cargo.toml, directory structure)
- [x] `core/scalar.rs` - ControlScalar trait (f32/f64)
- [x] `core/traits.rs` - Controller, Estimator, Plant traits
- [x] `core/signal.rs` - Setpoint, Feedback, ControlOutput
- [x] `core/saturation.rs` - OutputLimiter, RateLimiter
- [x] `pid/standard.rs` - PID controller with 2-DOF
- [x] `pid/anti_windup.rs` - Clamping and back-calculation
- [x] `pid/derivative_filter.rs` - IIR low-pass filter
- [x] `safety/watchdog.rs` - Watchdog timer
- [x] `safety/fault.rs` - FaultSeverity, FaultDef, FaultEvent
- [x] `safety/handler.rs` - FaultHandler (heapless)
- [x] `safety/monitor/` - Range, Rate, Timeout monitors
- [x] `safety/mod.rs` - SafetyMonitor aggregator
- [x] `sim/thermal_sim.rs` - First-order thermal plant
- [x] `sim/scope.rs` - Waveform recorder
- [x] `examples/pid_temperature.rs` - Closed-loop demo
- [x] Verification: 93 tests, clippy clean, no_std builds

## Phase 2: State Estimation + Advanced PID ✅ COMPLETE

- [x] `core/matrix.rs` - Fixed-size Matrix<S,R,C> with matmul, inv, etc.
- [x] `estimator/kalman.rs` - Linear Kalman Filter (N,M,I dims)
- [x] `estimator/ekf.rs` - Extended Kalman Filter with fn-ptr Jacobians
- [x] `estimator/complementary.rs` - Complementary filter (1D and 2D)
- [x] `pid/cascade.rs` - Cascade PID (outer/inner loops)
- [x] `pid/auto_tune.rs` - Relay feedback auto-tuner (Åström-Hägglund) + ZN rules
- [x] `pid/gain_schedule.rs` - Linear interpolated gain scheduling
- [x] `pid/incremental.rs` - Incremental (velocity-form) PID

## Phase 3: State-Space + LQR ✅ COMPLETE

- [x] `core/transfer_fn.rs` - IIR TransferFn + Biquad (LP/HP/notch)
- [x] `core/state_space.rs` - Discrete/continuous state-space, Euler+Tustin discretization
- [x] `state_feedback/lqr.rs` - DARE solver + LQR optimal controller

## Phase 4: Motor Control ✅ COMPLETE

- [x] `motor/transform/clarke.rs` - Clarke transform (abc→αβ)
- [x] `motor/transform/park.rs` - Park transform (αβ→dq) + inverse
- [x] `motor/transform/svpwm.rs` - Space vector PWM + SPWM
- [x] `motor/foc/current_loop.rs` - d/q axis PI current controllers
- [x] `motor/foc/speed_loop.rs` - Speed PI controller
- [x] `motor/foc/controller.rs` - Complete FOC pipeline
- [x] `motor/bldc/six_step.rs` - Hall-sensor six-step commutation
- [x] `motor/stepper/s_curve.rs` - Jerk-limited S-curve profile
- [x] `motor/encoder/incremental.rs` - Encoder with LP-filtered velocity

## Phase 5: Scheduler ✅ COMPLETE

- [x] `scheduler/fixed_rate.rs` - Period-accumulator fixed-rate task
- [x] `scheduler/multi_rate.rs` - Multi-rate + priority scheduler with overrun detection

## Extended Modules (Beyond Phase 5) ✅

- [x] `mpc/linear_mpc.rs` - LinearMpc<S,N,I,H> (gradient projection QP, warm-start)
- [x] `kinematics/forward.rs` - Transform2D/3D rigid body transforms
- [x] `kinematics/jacobian.rs` - Jacobian2R (2-DOF planar, pseudo-inverse, IK step)
- [x] `kinematics/serial/scara.rs` - SCARA 4-DOF FK/IK/Jacobian
- [x] `trajectory/bezier.rs` - BezierCurve/BezierPath (de Casteljau, arc length)
- [x] `motor/foc/sensorless.rs` - BackEmfObserver (flux integration, speed estimation)
- [x] `scheduler/timing.rs` - TaskTiming (EMA), DeadlineMonitor
- [x] `power/pll.rs` - SRF-PLL (phase-locked loop for grid/motor)
- [x] `power/converter/boost.rs` - Boost converter average model + PI controller
- [x] `power/converter/buck.rs` - Buck converter average model + PI controller
- [x] `state_feedback/robust/hinf.rs` - H∞ state feedback via modified DARE

## Phase 6: Advanced Control Expansion ✅ COMPLETE (2026-03-08)

### Bug Fixes
- [x] `motor/foc/overmodulation.rs` - Fixed FRAC_2_PI constant (was hardcoded literal)
- [x] `motor/encoder/sincos.rs` - Fixed velocity estimation (sign error → correct cross-product formula)
- [x] `motor/foc/load_observer.rs` - Fixed sign error in tau_load_dot (unstable → stable ESO)
- [x] `mpc/tracking_mpc.rs` - Fixed gradient computation via central-difference on total_cost()
- [x] `power/harmonic/thd.rs` - Fixed fundamental bin selection (now uses correct frequency bin)
- [x] `state_feedback/servo.rs` - Fixed prefilter design (correct discrete-time Nu formula)
- [x] `trajectory/clothoid.rs` - Fixed Fresnel small-t tolerance (y ≈ s³/6 leading term)

### MPC Expansion
- [x] `mpc/moving_horizon_estimator.rs` - MHE with sliding window, Gauss-Newton, arrival cost
- [x] `mpc/robust_mpc.rs` - Min-max robust MPC with polytopic uncertainty, constraint tightening
- [x] `mpc/stochastic_mpc.rs` - Scenario-based chance-constrained MPC, SAA, LCG scenarios
- [x] `mpc/multi_objective_mpc.rs` - Pareto-optimal MPC, weighted sum + ε-constraint methods
- [x] `mpc/multi_stage_mpc.rs` - Scenario tree MPC, non-anticipativity, CVaR risk measure

### Motor Module Expansion
- [x] `motor/param_id/pmsm_id.rs` - Online PMSM RLS parameter identification (Rs, Ld, Lq, λ_pm)
- [x] `motor/param_id/induction_id.rs` - MRAS induction motor ID (rotor time constant, Rs)
- [x] `motor/foc/mtpa.rs` - MTPA lookup table + linear interpolation for PMSM
- [x] `motor/foc/direct_thrust.rs` - Direct Thrust Control for linear PMSM
- [x] `motor/model/srm.rs` - SRM nonlinear torque/flux model, phase switching, ripple

### Estimator Expansion
- [x] `estimator/information_filter.rs` - Information-form KF, multi-sensor additive fusion
- [x] `estimator/sqrt_kalman.rs` - Square-root KF (Cholesky-factored P, Givens downdate)
- [x] `estimator/ensemble_kf.rs` - Ensemble KF, perturbed-observation update, LCG+Box-Muller
- [x] `estimator/marginalized_particle.rs` - Rao-Blackwellized particle filter, systematic resampling

### State Feedback Expansion
- [x] `state_feedback/adrc.rs` - ADRC with ESO, bandwidth-parameterized, 1st/2nd order
- [x] `state_feedback/integral_sliding_mode.rs` - ISMC, reaching-phase elimination, sat/sign laws
- [x] `state_feedback/model_free_control.rs` - Ultra-local model, iPID, algebraic F estimator
- [x] `state_feedback/backstepping.rs` - Recursive backstepping, virtual controls, CLF stability

### Adaptive Control
- [x] `adaptive/mrac_second_order.rs` - MRAC (MIT rule + Lyapunov), parameter projection
- [x] `adaptive/gain_scheduling.rs` - Scheduled PID with interpolation, anti-windup, hysteresis
- [x] `adaptive/self_tuning_regulator.rs` - STR with online RLS + minimum variance law

### Trajectory Expansion
- [x] `trajectory/bspline.rs` - B-spline (de Boor), velocity/acceleration, clamped uniform
- [x] `trajectory/polynomial.rs` - Min-jerk (5th), min-snap (7th), multi-segment stitching

### Simulation Expansion
- [x] `sim/nonlinear_pendulum.rs` - Full nonlinear pendulum + pendulum-on-cart (RK4)
- [x] `sim/dc_motor.rs` - DC motor plant (armature + mechanical), RK4, steady-state
- [x] `sim/three_tank.rs` - Three-tank hydraulic system (Torricelli), RK4, constraints

### Tests & Examples
- [x] `tests/state_feedback_validation/` - ADRC vs LQR, ISMC robustness, H∞, pole placement (15 tests)
- [x] `tests/mpc_advanced_validation/` - Robust MPC, MHE, stochastic, multi-objective (22 tests)
- [x] `tests/motor_advanced_validation/` - MTPA, SRM, PMSM ID, info filter (19 tests)
- [x] `tests/integration_extended/` - FOC+MTPA+MHE, ADRC disturbance, PLL pipeline (9 tests)
- [x] `examples/adrc_servo.rs` - ADRC servo position control with torque disturbance
- [x] `examples/robust_mpc_pendulum.rs` - Robust MPC for uncertain inverted pendulum
- [x] `examples/multi_sensor_fusion.rs` - Information filter fusing 3 IMUs

## Phase 7: Protocol, Filters & Kinematics ✅ COMPLETE (2026-03-08)

### Modbus RTU/TCP Protocol
- [x] `protocol/modbus/pdu.rs` - Request/Response encoding (FC01-FC06, FC16), exception handling
- [x] `protocol/modbus/rtu.rs` - RTU framing, CRC16, RtuMaster state machine, silent interval
- [x] `protocol/modbus/tcp.rs` - MBAP header, TcpSession with transaction counter
- [x] `protocol/modbus/register_map.rs` - RegisterBank<N>, CoilBank<N, BYTES> with bit-packing

### Signal Processing Filters
- [x] `core/filters/butterworth.rs` - Nth-order LP/HP/BP via bilinear transform + pre-warping
- [x] `core/filters/chebyshev.rs` - Chebyshev Type I/II, equiripple passband/stopband
- [x] `core/filters/moving_average.rs` - MovingAverage, EMA, MovingRms, MovingVariance
- [x] `core/filters/median_filter.rs` - MedianFilter<N>, MedianOf3, median3() network
- [x] `core/filters/fir.rs` - FirFilter, windowed-sinc design (Hamming/Hanning/Blackman)

### 6-DOF Inverse Kinematics
- [x] `kinematics/inverse/geometric_6dof.rs` - Pieper closed-form IK, 8-solution elbow/shoulder/wrist configs
- [x] `kinematics/inverse/numerical_ik.rs` - Levenberg-Marquardt IK with null-space joint-limit avoidance
- [x] `kinematics/workspace.rs` - DH parameters, FK from DH, workspace reachability analysis

## Phase 8: CANopen, Plant Models & Extended Safety ✅ COMPLETE (2026-03-08)

### CANopen Node Protocol
- [x] `protocol/canopen/nmt.rs` - NMT state machine, heartbeat producer, command processing
- [x] `protocol/canopen/object_dict.rs` - Static OD with DataType, AccessType, OdValue
- [x] `protocol/canopen/sdo.rs` - SDO server, expedited transfer, abort codes
- [x] `protocol/canopen/pdo.rs` - TPDO/RPDO mapping, pack/unpack, event timer

### Nonlinear Plant Models
- [x] `sim/quadrotor.rs` - 6-DOF quadrotor (Newton-Euler, ZYX rotation, RK4)
- [x] `sim/cart_pole.rs` - Cart-pole (Lagrangian EOM, Cramer's rule, RK4)
- [x] `sim/robotic_arm.rs` - 2-DOF planar arm (full M/C/G dynamics, RK4)
- [x] `core/linearization.rs` - Numerical Jacobian (central diff), ZOH discretize, controllability rank

### Extended Safety (IEC 61508 inspired)
- [x] `safety/sil.rs` - SIL 1-4 classification, PFD/PFH ranges, SafetyRequirement
- [x] `safety/redundancy_ext.rs` - DualChannel 1oo2, TMR 2oo3 voting, ComparatorConfig
- [x] `safety/diagnostic.rs` - DiagnosticCoverage, FaultTree (AND/OR gates), PFD computation
- [x] `safety/safe_state_ext.rs` - SafeStateMachine with latching, keyed reset, fault escalation

## Phase 9: Frequency Domain, Fuzzy Logic & Optimal Control ✅ COMPLETE (2026-03-08)

### Frequency Domain Analysis
- [x] `core/frequency_domain/bode.rs` - Bode plot, gain/phase margins, crossover frequencies
- [x] `core/frequency_domain/nyquist.rs` - Nyquist curve, winding number, distance to critical
- [x] `core/frequency_domain/sensitivity.rs` - Sensitivity/complementary sensitivity, H∞ norm, bandwidth
- [x] `core/frequency_domain/root_locus.rs` - Root locus, Cardano solver, Durand-Kerner, stability check

### Fuzzy Logic Control
- [x] `fuzzy/membership.rs` - Triangular, Trapezoidal, Gaussian, Sigmoid, Bell membership functions
- [x] `fuzzy/rule_base.rs` - LinguisticVar, FuzzyRule, RuleBase, T-norm (Product/Min)
- [x] `fuzzy/inference.rs` - Mamdani and Sugeno (TSK) inference engines
- [x] `fuzzy/defuzzify.rs` - CoG, MOM, bisector, largest/smallest of maxima
- [x] `fuzzy/fuzzy_pid.rs` - Fuzzy-PID hybrid with online gain adjustment

### Optimal Control (Shooting Methods)
- [x] `optimal/ode_solver.rs` - Euler, RK4, RK45 adaptive (Fehlberg), integrate utility
- [x] `optimal/single_shooting.rs` - Direct single shooting, gradient descent, Armijo line search
- [x] `optimal/multiple_shooting.rs` - Multiple shooting, continuity penalty, joint gradient
- [x] `optimal/pontryagin.rs` - Hamiltonian, co-state dynamics, bang-bang control, adjoint gradient

## Phase 10: Neural Networks, IMC/PFC & Validation ✅ COMPLETE (2026-03-08)

### Neural Network Control
- [x] `neural/activations.rs` - ReLU, LeakyReLU, Sigmoid, Tanh, Swish, Linear + derivatives
- [x] `neural/layer.rs` - DenseLayer with Xavier init, forward/backward, gradient accumulation
- [x] `neural/network.rs` - MLP (2-layer), MSE backprop, mini-batch training, weight export
- [x] `neural/rbf_network.rs` - RBF network with Gaussian kernels, online output weight training
- [x] `neural/neural_pid.rs` - Neural PID with 3 RBF networks adjusting Kp/Ki/Kd online

### Internal Model Control (IMC) & Predictive Functional Control (PFC)
- [x] `imc/imc_controller.rs` - IMC with Q-filter, model-mismatch feedback, saturation
- [x] `imc/smith_predictor.rs` - Smith predictor with compile-time dead-time, PI primary controller
- [x] `imc/pfc.rs` - PFC with precomputed G, free response rollout, first-order reference trajectory

### Integration Tests (Phase 9 Validation)
- [x] `tests/frequency_domain_validation/bode_margins.rs` - Gain/phase margins, sensitivity S+T=1
- [x] `tests/fuzzy_validation/fuzzy_inference.rs` - Mamdani/Sugeno output bounds, partition of unity
- [x] `tests/optimal_validation/shooting_convergence.rs` - Cost descent, continuity, bang-bang law

### Examples
- [x] `examples/bode_analysis.rs` - Bode/Nyquist stability analysis with lead-lag compensator
- [x] `examples/fuzzy_temperature.rs` - Mamdani fuzzy thermostat simulation
- [x] `examples/optimal_trajectory.rs` - Single shooting minimum-energy double integrator

## Phase 11: System ID, Power Electronics & Kalman Smoother ✅ COMPLETE (2026-03-08)

### System Identification
- [x] `sysid/arx.rs` - ARX model: batch LS + online RLS, FIT% metric, simulate
- [x] `sysid/armax.rs` - ARMAX via Extended Least Squares (ELS), iterative convergence
- [x] `sysid/instrumental_variables.rs` - IV and Refined IV for noise-robust identification
- [x] `sysid/validation.rs` - FIT%, autocorrelation, Ljung-Box whiteness, cross-correlation
- [x] `sysid/subspace.rs` - Simplified N4SID subspace identification (QR-based)

### Extended Power Electronics
- [x] `power/inverter/vsi.rs` - VSI with LCL filter (6-state), RK4, d/q PI with feed-forward
- [x] `power/mppt.rs` - P&O, InC, Fractional OCV MPPT + single-diode PV cell model
- [x] `power/active_filter.rs` - Sliding DFT harmonic detector, APF current reference, hysteresis controller

### Kalman Smoother & Batch Estimation
- [x] `estimator/rts_smoother.rs` - RTS backward smoother, covariance guaranteed ≤ filter
- [x] `estimator/fixed_interval_smoother.rs` - Bryson-Frazier two-filter smoother (BIF backward pass)
- [x] `estimator/batch_ml.rs` - ML estimation of Q/R via NLL gradient descent + KF
- [x] `estimator/em_algorithm.rs` - EM algorithm for full state-space model learning (A,C,Q,R)

## Phase 12: Flatness, Fractional PID & Networked Control ✅ COMPLETE (2026-03-08)

### Differential Flatness
- [x] `flatness/quadrotor_flat.rs` - Quadrotor inverse flat map, min-snap trajectory, FlatState
- [x] `flatness/unicycle_flat.rs` - Unicycle flat map (v,ω from ẋ,ẏ), path tracker with lookahead
- [x] `flatness/manipulator_flat.rs` - 2-DOF arm flat IK, Jacobian velocity/acceleration

### Fractional-Order PID
- [x] `pid/fractional/grunwald.rs` - Grünwald-Letnikov D^α operator, FracIntegrator/Differentiator
- [x] `pid/fractional/fopid.rs` - PI^λD^μ controller, conditional anti-windup, grid-search auto-tune
- [x] `pid/fractional/tustin_approx.rs` - Tustin s^α IIR approximation, num/den coefficient design

### Networked & Event-Triggered Control
- [x] `networked/event_triggered.rs` - Static (Tabuada), dynamic (Girard) triggers, ZOH, MIET
- [x] `networked/self_triggered.rs` - ISS-based next-trigger precomputation, self-triggered LQR
- [x] `networked/consensus.rs` - AgentGraph/Laplacian, average consensus, leader-following, dist. GD

## Phase 13: Geometric Control, Passivity & MPPI ✅ COMPLETE (2026-03-08)

### Geometric Control (Lie Groups)
- [x] `geometric/so3.rs` - SO(3) rotation group, exponential/logarithm map, hat/vee operators
- [x] `geometric/quaternion.rs` - Unit quaternion, SLERP, quaternion multiplication, to/from rotation matrix
- [x] `geometric/geometric_pd.rs` - Geometric PD controller on SO(3) (Lee 2010), attitude error on manifold
- [x] `geometric/se3.rs` - SE(3) wrench transform, adjoint map, body/spatial force transformation

### Passivity-Based Control
- [x] `passivity/port_hamiltonian.rs` - Port-Hamiltonian system structure, J/R/g matrices, passivity output
- [x] `passivity/ida_pbc.rs` - IDA-PBC energy shaping + damping injection, desired Hamiltonian assignment
- [x] `passivity/lyapunov.rs` - Storage function Lyapunov verifier, supply rate check, passivity certificate

### MPPI (Model Predictive Path Integral)
- [x] `mpc/mppi.rs` - MPPI stochastic optimal control, importance-weighted trajectory rollout, temperature tuning

## Phase 14: Robust Estimation, Advanced Trajectory & Benchmarks ✅ COMPLETE (2026-03-08)

### Robust Estimators
- [x] `estimator/huber_kalman.rs` - Huber M-estimator robust KF (iterative reweighted update, outlier rejection)
- [x] `estimator/variational_bayes_filter.rs` - Variational Bayes adaptive noise filter (online Q/R estimation)
- [x] `estimator/cauchy_estimator.rs` - Cauchy heavy-tail scalar estimator (closed-form update, fat-tailed likelihood)

### Advanced Trajectory Planning
- [x] `trajectory/rrt.rs` - RRT* deterministic path planning via LCG (asymptotically optimal, rewiring)
- [x] `trajectory/dubins.rs` - 6-type Dubins paths (LSL/RSR/LSR/RSL/RLR/LRL, shortest path selection)
- [x] `trajectory/time_optimal.rs` - Bang-bang time-optimal trajectory planning (TOTP, phase-plane method)

### Benchmarks
- [x] `benches/control_bench.rs` - Criterion benchmarks: PID step, Kalman update, Butterworth filter, Bezier eval

### Examples
- [x] `examples/geometric_attitude.rs` - Geometric PD attitude control on SO(3) with disturbance torque
- [x] `examples/mppi_obstacle.rs` - MPPI stochastic control for obstacle avoidance
- [x] `examples/passive_control.rs` - IDA-PBC passivity-based control for magnetic levitation

## Verification: 2501/2501 tests, clippy clean, 75,280 SLoC (402 Rust files) ✅

## Phase 16: Control Allocation, Vehicle Simulation & Iterative Learning Control ✅ COMPLETE (2026-03-08)

### Control Allocation
- [x] src/allocation/weighted_pseudo.rs — Weighted pseudo-inverse allocation with actuator limits and redistribution
- [x] src/allocation/prioritized.rs — Priority-based cascaded control allocation
- [x] src/allocation/linear_programming.rs — LP-based optimal allocation (simplex)
- [x] src/allocation/mod.rs

### Ground Vehicle Simulation
- [x] src/sim/bicycle.rs — Kinematic/dynamic bicycle model (lateral dynamics, slip angles)
- [x] src/sim/differential_drive.rs — Differential drive robot (velocity kinematics, odometry)
- [x] src/sim/vehicle_platoon.rs — N-vehicle platoon (spacing policy, string stability)

### Iterative Learning Control
- [x] src/ilc/p_type_ilc.rs — P-type ILC (Arimoto learning from repetition)
- [x] src/ilc/d_type_ilc.rs — D-type ILC (derivative-based learning)
- [x] src/ilc/norm_optimal_ilc.rs — Norm-optimal ILC (quadratic optimization)
- [x] src/ilc/mod.rs

### Examples
- [x] examples/control_allocation_uav.rs — Over-actuated quadrotor allocation
- [x] examples/bicycle_mpc.rs — Bicycle model with MPC path tracking
- [x] examples/ilc_repetitive.rs — ILC on a repetitive pick-and-place task

## Phase 15: Disturbance Observers, Adaptive Filters & Gaussian Processes ✅ COMPLETE (2026-03-08)

### Disturbance Observers
- [x] `src/disturbance/dob.rs` - Q-filter Disturbance Observer (bandwidth-parameterized)
- [x] `src/disturbance/ndob.rs` - Nonlinear DOB (ESO-based)
- [x] `src/disturbance/ude.rs` - Uncertainty and Disturbance Estimator
- [x] `src/disturbance/mod.rs`

### Adaptive Filters
- [x] `src/core/adaptive_filters/lms.rs` - LMS, NLMS, variable step-size
- [x] `src/core/adaptive_filters/rls.rs` - RLS with forgetting factor + sqrt-RLS
- [x] `src/core/adaptive_filters/affine_projection.rs` - APA algorithm
- [x] `src/core/adaptive_filters/mod.rs`

### Gaussian Process Regression
- [x] `src/gp/kernel.rs` - RBF, Matern 5/2, polynomial kernels
- [x] `src/gp/gp_regression.rs` - Exact GP with Cholesky, predictive mean/var, NLL
- [x] `src/gp/sparse_gp.rs` - Sparse GP with inducing points (FITC)
- [x] `src/gp/mod.rs`

### Examples
- [x] `examples/dob_motor.rs` - DOB for load torque rejection
- [x] `examples/lms_noise_cancel.rs` - LMS noise cancellation
- [x] `examples/gp_learning.rs` - GP regression for system learning

## Phase 17: Advanced SMC, Battery/Energy Models & Navigation ✅ COMPLETE (2026-03-08)

### Advanced Sliding Mode Control
- [x] src/state_feedback/super_twisting.rs — Super-twisting algorithm (2nd-order SMC, Levant 1993)
- [x] src/state_feedback/terminal_smc.rs — Terminal SMC with finite-time convergence
- [x] src/state_feedback/prescribed_time.rs — Prescribed-time control (time-varying high-gain)

### Battery & Energy Simulation
- [x] src/sim/battery.rs — Thevenin RC battery model (SOC, OCV, thermal coupling)
- [x] src/sim/fuel_cell.rs — PEMFC polymer electrolyte fuel cell model
- [x] src/sim/energy_storage.rs — Supercapacitor + hybrid battery/SC storage

### Navigation & Pose Estimation
- [x] src/navigation/dead_reckoning.rs — Wheel odometry + IMU fusion
- [x] src/navigation/ekf_slam_2d.rs — 2D EKF-SLAM (landmark-based)
- [x] src/navigation/pose_graph.rs — Linear pose chain least-squares

### Examples
- [x] examples/super_twisting_motor.rs — Super-twisting control rejecting matched disturbance
- [x] examples/battery_simulation.rs — Thevenin battery charge/discharge cycle
- [x] examples/ekf_slam_demo.rs — 2D EKF-SLAM with 3 landmarks

## Phase 18: Fault Detection, Extremum Seeking & Communication Effects ✅ COMPLETE (2026-03-08)

### Fault Detection & Isolation (FDI)
- [x] src/fdi/parity_space.rs — Parity vector FDI (redundancy relations from system model)
- [x] src/fdi/observer_fdi.rs — Observer-based FDI with structured residuals
- [x] src/fdi/hypothesis_test.rs — χ² test and SPRT sequential fault detection
- [x] src/fdi/mod.rs

### Extremum Seeking Control
- [x] src/extremum/gradient_esc.rs — Perturbation-based ESC (sinusoidal probing + HPF + integrator)
- [x] src/extremum/newton_esc.rs — Newton-based ESC (Hessian estimation, faster convergence)
- [x] src/extremum/mod.rs

### Quantization & Communication Effects
- [x] src/comm/quantizer.rs — Uniform and logarithmic quantizers, dynamic quantization
- [x] src/comm/packet_dropout.rs — Markov-chain dropout model, dropout-robust hold
- [x] src/comm/time_delay.rs — Pade approximation, finite-history delay buffer
- [x] src/comm/mod.rs

### Examples
- [x] examples/fdi_motor.rs — Observer-based fault detection on DC motor
- [x] examples/extremum_seeking.rs — ESC converging to peak of unknown static map
- [x] examples/quantized_control.rs — Quantization effects on PID closed-loop

## Phase 19: Repetitive Control, Bioinspired Optimization & Data-Driven Control ✅ COMPLETE (2026-03-08)

### Repetitive Control & 2-DOF
- [x] src/repetitive/repetitive_controller.rs — Plug-in repetitive controller for periodic disturbance rejection
- [x] src/repetitive/two_dof_controller.rs — 2-DOF controller (prefilter + feedback, optimal design)
- [x] src/repetitive/feedforward.rs — Dynamic inversion feedforward filter
- [x] src/repetitive/mod.rs

### Bioinspired Optimization
- [x] src/optim/particle_swarm.rs — Particle Swarm Optimization (PSO, LCG-based)
- [x] src/optim/genetic_algorithm.rs — Genetic Algorithm (tournament selection, crossover, mutation)
- [x] src/optim/simulated_annealing.rs — Simulated Annealing for continuous parameter optimization
- [x] src/optim/mod.rs

### Data-Driven Control
- [x] src/data_driven/vrft.rs — Virtual Reference Feedback Tuning (Campi & Savaresi 2002)
- [x] src/data_driven/correlation_tuning.rs — Correlation-based tuning (CbT)
- [x] src/data_driven/frit.rs — Fictitious Reference Iterative Tuning (FRIT)
- [x] src/data_driven/mod.rs

### Examples
- [x] examples/repetitive_control.rs — Repetitive rejection of periodic disturbance on servo
- [x] examples/pso_pid_tuning.rs — PSO auto-tuning PID gains on first-order plant
- [x] examples/vrft_tuning.rs — VRFT data-driven controller tuning

## Phase 20: Koopman Operators, Anti-Windup & Hybrid Systems ✅ COMPLETE (2026-03-08)

### Koopman Operator Methods
- [x] src/koopman/lifting_functions.rs — Polynomial, RBF, delay-embedding lifting maps
- [x] src/koopman/edmd.rs — Extended Dynamic Mode Decomposition (data-driven Koopman)
- [x] src/koopman/koopman_mpc.rs — Koopman-based linear MPC for nonlinear systems
- [x] src/koopman/mod.rs

### Anti-Windup for General Systems
- [x] src/antiwindup/aw_compensator.rs — Linear anti-windup compensator (quadratic recovery)
- [x] src/antiwindup/conditioning_technique.rs — Conditioning technique (I-PD + tracking)
- [x] src/antiwindup/observer_aw.rs — Observer-based anti-windup for output feedback
- [x] src/antiwindup/mod.rs

### Hybrid Automata & Switched Systems
- [x] src/hybrid/automaton.rs — Hybrid automaton (modes, guards, resets, invariants)
- [x] src/hybrid/switched_lti.rs — Switched LTI with dwell-time stability
- [x] src/hybrid/piecewise_affine.rs — Piecewise affine (PWA) system and controller
- [x] src/hybrid/mod.rs

### Examples
- [x] examples/koopman_pendulum.rs — Koopman linearization of nonlinear pendulum
- [x] examples/switched_controller.rs — Mode-switching control for hybrid plant
- [x] examples/antiwindup_demo.rs — AW compensator on saturated actuator plant

Verification: 2586 tests passing, 0 failing | SLoC: 77,783 (tokei src/) | 418 Rust files | clippy: 0 warnings (2026-03-09)

## Phase 21: Protocol Expansion ✅ COMPLETE (2026-03-09)

### EtherCAT Master/Slave
- [x] `protocol/ethercat/master.rs` - EtherCAT master state machine
- [x] `protocol/ethercat/slave.rs` - Slave device abstraction
- [x] `protocol/ethercat/dc.rs` - Distributed clocks (DC) synchronization
- [x] `protocol/ethercat/fmmu.rs` - Fieldbus Memory Management Unit mapping
- [x] `protocol/ethercat/mailbox.rs` - Mailbox protocol (CoE/FoE/EoE)
- [x] `protocol/ethercat/pdo.rs` - PDO mapping and exchange
- [x] `protocol/ethercat/sdo.rs` - SDO over EtherCAT (CoE)
- [x] `protocol/ethercat/drift_comp.rs` - Clock drift compensation
- [x] `protocol/ethercat/lss.rs` - Layer Setting Services

### CANopen (Extended)
- [x] `protocol/canopen/nmt.rs` - NMT state machine, heartbeat producer
- [x] `protocol/canopen/object_dict.rs` - Static OD with DataType/AccessType/OdValue
- [x] `protocol/canopen/sdo.rs` - SDO server, expedited transfer, abort codes
- [x] `protocol/canopen/pdo.rs` - TPDO/RPDO mapping, pack/unpack, event timer

## Future / Backlog

- [x] f32 fixed-point via `fixed` crate (planned 2026-04-27)
  - **Goal:** First-class fixed-point arithmetic alongside existing f32/f64 support so PID-style control loops can run on cost-sensitive MCUs without an FPU. Default features stay Pure Rust float; fixed-point is opt-in via a `fixed_point` feature flag. Selected algorithms (PID standard, derivative_filter, anti_windup) gain fixed-point genericity in this phase.
  - **Design:**
    - **Dependency.** Add `fixed` (latest stable on crates.io). no_std-friendly, Pure Rust.
    - **Feature flag.** New `fixed_point` Cargo feature, default off. Compiles in all four matrix points: `default`, `+fixed_point`, `+std+fixed_point`, `no_std+fixed_point`.
    - **Trait direction.** Introduce narrower `PidScalar` trait in `src/core/scalar.rs`: `Add+Sub+Mul+Neg+Copy+Debug+PartialOrd` + `from_int(i32)` + `saturating_{add,sub,mul}`. Blanket impl `impl<T: ControlScalar> PidScalar for T {}`. Fixed-point types implement `PidScalar` directly (not `ControlScalar`, which requires `Float`).
    - **Module layout.** New `src/core/fixed_point/`: `mod.rs`, `types.rs` (Q-format aliases Q15_16, Q1_31, Q3_29, Q7_24, Q31), `ops.rs` (saturating arithmetic, safe_div returning Result), `convert.rs` (from_f32_saturating, to_f32, from_int), `scalar_impl.rs` (PidScalar impls for each Q-format), `tests.rs` (unit tests).
    - **Algorithms made fixed-point-eligible this phase:** `pid::standard::Pid`, `pid::derivative_filter`, `pid::anti_windup` — bound relaxed from `ControlScalar` to `PidScalar`. Blanket impl ensures existing callers are unaffected.
    - **Explicitly out of scope:** any algorithm calling sin/exp/sqrt/log/tan/atan2 on the scalar (state-space, KF/EKF, filters, FOC, flatness, etc.).
    - **Example:** `examples/fixed_point_pid.rs` — first-order plant with Q15_16 PID step response, asserts convergence.
  - **Files:**
    - new: `src/core/fixed_point/{mod,types,ops,convert,scalar_impl,tests}.rs`
    - new: `examples/fixed_point_pid.rs`
    - new: `tests/fixed_point_validation/{mod,pid_step_response,saturation,roundtrip}.rs`
    - modified: `Cargo.toml` (dep + feature + example/test gating)
    - modified: `src/core/mod.rs`, `src/lib.rs`
    - modified: `src/core/scalar.rs` (new `PidScalar` trait + blanket impl)
    - modified: `src/pid/standard.rs`, `src/pid/derivative_filter.rs`, `src/pid/anti_windup.rs`
  - **Prerequisites:** Baseline green (cargo nextest --all-features + cargo clippy --all-features) before any code change.
  - **Tests:**
    - unit: round-trip f32 ↔ Q15_16; saturating add/sub at bounds; mul precision; div-by-zero returns Err; from_int for all Q-formats.
    - integration: PID step-response in Q15_16 matches f32 reference within 1% RMS error.
    - property (proptest): for random seeds in [-0.5, 0.5] Q15_16, (a+b)-b ≈ a within ULP; mul commutativity within saturation bounds.
  - **Risk:** Trait bound ripple (mitigated by blanket impl). Feature-gating: run all four feature combos. libm interop: float-only ops excluded from scope. No unwrap() in production code.
- [x] ROS2 bridge (full DDS transport) (completed 2026-04-28 — 8/8 DDS phases shipped)
- [x] Phase 23 — DDS user API (completed 2026-04-28 — Publisher<T>/Subscription<T>/Participant on top of full Phase 22 stack)


## Proposed follow-ups

### DDS transport — Phase decomposition (replaces: "ROS2 bridge (full DDS transport)")

The original ROS2 DDS item is vague (RTPS spec edition, QoS profile, transport choice, and target ROS2 distro are all undecided). It is decomposed into approvable phases. Pick one or more on the next `/ultra` run.
Open questions before any phase runs: (1) RTPS spec edition: 2.3 (broad compat) or 2.5 (latest)? (2) QoS coverage: ROS2-default profile only, or full DDS profile set? (3) Transport: UDP-only, or also SHM for intra-host? (4) Target ROS2 distro: Humble (LTS), Iron, or Jazzy?

- [x] Phase 22.1 — RTPS 2.3 wire-protocol foundation: zero-alloc no_std parser+serializer for the RTPS message format (planned 2026-04-27)
  - **Goal:** Complete, zero-allocation, no_std-compatible parser+serializer for RTPS 2.3 messages covering all 13 submessage kinds (DATA, DATA_FRAG, HEARTBEAT, HEARTBEAT_FRAG, ACKNACK, NACK_FRAG, GAP, INFO_TS, INFO_DST, INFO_SRC, INFO_REPLY, INFO_REPLY_IP4, PAD), ParameterList (40+ PID constants), and all RTPS primitive types. New `dds` feature, default off.
  - **Design:** `src/protocol/dds/` module — 21 files, all ≤ 400 lines, ~3000 LoC. Feature: `dds = ["protocol"]`. Zero-alloc parser borrows into input slice (`Message<'a>`); serializer writes to caller `&mut [u8]`. No modifications to `cdr_ser.rs`. Manual Display for RtpsError (no_std compatible). heapless::Vec<Submessage, 64> + heapless::Vec<Parameter, 32>.
  - **Files:** new `src/protocol/dds/{mod,error,byte_cursor,parser,serializer,tests}.rs` + `types/{mod,guid,locator,sequence,fragment,time,parameter}.rs` + `message/{mod,header}.rs` + `message/submessage/{mod,data,heartbeat,acknack,gap,info}.rs`; modified `Cargo.toml`, `src/protocol/mod.rs`.
  - **Tests:** 25+ unit tests: round-trip per submessage (13), header validation, truncated-buffer errors, endianness, SequenceNumberSet bitmap, ParameterList sentinel, structural fixture from spec bytes.
- [x] Phase 22.2 — UDPv4 transport with locator routing (dds-transport feature; std-only; TransportConfig, UdpTransport, port helpers, Locator↔SocketAddr conversion, loopback round-trip test).
- [x] Phase 22.3 — SPDP participant discovery (ParticipantBuiltinTopicData CDR encode/decode, SpdpParticipant beacon send/recv, discovered participant list with upsert; loopback test)
- [x] Phase 22.4 — SEDP endpoint discovery (PublicationBuiltinTopicData + SubscriptionBuiltinTopicData CDR encode/decode with CDR strings, 5 QoS policies, SedpParticipant announce/discover, loopback tests)
- [x] Phase 22.5 — Best-effort StatelessWriter / StatelessReader (HistoryCache, fire-and-forget DATA delivery, no ACK needed).
- [x] Phase 22.6 — Reliable StatefulWriter / StatefulReader (heartbeat/ACK/NACK cycle, sequence number tracking, retransmit queue).
- [x] Phase 22.7 — QoS policy matching (Reliability, History, Durability, Deadline, Liveliness — at minimum the ROS2-default profile).
- [x] Phase 22.8 — ROS2 builtin entity wiring (rt/Parameter, /rosout topic, namespace-prefixed topic name encoding, builtin endpoint set).

### DDS QoS extended policies

- [x] Phase 22.9 — Remaining DDS QoS wire-format types (Lifespan, Ownership, OwnershipStrength, DestinationOrder, ResourceLimits) + extended matcher (planned 2026-04-28)

### `#[allow(...)]` suppression cleanup

Six policy-violating `#[allow]` attributes exist in src/ and must be fixed (root cause, not silenced). The `lss.rs:22` module-wide `#[allow(unused)]` may indicate incomplete CANopen LSS scope.

- [x] Remove `#[allow(dead_code)]` in `src/core/filters/chebyshev.rs:183` — make code reachable or delete it.
- [x] Remove `#[allow(dead_code)]` in `src/estimator/fixed_interval_smoother.rs:127` — make code reachable or delete it.
- [x] Remove `#[allow(dead_code)]` in `src/state_feedback/integral_sliding_mode.rs:283` — make code reachable or delete it.
- [x] Remove module-wide `#[allow(unused)]` in `src/protocol/canopen/lss.rs:22` — investigate for incomplete LSS Protocol scope before deleting.
- [x] Remove `#[allow(dead_code)]` in `src/protocol/canopen/lss.rs:241` — make code reachable or delete it.
- [x] Remove `#[allow(dead_code)]` in `src/mpc/linear_mpc.rs:300` — make code reachable or delete it.

### Audit completion + baseline verification

- [x] Verify zero `unimplemented!()` / `todo!()` macros exist in src/.
- [x] Verify `cargo check --all-features` is green at HEAD.
- [x] Verify `cargo clippy --all-features --all-targets -- -D warnings` is green at HEAD (no-warnings policy hard gate; must be 0 warnings and 0 errors).

## Phase 23 — High-level DDS User API (`dds-api` feature) ✅ COMPLETE (2026-04-28)

- [x] Phase 23.1 — `DdsType` trait: `serialize(&self, buf: &mut [u8]) -> Result<usize, DdsApiError>` + `deserialize(payload: &[u8]) -> Result<Self, DdsApiError>` + `TYPE_NAME: &'static str`.
- [x] Phase 23.2 — `Sample<T>`: thin wrapper with `data: T` + `writer_guid_bytes: [u8; 16]`.
- [x] Phase 23.3 — `EntityIdAllocator`: process-global `AtomicU32` counter, `next_writer()` / `next_reader()` yielding distinct entity kinds.
- [x] Phase 23.4 — `DdsApiError`: unified error enum wrapping `StatefulError`, `DiscoveryError`, `TransportError`, `RtpsError` plus CDR-specific variants.
- [x] Phase 23.5 — `builtin_impls`: CDR LE encapsulation header helpers; `DdsType` impls for `heapless::String<256>`, `LogOwned`, `ParameterEventOwned`.
- [x] Phase 23.6 — `WriterEntry` / `Publisher<T>`: ephemeral UDP socket, `PublicationBuiltinTopicData`, typed `publish` method.
- [x] Phase 23.7 — `ReaderEntry` / `Subscription<T>`: ephemeral UDP socket, `SubscriptionBuiltinTopicData`, fixed-depth `raw_queue`, typed `take` method.
- [x] Phase 23.8 — `Participant`: SEDP metatraffic transport, explicit `add_peer`, `create_publisher`, `create_subscription`, `publish`, `take`, `spin_once` (announce → SEDP recv → match → heartbeat → recv data).
- [x] Phase 23.9 — Integration tests: 5 tests (`participant_create_and_peer_registration`, `cdr_string_dds_type_roundtrip`, `log_owned_roundtrip`, `publisher_subscription_create`, `end_to_end_publish_subscribe`) — all passing.
- [x] Bugfix: CDR encapsulation endianness detection in `make_cursor` (byte 1 bit-0 set → LE, not BE).
  - **Feature gate:** `dds-api = ["dds-ros2"]`
  - **Files:** `src/protocol/dds/api/{mod,error,dds_type,entity_id,builtin_impls,publisher,subscription,participant}.rs`; `tests/dds_api_integration/main.rs`
  - **Tests:** 5 integration tests + 2772 total tests passing, 0 failing; clippy clean.

## Phase 24 — DDS Production-Readiness (planned 2026-04-28)

- [x] Phase 24A — ROS2 standard message TypeSupport catalog (planned 2026-04-28)
  - **Goal:** `DdsType` impls for 35 ROS2 standard messages — `builtin_interfaces`, `std_msgs`, `geometry_msgs`, `sensor_msgs` — CDR-correct and TYPE_NAME-exact for rosidl interop.
  - **Design:** New `src/protocol/dds/ros2/msgs/` module (5 files) gated on existing `dds-api` feature. Hand-rolled CDR serialize/deserialize with alignment, 4-byte encapsulation header `[0x00,0x01,0x00,0x00]`, `derive(Default,Clone,Debug,PartialEq)` on every type. `TYPE_NAME` in form `<pkg>::msg::dds_::<Type>_`. Re-exported from `ros2/mod.rs`.
  - **Files:** `src/protocol/dds/ros2/msgs/mod.rs`, `builtin_interfaces.rs`, `std_msgs.rs`, `geometry_msgs.rs`, `sensor_msgs.rs` (new); `src/protocol/dds/ros2/mod.rs` (modified).
  - **Tests:** ≥70 unit tests (round-trip + TYPE_NAME literal + 5 byte-layout + 1 overflow).
  - **Risk:** CDR alignment errors — mitigated by byte-exact layout tests.

- [x] Phase 24B — Multicast SPDP + auto-discovery in Participant (planned 2026-04-28)
  - **Goal:** `Participant::new(domain_id, guid_prefix, qos)` auto-joins SPDP multicast `239.255.0.1:(7400+250*domain)`, sends periodic beacons, auto-discovers peers — no `add_peer` required for in-domain peers.
  - **Design:** Add `socket2 = "0.5"` dep (gated on `dds-transport`). New `transport/multicast_socket.rs` with `bind_multicast_reuse(port,group)` using SO_REUSEADDR+SO_REUSEPORT. `Participant` gains `spdp: SpdpParticipant`, `domain_id`, `last_beacon_at`, `auto_discovered_peers` fields. `spin_once` drives SPDP+auto-peer-promotion. Signature changes from `new(GuidPrefix,QosProfile)` to `new(domain_id,GuidPrefix,QosProfile)`.
  - **Files:** `src/protocol/dds/transport/multicast_socket.rs` (new); `Cargo.toml`, `transport/mod.rs`, `transport/udp.rs`, `discovery/spdp.rs`, `api/participant.rs`, `tests/dds_api_integration/main.rs` (modified).
  - **Tests:** 4 inline unit tests + 1 multicast integration test (domain 99, `#[ignore]` on non-Linux).
  - **Risk:** macOS multicast loopback — mitigated by explicit `set_multicast_loop_v4(true)` + `#[ignore]` gate.

- [x] Phase 24C — Concrete dds-api examples (planned 2026-04-28)
  - **Goal:** 3 executable examples: `ros2_chatter` (std_msgs), `ros2_imu_publisher` (sensor_msgs::Imu from estimator), `ros2_twist_subscriber` (geometry_msgs::Twist driving a sim).
  - **Design:** Self-contained loopback examples using two `Participant`s per process. Demonstrate TypeSupport catalog + typical control-system integration pattern.
  - **Files:** `examples/ros2_chatter.rs`, `examples/ros2_imu_publisher.rs`, `examples/ros2_twist_subscriber.rs` (new); `Cargo.toml` (3 `[[example]]` entries).
  - **Tests:** All three examples compile under `--all-features`.
  - **Risk:** sim type API mismatch — subagent reads actual types first.

- [x] Phase 24 — DDS production-readiness (completed 2026-04-28 — ROS2 message TypeSupport catalog (35 types) + multicast SPDP auto-discovery + 3 concrete examples)

## Phase 25 — ROS2 Services & Actions over DDS

- [x] Phase 25.1 — topic_naming.rs: fix service type-name suffix + add action naming (planned 2026-06-13)
  - **Goal:** Request/reply topic + type names exactly match rmw conventions; action topic/type naming supported.
  - **Design:** In `topic_naming.rs`: rename `TypeSuffix::Reply → TypeSuffix::Response`, map to `"_Response"` (keep `Plain`/`Request`). Add `TypeNamespace::Action => "action"`. Add `ActionSubtopic::{Feedback,Status}` + `encode_action_subtopic(out, action_name, sub)` producing `rt/<action>/_action/{feedback,status}`. Action *services* reuse the existing `encode_topic_name("<action>/_action/send_goal", ServiceRequest)` → `rq/<action>/_action/send_goalRequest`. `decode_topic_name` unchanged.
  - **Files:** `src/protocol/dds/ros2/topic_naming.rs`; re-export new `ActionSubtopic`/`encode_action_subtopic` in `src/protocol/dds/ros2/mod.rs` + `src/protocol/dds/mod.rs`.
  - **Prerequisites:** none (root of dependency tree).
  - **Tests:** `encode_type_name_srv_response`; `encode_action_type_send_goal_request`; `encode_action_subtopic_feedback`; `encode_action_subtopic_status`; `encode_service_subtopic_send_goal_request`.
  - **Risk:** TypeSuffix::Reply rename — no test asserts `_Reply_` type path; safe.

- [x] Phase 25.2 — SampleIdentity + request-header CDR codec (planned 2026-06-13)
  - **Goal:** Wire primitive for request/reply correlation, embedded at the front of the CDR body.
  - **Design:** `pub struct SampleIdentity { pub writer_guid: [u8;16], pub sequence_number: i64 }` (Copy/Eq). `serialize_inner`: 16 guid bytes + i64 LE (24 bytes total, 8-aligned). `deserialize_inner`: read 16 bytes + read_i64. Header seq is plain CDR int64 LE, NOT the RTPS SequenceNumber form.
  - **Files:** `src/protocol/dds/api/service/sample_identity.rs` (new).
  - **Prerequisites:** none.
  - **Tests:** `sample_identity_round_trip`; `sample_identity_wire_size` (body==24); `sample_identity_guid_bytes_order`.
  - **Risk:** CDR-int64 vs RTPS-SequenceNumber confusion — wire-size test pins it.

- [x] Phase 25.3 — ServiceClient<S> + Service/ServiceField traits + wrappers + reply filtering (planned 2026-06-13)
  - **Goal:** Type-safe client that sends Req and correlates Rep, robust to broadcast replies.
  - **Design:** Traits `ServiceField { serialize_inner/deserialize_inner }` and `Service { type Request: ServiceField; type Response: ServiceField; const REQUEST_TYPE_NAME; const RESPONSE_TYPE_NAME }`. Generic adapters `RequestWrapper<T>`/`ReplyWrapper<T>` implement `DdsType` composing SampleIdentity header + body `_inner`. `ServiceClient<S>` owns request Publisher, reply Subscription, `my_request_writer_guid: [u8;16]`, `next_request_seq: i64`, pending `heapless::Vec<i64,16>`. `send_request` assigns seq, wraps, publishes. `take_responses` filters by writer_guid.
  - **Files:** `src/protocol/dds/api/service/{mod,wrappers,client}.rs` (new).
  - **Prerequisites:** 25.1, 25.2.
  - **Tests:** `request_wrapper_round_trip`; `client_seq_monotonic`; `reply_filter_rejects_foreign_guid`.
  - **Risk:** Cap pending at 16 with oldest-eviction.

- [x] Phase 25.4 — ServiceServer<S> + Participant service plumbing (planned 2026-06-13)
  - **Goal:** Server receives requests, runs a callback, broadcasts correlated replies; Participant glue.
  - **Design:** `ServiceServer<S>` owns request Subscription + reply Publisher. `process<F: FnMut(&S::Request)->S::Response>` takes requests, calls handler, publishes ReplyWrapper with echoed SampleIdentity. Free fns `service::create_client`/`create_server`. Add `Participant::publisher_guid<T>(&self,&Publisher<T>)->Option<[u8;16]>`.
  - **Files:** `src/protocol/dds/api/service/server.rs` (new); `api/service/mod.rs`; `api/participant.rs` (accessor); `api/mod.rs` (re-export).
  - **Prerequisites:** 25.1, 25.2, 25.3.
  - **Tests:** Integration target `tests/dds_service_integration/main.rs`: `add_two_ints_request_reply`; `two_clients_no_cross_talk`; `server_handles_multiple_sequential`; `unmatched_service_no_reply`.
  - **Risk:** participant.rs edit — confine to Wave 0.

- [x] Phase 25.5 — srv message catalog: AddTwoInts + action_msgs (planned 2026-06-13)
  - **Goal:** Concrete request/response field structs + Service impls + standard action support messages.
  - **Design:** `example_interfaces` AddTwoInts Request/Response + Service impl. `action_msgs`: GoalInfo, GoalStatus (constants UNKNOWN=0..ABORTED=6), GoalStatusArray, CancelGoal Request/Response. Sequences follow JointState pattern.
  - **Files:** `src/protocol/dds/ros2/msgs/example_interfaces.rs`, `action_msgs.rs` (new); `msgs/mod.rs` + `ros2/mod.rs` re-exports.
  - **Prerequisites:** 25.1, 25.3.
  - **Tests:** round-trips; `goal_status_constants`; `goal_status_array_byte_layout`; type-name tests.
  - **Risk:** Sequence-of-nested-struct alignment — model on JointState.

- [x] Phase 25.6 — unique_identifier_msgs/UUID + GoalId helper (planned 2026-06-13)
  - **Goal:** 16-raw-byte goal identifier used throughout actions.
  - **Design:** `pub struct Uuid { pub uuid: [u8;16] }`. `serialize_inner`: `write_bytes(&uuid)` (fixed octet array, NO length prefix). `TYPE_NAME="unique_identifier_msgs::msg::dds_::UUID_"`. `Uuid::nil()`, `Uuid::from_bytes([u8;16])`.
  - **Files:** `src/protocol/dds/ros2/msgs/unique_identifier_msgs.rs` (new); `msgs/mod.rs` + `ros2/mod.rs` re-exports.
  - **Prerequisites:** none.
  - **Tests:** `uuid_round_trip`; `uuid_wire_size` (body==16, no prefix); `uuid_type_name`.
  - **Risk:** Mistaken 4-byte length prefix — explicit wire-size test.

- [x] Phase 25.7 — ActionServer<A> / ActionClient<A> + Fibonacci action (planned 2026-06-13)
  - **Goal:** Full ROS2 action orchestration on top of services + pub/sub.
  - **Design:** `trait Action { type Goal: ServiceField; type Result: ServiceField; type Feedback: DdsType+Clone; const ACTION_NAME_TYPE_PREFIX }`. ActionServer = 3 ServiceServers + 2 Publishers (feedback, status). ActionClient mirrors. Fibonacci example action. `heapless::Vec<(Uuid,i8)>` active goals.
  - **Files:** `src/protocol/dds/api/action/{mod,server,client}.rs`; `ros2/msgs/example_interfaces_action.rs` (new); re-exports in `api/mod.rs`, `ros2/mod.rs`, `msgs/mod.rs`.
  - **Prerequisites:** 25.1–25.6.
  - **Tests:** Integration target `tests/dds_action_integration/main.rs`: `fibonacci_goal_accept_and_result`; `feedback_flows_to_client`; `status_array_reflects_lifecycle`; `cancel_goal_marks_canceling`; `two_action_clients_isolated`.
  - **Risk:** get_result flakiness → handler drives goal terminal deterministically.

- [x] Phase 25.8 — docs + re-export audit + clippy/no_std/test sweep (planned 2026-06-13)
  - **Goal:** Ship-quality: consistent public surface, zero warnings, no_std-clean codecs, green workspace.
  - **Design:** Module docs on service/mod.rs + action/mod.rs. Audit no unwrap() outside #[cfg(test)]. Batch all pub use additions. Add [[test]] entries in Cargo.toml gated on dds-api feature.
  - **Files:** `api/mod.rs`, `dds/mod.rs`, `ros2/mod.rs`, `msgs/mod.rs`, `Cargo.toml`.
  - **Prerequisites:** 25.1–25.7.
  - **Tests:** cargo clippy --all-features --all-targets -- -D warnings clean; cargo nextest run --all-features green; `service_module_reexports_present` compile-test.
  - **Risk:** File-size creep — split if >2000 lines.
