# OxiCtl Development TODO

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

- [ ] f32 fixed-point via `fixed` crate
- [ ] ROS2 bridge (full DDS transport)
