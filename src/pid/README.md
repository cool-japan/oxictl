# oxictl::pid

PID controller module with industrial-grade features.

## Key Types

- **`PidConfig<S>`** - Builder for PID controllers. Supports `p()`, `pi()`, `pid()` constructors
- **`Pid<S>`** - The PID controller implementing `Controller<S>` trait
- **`AntiWindupMethod<S>`** - None, Clamping, BackCalculation
- **`DerivativeFilter<S>`** - First-order IIR low-pass for noise-free derivative

## Features

- 2-DOF PID with setpoint weights (beta, gamma)
- Derivative on measurement (avoids derivative kick)
- Configurable output limiting
- Anti-windup with clamping or back-calculation
- dt=0 guard, integral overflow prevention

## no_std Compatibility

Fully `no_std` + `no_alloc`. All operations are O(1).
