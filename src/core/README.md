# oxictl::core

Core traits and types for the control systems framework.

## Key Types

- **`ControlScalar`** - Trait abstracting f32/f64 for compile-time precision selection
- **`Controller<S>`** - Trait for any controller (PID, MPC, etc.) with `update()`, `reset()`, `is_saturated()`
- **`Plant<S>`** - Trait for dynamic systems with `step()`, `output()`, `state()`
- **`Estimator<S, N>`** - Trait for state estimators (Kalman, observers)
- **`Setpoint<S>`**, **`Feedback<S>`**, **`ControlOutput<S>`** - Type-safe signal wrappers
- **`OutputLimiter<S>`** - Min/max clamp with saturation detection
- **`RateLimiter<S>`** - Rate-of-change limiting

## no_std Compatibility

Fully `no_std` + `no_alloc`. All operations are O(1).
