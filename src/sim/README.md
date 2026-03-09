# oxictl::sim

Simulation environment for testing control systems. Requires `std` feature.

## Key Types

- **`ThermalPlant`** - First-order thermal model: `dT/dt = (1/tau)(K*u - (T - T_ambient))`
- **`Scope`** - Waveform recorder with CSV export, min/max analysis

## Usage

```rust
use oxictl::sim::{ThermalPlant, Scope};
use oxictl::core::traits::Plant;

let mut plant = ThermalPlant::new(25.0, 10.0, 100.0, 25.0);
let mut scope = Scope::new("temperature");

for _ in 0..1000 {
    plant.step(0.5, 0.01);
    scope.record(0.0, plant.output());
}
println!("{}", scope.to_csv());
```

## no_std Compatibility

This module requires `std` (uses `Vec` for scope data). Not available in `no_std` builds.
