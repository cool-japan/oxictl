//! oxigrid energy system bridge (stub).
//!
//! Integration point for connecting oxictl power control algorithms to
//! the oxigrid energy management and grid simulation ecosystem.
//!
//! Exposes grid state (voltage, frequency, power flow) and accepts
//! control commands (setpoints, switching signals).

/// Grid measurement point.
#[derive(Debug, Clone, Copy, Default)]
pub struct GridMeasurement {
    /// Grid frequency (Hz).
    pub frequency_hz: f64,
    /// RMS line-to-neutral voltage (V).
    pub voltage_rms: f64,
    /// Active power flow (W, positive = import).
    pub active_power_w: f64,
    /// Reactive power flow (VAR).
    pub reactive_power_var: f64,
    /// Power factor (-1..1).
    pub power_factor: f64,
    /// Measurement timestamp (µs).
    pub time_us: u64,
}

/// Control command to grid interface.
#[derive(Debug, Clone, Copy, Default)]
pub struct GridCommand {
    /// Active power setpoint (W). Positive = inject into grid.
    pub active_power_sp: f64,
    /// Reactive power setpoint (VAR).
    pub reactive_power_sp: f64,
    /// Voltage regulation setpoint (V). 0 = no regulation.
    pub voltage_sp: f64,
    /// Enable flag.
    pub enabled: bool,
}

/// Trait for oxigrid-compatible grid interfaces.
pub trait OxigridInterface {
    /// Read current grid measurement.
    fn read_measurement(&self) -> GridMeasurement;

    /// Send control command to grid interface.
    fn send_command(&mut self, cmd: GridCommand);

    /// Check if grid is connected and healthy.
    fn is_connected(&self) -> bool;
}

/// Stub oxigrid interface returning nominal values.
pub struct NullOxigridInterface {
    pub nominal_frequency_hz: f64,
    pub nominal_voltage_rms: f64,
    last_cmd: GridCommand,
}

impl NullOxigridInterface {
    pub fn new(nominal_frequency_hz: f64, nominal_voltage_rms: f64) -> Self {
        Self {
            nominal_frequency_hz,
            nominal_voltage_rms,
            last_cmd: GridCommand::default(),
        }
    }

    pub fn last_command(&self) -> GridCommand {
        self.last_cmd
    }
}

impl OxigridInterface for NullOxigridInterface {
    fn read_measurement(&self) -> GridMeasurement {
        GridMeasurement {
            frequency_hz: self.nominal_frequency_hz,
            voltage_rms: self.nominal_voltage_rms,
            active_power_w: 0.0,
            reactive_power_var: 0.0,
            power_factor: 1.0,
            time_us: 0,
        }
    }

    fn send_command(&mut self, cmd: GridCommand) {
        self.last_cmd = cmd;
    }

    fn is_connected(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_interface_returns_nominal() {
        let iface = NullOxigridInterface::new(50.0, 230.0);
        let m = iface.read_measurement();
        assert!((m.frequency_hz - 50.0).abs() < 1e-10);
        assert!((m.voltage_rms - 230.0).abs() < 1e-10);
        assert!(iface.is_connected());
    }

    #[test]
    fn send_command_stored() {
        let mut iface = NullOxigridInterface::new(50.0, 230.0);
        let cmd = GridCommand {
            active_power_sp: 1000.0,
            enabled: true,
            ..Default::default()
        };
        iface.send_command(cmd);
        assert!((iface.last_command().active_power_sp - 1000.0).abs() < 1e-10);
        assert!(iface.last_command().enabled);
    }
}
