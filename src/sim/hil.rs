//! Hardware-in-the-Loop (HIL) interface traits and simulation bridge.
//!
//! Provides a common abstraction layer for running control algorithms
//! against real hardware or hardware simulators, supporting:
//! - Deterministic time stepping (real-time or sim-time)
//! - I/O channel abstraction (analog in/out, digital in/out)
//! - Scenario scripting (step, ramp, sinusoidal inputs)
//! - Logging integration

/// Result type for HIL operations.
pub type HilResult<T> = Result<T, HilError>;

/// Errors from HIL operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HilError {
    /// Channel index out of range.
    ChannelOutOfRange,
    /// Timeout waiting for I/O.
    Timeout,
    /// HIL device not connected.
    NotConnected,
    /// Data overflow in buffer.
    Overflow,
}

/// Analog input channel descriptor.
#[derive(Debug, Clone, Copy)]
pub struct AnalogIn {
    pub channel: u8,
    /// Voltage range in volts (full scale).
    pub range_v: f32,
    /// Resolution in bits.
    pub resolution_bits: u8,
}

/// Analog output channel descriptor.
#[derive(Debug, Clone, Copy)]
pub struct AnalogOut {
    pub channel: u8,
    pub range_v: f32,
    pub resolution_bits: u8,
}

/// Digital I/O direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigitalDir {
    Input,
    Output,
}

/// Digital I/O channel descriptor.
#[derive(Debug, Clone, Copy)]
pub struct DigitalChannel {
    pub channel: u8,
    pub dir: DigitalDir,
    /// Active-high (true) or active-low (false).
    pub active_high: bool,
}

/// Trait for HIL device drivers (real or simulated).
///
/// Implementors bridge the control software to physical or virtual hardware.
pub trait HilDevice {
    /// Read a single analog input channel.
    fn read_analog(&self, channel: u8) -> HilResult<f32>;

    /// Write a single analog output channel.
    fn write_analog(&mut self, channel: u8, value: f32) -> HilResult<()>;

    /// Read a digital input channel.
    fn read_digital(&self, channel: u8) -> HilResult<bool>;

    /// Write a digital output channel.
    fn write_digital(&mut self, channel: u8, state: bool) -> HilResult<()>;

    /// Advance simulation time by `dt_us` microseconds (for sim HIL devices).
    fn step_time(&mut self, dt_us: u32);

    /// Current simulation time in microseconds.
    fn time_us(&self) -> u64;

    /// Reset all outputs and simulation state.
    fn reset(&mut self);
}

/// Waveform type for scenario scripting.
#[derive(Debug, Clone, Copy)]
pub enum Waveform {
    /// Constant value.
    Constant(f32),
    /// Step at time t_start_us.
    Step {
        value_before: f32,
        value_after: f32,
        t_start_us: u64,
    },
    /// Linear ramp from start to end over duration.
    Ramp {
        start: f32,
        end: f32,
        t_start_us: u64,
        duration_us: u64,
    },
    /// Sinusoidal: amplitude, frequency_hz, phase_rad, offset.
    Sine {
        amplitude: f32,
        frequency_hz: f32,
        phase_rad: f32,
        offset: f32,
    },
}

impl Waveform {
    /// Evaluate waveform at time `t_us` (microseconds).
    pub fn evaluate(&self, t_us: u64) -> f32 {
        match *self {
            Self::Constant(v) => v,
            Self::Step {
                value_before,
                value_after,
                t_start_us,
            } => {
                if t_us >= t_start_us {
                    value_after
                } else {
                    value_before
                }
            }
            Self::Ramp {
                start,
                end,
                t_start_us,
                duration_us,
            } => {
                if t_us <= t_start_us {
                    start
                } else if t_us >= t_start_us + duration_us {
                    end
                } else {
                    let frac = (t_us - t_start_us) as f32 / duration_us as f32;
                    start + frac * (end - start)
                }
            }
            Self::Sine {
                amplitude,
                frequency_hz,
                phase_rad,
                offset,
            } => {
                let t_s = t_us as f32 * 1e-6;
                let arg = 2.0 * core::f32::consts::PI * frequency_hz * t_s + phase_rad;
                offset + amplitude * arg.sin()
            }
        }
    }
}

/// Simulated HIL device backed by software models.
///
/// Useful for testing control software without physical hardware.
/// Stores up to `N_AI` analog inputs, `N_AO` analog outputs, `N_DIO` digital I/O.
pub struct SimHil<const N_AI: usize, const N_AO: usize, const N_DIO: usize> {
    analog_in: [f32; N_AI],
    analog_out: [f32; N_AO],
    digital_io: [bool; N_DIO],
    time_us: u64,
    /// Scripted waveforms for each analog input channel.
    waveforms: [Option<Waveform>; N_AI],
}

impl<const N_AI: usize, const N_AO: usize, const N_DIO: usize> SimHil<N_AI, N_AO, N_DIO> {
    pub fn new() -> Self {
        Self {
            analog_in: [0.0; N_AI],
            analog_out: [0.0; N_AO],
            digital_io: [false; N_DIO],
            time_us: 0,
            waveforms: core::array::from_fn(|_| None),
        }
    }

    /// Assign a waveform to an analog input channel.
    pub fn set_waveform(&mut self, channel: usize, waveform: Waveform) {
        if channel < N_AI {
            self.waveforms[channel] = Some(waveform);
        }
    }

    /// Manually set an analog input value (overrides waveform for one step).
    pub fn inject_analog(&mut self, channel: usize, value: f32) {
        if channel < N_AI {
            self.analog_in[channel] = value;
        }
    }
}

impl<const N_AI: usize, const N_AO: usize, const N_DIO: usize> Default
    for SimHil<N_AI, N_AO, N_DIO>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const N_AI: usize, const N_AO: usize, const N_DIO: usize> HilDevice
    for SimHil<N_AI, N_AO, N_DIO>
{
    fn read_analog(&self, channel: u8) -> HilResult<f32> {
        let ch = channel as usize;
        if ch >= N_AI {
            return Err(HilError::ChannelOutOfRange);
        }
        // If waveform active, return evaluated value
        if let Some(wf) = self.waveforms[ch] {
            Ok(wf.evaluate(self.time_us))
        } else {
            Ok(self.analog_in[ch])
        }
    }

    fn write_analog(&mut self, channel: u8, value: f32) -> HilResult<()> {
        let ch = channel as usize;
        if ch >= N_AO {
            return Err(HilError::ChannelOutOfRange);
        }
        self.analog_out[ch] = value;
        Ok(())
    }

    fn read_digital(&self, channel: u8) -> HilResult<bool> {
        let ch = channel as usize;
        if ch >= N_DIO {
            return Err(HilError::ChannelOutOfRange);
        }
        Ok(self.digital_io[ch])
    }

    fn write_digital(&mut self, channel: u8, state: bool) -> HilResult<()> {
        let ch = channel as usize;
        if ch >= N_DIO {
            return Err(HilError::ChannelOutOfRange);
        }
        self.digital_io[ch] = state;
        Ok(())
    }

    fn step_time(&mut self, dt_us: u32) {
        self.time_us += dt_us as u64;
    }

    fn time_us(&self) -> u64 {
        self.time_us
    }

    fn reset(&mut self) {
        self.analog_in = [0.0; N_AI];
        self.analog_out = [0.0; N_AO];
        self.digital_io = [false; N_DIO];
        self.time_us = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waveform_step() {
        let wf = Waveform::Step {
            value_before: 0.0,
            value_after: 5.0,
            t_start_us: 1000,
        };
        assert_eq!(wf.evaluate(0), 0.0);
        assert_eq!(wf.evaluate(999), 0.0);
        assert_eq!(wf.evaluate(1000), 5.0);
        assert_eq!(wf.evaluate(2000), 5.0);
    }

    #[test]
    fn waveform_ramp() {
        let wf = Waveform::Ramp {
            start: 0.0,
            end: 10.0,
            t_start_us: 0,
            duration_us: 1000,
        };
        assert_eq!(wf.evaluate(0), 0.0);
        assert!((wf.evaluate(500) - 5.0).abs() < 0.01);
        assert_eq!(wf.evaluate(1000), 10.0);
        assert_eq!(wf.evaluate(2000), 10.0);
    }

    #[test]
    fn waveform_sine() {
        let wf = Waveform::Sine {
            amplitude: 1.0,
            frequency_hz: 1000.0,
            phase_rad: 0.0,
            offset: 0.0,
        };
        // At t=0: sin(0)=0
        assert!((wf.evaluate(0)).abs() < 0.01);
        // At quarter period: sin(π/2) ≈ 1
        let quarter_us = 250u64; // 250μs = quarter of 1kHz
        assert!(
            (wf.evaluate(quarter_us) - 1.0).abs() < 0.01,
            "val={:.4}",
            wf.evaluate(quarter_us)
        );
    }

    #[test]
    fn sim_hil_analog_io() {
        let mut hil = SimHil::<4, 4, 8>::new();
        hil.write_analog(0, 3.3).unwrap();
        assert_eq!(hil.analog_out[0], 3.3);

        hil.inject_analog(1, 2.5);
        assert_eq!(hil.read_analog(1).unwrap(), 2.5);
    }

    #[test]
    fn sim_hil_digital_io() {
        let mut hil = SimHil::<2, 2, 4>::new();
        hil.write_digital(2, true).unwrap();
        assert!(hil.read_digital(2).unwrap());
        hil.write_digital(2, false).unwrap();
        assert!(!hil.read_digital(2).unwrap());
    }

    #[test]
    fn sim_hil_waveform() {
        let mut hil = SimHil::<2, 2, 2>::new();
        hil.set_waveform(0, Waveform::Constant(5.0));
        assert_eq!(hil.read_analog(0).unwrap(), 5.0);
        hil.step_time(1000);
        assert_eq!(hil.time_us(), 1000);
    }

    #[test]
    fn sim_hil_channel_out_of_range() {
        let hil = SimHil::<2, 2, 2>::new();
        assert_eq!(hil.read_analog(5), Err(HilError::ChannelOutOfRange));
    }

    #[test]
    fn sim_hil_reset() {
        let mut hil = SimHil::<2, 2, 2>::new();
        hil.write_analog(0, 3.3).unwrap();
        hil.step_time(5000);
        hil.reset();
        assert_eq!(hil.time_us(), 0);
        assert_eq!(hil.analog_out[0], 0.0);
    }
}
