//! kizzasi sensor stream bridge (stub).
//!
//! Provides a trait-based integration point for connecting the oxictl
//! control framework to the kizzasi real-time sensor stream ecosystem.
//!
//! This stub allows the crate to compile and be used standalone.
//! Full integration requires the `io-kizzasi` feature and a kizzasi runtime.

/// A timestamped sensor sample from a kizzasi stream.
#[derive(Debug, Clone, Copy)]
pub struct SensorSample {
    pub time_us: u64,
    pub value: f64,
    pub channel_id: u32,
}

/// Trait for kizzasi-compatible sensor stream producers.
///
/// Implementors push samples into the control loop tick function.
pub trait KizzasiSink {
    /// Accept an incoming sensor sample.
    fn on_sample(&mut self, sample: SensorSample);

    /// Called at the start of each control tick.
    fn tick_start(&mut self, time_us: u64);

    /// Called at the end of each control tick.
    fn tick_end(&mut self, time_us: u64);
}

/// Stub kizzasi sink that discards all samples (no-op).
pub struct NullKizzasiSink;

impl KizzasiSink for NullKizzasiSink {
    fn on_sample(&mut self, _sample: SensorSample) {}
    fn tick_start(&mut self, _time_us: u64) {}
    fn tick_end(&mut self, _time_us: u64) {}
}

/// Buffered kizzasi sink that stores the last N samples per channel.
pub struct BufferedKizzasiSink<const N: usize> {
    pub samples: [Option<SensorSample>; N],
    pub count: usize,
}

impl<const N: usize> BufferedKizzasiSink<N> {
    pub fn new() -> Self {
        Self {
            samples: [None; N],
            count: 0,
        }
    }

    /// Get the latest sample for a given channel_id.
    pub fn latest(&self, channel_id: u32) -> Option<SensorSample> {
        self.samples[..self.count.min(N)]
            .iter()
            .filter_map(|s| *s)
            .rfind(|s| s.channel_id == channel_id)
    }
}

impl<const N: usize> Default for BufferedKizzasiSink<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> KizzasiSink for BufferedKizzasiSink<N> {
    fn on_sample(&mut self, sample: SensorSample) {
        let idx = self.count % N;
        self.samples[idx] = Some(sample);
        self.count += 1;
    }

    fn tick_start(&mut self, _time_us: u64) {}
    fn tick_end(&mut self, _time_us: u64) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_sink_compiles() {
        let mut sink = NullKizzasiSink;
        sink.on_sample(SensorSample {
            time_us: 0,
            value: 1.0,
            channel_id: 0,
        });
        sink.tick_start(0);
        sink.tick_end(100);
    }

    #[test]
    fn buffered_sink_stores_samples() {
        let mut sink = BufferedKizzasiSink::<8>::new();
        sink.on_sample(SensorSample {
            time_us: 100,
            value: 1.5,
            channel_id: 1,
        });
        sink.on_sample(SensorSample {
            time_us: 200,
            value: 2.5,
            channel_id: 2,
        });
        sink.on_sample(SensorSample {
            time_us: 300,
            value: 3.5,
            channel_id: 1,
        });

        let latest = sink.latest(1).unwrap();
        assert!((latest.value - 3.5).abs() < 1e-10);
        assert_eq!(latest.time_us, 300);
    }

    #[test]
    fn buffered_sink_no_channel_returns_none() {
        let sink = BufferedKizzasiSink::<4>::new();
        assert!(sink.latest(99).is_none());
    }
}
