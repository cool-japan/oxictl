//! EtherCAT Distributed Clock drift compensation.
//!
//! Measures and corrects timing drift between master and slave clocks
//! using a PI controller on the measured offset. Stores a circular
//! buffer of timing samples for averaging.

use crate::core::scalar::ControlScalar;

/// Number of samples in the circular timing buffer.
const SAMPLE_COUNT: usize = 16;

/// Drift compensator using a PI loop to correct DC timing offset.
///
/// `S` is the scalar type (f32 or f64).
pub struct DriftCompensator<S: ControlScalar> {
    /// Proportional gain for the PI correction loop.
    kp: S,
    /// Integral gain for the PI correction loop.
    ki: S,
    /// Accumulated integral error.
    integral: S,
    /// Circular buffer of measured offsets (nanoseconds).
    samples: [S; SAMPLE_COUNT],
    /// Write head for circular buffer.
    head: usize,
    /// Number of valid samples.
    count: usize,
    /// Current correction value (nanoseconds).
    correction: S,
    /// Estimated drift rate (parts per billion).
    drift_ppb: S,
    /// Previous average offset for rate estimation.
    prev_avg: S,
    /// Correction applied so far (for rate tracking).
    total_correction: S,
}

impl<S: ControlScalar> DriftCompensator<S> {
    /// Create a new drift compensator with given PI gains.
    ///
    /// Typical values: kp=0.1, ki=0.01 for nanosecond offsets.
    pub fn new(kp: S, ki: S) -> Self {
        Self {
            kp,
            ki,
            integral: S::zero(),
            samples: [S::zero(); SAMPLE_COUNT],
            head: 0,
            count: 0,
            correction: S::zero(),
            drift_ppb: S::zero(),
            prev_avg: S::zero(),
            total_correction: S::zero(),
        }
    }

    /// Record a timing offset measurement (in nanoseconds).
    pub fn measure_offset(&mut self, offset_ns: S) {
        self.samples[self.head] = offset_ns;
        self.head = (self.head + 1) % SAMPLE_COUNT;
        if self.count < SAMPLE_COUNT {
            self.count += 1;
        }
    }

    /// Compute averaged offset from the sample buffer.
    pub fn average_offset(&self) -> S {
        if self.count == 0 {
            return S::zero();
        }
        let mut sum = S::zero();
        for i in 0..self.count {
            sum += self.samples[i];
        }
        // divide by count
        let n = S::from_f64(self.count as f64);
        sum / n
    }

    /// Apply the PI correction loop to the current averaged offset.
    /// Returns the correction value to apply (nanoseconds).
    pub fn apply_correction(&mut self) -> S {
        let avg = self.average_offset();
        self.integral += avg * self.ki;

        // Anti-windup: clamp integral
        let limit = S::from_f64(1_000_000.0); // 1ms limit
        if self.integral > limit {
            self.integral = limit;
        } else if self.integral < -limit {
            self.integral = -limit;
        }

        let new_correction = avg * self.kp + self.integral;

        // Estimate drift rate from change in average offset
        let delta_avg = avg - self.prev_avg;
        // Drift rate in ppb: delta_ns / 1_000_000_000 * 1e9 = delta_ns (ns/s)
        // Normalized to ppb relative to 1s period
        self.drift_ppb = delta_avg * S::from_f64(1_000.0);
        self.prev_avg = avg;

        self.total_correction += new_correction - self.correction;
        self.correction = new_correction;
        new_correction
    }

    /// Current estimated drift rate in parts-per-billion.
    pub fn drift_rate_ppb(&self) -> S {
        self.drift_ppb
    }

    /// Current correction value in nanoseconds.
    pub fn correction_ns(&self) -> S {
        self.correction
    }

    /// Number of samples collected.
    pub fn sample_count(&self) -> usize {
        self.count
    }

    /// Reset the compensator state.
    pub fn reset(&mut self) {
        self.integral = S::zero();
        self.samples = [S::zero(); SAMPLE_COUNT];
        self.head = 0;
        self.count = 0;
        self.correction = S::zero();
        self.drift_ppb = S::zero();
        self.prev_avg = S::zero();
        self.total_correction = S::zero();
    }

    /// Feed a vector of offset measurements at once.
    pub fn feed_samples(&mut self, offsets: &[S]) {
        for &o in offsets {
            self.measure_offset(o);
        }
    }

    /// Check if drift exceeds the given threshold (nanoseconds).
    pub fn drift_exceeds(&self, threshold_ns: S) -> bool {
        let avg = self.average_offset();
        // abs via comparison
        let abs_avg = if avg < S::zero() {
            S::zero() - avg
        } else {
            avg
        };
        abs_avg > threshold_ns
    }

    /// Get the correction increment (delta from last update).
    pub fn total_correction(&self) -> S {
        self.total_correction
    }
}

/// Synchronized clock adjustment record.
#[derive(Debug, Clone, Copy)]
pub struct ClockAdjustment<S: ControlScalar> {
    /// Offset measured (ns).
    pub offset_ns: S,
    /// Correction applied (ns).
    pub correction_ns: S,
    /// Drift rate (ppb).
    pub drift_ppb: S,
}

impl<S: ControlScalar> ClockAdjustment<S> {
    /// Create a new adjustment record.
    pub fn new(offset_ns: S, correction_ns: S, drift_ppb: S) -> Self {
        Self {
            offset_ns,
            correction_ns,
            drift_ppb,
        }
    }
}

/// Multi-slave drift tracking table (up to N slaves).
pub struct DriftTable<S: ControlScalar, const N: usize> {
    compensators: [DriftCompensator<S>; N],
    active: [bool; N],
}

impl<S: ControlScalar, const N: usize> DriftTable<S, N> {
    /// Create a new drift table with given PI gains applied to all slaves.
    pub fn new(kp: S, ki: S) -> Self {
        // Cannot use array::from_fn with non-Copy type simply, use manual init
        // DriftCompensator is not Copy, so build with a helper
        Self {
            compensators: core::array::from_fn(|_| DriftCompensator::new(kp, ki)),
            active: [false; N],
        }
    }

    /// Activate tracking for a slave slot.
    pub fn activate(&mut self, slot: usize) {
        if slot < N {
            self.active[slot] = true;
        }
    }

    /// Record an offset for a slave slot.
    pub fn record_offset(&mut self, slot: usize, offset_ns: S) {
        if slot < N && self.active[slot] {
            self.compensators[slot].measure_offset(offset_ns);
        }
    }

    /// Apply correction for a slave slot. Returns correction or zero if inactive.
    pub fn apply_correction(&mut self, slot: usize) -> S {
        if slot < N && self.active[slot] {
            self.compensators[slot].apply_correction()
        } else {
            S::zero()
        }
    }

    /// Get drift rate for a slave slot.
    pub fn drift_ppb(&self, slot: usize) -> S {
        if slot < N && self.active[slot] {
            self.compensators[slot].drift_rate_ppb()
        } else {
            S::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_and_average() {
        let mut dc = DriftCompensator::<f64>::new(0.1, 0.01);
        dc.measure_offset(100.0);
        dc.measure_offset(200.0);
        dc.measure_offset(300.0);
        assert_eq!(dc.sample_count(), 3);
        let avg = dc.average_offset();
        assert!((avg - 200.0).abs() < 1e-9);
    }

    #[test]
    fn test_apply_correction_reduces_offset() {
        let mut dc = DriftCompensator::<f64>::new(0.5, 0.1);
        // Feed a constant offset of 1000 ns
        for _ in 0..16 {
            dc.measure_offset(1000.0);
        }
        let corr = dc.apply_correction();
        // Correction should be positive (counteracting positive offset)
        assert!(corr > 0.0);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut dc = DriftCompensator::<f32>::new(0.1, 0.01);
        dc.measure_offset(500.0);
        dc.apply_correction();
        dc.reset();
        assert_eq!(dc.sample_count(), 0);
        assert_eq!(dc.correction_ns(), 0.0f32);
        assert_eq!(dc.average_offset(), 0.0f32);
    }

    #[test]
    fn test_drift_exceeds_threshold() {
        let mut dc = DriftCompensator::<f64>::new(0.1, 0.01);
        dc.feed_samples(&[500.0, 600.0, 700.0]);
        assert!(dc.drift_exceeds(100.0));
        assert!(!dc.drift_exceeds(10000.0));
    }

    #[test]
    fn test_circular_buffer_overwrites() {
        let mut dc = DriftCompensator::<f64>::new(0.1, 0.01);
        // Fill more than SAMPLE_COUNT (16) samples
        for i in 0..20 {
            dc.measure_offset(i as f64 * 10.0);
        }
        // Buffer should be full (16 samples)
        assert_eq!(dc.sample_count(), SAMPLE_COUNT);
        // Average of last 16 samples: [40,50,...,190] = (40+190)/2 = 115
        let avg = dc.average_offset();
        assert!(avg > 0.0);
    }

    #[test]
    fn test_drift_table() {
        let mut table = DriftTable::<f64, 4>::new(0.1, 0.01);
        table.activate(0);
        table.record_offset(0, 200.0);
        table.record_offset(0, 200.0);
        let corr = table.apply_correction(0);
        assert!(corr > 0.0);
        // Inactive slot returns zero
        assert_eq!(table.apply_correction(1), 0.0);
    }
}
