//! CANopen SYNC protocol (CiA 301).
//!
//! The SYNC message triggers synchronous PDO transmission. The SYNC producer
//! sends COB-ID 0x80 frames at a configured interval. Consumers count SYNC
//! events and fire registered callbacks.

/// Default SYNC COB-ID per CiA 301.
pub const SYNC_COB_ID: u16 = 0x80;

/// Maximum SYNC counter value before overflow (1–240 per CiA 301).
pub const SYNC_COUNTER_MAX: u8 = 240;

/// SYNC producer: sends periodic SYNC frames.
///
/// The counter is only transmitted when `counter_overflow` > 0 (synchronous
/// counter mode per CiA 301 section 7.3.3).
#[derive(Debug, Clone)]
pub struct SyncProducer {
    /// COB-ID for SYNC message (default 0x80).
    cob_id: u16,
    /// Current SYNC counter value (1–240, 0=disabled).
    counter: u8,
    /// Number of SYNCs before counter overflows (0=disabled).
    counter_overflow: u8,
    /// Total SYNCs generated.
    total_syncs: u64,
    /// Whether the producer is enabled.
    enabled: bool,
}

impl SyncProducer {
    /// Create a SYNC producer with the given COB-ID.
    pub fn new(cob_id: u16) -> Self {
        Self {
            cob_id,
            counter: 0,
            counter_overflow: 0,
            total_syncs: 0,
            enabled: true,
        }
    }

    /// Create a SYNC producer with counter mode.
    ///
    /// `overflow` sets the maximum counter value (1–240). When the counter
    /// reaches `overflow` it resets to 1 on the next SYNC.
    pub fn with_counter(cob_id: u16, overflow: u8) -> Self {
        let overflow = overflow.clamp(1, SYNC_COUNTER_MAX);
        Self {
            cob_id,
            counter: 1,
            counter_overflow: overflow,
            total_syncs: 0,
            enabled: true,
        }
    }

    /// Enable or disable SYNC generation.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Is the producer enabled?
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// COB-ID being used.
    pub fn cob_id(&self) -> u16 {
        self.cob_id
    }

    /// Current counter value (0 if counter mode disabled).
    pub fn counter(&self) -> u8 {
        self.counter
    }

    /// Total number of SYNCs generated.
    pub fn total_syncs(&self) -> u64 {
        self.total_syncs
    }

    /// Generate the next SYNC frame.
    ///
    /// Returns `Some([u8; 8])` with the CAN frame data, or `None` if disabled.
    /// In counter mode the first byte is the counter; otherwise DLC=0 (all zeros).
    pub fn generate(&mut self) -> Option<[u8; 8]> {
        if !self.enabled {
            return None;
        }
        self.total_syncs += 1;
        let mut frame = [0u8; 8];
        if self.counter_overflow > 0 {
            frame[0] = self.counter;
            // Advance counter
            self.counter += 1;
            if self.counter > self.counter_overflow {
                self.counter = 1;
            }
        }
        Some(frame)
    }

    /// Reset counter to initial state.
    pub fn reset_counter(&mut self) {
        self.counter = if self.counter_overflow > 0 { 1 } else { 0 };
    }

    /// Configure counter overflow value (1–240, 0=disable counter).
    pub fn set_counter_overflow(&mut self, overflow: u8) {
        if overflow == 0 {
            self.counter_overflow = 0;
            self.counter = 0;
        } else {
            self.counter_overflow = overflow.clamp(1, SYNC_COUNTER_MAX);
            self.counter = 1;
        }
    }
}

impl Default for SyncProducer {
    fn default() -> Self {
        Self::new(SYNC_COB_ID)
    }
}

/// Registered SYNC callback entry.
#[derive(Clone, Copy)]
struct SyncCallback {
    /// Synchronous counter period (fire every N syncs).
    period: u8,
    /// Callback function.
    callback: fn(u8),
}

/// SYNC consumer: receives SYNC frames and triggers callbacks.
///
/// Up to `CALLBACKS` callbacks can be registered. Each callback is called
/// with the current SYNC counter value.
pub struct SyncConsumer<const CALLBACKS: usize = 8> {
    /// Expected COB-ID.
    cob_id: u16,
    /// Current local SYNC counter.
    counter: u8,
    /// Counter overflow value (matches producer's setting).
    counter_overflow: u8,
    /// Total SYNCs received.
    total_received: u64,
    /// Registered callbacks.
    callbacks: [Option<SyncCallback>; CALLBACKS],
    /// Number of registered callbacks.
    n_callbacks: usize,
    /// Last received counter value (for overflow detection).
    last_counter: u8,
    /// Counter overflow events.
    overflow_count: u32,
}

impl<const CALLBACKS: usize> SyncConsumer<CALLBACKS> {
    /// Create a new SYNC consumer.
    pub fn new(cob_id: u16) -> Self {
        Self {
            cob_id,
            counter: 0,
            counter_overflow: 0,
            total_received: 0,
            callbacks: [None; CALLBACKS],
            n_callbacks: 0,
            last_counter: 0,
            overflow_count: 0,
        }
    }

    /// Set the counter overflow value (must match producer's setting).
    pub fn set_counter_overflow(&mut self, overflow: u8) {
        self.counter_overflow = overflow;
    }

    /// Register a callback function.
    ///
    /// `period` = 0 means fire on every SYNC. `period` > 0 means fire every
    /// `period` SYNCs.
    ///
    /// Returns false if the callback table is full.
    pub fn register_callback(&mut self, callback: fn(u8), period: u8) -> bool {
        if self.n_callbacks >= CALLBACKS {
            return false;
        }
        self.callbacks[self.n_callbacks] = Some(SyncCallback { period, callback });
        self.n_callbacks += 1;
        true
    }

    /// Process a received SYNC frame.
    ///
    /// `frame_data` is the raw 8-byte CAN frame. Returns `true` if the frame
    /// was a valid SYNC for this consumer's COB-ID.
    pub fn on_sync(&mut self, frame_data: &[u8; 8]) -> bool {
        self.total_received += 1;
        self.counter += 1;

        let recv_counter = frame_data[0];

        // Detect counter overflow
        if self.counter_overflow > 0 && recv_counter < self.last_counter {
            self.overflow_count += 1;
        }
        self.last_counter = recv_counter;

        // Fire callbacks
        for slot in self.callbacks[..self.n_callbacks].iter().flatten() {
            let fire = if slot.period == 0 {
                true
            } else {
                self.total_received % (slot.period as u64) == 0
            };
            if fire {
                (slot.callback)(recv_counter);
            }
        }
        true
    }

    /// COB-ID.
    pub fn cob_id(&self) -> u16 {
        self.cob_id
    }

    /// Total SYNCs received.
    pub fn total_received(&self) -> u64 {
        self.total_received
    }

    /// Number of counter overflow events detected.
    pub fn overflow_count(&self) -> u32 {
        self.overflow_count
    }

    /// Local SYNC counter.
    pub fn counter(&self) -> u8 {
        self.counter
    }

    /// Number of registered callbacks.
    pub fn n_callbacks(&self) -> usize {
        self.n_callbacks
    }
}

impl<const CALLBACKS: usize> Default for SyncConsumer<CALLBACKS> {
    fn default() -> Self {
        Self::new(SYNC_COB_ID)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU32, Ordering};

    static CALLBACK_COUNT: AtomicU32 = AtomicU32::new(0);

    fn test_callback(_counter: u8) {
        CALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    #[test]
    fn test_sync_producer_generate() {
        let mut prod = SyncProducer::new(SYNC_COB_ID);
        let frame = prod.generate().unwrap();
        assert_eq!(frame, [0u8; 8]); // no counter
        assert_eq!(prod.total_syncs(), 1);
    }

    #[test]
    fn test_sync_producer_counter_mode() {
        let mut prod = SyncProducer::with_counter(SYNC_COB_ID, 3);
        // Generate 4 SYNCs and check counter wraps at 3→1
        let f1 = prod.generate().unwrap();
        assert_eq!(f1[0], 1);
        let f2 = prod.generate().unwrap();
        assert_eq!(f2[0], 2);
        let f3 = prod.generate().unwrap();
        assert_eq!(f3[0], 3);
        let f4 = prod.generate().unwrap();
        assert_eq!(f4[0], 1); // wrapped
    }

    #[test]
    fn test_sync_producer_disabled() {
        let mut prod = SyncProducer::new(SYNC_COB_ID);
        prod.set_enabled(false);
        assert!(prod.generate().is_none());
    }

    #[test]
    fn test_sync_consumer_callback() {
        CALLBACK_COUNT.store(0, Ordering::Relaxed);
        let mut cons = SyncConsumer::<4>::new(SYNC_COB_ID);
        cons.register_callback(test_callback, 0);
        let frame = [0u8; 8];
        cons.on_sync(&frame);
        cons.on_sync(&frame);
        assert_eq!(CALLBACK_COUNT.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_sync_consumer_period_callback() {
        CALLBACK_COUNT.store(0, Ordering::Relaxed);
        let mut cons = SyncConsumer::<4>::new(SYNC_COB_ID);
        // Only call every 2 syncs
        cons.register_callback(test_callback, 2);
        let frame = [0u8; 8];
        for _ in 0..6 {
            cons.on_sync(&frame);
        }
        // Should fire on sync 2, 4, 6
        assert_eq!(CALLBACK_COUNT.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_sync_counter_overflow_detection() {
        let mut cons = SyncConsumer::<4>::new(SYNC_COB_ID);
        cons.set_counter_overflow(3);
        // Simulate counter values: 1, 2, 3, 1 (overflow on 4th)
        cons.on_sync(&[1, 0, 0, 0, 0, 0, 0, 0]);
        cons.on_sync(&[2, 0, 0, 0, 0, 0, 0, 0]);
        cons.on_sync(&[3, 0, 0, 0, 0, 0, 0, 0]);
        cons.on_sync(&[1, 0, 0, 0, 0, 0, 0, 0]); // overflow: 1 < 3
        assert_eq!(cons.overflow_count(), 1);
    }

    #[test]
    fn test_producer_consumer_roundtrip() {
        let mut prod = SyncProducer::with_counter(SYNC_COB_ID, 5);
        let mut cons = SyncConsumer::<4>::new(SYNC_COB_ID);
        cons.set_counter_overflow(5);

        for i in 1..=5 {
            let frame = prod.generate().unwrap();
            assert_eq!(frame[0], i);
            cons.on_sync(&frame);
        }
        // After 5 syncs, next should wrap to 1
        let wrap_frame = prod.generate().unwrap();
        assert_eq!(wrap_frame[0], 1);
        assert_eq!(prod.total_syncs(), 6);
        assert_eq!(cons.total_received(), 5);
    }
}
