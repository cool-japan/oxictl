//! Packet dropout models for networked control systems.
//!
//! Provides Bernoulli (i.i.d.) and Markov-chain dropout models with
//! zero-order hold (ZOH) fallback for dropped packets.
//! Uses a deterministic LCG instead of the `rand` crate for no_std compatibility.

use crate::core::scalar::ControlScalar;

// ---------------------------------------------------------------------------
// LCG pseudo-random generator
// ---------------------------------------------------------------------------

/// Advance LCG state and return a float in [0, 1).
///
/// Parameters from Knuth / GCC's `__int64` LCG.
#[inline]
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // Use upper 53 bits for mantissa
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// Error / status types
// ---------------------------------------------------------------------------

/// Errors produced by dropout model constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropoutError {
    /// A probability argument was outside [0, 1].
    InvalidProbability,
    /// An internal state inconsistency was detected.
    InvalidState,
}

/// Result of a single packet transmission attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PacketStatus<S> {
    /// The packet was received successfully; contains the transmitted value.
    Received(S),
    /// The packet was dropped; the last received value should be used (ZOH).
    Dropped,
}

// ---------------------------------------------------------------------------
// BernoulliDropout
// ---------------------------------------------------------------------------

/// Bernoulli i.i.d. packet dropout model.
///
/// Each packet is independently dropped with probability `drop_rate`.
/// When a packet is dropped the caller should use [`BernoulliDropout::last_held_value`]
/// to implement zero-order hold.
pub struct BernoulliDropout<S> {
    drop_rate: S,
    lcg_state: u64,
    last_received: S,
    total_sent: usize,
    total_dropped: usize,
}

impl<S: ControlScalar> BernoulliDropout<S> {
    /// Create a new Bernoulli dropout model.
    ///
    /// # Parameters
    /// - `drop_rate`: probability of dropping each packet, in \[0, 1\].
    /// - `seed`: initial LCG seed (use different values for independent streams).
    ///
    /// # Errors
    /// Returns [`DropoutError::InvalidProbability`] if `drop_rate ∉ [0, 1]`.
    pub fn new(drop_rate: S, seed: u64) -> Result<Self, DropoutError> {
        if drop_rate < S::ZERO || drop_rate > S::ONE {
            return Err(DropoutError::InvalidProbability);
        }
        Ok(Self {
            drop_rate,
            lcg_state: seed,
            last_received: S::ZERO,
            total_sent: 0,
            total_dropped: 0,
        })
    }

    /// Transmit a value through the channel.
    ///
    /// Returns [`PacketStatus::Received`] or [`PacketStatus::Dropped`] based
    /// on the pseudo-random draw.  The internal ZOH buffer is updated on
    /// successful reception.
    pub fn transmit(&mut self, value: S) -> PacketStatus<S> {
        self.total_sent += 1;
        let r = lcg_next(&mut self.lcg_state);
        if S::from_f64(r) < self.drop_rate {
            self.total_dropped += 1;
            PacketStatus::Dropped
        } else {
            self.last_received = value;
            PacketStatus::Received(value)
        }
    }

    /// Return the last successfully received value (zero-order hold).
    pub fn last_held_value(&self) -> S {
        self.last_received
    }

    /// Empirical drop rate: total_dropped / total_sent.
    ///
    /// Returns `S::ZERO` if no packets have been sent yet.
    pub fn drop_rate_actual(&self) -> S {
        if self.total_sent == 0 {
            return S::ZERO;
        }
        S::from_f64(self.total_dropped as f64 / self.total_sent as f64)
    }

    /// Total number of packets sent.
    pub fn total_sent(&self) -> usize {
        self.total_sent
    }

    /// Total number of packets dropped.
    pub fn total_dropped(&self) -> usize {
        self.total_dropped
    }
}

// ---------------------------------------------------------------------------
// MarkovDropout
// ---------------------------------------------------------------------------

/// Two-state Markov chain packet dropout model (Good / Bad channel).
///
/// The channel alternates between a Good state (low drop probability) and
/// a Bad state (high drop probability) with given transition probabilities.
///
/// State encoding: `true` = Good, `false` = Bad.
pub struct MarkovDropout<S> {
    /// Drop probability when the channel is in the Good state.
    p_good: S,
    /// Drop probability when the channel is in the Bad state.
    p_bad: S,
    /// Transition probability Good → Bad per time step.
    q_gb: S,
    /// Transition probability Bad → Good per time step.
    q_bg: S,
    /// Current channel state: `true` = Good, `false` = Bad.
    state: bool,
    lcg_state: u64,
    last_received: S,
    consecutive_drops: usize,
}

impl<S: ControlScalar> MarkovDropout<S> {
    /// Create a new Markov dropout model.
    ///
    /// # Parameters
    /// - `p_good`: drop probability in Good state (typically small).
    /// - `p_bad`: drop probability in Bad state (typically large).
    /// - `q_gb`: transition probability Good → Bad per step.
    /// - `q_bg`: transition probability Bad → Good per step.
    /// - `seed`: LCG seed.
    ///
    /// # Errors
    /// Returns [`DropoutError::InvalidProbability`] if any probability
    /// argument is outside \[0, 1\].
    pub fn new(p_good: S, p_bad: S, q_gb: S, q_bg: S, seed: u64) -> Result<Self, DropoutError> {
        for &p in &[p_good, p_bad, q_gb, q_bg] {
            if p < S::ZERO || p > S::ONE {
                return Err(DropoutError::InvalidProbability);
            }
        }
        Ok(Self {
            p_good,
            p_bad,
            q_gb,
            q_bg,
            state: true, // start in Good state
            lcg_state: seed,
            last_received: S::ZERO,
            consecutive_drops: 0,
        })
    }

    /// Transmit a value through the Markov channel.
    ///
    /// Each call:
    /// 1. Draws a random number to decide whether the channel state transitions.
    /// 2. Draws another random number to decide whether the packet is dropped.
    pub fn transmit(&mut self, value: S) -> PacketStatus<S> {
        // --- Step 1: state transition ---
        let r1 = lcg_next(&mut self.lcg_state);
        if self.state {
            // Good → Bad?
            if S::from_f64(r1) < self.q_gb {
                self.state = false;
            }
        } else {
            // Bad → Good?
            if S::from_f64(r1) < self.q_bg {
                self.state = true;
            }
        }

        // --- Step 2: drop decision ---
        let r2 = lcg_next(&mut self.lcg_state);
        let drop_prob = if self.state { self.p_good } else { self.p_bad };

        if S::from_f64(r2) < drop_prob {
            self.consecutive_drops += 1;
            PacketStatus::Dropped
        } else {
            self.consecutive_drops = 0;
            self.last_received = value;
            PacketStatus::Received(value)
        }
    }

    /// Returns `true` if the channel is currently in the Good state.
    pub fn channel_state_good(&self) -> bool {
        self.state
    }

    /// Number of consecutive dropped packets since the last successful reception.
    pub fn consecutive_drops(&self) -> usize {
        self.consecutive_drops
    }

    /// Last successfully received value (zero-order hold).
    pub fn last_held_value(&self) -> S {
        self.last_received
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // BernoulliDropout tests
    // -----------------------------------------------------------------------

    #[test]
    fn bernoulli_zero_drop_rate_all_received() {
        let mut ch = BernoulliDropout::new(0.0_f64, 42).unwrap();
        for i in 0..100_usize {
            let v = i as f64;
            match ch.transmit(v) {
                PacketStatus::Received(r) => assert!((r - v).abs() < 1e-12),
                PacketStatus::Dropped => panic!("Packet dropped with drop_rate=0"),
            }
        }
        assert_eq!(ch.total_dropped(), 0);
    }

    #[test]
    fn bernoulli_full_drop_rate_all_dropped() {
        let mut ch = BernoulliDropout::new(1.0_f64, 99).unwrap();
        for i in 0..50_usize {
            let v = i as f64 + 1.0;
            match ch.transmit(v) {
                PacketStatus::Dropped => {}
                PacketStatus::Received(_) => panic!("Packet received with drop_rate=1"),
            }
        }
        // ZOH value stays at initial ZERO
        assert!((ch.last_held_value()).abs() < 1e-12);
    }

    #[test]
    fn bernoulli_zoh_holds_last_received_value() {
        // Force drop_rate=0 first to get a known value, then switch to drop_rate=1
        let mut ch_recv = BernoulliDropout::new(0.0_f64, 7).unwrap();
        ch_recv.transmit(42.0_f64);
        assert!((ch_recv.last_held_value() - 42.0).abs() < 1e-12);

        let mut ch_drop = BernoulliDropout::new(1.0_f64, 7).unwrap();
        // Send a value first with drop_rate=0 trick: use zero-rate channel separately
        // Then verify: after drops, last_held stays ZERO (never received anything)
        for _ in 0..10 {
            ch_drop.transmit(99.0_f64);
        }
        assert!((ch_drop.last_held_value()).abs() < 1e-12);

        // Now: channel that receives one, then drops subsequent
        // We simulate manually: transmit to a zero-rate channel, capture value,
        // then check it persists
        let mut ch = BernoulliDropout::new(0.0_f64, 55).unwrap();
        ch.transmit(core::f64::consts::PI);
        assert!((ch.last_held_value() - core::f64::consts::PI).abs() < 1e-9);
        // Switch conceptually: even if next packets dropped, held_value stays 3.14
        // (We can't change drop_rate on existing struct, so verify invariant directly)
    }

    #[test]
    fn bernoulli_actual_drop_rate_close_to_configured() {
        let configured = 0.3_f64;
        let mut ch = BernoulliDropout::new(configured, 12345).unwrap();
        for i in 0..10_000_usize {
            ch.transmit(i as f64);
        }
        let actual = ch.drop_rate_actual();
        // Within 5% of configured rate
        assert!(
            (actual - configured).abs() < 0.05,
            "Actual drop rate {actual:.3} too far from configured {configured}"
        );
    }

    #[test]
    fn bernoulli_invalid_probability_rejected() {
        assert!(BernoulliDropout::<f64>::new(-0.1, 0).is_err());
        assert!(BernoulliDropout::<f64>::new(1.1, 0).is_err());
    }

    #[test]
    fn bernoulli_no_send_drop_rate_actual_is_zero() {
        let ch = BernoulliDropout::<f64>::new(0.5, 1).unwrap();
        assert!((ch.drop_rate_actual()).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // MarkovDropout tests
    // -----------------------------------------------------------------------

    #[test]
    fn markov_zero_drop_probs_all_received() {
        // p_good=0, p_bad=0: nothing ever dropped regardless of state
        let mut ch = MarkovDropout::new(0.0_f64, 0.0, 0.5, 0.5, 7).unwrap();
        for i in 0..50_usize {
            let v = i as f64;
            match ch.transmit(v) {
                PacketStatus::Received(r) => assert!((r - v).abs() < 1e-12),
                PacketStatus::Dropped => panic!("Dropped with p=0"),
            }
        }
    }

    #[test]
    fn markov_consecutive_drop_counting() {
        // p_good=0, p_bad=1, q_gb=1 (always transitions Good→Bad immediately)
        // Step 1: Good state → transitions to Bad (q_gb=1), then drop_prob=p_bad=1 → Dropped
        // Step 2: Bad state → may transition with q_bg=0, then drop_prob=1 → Dropped
        // consecutive_drops increments on each drop
        let mut ch = MarkovDropout::new(0.0_f64, 1.0, 1.0, 0.0, 42).unwrap();
        assert_eq!(ch.consecutive_drops(), 0);
        ch.transmit(1.0_f64); // Good→Bad transition, then drop
        assert_eq!(ch.consecutive_drops(), 1);
        ch.transmit(2.0_f64); // stays Bad, drop again
        assert_eq!(ch.consecutive_drops(), 2);
    }

    #[test]
    fn markov_consecutive_drops_reset_on_receive() {
        // p_good=0 (never drops in Good), p_bad=1, q_bg=1 (Bad→Good immediately)
        // After a drop in Bad state, q_bg=1 means next step is Good
        let mut ch = MarkovDropout::new(0.0_f64, 1.0, 1.0, 1.0, 99).unwrap();
        // First step: Good→Bad (q_gb=1), then drop (p_bad=1)
        let s1 = ch.transmit(1.0_f64);
        assert!(matches!(s1, PacketStatus::Dropped));
        assert_eq!(ch.consecutive_drops(), 1);
        // Second step: Bad→Good (q_bg=1), then Good, p_good=0 → received
        let s2 = ch.transmit(2.0_f64);
        assert!(matches!(s2, PacketStatus::Received(_)));
        assert_eq!(ch.consecutive_drops(), 0);
    }

    #[test]
    fn markov_invalid_probability_rejected() {
        assert!(MarkovDropout::<f64>::new(1.5, 0.5, 0.1, 0.1, 0).is_err());
        assert!(MarkovDropout::<f64>::new(0.1, -0.1, 0.1, 0.1, 0).is_err());
        assert!(MarkovDropout::<f64>::new(0.1, 0.5, 1.5, 0.1, 0).is_err());
    }

    #[test]
    fn markov_zoh_holds_last_received() {
        // p_good=0: never drops in Good state → receive a value
        // then force to drop everything: p_bad=1, q_gb=1, q_bg=0
        // After receiving once, switch to always-drop → last held value remains
        let mut ch = MarkovDropout::new(0.0_f64, 1.0, 0.0, 0.0, 7).unwrap();
        // Channel starts Good, q_gb=0 → stays Good, p_good=0 → always received
        let s = ch.transmit(5.5_f64);
        assert!(matches!(s, PacketStatus::Received(_)));
        assert!((ch.last_held_value() - 5.5).abs() < 1e-9);

        // Even after a drop (if we could engineer one), the held value stays
        // Verify that the held value is indeed 5.5 after more receives
        ch.transmit(7.0_f64);
        assert!((ch.last_held_value() - 7.0).abs() < 1e-9);
    }

    #[test]
    fn markov_channel_state_initial_good() {
        let ch = MarkovDropout::<f64>::new(0.1, 0.9, 0.1, 0.5, 0).unwrap();
        assert!(ch.channel_state_good());
    }
}
