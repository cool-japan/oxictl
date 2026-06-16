//! QoS policy structs for SEDP endpoint discovery.
//!
//! All policies wire-encode as CDR in the ParameterList of endpoint builtin topic data.

use crate::protocol::dds::byte_cursor::{ByteCursor, ByteWriter};
use crate::protocol::dds::error::RtpsError;
use crate::protocol::dds::types::time::Duration;

// ─── Duration constants for defaults ─────────────────────────────────────────

/// INFINITE duration: seconds = i32::MAX, fraction = u32::MAX.
const DURATION_INFINITE: Duration = Duration {
    seconds: 0x7FFF_FFFF,
    fraction: 0xFFFF_FFFF,
};

/// 100 ms in NTP fraction units: 0.1 * 2^32 ≈ 0x1999_9999.
const FRACTION_100MS: u32 = 0x1999_9999;

// ─── ReliabilityQosPolicy ────────────────────────────────────────────────────

/// Kind for `ReliabilityQosPolicy`. Values per OMG DDS spec 2.2.3.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReliabilityKind {
    /// 1 = BEST_EFFORT
    BestEffort = 1,
    /// 2 = RELIABLE
    Reliable = 2,
}

/// DDS QoS policy governing delivery reliability and blocking time.
///
/// CDR wire format (12 bytes): `[kind: i32][max_blocking_time.seconds: i32][max_blocking_time.fraction: u32]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReliabilityQosPolicy {
    pub kind: ReliabilityKind,
    pub max_blocking_time: Duration,
}

impl ReliabilityQosPolicy {
    /// Reader.kind ≤ Writer.kind (BestEffort < Reliable).
    ///
    /// `self` is the reader; `writer` is the offering writer.
    pub fn is_compatible_with(&self, writer: &Self) -> bool {
        (self.kind as i32) <= (writer.kind as i32)
    }

    /// RELIABLE delivery with a 100 ms max blocking time (ROS2 default).
    pub fn reliable() -> Self {
        Self {
            kind: ReliabilityKind::Reliable,
            max_blocking_time: Duration {
                seconds: 0,
                fraction: FRACTION_100MS,
            },
        }
    }

    /// BEST_EFFORT delivery with no blocking time.
    pub fn best_effort() -> Self {
        Self {
            kind: ReliabilityKind::BestEffort,
            max_blocking_time: Duration::zero(),
        }
    }

    /// Parse from CDR bytes (12 bytes expected in `cur`'s endianness).
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let kind_raw = cur.read_i32()?;
        let max_blocking_time = Duration::parse(cur)?;
        let kind = match kind_raw {
            2 => ReliabilityKind::Reliable,
            _ => ReliabilityKind::BestEffort, // unknown → default to BestEffort
        };
        Ok(Self {
            kind,
            max_blocking_time,
        })
    }

    /// Serialize to CDR bytes using `w`'s endianness (12 bytes written).
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.kind as i32)?;
        self.max_blocking_time.serialize(w)
    }
}

impl Default for ReliabilityQosPolicy {
    fn default() -> Self {
        Self::reliable()
    }
}

// ─── HistoryQosPolicy ────────────────────────────────────────────────────────

/// Kind for `HistoryQosPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryKind {
    /// 0 = KEEP_LAST (bounded depth)
    KeepLast = 0,
    /// 1 = KEEP_ALL (unlimited)
    KeepAll = 1,
}

/// DDS QoS policy governing sample history storage.
///
/// CDR wire format (8 bytes): `[kind: i32][depth: i32]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryQosPolicy {
    pub kind: HistoryKind,
    /// Depth only meaningful when `kind == KeepLast`.
    pub depth: i32,
}

impl HistoryQosPolicy {
    /// KEEP_LAST with depth 1 (ROS2 default).
    pub fn keep_last_1() -> Self {
        Self {
            kind: HistoryKind::KeepLast,
            depth: 1,
        }
    }

    /// Parse from CDR bytes (8 bytes expected).
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let kind_raw = cur.read_i32()?;
        let depth = cur.read_i32()?;
        let kind = match kind_raw {
            1 => HistoryKind::KeepAll,
            _ => HistoryKind::KeepLast,
        };
        Ok(Self { kind, depth })
    }

    /// Serialize to CDR bytes (8 bytes written).
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.kind as i32)?;
        w.write_i32(self.depth)
    }
}

impl Default for HistoryQosPolicy {
    fn default() -> Self {
        Self::keep_last_1()
    }
}

// ─── DurabilityQosPolicy ─────────────────────────────────────────────────────

/// Kind for `DurabilityQosPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityKind {
    /// 0 = VOLATILE
    Volatile = 0,
    /// 1 = TRANSIENT_LOCAL
    TransientLocal = 1,
    /// 2 = TRANSIENT
    Transient = 2,
    /// 3 = PERSISTENT
    Persistent = 3,
}

/// DDS QoS policy governing data persistence.
///
/// CDR wire format (4 bytes): `[kind: i32]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurabilityQosPolicy {
    pub kind: DurabilityKind,
}

impl DurabilityQosPolicy {
    /// Reader.kind ≤ Writer.kind (Volatile < TransientLocal < Transient < Persistent).
    ///
    /// `self` is the reader; `writer` is the offering writer.
    pub fn is_compatible_with(&self, writer: &Self) -> bool {
        (self.kind as i32) <= (writer.kind as i32)
    }

    /// VOLATILE (ROS2 default).
    pub fn volatile() -> Self {
        Self {
            kind: DurabilityKind::Volatile,
        }
    }

    /// Parse from CDR bytes (4 bytes expected).
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let kind_raw = cur.read_i32()?;
        let kind = match kind_raw {
            1 => DurabilityKind::TransientLocal,
            2 => DurabilityKind::Transient,
            3 => DurabilityKind::Persistent,
            _ => DurabilityKind::Volatile,
        };
        Ok(Self { kind })
    }

    /// Serialize to CDR bytes (4 bytes written).
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.kind as i32)
    }
}

impl Default for DurabilityQosPolicy {
    fn default() -> Self {
        Self::volatile()
    }
}

// ─── LivelinessQosPolicy ─────────────────────────────────────────────────────

/// Kind for `LivelinessQosPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LivelinessKind {
    /// 0 = AUTOMATIC
    Automatic = 0,
    /// 1 = MANUAL_BY_PARTICIPANT
    ManualByParticipant = 1,
    /// 2 = MANUAL_BY_TOPIC
    ManualByTopic = 2,
}

/// DDS QoS policy governing liveliness detection.
///
/// CDR wire format (12 bytes): `[kind: i32][lease_duration.seconds: i32][lease_duration.fraction: u32]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LivelinessQosPolicy {
    pub kind: LivelinessKind,
    pub lease_duration: Duration,
}

impl LivelinessQosPolicy {
    /// Reader.kind ≤ Writer.kind AND Reader.lease ≥ Writer.lease.
    ///
    /// `self` is the reader; `writer` is the offering writer.
    pub fn is_compatible_with(&self, writer: &Self) -> bool {
        if (self.kind as i32) > (writer.kind as i32) {
            return false;
        }
        cmp_duration(self.lease_duration, writer.lease_duration) != core::cmp::Ordering::Less
    }

    /// AUTOMATIC liveliness with INFINITE lease duration (ROS2 default).
    pub fn automatic_infinite() -> Self {
        Self {
            kind: LivelinessKind::Automatic,
            lease_duration: DURATION_INFINITE,
        }
    }

    /// Parse from CDR bytes (12 bytes expected).
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let kind_raw = cur.read_i32()?;
        let lease_duration = Duration::parse(cur)?;
        let kind = match kind_raw {
            1 => LivelinessKind::ManualByParticipant,
            2 => LivelinessKind::ManualByTopic,
            _ => LivelinessKind::Automatic,
        };
        Ok(Self {
            kind,
            lease_duration,
        })
    }

    /// Serialize to CDR bytes (12 bytes written).
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.kind as i32)?;
        self.lease_duration.serialize(w)
    }
}

impl Default for LivelinessQosPolicy {
    fn default() -> Self {
        Self::automatic_infinite()
    }
}

// ─── DeadlineQosPolicy ───────────────────────────────────────────────────────

/// DDS QoS policy governing worst-case deadline between successive samples.
///
/// CDR wire format (8 bytes): `[period.seconds: i32][period.fraction: u32]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeadlineQosPolicy {
    pub period: Duration,
}

impl DeadlineQosPolicy {
    /// Reader.period ≥ Writer.period (writer must publish at least as often as reader requires).
    ///
    /// `self` is the reader; `writer` is the offering writer.
    pub fn is_compatible_with(&self, writer: &Self) -> bool {
        cmp_duration(self.period, writer.period) != core::cmp::Ordering::Less
    }

    /// INFINITE period (ROS2 default — no deadline).
    pub fn infinite() -> Self {
        Self {
            period: DURATION_INFINITE,
        }
    }

    /// Parse from CDR bytes (8 bytes expected).
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let period = Duration::parse(cur)?;
        Ok(Self { period })
    }

    /// Serialize to CDR bytes (8 bytes written).
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        self.period.serialize(w)
    }
}

impl Default for DeadlineQosPolicy {
    fn default() -> Self {
        Self::infinite()
    }
}

// ─── LifespanQosPolicy ───────────────────────────────────────────────────────

/// DDS QoS policy governing maximum lifetime of a data sample.
///
/// CDR wire format (8 bytes): `[duration.seconds: i32][duration.fraction: u32]`
///
/// This policy is **not** Request-vs-Offer; it is applied locally by the writer to
/// discard samples that have lived past their expiry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifespanQosPolicy {
    /// Maximum sample age. `DURATION_INFINITE` means samples never expire.
    pub duration: Duration,
}

impl LifespanQosPolicy {
    /// INFINITE duration (default — samples never expire).  ROS2 default.
    pub fn infinite() -> Self {
        Self {
            duration: DURATION_INFINITE,
        }
    }

    /// Parse from CDR bytes (8 bytes expected).
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let duration = Duration::parse(cur)?;
        Ok(Self { duration })
    }

    /// Serialize to CDR bytes (8 bytes written).
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        self.duration.serialize(w)
    }
}

impl Default for LifespanQosPolicy {
    fn default() -> Self {
        Self::infinite()
    }
}

// ─── OwnershipQosPolicy ──────────────────────────────────────────────────────

/// Kind for `OwnershipQosPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipKind {
    /// 0 = SHARED — multiple writers are allowed; all samples are delivered.
    Shared = 0,
    /// 1 = EXCLUSIVE — only the highest-strength writer delivers samples.
    Exclusive = 1,
}

/// DDS QoS policy governing exclusive vs shared ownership of a topic instance.
///
/// CDR wire format (4 bytes): `[kind: i32]`
///
/// RxO rule: reader and writer must have the **same** ownership kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnershipQosPolicy {
    pub kind: OwnershipKind,
}

impl OwnershipQosPolicy {
    /// SHARED ownership (ROS2 default).
    pub fn shared() -> Self {
        Self {
            kind: OwnershipKind::Shared,
        }
    }

    /// EXCLUSIVE ownership — only the strongest writer publishes.
    pub fn exclusive() -> Self {
        Self {
            kind: OwnershipKind::Exclusive,
        }
    }

    /// RxO: reader and writer must have the same ownership kind.
    ///
    /// `self` is the reader; `writer` is the offering writer.
    pub fn is_compatible_with(&self, writer: &Self) -> bool {
        self.kind == writer.kind
    }

    /// Parse from CDR bytes (4 bytes expected).
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let kind_raw = cur.read_i32()?;
        let kind = match kind_raw {
            1 => OwnershipKind::Exclusive,
            _ => OwnershipKind::Shared,
        };
        Ok(Self { kind })
    }

    /// Serialize to CDR bytes (4 bytes written).
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.kind as i32)
    }
}

impl Default for OwnershipQosPolicy {
    fn default() -> Self {
        Self::shared()
    }
}

// ─── OwnershipStrengthQosPolicy ──────────────────────────────────────────────

/// DDS QoS policy expressing a writer's ownership strength when `OwnershipKind::Exclusive`.
///
/// CDR wire format (4 bytes): `[value: i32]`
///
/// This is a **writer-only** policy and is **not** subject to Request-vs-Offer matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OwnershipStrengthQosPolicy {
    /// Strength value.  Higher value wins exclusive access.  Default is 0.
    pub value: i32,
}

impl OwnershipStrengthQosPolicy {
    /// Construct with the given strength value.
    pub fn new(value: i32) -> Self {
        Self { value }
    }

    /// Parse from CDR bytes (4 bytes expected).
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let value = cur.read_i32()?;
        Ok(Self { value })
    }

    /// Serialize to CDR bytes (4 bytes written).
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.value)
    }
}

// ─── DestinationOrderQosPolicy ───────────────────────────────────────────────

/// Kind for `DestinationOrderQosPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DestinationOrderKind {
    /// 0 = BY_RECEPTION_TIMESTAMP (default) — order samples by the subscriber's clock.
    ByReceptionTimestamp = 0,
    /// 1 = BY_SOURCE_TIMESTAMP — order samples by the publisher's source timestamp.
    BySourceTimestamp = 1,
}

/// DDS QoS policy governing how samples are ordered within the same instance.
///
/// CDR wire format (4 bytes): `[kind: i32]`
///
/// RxO rule: reader.kind ≤ writer.kind (`ByReceptionTimestamp` < `BySourceTimestamp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DestinationOrderQosPolicy {
    pub kind: DestinationOrderKind,
}

impl DestinationOrderQosPolicy {
    /// BY_RECEPTION_TIMESTAMP (ROS2 default).
    pub fn by_reception_timestamp() -> Self {
        Self {
            kind: DestinationOrderKind::ByReceptionTimestamp,
        }
    }

    /// BY_SOURCE_TIMESTAMP.
    pub fn by_source_timestamp() -> Self {
        Self {
            kind: DestinationOrderKind::BySourceTimestamp,
        }
    }

    /// RxO: reader.kind ≤ writer.kind (`ByReception` < `BySource`).
    ///
    /// A reader that only requires reception-timestamp ordering is satisfied by a
    /// writer that offers source-timestamp ordering (which is strictly stronger).
    ///
    /// `self` is the reader; `writer` is the offering writer.
    pub fn is_compatible_with(&self, writer: &Self) -> bool {
        (self.kind as i32) <= (writer.kind as i32)
    }

    /// Parse from CDR bytes (4 bytes expected).
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let kind_raw = cur.read_i32()?;
        let kind = match kind_raw {
            1 => DestinationOrderKind::BySourceTimestamp,
            _ => DestinationOrderKind::ByReceptionTimestamp,
        };
        Ok(Self { kind })
    }

    /// Serialize to CDR bytes (4 bytes written).
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.kind as i32)
    }
}

impl Default for DestinationOrderQosPolicy {
    fn default() -> Self {
        Self::by_reception_timestamp()
    }
}

// ─── ResourceLimitsQosPolicy ─────────────────────────────────────────────────

/// DDS QoS policy governing resource capacity bounds on a reader or writer.
///
/// CDR wire format (12 bytes):
/// `[max_samples: i32][max_instances: i32][max_samples_per_instance: i32]`
///
/// The sentinel `-1` represents `LENGTH_UNLIMITED` for each field.
///
/// This policy is **not** subject to Request-vs-Offer matching; it is a local
/// resource-management hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceLimitsQosPolicy {
    /// Maximum total number of samples held in the history cache.  `-1` = unlimited.
    pub max_samples: i32,
    /// Maximum number of instances tracked.  `-1` = unlimited.
    pub max_instances: i32,
    /// Maximum samples stored per instance.  `-1` = unlimited.
    pub max_samples_per_instance: i32,
}

impl ResourceLimitsQosPolicy {
    /// `LENGTH_UNLIMITED` on all fields (ROS2 default).
    pub fn unlimited() -> Self {
        Self {
            max_samples: -1,
            max_instances: -1,
            max_samples_per_instance: -1,
        }
    }

    /// Parse from CDR bytes (12 bytes expected).
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, RtpsError> {
        let max_samples = cur.read_i32()?;
        let max_instances = cur.read_i32()?;
        let max_samples_per_instance = cur.read_i32()?;
        Ok(Self {
            max_samples,
            max_instances,
            max_samples_per_instance,
        })
    }

    /// Serialize to CDR bytes (12 bytes written).
    pub fn serialize(&self, w: &mut ByteWriter<'_>) -> Result<(), RtpsError> {
        w.write_i32(self.max_samples)?;
        w.write_i32(self.max_instances)?;
        w.write_i32(self.max_samples_per_instance)
    }
}

impl Default for ResourceLimitsQosPolicy {
    fn default() -> Self {
        Self::unlimited()
    }
}

// ─── Duration comparison helper ──────────────────────────────────────────────

/// Compare two [`Duration`] values for RxO checks.
///
/// `INFINITE` (`seconds = i32::MAX, fraction = u32::MAX`) is treated as greater
/// than any finite value. For two finite values: `seconds` is compared first,
/// then `fraction`.
pub(crate) fn cmp_duration(a: Duration, b: Duration) -> core::cmp::Ordering {
    const INF_S: i32 = 0x7FFF_FFFF;
    const INF_F: u32 = 0xFFFF_FFFF;
    let a_inf = a.seconds == INF_S && a.fraction == INF_F;
    let b_inf = b.seconds == INF_S && b.fraction == INF_F;
    match (a_inf, b_inf) {
        (true, true) => core::cmp::Ordering::Equal,
        (true, false) => core::cmp::Ordering::Greater,
        (false, true) => core::cmp::Ordering::Less,
        (false, false) => {
            let sc = a.seconds.cmp(&b.seconds);
            if sc != core::cmp::Ordering::Equal {
                sc
            } else {
                a.fraction.cmp(&b.fraction)
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const INFINITE: Duration = Duration {
        seconds: 0x7FFF_FFFF,
        fraction: 0xFFFF_FFFF,
    };

    fn dur(seconds: i32, fraction: u32) -> Duration {
        Duration { seconds, fraction }
    }

    // cmp_duration tests

    #[test]
    fn cmp_duration_both_infinite_equal() {
        assert_eq!(cmp_duration(INFINITE, INFINITE), core::cmp::Ordering::Equal);
    }

    #[test]
    fn cmp_duration_inf_greater_than_finite() {
        assert_eq!(
            cmp_duration(INFINITE, dur(1, 0)),
            core::cmp::Ordering::Greater
        );
    }

    #[test]
    fn cmp_duration_finite_less_than_inf() {
        assert_eq!(cmp_duration(dur(1, 0), INFINITE), core::cmp::Ordering::Less);
    }

    #[test]
    fn cmp_duration_seconds_dominate() {
        // 2s 0 fraction > 1s MAX fraction
        assert_eq!(
            cmp_duration(dur(2, 0), dur(1, 0xFFFF_FFFE)),
            core::cmp::Ordering::Greater
        );
    }

    #[test]
    fn cmp_duration_equal_seconds_fraction_breaks_tie() {
        assert_eq!(
            cmp_duration(dur(1, 500), dur(1, 200)),
            core::cmp::Ordering::Greater
        );
        assert_eq!(
            cmp_duration(dur(1, 200), dur(1, 500)),
            core::cmp::Ordering::Less
        );
    }

    #[test]
    fn cmp_duration_zero_less_than_one_sec() {
        assert_eq!(
            cmp_duration(dur(0, 0), dur(1, 0)),
            core::cmp::Ordering::Less
        );
    }

    // ── Helper: build a little-endian ByteCursor from a byte slice ────────────

    fn le_cursor(bytes: &[u8]) -> ByteCursor<'_> {
        use crate::protocol::dds::byte_cursor::Endianness;
        ByteCursor::new(bytes, Endianness::Little)
    }

    fn le_writer(buf: &mut [u8]) -> ByteWriter<'_> {
        use crate::protocol::dds::byte_cursor::Endianness;
        ByteWriter::new(buf, Endianness::Little)
    }

    // ── LifespanQosPolicy ─────────────────────────────────────────────────────

    #[test]
    fn lifespan_default_is_infinite() {
        let p = LifespanQosPolicy::default();
        assert_eq!(p.duration.seconds, 0x7FFF_FFFF);
        assert_eq!(p.duration.fraction, 0xFFFF_FFFF);
    }

    #[test]
    fn lifespan_serialize_parse_roundtrip() {
        // 5 seconds, 0 fraction
        let original = LifespanQosPolicy {
            duration: dur(5, 0),
        };
        let mut buf = [0u8; 8];
        original
            .serialize(&mut le_writer(&mut buf))
            .expect("serialize");
        let parsed = LifespanQosPolicy::parse(&mut le_cursor(&buf)).expect("parse");
        assert_eq!(original, parsed);
    }

    #[test]
    fn lifespan_serialize_parse_zero_duration() {
        let original = LifespanQosPolicy {
            duration: dur(0, 0),
        };
        let mut buf = [0u8; 8];
        original
            .serialize(&mut le_writer(&mut buf))
            .expect("serialize");
        let parsed = LifespanQosPolicy::parse(&mut le_cursor(&buf)).expect("parse");
        assert_eq!(original, parsed);
        assert_eq!(parsed.duration.seconds, 0);
        assert_eq!(parsed.duration.fraction, 0);
    }

    // ── OwnershipQosPolicy ────────────────────────────────────────────────────

    #[test]
    fn ownership_shared_default() {
        let p = OwnershipQosPolicy::default();
        assert_eq!(p.kind, OwnershipKind::Shared);
    }

    #[test]
    fn ownership_serialize_parse_shared() {
        let original = OwnershipQosPolicy::shared();
        let mut buf = [0u8; 4];
        original
            .serialize(&mut le_writer(&mut buf))
            .expect("serialize");
        let parsed = OwnershipQosPolicy::parse(&mut le_cursor(&buf)).expect("parse");
        assert_eq!(parsed.kind, OwnershipKind::Shared);
    }

    #[test]
    fn ownership_serialize_parse_exclusive() {
        let original = OwnershipQosPolicy::exclusive();
        let mut buf = [0u8; 4];
        original
            .serialize(&mut le_writer(&mut buf))
            .expect("serialize");
        let parsed = OwnershipQosPolicy::parse(&mut le_cursor(&buf)).expect("parse");
        assert_eq!(parsed.kind, OwnershipKind::Exclusive);
    }

    #[test]
    fn ownership_compatible_both_shared() {
        let reader = OwnershipQosPolicy::shared();
        let writer = OwnershipQosPolicy::shared();
        assert!(reader.is_compatible_with(&writer));
    }

    #[test]
    fn ownership_incompatible_shared_vs_exclusive() {
        let reader = OwnershipQosPolicy::shared();
        let writer = OwnershipQosPolicy::exclusive();
        assert!(!reader.is_compatible_with(&writer));
    }

    #[test]
    fn ownership_incompatible_exclusive_vs_shared() {
        let reader = OwnershipQosPolicy::exclusive();
        let writer = OwnershipQosPolicy::shared();
        assert!(!reader.is_compatible_with(&writer));
    }

    // ── OwnershipStrengthQosPolicy ────────────────────────────────────────────

    #[test]
    fn ownership_strength_default_zero() {
        let p = OwnershipStrengthQosPolicy::default();
        assert_eq!(p.value, 0);
    }

    #[test]
    fn ownership_strength_serialize_parse() {
        let original = OwnershipStrengthQosPolicy::new(42);
        let mut buf = [0u8; 4];
        original
            .serialize(&mut le_writer(&mut buf))
            .expect("serialize");
        let parsed = OwnershipStrengthQosPolicy::parse(&mut le_cursor(&buf)).expect("parse");
        assert_eq!(parsed.value, 42);
    }

    // ── DestinationOrderQosPolicy ─────────────────────────────────────────────

    #[test]
    fn destination_order_default_by_reception_timestamp() {
        let p = DestinationOrderQosPolicy::default();
        assert_eq!(p.kind, DestinationOrderKind::ByReceptionTimestamp);
    }

    #[test]
    fn destination_order_serialize_parse_by_source() {
        let original = DestinationOrderQosPolicy::by_source_timestamp();
        let mut buf = [0u8; 4];
        original
            .serialize(&mut le_writer(&mut buf))
            .expect("serialize");
        let parsed = DestinationOrderQosPolicy::parse(&mut le_cursor(&buf)).expect("parse");
        assert_eq!(parsed.kind, DestinationOrderKind::BySourceTimestamp);
    }

    #[test]
    fn destination_order_compatible_by_reception_reader_by_source_writer() {
        // reader=ByReception (0) <= writer=BySource (1) → compatible
        let reader = DestinationOrderQosPolicy::by_reception_timestamp();
        let writer = DestinationOrderQosPolicy::by_source_timestamp();
        assert!(reader.is_compatible_with(&writer));
    }

    #[test]
    fn destination_order_incompatible_by_source_reader_by_reception_writer() {
        // reader=BySource (1) > writer=ByReception (0) → incompatible
        let reader = DestinationOrderQosPolicy::by_source_timestamp();
        let writer = DestinationOrderQosPolicy::by_reception_timestamp();
        assert!(!reader.is_compatible_with(&writer));
    }

    // ── ResourceLimitsQosPolicy ───────────────────────────────────────────────

    #[test]
    fn resource_limits_unlimited_default() {
        let p = ResourceLimitsQosPolicy::default();
        assert_eq!(p.max_samples, -1);
        assert_eq!(p.max_instances, -1);
        assert_eq!(p.max_samples_per_instance, -1);
    }

    #[test]
    fn resource_limits_serialize_parse_finite() {
        let original = ResourceLimitsQosPolicy {
            max_samples: 100,
            max_instances: 10,
            max_samples_per_instance: 5,
        };
        let mut buf = [0u8; 12];
        original
            .serialize(&mut le_writer(&mut buf))
            .expect("serialize");
        let parsed = ResourceLimitsQosPolicy::parse(&mut le_cursor(&buf)).expect("parse");
        assert_eq!(parsed.max_samples, 100);
        assert_eq!(parsed.max_instances, 10);
        assert_eq!(parsed.max_samples_per_instance, 5);
    }
}
