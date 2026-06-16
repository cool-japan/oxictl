//! QoS endpoint compatibility matching per OMG DDS 1.4 §2.2.3.

use heapless::Vec;

use crate::protocol::dds::discovery::qos::{
    cmp_duration, DestinationOrderKind, DestinationOrderQosPolicy, DurabilityKind, LivelinessKind,
    OwnershipKind, OwnershipQosPolicy, ReliabilityKind,
};
use crate::protocol::dds::discovery::qos_profile::QosProfile;
use crate::protocol::dds::types::time::Duration;

/// A single QoS incompatibility found during endpoint matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompatibleQos {
    /// Reliability mismatch: reader requires stronger guarantee than writer offers.
    Reliability {
        reader: ReliabilityKind,
        writer: ReliabilityKind,
    },
    /// Durability mismatch: reader requires stronger persistence than writer offers.
    Durability {
        reader: DurabilityKind,
        writer: DurabilityKind,
    },
    /// Deadline mismatch: reader's required period is shorter than writer's offered period.
    Deadline {
        reader_period: Duration,
        writer_period: Duration,
    },
    /// Liveliness kind mismatch: reader requires stronger kind than writer offers.
    LivelinessKind {
        reader: LivelinessKind,
        writer: LivelinessKind,
    },
    /// Liveliness lease mismatch: reader's required lease is shorter than writer's offered lease.
    LivelinessLease {
        reader_lease: Duration,
        writer_lease: Duration,
    },
    /// Ownership kind mismatch: reader and writer must use the same ownership kind.
    Ownership {
        reader: OwnershipKind,
        writer: OwnershipKind,
    },
    /// DestinationOrder mismatch: reader requires BySourceTimestamp but writer only offers ByReceptionTimestamp.
    DestinationOrder {
        reader: DestinationOrderKind,
        writer: DestinationOrderKind,
    },
}

/// Check whether `reader` and `writer` QoS profiles are compatible.
///
/// Returns `Ok(())` if all 4 RxO policies are compatible.
/// Returns `Err(violations)` with every incompatibility found (never short-circuits).
/// The returned [`Vec`] has capacity 5 (one slot per distinct violation type).
pub fn match_endpoint_qos(
    reader: &QosProfile,
    writer: &QosProfile,
) -> Result<(), Vec<IncompatibleQos, 5>> {
    let mut v: Vec<IncompatibleQos, 5> = Vec::new();

    // Reliability: reader.kind <= writer.kind
    if !reader.reliability.is_compatible_with(&writer.reliability) {
        let _ = v.push(IncompatibleQos::Reliability {
            reader: reader.reliability.kind,
            writer: writer.reliability.kind,
        });
    }

    // Durability: reader.kind <= writer.kind
    if !reader.durability.is_compatible_with(&writer.durability) {
        let _ = v.push(IncompatibleQos::Durability {
            reader: reader.durability.kind,
            writer: writer.durability.kind,
        });
    }

    // Deadline: reader.period >= writer.period
    if !reader.deadline.is_compatible_with(&writer.deadline) {
        let _ = v.push(IncompatibleQos::Deadline {
            reader_period: reader.deadline.period,
            writer_period: writer.deadline.period,
        });
    }

    // Liveliness: report kind and lease mismatches as separate entries so callers can distinguish.
    let lv_kind_ok = (reader.liveliness.kind as i32) <= (writer.liveliness.kind as i32);
    let lv_lease_ok = cmp_duration(
        reader.liveliness.lease_duration,
        writer.liveliness.lease_duration,
    ) != core::cmp::Ordering::Less;

    if !lv_kind_ok {
        let _ = v.push(IncompatibleQos::LivelinessKind {
            reader: reader.liveliness.kind,
            writer: writer.liveliness.kind,
        });
    }
    if !lv_lease_ok {
        let _ = v.push(IncompatibleQos::LivelinessLease {
            reader_lease: reader.liveliness.lease_duration,
            writer_lease: writer.liveliness.lease_duration,
        });
    }

    if v.is_empty() {
        Ok(())
    } else {
        Err(v)
    }
}

/// Extended endpoint matcher that also checks `Ownership` and `DestinationOrder` RxO policies.
///
/// The base 5 checks (Reliability, Durability, Deadline, Liveliness ×2) are delegated to
/// [`match_endpoint_qos`].  The two additional policies are passed as explicit arguments
/// because they are not yet embedded in [`QosProfile`].
///
/// Returns `Ok(())` when all policies are compatible.  Returns `Err(violations)` with
/// every incompatibility found — never short-circuits.
/// The returned [`Vec`] has capacity 7 (5 from base checks + 2 new policies).
///
/// # Note on return size
/// The heapless fixed-capacity array cannot be heap-allocated in no_std contexts, so the
/// large-err lint is suppressed here — boxing is not available without `alloc`.
#[allow(clippy::result_large_err)]
pub fn match_endpoint_qos_extended(
    reader: &QosProfile,
    writer: &QosProfile,
    reader_ownership: &OwnershipQosPolicy,
    writer_ownership: &OwnershipQosPolicy,
    reader_dest_order: &DestinationOrderQosPolicy,
    writer_dest_order: &DestinationOrderQosPolicy,
) -> Result<(), Vec<IncompatibleQos, 7>> {
    let mut v: Vec<IncompatibleQos, 7> = Vec::new();

    // Run the base 5-slot RxO checks (Reliability, Durability, Deadline, Liveliness ×2).
    if let Err(base_errs) = match_endpoint_qos(reader, writer) {
        for e in base_errs.iter() {
            let _ = v.push(*e);
        }
    }

    // Ownership: both sides must have the same kind.
    if !reader_ownership.is_compatible_with(writer_ownership) {
        let _ = v.push(IncompatibleQos::Ownership {
            reader: reader_ownership.kind,
            writer: writer_ownership.kind,
        });
    }

    // DestinationOrder: reader.kind <= writer.kind.
    if !reader_dest_order.is_compatible_with(writer_dest_order) {
        let _ = v.push(IncompatibleQos::DestinationOrder {
            reader: reader_dest_order.kind,
            writer: writer_dest_order.kind,
        });
    }

    if v.is_empty() {
        Ok(())
    } else {
        Err(v)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::discovery::qos::{
        DeadlineQosPolicy, DestinationOrderQosPolicy, DurabilityQosPolicy, HistoryKind,
        HistoryQosPolicy, LivelinessQosPolicy, OwnershipQosPolicy, ReliabilityQosPolicy,
    };
    use crate::protocol::dds::discovery::qos_profile::QosProfile;
    use crate::protocol::dds::types::time::Duration;

    const DURATION_INFINITE: Duration = Duration {
        seconds: 0x7FFF_FFFF,
        fraction: 0xFFFF_FFFF,
    };

    fn dur(seconds: i32, fraction: u32) -> Duration {
        Duration { seconds, fraction }
    }

    fn base_profile() -> QosProfile {
        QosProfile::ros2_default()
    }

    // ── Reliability tests ────────────────────────────────────────────────────

    #[test]
    fn reliability_reader_reliable_writer_best_effort_incompatible() {
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.reliability = ReliabilityQosPolicy::reliable();
        writer.reliability = ReliabilityQosPolicy::best_effort();
        let result = match_endpoint_qos(&reader, &writer);
        let errs = result.expect_err("should be incompatible");
        assert!(errs
            .iter()
            .any(|e| matches!(e, IncompatibleQos::Reliability { .. })));
    }

    #[test]
    fn reliability_reader_best_effort_writer_reliable_compatible() {
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.reliability = ReliabilityQosPolicy::best_effort();
        writer.reliability = ReliabilityQosPolicy::reliable();
        assert!(match_endpoint_qos(&reader, &writer).is_ok());
    }

    // ── Durability tests ─────────────────────────────────────────────────────

    #[test]
    fn durability_reader_transient_local_writer_volatile_incompatible() {
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.durability = DurabilityQosPolicy {
            kind: DurabilityKind::TransientLocal,
        };
        writer.durability = DurabilityQosPolicy {
            kind: DurabilityKind::Volatile,
        };
        let errs = match_endpoint_qos(&reader, &writer).expect_err("should be incompatible");
        assert!(errs
            .iter()
            .any(|e| matches!(e, IncompatibleQos::Durability { .. })));
    }

    #[test]
    fn durability_reader_volatile_writer_persistent_compatible() {
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.durability = DurabilityQosPolicy {
            kind: DurabilityKind::Volatile,
        };
        writer.durability = DurabilityQosPolicy {
            kind: DurabilityKind::Persistent,
        };
        assert!(match_endpoint_qos(&reader, &writer).is_ok());
    }

    // ── Deadline tests ───────────────────────────────────────────────────────

    #[test]
    fn deadline_reader_period_less_than_writer_incompatible() {
        // reader requires 1s cadence; writer only offers 2s cadence → incompatible
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.deadline = DeadlineQosPolicy { period: dur(1, 0) };
        writer.deadline = DeadlineQosPolicy { period: dur(2, 0) };
        let errs = match_endpoint_qos(&reader, &writer).expect_err("should be incompatible");
        assert!(errs
            .iter()
            .any(|e| matches!(e, IncompatibleQos::Deadline { .. })));
    }

    #[test]
    fn deadline_reader_period_equal_writer_compatible() {
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.deadline = DeadlineQosPolicy { period: dur(1, 0) };
        writer.deadline = DeadlineQosPolicy { period: dur(1, 0) };
        assert!(match_endpoint_qos(&reader, &writer).is_ok());
    }

    #[test]
    fn deadline_reader_period_greater_writer_compatible() {
        // reader period 2s, writer period 1s → reader is less strict → compatible
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.deadline = DeadlineQosPolicy { period: dur(2, 0) };
        writer.deadline = DeadlineQosPolicy { period: dur(1, 0) };
        assert!(match_endpoint_qos(&reader, &writer).is_ok());
    }

    // ── Liveliness tests ─────────────────────────────────────────────────────

    #[test]
    fn liveliness_kind_reader_manual_topic_writer_automatic_incompatible() {
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.liveliness = LivelinessQosPolicy {
            kind: LivelinessKind::ManualByTopic,
            lease_duration: DURATION_INFINITE,
        };
        writer.liveliness = LivelinessQosPolicy {
            kind: LivelinessKind::Automatic,
            lease_duration: DURATION_INFINITE,
        };
        let errs = match_endpoint_qos(&reader, &writer).expect_err("should be incompatible");
        assert!(errs
            .iter()
            .any(|e| matches!(e, IncompatibleQos::LivelinessKind { .. })));
    }

    #[test]
    fn liveliness_kind_compatible_lease_too_short_on_writer_incompatible() {
        // reader lease = 100ms, writer lease = 200ms → reader wants ≥ 200ms from writer but
        // writer only offers 200ms while reader's required is 100ms.
        // Reader lease 100ms < writer lease 200ms → reader needs shorter, writer offers longer.
        // RxO: reader.lease >= writer.lease must hold; 100ms < 200ms → incompatible.
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.liveliness = LivelinessQosPolicy {
            kind: LivelinessKind::Automatic,
            lease_duration: dur(0, 0x1999_9999), // 100ms
        };
        writer.liveliness = LivelinessQosPolicy {
            kind: LivelinessKind::Automatic,
            lease_duration: dur(0, 0x3333_3333), // ~200ms
        };
        let errs = match_endpoint_qos(&reader, &writer).expect_err("should be incompatible");
        assert!(errs
            .iter()
            .any(|e| matches!(e, IncompatibleQos::LivelinessLease { .. })));
    }

    #[test]
    fn liveliness_lease_infinite_reader_finite_writer_compatible() {
        // reader lease = INF >= writer lease = 1s → compatible
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.liveliness = LivelinessQosPolicy {
            kind: LivelinessKind::Automatic,
            lease_duration: DURATION_INFINITE,
        };
        writer.liveliness = LivelinessQosPolicy {
            kind: LivelinessKind::Automatic,
            lease_duration: dur(1, 0),
        };
        assert!(match_endpoint_qos(&reader, &writer).is_ok());
    }

    // ── Multi-violation and ok tests ─────────────────────────────────────────

    #[test]
    fn match_returns_all_violations() {
        // Reliable reader + BestEffort writer AND TransientLocal reader + Volatile writer → 2 violations
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.reliability = ReliabilityQosPolicy::reliable();
        writer.reliability = ReliabilityQosPolicy::best_effort();
        reader.durability = DurabilityQosPolicy {
            kind: DurabilityKind::TransientLocal,
        };
        writer.durability = DurabilityQosPolicy {
            kind: DurabilityKind::Volatile,
        };
        let errs = match_endpoint_qos(&reader, &writer).expect_err("should have violations");
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn match_compatible_returns_ok() {
        let p = QosProfile::ros2_default();
        assert!(match_endpoint_qos(&p, &p).is_ok());
    }

    // ── History policy test (informational, not RxO) ─────────────────────────

    #[test]
    fn history_policy_not_checked_for_compatibility() {
        // Different history settings do not affect match result
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.history = HistoryQosPolicy {
            kind: HistoryKind::KeepAll,
            depth: 0,
        };
        writer.history = HistoryQosPolicy {
            kind: HistoryKind::KeepLast,
            depth: 1,
        };
        assert!(match_endpoint_qos(&reader, &writer).is_ok());
    }

    // ── match_endpoint_qos_extended tests ────────────────────────────────────

    fn shared_ownership() -> OwnershipQosPolicy {
        OwnershipQosPolicy::shared()
    }

    fn exclusive_ownership() -> OwnershipQosPolicy {
        OwnershipQosPolicy::exclusive()
    }

    fn by_reception() -> DestinationOrderQosPolicy {
        DestinationOrderQosPolicy::by_reception_timestamp()
    }

    fn by_source() -> DestinationOrderQosPolicy {
        DestinationOrderQosPolicy::by_source_timestamp()
    }

    #[test]
    fn extended_match_ownership_incompatible() {
        let p = base_profile();
        // reader=Exclusive, writer=Shared → ownership mismatch
        let errs = match_endpoint_qos_extended(
            &p,
            &p,
            &exclusive_ownership(),
            &shared_ownership(),
            &by_reception(),
            &by_reception(),
        )
        .expect_err("should be incompatible");
        assert!(errs
            .iter()
            .any(|e| matches!(e, IncompatibleQos::Ownership { .. })));
    }

    #[test]
    fn extended_match_destination_order_incompatible() {
        let p = base_profile();
        // reader=BySource, writer=ByReception → destination order mismatch
        let errs = match_endpoint_qos_extended(
            &p,
            &p,
            &shared_ownership(),
            &shared_ownership(),
            &by_source(),
            &by_reception(),
        )
        .expect_err("should be incompatible");
        assert!(errs
            .iter()
            .any(|e| matches!(e, IncompatibleQos::DestinationOrder { .. })));
    }

    #[test]
    fn extended_match_base_and_extra_violations_all_returned() {
        // Reliable reader + BestEffort writer (base violation)
        // + Exclusive reader + Shared writer (ownership violation)
        // + BySource reader + ByReception writer (dest-order violation)
        let mut reader = base_profile();
        let mut writer = base_profile();
        reader.reliability = ReliabilityQosPolicy::reliable();
        writer.reliability = ReliabilityQosPolicy::best_effort();

        let errs = match_endpoint_qos_extended(
            &reader,
            &writer,
            &exclusive_ownership(),
            &shared_ownership(),
            &by_source(),
            &by_reception(),
        )
        .expect_err("should have violations");

        assert!(errs.len() >= 3);
        assert!(errs
            .iter()
            .any(|e| matches!(e, IncompatibleQos::Reliability { .. })));
        assert!(errs
            .iter()
            .any(|e| matches!(e, IncompatibleQos::Ownership { .. })));
        assert!(errs
            .iter()
            .any(|e| matches!(e, IncompatibleQos::DestinationOrder { .. })));
    }

    #[test]
    fn extended_match_all_compatible() {
        let p = base_profile();
        // reader=Shared, writer=Shared; reader=ByReception, writer=BySource → all compatible
        let result = match_endpoint_qos_extended(
            &p,
            &p,
            &shared_ownership(),
            &shared_ownership(),
            &by_reception(),
            &by_source(),
        );
        assert!(result.is_ok());
    }
}
