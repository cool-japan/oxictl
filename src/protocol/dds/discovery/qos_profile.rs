//! QosProfile — aggregator of all 5 DDS QoS policies with ROS2 named profiles.

use crate::protocol::dds::discovery::qos::{
    DeadlineQosPolicy, DurabilityKind, DurabilityQosPolicy, HistoryKind, HistoryQosPolicy,
    LivelinessKind, LivelinessQosPolicy, ReliabilityKind, ReliabilityQosPolicy,
};
use crate::protocol::dds::types::time::Duration;

// Convenience constants

const DURATION_INFINITE: Duration = Duration {
    seconds: 0x7FFF_FFFF,
    fraction: 0xFFFF_FFFF,
};

/// 100 ms in NTP fraction units: 0.1 * 2^32 ≈ 0x1999_9999.
const FRACTION_100MS: u32 = 0x1999_9999;

/// Aggregated QoS profile for an endpoint (reader or writer).
///
/// All five standard DDS policies are present. History is informational (not RxO);
/// the other four participate in endpoint matching via [`match_endpoint_qos`].
///
/// [`match_endpoint_qos`]: crate::protocol::dds::discovery::qos_match::match_endpoint_qos
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QosProfile {
    pub reliability: ReliabilityQosPolicy,
    pub history: HistoryQosPolicy,
    pub durability: DurabilityQosPolicy,
    pub deadline: DeadlineQosPolicy,
    pub liveliness: LivelinessQosPolicy,
}

impl QosProfile {
    /// `rmw_qos_profile_default`: Reliable / KeepLast=10 / Volatile / INF deadline / Auto+INF liveliness.
    pub const fn ros2_default() -> Self {
        Self {
            reliability: ReliabilityQosPolicy {
                kind: ReliabilityKind::Reliable,
                max_blocking_time: Duration {
                    seconds: 0,
                    fraction: FRACTION_100MS,
                },
            },
            history: HistoryQosPolicy {
                kind: HistoryKind::KeepLast,
                depth: 10,
            },
            durability: DurabilityQosPolicy {
                kind: DurabilityKind::Volatile,
            },
            deadline: DeadlineQosPolicy {
                period: DURATION_INFINITE,
            },
            liveliness: LivelinessQosPolicy {
                kind: LivelinessKind::Automatic,
                lease_duration: DURATION_INFINITE,
            },
        }
    }

    /// `rmw_qos_profile_sensor_data`: BestEffort / KeepLast=5 / Volatile / INF / Auto+INF.
    pub const fn ros2_sensor_data() -> Self {
        Self {
            reliability: ReliabilityQosPolicy {
                kind: ReliabilityKind::BestEffort,
                // inline Duration::zero() since Duration::zero() is not const fn
                max_blocking_time: Duration {
                    seconds: 0,
                    fraction: 0,
                },
            },
            history: HistoryQosPolicy {
                kind: HistoryKind::KeepLast,
                depth: 5,
            },
            durability: DurabilityQosPolicy {
                kind: DurabilityKind::Volatile,
            },
            deadline: DeadlineQosPolicy {
                period: DURATION_INFINITE,
            },
            liveliness: LivelinessQosPolicy {
                kind: LivelinessKind::Automatic,
                lease_duration: DURATION_INFINITE,
            },
        }
    }

    /// `rmw_qos_profile_services_default`: Reliable / KeepLast=10 / Volatile / INF / Auto+INF.
    ///
    /// Same as default for general use; explicit for documentation clarity.
    pub const fn ros2_services_default() -> Self {
        Self::ros2_default()
    }

    /// `rmw_qos_profile_parameters`: Reliable / KeepLast=1000 / Volatile / INF / Auto+INF.
    pub const fn ros2_parameters() -> Self {
        Self {
            reliability: ReliabilityQosPolicy {
                kind: ReliabilityKind::Reliable,
                max_blocking_time: Duration {
                    seconds: 0,
                    fraction: FRACTION_100MS,
                },
            },
            history: HistoryQosPolicy {
                kind: HistoryKind::KeepLast,
                depth: 1000,
            },
            durability: DurabilityQosPolicy {
                kind: DurabilityKind::Volatile,
            },
            deadline: DeadlineQosPolicy {
                period: DURATION_INFINITE,
            },
            liveliness: LivelinessQosPolicy {
                kind: LivelinessKind::Automatic,
                lease_duration: DURATION_INFINITE,
            },
        }
    }

    /// `rmw_qos_profile_parameter_events`: Reliable / KeepLast=1000 / Volatile / INF / Auto+INF.
    pub const fn ros2_parameter_events() -> Self {
        Self::ros2_parameters()
    }

    /// `/rosout` profile: Reliable / KeepLast=1000 / TransientLocal / INF / Auto+INF.
    pub const fn ros2_rosout() -> Self {
        Self {
            reliability: ReliabilityQosPolicy {
                kind: ReliabilityKind::Reliable,
                max_blocking_time: Duration {
                    seconds: 0,
                    fraction: FRACTION_100MS,
                },
            },
            history: HistoryQosPolicy {
                kind: HistoryKind::KeepLast,
                depth: 1000,
            },
            durability: DurabilityQosPolicy {
                kind: DurabilityKind::TransientLocal,
            },
            deadline: DeadlineQosPolicy {
                period: DURATION_INFINITE,
            },
            liveliness: LivelinessQosPolicy {
                kind: LivelinessKind::Automatic,
                lease_duration: DURATION_INFINITE,
            },
        }
    }

    /// `rclcpp::ClockQoS`: BestEffort / KeepLast=1 / Volatile / INF deadline / Auto+INF liveliness.
    /// Used for the `/clock` topic (ROS time source).
    pub const fn ros2_clock() -> Self {
        Self {
            reliability: ReliabilityQosPolicy {
                kind: ReliabilityKind::BestEffort,
                max_blocking_time: Duration {
                    seconds: 0,
                    fraction: 0,
                },
            },
            history: HistoryQosPolicy {
                kind: HistoryKind::KeepLast,
                depth: 1,
            },
            durability: DurabilityQosPolicy {
                kind: DurabilityKind::Volatile,
            },
            deadline: DeadlineQosPolicy {
                period: DURATION_INFINITE,
            },
            liveliness: LivelinessQosPolicy {
                kind: LivelinessKind::Automatic,
                lease_duration: DURATION_INFINITE,
            },
        }
    }
}

impl Default for QosProfile {
    fn default() -> Self {
        Self::ros2_default()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dds::discovery::qos_match::match_endpoint_qos;

    #[test]
    fn ros2_default_profile_fields() {
        let p = QosProfile::ros2_default();
        assert_eq!(p.reliability.kind, ReliabilityKind::Reliable);
        assert_eq!(p.history.kind, HistoryKind::KeepLast);
        assert_eq!(p.history.depth, 10);
        assert_eq!(p.durability.kind, DurabilityKind::Volatile);
    }

    #[test]
    fn ros2_sensor_data_profile_fields() {
        let p = QosProfile::ros2_sensor_data();
        assert_eq!(p.reliability.kind, ReliabilityKind::BestEffort);
        assert_eq!(p.history.kind, HistoryKind::KeepLast);
        assert_eq!(p.history.depth, 5);
    }

    #[test]
    fn ros2_rosout_uses_transient_local() {
        let p = QosProfile::ros2_rosout();
        assert_eq!(p.durability.kind, DurabilityKind::TransientLocal);
        assert_eq!(p.history.depth, 1000);
    }

    #[test]
    fn ros2_default_self_compatible() {
        let p = QosProfile::ros2_default();
        assert!(match_endpoint_qos(&p, &p).is_ok());
    }
}
