//! ROS2 Quality of Service (QoS) profiles.
//!
//! QoS policies control how messages are delivered between publishers
//! and subscribers. This module provides the standard ROS2 QoS profiles
//! matching the rmw_qos_profile_t structure.

/// Reliability policy: how messages are delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReliabilityKind {
    /// Best effort delivery - may lose messages on poor connections.
    BestEffort,
    /// Reliable delivery - retransmits lost messages.
    Reliable,
}

/// Durability policy: persistence of messages for late joiners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityKind {
    /// No persistence - late joiners receive no old messages.
    Volatile,
    /// Publisher stores messages - late joiners receive last N messages.
    TransientLocal,
}

/// History policy: how many messages to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryKind {
    /// Keep only the last N messages (N=depth).
    KeepLast(u32),
    /// Keep all messages.
    KeepAll,
}

impl HistoryKind {
    /// Depth for KeepLast, 0 for KeepAll.
    pub fn depth(&self) -> u32 {
        match self {
            HistoryKind::KeepLast(n) => *n,
            HistoryKind::KeepAll => 0,
        }
    }

    /// Is this KeepAll?
    pub fn is_keep_all(&self) -> bool {
        matches!(self, HistoryKind::KeepAll)
    }
}

/// Liveliness policy: how liveness is asserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LivelinessKind {
    /// System auto-asserts liveliness.
    Automatic,
    /// Publisher manually asserts liveliness.
    ManualByTopic,
}

/// Deadline: maximum time between successive messages.
///
/// Duration in nanoseconds. 0 means unspecified (infinite).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Duration {
    /// Seconds component.
    pub sec: i32,
    /// Nanoseconds component (0–999_999_999).
    pub nsec: u32,
}

impl Duration {
    /// Duration of zero (use as "infinite" / unset).
    pub const ZERO: Self = Self { sec: 0, nsec: 0 };
    /// Default / infinite duration.
    pub const INFINITE: Self = Self {
        sec: i32::MAX,
        nsec: u32::MAX,
    };

    /// Create from seconds and nanoseconds.
    pub const fn new(sec: i32, nsec: u32) -> Self {
        Self { sec, nsec }
    }

    /// Create from milliseconds.
    pub const fn from_millis(ms: u32) -> Self {
        Self {
            sec: (ms / 1000) as i32,
            nsec: (ms % 1000) * 1_000_000,
        }
    }

    /// Total nanoseconds (saturating for large values).
    pub fn total_ns(&self) -> i64 {
        (self.sec as i64) * 1_000_000_000 + (self.nsec as i64)
    }

    /// Is this the "infinite" sentinel?
    pub fn is_infinite(&self) -> bool {
        self.sec == i32::MAX
    }
}

/// Complete QoS profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QosProfile {
    /// History policy.
    pub history: HistoryKind,
    /// Reliability policy.
    pub reliability: ReliabilityKind,
    /// Durability policy.
    pub durability: DurabilityKind,
    /// Deadline duration.
    pub deadline: Duration,
    /// Lifespan: max age of a message.
    pub lifespan: Duration,
    /// Liveliness policy.
    pub liveliness: LivelinessKind,
    /// Liveliness lease duration.
    pub liveliness_lease: Duration,
    /// Avoid ROS namespace conventions if true.
    pub avoid_ros_namespace: bool,
}

impl QosProfile {
    /// Default ROS2 QoS profile:
    /// KeepLast(10), Reliable, Volatile, infinite deadline.
    pub const DEFAULT: Self = Self {
        history: HistoryKind::KeepLast(10),
        reliability: ReliabilityKind::Reliable,
        durability: DurabilityKind::Volatile,
        deadline: Duration::INFINITE,
        lifespan: Duration::INFINITE,
        liveliness: LivelinessKind::Automatic,
        liveliness_lease: Duration::INFINITE,
        avoid_ros_namespace: false,
    };

    /// Sensor data profile:
    /// KeepLast(5), BestEffort, Volatile — for high-rate sensor streams.
    pub const SENSOR_DATA: Self = Self {
        history: HistoryKind::KeepLast(5),
        reliability: ReliabilityKind::BestEffort,
        durability: DurabilityKind::Volatile,
        deadline: Duration::INFINITE,
        lifespan: Duration::INFINITE,
        liveliness: LivelinessKind::Automatic,
        liveliness_lease: Duration::INFINITE,
        avoid_ros_namespace: false,
    };

    /// Parameters profile:
    /// KeepLast(1000), Reliable, TransientLocal — for parameter events.
    pub const PARAMETERS: Self = Self {
        history: HistoryKind::KeepLast(1000),
        reliability: ReliabilityKind::Reliable,
        durability: DurabilityKind::TransientLocal,
        deadline: Duration::INFINITE,
        lifespan: Duration::INFINITE,
        liveliness: LivelinessKind::Automatic,
        liveliness_lease: Duration::INFINITE,
        avoid_ros_namespace: false,
    };

    /// Services default profile:
    /// KeepLast(10), Reliable, Volatile — for service calls.
    pub const SERVICES_DEFAULT: Self = Self {
        history: HistoryKind::KeepLast(10),
        reliability: ReliabilityKind::Reliable,
        durability: DurabilityKind::Volatile,
        deadline: Duration::INFINITE,
        lifespan: Duration::INFINITE,
        liveliness: LivelinessKind::Automatic,
        liveliness_lease: Duration::INFINITE,
        avoid_ros_namespace: false,
    };

    /// Best available profile (matches DDS defaults).
    pub const BEST_AVAILABLE: Self = Self {
        history: HistoryKind::KeepLast(1),
        reliability: ReliabilityKind::BestEffort,
        durability: DurabilityKind::Volatile,
        deadline: Duration::ZERO,
        lifespan: Duration::INFINITE,
        liveliness: LivelinessKind::Automatic,
        liveliness_lease: Duration::INFINITE,
        avoid_ros_namespace: false,
    };

    /// Real-time profile: small depth, best effort, tight deadline.
    pub const REAL_TIME: Self = Self {
        history: HistoryKind::KeepLast(1),
        reliability: ReliabilityKind::BestEffort,
        durability: DurabilityKind::Volatile,
        deadline: Duration::from_millis(10), // 10ms deadline
        lifespan: Duration::from_millis(10),
        liveliness: LivelinessKind::Automatic,
        liveliness_lease: Duration::INFINITE,
        avoid_ros_namespace: false,
    };

    /// Create a custom profile with fluent builder style.
    pub const fn new() -> Self {
        Self::DEFAULT
    }

    /// Set reliability.
    pub const fn with_reliability(mut self, r: ReliabilityKind) -> Self {
        self.reliability = r;
        self
    }

    /// Set durability.
    pub const fn with_durability(mut self, d: DurabilityKind) -> Self {
        self.durability = d;
        self
    }

    /// Set history.
    pub const fn with_history(mut self, h: HistoryKind) -> Self {
        self.history = h;
        self
    }

    /// Set deadline.
    pub const fn with_deadline(mut self, d: Duration) -> Self {
        self.deadline = d;
        self
    }

    /// Set lifespan.
    pub const fn with_lifespan(mut self, l: Duration) -> Self {
        self.lifespan = l;
        self
    }

    /// Check if this profile is compatible with another for pub-sub matching.
    ///
    /// Compatibility rules (simplified per DDS):
    /// - Reliability: reliable publisher is compatible with best-effort subscriber,
    ///   but not vice versa.
    /// - Durability: transient-local publisher is compatible with volatile subscriber,
    ///   but not vice versa.
    pub fn is_compatible_with_subscriber(&self, subscriber: &QosProfile) -> bool {
        // Publisher reliability >= subscriber reliability
        let rel_ok = !matches!(
            (self.reliability, subscriber.reliability),
            (ReliabilityKind::BestEffort, ReliabilityKind::Reliable)
        );
        // Publisher durability >= subscriber durability
        let dur_ok = !matches!(
            (self.durability, subscriber.durability),
            (DurabilityKind::Volatile, DurabilityKind::TransientLocal)
        );
        rel_ok && dur_ok
    }

    /// Depth of the history queue.
    pub fn history_depth(&self) -> u32 {
        self.history.depth()
    }

    /// Whether this is a reliable profile.
    pub fn is_reliable(&self) -> bool {
        self.reliability == ReliabilityKind::Reliable
    }

    /// Whether this profile uses transient-local durability.
    pub fn is_transient_local(&self) -> bool {
        self.durability == DurabilityKind::TransientLocal
    }
}

impl Default for QosProfile {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_sensor_data() {
        let qos = QosProfile::SENSOR_DATA;
        assert_eq!(qos.reliability, ReliabilityKind::BestEffort);
        assert_eq!(qos.durability, DurabilityKind::Volatile);
        assert_eq!(qos.history_depth(), 5);
    }

    #[test]
    fn test_preset_parameters() {
        let qos = QosProfile::PARAMETERS;
        assert_eq!(qos.reliability, ReliabilityKind::Reliable);
        assert_eq!(qos.durability, DurabilityKind::TransientLocal);
        assert!(qos.is_transient_local());
        assert_eq!(qos.history_depth(), 1000);
    }

    #[test]
    fn test_compatibility_reliable_pub_besteffort_sub() {
        let pub_qos = QosProfile::DEFAULT; // Reliable
        let sub_qos = QosProfile::SENSOR_DATA; // BestEffort
        assert!(pub_qos.is_compatible_with_subscriber(&sub_qos));
    }

    #[test]
    fn test_incompatibility_besteffort_pub_reliable_sub() {
        let pub_qos = QosProfile::SENSOR_DATA; // BestEffort
        let sub_qos = QosProfile::DEFAULT; // Reliable
        assert!(!pub_qos.is_compatible_with_subscriber(&sub_qos));
    }

    #[test]
    fn test_builder_pattern() {
        let qos = QosProfile::new()
            .with_reliability(ReliabilityKind::BestEffort)
            .with_durability(DurabilityKind::TransientLocal)
            .with_history(HistoryKind::KeepLast(20));
        assert_eq!(qos.reliability, ReliabilityKind::BestEffort);
        assert!(qos.is_transient_local());
        assert_eq!(qos.history_depth(), 20);
    }

    #[test]
    fn test_duration_from_millis() {
        let d = Duration::from_millis(1500);
        assert_eq!(d.sec, 1);
        assert_eq!(d.nsec, 500_000_000);
        let total = d.total_ns();
        assert_eq!(total, 1_500_000_000);
    }

    #[test]
    fn test_history_kind() {
        let h = HistoryKind::KeepLast(10);
        assert_eq!(h.depth(), 10);
        assert!(!h.is_keep_all());
        assert!(HistoryKind::KeepAll.is_keep_all());
    }

    #[test]
    fn test_durability_incompatibility() {
        let pub_qos = QosProfile::DEFAULT; // Volatile
        let sub_qos = QosProfile::PARAMETERS; // TransientLocal
        assert!(!pub_qos.is_compatible_with_subscriber(&sub_qos));
    }
}
